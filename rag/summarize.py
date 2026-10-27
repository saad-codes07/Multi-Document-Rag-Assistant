import os
import time
import unicodedata

from langchain_core.prompts import ChatPromptTemplate

BATCH_CHARS = 9000        # text per LLM call (about 2,700 tokens)
COMBINE_CHARS = 9000      # max size of the notes merged in one call
TPM_BUDGET = 7500         # stay under the free-tier limit of 8,000 tokens/minute
CHARS_PER_TOKEN = 3.3     # rough estimate used to count tokens
PROMPT_OVERHEAD = 250     # tokens used by the instructions
OUTPUT_ALLOWANCE = 700    # tokens we expect the reply to use

MAP_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are reading one part of a longer document. Use ONLY the text "
        "provided and no outside knowledge. Write short bullet points with "
        "the key points of this part. Copy names, numbers and units exactly "
        "as written: never convert, round or recompute them. Do not say "
        "that this is only a part of the document.",
    ),
    ("human", "{text}"),
])

REDUCE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are given bullet-point notes taken from consecutive parts of one "
        "document. Merge them into a single summary using ONLY these notes "
        "and no outside knowledge. Remove repetition.\n\n"
        "Rules for numbers: copy numbers and units exactly as they appear in "
        "the notes (for example keep '10,000 steps' as it is; do not turn it "
        "into a percentage). If the notes give a range of settings, report "
        "the range, not a single value. If you are unsure about a figure, "
        "leave it out.\n\n"
        "Format:\n"
        "Overview: one or two sentences saying what the document is about.\n"
        "Main points: 6 to 8 bullet points, each at most two sentences. "
        "Prefer the key ideas and headline results over listing every number.\n"
        "Write formulas in plain text, not LaTeX.",
    ),
    ("human", "{text}"),
])


class TokenBudget:
    """Counts the tokens sent in the last 60 seconds and makes us wait when
    the next request would go over the free-tier tokens-per-minute limit."""

    def __init__(self, tokens_per_minute):
        self.limit = tokens_per_minute
        self.events = []  # list of (time, tokens)

    def wait_for(self, tokens, progress=print):
        while True:
            now = time.time()
            self.events = [(t, n) for t, n in self.events if now - t < 60]
            used = sum(n for _, n in self.events)
            if not self.events or used + tokens <= self.limit:
                self.events.append((now, tokens))
                return
            wait = max(1, int(self.events[0][0] + 60 - now) + 1)
            progress(f"  (waiting {wait}s to stay under the free-tier limit)")
            time.sleep(wait)


BUDGET = TokenBudget(TPM_BUDGET)


def estimate_tokens(text):
    return int(len(text) / CHARS_PER_TOKEN) + PROMPT_OVERHEAD + OUTPUT_ALLOWANCE


def clean_output(text):
    """Turn special spaces and symbols the model may emit into plain text."""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u2010", "-").replace("\u2011", "-")
    return text.strip()


def invoke_with_budget(chain, text, progress=print, retries=4):
    """Call the LLM once, waiting if needed to respect the rate limit."""
    tokens = estimate_tokens(text)
    for attempt in range(1, retries + 1):
        BUDGET.wait_for(tokens, progress)
        try:
            return clean_output(chain.invoke({"text": text}).content)
        except Exception as error:
            message = str(error).lower()
            # Check 413 first: its message also contains "rate_limit"
            if "413" in message or "request too large" in message:
                raise RuntimeError(
                    "A request was too large for the free-tier limit. "
                    "Lower BATCH_CHARS in rag/summarize.py."
                ) from error
            is_rate_limit = (
                "429" in message or "rate limit" in message or "rate_limit" in message
            )
            if is_rate_limit and attempt < retries:
                wait = 20 * attempt
                progress(f"  (rate limit reached, waiting {wait} seconds...)")
                time.sleep(wait)
            else:
                raise


def make_batches(texts, max_chars):
    """Group a list of texts into batches of at most max_chars characters."""
    batches, current, size = [], [], 0
    for text in texts:
        if current and size + len(text) > max_chars:
            batches.append("\n\n".join(current))
            current, size = [], 0
        current.append(text)
        size += len(text)
    if current:
        batches.append("\n\n".join(current))
    return batches


def summarize_texts(texts, llm, progress=print):
    """Map-reduce summary of a list of text pieces."""
    map_chain = MAP_PROMPT | llm
    reduce_chain = REDUCE_PROMPT | llm

    # MAP: notes for every batch of text, one batch at a time
    batches = make_batches(texts, BATCH_CHARS)
    notes = []
    for number, batch in enumerate(batches, start=1):
        progress(f"  reading part {number} of {len(batches)}...")
        notes.append(invoke_with_budget(map_chain, batch, progress))

    # REDUCE: merge notes in rounds until they fit in one call
    while len(notes) > 1 and sum(len(n) for n in notes) > COMBINE_CHARS:
        groups = make_batches(notes, COMBINE_CHARS)
        if len(groups) == len(notes):
            break  # cannot shrink further
        progress(f"  merging {len(notes)} sets of notes into {len(groups)}...")
        notes = [invoke_with_budget(reduce_chain, g, progress) for g in groups]

    progress("  writing the final summary...")
    return invoke_with_budget(reduce_chain, "\n\n".join(notes), progress)

REFERENCE_HINTS = (
    "arxiv preprint", "in proceedings", "proceedings of", "journal of",
    "conference on", "association for computational",
)


def looks_like_references(text):
    """True for chunks that look like a bibliography."""
    lowered = " ".join(text.lower().split())
    hits = sum(lowered.count(hint) for hint in REFERENCE_HINTS)
    return hits >= 3

def summarize_documents(chunks, llm, only_file=None, progress=print):
    """Summarize each uploaded file separately.

    Returns a dictionary: {file name: summary}.
    """
    by_file = {}
    for chunk in chunks:
        name = os.path.basename(chunk.metadata.get("source", "document"))
        by_file.setdefault(name, []).append(chunk.page_content)

        summaries = {}
    for name, texts in by_file.items():
        if only_file and name != only_file:
            continue
        kept = [t for t in texts if not looks_like_references(t)]
        progress(
            f"Summarizing {name} ({len(kept)} of {len(texts)} chunks; "
            f"skipping {len(texts) - len(kept)} that look like references)..."
        )
        summaries[name] = summarize_texts(kept, llm, progress)
    return summaries