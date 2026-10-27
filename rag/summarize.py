import os
import time

from langchain_core.prompts import ChatPromptTemplate

BATCH_CHARS = 12000     # how much text goes into one LLM call
COMBINE_CHARS = 12000   # max size of the notes combined in one reduce call

MAP_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are reading one part of a longer document. Use ONLY the text "
        "provided and no outside knowledge. Write short bullet points with "
        "the key points of this part. Keep specific names, numbers and "
        "results exactly as written. Do not say that this is only a part "
        "of the document.",
    ),
    ("human", "{text}"),
])

REDUCE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are given bullet-point notes taken from consecutive parts of one "
        "document. Merge them into a single summary using ONLY these notes "
        "and no outside knowledge. Remove repetition.\n\n"
        "Format:\n"
        "Overview: one or two sentences saying what the document is about.\n"
        "Main points: 6 to 8 bullet points, each at most two sentences. "
        "Prefer the key ideas and headline results over listing every number. "
        "If the notes give a range of settings, report the range, not a "
        "single value.\n"
        "Write formulas in plain text, not LaTeX.",
    ),
    ("human", "{text}"),
])


def invoke_with_retry(chain, inputs, retries=4):
    """Call the LLM; if the free-tier rate limit is hit, wait and try again."""
    for attempt in range(1, retries + 1):
        try:
            return chain.invoke(inputs).content.strip()
        except Exception as error:
            message = str(error).lower()
            is_rate_limit = "429" in message or "rate limit" in message
            if is_rate_limit and attempt < retries:
                wait = 20 * attempt
                print(f"  (rate limit reached, waiting {wait} seconds...)")
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

    # MAP: notes for every batch of text
    batches = make_batches(texts, BATCH_CHARS)
    notes = []
    for number, batch in enumerate(batches, start=1):
        progress(f"  reading part {number} of {len(batches)}...")
        notes.append(invoke_with_retry(map_chain, {"text": batch}))

    # REDUCE: merge notes in rounds until they fit in one call
    while len(notes) > 1 and sum(len(n) for n in notes) > COMBINE_CHARS:
        groups = make_batches(notes, COMBINE_CHARS)
        if len(groups) == len(notes):
            break  # cannot shrink further
        progress(f"  merging {len(notes)} sets of notes into {len(groups)}...")
        notes = [invoke_with_retry(reduce_chain, {"text": g}) for g in groups]

    progress("  writing the final summary...")
    return invoke_with_retry(reduce_chain, {"text": "\n\n".join(notes)})


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
        progress(f"Summarizing {name} ({len(texts)} chunks)...")
        summaries[name] = summarize_texts(texts, llm, progress)
    return summaries