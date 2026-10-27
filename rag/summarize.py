import os
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_core.prompts import ChatPromptTemplate

BATCH_CHARS = 20000     # how much text goes into one LLM call
COMBINE_CHARS = 12000   # max size of the notes combined in one reduce call
MAX_PARALLEL = 3        # how many LLM calls run at the same time

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


def clean_output(text):
    """Turn special spaces and symbols the model may emit into plain text."""
    return unicodedata.normalize("NFKC", text).strip()


def invoke_with_retry(chain, inputs, retries=4):
    """Call the LLM; if the free-tier rate limit is hit, wait and try again."""
    for attempt in range(1, retries + 1):
        try:
            return clean_output(chain.invoke(inputs).content)
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


def run_in_parallel(chain, texts, progress, label):
    """Run the chain on every text, several at a time, keeping the order."""
    results = [None] * len(texts)
    with ThreadPoolExecutor(max_workers=MAX_PARALLEL) as pool:
        futures = {
            pool.submit(invoke_with_retry, chain, {"text": text}): index
            for index, text in enumerate(texts)
        }
        done = 0
        for future in as_completed(futures):
            results[futures[future]] = future.result()
            done += 1
            progress(f"  {label}: {done} of {len(texts)} done")
    return results


def summarize_texts(texts, llm, progress=print):
    """Map-reduce summary of a list of text pieces."""
    map_chain = MAP_PROMPT | llm
    reduce_chain = REDUCE_PROMPT | llm

    # MAP: notes for every batch of text (several batches at once)
    batches = make_batches(texts, BATCH_CHARS)
    notes = run_in_parallel(map_chain, batches, progress, "reading")

    # REDUCE: merge notes in rounds until they fit in one call
    while len(notes) > 1 and sum(len(n) for n in notes) > COMBINE_CHARS:
        groups = make_batches(notes, COMBINE_CHARS)
        if len(groups) == len(notes):
            break  # cannot shrink further
        progress(f"  merging {len(notes)} sets of notes into {len(groups)}...")
        notes = run_in_parallel(reduce_chain, groups, progress, "merging")

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