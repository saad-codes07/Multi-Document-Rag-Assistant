import json
import re
import sys
import time
import unicodedata

from langchain_core.prompts import ChatPromptTemplate

from eval_set import TESTS
from rag.chain import PROMPT, format_context, get_llm, retrieve
from rag.hybrid import build_hybrid_retriever
from rag.loader import load_pdf, split_documents
from rag.rerank import load_reranker
from rag.summarize import TokenBudget
from rag.vectorstore import get_embeddings

PDF_FILES = ["data/paper.pdf", "data/bert.pdf"]

# python eval_answers.py            -> the full upgraded pipeline
# python eval_answers.py baseline   -> the original system (dense search, top 3)
# add the word "quick" to try only 6 questions first
MODE = "baseline" if "baseline" in sys.argv else "full"
QUICK = "quick" in sys.argv

# Questions the documents should NOT be able to answer.
# The correct behaviour is to refuse.
OUT_OF_SCOPE = [
    "Who won the football World Cup?",
    "What is the capital of France?",
    "What BLEU score did GPT-4 get on English-to-German translation?",
    "What accuracy did ResNet achieve on ImageNet?",
    "How many parameters does GPT-3 have?",
]

# Facts a correct answer must contain. Each entry is a list of requirements;
# ALL must be present. Inside a requirement, "a|b" means a OR b.
# Text is compared after lowercasing and removing spaces, commas and asterisks.
EXPECT = {
    "What optimizer was used?": ["adam"],
    "How many training steps did the big models run?": ["300000|300k"],
    "What BLEU score did the model get on English-to-German translation?": ["28.4"],
    "How many P100 GPUs were used for training?": ["eight|8nvidia|8p100|8gpu"],
    "What value of warmup_steps was used?": ["4000"],
    "How many sentence pairs are in the WMT 2014 English-German dataset?": ["4.5million"],
    "How large is the shared vocabulary for byte-pair encoding?": ["37000"],
    "What dropout rate did the English-to-French big model use?": ["0.1"],
    "By what factor are the dot products scaled in attention?": ["sqrt|squareroot|√"],
    "Which network components does the Transformer dispense with?": ["recurrence", "convolution"],
    "What does BERT stand for?": ["bidirectionalencoderrepresentations"],
    "On how many NLP tasks did BERT obtain new state-of-the-art results?": ["eleven|11"],
    "What GLUE score did BERT achieve?": ["80.5"],
    "What MultiNLI accuracy did BERT reach?": ["86.7"],
    "What was the SQuAD v1.1 Test F1 score?": ["93.2"],
    "What was the SQuAD v2.0 Test F1 score?": ["83.1"],
    "What data was BERT pre-trained on?": ["bookscorpus", "wikipedia"],
    "What percentage of tokens are masked in the masked language model?": ["15%"],
    "How many parameters does BERT-base have?": ["110m|110million"],
    "What are the two pre-training tasks of BERT?": ["masked", "nextsentence"],
}

JUDGE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a strict evaluator of a question-answering system. You are "
        "given a QUESTION, the CONTEXT passages the system was allowed to use, "
        "and the system's ANSWER. Judge using ONLY the context, never your own "
        "knowledge, even if you believe the answer is true or false in the "
        "real world.\n\n"
        "faithful: true only if every factual claim in the ANSWER (numbers, "
        "names, results) is directly supported by the CONTEXT. Ignore "
        "formatting.\n"
        "relevant: true if the ANSWER addresses what the QUESTION asks.\n\n"
        "Reply with ONLY a JSON object and no other text:\n"
        "{{\"faithful\": true or false, \"unsupported\": \"the unsupported "
        "claims, or empty text\", \"relevant\": true or false}}",
    ),
    ("human", "QUESTION:\n{question}\n\nCONTEXT:\n{context}\n\nANSWER:\n{answer}"),
])

# Free-tier limit: 8,000 tokens per minute. Stay a little under it.
BUDGET = TokenBudget(7000)
CHARS_PER_TOKEN = 3.3
OUTPUT_ALLOWANCE = 800


def clean(text):
    return unicodedata.normalize("NFKC", text).strip()


def norm(text):
    """Lowercase and remove spaces, commas and asterisks, so that
    '300,000' and '300 000' and '300000' all look the same."""
    return re.sub(r"[\s,*_]+", "", unicodedata.normalize("NFKC", text).lower())


def invoke_limited(chain, inputs, prompt_chars, retries=4):
    """Call the LLM while respecting the free-tier tokens-per-minute limit."""
    tokens = int(prompt_chars / CHARS_PER_TOKEN) + OUTPUT_ALLOWANCE
    for attempt in range(1, retries + 1):
        BUDGET.wait_for(tokens)
        try:
            return chain.invoke(inputs).content
        except Exception as error:
            message = str(error).lower()
            if "413" in message or "request too large" in message:
                raise
            is_rate_limit = (
                "429" in message or "rate limit" in message or "rate_limit" in message
            )
            if is_rate_limit and attempt < retries:
                time.sleep(20 * attempt)
            else:
                raise


def is_refusal(answer):
    text = answer.lower().replace("\u2019", "'")
    return "find that in the document" in text


def is_correct(question, answer):
    requirements = EXPECT.get(question)
    if requirements is None:
        return None
    text = norm(answer)
    return all(
        any(alt in text for alt in req.split("|")) for req in requirements
    )


def parse_judge(text):
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def evaluate_question(kind, question, retriever, reranker, llm):
    # 1. Retrieve (the same steps the app performs)
    if MODE == "full":
        docs = retrieve(question, retriever, reranker)
    else:
        docs = retriever.dense_search(question, k=3)
    context = format_context(docs)

    # 2. Answer
    answer = clean(invoke_limited(
        PROMPT | llm,
        {"context": context, "history": [], "question": question},
        len(context) + len(question) + 700,
    ))
    refused = is_refusal(answer)

    # 3. Judge (skipped for refusals: there is no claim to check)
    judge = None
    if not refused:
        judge_text = invoke_limited(
            JUDGE_PROMPT | llm,
            {"question": question, "context": context, "answer": answer},
            len(context) + len(question) + len(answer) + 900,
        )
        judge = parse_judge(judge_text) or {"error": "judge reply could not be read"}

    return {
        "kind": kind,
        "question": question,
        "answer": answer,
        "refused": refused,
        "correct": is_correct(question, answer) if kind == "in" else None,
        "judge": judge,
        "chunks": [
            "{} p.{}".format(d.metadata.get("source", "?"), d.metadata.get("page_label", "?"))
            for d in docs
        ],
    }


def percent(count, total):
    return f"{count}/{total} ({100 * count / total:.0f}%)" if total else "n/a"


def print_summary(records):
    inside = [r for r in records if r["kind"] == "in"]
    outside = [r for r in records if r["kind"] == "out"]

    print("\n" + "=" * 60)
    print(f"RESULTS  (mode: {MODE})")
    print("=" * 60)

    if inside:
        answered = [r for r in inside if not r["refused"]]
        judged = [r for r in answered if r["judge"] and "error" not in r["judge"]]
        print(f"\nQuestions the documents CAN answer: {len(inside)}")
        print("  answered (did not refuse):  ", percent(len(answered), len(inside)))
        print("  correct (right fact present):",
              percent(sum(1 for r in inside if r["correct"]), len(inside)))
        print("  faithful (among answered):  ",
              percent(sum(1 for r in judged if r["judge"].get("faithful")), len(judged)))
        print("  relevant (among answered):  ",
              percent(sum(1 for r in judged if r["judge"].get("relevant")), len(judged)))

        print("\n  Problems:")
        any_problem = False
        for r in inside:
            reasons = []
            if r["refused"]:
                reasons.append("refused")
            elif r["correct"] is False:
                reasons.append("right fact missing")
            if r["judge"] and r["judge"].get("faithful") is False:
                reasons.append("unfaithful: " + str(r["judge"].get("unsupported", ""))[:100])
            if r["judge"] and r["judge"].get("relevant") is False:
                reasons.append("not relevant")
            if r["judge"] and "error" in r["judge"]:
                reasons.append("judge reply unreadable")
            if reasons:
                any_problem = True
                print(f"   - {r['question']}")
                print(f"       {'; '.join(reasons)}")
                print(f"       answer: {r['answer'][:150]!r}")
        if not any_problem:
            print("   none")

    if outside:
        refused = [r for r in outside if r["refused"]]
        print(f"\nQuestions the documents should NOT answer: {len(outside)}")
        print("  correctly refused:", percent(len(refused), len(outside)))
        for r in outside:
            if not r["refused"]:
                faithful = r["judge"].get("faithful") if r["judge"] else None
                print(f"   - answered anyway: {r['question']}")
                print(f"       answer: {r['answer'][:150]!r}  (faithful to context: {faithful})")


def main():
    chunks = []
    for path in PDF_FILES:
        chunks.extend(split_documents(load_pdf(path)))
    retriever = build_hybrid_retriever(chunks, get_embeddings())
    reranker = load_reranker() if MODE == "full" else None
    llm = get_llm()

    in_scope = [t["q"] for t in TESTS]
    out_scope = list(OUT_OF_SCOPE)
    if QUICK:
        in_scope = in_scope[::5]
        out_scope = out_scope[:2]

    jobs = [("in", q) for q in in_scope] + [("out", q) for q in out_scope]
    print(f"Mode: {MODE}. {len(jobs)} questions. This takes several minutes "
          f"because of the free-tier token limit.")

    records = []
    started = time.time()
    for number, (kind, question) in enumerate(jobs, start=1):
        print(f"\n[{number}/{len(jobs)}] ({kind}-scope) {question}")
        record = evaluate_question(kind, question, retriever, reranker, llm)
        records.append(record)
        verdict = "refused" if record["refused"] else "answered"
        print(f"   -> {verdict} | correct: {record['correct']} | judge: {record['judge']}")

        # Save after every question, so nothing is lost if it stops early
        with open(f"eval_results_{MODE}.json", "w", encoding="utf-8") as file:
            json.dump(records, file, indent=2, ensure_ascii=False)

    print_summary(records)
    print(f"\n(took {(time.time() - started) / 60:.1f} minutes; "
          f"details saved in eval_results_{MODE}.json)")


if __name__ == "__main__":
    main()