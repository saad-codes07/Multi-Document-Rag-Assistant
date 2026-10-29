# Each test has:
#   q      - the question
#   file   - which PDF holds the answer
#   phrase - a short piece of text that must appear in a chunk for that chunk
#            to count as "the chunk that contains the answer"
TESTS = [
    # ---------- paper.pdf (Attention Is All You Need) ----------
    {"q": "What optimizer was used?", "file": "paper.pdf",
     "phrase": "adam optimizer"},
    {"q": "How many training steps did the big models run?", "file": "paper.pdf",
     "phrase": "300,000 steps"},
    {"q": "What BLEU score did the model get on English-to-German translation?",
     "file": "paper.pdf", "phrase": "28.4 bleu"},
    {"q": "How many P100 GPUs were used for training?", "file": "paper.pdf",
     "phrase": "8 nvidia p100 gpus"},
    {"q": "What value of warmup_steps was used?", "file": "paper.pdf",
     "phrase": "warmup_steps = 4000"},
    {"q": "How many sentence pairs are in the WMT 2014 English-German dataset?",
     "file": "paper.pdf", "phrase": "4.5 million sentence pairs"},
    {"q": "How large is the shared vocabulary for byte-pair encoding?",
     "file": "paper.pdf", "phrase": "37000 tokens"},
    {"q": "What dropout rate did the English-to-French big model use?",
     "file": "paper.pdf", "phrase": "pdrop = 0.1"},
    {"q": "By what factor are the dot products scaled in attention?",
     "file": "paper.pdf", "phrase": "scale the dot products by"},
    {"q": "Which network components does the Transformer dispense with?",
     "file": "paper.pdf", "phrase": "dispensing with recurrence and convolutions"},

    # ---------- bert.pdf ----------
    {"q": "What does BERT stand for?", "file": "bert.pdf",
     "phrase": "bidirectional encoder representations"},
    {"q": "On how many NLP tasks did BERT obtain new state-of-the-art results?",
     "file": "bert.pdf", "phrase": "eleven natural language"},
    {"q": "What GLUE score did BERT achieve?", "file": "bert.pdf",
     "phrase": "80.5%"},
    {"q": "What MultiNLI accuracy did BERT reach?", "file": "bert.pdf",
     "phrase": "86.7%"},
    {"q": "What was the SQuAD v1.1 Test F1 score?", "file": "bert.pdf",
     "phrase": "93.2"},
    {"q": "What was the SQuAD v2.0 Test F1 score?", "file": "bert.pdf",
     "phrase": "83.1"},
    {"q": "What data was BERT pre-trained on?", "file": "bert.pdf",
     "phrase": "bookscorpus"},
    {"q": "What percentage of tokens are masked in the masked language model?",
     "file": "bert.pdf", "phrase": "15% of all wordpiece tokens"},
    {"q": "How many parameters does BERT-base have?", "file": "bert.pdf",
     "phrase": "110m"},
    {"q": "What are the two pre-training tasks of BERT?", "file": "bert.pdf",
     "phrase": "next sentence prediction"},
]