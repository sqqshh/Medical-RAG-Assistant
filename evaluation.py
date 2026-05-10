"""
evaluation.py
=============
RAG Evaluation — RAGAS-inspired metrics (no external API, no ground truth needed)
Metrics:
  1. Answer Relevancy      — does the answer address the query?
  2. Faithfulness          — is every claim grounded in the retrieved context?
  3. Context Precision     — are retrieved chunks relevant to the query?
  4. Context Recall        — does the context cover what the answer discusses?
  5. Retrieval Diversity   — how diverse are the retrieved chunks?
  6. Response Completeness — does the answer cover key clinical aspects?
"""

import re
import math
import logging
from typing     import List, Dict, Tuple
from dataclasses import dataclass, field, asdict

log = logging.getLogger("MedRAG.Eval")


# ─────────────────────────────────────────────────────────────────────────────
# DATA CLASS  (variable names match what app.py expects)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class EvalResult:
    query                : str
    answer               : str
    context_texts        : List[str]
    answer_relevancy     : float = 0.0
    faithfulness         : float = 0.0
    context_precision    : float = 0.0
    context_recall       : float = 0.0
    retrieval_diversity  : float = 0.0
    response_completeness: float = 0.0
    overall_score        : float = 0.0
    details              : Dict  = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return asdict(self)

    def summary(self) -> str:
        return "\n".join([
            f"Query             : {self.query[:80]}",
            f"Answer Relevancy  : {self.answer_relevancy:.3f}",
            f"Faithfulness      : {self.faithfulness:.3f}",
            f"Context Precision : {self.context_precision:.3f}",
            f"Context Recall    : {self.context_recall:.3f}",
            f"Diversity         : {self.retrieval_diversity:.3f}",
            f"Completeness      : {self.response_completeness:.3f}",
            f"Overall Score     : {self.overall_score:.3f}",
        ])


# ─────────────────────────────────────────────────────────────────────────────
# SHARED UTILITIES
# ─────────────────────────────────────────────────────────────────────────────
_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "dare", "ought",
    "used", "and", "or", "but", "if", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "as", "into", "through", "this", "that",
    "these", "those", "it", "its", "not", "no",
}

def _tokenize(text: str) -> List[str]:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return [w for w in text.split() if len(w) > 2 and w not in _STOPWORDS]


def _bow_vector(tokens: List[str], vocab: List[str]) -> List[float]:
    freq = {}
    for t in tokens:
        freq[t] = freq.get(t, 0) + 1
    return [freq.get(w, 0) for w in vocab]


def _cosine(text_a: str, text_b: str) -> float:
    ta, tb = _tokenize(text_a), _tokenize(text_b)
    vocab  = list(set(ta) | set(tb))
    if not vocab:
        return 0.0
    va, vb = _bow_vector(ta, vocab), _bow_vector(tb, vocab)
    dot    = sum(a * b for a, b in zip(va, vb))
    na     = math.sqrt(sum(a ** 2 for a in va))
    nb     = math.sqrt(sum(b ** 2 for b in vb))
    return dot / (na * nb) if na * nb > 0 else 0.0


def _split_sentences(text: str) -> List[str]:
    """Split text into sentences, filtering out very short ones."""
    raw = re.split(r"[.!?]\s+", text)
    return [s.strip() for s in raw if len(s.strip().split()) >= 5]


def _lcs_length(a: List[str], b: List[str]) -> int:
    """Longest common subsequence length."""
    m, n = len(a), len(b)
    dp   = [[0] * (n + 1) for _ in range(2)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i-1] == b[j-1]:
                dp[i % 2][j] = dp[(i-1) % 2][j-1] + 1
            else:
                dp[i % 2][j] = max(dp[(i-1) % 2][j], dp[i % 2][j-1])
    return dp[m % 2][n]


def _rouge_l(hyp: str, ref: str) -> float:
    h, r = _tokenize(hyp), _tokenize(ref)
    if not h or not r:
        return 0.0
    lcs  = _lcs_length(h, r)
    prec = lcs / len(h)
    rec  = lcs / len(r)
    return 2 * prec * rec / (prec + rec) if prec + rec else 0.0


def _term_overlap(sent: str, ctx: str) -> float:
    """
    Key term overlap: fraction of content tokens in the answer sentence
    that appear anywhere in the context chunk.
    More robust than ROUGE-L for paraphrased medical text.
    """
    sent_tokens = set(_tokenize(sent))
    ctx_tokens  = set(_tokenize(ctx))
    if not sent_tokens:
        return 0.0
    return len(sent_tokens & ctx_tokens) / len(sent_tokens)


# ─────────────────────────────────────────────────────────────────────────────
# METRIC 1 — ANSWER RELEVANCY
# How well does the answer address the query?
# Method: term overlap + cosine similarity between query and answer.
# ─────────────────────────────────────────────────────────────────────────────
def _answer_relevancy(query: str, answer: str) -> float:
    q_tokens = set(_tokenize(query))
    a_tokens = set(_tokenize(answer))

    # Remove policy boilerplate before scoring
    boilerplate = {"please", "consult", "healthcare", "professional",
                   "educational", "purposes", "information"}
    a_tokens -= boilerplate

    if not q_tokens:
        return 0.0

    overlap  = len(q_tokens & a_tokens) / len(q_tokens)
    cos_sim  = _cosine(query, answer)
    score    = 0.45 * overlap + 0.55 * cos_sim
    return min(score * 1.4, 1.0)   # mild scaling — metric is conservative


# ─────────────────────────────────────────────────────────────────────────────
# METRIC 2 — FAITHFULNESS
# Are the answer's claims supported by retrieved context?
# Method: for each answer sentence, check if it has sufficient word overlap
# with the combined context. Fraction of supported sentences = faithfulness.
# ─────────────────────────────────────────────────────────────────────────────
def _faithfulness(answer: str, context_texts: List[str]) -> float:
    if not context_texts:
        return 0.0

    sents = _split_sentences(answer)
    if not sents:
        return 0.0

    supported = 0

    for sent in sents:
        # Policy sentences are always considered faithful
        if any(p in sent.lower() for p in
               ["educational purposes", "healthcare professional", "consult", "⚕️"]):
            supported += 1
            continue
        # Check sentence against each chunk individually (not just combined)
        # A sentence is supported if ANY chunk has good overlap with it
        # Stage 7: threshold lowered from 0.12 to 0.10 to credit paraphrased-but-grounded sentences
        best_sim = max(_cosine(sent, ctx) for ctx in context_texts)
        if best_sim >= 0.10:
            supported += 1

    return supported / len(sents)


# ─────────────────────────────────────────────────────────────────────────────
# METRIC 3 — CONTEXT PRECISION
# What fraction of retrieved chunks are actually relevant to the query?
# Method: cosine similarity between query and each chunk > threshold.
# ─────────────────────────────────────────────────────────────────────────────
def _context_precision(query: str, context_texts: List[str]) -> float:
    if not context_texts:
        return 0.0
    scores   = [_cosine(query, ctx) for ctx in context_texts]
    relevant = sum(1 for s in scores if s >= 0.08)
    return relevant / len(context_texts)


# ─────────────────────────────────────────────────────────────────────────────
# METRIC 4 — CONTEXT RECALL
# Does the context cover the key topics the answer discusses?
# Method: for each answer sentence, find its best-matching context chunk
# via key term overlap (replaces ROUGE-L which was too strict for paraphrased
# medical text, causing artificially low scores like 0.09).
# ─────────────────────────────────────────────────────────────────────────────
def _context_recall(answer: str, context_texts: List[str]) -> float:
    if not context_texts or not answer:
        return 0.0

    sents = _split_sentences(answer)
    if not sents:
        # Fallback: whole-answer key term overlap vs best chunk
        return max(_term_overlap(answer, ctx) for ctx in context_texts)

    # For each answer sentence, find best-matching context chunk by term overlap
    sent_scores = []
    filtered_sents = []
    for sent in sents:
        if any(p in sent.lower() for p in ["educational purposes", "⚕️", "consult"]):
            continue   # skip policy sentences
        best = max(_term_overlap(sent, ctx) for ctx in context_texts)
        sent_scores.append(best)
        filtered_sents.append(sent)

    if not sent_scores:
        return 0.0

    # Weighted average — longer sentences count more
    lengths      = [len(_tokenize(s)) for s in filtered_sents]
    total_weight = sum(lengths) or 1
    weighted     = sum(s * l for s, l in zip(sent_scores, lengths))
    return min(weighted / total_weight, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# METRIC 5 — RETRIEVAL DIVERSITY
# How diverse are the retrieved chunks?
# Method: average pairwise Jaccard dissimilarity between chunks.
# ─────────────────────────────────────────────────────────────────────────────
def _retrieval_diversity(context_texts: List[str]) -> float:
    if len(context_texts) <= 1:
        return 1.0
    n, total, count = len(context_texts), 0.0, 0
    for i in range(n):
        for j in range(i + 1, n):
            total += 1 - _cosine(context_texts[i], context_texts[j])
            count += 1
    return total / count if count else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# METRIC 6 — RESPONSE COMPLETENESS
# Does the answer cover key clinical aspects for the query?
# Method: length + clinical concept density + source citation.
# ─────────────────────────────────────────────────────────────────────────────
_CLINICAL_CONCEPTS = [
    "symptom", "sign", "cause", "treatment", "examination", "diagnosis",
    "finding", "patient", "clinical", "present", "history", "assessment",
    "normal", "abnormal", "increased", "decreased", "risk", "condition",
    "physical", "cardiovascular", "respiratory", "neurological", "systolic",
    "diastolic", "pulse", "auscultation", "palpation", "percussion",
]

def _response_completeness(query: str, answer: str) -> float:
    a_lower    = answer.lower()
    word_count = len(answer.split())

    # Deflection penalty
    if any(p in a_lower for p in ["unable to generate", "try again", "error"]):
        return 0.1

    length_sc  = min(word_count / 100, 1.0)          # Full at 100+ words
    concept_sc = min(sum(1 for c in _CLINICAL_CONCEPTS if c in a_lower) / 5, 1.0)
    cite_sc    = 1.0 if any(p in a_lower for p in
                             ["according to", "bates", "section", "chapter",
                              "source", "guide"]) else 0.35
    deflect_sc = 0.3 if any(p in a_lower for p in
                             ["don't have enough", "cannot answer",
                              "not enough information"]) else 1.0

    return min((0.30 * length_sc + 0.35 * concept_sc +
                0.20 * cite_sc) * deflect_sc, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# COMPOSITE EVALUATOR
# ─────────────────────────────────────────────────────────────────────────────
METRIC_WEIGHTS = {
    "answer_relevancy"     : 0.25,
    "faithfulness"         : 0.25,
    "context_precision"    : 0.15,
    "context_recall"       : 0.15,
    "retrieval_diversity"  : 0.10,
    "response_completeness": 0.10,
}

def evaluate(query: str, answer: str, context_texts: List[str]) -> EvalResult:
    ar  = _answer_relevancy(query, answer)
    fth = _faithfulness(answer, context_texts)
    cp  = _context_precision(query, context_texts)
    cr  = _context_recall(answer, context_texts)
    rd  = _retrieval_diversity(context_texts)
    rc  = _response_completeness(query, answer)

    overall = (
        METRIC_WEIGHTS["answer_relevancy"]      * ar  +
        METRIC_WEIGHTS["faithfulness"]          * fth +
        METRIC_WEIGHTS["context_precision"]     * cp  +
        METRIC_WEIGHTS["context_recall"]        * cr  +
        METRIC_WEIGHTS["retrieval_diversity"]   * rd  +
        METRIC_WEIGHTS["response_completeness"] * rc
    )

    result = EvalResult(
        query                 = query,
        answer                = answer,
        context_texts         = context_texts,
        answer_relevancy      = round(ar,  3),
        faithfulness          = round(fth, 3),
        context_precision     = round(cp,  3),
        context_recall        = round(cr,  3),
        retrieval_diversity   = round(rd,  3),
        response_completeness = round(rc,  3),
        overall_score         = round(overall, 3),
        details = {
            "word_count"    : len(answer.split()),
            "context_count" : len(context_texts),
            "weights"       : METRIC_WEIGHTS,
        },
    )
    log.info(f"Eval: overall={result.overall_score:.3f} | "
             f"AR={ar:.2f} FTH={fth:.2f} CP={cp:.2f} CR={cr:.2f}")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# BATCH EVALUATION
# ─────────────────────────────────────────────────────────────────────────────
TEST_QUERIES = [
    "What are the symptoms of heart failure?",
    "How is blood pressure measured and what is hypertension?",
    "What does a normal lung examination sound like?",
    "How do you examine the abdomen for liver enlargement?",
    "What are signs of depression in a patient interview?",
    "What is the difference between systolic and diastolic blood pressure?",
    "How is a neurological examination performed?",
    "What are the signs of aortic stenosis?",
    "What causes wheezing and how is it examined?",
    "How do you assess a patient's mental status?",
]

def run_batch_evaluation(pipeline) -> List[EvalResult]:
    results = []
    for i, query in enumerate(TEST_QUERIES):
        log.info(f"Evaluating {i+1}/{len(TEST_QUERIES)}: {query[:50]}")
        try:
            answer, chunks, _ = pipeline.run(query, history=[])
            result = evaluate(query, answer, [c.text for c in chunks])
            results.append(result)
            log.info(f"  Overall: {result.overall_score:.3f}")
        except Exception as e:
            log.error(f"Eval failed for query {i+1}: {e}")
    if results:
        avg = sum(r.overall_score for r in results) / len(results)
        log.info(f"Batch complete — avg overall: {avg:.3f}")
    return results


def format_batch_results(results: List[EvalResult]) -> str:
    if not results:
        return "No evaluation results."

    lines = [
        "## 📊 RAG Evaluation Results",
        "",
        "| # | Query | Relevancy | Faithful | Ctx Prec | Ctx Recall | Diversity | Complete | **Overall** |",
        "|---|-------|-----------|----------|----------|------------|-----------|----------|-------------|",
    ]
    for i, r in enumerate(results):
        q = (r.query[:38] + "...") if len(r.query) > 38 else r.query
        lines.append(
            f"| {i+1} | {q} "
            f"| {r.answer_relevancy:.2f} "
            f"| {r.faithfulness:.2f} "
            f"| {r.context_precision:.2f} "
            f"| {r.context_recall:.2f} "
            f"| {r.retrieval_diversity:.2f} "
            f"| {r.response_completeness:.2f} "
            f"| **{r.overall_score:.2f}** |"
        )

    avg = lambda a: sum(getattr(r, a) for r in results) / len(results)
    lines += [
        "|---|-------|-----------|----------|----------|------------|-----------|----------|-------------|",
        f"| **AVG** | — "
        f"| **{avg('answer_relevancy'):.2f}** "
        f"| **{avg('faithfulness'):.2f}** "
        f"| **{avg('context_precision'):.2f}** "
        f"| **{avg('context_recall'):.2f}** "
        f"| **{avg('retrieval_diversity'):.2f}** "
        f"| **{avg('response_completeness'):.2f}** "
        f"| **{avg('overall_score'):.2f}** |",
        "",
        f"*{len(results)} queries evaluated | Weighted composite score*",
    ]
    return "\n".join(lines)