"""
rag_pipeline.py
===============
Medical RAG Pipeline — Bates' Guide to Physical Examination
Stages: Query Expansion → Hybrid Retrieval (Dense+BM25) → Cross-Encoder
        Rerank → MMR → Prompt → Qwen2.5-1.5B-Instruct (local CPU inference)
"""

import os, re, math, time, logging, zipfile, threading, hashlib, json
from pathlib import Path
from typing  import List, Dict, Tuple
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import torch

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("MedRAG")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = Path("/tmp/medical_rag")
CHROMA_DIR = BASE_DIR / "chromadb_store"
QUERY_LOG  = BASE_DIR / "query_log.jsonl"

# ── Environment ───────────────────────────────────────────────────────────────
HF_DATASET_REPO = os.environ.get("HF_DATASET_REPO", "")
HF_TOKEN        = os.environ.get("HF_TOKEN", "")

# ── Models ────────────────────────────────────────────────────────────────────
EMBED_MODEL_NAME   = "BAAI/bge-base-en-v1.5"
CROSS_ENCODER_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
LLM_MODEL_NAME     = "Qwen/Qwen2.5-1.5B-Instruct"

# ── Retrieval ─────────────────────────────────────────────────────────────────
COLLECTION_NAME  = "bates_medical_rag"
TOP_K_DENSE      = 20
TOP_K_BM25       = 20
TOP_K_RERANK     = 10
TOP_K_MMR        = 5
MMR_LAMBDA       = 0.65
SIMILARITY_FLOOR = 0.35
DENSE_WEIGHT     = 0.60
BM25_WEIGHT      = 0.40
K_RRF            = 60

# ── Generation ────────────────────────────────────────────────────────────────
MAX_NEW_TOKENS     = 400
TEMPERATURE        = 0.25
TOP_P              = 0.85
REPETITION_PENALTY = 1.15
MAX_CTX_CHARS      = 5000
MAX_HISTORY_TURNS  = 3


# ─────────────────────────────────────────────────────────────────────────────
# DATA CLASS
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class RetrievedChunk:
    text         : str
    metadata     : Dict
    dense_score  : float = 0.0
    bm25_score   : float = 0.0
    fusion_score : float = 0.0
    ce_score     : float = 0.0
    final_score  : float = 0.0

    @property
    def source_label(self) -> str:
        m = self.metadata
        return (f"{m.get('chapter_name','')[:45]} › "
                f"{m.get('section_name','')[:35]}  (p.{m.get('page_start','?')})")


# ─────────────────────────────────────────────────────────────────────────────
# 1. CHROMADB LOADER
# ─────────────────────────────────────────────────────────────────────────────
class ChromaLoader:
    _collection = None
    _lock       = threading.Lock()

    @classmethod
    def get_collection(cls):
        with cls._lock:
            if cls._collection is None:
                cls._collection = cls._load()
        return cls._collection

    @classmethod
    def _load(cls):
        import chromadb
        from chromadb.config import Settings
        from huggingface_hub import hf_hub_download

        BASE_DIR.mkdir(parents=True, exist_ok=True)
        if not CHROMA_DIR.exists() or not any(CHROMA_DIR.iterdir()):
            log.info(f"Downloading ChromaDB from HF Dataset: {HF_DATASET_REPO}")
            zip_path = hf_hub_download(
                repo_id   = HF_DATASET_REPO,
                filename  = "chromadb_store.zip",
                repo_type = "dataset",
                token     = HF_TOKEN or None,
                local_dir = str(BASE_DIR),
            )
            log.info("Extracting...")
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(BASE_DIR)
            log.info(f"Extracted to {CHROMA_DIR}")

        client     = chromadb.PersistentClient(
            path     = str(CHROMA_DIR),
            settings = Settings(anonymized_telemetry=False),
        )
        collection = client.get_collection(COLLECTION_NAME)
        log.info(f"ChromaDB ready — {collection.count()} documents")
        return collection


# ─────────────────────────────────────────────────────────────────────────────
# 2. EMBEDDING MODEL
# ─────────────────────────────────────────────────────────────────────────────
class EmbedModel:
    _inst = None
    _lock = threading.Lock()

    @classmethod
    def get(cls):
        with cls._lock:
            if cls._inst is None:
                cls._inst = cls._load()
        return cls._inst

    @classmethod
    def _load(cls):
        from sentence_transformers import SentenceTransformer
        device = "cuda" if torch.cuda.is_available() else "cpu"
        log.info(f"Loading embed model {EMBED_MODEL_NAME} on {device}")
        m = SentenceTransformer(EMBED_MODEL_NAME, device=device)
        m.eval()
        log.info("Embed model ready")
        return m

    @classmethod
    def encode_query(cls, text: str) -> np.ndarray:
        with torch.no_grad():
            return cls.get().encode(
                [f"Represent this medical question: {text}"],
                normalize_embeddings = True,
                convert_to_numpy     = True,
                show_progress_bar    = False,
            )


# ─────────────────────────────────────────────────────────────────────────────
# 3. CROSS-ENCODER
# ─────────────────────────────────────────────────────────────────────────────
class CrossEncoder:
    _inst = None
    _lock = threading.Lock()

    @classmethod
    def get(cls):
        with cls._lock:
            if cls._inst is None:
                cls._inst = cls._load()
        return cls._inst

    @classmethod
    def _load(cls):
        from sentence_transformers import CrossEncoder as CE
        log.info(f"Loading cross-encoder {CROSS_ENCODER_NAME}")
        m = CE(CROSS_ENCODER_NAME, max_length=512)
        log.info("Cross-encoder ready")
        return m

    @classmethod
    def score(cls, query: str, texts: List[str]) -> List[float]:
        pairs = [[query, t[:600]] for t in texts]
        return cls.get().predict(pairs, show_progress_bar=False).tolist()


# ─────────────────────────────────────────────────────────────────────────────
# 4. LLM — Qwen2.5-1.5B-Instruct loaded locally
# ─────────────────────────────────────────────────────────────────────────────
class LLMModel:
    """
    Qwen/Qwen2.5-1.5B-Instruct loaded locally on CPU.
    fp32 on CPU — no GPU or bitsandbytes needed.
    Model is ~3GB RAM, generation ~60-120s on CPU Basic.
    """
    _bundle = None
    _lock   = threading.Lock()

    @classmethod
    def get(cls):
        with cls._lock:
            if cls._bundle is None:
                cls._bundle = cls._load()
        return cls._bundle

    @classmethod
    def _load(cls):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        log.info(f"Loading {LLM_MODEL_NAME} locally on CPU...")
        tok = AutoTokenizer.from_pretrained(
            LLM_MODEL_NAME,
            trust_remote_code = True,
        )
        mdl = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL_NAME,
            torch_dtype       = torch.float32,
            device_map        = "cpu",
            trust_remote_code = True,
            low_cpu_mem_usage = True,
        )
        mdl.eval()
        log.info("LLM ready")
        return {"model": mdl, "tokenizer": tok}

    @classmethod
    def generate(cls, prompt: str) -> str:
        bundle = cls.get()
        tok, mdl = bundle["tokenizer"], bundle["model"]
        try:
            inputs = tok(
                prompt,
                return_tensors = "pt",
                truncation     = True,
                max_length     = 3072,
            )
            with torch.no_grad():
                out = mdl.generate(
                    **inputs,
                    max_new_tokens     = MAX_NEW_TOKENS,
                    temperature        = TEMPERATURE,
                    top_p              = TOP_P,
                    repetition_penalty = REPETITION_PENALTY,
                    do_sample          = True,
                    pad_token_id       = tok.eos_token_id,
                    eos_token_id       = tok.eos_token_id,
                )
            new_ids = out[0][inputs["input_ids"].shape[1]:]
            result  = tok.decode(new_ids, skip_special_tokens=True).strip()
            log.info(f"Generation done — {len(result)} chars")
            return result
        except Exception as e:
            log.error(f"LLM generation error: {type(e).__name__}: {e}")
            return (
                "I encountered an error generating a response. "
                "Please try again in a moment."
            )


# ─────────────────────────────────────────────────────────────────────────────
# 5. BM25 INDEX (disk-cached)
# ─────────────────────────────────────────────────────────────────────────────
class BM25Index:
    """Built once from ChromaDB corpus, cached to disk to avoid rebuild on restart."""
    _inst      = None
    _lock      = threading.Lock()
    CACHE_PATH = BASE_DIR / "bm25_cache.pkl"

    @classmethod
    def get(cls):
        with cls._lock:
            if cls._inst is None:
                cls._inst = cls._load_or_build()
        return cls._inst

    @classmethod
    def _load_or_build(cls):
        import pickle
        if cls.CACHE_PATH.exists():
            log.info("Loading BM25 index from disk cache...")
            try:
                with open(cls.CACHE_PATH, "rb") as f:
                    bundle = pickle.load(f)
                log.info(f"BM25 cache loaded — {len(bundle['docs'])} documents")
                return bundle
            except Exception as e:
                log.warning(f"Cache load failed ({e}), rebuilding...")
        return cls._build()

    @classmethod
    def _build(cls):
        import pickle
        from rank_bm25 import BM25Okapi
        log.info("Building BM25 index from ChromaDB corpus...")
        col  = ChromaLoader.get_collection()
        all_docs, all_ids, all_metas = [], [], []
        offset = 0
        while offset < col.count():
            res = col.get(limit=1000, offset=offset,
                          include=["documents", "metadatas"])
            all_docs.extend(res["documents"])
            all_ids.extend(res["ids"])
            all_metas.extend(res["metadatas"])
            offset += 1000

        tokenized = [cls._tokenize(d) for d in all_docs]
        bundle    = {"index": BM25Okapi(tokenized),
                     "docs": all_docs, "ids": all_ids, "metas": all_metas}
        try:
            cls.CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(cls.CACHE_PATH, "wb") as f:
                pickle.dump(bundle, f)
            log.info(f"BM25 index cached — {len(all_docs)} documents")
        except Exception as e:
            log.warning(f"Could not cache BM25 index: {e}")
        return bundle

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        text = text.lower()
        text = re.sub(r"[^\w\s]", " ", text)
        return [w for w in text.split() if len(w) > 1]

    @classmethod
    def search(cls, query: str, top_k: int = TOP_K_BM25) -> List[Dict]:
        bundle   = cls.get()
        tokens   = cls._tokenize(query)
        scores   = bundle["index"].get_scores(tokens)
        top_idxs = np.argsort(scores)[::-1][:top_k]
        return [
            {"id": bundle["ids"][i], "text": bundle["docs"][i],
             "meta": bundle["metas"][i], "score": float(scores[i])}
            for i in top_idxs if scores[i] > 0
        ]


# ─────────────────────────────────────────────────────────────────────────────
# 6. QUERY EXPANSION
# ─────────────────────────────────────────────────────────────────────────────
MEDICAL_SYNONYMS: Dict[str, List[str]] = {
    "heart failure"       : ["cardiac failure", "congestive heart failure", "CHF"],
    "heart attack"        : ["myocardial infarction", "MI", "acute coronary syndrome"],
    "high blood pressure" : ["hypertension", "HTN"],
    "low blood pressure"  : ["hypotension"],
    "stroke"              : ["cerebrovascular accident", "CVA", "TIA"],
    "diabetes"            : ["diabetes mellitus", "hyperglycemia", "DM"],
    "shortness of breath" : ["dyspnea", "breathlessness", "respiratory distress"],
    "chest pain"          : ["angina", "chest tightness", "precordial pain"],
    "depression"          : ["major depressive disorder", "MDD", "depressive episode"],
    "anxiety"             : ["anxiety disorder", "panic disorder"],
    "kidney"              : ["renal", "nephro"],
    "liver"               : ["hepatic", "hepato"],
    "lung"                : ["pulmonary", "respiratory"],
    "stomach"             : ["gastric", "gastrointestinal"],
    "headache"            : ["cephalalgia", "migraine"],
    "fatigue"             : ["tiredness", "weakness", "malaise"],
    "swelling"            : ["edema", "oedema"],
    "fever"               : ["pyrexia", "febrile"],
    "dizziness"           : ["vertigo", "lightheadedness", "syncope"],
    "cough"               : ["productive cough", "hemoptysis"],
    "joint pain"          : ["arthralgia", "arthritis"],
}

EXAM_BIAS_SUFFIX = "physical examination findings signs symptoms"

def expand_query(query: str) -> List[str]:
    queries     = [query]
    query_lower = query.lower()
    added_terms = []
    for lay_term, medical_terms in MEDICAL_SYNONYMS.items():
        if lay_term in query_lower:
            added_terms.extend(medical_terms[:2])
        for med in medical_terms:
            if med.lower() in query_lower:
                added_terms.append(lay_term)
                break
    if added_terms:
        queries.append(query + " " + " ".join(added_terms))
    else:
        queries.append(query + " " + EXAM_BIAS_SUFFIX)
    queries = queries[:2]
    log.info(f"Query expanded: {len(queries)} variants")
    return queries


# ─────────────────────────────────────────────────────────────────────────────
# 7. HYBRID RETRIEVAL + RRF FUSION
# ─────────────────────────────────────────────────────────────────────────────
def hybrid_retrieve(query: str, expanded_queries: List[str]) -> List[RetrievedChunk]:
    col         = ChromaLoader.get_collection()
    rrf_scores  : Dict[str, float]          = defaultdict(float)
    chunk_store : Dict[str, RetrievedChunk] = {}

    for q in expanded_queries:
        q_emb = EmbedModel.encode_query(q)
        res   = col.query(query_embeddings=q_emb.tolist(), n_results=TOP_K_DENSE,
                          include=["documents", "metadatas", "distances"])
        for rank, (doc, meta, dist) in enumerate(
            zip(res["documents"][0], res["metadatas"][0], res["distances"][0])
        ):
            cid   = hashlib.md5(doc[:200].encode()).hexdigest()
            score = 1 - dist
            if score < SIMILARITY_FLOOR:
                continue
            rrf_scores[cid] += DENSE_WEIGHT * (1 / (K_RRF + rank + 1))
            if cid not in chunk_store:
                chunk_store[cid] = RetrievedChunk(text=doc, metadata=meta,
                                                  dense_score=score)
            else:
                chunk_store[cid].dense_score = max(chunk_store[cid].dense_score, score)

    bm25_results = BM25Index.search(query, top_k=TOP_K_BM25)
    max_bm25     = max((r["score"] for r in bm25_results), default=1.0)
    for rank, r in enumerate(bm25_results):
        cid = hashlib.md5(r["text"][:200].encode()).hexdigest()
        rrf_scores[cid] += BM25_WEIGHT * (1 / (K_RRF + rank + 1))
        if cid not in chunk_store:
            chunk_store[cid] = RetrievedChunk(text=r["text"], metadata=r["meta"],
                                              bm25_score=r["score"]/max_bm25)
        else:
            chunk_store[cid].bm25_score = r["score"] / max_bm25

    for cid, chunk in chunk_store.items():
        chunk.fusion_score = rrf_scores[cid]

    ranked = sorted(chunk_store.values(), key=lambda c: c.fusion_score, reverse=True)
    log.info(f"Hybrid retrieval: {len(ranked)} candidates")
    return ranked[:TOP_K_RERANK]


# ─────────────────────────────────────────────────────────────────────────────
# 8. CROSS-ENCODER RERANKING
# ─────────────────────────────────────────────────────────────────────────────
def rerank_with_cross_encoder(query: str, chunks: List[RetrievedChunk]) -> List[RetrievedChunk]:
    if not chunks:
        return []
    scores = CrossEncoder.score(query, [c.text for c in chunks])
    for chunk, sc in zip(chunks, scores):
        chunk.ce_score    = float(sc)
        chunk.final_score = 0.70 * (1 / (1 + math.exp(-sc))) + 0.30 * chunk.fusion_score
    reranked = sorted(chunks, key=lambda c: c.final_score, reverse=True)
    log.info(f"Reranking done — top CE score: {reranked[0].ce_score:.3f}")
    return reranked


# ─────────────────────────────────────────────────────────────────────────────
# 9. MMR DIVERSIFICATION
# ─────────────────────────────────────────────────────────────────────────────
def mmr_select(chunks: List[RetrievedChunk], top_k: int = TOP_K_MMR) -> List[RetrievedChunk]:
    if len(chunks) <= top_k:
        return chunks
    selected, remaining = [], list(chunks)
    word_sets = {id(c): set(c.text.lower().split()) for c in remaining}
    while len(selected) < top_k and remaining:
        if not selected:
            best = max(remaining, key=lambda c: c.final_score)
        else:
            sel_words = [word_sets[id(s)] for s in selected]
            best = max(
                remaining,
                key=lambda c: (
                    MMR_LAMBDA * c.final_score -
                    (1 - MMR_LAMBDA) * max(
                        len(word_sets[id(c)] & sw) / max(len(word_sets[id(c)] | sw), 1)
                        for sw in sel_words
                    )
                )
            )
        selected.append(best)
        remaining.remove(best)
    log.info(f"MMR selected {len(selected)} chunks")
    return selected


# ─────────────────────────────────────────────────────────────────────────────
# 10. PROMPT ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are MedAssist, a clinical education assistant based on Bates' Guide to Physical Examination.
RULES:
1. Answer ONLY using the CONTEXT provided. Copy relevant sentences from the context directly into your answer where possible.
2. If context is insufficient, say so honestly. Never diagnose a patient.
3. Cite the source section: "According to [section name]..."
4. End every answer with: "⚕️ For educational purposes only. Consult a healthcare professional." """


def build_prompt(query: str, chunks: List[RetrievedChunk],
                 history: List[Tuple[str, str]]) -> str:
    ctx_blocks, total = [], 0
    for i, c in enumerate(chunks):
        block = (f"[Source {i+1}: {c.metadata.get('chapter_name','')[:40]} | "
                 f"{c.metadata.get('section_name','')[:35]} | "
                 f"p.{c.metadata.get('page_start','?')}]\n{c.text}")
        if total + len(block) > MAX_CTX_CHARS:
            break
        ctx_blocks.append(block)
        total += len(block)

    context_str = "\n\n".join(ctx_blocks) or "No relevant context retrieved."

    emergency_note = (
        "⚠️ If this is a medical emergency, call 911 immediately.\n\n"
        if any(t in query.lower() for t in
               ["chest pain", "can't breathe", "stroke", "unconscious",
                "heart attack", "severe bleeding"])
        else ""
    )

    history_str = ""
    for user_msg, asst_msg in history[-MAX_HISTORY_TURNS:]:
        history_str += f"Previous Q: {user_msg[:200]}\nPrevious A: {asst_msg[:300]}\n\n"

    # Qwen2.5 native chat template format
    user_msg = (
        f"{emergency_note}"
        f"CONTEXT FROM BATES GUIDE:\n{context_str}\n\n"
        f"QUESTION: {query}\n\n"
        f"Answer using ONLY the context above. Be clear and cite the source section."
    )
    prompt = f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
    for u, a in history[-MAX_HISTORY_TURNS:]:
        prompt += f"<|im_start|>user\n{u[:200]}<|im_end|>\n"
        prompt += f"<|im_start|>assistant\n{a[:350]}<|im_end|>\n"
    prompt += f"<|im_start|>user\n{user_msg}<|im_end|>\n<|im_start|>assistant\n"
    return prompt


# ─────────────────────────────────────────────────────────────────────────────
# 11. MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
class MedicalRAGPipeline:
    def __init__(self):
        log.info("Initialising MedicalRAGPipeline...")
        self.collection = ChromaLoader.get_collection()
        self.embed      = EmbedModel.get()
        self.ce         = CrossEncoder.get()
        self.llm_client = LLMModel.get()
        self.bm25       = BM25Index.get()
        log.info("MedicalRAGPipeline ready ✓")

    def run(self, query: str,
            history: List[Tuple[str, str]]) -> Tuple[str, List[RetrievedChunk], Dict]:
        t0, timings = time.time(), {}

        t = time.time()
        expanded = expand_query(query)
        timings["query_expansion_ms"] = round((time.time() - t) * 1000)

        t = time.time()
        candidates = hybrid_retrieve(query, expanded)
        timings["hybrid_retrieval_ms"] = round((time.time() - t) * 1000)

        if not candidates:
            return (
                "I couldn't find relevant information in Bates' Guide for your question. "
                "Please try rephrasing or consult a healthcare professional.\n\n"
                "⚕️ For educational purposes only.",
                [], timings
            )

        t = time.time()
        reranked = rerank_with_cross_encoder(query, candidates)
        timings["reranking_ms"] = round((time.time() - t) * 1000)

        t = time.time()
        final_chunks = mmr_select(reranked, top_k=TOP_K_MMR)
        timings["mmr_ms"] = round((time.time() - t) * 1000)

        t = time.time()
        prompt = build_prompt(query, final_chunks, history)
        timings["prompt_build_ms"] = round((time.time() - t) * 1000)
        log.info(f"Prompt built — {len(prompt)} chars")

        t = time.time()
        answer = LLMModel.generate(prompt)
        timings["generation_ms"] = round((time.time() - t) * 1000)

        timings["total_ms"] = round((time.time() - t0) * 1000)
        log.info(f"Pipeline complete in {timings['total_ms']}ms")

        # ── Confidence signal ─────────────────────────────────────────────────
        # Low confidence if top chunk relevance is weak or very few chunks passed
        # the similarity floor. Surfaced in app.py as a warning banner.
        top_score      = final_chunks[0].final_score if final_chunks else 0.0
        avg_ce         = (sum(c.ce_score for c in final_chunks) / len(final_chunks)
                          if final_chunks else 0.0)
        low_confidence = top_score < 0.35 or avg_ce < 0.0 or len(final_chunks) < 3
        timings["top_chunk_score"] = round(top_score, 3)
        timings["avg_ce_score"]    = round(avg_ce, 3)
        timings["low_confidence"]  = low_confidence

        # ── Query logging (enterprise audit trail) ────────────────────────────
        try:
            BASE_DIR.mkdir(parents=True, exist_ok=True)
            log_entry = {
                "ts"            : time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "query"         : query,
                "answer_chars"  : len(answer),
                "chunks_used"   : len(final_chunks),
                "top_score"     : timings["top_chunk_score"],
                "avg_ce"        : timings["avg_ce_score"],
                "low_confidence": low_confidence,
                "total_ms"      : timings["total_ms"],
                "sources"       : [
                    {"chapter": c.metadata.get("chapter_name", "")[:40],
                     "section": c.metadata.get("section_name", "")[:35],
                     "page"   : c.metadata.get("page_start", "?")}
                    for c in final_chunks
                ],
            }
            with open(QUERY_LOG, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry) + "\n")
        except Exception as e:
            log.warning(f"Query log write failed: {e}")

        return answer, final_chunks, timings