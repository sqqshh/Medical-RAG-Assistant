# Medical RAG Pipeline

## Stages

### 1. Query Expansion

Expands medical terminology using synonym dictionaries.

Example:

```text
heart attack → myocardial infarction, MI
```

---

### 2. Hybrid Retrieval

Combines:

- Dense vector retrieval
- Sparse BM25 retrieval

Using Reciprocal Rank Fusion.

---

### 3. Cross-Encoder Reranking

Refines semantic relevance using:

```text
cross-encoder/ms-marco-MiniLM-L-6-v2
```

---

### 4. MMR Diversification

Reduces redundancy while maximizing coverage.

---

### 5. Prompt Engineering

Builds grounded prompts using retrieved evidence.

---

### 6. Local LLM Generation

Uses:

```text
Qwen2.5-1.5B-Instruct
```

Running fully locally on CPU.
