# Evaluation Framework

Custom lightweight RAG evaluation inspired by RAGAS.

## Metrics

### Answer Relevancy
Measures alignment between query and response.

### Faithfulness
Checks whether claims are grounded in retrieved context.

### Context Precision
Fraction of retrieved chunks relevant to the query.

### Context Recall
Measures coverage of answer concepts by retrieved chunks.

### Retrieval Diversity
Measures redundancy between chunks.

### Response Completeness
Checks clinical coverage and citation quality.

---

## Composite Score

Weighted average:

| Metric | Weight |
|---|---|
| Relevancy | 0.25 |
| Faithfulness | 0.25 |
| Precision | 0.15 |
| Recall | 0.15 |
| Diversity | 0.10 |
| Completeness | 0.10 |
