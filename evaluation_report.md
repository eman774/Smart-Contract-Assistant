# RAG Evaluation Report
*2026-03-06 22:31*

*Based on 3 question(s) from your session.*

## Summary
| Metric | Value | Meaning |
|---|---|---|
| Hit Rate | 100% | Questions with relevant chunks found |
| Citation Rate | 100% | Answers with source references |
| Avg Relevance | 67% | How well answers matched questions (LLM-as-Judge) |
| Avg Groundedness | 60% | How much answers were based on document |
| Avg Retrieval Time | 0.037s | Time to find relevant chunks |
| Avg Answer Time | 0.59s | Total time to generate answer |

## Per-Question Results
### Q1: What is this contract about?
- **Chunks:** 5/5 | **Retrieval:** 0.032s | **Answer time:** 0.48s
- **Relevance:** 100% | **Groundedness:** 100% | **Citation:** ✅ | **Quality:** Good
- **Judge (LLM-as-Judge):** The answer is based directly on the document title "SERVICE AGREEMENT" found on Page 1.
- **Answer:** *This contract is about a service agreement between a principal and a service provider.   Found on Page 1: "SERVICE AGREEMENT"  --- 📄 **Source:** sample-service-agreement.pdf...*

### Q2: What are the payment terms?
- **Chunks:** 5/5 | **Retrieval:** 0.032s | **Answer time:** 0.53s
- **Relevance:** 0% | **Groundedness:** 0% | **Citation:** ✅ | **Quality:** Not found in document
- **Judge (LLM-as-Judge):** The answer is not based on the provided document as the payment terms are not mentioned.
- **Answer:** *This information was not found in the document.  --- 📄 **Source:** sample-service-agreement.pdf...*

### Q3: What are the main obligations?
- **Chunks:** 5/5 | **Retrieval:** 0.048s | **Answer time:** 0.75s
- **Relevance:** 100% | **Groundedness:** 80% | **Citation:** ✅ | **Quality:** Detailed
- **Judge (LLM-as-Judge):** The answer accurately identifies the main obligations but only partially addresses the specific request for the service provider's responsibilities.
- **Answer:** *The main obligations can be found in sections 8 (Service provider's responsibilities) and 9-12 (Principal's responsibilities).  For the Service provider's responsibilities:  - Provide services as and when the principal reasonably requires (8.1) - Car...*

## Known Limitations
- Single document at a time
- English contracts only
- LLM response time varies with free API tier
- LLM-as-Judge scores are approximations
- Guardrails use semantic similarity with fixed threshold