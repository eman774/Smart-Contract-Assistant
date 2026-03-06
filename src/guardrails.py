"""
guardrails.py
=============
Two-layer guardrail system:
1. Semantic similarity check using embeddings (not just keywords)
2. Grounding check to detect potential hallucinations
"""

import numpy as np


def cosine_similarity(vec1: list, vec2: list) -> float:
    """Calculate cosine similarity between two vectors."""
    a = np.array(vec1)
    b = np.array(vec2)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


def is_relevant(question: str, context: str) -> tuple[bool, str]:
    """
    Semantic similarity guardrail.
    Uses the embedding model to compare question vs document context.
    Much more accurate than keyword matching.
    """
    if not context.strip():
        return False, "No document has been uploaded yet."

    # Hard block: clearly off-topic questions
    off_topic = [
        "weather", "stock price", "recipe", "cook", "sport score",
        "movie", "music", "song", "celebrity", "joke", "how to hack",
        "capital of", "translate to", "what is 2+2",
    ]
    q_lower = question.lower()
    for kw in off_topic:
        if kw in q_lower:
            return False, "This question is not related to the uploaded contract."

    try:
        from src.ingestion import get_embeddings
        embeddings = get_embeddings()

        # Embed the question
        q_vec = embeddings.embed_query(question)

        # Embed a sample of the context (first 500 chars to save time)
        ctx_sample = context[:500]
        c_vec = embeddings.embed_query(ctx_sample)

        # Calculate semantic similarity
        similarity = cosine_similarity(q_vec, c_vec)

        # Threshold: if similarity < 0.15, question is likely off-topic
        if similarity < 0.15:
            return False, (
                "This question does not appear to be related to the uploaded contract. "
                f"(Semantic similarity: {similarity:.2f})"
            )

        return True, ""

    except Exception:
        # Fallback to keyword check if embedding fails
        stop_words = {"what", "who", "when", "where", "why", "how", "is", "are",
                      "the", "a", "an", "in", "of", "to", "and", "or", "this",
                      "contract", "document", "about", "mentioned", "main", "key"}
        q_words     = set(q_lower.split()) - stop_words
        ctx_lower   = context.lower()
        matches     = sum(1 for w in q_words if w in ctx_lower)

        if matches == 0 and len(q_words) > 3:
            return False, "This question does not appear to be related to the uploaded contract."

        return True, ""


def check_grounding(answer: str, context: str) -> tuple[bool, str]:
    """
    Checks if the answer is suspiciously long compared to context.
    """
    if len(answer) > len(context) * 2:
        warning = "\n\n⚠️ *Warning: This answer may contain information beyond the document. Please verify.*"
        return False, warning
    return True, ""
