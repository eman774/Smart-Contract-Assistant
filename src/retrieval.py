"""
retrieval.py
============
Takes a question, retrieves relevant chunks from Chroma,
sends them to the LLM via OpenRouter, and returns the answer with sources.
Includes guardrails to prevent off-topic and hallucinated answers.
"""

from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from src.ingestion import get_embeddings
from src.config import OPENROUTER_API_KEY, LLM_MODEL, TOP_K, CHROMA_DIR

# Load vector store ONCE
_vectorstore = None


def get_vectorstore():
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = Chroma(
            collection_name="contracts",
            embedding_function=get_embeddings(),
            persist_directory=CHROMA_DIR,
        )
    return _vectorstore


def reload_vectorstore():
    """Call this after ingesting a new document."""
    global _vectorstore
    _vectorstore = None
    import gc
    gc.collect()
    return get_vectorstore()


def get_llm(max_tokens: int = 1024):
    return ChatOpenAI(
        model=LLM_MODEL,
        openai_api_key=OPENROUTER_API_KEY,
        openai_api_base="https://api.groq.com/openai/v1",
        temperature=0,
        max_tokens=max_tokens,
        default_headers={
            "HTTP-Referer": "http://localhost:7860",
            "X-Title": "Smart Document Assistant",
        },
    )


SYSTEM_PROMPT = """You are a professional document analysis assistant.
Answer questions strictly based on the document excerpts provided below.

Rules:
- Only use information from the provided excerpts.
- If the answer is not in the document, say: "This information was not found in the document."
- Always mention where you found the information.
- Never make up or assume any information.
- Keep answers concise and clear.

Document excerpts:
{context}
"""

QA_PROMPT = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(SYSTEM_PROMPT),
    HumanMessagePromptTemplate.from_template("{question}"),
])


def answer(question: str, chat_history: list) -> tuple[str, list]:
    vs        = get_vectorstore()
    retriever = vs.as_retriever(search_kwargs={"k": TOP_K})

    # Retrieve relevant chunks
    source_docs = retriever.invoke(question)

    if not source_docs:
        return "No relevant information found in the document.", []

    # Build context
    context = "\n\n---\n\n".join([d.page_content for d in source_docs])

    # ── Guardrail: check if question is relevant ──────────────────────────
    from src.guardrails import is_relevant
    relevant, reason = is_relevant(question, context)
    if not relevant:
        return f"⚠️ **Guardrail:** {reason}", []

    # Build prompt and call LLM
    prompt   = QA_PROMPT.format_messages(context=context, question=question)
    result   = get_llm().invoke(prompt)
    response = result.content

    # ── Guardrail: check if answer is grounded ────────────────────────────
    from src.guardrails import check_grounding
    grounded, warning = check_grounding(response, context)
    if not grounded:
        response += f"\n\n{warning}"

    # Append sources
    if source_docs:
        sources = list({d.metadata.get("source", "?") for d in source_docs})
        response += f"\n\n---\n📄 **Source:** {', '.join(sources)}"

    return response, source_docs


def summarize(file_name: str) -> str:
    vs   = get_vectorstore()
    docs = vs.similarity_search(
        "main topic summary key points overview introduction conclusion",
        k=15,
    )

    filtered = [d for d in docs if d.metadata.get("source") == file_name] or docs
    combined = "\n\n---\n\n".join(
        [f"[Chunk {i+1}]\n{d.page_content}" for i, d in enumerate(filtered[:15])]
    )

    prompt = f"""You are a senior legal analyst specializing in contract review.

Read the contract below carefully and write a professional, structured summary that a non-lawyer can understand.

Use exactly these sections:

## 1. Contract Type & Purpose
What type of contract is this? What is its main purpose?

## 2. Parties Involved
Who are the parties? List their full names and roles (e.g. Service Provider, Client, Employer).

## 3. Key Dates & Duration
When does the contract start and end? Are there any important deadlines or renewal terms?

## 4. Main Obligations
What must each party do? List the key responsibilities for each side clearly.

## 5. Payment Terms
What are the payment amounts, schedule, and method? Any penalties for late payment?

## 6. Termination Conditions
How can this contract be ended? What are the notice periods and conditions?

## 7. Key Risks & Clauses to Watch
Are there any unusual clauses, limitations of liability, or important risks the reader should be aware of?

## 8. Overall Assessment
In 2-3 sentences, summarize what this contract means in plain language.

RULES:
- Write in clear, simple English that a non-lawyer can understand
- Use full sentences and paragraphs, not just bullet points
- If a section is not covered in the contract, write: "Not specified in this contract"
- Do NOT make up information — only use what is in the document

Contract text:
{combined}

Write the full professional summary now:"""

    result = get_llm(max_tokens=2048).invoke(prompt)
    return result.content
