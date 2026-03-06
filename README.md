# 📄 Smart Contract Assistant

> An intelligent contract analysis tool powered by RAG (Retrieval Augmented Generation).  
> Upload any contract, ask questions in plain English, and get accurate answers with source citations.

Built with **LangChain · Groq · Chroma · Gradio · FastAPI · LangServe**

---

## ✨ Features

| Feature | Description |
|---|---|
| 📤 **Document Upload** | Supports PDF and DOCX contracts |
| 💬 **Conversational Q&A** | Ask questions with full chat history |
| 🛡️ **Guardrails** | Semantic similarity blocks off-topic questions |
| 📋 **Auto Summary** | Structured 8-section contract summary |
| 📊 **Evaluation** | LLM-as-Judge scoring with downloadable report |
| 🔌 **REST API** | FastAPI endpoints for developer integration |
| 🦜 **LangServe** | Interactive chain playground at `/qa/playground` |

---

## 🗂️ Project Structure

```
RAG_project/
│
├── app.py                   # Gradio UI — main user interface
├── server.py                # FastAPI + LangServe backend API
├── requirements.txt         # Python dependencies
├── .env                     # API keys (never commit this)
│
├── src/
│   ├── __init__.py
│   ├── config.py            # Settings: model, chunk size, paths
│   ├── ingestion.py         # PDF/DOCX parsing, chunking, embedding
│   ├── retrieval.py         # Vector search + LLM answer generation
│   ├── guardrails.py        # Semantic similarity input validation
│   └── evaluation.py        # LLM-as-Judge evaluation pipeline
│
└── data/
    └── chroma_db/           # Local vector database (auto-created)
```

---

## 🏗️ System Architecture

```
                        ┌─────────────────────────────────────┐
                        │           RAG Pipeline               │
                        │                                      │
  PDF / DOCX  ─────────►│  Extract ──► Chunk ──► Embed        │
                        │                          │           │
                        │                     Chroma DB        │
                        │                          │           │
  User Question ────────►│  Embed Question          │           │
                        │       │                  │           │
                        │       ▼                  ▼           │
                        │    Semantic Search ──► Top-K Chunks  │
                        │                          │           │
                        │                     Guardrails       │
                        │                          │           │
                        │                    LLM (Groq)        │
                        │                          │           │
                        │               Answer + Source Cite   │
                        └─────────────────────────────────────┘
                                          │
                   ┌──────────────────────┼──────────────────────┐
                   ▼                      ▼                       ▼
            Gradio UI               FastAPI REST            LangServe
          (localhost:7860)        (localhost:8000)       (/qa/playground)
```

---

## 🛠️ Technology Stack

| Layer | Technology |
|---|---|
| LLM | Groq — `llama-3.1-8b-instant` (free tier) |
| Embeddings | SentenceTransformers — `all-MiniLM-L6-v2` (local) |
| Vector Store | ChromaDB (local persistence) |
| RAG Framework | LangChain + LCEL |
| UI | Gradio 3.x |
| Backend API | FastAPI + Uvicorn |
| Chain Serving | LangServe |
| File Parsing | PyMuPDF (PDF), python-docx (DOCX) |
| Evaluation | LLM-as-Judge + heuristic fallback |

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.10+
- A free [Groq API key](https://console.groq.com)

### 1. Clone the repository
```bash
git clone https://github.com/eman774/RAG_project.git
cd RAG_project
```

### 2. Create a virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac / Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure API key
Create a `.env` file in the root folder:
```
GROQ_API_KEY=gsk_your_key_here
```
Get your free key at: https://console.groq.com

### 5. Run the application

**Gradio UI:**
```bash
python app.py
```
Open: **http://127.0.0.1:7860**

**FastAPI + LangServe (optional, separate terminal):**
```bash
pip install sse-starlette
python server.py
```
Open: **http://127.0.0.1:8000/docs**

---

## 💬 How to Use

### Gradio UI
1. Go to the **Upload Document** tab
2. Upload a PDF or DOCX contract
3. Go to the **Chat** tab and ask questions
4. Use the **Summary** tab to generate a full contract summary
5. Use the **Evaluation** tab to measure system quality

### REST API
```python
import requests

# 1. Upload a contract
with open("contract.pdf", "rb") as f:
    requests.post("http://localhost:8000/ingest", files={"file": f})

# 2. Ask a question
response = requests.post("http://localhost:8000/chat",
    json={"question": "Who are the parties in this contract?"})
print(response.json()["answer"])

# 3. Generate summary
summary = requests.post("http://localhost:8000/summarize")
print(summary.json()["summary"])
```

### LangServe Playground
1. Run `python server.py`
2. Open `http://127.0.0.1:8000/qa/playground`
3. Type: `{"question": "What is this contract about?"}`
4. Click **Start**

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | API status and available routes |
| GET | `/health` | Health check |
| POST | `/ingest` | Upload and process a PDF or DOCX file |
| POST | `/chat` | Ask a question about the contract |
| POST | `/summarize` | Generate a structured contract summary |
| GET | `/evaluate` | Run retrieval evaluation metrics |
| POST | `/qa/invoke` | LangServe — invoke the RAG chain directly |
| GET | `/qa/playground` | LangServe — interactive chain playground |

---

## 🛡️ Guardrails

The system uses two layers of protection:

**Input Guardrail** — Before answering, the question is compared to the document using cosine similarity on embeddings. If the semantic similarity is too low, the question is blocked with a clear message.

**Output Guardrail** — After generating an answer, the system checks whether the answer length is disproportionate to the retrieved context, which may indicate hallucination.

---

## 📊 Evaluation Pipeline

The evaluation tab runs automatically after you ask questions in Chat:

- **Hit Rate** — Did the retriever find relevant chunks for every question?
- **Relevance Score** — Does the answer actually address the question? (LLM-as-Judge)
- **Groundedness Score** — Is the answer based on the document, not made up? (LLM-as-Judge)
- **Citation Rate** — Did every answer include a source reference?
- **Response Time** — How fast is retrieval and answer generation?

Results are displayed in the UI and can be downloaded as a Markdown report.

---

## ⚠️ Known Limitations

- Processes one document at a time — uploading a new file replaces the previous one
- English contracts only — other languages may produce poor results
- Free Groq API has daily rate limits
- LLM-as-Judge scores are approximations, not ground truth
- Not suitable for production legal or compliance use without further validation

---

## 🚀 Future Enhancements

- [ ] Multi-document search
- [ ] Arabic language support
- [ ] Domain-specific fine-tuned legal model
- [ ] User authentication and role-based access
- [ ] Docker containerization
- [ ] Cloud deployment (AWS / Azure)

---

## 📝 Project Context

This project was built as part of the **NVIDIA DLI Workshop** on LLM Pipelines and RAG Applications.

> **Domain:** LLM Pipelines · LangChain · Vector Stores · Gradio · LangServe  
> **Type:** Workshop Application Project
