"""
ingestion.py
============
Reads a PDF or DOCX file, splits it into chunks,
generates embeddings, and stores them in Chroma.
"""

import traceback
from pathlib import Path
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import fitz
from docx import Document as DocxFile
from src.config import CHUNK_SIZE, CHUNK_OVERLAP, CHROMA_DIR

# Load embedding model ONCE when the app starts
print("Loading embedding model... (first time only)")
_embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)
print("Embedding model loaded ✅")


def get_embeddings():
    return _embeddings


def read_pdf(path: str) -> str:
    doc = fitz.open(path)
    pages = []
    for i, page in enumerate(doc, start=1):
        text = page.get_text("text").strip()
        if text:
            pages.append(f"[Page {i}]\n{text}")
    doc.close()
    if not pages:
        raise ValueError("The PDF is empty or contains no readable text.")
    return "\n\n".join(pages)


def read_docx(path: str) -> str:
    doc = DocxFile(path)
    paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    if not paragraphs:
        raise ValueError("The DOCX file is empty.")
    return "\n\n".join(paragraphs)


def read_file(path: str) -> str:
    suffix = Path(path).suffix.lower()
    if suffix == ".pdf":
        return read_pdf(path)
    elif suffix in (".docx", ".doc"):
        return read_docx(path)
    else:
        raise ValueError(f"Unsupported file type '{suffix}'. Please upload a PDF or DOCX.")


def split_text(text: str, file_name: str) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " "],
    )
    chunks = splitter.split_text(text)
    return [
        Document(
            page_content=chunk,
            metadata={"source": file_name, "chunk": i, "total": len(chunks)},
        )
        for i, chunk in enumerate(chunks)
    ]


def ingest(file_path: str) -> dict:
    try:
        file_name = Path(file_path).name
        print(f"[1/3] Reading: {file_name}")
        text = read_file(file_path)

        print(f"[2/3] Splitting into chunks...")
        docs = split_text(text, file_name)

        print(f"[3/3] Storing {len(docs)} chunks in Chroma...")

        # Delete old collection and create fresh one (avoids Windows file lock issue)
        import chromadb
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        try:
            client.delete_collection("contracts")
            print("Old collection deleted ✅")
        except Exception:
            pass  # collection didn't exist yet

        # Now store new documents
        Chroma.from_documents(
            documents=docs,
            embedding=_embeddings,
            persist_directory=CHROMA_DIR,
            collection_name="contracts",
        )
        print("Done! ✅")

        return {
            "file_name": file_name,
            "characters": len(text),
            "chunks": len(docs),
        }
    except Exception as e:
        print(f"ERROR: {e}")
        traceback.print_exc()
        raise
