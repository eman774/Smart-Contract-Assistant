import os
from dotenv import load_dotenv

load_dotenv()

# Groq API Key
OPENROUTER_API_KEY = os.getenv("GROQ_API_KEY", "")

# Groq free model - fast!
LLM_MODEL = "llama-3.1-8b-instant"

# Chunking settings
CHUNK_SIZE    = 800
CHUNK_OVERLAP = 100

# How many chunks to retrieve per question
TOP_K = 5

# Where to save the vector database
CHROMA_DIR = "./data/chroma_db"