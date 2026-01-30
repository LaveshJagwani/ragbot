import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceInferenceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq

load_dotenv(dotenv_path=Path(".env"))

# ---- Embeddings ----
from langchain_huggingface import HuggingFaceInferenceEmbeddings

embeddings = HuggingFaceInferenceEmbeddings(
    api_key=os.getenv("HUGGINGFACE_API_KEY"),
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# ---- Vector Store ----
vectorstore = PineconeVectorStore(
    index_name=os.getenv("PINECONE_INDEX_NAME"),
    embedding=embeddings,
    pinecone_api_key=os.getenv("PINECONE_API_KEY"),
)

# ---- LLM ----
llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.1-8b-instant",
    temperature=0
)
