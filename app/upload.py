import os
import tempfile
from fastapi import UploadFile

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore

from app.config import embeddings
from app.pinecone_utils import get_or_create_index


def process_pdf_and_index(company_id: str, file: UploadFile):
    # Save PDF temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file.file.read())
        tmp_path = tmp.name

    # Load PDF
    loader = PyPDFLoader(tmp_path)
    docs = loader.load()

    # Chunk
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(docs)

    # Add metadata
    for chunk in chunks:
        chunk.metadata["company_id"] = company_id
        chunk.metadata["source"] = file.filename

    # Pinecone index
    index = get_or_create_index(company_id)

    vectorstore = PineconeVectorStore(
        index=index,
        embedding=embeddings
    )

    vectorstore.add_documents(chunks)

    os.remove(tmp_path)

    return {
        "company_id": company_id,
        "chunks_added": len(chunks),
        "file": file.filename
    }
