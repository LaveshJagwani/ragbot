import os
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore

from app.config import embeddings, llm
from app.memory import get_history, append_history

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))


def format_history(history):
    return "\n".join(
        f"User: {q}\nAssistant: {a}"
        for q, a in history
    )


def get_vectorstore(company_id: str):
    index = pc.Index(company_id)
    return PineconeVectorStore(
        index=index,
        embedding=embeddings
    )


def ask_rag(question: str, session_id: str, company_id: str):
    history = get_history(session_id)
    history_text = format_history(history)

    vectorstore = get_vectorstore(company_id)
    docs = vectorstore.similarity_search(question, k=4)

    context = "\n\n".join(doc.page_content for doc in docs)

    prompt = f"""
You are a helpful assistant.

Conversation so far:
{history_text}

Use ONLY the context below.
If the answer is not present, say "I don't know".

Context:
{context}

User question:
{question}
"""

    answer = llm.invoke(prompt).content

    append_history(session_id, question, answer)
    return answer
