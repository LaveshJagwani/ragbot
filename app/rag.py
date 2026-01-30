from app.config import vectorstore, llm
from app.memory import get_history, append_history

def format_history(history):
    return "\n".join(
        f"User: {q}\nAssistant: {a}"
        for q, a in history
    )

def ask_rag(question: str, session_id: str):
    history = get_history(session_id)
    history_text = format_history(history)

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
