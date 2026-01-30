from fastapi import FastAPI
from app.schemas import AskRequest, AskResponse
from app.rag import ask_rag
from app.memory import reset_session

app = FastAPI(title="RAG Bot with Session Memory")


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    answer = ask_rag(
        question=request.question,
        session_id=request.session_id
    )
    return {"answer": answer}


@app.post("/reset")
def reset(session_id: str):
    reset_session(session_id)
    return {"status": "session reset"}
