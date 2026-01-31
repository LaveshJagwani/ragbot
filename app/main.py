from fastapi import FastAPI, UploadFile, File, Form
from app.schemas import AskRequest, AskResponse
from app.rag import ask_rag
from app.upload import process_pdf_and_index
from app.memory import reset_session
from app.delete import delete_document
from app.schemas import DeleteDocumentRequest
from app.list_documents import list_documents



app = FastAPI(title="Multi-tenant RAG API")


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    answer = ask_rag(
        question=req.question,
        session_id=req.session_id,
        company_id=req.company_id
    )
    return {"answer": answer}


@app.post("/upload")
async def upload_pdf(
    company_id: str = Form(...),
    file: UploadFile = File(...)
):
    if file.content_type != "application/pdf":
        return {"error": "Only PDF files are supported"}

    return process_pdf_and_index(company_id, file)

@app.delete("/delete_document")
def delete_doc(req: DeleteDocumentRequest):
    return delete_document(
        company_id=req.company_id,
        file_name=req.file_name
    )

@app.get("/documents/{company_id}")
def get_documents(company_id: str):
    return list_documents(company_id)


@app.post("/reset")
def reset(session_id: str):
    reset_session(session_id)
    return {"status": "session reset"}
