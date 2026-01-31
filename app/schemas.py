from pydantic import BaseModel

class AskRequest(BaseModel):
    session_id: str
    company_id: str
    question: str


class AskResponse(BaseModel):
    answer: str

class DeleteDocumentRequest(BaseModel):
    company_id: str
    file_name: str
