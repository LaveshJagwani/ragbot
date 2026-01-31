import os
from pinecone import Pinecone

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))


def delete_document(company_id: str, file_name: str):
    index = pc.Index(company_id)

    index.delete(
        filter={
            "company_id": {"$eq": company_id},
            "source": {"$eq": file_name}
        }
    )

    return {
        "company_id": company_id,
        "deleted_file": file_name,
        "status": "deleted"
    }
