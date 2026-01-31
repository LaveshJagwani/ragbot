import os
from collections import defaultdict
from pinecone import Pinecone

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))


def list_documents(company_id: str):
    index = pc.Index(company_id)

    # We query with a dummy vector but filter by metadata
    # Pinecone requires a vector for queries
    stats = defaultdict(int)

    # Use index.query with metadata-only fetch
    res = index.query(
        vector=[0.0] * 384,  # dummy vector
        top_k=10000,
        include_metadata=True,
        filter={
            "company_id": {"$eq": company_id}
        }
    )

    for match in res.matches:
        source = match.metadata.get("source", "unknown")
        stats[source] += 1

    return {
        "company_id": company_id,
        "documents": [
            {"file_name": k, "chunks": v}
            for k, v in stats.items()
        ]
    }
