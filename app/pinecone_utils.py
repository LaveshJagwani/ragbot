import os
from pinecone import Pinecone, ServerlessSpec

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

DIMENSION = 384  # MiniLM embeddings
METRIC = "cosine"
CLOUD = "aws"
REGION = "us-east-1"


def get_or_create_index(index_name: str):
    existing = pc.list_indexes().names()

    if index_name not in existing:
        pc.create_index(
            name=index_name,
            dimension=DIMENSION,
            metric=METRIC,
            spec=ServerlessSpec(
                cloud=CLOUD,
                region=REGION
            )
        )

    return pc.Index(index_name)
