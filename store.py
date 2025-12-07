import os

from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance, HnswConfigDiff, PointStruct
from config import QDRANT_COLLECTION_BASIC, QDRANT_COLLECTION_DETAIL, QDRANT_HOST, QDRANT_PORT
from dotenv import load_dotenv
load_dotenv()

qdrant = QdrantClient(QDRANT_HOST, port=QDRANT_PORT, api_key=os.getenv("QDRANT_API_KEY"), timeout=60)

def init_collection():
    collections = qdrant.get_collections().collections
    existing = [c.name for c in collections]

    if QDRANT_COLLECTION_BASIC in existing:
        print(f"[QDRANT] Collection '{QDRANT_COLLECTION_BASIC}' already exists. Skip creating.")
        return
    else:
        qdrant.create_collection(
            collection_name=QDRANT_COLLECTION_BASIC,
            vectors_config={
                "text_vector": VectorParams(size=384, distance=Distance.COSINE),
                "name_vector": VectorParams(size=384, distance=Distance.COSINE)
            },
            hnsw_config=HnswConfigDiff(on_disk=True)
        )
        print(f"[QDRANT] Collection '{QDRANT_COLLECTION_BASIC}' created.")

    if QDRANT_COLLECTION_DETAIL in existing:
        print(f"[QDRANT] Collection '{QDRANT_COLLECTION_DETAIL}' already exists. Skip creating.")
        return
    else:
        qdrant.create_collection(
            collection_name=QDRANT_COLLECTION_DETAIL,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
            hnsw_config=HnswConfigDiff(on_disk=True)
        )
        print(f"[QDRANT] Collection '{QDRANT_COLLECTION_DETAIL}' created.")

# def upsert_batch(batch):
#     qdrant.upsert(
#         collection_name=QDRANT_COLLECTION,
#         points=batch
#     )

def upsert_batch(collection_name: str, points: list):
    """
    points: list of dicts {'id': id, 'vector': vector (optional), 'payload': {...}}
    """
    point_structs = []
    for p in points:
        if "vector" in p and p["vector"] is not None:
            point_structs.append(
                PointStruct(id=int(p["id"]), vector=p["vector"], payload=p["payload"])
            )
        else:
            point_structs.append(
                PointStruct(id=int(p["id"]), payload=p["payload"])
            )

    qdrant.upsert(collection_name=collection_name, points=point_structs)

def retrieve_by_id(collection_name: str, id_):
    """id 단일 조회 (payload 반환)"""
    res = qdrant.retrieve(collection_name=collection_name, ids=[int(id_)])
    if res and len(res) > 0:
        return res[0].payload
    return None

def search_vectors(collection_name: str, vector, limit=3, filter=None):
    """벡터 검색 래퍼"""
    results = qdrant.search(collection_name=collection_name, query_vector=vector, limit=limit, query_filter=filter)
    return results