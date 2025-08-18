"""HNSW example (simplified, OpenAI removed).

This example previously demonstrated custom LLM + OpenAI embeddings.
After purge it now only shows how to configure HNSW storage with a local
SentenceTransformer embedding model. Replace model funcs with real Gemini
calls automatically via default GraphRAG settings or provide stubs.
"""

import os
import logging
import numpy as np
from sentence_transformers import SentenceTransformer
from nano_graphrag import GraphRAG, QueryParam
from nano_graphrag._storage import HNSWVectorStorage
from nano_graphrag._utils import wrap_embedding_func_with_attrs

logging.basicConfig(level=logging.INFO)

WORKING_DIR = "./nano_graphrag_cache_using_hnsw_as_vectorDB"

EMBED_MODEL = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2", cache_folder=WORKING_DIR, device="cpu"
)


@wrap_embedding_func_with_attrs(
    embedding_dim=EMBED_MODEL.get_sentence_embedding_dimension(),
    max_token_size=EMBED_MODEL.max_seq_length,
)
async def local_embedding(texts: list[str]) -> np.ndarray:
    return EMBED_MODEL.encode(texts, normalize_embeddings=True)


def _reset():
    for f in [
        "vdb_entities.json",
        "kv_store_full_docs.json",
        "kv_store_text_chunks.json",
        "kv_store_community_reports.json",
        "graph_chunk_entity_relation.graphml",
    ]:
        p = os.path.join(WORKING_DIR, f)
        if os.path.exists(p):
            os.remove(p)


def build_index():
    _reset()
    with open("./tests/mock_data.txt", encoding="utf-8-sig") as f:
        text = f.read()
    rag = GraphRAG(
        working_dir=WORKING_DIR,
        vector_db_storage_cls=HNSWVectorStorage,
        vector_db_storage_cls_kwargs={"max_elements": 1000000, "ef_search": 200, "M": 50},
        embedding_func=local_embedding,
        enable_naive_rag=True,
    )
    rag.insert(text)


def run_queries():
    rag = GraphRAG(
        working_dir=WORKING_DIR,
        vector_db_storage_cls=HNSWVectorStorage,
        vector_db_storage_cls_kwargs={"max_elements": 1000000, "ef_search": 200, "M": 50},
        embedding_func=local_embedding,
        enable_naive_rag=True,
    )
    print(rag.query("What are the top themes in this story?", param=QueryParam(mode="global")))
    print(rag.query("What are the top themes in this story?", param=QueryParam(mode="local")))


if __name__ == "__main__":
    build_index()
    run_queries()
