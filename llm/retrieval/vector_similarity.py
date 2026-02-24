from pathlib import Path
from typing import Any, Dict, List
import heapq
import json
import math
import numpy as np

from supabase import Client
from llm.lib.supabase import get_supabase, query_vector_db
from llm.provider import get_embedding_model


def vector_search_db(queries: list[str] | str,
                     embedding_provider: str = "openrouter",
                     embedding_model: str = "sentence-transformers/all-minilm-l12-v2",
                     document_ids: list[int] | None = None,
                     client: Client = get_supabase(),
                     match_count=5):
    model = get_embedding_model(embedding_provider, embedding_model)
    if isinstance(queries, str):
        query_embeddings = model.embed([queries])
        query_vector = query_embeddings[0].vector
        results = query_vector_db(query_embedding=query_vector,
                                  document_ids=document_ids, match_count=match_count, client=client)
    else:
        results = []
        query_embeddings = model.embed(queries)
        for em in query_embeddings:
            query_vector = em.vector
            result = query_vector_db(query_embedding=query_vector,
                                     document_ids=document_ids, match_count=match_count,)
            results.extend(result)
    return results


def vector_search_local(queries: list[str] | str,
                        corpus: Path | List[Any],
                        embedding_provider: str = "openrouter",
                        embedding_model: str = "sentence-transformers/all-minilm-l12-v2",
                        match_count=5):
    model = get_embedding_model(embedding_provider, embedding_model)
    corpus = _load_corpus(corpus) if isinstance(corpus, Path) else corpus
    if isinstance(queries, str):
        query_embeddings = model.embed([queries])
        query_vector = query_embeddings[0].vector
        results = query_local_corpus(
            corpus=corpus, query_embedding=query_vector, match_count=match_count)
    else:
        results = []
        query_embeddings = model.embed(queries)
        for em in query_embeddings:
            query_vector = em.vector
            result = query_local_corpus(
                corpus=corpus, query_embedding=query_vector, match_count=match_count)
            results.extend(result)
    return results


def _load_corpus(corpus) -> List[Any]:
    with open(corpus) as f:
        return json.load(f)


def query_local_corpus(
    corpus: List[Dict[str, Any]],
    query_embedding: List[float],
    match_count: int,
) -> List[Dict[str, Any]]:
    """
    Return up to `match_count` corpus items with highest cosine similarity to query_embedding.
    Each returned dict is the original item with an added "similarity" float key.
    Items with missing or zero embeddings are ignored.
    """
    # collect items that actually have an embedding
    items = [el for el in corpus if el.get("embedding") is not None]
    if not items or match_count <= 0:
        return []

    emb_array = np.asarray([el["embedding"] for el in items], dtype=np.float32)
    q = np.asarray(query_embedding, dtype=np.float32)

    q_norm = np.linalg.norm(q)
    if q_norm == 0:
        return []

    emb_norms = np.linalg.norm(emb_array, axis=1)
    valid = emb_norms > 0

    sims = np.zeros(len(items), dtype=np.float32)
    if valid.any():
        # efficient dot product across rows
        sims[valid] = (emb_array[valid] @ q) / (emb_norms[valid] * q_norm)

    k = min(match_count, len(sims))
    if k == 0:
        return []

    top_idx = np.argpartition(sims, -k)[-k:]
    top_idx = top_idx[np.argsort(sims[top_idx])[::-1]]

    return [{**items[i], "similarity": float(sims[i])} for i in top_idx]
