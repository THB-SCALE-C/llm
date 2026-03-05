from pathlib import Path
from typing import Any, Dict, List
import json
import numpy as np
from supabase import Client
from llm.provider import get_embedding_model


def vector_search_db(queries: list[str] | str,
                     embedding_provider: str = "openrouter",
                     embedding_model: str = "sentence-transformers/all-minilm-l12-v2",
                     document_ids: list[int] | None = None,
                     client: Client|None = None,
                     match_count=5):
    from llm.lib.supabase import get_supabase, query_vector_db
    if not client:
        client = get_supabase()
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
                        corpus: Path | str,
                        embedding_provider: str = "openrouter",
                        embedding_model: str = "sentence-transformers/all-minilm-l12-v2",
                        match_count=5):
    model = get_embedding_model(embedding_provider, embedding_model)
    corpus = Path(corpus)
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


def query_local_corpus(
    corpus: Path,
    query_embedding: List[float],
    match_count: int,
    threshold:float|None = None
) -> List[Dict[str, Any]]:
    """
    Return up to `match_count` corpus items with highest cosine similarity to query_embedding.
    Each returned dict is the original item with an added "similarity" float key.
    Items with missing or zero embeddings are ignored.
    threshold allows to keep only similar embeddings.
    """
    similarities = []
    with open(corpus) as f:
        for i, line in enumerate(f.readlines()):
            data = json.loads(line)
            embedding = data.get("embedding")
            if not embedding:
                print(f"ERROR NO EMBEDDING FOUND IN LINE {i}")
                continue
            sim = _calculate_cosine_similarity(query_embedding, embedding)
            if threshold and sim<threshold:
                continue
            similarities.append((data, sim))

        return sorted(
            similarities,
            key=lambda x: x[1],
            reverse=True
        )[:match_count]


def _calculate_cosine_similarity(a: List[float], b: List[float]) -> float:
    """
    Compute cosine similarity between two embedding vectors.

    Args:
        a (np.ndarray): 1D embedding vector
        b (np.ndarray): 1D embedding vector

    Returns:
        float: cosine similarity score
    """
    a_vec = np.asarray(a, dtype=np.float32)
    b_vec = np.asarray(b, dtype=np.float32)

    denom = np.linalg.norm(a_vec) * np.linalg.norm(b_vec)
    if denom == 0:
        return 0.0

    return float(np.dot(a_vec, b_vec) / denom)
