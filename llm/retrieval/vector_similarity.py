from llm.lib.supabase import query_vector_db
from llm.provider import get_embedding_model


def retrieve_by_similarity(queries: list[str] | str, 
                           embedding_provider: str = "openrouter", 
                           embedding_model: str = "sentence-transformers/all-minilm-l12-v2", 
                           document_ids: list[int] | None = None, match_count=5):
    model = get_embedding_model(embedding_provider, embedding_model)
    if isinstance(queries, str):
        query_embeddings = model.embed([queries])
        query_vector = query_embeddings[0].vector
        results = query_vector_db(query_embedding=query_vector,
                        document_ids=document_ids, match_count=match_count,)
    else:
        results = []
        query_embeddings = model.embed(queries)
        for em in query_embeddings:
            query_vector = em.vector
            result = query_vector_db(query_embedding=query_vector,
                        document_ids=document_ids, match_count=match_count,)
            results.extend(result)
    return results 

