from typing import Any

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from supabase import Client

from llm.lib.supabase import get_supabase, query_vector_db
from llm.provider import get_embedding_model


class DenseSupabaseRetriever(BaseRetriever):
    embedding_provider: str = "openrouter"
    embedding_model: str = "sentence-transformers/all-minilm-l12-v2"
    document_ids: list[int] | None = None
    corpus_id: int | None = None
    experiment_id: int | None = None
    match_count: int = 5
    client: Client | None = None

    def _to_documents(self, rows: list[dict[str, Any]]) -> list[Document]:
        docs: list[Document] = []
        for row in rows:
            metadata = dict(row.get("meta") or {})
            metadata["id"] = row.get("id")
            metadata["similarity"] = row.get("similarity")
            docs.append(Document(page_content=row.get("content", ""), metadata=metadata))
        return docs

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        model = get_embedding_model(self.embedding_provider, self.embedding_model)
        query_embedding = model.embed(query)[0].vector
        rows = query_vector_db(
            query_embedding=query_embedding,
            client=self.client or get_supabase(),
            document_ids=self.document_ids,
            corpus_id=self.corpus_id,
            experiment_id=self.experiment_id,
            match_count=self.match_count,
        )
        return self._to_documents(rows)


def dense_search_db(
    query: str,
    embedding_provider: str = "openrouter",
    embedding_model: str = "sentence-transformers/all-minilm-l12-v2",
    document_ids: list[int] | None = None,
    corpus_id: int | None = None,
    experiment_id: int | None = None,
    client: Client | None = None,
    match_count: int = 5,
) -> list[Document]:
    retriever = DenseSupabaseRetriever(
        embedding_provider=embedding_provider,
        embedding_model=embedding_model,
        document_ids=document_ids,
        corpus_id=corpus_id,
        experiment_id=experiment_id,
        client=client,
        match_count=match_count,
    )
    return retriever.invoke(query)
