from typing import TYPE_CHECKING

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from supabase import Client

from llm.retrieval.dense import DenseSupabaseRetriever
from llm.retrieval.sparse import SparseSupabaseRetriever

if TYPE_CHECKING:
    from langchain_classic.retrievers import MergerRetriever as _MergerRetriever


def _get_merger_retriever_cls():
    try:
        from langchain_classic.retrievers import MergerRetriever

        return MergerRetriever
    except Exception:
        try:
            from langchain.retrievers import MergerRetriever

            return MergerRetriever
        except Exception as exc:
            raise ImportError(
                "MergerRetriever not available. Install a LangChain package that provides it "
                "(e.g. langchain-classic) and retry."
            ) from exc


class HybridSupabaseRetriever(BaseRetriever):
    embedding_provider: str = "openrouter"
    embedding_model: str = "sentence-transformers/all-minilm-l12-v2"
    document_ids: list[int] | None = None
    corpus_id: int | None = None
    experiment_id: int | None = None
    dense_match_count: int = 5
    sparse_match_count: int = 5
    match_count: int = 5
    client: Client | None = None

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        dense = DenseSupabaseRetriever(
            embedding_provider=self.embedding_provider,
            embedding_model=self.embedding_model,
            document_ids=self.document_ids,
            corpus_id=self.corpus_id,
            experiment_id=self.experiment_id,
            match_count=self.dense_match_count,
            client=self.client,
        )
        sparse = SparseSupabaseRetriever(
            document_ids=self.document_ids,
            corpus_id=self.corpus_id,
            experiment_id=self.experiment_id,
            match_count=self.sparse_match_count,
            client=self.client,
        )
        merger_cls = _get_merger_retriever_cls()
        merger: "_MergerRetriever" = merger_cls(retrievers=[dense, sparse])
        return merger.invoke(query)[: self.match_count]


def hybrid_search_db(
    query: str,
    embedding_provider: str = "openrouter",
    embedding_model: str = "sentence-transformers/all-minilm-l12-v2",
    document_ids: list[int] | None = None,
    corpus_id: int | None = None,
    experiment_id: int | None = None,
    client: Client | None = None,
    dense_match_count: int = 5,
    sparse_match_count: int = 5,
    match_count: int = 5,
) -> list[Document]:
    retriever = HybridSupabaseRetriever(
        embedding_provider=embedding_provider,
        embedding_model=embedding_model,
        document_ids=document_ids,
        corpus_id=corpus_id,
        experiment_id=experiment_id,
        dense_match_count=dense_match_count,
        sparse_match_count=sparse_match_count,
        match_count=match_count,
        client=client,
    )
    return retriever.invoke(query)
