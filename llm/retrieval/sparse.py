from typing import Any

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from supabase import Client

from llm.lib.supabase import get_supabase, query_text_search


class SparseSupabaseRetriever(BaseRetriever):
    document_ids: list[int] | None = None
    corpus_id: int | None = None
    experiment_id: int | None = None
    match_count: int = 5
    client: Client | None = None

    @staticmethod
    def _build_text_search_query(query: str) -> str:
        normalized = " ".join(query.split())
        if " " not in normalized:
            return normalized

        terms = []
        for raw_term in normalized.split(" "):
            term = raw_term.strip("\"'")
            if term:
                terms.append(term.replace("'", "''"))

        if len(terms) <= 1:
            return normalized

        required_terms = " & ".join(f"'{term}'" for term in terms)
        optional_terms = " | ".join(f"'{term}'" for term in terms)
        return f"({required_terms}) | ({optional_terms})"

    def _to_documents(self, rows: list[dict[str, Any]]) -> list[Document]:
        docs: list[Document] = []
        for row in rows[: self.match_count]:
            metadata = dict(row.get("meta") or {})
            if "id" in row:
                metadata["id"] = row["id"]
            if "document_id" in row:
                metadata["document_id"] = row["document_id"]
            docs.append(Document(page_content=row.get("content", ""), metadata=metadata))
        return docs

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        rows = query_text_search(
            query=self._build_text_search_query(query),
            client=self.client or get_supabase(),
            document_ids=self.document_ids,
            corpus_id=self.corpus_id,
            experiment_id=self.experiment_id,
            match_count=self.match_count,
        )
        return self._to_documents(rows or [])


def sparse_search_db(
    query: str,
    document_ids: list[int] | None = None,
    corpus_id: int | None = None,
    experiment_id: int | None = None,
    client: Client | None = None,
    match_count: int = 5,
) -> list[Document]:
    retriever = SparseSupabaseRetriever(
        document_ids=document_ids,
        corpus_id=corpus_id,
        experiment_id=experiment_id,
        client=client,
        match_count=match_count,
    )
    return retriever.invoke(query)
