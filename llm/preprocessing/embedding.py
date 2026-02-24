
import os
from pathlib import Path
from logging import Logger
import re
from typing import Any, Callable, List

from dotenv import load_dotenv
from langchain_text_splitters import TextSplitter
from llm.lib.supabase import get_storage_object, get_supabase, upload_document_sections
from llm.lib.types import Section
from llm.lib.utils import get_logger
from llm.preprocessing.chunking import pdf_buffer_to_chunks, split_pages
from llm.provider import get_embedding_model

load_dotenv()
BUCKET = os.getenv("STORAGE_BUCKET", "")

def _resolve_storage_paths(
    documents: list[str] | None,
    folder: Path | None,
) -> list[str] | None:
    """Build storage paths used to filter documents in Supabase."""
    if not documents:
        return None
    if not folder:
        return documents
    return [str(folder / document) for document in documents]


def _fetch_documents(client: Any, file_paths: list[str] | None) -> list[dict[str, Any]]:
    """Fetch target documents, optionally filtered by storage paths."""
    query = client.table("documents_with_storage_path").select("id, name")
    if file_paths:
        query = query.in_("storage_object_path", file_paths)
    response = query.execute()
    return response.data or []


def _fetch_embedded_document_ids(client: Any, document_ids: list[Any]) -> set[Any]:
    """Fetch document IDs that already have sections to avoid re-embedding."""
    if not document_ids:
        return set()
    response = (
        client.table("document_sections")
        .select("document_id")
        .in_("document_id", document_ids)
        .execute()
    )
    return {row["document_id"] for row in (response.data or [])}


def _build_sections(chunks: list[Any], embeddings: list[Any], document_id: Any) -> list[Section]:
    """Create section rows by pairing each chunk with its embedding."""
    sections: list[Section] = []
    for chunk, embedding in zip(chunks, embeddings):
        _raise_matching_error(chunk.text, embedding.text)
        sections.append(
            Section(
                content=chunk.text,
                document_id=document_id,
                embedding=embedding.vector,
                meta={"page_number": chunk.page_number},
            )
        )
    return sections


def create_embeddings_in_db(
    documents: list[str] | None = None,
    folder: Path | None = None,
    bucket: str = BUCKET,
    separator: str = r"\n\n",
    embedding_model: str = "sentence-transformers/all-minilm-l12-v2",
    embedding_provider: str = "openrouter",
    client: Any | None = None,
    logger: Logger | None = None,
) -> None:
    """Create embeddings for documents and upload section rows to Supabase."""
    client = client or get_supabase()
    logger = logger or get_logger()

    file_paths = _resolve_storage_paths(documents, folder)
    docs = _fetch_documents(client, file_paths)
    logger.debug(f"retrieved {len(docs)} documents.")

    # Fast skip-path: fetch existing section IDs in one query.
    embedded_document_ids = _fetch_embedded_document_ids(client, [doc["id"] for doc in docs])
    model = get_embedding_model(embedding_provider, embedding_model)

    for doc in docs:
        document_id = doc["id"]
        if document_id in embedded_document_ids:
            logger.debug(f"document with id {document_id} already embedded. Skipping...")
            continue

        buffer = get_storage_object(doc["name"], client, bucket)
        chunks = list(pdf_buffer_to_chunks(buffer, separator))
        if not chunks:
            logger.debug(f"document with id {document_id} produced no chunks. Skipping...")
            continue

        embeddings = model.embed([chunk.text for chunk in chunks])
        sections = _build_sections(chunks, embeddings, document_id)
        logger.debug(f"created {len(sections)} sections.")

        if sections:
            res = upload_document_sections(sections=sections, client=client)
            logger.debug(f"superbase responded on upload: {res}")


def create_embeddings(
    documents: list[Any],
    separator:str|re.Pattern[str]|TextSplitter|None = None,
    embedding_model: str = "sentence-transformers/all-minilm-l12-v2",
    embedding_provider: str = "openrouter",
    logger: Logger | None = None,
) -> List[Any]:
    """Create embeddings for documents and upload section rows to Supabase."""
    logger = logger or get_logger()
    model = get_embedding_model(embedding_provider, embedding_model)

    sections = []
    for doc in documents:
        document_id = doc["id"]
        if separator:
            chunks = split_pages(doc, separator)
        else:
            chunks = split_pages(doc)
        if not chunks:
            logger.debug(f"document with id {document_id} produced no chunks. Skipping...")
            continue

        embeddings = model.embed([chunk.text for chunk in chunks])
        sections.extend(_build_sections(list(chunks), embeddings, document_id))
        logger.debug(f"created {len(sections)} sections.")
    return sections



def _raise_matching_error(text1, text2):
    if text1 == text2:
        return True
    else:
        raise ValueError("chunk and embedding incoherent.")
