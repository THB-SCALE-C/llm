
import os
from pathlib import Path
from logging import Logger
import re
from typing import Any, Callable, Generator, List, Tuple

from dotenv import load_dotenv
from langchain_text_splitters import TextSplitter
from supabase import Client
from llm.lib.types import Section
from llm.lib.utils import get_logger
from llm.preprocessing.chunking import Chunk, md_buffer_to_chunks, pdf_buffer_to_chunks, txt_buffer_to_chunks
from llm.provider import get_embedding_model
from llm.provider.base.embedding_model import Embedding

load_dotenv()
BUCKET = os.getenv("STORAGE_BUCKET", "")

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


def _build_sections(chunks: list[Chunk], embeddings: list[Embedding], document_id: Any) -> list[Section]:
    """Create section rows by pairing each chunk with its embedding."""
    sections: list[Section] = []
    for chunk, embedding in zip(chunks, embeddings):
        _raise_matching_error(chunk.text, embedding.text)
        sections.append(
            Section(
                content=chunk.text,
                document_id=document_id,
                embedding=embedding.vector,
                meta=chunk.meta,
            )
        )
    return sections


def create_embeddings_from_db(
    documents: list[str] | None = None,
    folder: Path | None = None,
    bucket: str = BUCKET,
    separator: str = r"\n\n",
    embedding_model: str = "sentence-transformers/all-minilm-l12-v2",
    embedding_provider: str = "openrouter",
    client: Client | None = None,
    logger: Logger | None = None,
    skip_existing_embeddings:bool=True,
) -> List[Section]:
    """Create embeddings for documents and upload section rows to Supabase."""
    from llm.lib.supabase import get_documents_buffers, get_supabase
    client = client or get_supabase()
    logger = logger or get_logger()

    docs_and_buffers = list(get_documents_buffers(bucket, documents, folder, client, logger))
    if not docs_and_buffers:
        return []
    docs, buffers = zip(*docs_and_buffers)

    # Fast skip-path: fetch existing section IDs in one query.
    embedded_document_ids = _fetch_embedded_document_ids(client, [doc["id"] for doc in docs])
    model = get_embedding_model(embedding_provider, embedding_model)

    sections = []
    for doc, buffer in zip(docs, buffers):
        document_id = doc["id"]
        if skip_existing_embeddings and document_id in embedded_document_ids:
            logger.debug(f"document with id {document_id} already embedded. Skipping...")
            continue
        if not buffer:
            logger.debug(f"document with id {document_id} has no buffer. Skipping...")
            continue

        suffix = Path(doc.get("name", "")).suffix.lower()
        if suffix == ".pdf":
            chunker: Callable[..., List[Chunk]] = pdf_buffer_to_chunks
        elif suffix == ".md":
            chunker = md_buffer_to_chunks
        elif suffix == ".txt":
            chunker = txt_buffer_to_chunks
        else:
            logger.debug(f"document with id {document_id} has unsupported extension '{suffix}'. Skipping...")
            continue

        chunks = chunker(buffer, separator)
        if not chunks:
            logger.debug(f"document with id {document_id} produced no chunks. Skipping...")
            continue

        embeddings = model.embed([chunk.text for chunk in chunks])
        _sections = _build_sections(chunks, embeddings, document_id)
        logger.debug(f"created {len(_sections)} sections.")
        sections.extend(_sections)
    return sections

def load_local_documents(path: Path, expected_extensions: set[str]) -> List[Tuple[bytes, dict]]:
    loaded_documents = []
    normalized_extensions = {ext.lower() for ext in expected_extensions}

    def _recursively_load_files(path: Path, loaded_documents: list):
        if path.is_dir():
            for entry in os.scandir(path):
                _recursively_load_files(Path(entry.path), loaded_documents)
    
        if path.is_file() and path.suffix.lower() in normalized_extensions:
            with open(path, "rb") as f:
                meta = dict(name=str(path))
                loaded_documents.append((f.read(), meta))
    _recursively_load_files(path, loaded_documents)
    return loaded_documents
    


def create_embeddings_from_local_documents(
    document_folder: str | Path,
    separator:str|re.Pattern[str]|TextSplitter|None = None,
    embedding_model: str = "sentence-transformers/all-minilm-l12-v2",
    embedding_provider: str = "openrouter",
    logger: Logger | None = None,
) -> List[Section]:
    """Create embeddings for local .pdf, .md and .txt documents."""
    logger = logger or get_logger()
    model = get_embedding_model(embedding_provider, embedding_model)

    path = Path(document_folder)
    loaded_documents: list = load_local_documents(path, {".pdf", ".md", ".txt"})
    sections = []
    for i,(doc,meta) in enumerate(loaded_documents):
        suffix = Path(meta.get("name", "")).suffix.lower()
        if suffix == ".pdf":
            chunker: Callable[..., List[Chunk]] = pdf_buffer_to_chunks
        elif suffix == ".md":
            chunker = md_buffer_to_chunks
        else:
            chunker = txt_buffer_to_chunks

        if separator:
            chunks = chunker(doc, separator, meta)
        else:
            chunks = chunker(doc, meta=meta)
        if not chunks:
            logger.debug(f"document with id {i} produced no chunks. Skipping...")
            continue
        
        embeddings = model.embed([chunk.text for chunk in chunks])
        secs=_build_sections(list(chunks), embeddings, i)
        sections.extend(secs)
        logger.debug(f"created {len(sections)} sections.")
    return sections



def _raise_matching_error(text1, text2):
    if text1 == text2:
        return True
    else:
        raise ValueError("chunk and embedding incoherent.")
