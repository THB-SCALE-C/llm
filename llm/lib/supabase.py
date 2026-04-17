from logging import Logger
import os
from pathlib import Path
import time
from typing import Any, Generator, Iterable, Tuple, TypedDict

from dotenv import load_dotenv
import httpx
from pydantic import BaseModel, Field
from pydantic.v1.utils import deep_update
from supabase.client import Client
from supabase.lib.client_options import SyncClientOptions

from llm.lib.utils import get_logger


class QueryResult(TypedDict):
    id: int
    document_id: int
    content: str
    similarity: float
    meta: dict


# Backward compatibility for callers importing the old name.
query_result = QueryResult


class DocumentCorpusRow(TypedDict):
    id: int
    created_at: str
    description: str | None
    name: str | None


# ------------------------------
# Client / environment utilities
# ------------------------------
def ensure_env() -> tuple[str, str]:
    load_dotenv()

    key = os.getenv("SUPABASE_API_KEY", "")
    if not key:
        raise RuntimeError("Supabase key is not provided.")

    url = os.getenv("SUPABASE_API_URL", "")
    if not url:
        raise RuntimeError("Supabase URL is not provided.")

    return url, key


def get_supabase() -> Client:
    url, key = ensure_env()
    schema = os.getenv("DB_SCHEMA", "llm")
    # postgrest defaults to HTTP/2; we use HTTP/1.1 to avoid occasional
    # h2 connection state errors in long-running notebook update loops.
    http_client = httpx.Client(http2=False, follow_redirects=True)
    return Client(
        url,
        key,
        options=SyncClientOptions(schema=schema, httpx_client=http_client),
    )


def _resolve_client(client: Client | None) -> Client:
    return client or get_supabase()


def _resolve_bucket(bucket: str | None) -> str:
    return bucket or os.getenv("STORAGE_BUCKET", "documents")


def _execute_with_retry(
    build_query: Any,
    attempts: int = 4,
    initial_backoff_seconds: float = 0.2,
):
    """
    Execute a postgrest query with retry for transient transport-level errors.
    """
    for attempt in range(attempts):
        try:
            return build_query().execute()
        except (httpx.LocalProtocolError, httpx.RemoteProtocolError, httpx.TransportError):
            if attempt >= attempts - 1:
                raise
            time.sleep(initial_backoff_seconds * (2**attempt))


# ----------------
# Storage utilities
# ----------------
def get_storage_object(
    name: str,
    client: Client | None = None,
    bucket: str | None = None,
) -> bytes | None:
    client = _resolve_client(client)
    bucket = _resolve_bucket(bucket)
    try:
        return client.storage.from_(bucket).download(name)
    except Exception as e:
        print(e)
        return None


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
    query = client.table("documents_with_storage_path").select("id, name, storage_object_path")
    if file_paths:
        query = query.in_("storage_object_path", file_paths)
    response = query.execute()
    return response.data or []


def _normalize_extensions(extensions: Iterable[str] | None) -> set[str] | None:
    """Normalize values like 'pdf' or '.PDF' to {'.pdf'} for suffix checks."""
    if not extensions:
        return None
    normalized = {
        (f".{ext.lstrip('.').lower()}")
        for ext in extensions
        if ext and ext.strip()
    }
    return normalized or None


def _filter_documents_by_extension(
    docs: list[dict[str, Any]],
    extensions: set[str] | None,
) -> list[dict[str, Any]]:
    if not extensions:
        return docs

    filtered: list[dict[str, Any]] = []
    for doc in docs:
        storage_key = str(doc.get("storage_object_path") or doc.get("name") or "")
        suffix = Path(storage_key).suffix.lower()
        if suffix in extensions:
            filtered.append(doc)
    return filtered


def get_documents_buffers(
    bucket: str | None = None,
    documents: list[str] | None = None,
    folder: Path | str | None = None,
    client: Client | None = None,
    logger: Logger | None = None,
    extensions: list[str] | None = None,
) -> Generator[tuple[dict[str, Any], bytes | None]]:
    """
    Fetch document rows and yield each row with its storage buffer.

    `extensions` supports values with or without leading dots, e.g. ['pdf', '.md'].
    """
    client = _resolve_client(client)
    logger = logger or get_logger()
    bucket = _resolve_bucket(bucket)
    folder = Path(folder) if folder else None

    file_paths = _resolve_storage_paths(documents, folder)
    docs = _fetch_documents(client, file_paths)
    docs = _filter_documents_by_extension(docs, _normalize_extensions(extensions))
    logger.debug(f"retrieved {len(docs)} documents.")

    for doc in docs:
        storage_key = str(doc.get("storage_object_path") or doc.get("name"))
        yield doc, get_storage_object(storage_key, client=client, bucket=bucket)


# -----------------------
# Retrieval query helpers
# -----------------------
def query_vector_db(
    query_embedding: list[float],
    client: Client | None = None,
    document_ids: list[int] | None = None,
    corpus_id:int|None=None,
    experiment_id:int|None=None,
    match_count: int = 5,
) -> list[QueryResult]:
    client = _resolve_client(client)
    params = {
            "query_embedding": query_embedding,
            "match_count": match_count,
        }
    if document_ids:
        params.update({"document_ids": document_ids})
    if corpus_id:
        params.update({"corpus_id":corpus_id})
    if experiment_id:
        params.update({"experiment_id":experiment_id})
    
    response = client.rpc(
        "match_document_sections",
        params,
    ).execute()
    return response.data or []


def query_text_search(
    query: str,
    client: Client | None = None,
    corpus_id:int|None=None,
    experiment_id:int|None=None,
    document_ids: list[int] | None = None,
    match_count: int | None = None,
) -> list[dict[str, Any]]:
    """
    Run full-text search over `document_sections.content`.

    query example: "'phishing' & 'malware'" where `&` and `|` are logical operators.
    """
    client = _resolve_client(client)
    request = client.from_("document_sections").select("*")
    if document_ids:
        request = request.in_("document_id", document_ids)
    if corpus_id:
        request = request.eq("corpus_id", corpus_id)
    if experiment_id:
        request = request.eq("experiment_id", experiment_id)
    request = request.text_search("content", query)
    data = request.execute().data
    return data[:match_count] if match_count else data


# ------------------------
# Experiment table helpers
# ------------------------
def create_document_corpus(
    name: str | None = None,
    description: str | None = None,
    meta:dict|None =None,
    client: Client | None = None,
) -> DocumentCorpusRow:
    client = _resolve_client(client)
    payload = {
        "name": name,
        "description": description,
        "meta":meta,
    }
    response = client.table("document_corpus").insert(payload).execute()
    rows = response.data or []
    if not rows:
        raise RuntimeError("Failed to create document_corpus row.")
    return rows[0]


def get_document_corpus(
    corpus_id: int,
    client: Client | None = None,
) -> DocumentCorpusRow | None:
    client = _resolve_client(client)
    response = (
        client
        .table("document_corpus")
        .select("*")
        .eq("id", corpus_id)
        .limit(1)
        .execute()
    )
    rows = response.data or []
    return rows[0] if rows else None


def get_document_corpus_by_name(
    name: str,
    client: Client | None = None,
) -> DocumentCorpusRow | None:
    client = _resolve_client(client)
    response = (
        client
        .table("document_corpus")
        .select("*")
        .eq("name", name)
        .limit(1)
        .execute()
    )
    rows = response.data or []
    return rows[0] if rows else None


def list_document_corpora(client: Client | None = None) -> list[DocumentCorpusRow]:
    client = _resolve_client(client)
    response = client.table("document_corpus").select("*").order("id", desc=False).execute()
    return response.data or []


def get_or_create_document_corpus(
    name: str,
    description: str | None = None,
    client: Client | None = None,
) -> DocumentCorpusRow:
    existing = get_document_corpus_by_name(name, client=client)
    if existing:
        return existing
    return create_document_corpus(name=name, description=description, client=client)


def get_chunking_experiment_by_name(
    experiment_name: str,
    client: Client | None = None,
) -> dict[str, Any] | None:
    client = _resolve_client(client)
    response = (
        client
        .table("chunking_experiment")
        .select("*")
        .eq("experiment_name", experiment_name)
        .limit(1)
        .execute()
    )
    rows = response.data or []
    return rows[0] if rows else None


def create_chunking_experiment(
    experiment_name: str,
    chunking_strategy: str,
    corpus_id:int,
    notes: str | None = None,
    meta: dict[str, Any] | None = None,
    client: Client | None = None,
) -> dict[str, Any]:
    client = _resolve_client(client)
    payload = {
        "corpus_id":corpus_id,
        "experiment_name": experiment_name,
        "chunking_strategy": chunking_strategy,
        "notes": notes,
        "meta": meta or {},
    }
    response = client.table("chunking_experiment").insert(payload).execute()
    rows = response.data or []
    if not rows:
        raise RuntimeError("Failed to create chunking_experiment row.")
    return rows[0]

def get_or_create_chunking_experiment(
    experiment_name: str,
    chunking_strategy: str | None = None,
    corpus_id:int | None = None,
    notes: str | None = None,
    meta: dict[str, Any] | None = None,
    client: Client | None = None,
) -> dict[str, Any]:
    client = _resolve_client(client)
    existing = get_chunking_experiment_by_name(experiment_name, client=client)
    if existing:
        existing_notes = existing.get("notes")
        merged_notes = f"{existing_notes}\n{notes}" if existing_notes and notes else (notes or existing_notes)
        merged_meta = deep_update(existing.get("meta") or {}, meta or {})
        payload = {
            "corpus_id": corpus_id,
            "notes": merged_notes,
            "meta": merged_meta,
        }
        response = (
            client
            .table("chunking_experiment")
            .update(payload)
            .eq("id", existing["id"])
            .execute()
        )
        rows = response.data or []
        if not rows:
            raise RuntimeError("Failed to update chunking_experiment row.")
        return rows[0]
    if not corpus_id or not chunking_strategy:
        raise Exception("Must set `corpus_id` and `chunking_strategy`")
    return create_chunking_experiment(
        corpus_id=corpus_id,
        experiment_name=experiment_name,
        chunking_strategy=chunking_strategy,
        notes=notes,
        meta=meta,
        client=client,
    )


def upsert_document_preprocessed(
    document_id: int,
    content: str,
    preprocessing_experiment_id: str | None = None,
    preprocessing_comment: str | None = None,
    meta: dict[str, Any] | None = None,
    corpus_id: int | None = None,
    client: Client | None = None,
    on_conflict: str | None = None,
) -> dict[str, Any]:
    client = _resolve_client(client)
    payload: dict[str, Any] = {
        "document_id": document_id,
        "content": content,
        "corpus_id": corpus_id,
        "preprocessing_experiment_id": preprocessing_experiment_id,
        "preprocessing_comment": preprocessing_comment,
        "meta": meta or {},
    }
    conflict_target = on_conflict or ("document_id,corpus_id" if corpus_id is not None else "document_id")
    response = (
        client
        .table("document_preprocessed")
        .upsert(payload, on_conflict=conflict_target)
        .execute()
    )
    rows = response.data or []
    if not rows:
        raise RuntimeError("Failed to upsert document_preprocessed row.")
    return rows[0]


def get_document_preprocessed(
    document_id: int | None = None,
    corpus_id: int | None = None,
    client: Client | None = None,
) -> dict[str, Any] | None:
    if document_id is None and corpus_id is None:
        raise ValueError("Either document_id or corpus_id must be provided.")
    client = _resolve_client(client)
    query = client.table("document_preprocessed").select("*")
    if document_id is not None:
        query = query.eq("document_id", document_id)
    if corpus_id is not None:
        query = query.eq("corpus_id", corpus_id)
    response = query.limit(1).execute()
    rows = response.data or []
    return rows[0] if rows else None 

def get_corpus_documents(
    corpus_id: int,
    client: Client | None = None,
) -> dict[str, Any] | None:
    """
    Return corpus row and related documents for a given corpus id.

    Uses two fast queries:
    1) fetch corpus metadata
    2) fetch linked documents through `document_preprocessed`
    """
    client = _resolve_client(client)
    corpus = get_document_corpus(corpus_id, client=client)
    if not corpus:
        return None

    response = (
        client
        .table("document_preprocessed")
        .select("document_id, documents(*)")
        .eq("corpus_id", corpus_id)
        .execute()
    )
    rows = response.data or []

    documents: list[dict[str, Any]] = []
    seen_ids: set[int] = set()
    for row in rows:
        document = row.get("documents")
        if not document:
            continue
        document_id = document.get("id")
        if isinstance(document_id, int) and document_id in seen_ids:
            continue
        if isinstance(document_id, int):
            seen_ids.add(document_id)
        documents.append(document)

    return {
        "corpus": corpus,
        "documents": documents,
    }

def list_document_preprocessed(
    document_id: int | None = None,
    corpus_id: int | None = None,
    client: Client | None = None,
) -> list[dict[str, Any]]:
    """List preprocessed rows with optional filters for document and corpus."""
    client = _resolve_client(client)
    query = client.table("document_preprocessed").select("*")
    if document_id is not None:
        query = query.eq("document_id", document_id)
    if corpus_id is not None:
        query = query.eq("corpus_id", corpus_id)
    response = query.execute()
    return response.data or []


def list_document_sections(
    corpus_id: int | None = None,
    chunking_experiment_id: int | None = None,
    document_preprocessed_id: int | None = None,
    with_embeddings: bool | None = None,
    client: Client | None = None,
) -> list[dict[str, Any]]:
    """
    List section rows with optional corpus/experiment/document filters.

    Set `with_embeddings=True` to only return rows with vectors, `False` for rows
    without vectors, or `None` for no embedding filter.
    """
    client = _resolve_client(client)
    query = client.table("document_sections").select("*")
    if corpus_id is not None:
        query = query.eq("corpus_id", corpus_id)
    if chunking_experiment_id is not None:
        query = query.eq("chunking_experiment_id", chunking_experiment_id)
    if document_preprocessed_id is not None:
        query = query.eq("document_preprocessed_id", document_preprocessed_id)
    if with_embeddings is True:
        query = query.not_.is_("embedding", "null")
    elif with_embeddings is False:
        query = query.is_("embedding", "null")
    response = query.execute()
    return response.data or []


class Section(BaseModel):
    document_id: str | int
    content: str
    embedding: list[float]| None = None
    meta: dict = Field(default_factory=dict)
    document_preprocessed_id: int 
    chunking_experiment_id: int 

def upload_document_sections(
    sections: list[Section],
    client: Client | None = None,
    corpus_id:int | None =None,
) -> list[dict[str, Any]]:
    """Insert `document_sections` rows, applying optional default FK values."""
    client = _resolve_client(client)
    section_rows: list[dict[str, Any]] = []

    for section in sections:
        row = section.model_dump()
        row["corpus_id"] = corpus_id
        section_rows.append(row)

    response = client.table("document_sections").insert(section_rows).execute()
    return response.data or []

def upsert_document_sections(
    rows: list[dict[str, Any]],
    client: Client | None = None,
    on_conflict: str = "content,meta",
    # ignore_existing_embedding:bool = True,
) -> list[dict[str, Any]]:
    """
    Upsert `document_sections` rows in bulk.
    """
    if not rows:
        return []

    client = _resolve_client(client)
    rows_to_upsert = rows
    response = _execute_with_retry(
        lambda: client.table("document_sections").upsert(rows_to_upsert, on_conflict=on_conflict)
    )
    return response.data or []

def add_embeddings_document_sections(
    rows: list[Tuple[int, list]],
    client: Client | None = None,
    ignore_existing_embedding:bool = True,
):
    if not rows:
        return []

    client = _resolve_client(client)
    response = client.table("document_sections").select("*").execute()
    db_rows = response.data or []
    db_rows_by_id = {row["id"]: row for row in db_rows if row.get("id") is not None}

    # Keep only the last embedding per id to avoid duplicate upsert work.
    embeddings_by_id: dict[int, list] = {}
    for section_id, embedding in rows:
        embeddings_by_id[section_id] = embedding

    rows_to_upsert: list[dict[str, Any]] = []
    for section_id, embedding in embeddings_by_id.items():
        db_row = db_rows_by_id.get(section_id)
        if not db_row:
            continue
        if ignore_existing_embedding and db_row.get("embedding") is not None:
            continue

        upsert_row = dict(db_row)
        upsert_row.pop("id", None)
        upsert_row["content"] = db_row.get("content")
        upsert_row["meta"] = db_row.get("meta") or {}
        upsert_row["embedding"] = embedding
        rows_to_upsert.append(upsert_row)

    if not rows_to_upsert:
        return []

    return upsert_document_sections(
        rows=rows_to_upsert,
        client=client,
        on_conflict="meta,content",
    )
