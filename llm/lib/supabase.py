from logging import Logger
import os
from pathlib import Path
from typing import Any, Generator, Tuple, TypedDict
from dotenv import load_dotenv
from supabase.client import Client
from supabase.lib.client_options import SyncClientOptions
from llm.lib.types import Section
from llm.lib.utils import get_logger

def ensure_env():
    load_dotenv()

    KEY = os.getenv("SUPABASE_API_KEY", "")
    if not KEY:
        raise RuntimeError("Supabase key is not provided.")

    URL = os.getenv("SUPABASE_API_URL", "")
    if not URL:
        raise RuntimeError("Supabase URL is not provided.")
    return URL,KEY

SCHEMA = os.getenv("DB_SCHEMA", "llm")
STORAGE_BUCKET = os.getenv("STORAGE_BUCKET", "")


def get_supabase() -> Client:
    URL,KEY = ensure_env()
    client: Client = Client(URL, KEY, options=SyncClientOptions(schema=SCHEMA))
    return client


def get_storage_object(name: str, client=get_supabase(), bucket: str = STORAGE_BUCKET) -> bytes | None:
    try:
        return client.storage.from_(bucket).download(name)
    except Exception as e:
        print(e)
        return None

def upload_document_sections(sections: list[Section], client=get_supabase()):
    _sections = [s.model_dump() for s in sections]
    response = (
        client
        .table("document_sections")
        .insert(_sections)
        .execute()
    )

    return response.data


class query_result(TypedDict):
    id: int
    content: str
    similarity: float
    meta:dict


def query_vector_db(query_embedding: list[float], client: Client = get_supabase(), document_ids: list[int] | None = None, match_count=5) -> list[query_result]:
    return client.rpc(
        "match_document_sections",
        {
            "query_embedding": query_embedding,
            "match_count": match_count,
            **({"document_ids": document_ids, } if document_ids else {})
        }
    ).execute().data  # type: ignore


def query_text_search(query:str, client: Client = get_supabase()):
    """
    Docstring for query_text_search
    
    :param query: Example = "'phishing' & 'malware'"; use & or | as logical operators
    :type query: str
    :param client: supabase_client
    :type client: Client
    """
    return client.from_("document_sections").select("*").text_search("content",query).execute().data


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


def get_documents_buffers(
    bucket: str,
    documents: list[str] | None = None,
    folder: Path | None = None,
    client: Client | None = None,
    logger: Logger | None = None,
) -> Generator[Tuple[dict,bytes|None]]:
    """Fetch documents and return table entry and respective buffers."""
    client = client or get_supabase()
    logger = logger or get_logger()

    file_paths = _resolve_storage_paths(documents, folder)
    docs = _fetch_documents(client, file_paths)
    logger.debug(f"retrieved {len(docs)} documents.")

    for doc in docs:
        yield doc,get_storage_object(doc["name"], client, bucket)