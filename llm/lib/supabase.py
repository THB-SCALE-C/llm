import os
from typing import TypedDict
from dotenv import load_dotenv
from openai import BaseModel
from supabase.client import Client
from supabase.lib.client_options import SyncClientOptions
from llm.lib.types import Section


load_dotenv()

KEY = os.getenv("SUPABASE_API_KEY", "")
if not KEY:
    raise RuntimeError("Supabase key is not provided.")

URL = os.getenv("SUPABASE_API_URL", "")
if not URL:
    raise RuntimeError("Supabase URL is not provided.")

SCHEMA = os.getenv("DB_SCHEMA", "llm")
STORAGE_BUCKET = os.getenv("STORAGE_BUCKET", "")


def get_supabase() -> Client:
    client: Client = Client(URL, KEY, options=SyncClientOptions(schema=SCHEMA))
    return client


def get_storage_object(name: str, client=get_supabase(), bucket: str = STORAGE_BUCKET) -> bytes:
    return client.storage.from_(bucket).download(name)


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
