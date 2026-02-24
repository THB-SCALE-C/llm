from pathlib import Path
from typing import Any, List

from supabase import Client
from llm.lib.supabase import get_supabase, query_text_search


def text_search_db(query: str, client: Client = get_supabase()):
    return query_text_search(query=query, client=client)


def text_search_local(
    queries: list[str] | str,
    corpus: Path | List[Any],
    match_count: int = 5,
):
    raise NotImplementedError()