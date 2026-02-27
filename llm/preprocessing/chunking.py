from io import BytesIO
from logging import Logger
import os
from pathlib import Path
import re
from typing import Any, Callable, Dict, Generator, List, Literal, Tuple, TypedDict
from pydantic import BaseModel
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter, TextSplitter

from llm.lib.types import Chunk, ChunkedDocument
from llm.lib.utils import get_logger


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


def pdf_buffer_to_pages(buffer: bytes, meta: dict = {}) -> Generator[Chunk]:
    pdf_stream = BytesIO(buffer)
    reader = PdfReader(pdf_stream)
    for i, page in enumerate(reader.pages, start=1):
        meta = {**meta, "page_number": i}
        yield Chunk(id=i, meta=meta, text=page.extract_text())


splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", ".", "?", "!"], chunk_size=450, chunk_overlap=50)


def pdf_buffer_to_chunks(buffer: bytes, separator: str | re.Pattern[str] | TextSplitter = splitter,
                         meta: dict | Literal[False] = {},) -> Tuple[List[Chunk],dict]:
    pages = pdf_buffer_to_pages(buffer, meta=meta if meta is not False else {})
    return list(_to_chunks(list(pages), separator)),{}


def txt_buffer_to_chunks(
    buffer: bytes,
    separator: str | re.Pattern[str] | TextSplitter = splitter,
    meta: dict | Literal[False] = {},
) -> Tuple[List[Chunk],dict]:
    text = buffer.decode("utf-8", errors="ignore")
    if not text:
        return [],{}
    return list(_to_chunks([Chunk(id=0, text=text, meta=meta if meta is not False else {})], separator)),{}


def md_buffer_to_chunks(
    buffer: bytes,
    separator: str | re.Pattern[str] | TextSplitter = splitter,
    meta: dict | Literal[False] = {},
) -> Tuple[List[Chunk],dict]:
    text = buffer.decode("utf-8", errors="ignore")
    if not text:
        return [],{}
    _meta = {}
    if text.startswith("---"):
        _, meta_str, *rest = text.split("---")
        meta_items = [line.replace("\r", "").split(":", 1)
                      for line in meta_str.split("\n") if ":" in line]
        _meta = dict(meta_items)
        text = "---".join(rest)
    return list(_to_chunks([Chunk(id=0, text=text, meta=meta if meta is not False else {})], separator)),_meta


def create_chunks_from_local_documents(
    document_folder: str | Path,
    separator:str|re.Pattern[str]|TextSplitter|None = None,
    logger: Logger | None = None,
    include_meta_in_chunks:bool=False
) -> list[ChunkedDocument]:
    """Create chunks for local .pdf, .md and .txt documents."""
    logger = logger or get_logger()

    path = Path(document_folder)
    loaded_documents: list = load_local_documents(path, {".pdf", ".md", ".txt"})
    docs = []
    for i,(doc,meta) in enumerate(loaded_documents):
        suffix = Path(meta.get("name", "")).suffix.lower()
        if suffix == ".pdf":
            chunker: Callable[..., Tuple[List[Chunk],dict]] = pdf_buffer_to_chunks
        elif suffix == ".md":
            chunker = md_buffer_to_chunks
        else:
            chunker = txt_buffer_to_chunks

        meta = dict(**(meta if include_meta_in_chunks else {}), document_id=i)
        if separator:
            chunks,_meta = chunker(doc, separator, meta, )
        else:
            chunks,_meta = chunker(doc, meta= meta)
        if not chunks:
            logger.debug(f"document with id {i} produced no chunks. Skipping...")
            continue
        docs.append(ChunkedDocument(id=i, chunks=chunks, meta={**meta,**_meta}))
        logger.debug(f"created {len(chunks)} chunks.")
    return docs

##########


def _to_chunks(pages: List[Chunk], separator: str | re.Pattern[str] | TextSplitter = splitter, meta={}):
    if not separator:
        raise ValueError("Must contain separators.")
    _chunks = []
    for i, page in enumerate(pages):
        enhanced_page_text = _enhance_page_text(i, pages)
        if isinstance(separator, TextSplitter):
            splits = separator.split_text(enhanced_page_text)
        else:
            splits = re.split(separator, page.text)
        for j, split in enumerate(splits):
            if split:
                _chunks.append(Chunk(id=f"{i}-{j}", text=split, meta=dict(**meta, **page.meta, page_number=i)))
    return _chunks

def _enhance_page_text(idx, pages: List[Chunk]):
    text = pages[idx].text
    try:
        extra = pages[idx+1].text.split(".")[0][:1000]
        text += extra
    except:
        pass
    try:
        extra = pages[idx].text.split(".")[-1][:1000]
        text = extra + text
    except:
        pass
    return text
