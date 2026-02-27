from io import BytesIO
import re
from typing import Any, Dict, Generator, List, Literal, Tuple, TypedDict
from pydantic import BaseModel
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter, TextSplitter


class Chunk(BaseModel):
    id: int | str
    text: str
    meta: dict


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

##########


def _to_chunks(pages: List[Chunk], separator: str | re.Pattern[str] | TextSplitter = splitter):
    if not separator:
        raise ValueError("Must contain separators.")

    for i, page in enumerate(pages):
        enhanced_page_text = _enhance_page_text(i, pages)
        if isinstance(separator, TextSplitter):
            splits = separator.split_text(enhanced_page_text)
        else:
            splits = re.split(separator, page.text)
        for j, split in enumerate(splits):
            yield Chunk(id=f"{i}-{j}", text=split, meta={"document_id": page.id, **page.meta})


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
