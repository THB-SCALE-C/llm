from io import BytesIO
import re
from typing import Any, Dict, Generator, List, TypedDict
from pydantic import BaseModel
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter, TextSplitter

class Chunk(BaseModel):
    text: str
    meta:dict


def pdf_buffer_to_pages(buffer: bytes,meta:dict={}) -> Generator[Chunk]:
    pdf_stream = BytesIO(buffer)
    reader = PdfReader(pdf_stream)
    for i, page in enumerate(reader.pages, start=1):
        meta = {**meta, "page_number":i}
        yield Chunk(meta=meta, text=page.extract_text())

splitter = RecursiveCharacterTextSplitter(separators=["\n\n",".","?","!"],chunk_size=450, chunk_overlap=50)

def pdf_buffer_to_chunks(buffer:bytes, separator:str|re.Pattern[str]|TextSplitter = splitter, meta:dict={}) -> List[Chunk]:
    pages = pdf_buffer_to_pages(buffer, meta=dict(**meta))
    return list(_to_chunks(list(pages), separator))


def txt_buffer_to_chunks(
    buffer: bytes,
    separator: str | re.Pattern[str] | TextSplitter = splitter,
    meta: dict = {},
) -> List[Chunk]:
    text = buffer.decode("utf-8", errors="ignore")
    if not text:
        return []
    return list(_to_chunks([Chunk(text=text, meta=dict(**meta))], separator))


def md_buffer_to_chunks(
    buffer: bytes,
    separator: str | re.Pattern[str] | TextSplitter = splitter,
    meta: dict = {},
) -> List[Chunk]:
    return txt_buffer_to_chunks(buffer, separator=separator, meta=meta)

##########

def _to_chunks(pages:List[Chunk],separator:str|re.Pattern[str]|TextSplitter = splitter):
    if not separator:
        raise ValueError("Must contain separators.")
        
    if isinstance(separator, TextSplitter):
        for i,page in enumerate(pages):
            enhanced_page_text = _enhance_page_text(i, pages)
            splits =  separator.split_text(enhanced_page_text)
            for split in splits:
                yield Chunk(text=split, meta=page.meta)

    else:
        for page in pages:
            for chunk in re.split(separator,page.text):
                yield Chunk(text=chunk, meta=page.meta)

def _enhance_page_text(idx, pages:List[Chunk]):
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
