from io import BytesIO
import re
from typing import Any, Dict, Generator, List, TypedDict
from pydantic import BaseModel
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter, TextSplitter

class Chunk(BaseModel):
    text: str
    page_number: int


def pdf_buffer_to_pages(buffer: bytes) -> Generator[Chunk]:
    pdf_stream = BytesIO(buffer)
    reader = PdfReader(pdf_stream)
    for i, page in enumerate(reader.pages, start=1):
        yield Chunk(page_number=i, text=page.extract_text())

splitter = RecursiveCharacterTextSplitter(chunk_size=450, chunk_overlap=50)

def pdf_buffer_to_chunks(buffer:bytes, separator:str|re.Pattern[str]|TextSplitter = splitter) -> Generator[Chunk]:
    pages = pdf_buffer_to_pages(buffer)
    return _to_chunks(list(pages), separator)

def split_pages(texts:list[Chunk], separator:str|re.Pattern[str]|TextSplitter = splitter) -> Generator[Chunk]:
    return _to_chunks(texts, separator)

##########

def _to_chunks(pages:List[Chunk],separator:str|re.Pattern[str]|TextSplitter = splitter):
    if not separator:
        raise ValueError("Must contain separators.")
        
    if isinstance(separator, TextSplitter):
        for i,page in enumerate(pages):
            enhanced_page_text = _enhance_page_text(i, pages)
            splits =  separator.split_text(enhanced_page_text)
            for split in splits:
                yield Chunk(text=split, page_number=page.page_number)

    else:
        for page in pages:
            for chunk in re.split(separator,page.text):
                yield Chunk(text=chunk, page_number=page.page_number)

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