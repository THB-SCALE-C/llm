from io import BytesIO
import re
from typing import Any, Dict, Generator, TypedDict
from pydantic import BaseModel
from pypdf import PdfReader


class Chunk(BaseModel):
    text: str
    page_number: int


def pdf_buffer_to_pages(buffer: bytes) -> Generator[Chunk]:
    pdf_stream = BytesIO(buffer)
    reader = PdfReader(pdf_stream)
    for i, page in enumerate(reader.pages, start=1):
        yield Chunk(page_number=i, text=page.extract_text())


def pdf_buffer_to_sections(buffer:bytes, separator:str|re.Pattern[str] = r"\n\s+|\n\n") -> Generator[Chunk]:
    pages = pdf_buffer_to_pages(buffer)
    if not separator:
        raise ValueError("Must contain separators.")
    for page in pages:
        for chunk in re.split(separator,page.text):
            yield Chunk(text=chunk, page_number=page.page_number)
