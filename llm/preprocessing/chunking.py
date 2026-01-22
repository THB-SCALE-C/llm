from io import BytesIO
from typing import Any, Dict, Generator, TypedDict
from pypdf import PdfReader


class Chunk(TypedDict):
    text: str
    num: int


def pdf_buffer_to_pages(buffer: bytes) -> Generator[Chunk]:
    pdf_stream = BytesIO(buffer)
    reader = PdfReader(pdf_stream)
    for i, page in enumerate(reader.pages, start=1):
        yield {"text": page.extract_text(), "num": i}



