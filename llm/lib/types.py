from pydantic import BaseModel


class Chunk(BaseModel):
    id: int | str
    text: str
    meta: dict


class Section(BaseModel):
    document_id:str|int
    content:str
    embedding:list[float]
    meta:dict={}

class EmbeddedDocument(BaseModel):
    id:str|int
    sections:list[Section]
    meta:dict = {}


class ChunkedDocument(BaseModel):
    id:str|int
    chunks:list[Chunk]
    meta:dict = {}