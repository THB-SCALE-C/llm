from pydantic import BaseModel


class Section(BaseModel):
    document_id:str|int
    content:str
    embedding:list[float]
    meta:dict={}

class Document(BaseModel):
    id:str|int
    sections:list[Section]
    meta:dict = {}