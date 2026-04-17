from pydantic import BaseModel, Field


class Chunk(BaseModel):
    id: int | str
    text: str
    meta: dict = Field(default_factory=dict)


class Section(BaseModel):
    document_id: str | int
    content: str
    embedding: list[float]
    meta: dict = Field(default_factory=dict)
    document_preprocessed_id: int | None = None
    chunking_experiment_id: int | None = None

class EmbeddedDocument(BaseModel):
    id: str | int
    sections: list[Section]
    meta: dict = Field(default_factory=dict)


class ChunkedDocument(BaseModel):
    id: str | int
    chunks: list[Chunk]
    meta: dict = Field(default_factory=dict)
