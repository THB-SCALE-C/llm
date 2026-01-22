from typing import Iterable

from sqlalchemy import BIGINT, Column, DateTime, Integer, Text, func, Index
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Session, declarative_base
from pgvector.sqlalchemy import Vector

Base = declarative_base()


class Chunk(Base):
    __tablename__ = "chunks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    text = Column(Text, nullable=False)
    embedding = Column(Vector(1536), nullable=False)
    document_id = Column(BIGINT,  nullable=True)
    meta = Column(JSONB, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    __table_args__ = (
        Index(
            "ix_chunks_embedding_ivfflat",
            embedding,
            postgresql_using="ivfflat",
            postgresql_with={"lists": 100},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )


def search_chunks(session: Session, query_embedding: Iterable[float], limit: int = 5) -> list["Chunk"]:
    """Return the closest chunks ordered by cosine distance."""
    distance = Chunk.embedding.cosine_distance(query_embedding)
    return (
        session.query(Chunk)
        .order_by(distance)  # uses ivfflat cosine index for speed
        .limit(limit)
        .all()
    )
