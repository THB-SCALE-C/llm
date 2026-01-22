import os

from dotenv import load_dotenv
from sqlalchemy import create_engine, text

from retrieval.vector_similarity.db.chunks import Base, Chunk

load_dotenv()

DATABASE_URL = os.environ.get("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL env var is required to initialize the database.")

SCHEMA = os.environ.get("DB_SCHEMA", "LLM")

# build engine once
engine = create_engine(DATABASE_URL, future=True)


def ensure_schema_and_extension() -> None:
    with engine.begin() as conn:
        conn.execute(text('CREATE EXTENSION IF NOT EXISTS "vector"'))
        conn.execute(text(f'CREATE SCHEMA IF NOT EXISTS "{SCHEMA}"'))


def create_tables() -> None:
    # make sure the generated table/index land in the configured schema
    Base.metadata.schema = SCHEMA
    Chunk.__table__.schema = SCHEMA
    Base.metadata.create_all(bind=engine)


def main() -> None:
    ensure_schema_and_extension()
    create_tables()


if __name__ == "__main__":
    main()
