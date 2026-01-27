

import os
from pathlib import Path
from logging import Logger
from dotenv import load_dotenv
from llm.lib.supabase import get_storage_object, get_supabase, upload_document_sections
from llm.lib.types import Section
from llm.lib.utils import get_logger
from llm.preprocessing.chunking import pdf_buffer_to_sections
from llm.provider import get_embedding_model

load_dotenv()
BUCKET = os.getenv("STORAGE_BUCKET", "")


def create_embeddings(documents: list[str] | None = None,
                      folder: Path | None = None,
                      bucket: str = BUCKET,
                      separator: str = r"\n\n",
                      embedding_model:str="sentence-transformers/all-minilm-l12-v2",
                      embedding_provider:str="openrouter",
                      logger = get_logger()):
    client = get_supabase()
    if documents:
        if folder:
            file_paths = [str(folder / d) for d in documents]
        else:
            file_paths = documents
        response = (
            client.table("documents_with_storage_path")
            .select("id, name")
            .in_("storage_object_path", file_paths)
            .execute()
        )
    else:
        response = (
            client.table("documents_with_storage_path")
            .select("id, name")
            .execute()
        )

    docs: list[dict] = response.data  # type:ignore
    logger.debug(f"retrieved {len(docs)} documents.")

    ### ======== EMBEDDING ========= ###
    model = get_embedding_model(embedding_provider, embedding_model)
    # embed document wise
    for doc in docs:
        # check for existing sections
        if (
            client.table("document_sections")
            .select("*")
            .eq("document_id",doc["id"])
            .execute()
        ).data:
            print(f"document with id {doc["id"]} already embedded. Skipping...")
            logger.debug(f"document with id {doc["id"]} already embedded. Skipping...")
            continue

        buffer = get_storage_object(doc["name"], client, bucket)
        chunks = list(pdf_buffer_to_sections(buffer, separator))
        texts = [c.text for c in chunks]
        embeddings = model.embed(texts)
        sections: list[Section] = [
            Section(content=chunk.text,
                    document_id=doc["id"],
                    embedding=embedding.vector, meta={
                        "page_number": chunk.page_number
                    }) for chunk, embedding in zip(chunks, embeddings) if _raise_matching_error(chunk.text, embedding.text)
        ]
        logger.debug(f"created {len(sections)} sections.")
        if sections:
            res = upload_document_sections(sections=sections,client=client)
            logger.debug(f"superbase responded opn upload: {res}")

def _raise_matching_error(text1, text2):
    if text1 == text2:
        return True
    else:
        raise ValueError("chunk and embedding incoherent.")
