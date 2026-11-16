# LLM

A Python library for working with language models through DSPy and building
retrieval-augmented generation (RAG) workflows with document embeddings and
Supabase. It provides provider adapters, document chunking, embedding generation,
and LangChain-compatible dense, sparse, and hybrid retrievers.

The project is a library imported into Python scripts or notebooks. There is no
command-line application, web server, or complete question-answering pipeline.
Generation, retrieval, and persistence can be composed by the calling application.

## Contents

- [Requirements and setup](#requirements-and-setup)
- [Configuration](#configuration)
- [Language models with DSPy](#language-models-with-dspy)
- [Embedding providers](#embedding-providers)
- [Document chunking](#document-chunking)
- [Document embedding workflows](#document-embedding-workflows)
- [Supabase integration](#supabase-integration)
- [Retrieval](#retrieval)
- [Ollama and SSH forwarding](#ollama-and-ssh-forwarding)
- [Repository layout](#repository-layout)
- [Development and limitations](#development-and-limitations)

## Requirements and setup

- Python **3.12 or newer**, as declared in `pyproject.toml`.
- [uv](https://docs.astral.sh/uv/) for the checked-in dependency lockfile.
- Credentials for the hosted providers you use, or a reachable Ollama server.
- A separately configured Supabase project for database-backed workflows.

From the repository root:

```sh
uv sync --locked
uv run python -c "import llm; print('llm imported successfully')"
```

Run your own scripts from the repository root with `uv run python your_script.py`.
The lockfile treats this project as a virtual project: dependencies are installed,
but the `llm` package itself is imported from the checkout. The repository does not
configure a build backend or an installed command-line entry point.

`.python-version` selects Python **3.14** by default. The local dependency stack
emits a warning about LangChain's use of Pydantic v1 under Python 3.14. If needed,
select another interpreter explicitly, for example `uv sync --locked --python 3.13`.
The declared Python range is not backed by a checked-in compatibility test matrix.

## Configuration

Create a `.env` file in the repository root, or set environment variables before
starting Python. The library calls `load_dotenv()` in its configuration helpers.
`.env` and `.env.*` are ignored by Git.

Only set the values needed for your workflow:

```dotenv
# Provider adapters use these exact spellings.
Openai_API_KEY=your-openai-key
Openrouter_API_KEY=your-openrouter-key
Google_API_KEY=your-google-key

# GoogleEmbeddingModel also constructs genai.Client() without passing its key.
# Set the Google SDK environment variable as well when using Google embeddings.
GOOGLE_API_KEY=your-google-key

# Supabase database and storage
SUPABASE_API_URL=https://your-project.supabase.co
SUPABASE_API_KEY=your-supabase-key
DB_SCHEMA=llm
STORAGE_BUCKET=documents

# Optional SSH forwarding for a remote Ollama instance
SSH_HOST=user@your-ollama-host
LOCAL_PORT=18080
REMOTE_PORT=11434
```

| Variable | Used for | Default / requirement |
| --- | --- | --- |
| `Openai_API_KEY` | OpenAI wrappers | Required unless `key=` is passed directly |
| `Openrouter_API_KEY` | OpenRouter wrappers | Required unless `key=` is passed directly |
| `Google_API_KEY` | Google wrappers' shared key lookup | Required unless `key=` is passed directly |
| `GOOGLE_API_KEY` | Google embedding SDK authentication | Configure separately; the wrapper does not forward its key to the SDK client |
| `SUPABASE_API_URL` | Supabase project URL | Required by `get_supabase()` |
| `SUPABASE_API_KEY` | Supabase project key | Required by `get_supabase()` |
| `DB_SCHEMA` | Supabase database schema | `llm` |
| `STORAGE_BUCKET` | Supabase Storage bucket | `documents` |
| `SSH_HOST` | SSH destination for `ensure_tunnel()` | Required when creating a tunnel |
| `LOCAL_PORT` | Ollama tunnel and model-management port | `18080` |
| `REMOTE_PORT` | Remote Ollama port forwarded through SSH | `11434` |

The mixed-case provider names come from `provider.capitalize()` in
`get_provider_key()`. Use the exact names above for portability across operating
systems. Standard uppercase provider variables alone do not satisfy this lookup
on case-sensitive systems. Provider classes accept an explicit `key=`; the
`get_embedding_model()` factory accepts only a provider name and model name.

Load `.env` before importing `llm` when relying on SSH settings: the Ollama helper
reads those values at import time. See [Ollama and SSH forwarding](#ollama-and-ssh-forwarding).

## Language models with DSPy

Each language model class extends the shared `DspyLM`, which extends `dspy.LM`.
Import the class from its provider module:

| Provider | Class | Prefix added to the supplied model name |
| --- | --- | --- |
| OpenAI | `llm.provider.openai.OpenaiLM` | `openai/` |
| OpenRouter | `llm.provider.openrouter.OpenrouterLM` | `openrouter/` |
| Google | `llm.provider.google.GoogleLM` | `gemini/` |
| Ollama | `llm.provider.ollama.OllamaLM` | `ollama_chat/` |

For example, configure a DSPy predictor with OpenAI:

```python
import dspy

from llm.provider.openai import OpenaiLM

lm = OpenaiLM(
    model="gpt-4o-mini",
    temperature=0.2,
    max_tokens=512,
)
dspy.configure(lm=lm)

answer = dspy.Predict("question -> answer")
result = answer(question="What is retrieval-augmented generation?")
print(result.answer)
```

Model names in these examples are illustrative; choose one available to your
provider account. Supply the model identifier without the wrapper's prefix. An
OpenRouter identifier can itself contain a vendor prefix, such as
`openai/gpt-oss-120b`, producing `openrouter/openai/gpt-oss-120b` internally.

The wrappers expose DSPy options including `model_type` (`chat`, `text`, or
`responses`), `temperature`, `max_tokens`, caching, callbacks, retry count, and
additional keyword arguments. Support for individual options depends on the
underlying model and provider. Shared defaults include `temperature=0.5`,
`cache=True`, and `num_retries=3`.

## Embedding providers

Use the factory for a common interface:

```python
from llm.provider import get_embedding_model

model = get_embedding_model("openai", "text-embedding-3-small")

# Both calls return list[Embedding], including the single-string call.
single = model.embed("A document about information retrieval.")
batch = model.embed(["First document", "Second document"], step=32)

print(single[0].text)
print(len(single[0].vector))
```

Each `Embedding` contains `text: str` and `vector: list[float]`. `step` divides a
list input into batches. Without it, the full list is sent in one call.

| Factory provider | Implementation | Model behavior |
| --- | --- | --- |
| `openai` | `OpenaiEmbeddingModel` | Requires an explicit model |
| `openrouter` | `OpenrouterEmbeddingModel` | Class default: `sentence-transformers/all-minilm-l12-v2` |
| `google` | `GoogleEmbeddingModel` | Class default: `gemini-embedding-001` |
| `ollama` | `OllamaEmbeddingModel` | Class default: `mxbai-embed-large`; checks/pulls the model on construction |

The factory requires both arguments even when the underlying class has a default.
Provider names are lowercase and case-sensitive. Unknown names raise
`ValueError("Embedding provider not found.")`.

Use the same embedding model and vector dimensions for stored sections and query
embeddings. The library does not store or enforce this configuration automatically.

Embedding API exceptions are printed and converted to empty results by the base
class. Always check the returned count. With `step=`, a failed intermediate batch
can also misalign later vectors with the original texts because results are
collected and paired with the input afterward. Validate results before persistence
or retry batches individually when reliable text/vector alignment is essential.

## Document chunking

Local ingestion recursively reads `.pdf`, `.md`, and `.txt` files, matching file
extensions case-insensitively:

```python
from llm.preprocessing.chunking import create_chunks_from_local_documents

documents = create_chunks_from_local_documents(
    document_folder="./documents",
    separator=r"\n\n",
    include_meta_in_chunks=True,
)

for document in documents:
    print(document.id, document.meta)
    for chunk in document.chunks:
        print(chunk.id, chunk.text, chunk.meta)
```

`separator` accepts a regex string, a compiled regex, or a LangChain
`TextSplitter`. When omitted, the default `RecursiveCharacterTextSplitter` uses:

- `chunk_size=450` characters;
- `chunk_overlap=50` characters;
- separators `\n\n`, `.`, `?`, and `!`.

The default splitter path enhances each page's text with neighboring text before
splitting; this can repeat content. A regex separator splits the original page
text directly.

For individual buffers, use `pdf_buffer_to_chunks()`, `md_buffer_to_chunks()`, or
`txt_buffer_to_chunks()`. They return `(list[Chunk], metadata)`. PDF extraction uses
`pypdf` and does not provide OCR. Text and Markdown are decoded as UTF-8 with
decoding errors ignored. Markdown front matter is parsed as simple `key: value`
lines, rather than full YAML, and returned as document metadata.

### Data models

The preprocessing models are defined in `llm.lib.types`:

| Model | Fields |
| --- | --- |
| `Chunk` | `id: int \| str`, `text: str`, `meta: dict` |
| `ChunkedDocument` | `id: str \| int`, `chunks: list[Chunk]`, `meta: dict` |
| `Section` | `document_id`, `content`, `embedding`, `meta`, optional `document_preprocessed_id` and `chunking_experiment_id` |
| `EmbeddedDocument` | `id: str \| int`, `sections: list[Section]`, `meta: dict` |

Local document IDs are assigned by enumeration during directory traversal. They
are not stable identifiers or existing Supabase document IDs. When
`include_meta_in_chunks=True`, file-path metadata is retained alongside the local
document ID. The current chunking implementation records zero-based
`meta["page_number"]`; `pdf_buffer_to_pages()` itself yields one-based page IDs.

## Document embedding workflows

### Local files

Use separate chunking and embedding steps:

```python
from llm.preprocessing.chunking import create_chunks_from_local_documents
from llm.preprocessing.embedding import create_embeddings_from_chunked_documents

chunked = create_chunks_from_local_documents(
    "./documents",
    separator=r"\n\n",
    include_meta_in_chunks=True,
)

embedded = create_embeddings_from_chunked_documents(
    chunked,
    embedding_provider="openai",
    embedding_model="text-embedding-3-small",
)

for source, document in zip(chunked, embedded):
    if len(source.chunks) != len(document.sections):
        raise RuntimeError(f"Incomplete embeddings for document {source.id}")
    print(document.id, len(document.sections))
```

These functions return in-memory objects. Map local IDs to database document IDs
and supply any required foreign keys before uploading to Supabase.

There is also a `create_embeddings_from_local_documents()` convenience function,
but its current implementation passes Python's built-in `id` function as the
section's `document_id`, causing validation to fail for non-empty chunks. Use the
two-step workflow above until that implementation is corrected.

### Files stored in Supabase

`create_embeddings_from_db()` reads document metadata and downloads files from
Supabase Storage:

```python
from llm.lib.supabase import get_supabase
from llm.preprocessing.embedding import create_embeddings_from_db

client = get_supabase()
embedded = create_embeddings_from_db(
    documents=["handbook.pdf"],
    bucket="documents",
    embedding_provider="openai",
    embedding_model="text-embedding-3-small",
    separator=r"\n\n",
    client=client,
)

for document in embedded:
    print(document.id, len(document.sections))
```

Document filters match `storage_object_path`. With `folder=`, the folder is joined
to each document name using `pathlib.Path`; ensure the resulting separator matches
your storage keys. With no document filter, all rows in
`documents_with_storage_path` are considered.

The function skips documents that already have *any* section row by default; it
does not check that all sections have embeddings or filter the skip check by
corpus or experiment. Set `skip_existing_embeddings=False` to regenerate.
Missing downloads, unsupported extensions, and documents producing no chunks are
skipped. Despite its docstring, this function **returns objects without uploading
them**. Persist sections explicitly after checking the output.

## Supabase integration

`get_supabase()` creates a synchronous Supabase client for `DB_SCHEMA` and uses an
HTTP/1.1 `httpx` client. Most helpers also accept an existing `client=`.

### Required database contract

The repository contains no SQL schema, migrations, storage provisioning, or access
policies. Provision the following objects separately for the helpers you use, and
expose the selected schema through the Supabase API with appropriate permissions.

| Object | Expected use |
| --- | --- |
| `documents` | Document records referenced by preprocessing and sections |
| `documents_with_storage_path` | Table/view exposing `id`, `name`, and `storage_object_path` for downloads |
| `document_corpus` | Corpus records with a name, description, and metadata |
| `document_preprocessed` | Content linked to a document and optional corpus, with preprocessing metadata |
| `chunking_experiment` | Named chunking strategies linked to a corpus, with notes and metadata |
| `document_sections` | Section content, vector embeddings, metadata, and document/experiment links |
| `match_document_sections` | RPC for vector similarity search |
| Storage bucket | File bytes, default bucket name `documents` |

The vector-search RPC receives `query_embedding` and `match_count`, plus optional
`document_ids`, `corpus_id`, and `experiment_id`. Retrieval expects rows containing
`content`, `id`, `meta`, and `similarity`; the query-result type also declares
`document_id`. The vector column and RPC must match your embedding dimensions.

Sparse retrieval calls PostgREST full-text search on `document_sections.content`.
Configure suitable full-text search support in the database for your data and
language. The helper does not add a ranking expression or explicit ordering.

Preprocessing upserts require unique constraints matching `document_id`, or
`document_id,corpus_id` when a corpus is supplied. Section upserts default to the
conflict target `content,meta`; configure a matching constraint or provide the
appropriate `on_conflict=` value. Nested document reads also require a relationship
from `document_preprocessed` to `documents`.

Retrieval's `experiment_id` filter uses an `experiment_id` column for sparse search
and an RPC argument of that name for dense search. Section listing and upload
helpers use `chunking_experiment_id`. Confirm how your external schema maps these
names; the library does not translate between them.

### Persist generated sections

After generating `embedded` from database-backed documents as above, build section
rows with the required preprocessing and experiment links:

```python
from llm.lib.supabase import Section as DatabaseSection, upload_document_sections

# Replace these example IDs with existing rows for the document being processed.
corpus_id = 1
preprocessed_id = 10
chunking_experiment_id = 2

document = embedded[0]  # Check that the generation result is non-empty first.
sections = [
    DatabaseSection(
        document_id=section.document_id,
        content=section.content,
        embedding=section.embedding,
        meta=section.meta,
        document_preprocessed_id=preprocessed_id,
        chunking_experiment_id=chunking_experiment_id,
    )
    for section in document.sections
]

if sections:
    inserted = upload_document_sections(sections, client=client, corpus_id=corpus_id)
    print(f"Inserted {len(inserted)} sections")
```

`llm.lib.supabase.Section` is distinct from `llm.lib.types.Section`: the database
model requires both linkage IDs and allows an absent embedding. `upload_document_sections()`
inserts rows and adds the supplied `corpus_id`; it does not upsert or deduplicate.

### Other database helpers

All of these are available from `llm.lib.supabase`:

| Area | Helpers |
| --- | --- |
| Storage | `get_storage_object`, `get_documents_buffers` |
| Corpora | `create_document_corpus`, `get_document_corpus`, `get_document_corpus_by_name`, `list_document_corpora`, `get_or_create_document_corpus`, `get_corpus_documents` |
| Experiments | `create_chunking_experiment`, `get_chunking_experiment_by_name`, `get_or_create_chunking_experiment` |
| Preprocessed content | `upsert_document_preprocessed`, `get_document_preprocessed`, `list_document_preprocessed` |
| Sections | `list_document_sections`, `upload_document_sections`, `upsert_document_sections`, `add_embeddings_document_sections` |
| Raw queries | `query_vector_db`, `query_text_search` |

`get_documents_buffers()` yields `(document_row, bytes_or_none)` and supports
extension filters such as `extensions=["pdf", ".md"]`. Failed downloads print the
exception and return `None`.

`get_or_create_chunking_experiment()` updates existing experiments: it appends
notes, deep-merges metadata, and writes the supplied corpus ID. Pass `corpus_id`
when updating as well as creating to avoid replacing it with `None`.

`list_document_sections(with_embeddings=True)` selects sections with vectors;
`False` selects sections without vectors, and `None` leaves them unfiltered.
`add_embeddings_document_sections()` accepts `(section_id, vector)` pairs, skips
already embedded rows by default, and persists via a content/metadata upsert.
It reads the section table before applying updates.

Only the bulk section upsert path has an explicit transport-error retry helper:
up to four attempts with exponential backoff starting at 0.2 seconds. This is not
a general retry policy for every database or provider operation.

## Retrieval

The retrieval modules provide LangChain `BaseRetriever` subclasses and convenience
functions. Results are lists of `langchain_core.documents.Document` objects.

```python
from llm.lib.supabase import get_supabase
from llm.retrieval.dense import dense_search_db
from llm.retrieval.sparse import sparse_search_db
from llm.retrieval.hybrid import hybrid_search_db

client = get_supabase()

dense = dense_search_db(
    "How are documents indexed?",
    embedding_provider="openai",
    embedding_model="text-embedding-3-small",
    corpus_id=1,
    match_count=5,
    client=client,
)

sparse = sparse_search_db("document indexing", corpus_id=1, client=client)

hybrid = hybrid_search_db(
    "document indexing",
    embedding_provider="openai",
    embedding_model="text-embedding-3-small",
    dense_match_count=5,
    sparse_match_count=5,
    match_count=5,
    corpus_id=1,
    client=client,
)

for result in hybrid:
    print(result.page_content, result.metadata)
```

| Retriever | Behavior | Added result metadata |
| --- | --- | --- |
| `DenseSupabaseRetriever` | Embeds the query, then calls `match_document_sections` | `id`, `similarity` |
| `SparseSupabaseRetriever` | Converts query words to a PostgreSQL full-text expression | `id`, `document_id` when present |
| `HybridSupabaseRetriever` | Merges dense and sparse results through LangChain's `MergerRetriever`, then truncates | Metadata from each source |

Existing row `meta` values are retained, with the fields above added or replaced.
All three support `document_ids`, `corpus_id`, `experiment_id`, and `client`.
Dense and hybrid retrieval default to OpenRouter's
`sentence-transformers/all-minilm-l12-v2`; pass explicit embedding settings when
your stored vectors use another model.

For repeated use or LangChain composition, instantiate a retriever and invoke it:

```python
from llm.retrieval.sparse import SparseSupabaseRetriever

retriever = SparseSupabaseRetriever(client=client, corpus_id=1, match_count=3)
results = retriever.invoke("document indexing")
```

Sparse multiword queries combine AND and OR expressions. The wrapper is intended
for plain query words; use `query_text_search()` directly for an explicit
PostgreSQL query such as `"'phishing' & 'malware'"`. Sparse result limits are applied
in Python after fetching matches.

Hybrid retrieval interleaves its sources through the merger. It does not perform
score fusion, reranking, or deduplication, so the same section can appear twice.
It loads `MergerRetriever` from `langchain_classic`, with a fallback to the older
`langchain.retrievers` import path.

## Ollama and SSH forwarding

The Ollama provider clients use the fixed URL `http://localhost:18080` in
`llm/lib/url.py`. Model management uses `LOCAL_PORT` from the environment, so
changing that variable alone does not change the provider client URL. Keep the
default port unless updating the code to use a different endpoint.

For a local Ollama server, make it reachable at port `18080`. For a remote server,
forward that port to the remote Ollama port, either manually or with the helpers:

```python
from dotenv import load_dotenv

load_dotenv()  # Must run before importing the Ollama helpers.

from llm.lib.remote_ollama import ensure_tunnel, wait_until_ready
from llm.provider.ollama import OllamaEmbeddingModel

tunnel = ensure_tunnel()
try:
    wait_until_ready()
    model = OllamaEmbeddingModel("mxbai-embed-large")
    embeddings = model.embed("A local embedding example.")
finally:
    if tunnel is not None:
        tunnel.terminate()
        tunnel.wait()
```

The tunnel helper runs `ssh -N -L LOCAL_PORT:localhost:REMOTE_PORT SSH_HOST`, so SSH
must be installed and authentication configured. It returns a subprocess when it
starts a tunnel, or `None` when the endpoint already responds. The caller owns
the process lifetime.

Both Ollama model classes call `ensure_model()` during construction, checking
`/api/tags` and requesting `/api/pull` when the model name is missing. This can
download a model. They do not create a tunnel automatically; `OllamaLM`'s
`create_tunnel` argument is currently unused.

## Repository layout

```text
llm/
  __init__.py                   # Exposes provider and retrieval modules
  provider/
    __init__.py                 # get_embedding_model() factory
    base/
      dspy_lm_base.py           # Shared DSPy LM wrapper
      embedding_model.py       # Embedding model interface and result model
    openai.py                   # OpenAI LM and embedding adapters
    openrouter.py               # OpenRouter LM and embedding adapters
    google.py                   # Google LM and embedding adapters
    ollama.py                   # Ollama LM and embedding adapters
  preprocessing/
    chunking.py                 # Local and buffer-based document chunking
    embedding.py                # Chunk-to-vector workflows
  retrieval/
    dense.py                    # Vector retrieval
    sparse.py                   # Full-text retrieval
    hybrid.py                   # Merged retrieval
  lib/
    types.py                    # Chunk/document/section models
    supabase.py                 # Storage, persistence, and query helpers
    remote_ollama.py            # SSH forwarding and model availability
    url.py                      # Fixed provider endpoints
    utils.py                    # Environment keys and logging
pyproject.toml                  # Project version and dependency declarations
uv.lock                         # Locked dependency resolution
.python-version                 # Default Python interpreter selection
```

The declared project version is `0.3.5`. Dependencies include DSPy, Google Gen AI,
Ollama, LangChain, Supabase, pypdf, SQLAlchemy, and pgvector. Some imports, including
OpenAI, Requests, HTTPX, and Pydantic, currently rely on transitive dependencies.

## Development and limitations

There is no checked-in automated test suite, CI workflow, or lint/type-check
configuration. A local `test.ipynb` is ignored and is not part of the reproducible
repository. For a basic check without provider calls or database access, run:

```sh
uv run python -m compileall -q llm
uv run python -c "from llm.preprocessing.chunking import txt_buffer_to_chunks; chunks, _ = txt_buffer_to_chunks(b'First\n\nSecond', separator=r'\n\n'); assert [c.text for c in chunks] == ['First', 'Second']; print('chunking OK')"
```

Integration checks require your own credentials, models, database objects, and
storage data. Importing and chunking locally do not establish that a provider or
Supabase deployment is configured correctly.

Common issues to check:

| Symptom | Check |
| --- | --- |
| `Please provide ... in .env` | Exact mixed-case provider variable, working directory, or explicit `key=` |
| Google embeddings fail to authenticate | Both wrapper key lookup and the SDK's separate environment configuration |
| `Supabase key is not provided.` / `Supabase URL is not provided.` | `SUPABASE_API_KEY` and `SUPABASE_API_URL` |
| Missing table, relationship, or RPC | External schema, API exposure, permissions, and database contract above |
| Vector dimension errors or poor dense results | Matching ingestion/query model and database vector dimensions |
| Missing or incomplete embeddings | Printed provider errors and returned embedding counts |
| Dense search raises an index error | Query embedding may have failed and returned an empty list |
| Local convenience embedding raises validation errors | Use the documented two-step chunking/embedding workflow |
| Ollama connection fails | Endpoint at port `18080`, SSH tunnel, and matching model-management port |
| Hybrid results contain duplicates | Current merge behavior does not deduplicate |

`pyproject.toml` refers to `README.md`, while the tracked documentation filename is
`readme.md`. Align that casing before introducing packaging on a case-sensitive
filesystem. The repository does not currently include a license file.
