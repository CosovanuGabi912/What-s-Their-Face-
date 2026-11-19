"""search_knowledge (Part A, issue #9): retrieval from the ChromaDB knowledge collection.

The query is embedded with config.EMBED_QUERY_PREFIX through LM Studio, the same model that embedded the
chunks (scripts/ingest.py adds config.EMBED_DOC_PREFIX to every chunk before embedding). The optional breed and
section arguments become a Chroma metadata filter. Chroma accepts one condition as {"breed": x}, but two
conditions need {"$and": [{"breed": x}, {"section": y}]}; a dict with two keys is rejected (_build_where).

Behaviour:
- breed and section are validated with shared.schemas.SearchKnowledgeInput (one source of truth with the stub):
  any known spelling, a partial name that fits exactly one breed ("Labrador") and section synonyms ("Diet")
  are resolved.
- A breed or section that still does not resolve ("NonExistentBreed999") returns 0 chunks, never an error.
  The agent then says it does not know (acceptance criterion of issue #9).
- A breed+section filter that matches nothing falls back to the breed filter alone (logged).
- A broken setup (LM Studio not running or serving the wrong model, collection not ingested, wrong vector size,
  any other Chroma failure) raises a RuntimeError that says what actually failed, because an empty result would
  look like "nothing relevant" and hide the real problem.

The Chroma client and the collections are cached for the life of the process. After the data is re-unzipped or
`scripts/ingest.py --all` re-ingests, RESTART the MCP server (the Chainlit app starts it), or call
reset_caches(): a running server keeps the old database open and can serve stale chunks.

Rules for tool code (see the notes on issue #9): never write to stdout (the MCP stdio stream); use logging.
The Chroma client and the embedding client are created once, on first use, not on every call and not at import.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Optional

from pydantic import ValidationError

import config
from shared.schemas import KnowledgeChunk, SearchKnowledgeInput, SearchKnowledgeOutput

log = logging.getLogger(__name__)

KNOWLEDGE_COLLECTION = "knowledge"  # the section-chunked collection (scripts/ingest.py); "knowledge_90w" is the baseline


# --- Clients, created once --------------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _embedding_client():
    from openai import OpenAI

    return OpenAI(base_url=config.LM_STUDIO_BASE_URL, api_key=config.LM_STUDIO_API_KEY, timeout=30, max_retries=1)


@lru_cache(maxsize=1)
def _chroma_client():
    import chromadb

    if not (config.CHROMA_KNOWLEDGE_DIR / "chroma.sqlite3").exists():  # PersistentClient would create an empty DB
        raise RuntimeError(f"No knowledge database in {config.CHROMA_KNOWLEDGE_DIR}. Unzip the data from Part A "
                           "or run `python scripts/ingest.py --all` (see the README, 'How to run').")
    return chromadb.PersistentClient(path=str(config.CHROMA_KNOWLEDGE_DIR))


def _is_not_found(error: Exception) -> bool:
    """True only for "this collection does not exist" (NotFoundError; older chromadb raised ValueError)."""
    return type(error).__name__ == "NotFoundError" or (isinstance(error, ValueError) and "does not exist" in str(error))


@lru_cache(maxsize=4)
def _collection(name: str):
    try:
        return _chroma_client().get_collection(name)
    except RuntimeError:
        raise
    except Exception as error:  # noqa: BLE001 - chromadb raises different errors by version
        if _is_not_found(error):
            raise RuntimeError(f"Collection {name!r} not found in {config.CHROMA_KNOWLEDGE_DIR}: run "
                               f"`python scripts/ingest.py --all`.") from error
        raise RuntimeError(f"Cannot open collection {name!r} in {config.CHROMA_KNOWLEDGE_DIR}: {error}. This is not "
                           "a missing collection, so re-running ingest.py will not fix it (chromadb version "
                           "mismatch, or the database is locked by another process).") from error


def reset_caches() -> None:
    """Forget the Chroma client and collections (after re-ingesting, and in tests)."""
    _chroma_client.cache_clear()
    _collection.cache_clear()


def _embed(text: str) -> list[float]:
    """Embed a question with the query prefix. Failures name the fix instead of a stack of HTTP errors."""
    try:
        response = _embedding_client().embeddings.create(model=config.EMBED_MODEL_ID,
                                                         input=config.EMBED_QUERY_PREFIX + text)
    except Exception as error:  # noqa: BLE001 - connection refused, model not loaded, timeout, ...
        raise RuntimeError(f"Cannot embed the question with {config.EMBED_MODEL_ID!r} at {config.LM_STUDIO_BASE_URL}: "
                           f"{str(error).rstrip('.')}. Start the LM Studio server and load the embedding model "
                           f"(nomic-embed-text-v1.5 Q4_K_M).") from error
    data = getattr(response, "data", None)
    embedding = data[0].embedding if data else None
    if not embedding:
        raise RuntimeError(f"LM Studio at {config.LM_STUDIO_BASE_URL} returned no embedding for "
                           f"{config.EMBED_MODEL_ID!r}. A chat model may be loaded under that ID: load the embedding "
                           "model (nomic-embed-text-v1.5 Q4_K_M) in LM Studio.")
    return list(embedding)


# --- Filters ----------------------------------------------------------------------------------------------
def _build_where(breed: Optional[str], section: Optional[str]) -> Optional[dict]:
    """Chroma metadata filter: none, one condition as it is, two conditions joined with $and."""
    conditions = [{key: value} for key, value in (("breed", breed), ("section", section)) if value]
    if not conditions:
        return None
    return conditions[0] if len(conditions) == 1 else {"$and": conditions}


def _resolve_filters(breed: Optional[str], section: Optional[str]) -> Optional[tuple[Optional[str], Optional[str]]]:
    """(breed, section) as stored in Chroma, resolved by the contract validators; None when one is unknown."""
    try:
        args = SearchKnowledgeInput.model_validate({"query": "x", "breed": breed, "section": section})
    except ValidationError as error:
        log.info("search_knowledge: %s -> returning no chunks", "; ".join(e["msg"] for e in error.errors()))
        return None
    return args.breed, args.section


# --- The tool ---------------------------------------------------------------------------------------------
def _query(collection, embedding: list[float], where: Optional[dict], top_k: int) -> list[KnowledgeChunk]:
    """collection.query as chunks, with Chroma failures turned into messages that say what failed."""
    try:
        result = collection.query(query_embeddings=[embedding], n_results=top_k, where=where,
                                  include=["documents", "metadatas", "distances"])
    except Exception as error:  # noqa: BLE001
        if "dimension" in f"{type(error).__name__} {error}".lower():
            raise RuntimeError(f"The question embedding has {len(embedding)} dimensions but collection "
                               f"{collection.name!r} was built with another size ({error}). LM Studio is serving a "
                               f"different model than the one that ingested the data: load {config.EMBED_MODEL_ID!r} "
                               "(nomic-embed-text-v1.5 Q4_K_M).") from error
        raise RuntimeError(f"The search in collection {collection.name!r} failed: {error}") from error
    return [
        KnowledgeChunk(chunk_id=meta["chunk_id"], breed=meta["breed"], section=meta["section"], text=document,
                       score=round(1.0 - distance, 4))  # Chroma returns the cosine distance
        for document, meta, distance in zip(result["documents"][0], result["metadatas"][0], result["distances"][0])
    ]


def _search(query: str, breed: Optional[str] = None, section: Optional[str] = None, *,
            collection_name: str = KNOWLEDGE_COLLECTION, top_k: int = config.SEARCH_TOP_K) -> SearchKnowledgeOutput:
    query = (query or "").strip()
    if not query:
        return SearchKnowledgeOutput(chunks=[])
    filters = _resolve_filters(breed, section)
    if filters is None:
        return SearchKnowledgeOutput(chunks=[])
    breed, section = filters

    collection = _collection(collection_name)
    embedding = _embed(query)
    chunks = _query(collection, embedding, _build_where(breed, section), top_k)
    if not chunks and breed and section:  # e.g. Siberian Husky has no Exercise chunk: keep the breed, drop the section
        log.info("search_knowledge: no chunk for breed=%s section=%s, retrying with the breed only", breed, section)
        section = None
        chunks = _query(collection, embedding, _build_where(breed, None), top_k)
    log.info("search_knowledge: breed=%s section=%s -> %s", breed, section, [c.chunk_id for c in chunks])
    return SearchKnowledgeOutput(chunks=chunks)


def search_knowledge(query: str, breed: str | None = None, section: str | None = None) -> SearchKnowledgeOutput:
    """The 3 chunks closest to `query`, optionally limited to one breed and/or one section."""
    return _search(query, breed, section)
