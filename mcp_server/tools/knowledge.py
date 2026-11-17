"""search_knowledge (Part A, issue #9): retrieval from the ChromaDB knowledge collection.

The query is embedded with config.EMBED_QUERY_PREFIX through LM Studio, the same model that embedded the
chunks (scripts/ingest.py adds config.EMBED_DOC_PREFIX to every chunk before embedding). The optional breed and
section arguments become a Chroma metadata filter. Chroma accepts one condition as {"breed": x}, but two
conditions need {"$and": [{"breed": x}, {"section": y}]}; a dict with two keys is rejected (_build_where).

Behaviour:
- An unknown breed or section returns 0 chunks, never an error. The agent then says it does not know.
- A broken setup (LM Studio not running, collection not ingested) raises a RuntimeError that says what to do,
  because an empty result would look like "nothing relevant" and hide the real problem.

Rules for tool code (see the notes on issue #9): never write to stdout (the MCP stdio stream); use logging.
The Chroma client and the embedding client are created once, on first use, not on every call and not at import.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Optional

import config
from shared.breeds import to_display
from shared.schemas import KnowledgeChunk, SearchKnowledgeOutput

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


@lru_cache(maxsize=4)
def _collection(name: str):
    try:
        return _chroma_client().get_collection(name)
    except RuntimeError:
        raise
    except Exception as error:  # noqa: BLE001 - chromadb raises different errors by version
        raise RuntimeError(f"Collection {name!r} not found in {config.CHROMA_KNOWLEDGE_DIR}: run "
                           f"`python scripts/ingest.py --all`.") from error


def _embed(text: str) -> list[float]:
    """Embed a question with the query prefix. Failures name the fix instead of a stack of HTTP errors."""
    try:
        response = _embedding_client().embeddings.create(model=config.EMBED_MODEL_ID,
                                                         input=config.EMBED_QUERY_PREFIX + text)
    except Exception as error:  # noqa: BLE001 - connection refused, model not loaded, timeout, ...
        raise RuntimeError(f"Cannot embed the question with {config.EMBED_MODEL_ID!r} at {config.LM_STUDIO_BASE_URL}: "
                           f"{error}. Start the LM Studio server and load the embedding model "
                           f"(nomic-embed-text-v1.5 Q4_K_M).") from error
    return response.data[0].embedding


# --- Filters ----------------------------------------------------------------------------------------------
def _build_where(breed: Optional[str], section: Optional[str]) -> Optional[dict]:
    """Chroma metadata filter: none, one condition as it is, two conditions joined with $and."""
    conditions = [{key: value} for key, value in (("breed", breed), ("section", section)) if value]
    if not conditions:
        return None
    return conditions[0] if len(conditions) == 1 else {"$and": conditions}


def _canonical_section(section: Optional[str]) -> Optional[str]:
    """The config.KNOWLEDGE_SECTIONS spelling of `section` (any case), or None when it is not one of them."""
    wanted = (section or "").strip().casefold()
    return next((s for s in config.KNOWLEDGE_SECTIONS if s.casefold() == wanted), None)


# --- The tool ---------------------------------------------------------------------------------------------
def _search(query: str, breed: Optional[str] = None, section: Optional[str] = None, *,
            collection_name: str = KNOWLEDGE_COLLECTION, top_k: int = config.SEARCH_TOP_K) -> SearchKnowledgeOutput:
    query, breed, section = (query or "").strip(), (breed or "").strip() or None, (section or "").strip() or None
    if not query:
        return SearchKnowledgeOutput(chunks=[])
    if breed is not None:
        breed = to_display(breed)  # any known spelling -> the display name stored in the metadata
        if breed is None:
            log.info("search_knowledge: unknown breed, returning no chunks")
            return SearchKnowledgeOutput(chunks=[])
    if section is not None:
        section = _canonical_section(section)
        if section is None:
            log.info("search_knowledge: unknown section, returning no chunks")
            return SearchKnowledgeOutput(chunks=[])

    result = _collection(collection_name).query(
        query_embeddings=[_embed(query)], n_results=top_k, where=_build_where(breed, section),
        include=["documents", "metadatas", "distances"])
    chunks = [
        KnowledgeChunk(chunk_id=meta["chunk_id"], breed=meta["breed"], section=meta["section"], text=document,
                       score=round(1.0 - distance, 4))  # Chroma returns the cosine distance
        for document, meta, distance in zip(result["documents"][0], result["metadatas"][0], result["distances"][0])
    ]
    log.info("search_knowledge: breed=%s section=%s -> %s", breed, section, [c.chunk_id for c in chunks])
    return SearchKnowledgeOutput(chunks=chunks)


def search_knowledge(query: str, breed: str | None = None, section: str | None = None) -> SearchKnowledgeOutput:
    """The 3 chunks closest to `query`, optionally limited to one breed and/or one section."""
    return _search(query, breed, section)
