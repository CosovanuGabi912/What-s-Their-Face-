"""Tests for mcp_server/tools/knowledge.py (issue #9).

Two layers:
    - offline tests on a small temporary Chroma collection with made-up 2-D vectors (no LM Studio, no data);
    - live tests on the real knowledge collection and LM Studio: skipped, with the reason, when either is missing.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import config
from mcp_server.tools import knowledge
from shared.schemas import SearchKnowledgeOutput


# --- Offline: filters, unknown values, prefix -------------------------------------------------------------
DOCS = [  # id, breed, section, vector (distance to the query vector [1, 0] grows down the list)
    ("kb:siberian-husky:overview:0", "Siberian Husky", "Overview", [1.0, 0.0]),
    ("kb:siberian-husky:health:0", "Siberian Husky", "Health", [0.8, 0.6]),
    ("kb:german-shepherd-dog:overview:0", "German Shepherd Dog", "Overview", [0.6, 0.8]),
    ("kb:beagle:health:0", "Beagle", "Health", [0.0, 1.0]),
]


@pytest.fixture
def small_db(tmp_path, monkeypatch):
    chromadb = pytest.importorskip("chromadb")
    collection = chromadb.PersistentClient(path=str(tmp_path)).get_or_create_collection(
        "knowledge", metadata={"hnsw:space": "cosine"})
    collection.upsert(ids=[d[0] for d in DOCS], documents=[f"{d[1]} - {d[2]}\ntext of {d[0]}" for d in DOCS],
                      metadatas=[{"chunk_id": d[0], "breed": d[1], "section": d[2], "source": "wikipedia"} for d in DOCS],
                      embeddings=[d[3] for d in DOCS])
    monkeypatch.setattr(knowledge, "_collection", lambda name: collection)
    monkeypatch.setattr(knowledge, "_embed", lambda text: [1.0, 0.0])
    return collection


def ids(output: SearchKnowledgeOutput) -> list[str]:
    return [chunk.chunk_id for chunk in output.chunks]


def test_build_where_none_one_or_two_conditions():
    assert knowledge._build_where(None, None) is None
    assert knowledge._build_where("Beagle", None) == {"breed": "Beagle"}
    assert knowledge._build_where(None, "Health") == {"section": "Health"}
    assert knowledge._build_where("Beagle", "Health") == {"$and": [{"breed": "Beagle"}, {"section": "Health"}]}


def test_chroma_rejects_a_filter_with_two_keys(small_db):
    """The bug in the issue's example code: this is why two conditions are joined with $and."""
    with pytest.raises(Exception):
        small_db.query(query_embeddings=[[1.0, 0.0]], n_results=1, where={"breed": "Beagle", "section": "Health"})


def test_no_filter_returns_the_three_closest(small_db):
    out = knowledge.search_knowledge("anything")
    assert ids(out) == ["kb:siberian-husky:overview:0", "kb:siberian-husky:health:0", "kb:german-shepherd-dog:overview:0"]
    assert out.chunks[0].score == pytest.approx(1.0) and out.chunks[0].score > out.chunks[1].score


def test_breed_filter(small_db):
    out = knowledge.search_knowledge("anything", breed="Siberian Husky")
    assert ids(out) == ["kb:siberian-husky:overview:0", "kb:siberian-husky:health:0"]
    assert {c.breed for c in out.chunks} == {"Siberian Husky"}


def test_breed_and_section_filter_together(small_db):
    out = knowledge.search_knowledge("anything", breed="Siberian Husky", section="Health")
    assert ids(out) == ["kb:siberian-husky:health:0"]


def test_section_filter_alone_and_in_any_case(small_db):
    assert ids(knowledge.search_knowledge("anything", section="health")) == ["kb:siberian-husky:health:0", "kb:beagle:health:0"]


def test_any_known_breed_spelling_is_resolved(small_db):
    assert ids(knowledge.search_knowledge("anything", breed="german shepherd")) == ["kb:german-shepherd-dog:overview:0"]


def test_known_breed_without_a_matching_chunk_returns_nothing(small_db):
    assert ids(knowledge.search_knowledge("anything", breed="Beagle", section="Overview")) == []


@pytest.mark.parametrize("breed, section", [("NonExistentBreed999", None), (None, "Nonsense"),
                                            ("Beagle", "Nonsense"), ("NonExistentBreed999", "Health")])
def test_unknown_breed_or_section_returns_no_chunks_and_never_embeds(small_db, monkeypatch, breed, section):
    def no_embedding(text):
        raise AssertionError("an unknown breed or section must not reach LM Studio")

    monkeypatch.setattr(knowledge, "_embed", no_embedding)
    assert knowledge.search_knowledge("nutrition", breed=breed, section=section).chunks == []


def test_blank_query_returns_no_chunks(small_db):
    assert knowledge.search_knowledge("   ").chunks == []


def test_query_is_embedded_with_the_query_prefix(monkeypatch):
    sent = []
    fake = SimpleNamespace(embeddings=SimpleNamespace(create=lambda model, input: sent.append((model, input)) or
                                                      SimpleNamespace(data=[SimpleNamespace(embedding=[0.5])])))
    monkeypatch.setattr(knowledge, "_embedding_client", lambda: fake)
    assert knowledge._embed("how much exercise") == [0.5]
    assert sent == [(config.EMBED_MODEL_ID, "search_query: how much exercise")]


def test_chunks_are_embedded_at_ingestion_with_the_document_prefix():
    """#9 asks to verify the search_document: prefix; it is added by scripts/ingest.py before embedding."""
    from scripts.ingest import Embedder

    sent = []
    embedder = Embedder.__new__(Embedder)
    embedder.client = SimpleNamespace(embeddings=SimpleNamespace(create=lambda model, input: sent.append(input) or
                                      SimpleNamespace(data=[SimpleNamespace(index=0, embedding=[0.1])])))
    embedder.embed(["Siberian Husky - Overview\ntext"])
    assert sent == [["search_document: Siberian Husky - Overview\ntext"]]


def test_clients_are_created_once():
    pytest.importorskip("openai")
    assert knowledge._embedding_client() is knowledge._embedding_client()


def test_missing_database_raises_a_clear_error(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CHROMA_KNOWLEDGE_DIR", tmp_path / "nothing_here")
    knowledge._chroma_client.cache_clear()
    knowledge._collection.cache_clear()
    try:
        with pytest.raises(RuntimeError, match="ingest.py"):
            knowledge.search_knowledge("exercise", breed="Siberian Husky")
        assert not (tmp_path / "nothing_here").exists()  # a failed search must not create an empty database
    finally:
        knowledge._chroma_client.cache_clear()
        knowledge._collection.cache_clear()


def test_lm_studio_down_raises_a_clear_error(small_db, monkeypatch):
    monkeypatch.undo()  # keep small_db's collection but use the real _embed against a dead port
    monkeypatch.setattr(knowledge, "_collection", lambda name: small_db)
    monkeypatch.setattr(config, "LM_STUDIO_BASE_URL", "http://127.0.0.1:9/v1")
    knowledge._embedding_client.cache_clear()
    try:
        with pytest.raises(RuntimeError, match="LM Studio"):
            knowledge.search_knowledge("exercise")
    finally:
        knowledge._embedding_client.cache_clear()


# --- Live: the real collection and LM Studio -----------------------------------------------------------------
@pytest.fixture(scope="module")
def live():
    knowledge._chroma_client.cache_clear()
    knowledge._collection.cache_clear()
    try:
        if knowledge._collection(knowledge.KNOWLEDGE_COLLECTION).count() == 0:
            pytest.skip("the knowledge collection is empty: unzip the data from Part A or run scripts/ingest.py --all")
    except RuntimeError as error:
        pytest.skip(str(error))
    try:
        knowledge._embed("connection check")
    except RuntimeError as error:
        pytest.skip(f"LM Studio is not available: {error}")


def test_live_husky_exercise_question_returns_husky_chunks(live):
    out = knowledge.search_knowledge(query="how much exercise", breed="Siberian Husky")
    assert len(out.chunks) >= 1
    assert all(chunk.breed == "Siberian Husky" for chunk in out.chunks)


def test_live_breed_and_section_filter_together(live):
    out = knowledge.search_knowledge(query="what is the breed like", breed="Siberian Husky", section="Overview")
    assert out.chunks and {(c.breed, c.section) for c in out.chunks} == {("Siberian Husky", "Overview")}


def test_live_unknown_breed_returns_no_chunks_without_raising(live):
    assert knowledge.search_knowledge(query="nutrition", breed="NonExistentBreed999").chunks == []


def test_live_result_matches_the_contract_and_the_server_tool(live):
    from mcp_server import server

    out = knowledge.search_knowledge(query="grooming needs", breed="Poodle (Toy)")
    assert out.chunks and len(out.chunks) <= config.SEARCH_TOP_K
    SearchKnowledgeOutput.model_validate(out.model_dump())
    via_server = server.search_knowledge(query="grooming needs", breed="Poodle (Toy)")
    assert [c.chunk_id for c in via_server.chunks] == [c.chunk_id for c in out.chunks]


# --- scripts/eval_search_breed_filter.py ---------------------------------------------------------------------
def test_eval_script_takes_the_breed_from_the_expected_chunk_id():
    from scripts.eval_search_breed_filter import breed_of
    from shared.breeds import load_breeds_map, slugify

    by_slug = {slugify(display): display for display in load_breeds_map()}
    assert breed_of("kb:siberian-husky:exercise:0", by_slug) == "Siberian Husky"
    assert breed_of("kb:german-shepherd-dog:history:6", by_slug) == "German Shepherd Dog"
    assert len(by_slug) == len(load_breeds_map())  # no two breeds share a slug
    with pytest.raises(SystemExit):
        breed_of("kb:no-such-breed:overview:0", by_slug)
