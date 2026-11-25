"""Tests for mcp_server/tools/knowledge.py (issue #9).

Two layers:
    - offline tests on a small temporary Chroma collection with made-up 2-D vectors (no LM Studio, no data);
    - live tests on the real knowledge collection and LM Studio: skipped, with the reason, when either is missing.
"""

from __future__ import annotations

from types import SimpleNamespace

import anyio
import pytest

import config
from mcp_server.tools import knowledge, stubs
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


def test_section_synonym_is_resolved(small_db):
    assert ids(knowledge.search_knowledge("anything", section="Diseases")) == ["kb:siberian-husky:health:0", "kb:beagle:health:0"]


def test_section_synonyms_point_at_known_sections():
    assert set(config.SECTION_SYNONYMS.values()) <= set(config.KNOWLEDGE_SECTIONS)


def test_breed_and_section_without_a_chunk_falls_back_to_the_breed(small_db, caplog):
    with caplog.at_level("INFO", logger=knowledge.log.name):
        out = knowledge.search_knowledge("anything", breed="Beagle", section="Overview")
    assert ids(out) == ["kb:beagle:health:0"]
    assert "retrying with the breed only" in caplog.text


def test_fallback_keeps_the_breed(small_db):
    out = knowledge.search_knowledge("anything", breed="Siberian Husky", section="Exercise")
    assert {c.breed for c in out.chunks} == {"Siberian Husky"}


def test_breed_without_any_chunk_returns_nothing(small_db):
    small_db.delete(where={"breed": "Beagle"})
    assert ids(knowledge.search_knowledge("anything", breed="Beagle", section="Overview")) == []


@pytest.mark.parametrize("breed, section, embedded, expected", [
    # unknown breed: no filter, the name goes into the query
    ("NonExistentBreed999", None, "NonExistentBreed999: nutrition",
     ["kb:siberian-husky:overview:0", "kb:siberian-husky:health:0", "kb:german-shepherd-dog:overview:0"]),
    ("Labrador", None, "Labrador: nutrition",
     ["kb:siberian-husky:overview:0", "kb:siberian-husky:health:0", "kb:german-shepherd-dog:overview:0"]),
    # unknown section: dropped
    (None, "Nonsense", "nutrition",
     ["kb:siberian-husky:overview:0", "kb:siberian-husky:health:0", "kb:german-shepherd-dog:overview:0"]),
    ("Beagle", "Nonsense", "nutrition", ["kb:beagle:health:0"]),
    # unknown breed, known section: only the section filters
    ("NonExistentBreed999", "Health", "NonExistentBreed999: nutrition",
     ["kb:siberian-husky:health:0", "kb:beagle:health:0"]),
])
def test_unknown_breed_or_section_is_not_a_filter(small_db, monkeypatch, breed, section, embedded, expected):
    sent = []
    monkeypatch.setattr(knowledge, "_embed", lambda text: sent.append(text) or [1.0, 0.0])
    assert ids(knowledge.search_knowledge("nutrition", breed=breed, section=section)) == expected
    assert sent == [embedded]


@pytest.mark.parametrize("tool", [stubs.search_knowledge, knowledge.search_knowledge], ids=["stub", "real"])
@pytest.mark.parametrize("breed, section", [(None, None), ("NonExistentBreed999", None), ("Labrador", "Diet"),
                                            ("small", "Nonsense"), ("Beagle", "Nonsense"), (None, "Nonsense")])
def test_stub_and_real_tool_return_chunks_for_the_same_inputs(small_db, tool, breed, section):
    small_db.upsert(ids=["kb:beagle:feeding:0"], documents=["Beagle - Feeding text"], embeddings=[[0.5, 0.5]],
                    metadatas=[{"chunk_id": "kb:beagle:feeding:0", "breed": "Beagle", "section": "Feeding",
                                "source": "wikipedia"}])
    assert tool("nutrition", breed=breed, section=section).chunks


@pytest.mark.parametrize("tool", [stubs.search_knowledge, knowledge.search_knowledge], ids=["stub", "real"])
@pytest.mark.parametrize("query", ["", "   "])
def test_stub_and_real_tool_return_nothing_for_a_blank_query(small_db, tool, query):
    assert tool(query, breed="NonExistentBreed999").chunks == []


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


def test_empty_embedding_response_raises_a_clear_error(monkeypatch):
    fake = SimpleNamespace(embeddings=SimpleNamespace(create=lambda model, input: SimpleNamespace(data=[])))
    monkeypatch.setattr(knowledge, "_embedding_client", lambda: fake)
    with pytest.raises(RuntimeError, match="returned no embedding"):
        knowledge._embed("exercise")


def test_wrong_vector_dimension_raises_a_clear_error(small_db, monkeypatch):
    monkeypatch.setattr(knowledge, "_embed", lambda text: [1.0, 0.0, 0.0])  # the collection holds 2-D vectors
    with pytest.raises(RuntimeError, match="dimensions"):
        knowledge.search_knowledge("exercise")


def test_other_chroma_query_errors_say_what_failed(small_db, monkeypatch):
    def broken(**kwargs):
        raise OSError("database is locked")

    monkeypatch.setattr(small_db, "query", broken)
    with pytest.raises(RuntimeError, match="database is locked"):
        knowledge.search_knowledge("exercise")


def test_only_a_missing_collection_points_to_ingest(monkeypatch):
    chromadb = pytest.importorskip("chromadb")
    import chromadb.errors

    class FakeClient:
        def __init__(self, error):
            self.error = error

        def get_collection(self, name):
            raise self.error

    knowledge._collection.cache_clear()
    try:
        monkeypatch.setattr(knowledge, "_chroma_client", lambda: FakeClient(chromadb.errors.NotFoundError("nope")))
        with pytest.raises(RuntimeError, match="ingest.py"):
            knowledge._collection("knowledge")
        knowledge._collection.cache_clear()
        monkeypatch.setattr(knowledge, "_chroma_client", lambda: FakeClient(OSError("schema version mismatch")))
        with pytest.raises(RuntimeError, match="schema version mismatch") as info:
            knowledge._collection("knowledge")
        assert "run `python scripts/ingest.py" not in str(info.value)  # a re-ingest would not fix this
    finally:
        knowledge._collection.cache_clear()


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


def test_live_unknown_breed_is_searched_without_a_filter(live):
    assert knowledge.search_knowledge(query="nutrition", breed="NonExistentBreed999").chunks
    out = knowledge.search_knowledge(query="how much should I feed", breed="Labrador")
    assert "Labrador Retriever" in {chunk.breed for chunk in out.chunks}


def test_live_result_matches_the_contract_and_the_server_tool(live, monkeypatch):
    from mcp_server import server

    out = knowledge.search_knowledge(query="grooming needs", breed="Poodle (Toy)")
    assert out.chunks and len(out.chunks) <= config.SEARCH_TOP_K
    SearchKnowledgeOutput.model_validate(out.model_dump())
    monkeypatch.setattr(config, "STUB_TOOLS", set())  # the server must route to the real tool
    via_server = anyio.run(lambda: server.search_knowledge(query="grooming needs", breed="Poodle (Toy)"))
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


def test_golden_sha256_ignores_line_endings(tmp_path):
    from scripts.eval_search_breed_filter import golden_sha256

    (tmp_path / "lf.jsonl").write_bytes(b'{"a": 1}\n{"b": 2}\n')
    (tmp_path / "crlf.jsonl").write_bytes(b'{"a": 1}\r\n{"b": 2}\r\n')
    assert golden_sha256(tmp_path / "lf.jsonl") == golden_sha256(tmp_path / "crlf.jsonl")
