"""The embedding half of `index_db`, plus `storage._embed_and_store`.

These are the vector paths behind `retrieval.backend: embedding`. They were
uncovered because the whole embeddings suite was being skipped, so the
blob round-trip, the sqlite-vec probe and its brute-force fallback, and the
write-through from `store.put_*` all ran untested.
"""

from __future__ import annotations

import hashlib
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pytest

from vouch import index_db
from vouch.embeddings import register
from vouch.embeddings.base import DEFAULT_MODEL_NAME, Embedder
from vouch.models import Claim, Entity, Page
from vouch.storage import KBStore


class _HashEmbedder(Embedder):
    name = "mock"
    version = "1"
    dim = 8

    def encode(self, text: str) -> np.ndarray:
        h = hashlib.sha256(text.encode()).digest()
        out = np.array([h[i] / 255.0 for i in range(self.dim)], dtype=np.float32)
        norm = float(np.linalg.norm(out))
        if norm > 0:
            out /= norm
        return out


@pytest.fixture
def store(tmp_path: Path) -> KBStore:
    return KBStore.init(tmp_path / "kb")


@pytest.fixture
def embedder() -> None:
    register(DEFAULT_MODEL_NAME, _HashEmbedder)


def _vec(text: str) -> np.ndarray:
    return _HashEmbedder().encode(text)


def _put(store: KBStore, kind: str, eid: str, text: str) -> None:
    with index_db.open_db(store.kb_dir) as conn:
        index_db.put_embedding(
            conn, kind=kind, id=eid, vec=_vec(text),
            content_hash=hashlib.sha256(text.encode()).hexdigest(),
            model="mock", model_version="1", dim=8,
        )


# --- blob round-trip ------------------------------------------------------


def test_vec_blob_round_trip_preserves_the_vector() -> None:
    vec = _vec("round trip me")
    restored = index_db._blob_to_vec(index_db._vec_to_blob(vec), 8)
    assert np.allclose(vec, restored)


def test_put_and_get_embedding_round_trip(store: KBStore) -> None:
    _put(store, "claim", "c1", "stored claim text")
    got = index_db.get_embedding(store.kb_dir, kind="claim", id="c1")
    assert got is not None
    vec, content_hash, model = got
    assert np.allclose(vec, _vec("stored claim text"))
    assert model == "mock"
    assert content_hash == hashlib.sha256(b"stored claim text").hexdigest()


def test_get_embedding_returns_none_when_absent(store: KBStore) -> None:
    assert index_db.get_embedding(store.kb_dir, kind="claim", id="ghost") is None


def test_put_embedding_replaces_an_existing_row(store: KBStore) -> None:
    # INSERT OR REPLACE: re-embedding the same artifact must not raise a
    # UNIQUE violation on (kind, id)
    _put(store, "claim", "c1", "first text")
    _put(store, "claim", "c1", "second text")
    got = index_db.get_embedding(store.kb_dir, kind="claim", id="c1")
    assert got is not None
    assert np.allclose(got[0], _vec("second text"))


# --- embedding meta -------------------------------------------------------


def test_embedding_meta_round_trip(store: KBStore) -> None:
    index_db.set_embedding_meta(store.kb_dir, model="mock", version="1", dim=8)
    meta = index_db.get_embedding_meta(store.kb_dir)
    assert meta["embedding_model"] == "mock"
    assert meta["embedding_model_version"] == "1"
    assert meta["embedding_dim"] == "8"


def test_embedding_meta_is_empty_on_a_fresh_index(store: KBStore) -> None:
    assert index_db.get_embedding_meta(store.kb_dir) == {}


# --- sqlite-vec probe -----------------------------------------------------


def test_load_sqlite_vec_reports_false_without_the_extension(
    store: KBStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    # the [embeddings-fast] extra is absent here, so the probe must degrade
    # rather than raise -- search_embedding relies on that to fall back
    monkeypatch.setitem(sys.modules, "sqlite_vec", None)
    with index_db.open_db(store.kb_dir) as conn:
        assert index_db._load_sqlite_vec(conn) is False


def test_load_sqlite_vec_handles_a_build_without_extension_support() -> None:
    # python built against a sqlite without loadable-extension support has no
    # enable_load_extension at all; the probe must return False, not blow up
    class _NoLoader:
        pass

    assert index_db._load_sqlite_vec(_NoLoader()) is False  # type: ignore[arg-type]


def test_load_sqlite_vec_handles_a_disabled_loader() -> None:
    class _Refuses:
        def enable_load_extension(self, _flag: bool) -> None:
            raise sqlite3.OperationalError("extension loading disabled")

    assert index_db._load_sqlite_vec(_Refuses()) is False  # type: ignore[arg-type]


# --- search_embedding (brute-force fallback) -----------------------------


def test_search_embedding_ranks_the_exact_match_first(store: KBStore) -> None:
    _put(store, "claim", "c1", "the review gate is load-bearing")
    _put(store, "claim", "c2", "something entirely unrelated")
    hits = index_db.search_embedding(
        store.kb_dir, query_vec=_vec("the review gate is load-bearing")
    )
    assert hits[0][1] == "c1"
    assert hits[0][3] == pytest.approx(1.0, abs=1e-4)


def test_search_embedding_filters_by_kind(store: KBStore) -> None:
    _put(store, "claim", "c1", "shared text")
    _put(store, "page", "p1", "shared text")
    hits = index_db.search_embedding(
        store.kb_dir, query_vec=_vec("shared text"), kinds=("page",)
    )
    assert [h[0] for h in hits] == ["page"]


def test_search_embedding_honours_min_score(store: KBStore) -> None:
    _put(store, "claim", "c1", "the matching text")
    _put(store, "claim", "c2", "a different string")
    hits = index_db.search_embedding(
        store.kb_dir, query_vec=_vec("the matching text"), min_score=0.999
    )
    assert [h[1] for h in hits] == ["c1"]


def test_search_embedding_honours_limit(store: KBStore) -> None:
    for i in range(5):
        _put(store, "claim", f"c{i}", f"text number {i}")
    hits = index_db.search_embedding(
        store.kb_dir, query_vec=_vec("text number 1"), limit=2
    )
    assert len(hits) == 2


def test_search_embedding_on_an_empty_index(store: KBStore) -> None:
    assert index_db.search_embedding(store.kb_dir, query_vec=_vec("nothing")) == []


def test_search_embedding_tolerates_a_zero_query_vector(store: KBStore) -> None:
    _put(store, "claim", "c1", "some text")
    hits = index_db.search_embedding(
        store.kb_dir, query_vec=np.zeros(8, dtype=np.float32)
    )
    assert all(h[3] == pytest.approx(0.0) for h in hits)


# --- search_embeddings (legacy json table) -------------------------------


def test_legacy_search_embeddings_ranks_by_cosine(store: KBStore) -> None:
    # NOTE: `search_embeddings` (plural) reads the legacy `embeddings` table and
    # has no callers left in src/ or tests/. Covered here so the number is
    # honest, but deleting it would be the better fix.
    with index_db.open_db(store.kb_dir) as conn:
        index_db.index_embedding(
            conn, kind="claim", id="c1", vec=_vec("target text").tolist()
        )
        index_db.index_embedding(
            conn, kind="claim", id="c2", vec=_vec("other text").tolist()
        )
    hits = index_db.search_embeddings(
        store.kb_dir, _vec("target text").tolist()
    )
    assert hits[0][1] == "c1"


def test_deindex_removes_the_legacy_embeddings_row(store: KBStore) -> None:
    # deindex()'s own docstring promises to remove "the embedding row for
    # any kind" but only cleared embedding_index, never the legacy
    # `embeddings` table search_embeddings (plural) reads — a deleted
    # artifact's vector leaked there forever.
    with index_db.open_db(store.kb_dir) as conn:
        index_db.index_embedding(
            conn, kind="claim", id="ghost", vec=_vec("gone now").tolist()
        )
        conn.commit()
        index_db.deindex(conn, kind="claim", id="ghost")
        conn.commit()
    assert index_db.search_embeddings(store.kb_dir, _vec("gone now").tolist()) == []


def test_reset_clears_the_legacy_embeddings_table(store: KBStore) -> None:
    # reset()'s own docstring warns "leaving stale rows here means semantic
    # search can return orphaned hits after a reindex" but never cleared
    # the legacy `embeddings` table itself.
    with index_db.open_db(store.kb_dir) as conn:
        index_db.index_embedding(
            conn, kind="claim", id="c1", vec=_vec("will be reset").tolist()
        )
        conn.commit()
    index_db.reset(store.kb_dir)
    assert index_db.search_embeddings(store.kb_dir, _vec("will be reset").tolist()) == []


def test_legacy_search_embeddings_rejects_an_empty_query(store: KBStore) -> None:
    assert index_db.search_embeddings(store.kb_dir, []) == []


def test_legacy_search_embeddings_rejects_a_zero_query(store: KBStore) -> None:
    assert index_db.search_embeddings(store.kb_dir, [0.0] * 8) == []


def test_legacy_search_embeddings_skips_mismatched_dims(store: KBStore) -> None:
    with index_db.open_db(store.kb_dir) as conn:
        index_db.index_embedding(conn, kind="claim", id="c1", vec=[1.0, 0.0])
    assert index_db.search_embeddings(store.kb_dir, _vec("q").tolist()) == []


def test_legacy_search_embeddings_honours_limit(store: KBStore) -> None:
    with index_db.open_db(store.kb_dir) as conn:
        for i in range(4):
            index_db.index_embedding(
                conn, kind="claim", id=f"c{i}", vec=_vec(f"t{i}").tolist()
            )
    hits = index_db.search_embeddings(
        store.kb_dir, _vec("t1").tolist(), limit=2
    )
    assert len(hits) == 2


# --- _snippet_for --------------------------------------------------------


def test_snippet_falls_back_to_the_id_when_no_file_exists(store: KBStore) -> None:
    assert index_db._snippet_for(store.kb_dir, "claim", "ghost") == "ghost"


def test_snippet_reads_the_yaml_artifact(store: KBStore) -> None:
    src = store.put_source(b"e")
    store.put_claim(Claim(id="c1", text="snippet source text", evidence=[src.id]))
    snippet = index_db._snippet_for(store.kb_dir, "claims", "c1")
    assert snippet == "c1" or "\n" not in snippet


# --- semantic search availability ----------------------------------------


def test_semantic_search_available_with_an_embedder(
    store: KBStore, embedder: None
) -> None:
    assert index_db.semantic_search_available() is True


def test_semantic_search_unavailable_without_an_embedder(store: KBStore) -> None:
    # the suite-wide registry isolation means no adapter is registered here
    assert index_db.semantic_search_available() is False


def test_search_semantic_degrades_without_an_embedder(store: KBStore) -> None:
    assert index_db.search_semantic(store.kb_dir, "anything") == []


def test_search_semantic_finds_the_match(store: KBStore, embedder: None) -> None:
    _put(store, "claim", "c1", "the review gate is load-bearing")
    hits = index_db.search_semantic(
        store.kb_dir, "the review gate is load-bearing"
    )
    assert [h[1] for h in hits] == ["c1"]


def test_search_semantic_caches_the_query_vector(
    store: KBStore, embedder: None
) -> None:
    _put(store, "claim", "c1", "cache me")
    first = index_db.search_semantic(store.kb_dir, "cache me")
    second = index_db.search_semantic(store.kb_dir, "cache me")
    assert first == second


# --- storage write-through ------------------------------------------------


def test_put_claim_writes_an_embedding(store: KBStore, embedder: None) -> None:
    src = store.put_source(b"e")
    store.put_claim(Claim(id="c1", text="embedded on write", evidence=[src.id]))
    assert index_db.get_embedding(store.kb_dir, kind="claim", id="c1") is not None


def test_put_page_writes_an_embedding(store: KBStore, embedder: None) -> None:
    store.put_page(Page(id="p1", title="a page", body="prose"))
    assert index_db.get_embedding(store.kb_dir, kind="page", id="p1") is not None


def test_put_entity_writes_an_embedding(store: KBStore, embedder: None) -> None:
    store.put_entity(Entity(id="e1", name="acme-example", type="company"))
    assert index_db.get_embedding(store.kb_dir, kind="entity", id="e1") is not None


def test_updating_a_claim_replaces_its_embedding(
    store: KBStore, embedder: None
) -> None:
    # write-through runs on update_claim too; the second write must replace the
    # embedding row rather than raise a UNIQUE violation on (kind, id)
    src = store.put_source(b"e")
    claim = store.put_claim(Claim(id="c1", text="first text", evidence=[src.id]))
    before = index_db.get_embedding(store.kb_dir, kind="claim", id="c1")
    store.update_claim(claim.model_copy(update={"text": "second text"}))
    after = index_db.get_embedding(store.kb_dir, kind="claim", id="c1")
    assert before is not None and after is not None
    assert not np.allclose(before[0], after[0])


def test_put_claim_refuses_to_overwrite(store: KBStore, embedder: None) -> None:
    src = store.put_source(b"e")
    store.put_claim(Claim(id="c1", text="first text", evidence=[src.id]))
    with pytest.raises(ValueError, match="already exists"):
        store.put_claim(Claim(id="c1", text="second text", evidence=[src.id]))


def test_put_claim_without_an_embedder_skips_embedding(store: KBStore) -> None:
    src = store.put_source(b"e")
    store.put_claim(Claim(id="c1", text="no embedder here", evidence=[src.id]))
    assert index_db.get_embedding(store.kb_dir, kind="claim", id="c1") is None


def test_embed_and_store_is_a_noop_for_empty_text(
    store: KBStore, embedder: None
) -> None:
    store._embed_and_store(kind="claim", id="c-empty", text="")
    assert index_db.get_embedding(store.kb_dir, kind="claim", id="c-empty") is None
