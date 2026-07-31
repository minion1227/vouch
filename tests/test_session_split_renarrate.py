"""Re-narration of an already-filed mechanical session summary.

`_try_renarrate` is the second-chance path: a session that was rolled up
mechanically by `vouch-capture` gets narrated into topical pages later, and the
mechanical proposal is rejected as superseded. Every branch here decides
whether a pending proposal survives, so an untested failure mode either strands
the mechanical rollup or drops it without a replacement.

Also covers `build_session_rows`, which is what `kb.list_sessions` and
`vouch session list` render.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from vouch import capture, session_split
from vouch import compile as compile_mod
from vouch.llm_draft import LLMDraftError
from vouch.models import Page, PageStatus, Proposal, ProposalKind, ProposalStatus
from vouch.session_split import SPLIT_ACTOR, SplitConfig, load_split_config
from vouch.storage import KBStore


@pytest.fixture
def store(tmp_path: Path) -> KBStore:
    return KBStore.init(tmp_path)


def _mechanical(
    store: KBStore,
    *,
    session_id: str = "s1",
    proposal_id: str = "pr-mech",
    proposed_by: str = capture.CAPTURE_ACTOR,
    page_type: str = capture.CAPTURE_PAGE_TYPE,
    kind: ProposalKind = ProposalKind.PAGE,
    title: str = "session s1",
    body: str = "## did a thing\n\nobserved something",
    tags: list[str] | None = None,
) -> Proposal:
    return store.put_proposal(
        Proposal(
            id=proposal_id,
            kind=kind,
            proposed_by=proposed_by,
            session_id=session_id,
            status=ProposalStatus.PENDING,
            payload={
                "type": page_type,
                "title": title,
                "body": body,
                "tags": tags or [],
            },
        )
    )


# --- _eligible_mechanical_proposal ---------------------------------------


def test_eligible_finds_the_mechanical_rollup(store: KBStore) -> None:
    _mechanical(store)
    found = session_split._eligible_mechanical_proposal(store, "s1")
    assert found is not None
    assert found.id == "pr-mech"


def test_eligible_ignores_a_non_page_proposal(store: KBStore) -> None:
    src = store.put_source(b"e")
    store.put_proposal(
        Proposal(
            id="pr-claim",
            kind=ProposalKind.CLAIM,
            proposed_by=capture.CAPTURE_ACTOR,
            session_id="s1",
            status=ProposalStatus.PENDING,
            payload={"text": "x", "evidence": [src.id]},
        )
    )
    assert session_split._eligible_mechanical_proposal(store, "s1") is None


def test_eligible_ignores_a_page_of_another_type(store: KBStore) -> None:
    _mechanical(store, page_type="concept")
    assert session_split._eligible_mechanical_proposal(store, "s1") is None


def test_eligible_ignores_another_session(store: KBStore) -> None:
    _mechanical(store, session_id="other")
    assert session_split._eligible_mechanical_proposal(store, "s1") is None


def test_eligible_ignores_an_already_narrated_proposal(store: KBStore) -> None:
    # a session-split proposal is already narrated -- re-narrating would loop
    _mechanical(store, proposed_by=SPLIT_ACTOR)
    assert session_split._eligible_mechanical_proposal(store, "s1") is None


def test_eligible_returns_none_on_an_empty_queue(store: KBStore) -> None:
    assert session_split._eligible_mechanical_proposal(store, "s1") is None


# --- build_renarrate_prompt ----------------------------------------------


def test_renarrate_prompt_includes_the_record_body(store: KBStore) -> None:
    prompt = session_split.build_renarrate_prompt(
        store, "the session body", title="session s1", max_pages=3
    )
    assert "the session body" in prompt
    assert "SESSION RECORD TITLE: session s1" in prompt


def test_renarrate_prompt_omits_an_empty_title(store: KBStore) -> None:
    prompt = session_split.build_renarrate_prompt(
        store, "body", title="", max_pages=3
    )
    assert "SESSION RECORD TITLE" not in prompt


def test_renarrate_prompt_says_none_when_no_topics_are_taken(
    store: KBStore,
) -> None:
    prompt = session_split.build_renarrate_prompt(
        store, "body", title="t", max_pages=3
    )
    assert "- (none)" in prompt


def test_renarrate_prompt_lists_durable_and_pending_topics(
    store: KBStore,
) -> None:
    store.put_page(Page(id="p1", title="the review gate"))
    store.put_proposal(
        Proposal(
            id="pr-pending-page",
            kind=ProposalKind.PAGE,
            proposed_by="agent",
            status=ProposalStatus.PENDING,
            payload={"type": "concept", "title": "retrieval backends", "body": "x"},
        )
    )
    prompt = session_split.build_renarrate_prompt(
        store, "body", title="t", max_pages=3
    )
    assert "- the review gate" in prompt
    pending = compile_mod._pending_page_names(store)
    if "retrieval backends" in pending:
        assert "- retrieval backends [pending]" in prompt


def test_renarrate_prompt_and_file_drafts_ignore_archived_pages(
    store: KBStore,
) -> None:
    """Archived pages must not block TAKEN TOPICS or draft collision (#712)."""
    store.put_page(Page(
        id="archived-session-topic",
        title="retired session topic XYZ",
        status=PageStatus.ARCHIVED,
    ))
    store.put_page(Page(
        id="live-topic",
        title="landed the review gate",
        status=PageStatus.ACTIVE,
    ))
    prompt = session_split.build_renarrate_prompt(
        store, "body", title="t", max_pages=3,
    )
    assert "retired session topic XYZ" not in prompt
    assert "landed the review gate" in prompt

    ids, dropped = session_split._file_drafts(
        store,
        "s-arch",
        [{"title": "retired session topic XYZ", "body": "rewrote the retired topic"}],
        max_pages=3,
    )
    assert len(ids) == 1
    assert dropped == []


# --- _try_renarrate ------------------------------------------------------


def test_renarrate_returns_none_without_an_eligible_proposal(
    store: KBStore,
) -> None:
    assert session_split._try_renarrate(store, "s1", split_cfg=SplitConfig()) is None


def test_renarrate_skips_when_no_llm_is_configured(store: KBStore) -> None:
    prop = _mechanical(store)
    out = session_split._try_renarrate(
        store, "s1", split_cfg=SplitConfig(llm_cmd=None)
    )
    assert out is not None
    assert out["skipped"] == "not-configured"
    assert out["proposal_id"] == prop.id
    # the mechanical rollup must survive an unconfigured re-narration
    assert store.get_proposal(prop.id).status == ProposalStatus.PENDING


def test_renarrate_skips_and_keeps_the_rollup_when_the_llm_fails(
    store: KBStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    prop = _mechanical(store)

    def _boom(*_a: Any, **_k: Any) -> str:
        raise LLMDraftError("llm exited 1")

    monkeypatch.setattr(session_split.llm_draft, "run_llm", _boom)
    out = session_split._try_renarrate(
        store, "s1", split_cfg=SplitConfig(llm_cmd="false")
    )
    assert out is not None
    assert out["skipped"] == "llm-failed"
    assert store.get_proposal(prop.id).status == ProposalStatus.PENDING


def test_renarrate_skips_when_the_llm_yields_no_valid_drafts(
    store: KBStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    prop = _mechanical(store)
    monkeypatch.setattr(session_split.llm_draft, "run_llm", lambda *_a, **_k: "[]")
    monkeypatch.setattr(session_split.llm_draft, "parse_drafts", lambda *_a, **_k: [])
    out = session_split._try_renarrate(
        store, "s1", split_cfg=SplitConfig(llm_cmd="true")
    )
    assert out is not None
    assert out["skipped"] == "llm-failed"
    assert store.get_proposal(prop.id).status == ProposalStatus.PENDING


def test_renarrate_files_pages_and_supersedes_the_rollup(
    store: KBStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    prop = _mechanical(store)
    monkeypatch.setattr(session_split.llm_draft, "run_llm", lambda *_a, **_k: "x")
    monkeypatch.setattr(
        session_split.llm_draft,
        "parse_drafts",
        lambda *_a, **_k: [
            {"title": "the coverage grind", "body": "narrated prose"},
        ],
    )
    out = session_split._try_renarrate(
        store, "s1", split_cfg=SplitConfig(llm_cmd="true", max_pages=3)
    )
    assert out is not None
    assert out["mode"] == "renarrated"
    assert out["summarized"] is True
    assert out["superseded"] == prop.id
    assert out["summary_proposal_ids"]
    # the mechanical rollup is rejected, not left as a duplicate
    assert store.get_proposal(prop.id).status == ProposalStatus.REJECTED


def test_renarrate_falls_back_to_the_compile_llm_cmd(
    store: KBStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    _mechanical(store)
    store.config_path.write_text(
        "compile:\n  llm_cmd: 'true'\n", encoding="utf-8"
    )
    monkeypatch.setattr(session_split.llm_draft, "run_llm", lambda *_a, **_k: "x")
    monkeypatch.setattr(
        session_split.llm_draft,
        "parse_drafts",
        lambda *_a, **_k: [{"title": "a topic", "body": "prose"}],
    )
    out = session_split._try_renarrate(
        store, "s1", split_cfg=SplitConfig(llm_cmd=None)
    )
    assert out is not None
    assert out["mode"] == "renarrated"


# --- _skip ---------------------------------------------------------------


def test_skip_envelope_shape() -> None:
    out = session_split._skip("s1", "not-configured", proposal_id="pr-1")
    assert out["mode"] == "skipped"
    assert out["skipped"] == "not-configured"
    assert out["summarized"] is False
    assert out["captured"] == 0
    assert out["summary_proposal_ids"] == []
    assert out["proposal_id"] == "pr-1"


# --- build_session_rows --------------------------------------------------


def test_session_rows_empty_on_a_fresh_kb(store: KBStore) -> None:
    assert session_split.build_session_rows(store) == []


def test_session_rows_lists_a_mechanical_rollup_as_unsummarized(
    store: KBStore,
) -> None:
    _mechanical(store)
    rows = session_split.build_session_rows(store)
    assert len(rows) == 1
    assert rows[0]["stage"] == "pending"
    assert rows[0]["summarized"] is False
    assert rows[0]["proposal_id"] == "pr-mech"


def test_session_rows_marks_a_split_proposal_summarized(store: KBStore) -> None:
    _mechanical(store, proposed_by=SPLIT_ACTOR)
    assert session_split.build_session_rows(store)[0]["summarized"] is True


def test_session_rows_marks_a_split_tagged_proposal_summarized(
    store: KBStore,
) -> None:
    _mechanical(store, tags=["split"])
    assert session_split.build_session_rows(store)[0]["summarized"] is True


def test_session_rows_lists_an_open_buffer(store: KBStore) -> None:
    caps = capture.captures_dir(store)
    caps.mkdir(parents=True, exist_ok=True)
    (caps / "s-open.jsonl").write_text(
        json.dumps({"ts": 1000.0, "text": "did a thing"}) + "\n", encoding="utf-8"
    )
    rows = session_split.build_session_rows(store)
    buffers = [r for r in rows if r["stage"] == "buffer"]
    assert len(buffers) == 1
    assert buffers[0]["session_id"] == "s-open"
    assert buffers[0]["summarized"] is False
    assert buffers[0]["observations"] == 1
    assert buffers[0]["last_activity"] is not None


def test_session_rows_buffer_without_timestamps_has_no_last_activity(
    store: KBStore,
) -> None:
    caps = capture.captures_dir(store)
    caps.mkdir(parents=True, exist_ok=True)
    (caps / "s-nots.jsonl").write_text(
        json.dumps({"text": "no ts here"}) + "\n", encoding="utf-8"
    )
    rows = [r for r in session_split.build_session_rows(store) if r["stage"] == "buffer"]
    assert rows[0]["last_activity"] is None


def test_session_rows_does_not_double_list_a_filed_session(
    store: KBStore,
) -> None:
    _mechanical(store, session_id="s1")
    caps = capture.captures_dir(store)
    caps.mkdir(parents=True, exist_ok=True)
    (caps / "s1.jsonl").write_text(
        json.dumps({"ts": 1000.0, "text": "leftover buffer"}) + "\n", encoding="utf-8"
    )
    rows = session_split.build_session_rows(store)
    assert [r["stage"] for r in rows] == ["pending"]


def test_session_rows_sorts_newest_activity_first(store: KBStore) -> None:
    caps = capture.captures_dir(store)
    caps.mkdir(parents=True, exist_ok=True)
    (caps / "older.jsonl").write_text(
        json.dumps({"ts": 1000.0, "text": "a"}) + "\n", encoding="utf-8"
    )
    (caps / "newer.jsonl").write_text(
        json.dumps({"ts": 9000.0, "text": "b"}) + "\n", encoding="utf-8"
    )
    rows = session_split.build_session_rows(store)
    assert [r["session_id"] for r in rows] == ["newer", "older"]


# --- load_split_config edge cases ----------------------------------------


def test_split_config_unreadable_file_falls_back(store: KBStore) -> None:
    store.config_path.unlink()
    assert load_split_config(store) == SplitConfig()


def test_split_config_scalar_document_falls_back(store: KBStore) -> None:
    store.config_path.write_text("just-a-string\n", encoding="utf-8")
    assert load_split_config(store) == SplitConfig()


def test_split_config_without_a_capture_block_falls_back(store: KBStore) -> None:
    store.config_path.write_text("review:\n  gate: true\n", encoding="utf-8")
    assert load_split_config(store) == SplitConfig()


def test_split_config_reads_every_field(store: KBStore) -> None:
    store.config_path.write_text(
        "capture:\n"
        "  split:\n"
        "    enabled: false\n"
        "    llm_cmd: 'true'\n"
        "    threshold_observations: 7\n"
        "    max_pages: 2\n"
        "    timeout_seconds: 1.5\n"
        "    max_input_chars: 900\n",
        encoding="utf-8",
    )
    cfg = load_split_config(store)
    assert cfg.enabled is False
    assert cfg.llm_cmd == "true"
    assert cfg.threshold_observations == 7
    assert cfg.max_pages == 2
    assert cfg.timeout_seconds == 1.5
    assert cfg.max_input_chars == 900


def test_split_config_empty_llm_cmd_becomes_none(store: KBStore) -> None:
    store.config_path.write_text(
        "capture:\n  split:\n    llm_cmd: ''\n", encoding="utf-8"
    )
    assert load_split_config(store).llm_cmd is None
