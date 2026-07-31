"""Correction capture — propose-only, bounded, deduped (#430).

The load-bearing invariant is in the module docstring: a detected correction
becomes a *proposal*, never a write. Everything else here is the guards that
keep an ambient heuristic from becoming a reviewer's problem.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from vouch import correction
from vouch.capabilities import capabilities
from vouch.jsonl_server import HANDLERS
from vouch.models import ProposalKind, ProposalStatus
from vouch.storage import KBStore

CORRECTION = "no, we deploy from main not release"


@pytest.fixture
def store(tmp_path: Path) -> KBStore:
    s = KBStore.init(tmp_path)
    s.config_path.write_text(
        "review:\n  approver_role: trusted-agent\n", encoding="utf-8",
    )
    return s


# --- the invariant ---------------------------------------------------------


def test_a_correction_lands_as_a_pending_proposal(store: KBStore) -> None:
    report = correction.capture(store, prompt=CORRECTION, session_id="s1")

    assert report["captured"] is True
    proposal = store.get_proposal(report["proposal_id"])
    assert proposal.status is ProposalStatus.PENDING
    assert proposal.kind is ProposalKind.CLAIM
    assert proposal.proposed_by == correction.CORRECTION_ACTOR
    assert proposal.rationale == correction.CORRECTION_RATIONALE
    assert proposal.session_id == "s1"
    # nothing durable was written
    assert store.list_claims() == []


def test_the_origin_is_visible_in_the_queue(store: KBStore) -> None:
    report = correction.capture(store, prompt=CORRECTION)
    proposal = store.get_proposal(report["proposal_id"])
    assert correction.CORRECTION_TAG in proposal.payload["tags"]
    assert proposal.proposed_by == correction.CORRECTION_ACTOR


def test_the_module_has_no_path_to_approve() -> None:
    """The whole design constraint. An import of `approve` here would be a
    write path that skips the human at the gate."""
    source = Path(correction.__file__).read_text(encoding="utf-8")
    assert "propose_quoted_claim" in source
    assert not hasattr(correction, "approve")
    assert "approve" not in {
        line.split()[-1] for line in source.splitlines()
        if line.strip().startswith(("import ", "from "))
    }


def test_the_claim_carries_a_verifiable_receipt(store: KBStore) -> None:
    """The correction is quoted verbatim out of its own source, so the gate
    can check it by string comparison instead of trusting a paraphrase."""
    from vouch import receipts

    report = correction.capture(store, prompt=CORRECTION)
    proposal = store.get_proposal(report["proposal_id"])
    evidence_id = proposal.payload["evidence"][0]
    evidence = store.get_evidence(evidence_id)
    assert evidence.source_id == report["source_id"]
    result = receipts.verify_receipt(
        evidence, store.read_source_content(evidence.source_id)
    )
    assert result.status is receipts.ReceiptStatus.VERIFIED


# --- detection -------------------------------------------------------------


@pytest.mark.parametrize(
    "prompt",
    [
        "no, we deploy from main not release",
        "Nope, the config lives in etc not var",
        "actually, the retry limit should be 5",
        "that's wrong — the worker uses redis",
        "not quite, we always run mypy first",
        "I said the timeout is 30 seconds",
    ],
)
def test_pushback_openers_are_detected(prompt: str) -> None:
    assert correction.detect(prompt) is not None


@pytest.mark.parametrize(
    "prompt",
    [
        "please add a retry to the worker",
        "there is no config file in this repo",  # "no" mid-sentence
        "no.",                                    # disagreement without content
        "nope",
        "",
        "   ",
    ],
)
def test_ordinary_prompts_do_not_trip_the_heuristic(prompt: str) -> None:
    assert correction.detect(prompt) is None


def test_the_opener_is_stripped_from_the_captured_text(store: KBStore) -> None:
    """"no, we deploy from main" is worth keeping as "we deploy from main" —
    the disagreement is context, the assertion is the knowledge."""
    assert correction.detect(CORRECTION) == "we deploy from main not release"
    report = correction.capture(store, prompt=CORRECTION)
    assert report["text"] == "we deploy from main not release"


def test_a_long_prompt_is_a_new_task_not_a_correction() -> None:
    assert correction.detect("no, " + "x y z " * 200) is None


def test_a_non_correction_reports_why_nothing_was_filed(store: KBStore) -> None:
    report = correction.capture(store, prompt="please add a retry to the worker")
    assert report == {"captured": False, "reason": "not_a_correction"}
    assert store.list_proposals(ProposalStatus.PENDING) == []


# --- guards ----------------------------------------------------------------


def test_the_per_session_cap_bounds_one_run(store: KBStore) -> None:
    store.config_path.write_text(
        "review:\n  approver_role: trusted-agent\n"
        "capture:\n  correction:\n    max_per_session: 2\n",
        encoding="utf-8",
    )
    assert correction.capture(
        store, prompt="no, we deploy from main", session_id="s1"
    )["captured"]
    assert correction.capture(
        store, prompt="actually the retry limit is five attempts", session_id="s1"
    )["captured"]

    third = correction.capture(
        store, prompt="nope, the worker queue lives in redis", session_id="s1"
    )
    assert third == {
        "captured": False, "reason": "session_cap", "cap": 2, "filed": 2,
    }
    assert len(store.list_proposals(ProposalStatus.PENDING)) == 2

    # the cap is per session — a different session starts fresh
    assert correction.capture(
        store, prompt="nope, the worker queue lives in redis", session_id="s2"
    )["captured"]


def test_the_cap_is_counted_from_the_queue_not_from_memory(store: KBStore) -> None:
    """So it survives a process restart — the guard has to hold for an
    unattended capture, which is the case it exists for."""
    store.config_path.write_text(
        "review:\n  approver_role: trusted-agent\n"
        "capture:\n  correction:\n    max_per_session: 1\n",
        encoding="utf-8",
    )
    correction.capture(store, prompt="no, we deploy from main", session_id="s1")
    fresh = KBStore(store.kb_dir.parent)
    assert correction.capture(
        store=fresh, prompt="actually the retry limit is five", session_id="s1",
    )["reason"] == "session_cap"


def test_a_repeated_correction_is_suppressed(store: KBStore) -> None:
    first = correction.capture(store, prompt=CORRECTION, session_id="s1")
    assert first["captured"] is True

    again = correction.capture(
        store, prompt="no, we deploy from main, not from release", session_id="s2"
    )
    assert again["captured"] is False
    assert again["reason"] == "duplicate"
    assert again["duplicate_of"] == first["proposal_id"]


def test_an_unrelated_correction_still_files(store: KBStore) -> None:
    correction.capture(store, prompt=CORRECTION, session_id="s1")
    other = correction.capture(
        store, prompt="actually the retry limit should be five attempts",
        session_id="s1",
    )
    assert other["captured"] is True


def test_config_can_turn_it_off(store: KBStore) -> None:
    store.config_path.write_text(
        "review:\n  approver_role: trusted-agent\n"
        "capture:\n  correction:\n    enabled: false\n",
        encoding="utf-8",
    )
    assert correction.load_config(store).enabled is False
    assert correction.capture(store, prompt=CORRECTION) == {
        "captured": False, "reason": "disabled",
    }
    assert store.list_proposals(ProposalStatus.PENDING) == []


def test_defaults_when_no_capture_block_is_configured(store: KBStore) -> None:
    cfg = correction.load_config(store)
    assert cfg.enabled is True
    assert cfg.max_per_session == correction.DEFAULT_MAX_PER_SESSION


def test_a_quoted_true_does_not_read_as_off(store: KBStore) -> None:
    store.config_path.write_text(
        "review:\n  approver_role: trusted-agent\n"
        'capture:\n  correction:\n    enabled: "true"\n',
        encoding="utf-8",
    )
    assert correction.load_config(store).enabled is True


def test_a_pasted_credential_is_masked_before_it_is_filed(store: KBStore) -> None:
    report = correction.capture(
        store,
        prompt="no, the token is AKIAIOSFODNN7EXAMPLE and it must be rotated",
    )
    if report["captured"]:
        assert "AKIAIOSFODNN7EXAMPLE" not in report["text"]


# --- hook integration ------------------------------------------------------


def test_maybe_capture_never_raises_into_the_hook(store: KBStore) -> None:
    """The UserPromptSubmit contract: a broken KB drops the correction, it
    does not break the turn."""
    assert correction.maybe_capture(store, prompt="hello there") is None
    assert correction.maybe_capture(store, prompt=CORRECTION) is not None

    broken = KBStore(store.kb_dir.parent)
    broken.kb_dir = Path("/nonexistent/vouch")  # type: ignore[misc]
    assert correction.maybe_capture(broken, prompt=CORRECTION) is None


def test_the_prompt_hook_files_a_correction(store: KBStore) -> None:
    import json

    from vouch import hooks

    hooks.build_claude_prompt_hook(
        store, json.dumps({"prompt": CORRECTION, "session_id": "hook-1"}),
    )
    pending = store.list_proposals(ProposalStatus.PENDING)
    assert [p.proposed_by for p in pending] == [correction.CORRECTION_ACTOR]


# --- registration ----------------------------------------------------------


def test_capture_correction_registered_on_every_surface() -> None:
    method = "kb.capture_correction"
    assert method in set(capabilities().methods)
    assert method in HANDLERS
    from vouch.server import mcp

    assert mcp._tool_manager.get_tool("kb_capture_correction") is not None

    from vouch.cli import cli

    assert "capture-correction" in cli.commands


# --- the surfaces, exercised rather than merely registered -----------------


def test_cli_capture_correction_files_a_proposal(
    store: KBStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    import json as _json

    from click.testing import CliRunner

    from vouch.cli import cli

    monkeypatch.chdir(store.root)
    result = CliRunner().invoke(
        cli,
        ["capture-correction", CORRECTION, "--session-id", "cli-1",
         "--context", "the agent said release"],
    )
    assert result.exit_code == 0, result.output
    report = _json.loads(result.output)
    assert report["captured"] is True
    pending = store.list_proposals(ProposalStatus.PENDING)
    assert [p.kind for p in pending] == [ProposalKind.CLAIM]


def test_mcp_capture_correction_files_a_proposal(
    store: KBStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    from vouch import server

    monkeypatch.chdir(store.root)
    report = server.kb_capture_correction(prompt=CORRECTION, session_id="mcp-1")
    assert report["captured"] is True
    assert store.list_proposals(ProposalStatus.PENDING)


# --- config coercion and the guards ----------------------------------------


def test_overlap_is_zero_when_either_side_has_no_signal() -> None:
    # all stopwords / too-short tokens on one side -> no shared signal to score
    assert correction.overlap("we do it", "we deploy from main") == 0.0
    assert correction.overlap("we deploy from main", "") == 0.0


def test_config_falls_back_on_an_unreadable_or_odd_document(
    store: KBStore
) -> None:
    store.config_path.write_text("capture: [unclosed\n", encoding="utf-8")
    assert correction.load_config(store) == correction.CorrectionConfig()
    store.config_path.write_text("just-a-string\n", encoding="utf-8")
    assert correction.load_config(store) == correction.CorrectionConfig()
    store.config_path.write_text("capture: not-a-mapping\n", encoding="utf-8")
    assert correction.load_config(store) == correction.CorrectionConfig()
    store.config_path.write_text(
        "capture:\n  correction: not-a-mapping\n", encoding="utf-8"
    )
    assert correction.load_config(store) == correction.CorrectionConfig()


def test_config_typos_coerce_to_defaults(store: KBStore) -> None:
    # A config mistake must not take down an ambient capture path.
    store.config_path.write_text(
        "capture:\n  correction:\n"
        "    max_per_session: many\n"
        "    min_chars: lots\n"
        "    dedup_threshold: highish\n",
        encoding="utf-8",
    )
    cfg = correction.load_config(store)
    assert cfg.max_per_session == correction.DEFAULT_MAX_PER_SESSION
    assert cfg.min_chars == correction.DEFAULT_MIN_CHARS
    assert cfg.dedup_threshold == correction.DEFAULT_DEDUP_THRESHOLD


def test_a_correction_that_is_only_its_opener_is_not_knowledge(
    store: KBStore
) -> None:
    # "no, actually" strips to nothing; "no, it is" strips below min_chars.
    for prompt in ("no, actually", "no, wrong"):
        report = correction.capture(store, prompt=prompt, session_id="s1", agent="a")
        assert report["captured"] is False
    assert store.list_proposals(ProposalStatus.PENDING) == []


def test_pushback_without_an_assertion_is_not_knowledge() -> None:
    # Disagreement with no claim in it — nothing here belongs in a KB.
    assert correction.detect("no, that one over there instead please") is None


def test_dedup_catches_an_already_approved_claim(store: KBStore) -> None:
    from vouch.proposals import approve, propose_claim

    src = store.put_source(b"we deploy from main not release")
    pr = propose_claim(
        store, text="we deploy from main not release", evidence=[src.id],
        proposed_by="agent-a",
    )
    approve(store, pr.id, approved_by="human-b")
    report = correction.capture(store, prompt=CORRECTION, session_id="s1")
    assert report["captured"] is False
    assert report["reason"] == "duplicate"


def test_a_short_prompt_is_ignored_before_anything_else(store: KBStore) -> None:
    assert correction.capture(store, prompt="no", session_id="s1", agent="a")[
        "captured"
    ] is False


def test_dedup_catches_a_pending_correction_from_another_session(
    store: KBStore
) -> None:
    first = correction.capture(store, prompt=CORRECTION, session_id="s1", agent="a")
    assert first["captured"] is True
    again = correction.capture(
        store, prompt="no, we deploy from main not release branch",
        session_id="s2", agent="a",
    )
    assert again["captured"] is False
    assert again["reason"] == "duplicate"
    assert len(store.list_proposals(ProposalStatus.PENDING)) == 1


def test_dedup_folds_in_the_embedding_hits_when_the_extra_is_present(
    store: KBStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The embedding path is additive on top of the lexical guard, and is
    stubbed here so the branch is exercised with or without the extra
    installed — a base install must still dedup, which is why the lexical
    pass runs first and this one only adds."""
    import sys
    import types

    module = types.ModuleType("vouch.embeddings.similarity")
    module.find_similar_on_propose = lambda store, text: [  # type: ignore[attr-defined]
        {"artifact_id": None},              # ignored: not a string
        {"artifact_id": "semantic-twin"},
    ]
    monkeypatch.setitem(sys.modules, "vouch.embeddings.similarity", module)
    report = correction.capture(
        store, prompt="no, the release train leaves on thursdays now",
        session_id="s3", agent="a",
    )
    assert report["captured"] is False
    assert report["reason"] == "duplicate"
    assert report["duplicate_of"] == "semantic-twin"


def test_dedup_still_works_on_a_base_install(
    store: KBStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without the `[embeddings]` extra the import fails and the lexical guard
    is all there is — which is the whole reason it runs first. Pinned, because
    dedup that silently stops working is how an unattended capture floods a
    review queue."""
    import sys

    monkeypatch.setitem(sys.modules, "vouch.embeddings.similarity", None)
    assert correction.capture(
        store, prompt=CORRECTION, session_id="s1"
    )["captured"] is True
    again = correction.capture(
        store, prompt="no, we deploy from main not release", session_id="s2"
    )
    assert again["reason"] == "duplicate"
