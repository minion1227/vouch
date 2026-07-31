"""Operator profile page — issue #614."""

from __future__ import annotations

import json
import shlex
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from vouch import audit, proposals
from vouch import compile as compile_mod
from vouch.cli import cli
from vouch.compile import (
    DEFAULT_PROFILE_TAGS,
    PROFILE_PAGE_TITLE,
    CompileConfig,
    CompileError,
    build_profile_prompt,
    compile_profile,
    select_profile_claims,
)
from vouch.storage import KBStore


@pytest.fixture
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> KBStore:
    kb = KBStore.init(tmp_path)
    monkeypatch.chdir(kb.root)
    return kb


def _claim(store: KBStore, text: str, tags: list[str]) -> str:
    src = store.put_source(text.encode("utf-8") + b" source bytes")
    pr = proposals.propose_claim(
        store, text=text, evidence=[src.id], proposed_by="agent", tags=tags
    )
    return proposals.approve(store, pr.id, approved_by="reviewer").id


def _llm(drafts: list[dict]) -> str:
    """An llm_cmd that ignores stdin and prints canned drafts."""
    return "python3 -c " + shlex.quote(
        f"import sys; sys.stdout.write({json.dumps(drafts)!r})"
    )


def _draft(claim_ids: list[str], *, title: str = PROFILE_PAGE_TITLE) -> dict:
    body = "## Preferences\n" + "\n".join(
        f"Behavioural statement number {i} [claim: {cid}]."
        for i, cid in enumerate(claim_ids)
    )
    return {
        "title": title, "type": "concept", "summary": "how the operator works",
        "tags": ["operator", "profile", "preferences"],
        "body": body, "claims": list(claim_ids),
    }


def _pending_pages(store: KBStore) -> list:
    """Pending PAGE proposals only — the fixture's claim proposals are decided."""
    from vouch.models import ProposalKind, ProposalStatus

    return [
        pr for pr in store.list_proposals()
        if pr.kind is ProposalKind.PAGE and pr.status is ProposalStatus.PENDING
    ]


# --- selection is opt-in, never inferred ---------------------------------


def test_only_tagged_claims_are_selected(store: KBStore) -> None:
    """The restraint is the feature: a tag is something a human approved."""
    pref = _claim(store, "i prefer tabs over spaces in go", ["preference"])
    conv = _claim(store, "we always squash-merge into main", ["convention"])
    noise = _claim(store, "kubernetes ingress uses nginx", ["infra"])

    selected = [c.id for c in select_profile_claims(store)]
    assert pref in selected
    assert conv in selected
    assert noise not in selected


def test_first_person_prose_alone_does_not_qualify(store: KBStore) -> None:
    """Guessing 'this sentence is about you' is the inference #614 rejects."""
    untagged = _claim(store, "i always review my own diffs before pushing", [])
    assert untagged not in [c.id for c in select_profile_claims(store)]


@pytest.mark.parametrize("tag", DEFAULT_PROFILE_TAGS)
def test_every_default_tag_selects(store: KBStore, tag: str) -> None:
    cid = _claim(store, f"a claim carrying the {tag} tag", [tag])
    assert [c.id for c in select_profile_claims(store)] == [cid]


def test_tag_matching_is_case_insensitive(store: KBStore) -> None:
    cid = _claim(store, "i prefer tabs over spaces", ["Preference"])
    assert [c.id for c in select_profile_claims(store)] == [cid]


def test_retracted_claims_are_never_selected(store: KBStore) -> None:
    from vouch import lifecycle

    cid = _claim(store, "i preferred spaces once", ["preference"])
    lifecycle.archive(store, claim_id=cid, actor="reviewer")
    assert select_profile_claims(store) == []


def test_the_profile_entity_also_selects(store: KBStore) -> None:
    from vouch.models import Entity, EntityType

    store.put_entity(Entity(id="the-operator", name="operator",
                            type=EntityType.PERSON))
    src = store.put_source(b"a claim naming the operator entity")
    pr = proposals.propose_claim(
        store, text="the operator signs off releases", evidence=[src.id],
        proposed_by="agent", entities=["the-operator"],
    )
    cid = proposals.approve(store, pr.id, approved_by="reviewer").id

    cfg = CompileConfig(profile_entity="the-operator")
    assert [c.id for c in select_profile_claims(store, config=cfg)] == [cid]


def test_configured_tags_override_the_defaults(store: KBStore) -> None:
    store.config_path.write_text(
        yaml.safe_dump({"compile": {"profile_tags": ["habit"]}}), encoding="utf-8"
    )
    habit = _claim(store, "i batch reviews on fridays", ["habit"])
    _claim(store, "i prefer tabs", ["preference"])

    assert [c.id for c in select_profile_claims(store)] == [habit]


@pytest.mark.parametrize("raw", ["not-a-list", [], [" ", ""]])
def test_malformed_profile_tags_fall_back_to_defaults(
    store: KBStore, raw: object
) -> None:
    store.config_path.write_text(
        yaml.safe_dump({"compile": {"profile_tags": raw}}), encoding="utf-8"
    )
    assert compile_mod.load_config(store).profile_tags == DEFAULT_PROFILE_TAGS


# --- the prompt refuses to invite inference ------------------------------


def test_prompt_bans_psychometric_inference(store: KBStore) -> None:
    """The whole point: the honest version is a compiled page, not a profile."""
    _claim(store, "i prefer tabs over spaces", ["preference"])
    prompt = build_profile_prompt(store, select_profile_claims(store))

    lowered = prompt.lower()
    assert "psychometric" in lowered
    for banned in ("big five", "mbti", "disc"):
        assert banned in lowered
    assert "do not infer personality" in lowered


def test_prompt_shows_only_the_selected_claims(store: KBStore) -> None:
    _claim(store, "i prefer tabs over spaces", ["preference"])
    noise = _claim(store, "kubernetes ingress uses nginx", ["infra"])
    prompt = build_profile_prompt(store, select_profile_claims(store))
    assert noise not in prompt


def test_prompt_requires_a_citation_on_every_sentence(store: KBStore) -> None:
    _claim(store, "i prefer tabs over spaces", ["preference"])
    prompt = build_profile_prompt(store, select_profile_claims(store))
    assert "[claim: <claim-id>]" in prompt
    assert "Uncited prose is dropped" in prompt


def test_no_profile_material_is_a_clean_error(store: KBStore) -> None:
    _claim(store, "kubernetes ingress uses nginx", ["infra"])
    with pytest.raises(CompileError, match="nothing to profile"):
        build_profile_prompt(store, select_profile_claims(store))


# --- the compile pass -----------------------------------------------------


def test_profile_lands_as_a_pending_proposal(store: KBStore) -> None:
    """The review step *is* the feature."""
    pref = _claim(store, "i prefer tabs over spaces", ["preference"])
    report = compile_profile(store, llm_cmd=_llm([_draft([pref])]))

    assert len(report.proposed) == 1
    assert report.dropped == []
    assert len(_pending_pages(store)) == 1
    # nothing durable was written
    assert store.list_pages() == []


def test_refresh_reproposes_rather_than_rewriting(store: KBStore) -> None:
    """History of what the system believed about you must stay auditable."""
    pref = _claim(store, "i prefer tabs over spaces", ["preference"])
    llm = _llm([_draft([pref])])

    first = compile_profile(store, llm_cmd=llm)
    proposals.approve(store, first.proposed[0]["proposal_id"], approved_by="rev")
    second = compile_profile(store, llm_cmd=llm)

    assert second.proposed, second.dropped
    assert second.proposed[0]["proposal_id"] != first.proposed[0]["proposal_id"]
    # the approved page is untouched; the new belief is a fresh proposal
    assert len(store.list_pages()) == 1


def test_a_draft_citing_outside_the_profile_set_is_dropped(
    store: KBStore
) -> None:
    """A model must not smuggle an unselected claim into a page about a person."""
    pref = _claim(store, "i prefer tabs over spaces", ["preference"])
    noise = _claim(store, "kubernetes ingress uses nginx", ["infra"])

    draft = _draft([pref])
    draft["claims"] = [pref, noise]
    report = compile_profile(store, llm_cmd=_llm([draft]))

    assert report.proposed == []
    assert "outside the profile set" in report.dropped[0]["reason"]
    assert noise in report.dropped[0]["reason"]


def test_an_uncited_draft_is_dropped(store: KBStore) -> None:
    pref = _claim(store, "i prefer tabs over spaces", ["preference"])
    draft = _draft([pref])
    draft["body"] = (
        "## Preferences\nThey are a careful and methodical engineer.\n"
        "They tend to avoid risk.\nThey prefer detail over speed.\n"
    )
    report = compile_profile(store, llm_cmd=_llm([draft]))

    assert report.proposed == []
    assert report.dropped


def test_dry_run_files_nothing(store: KBStore) -> None:
    pref = _claim(store, "i prefer tabs over spaces", ["preference"])
    report = compile_profile(store, llm_cmd=_llm([_draft([pref])]), dry_run=True)

    assert report.proposed[0]["proposal_id"] == "(dry-run)"
    assert _pending_pages(store) == []


def test_only_one_page_is_drafted(store: KBStore) -> None:
    """A profile is one page; extra drafts are ignored, not filed."""
    pref = _claim(store, "i prefer tabs over spaces", ["preference"])
    drafts = [_draft([pref]), _draft([pref], title="a second page")]
    report = compile_profile(store, llm_cmd=_llm(drafts))

    assert len(report.proposed) == 1
    assert len(_pending_pages(store)) == 1


def test_missing_llm_cmd_is_a_clean_error(store: KBStore) -> None:
    _claim(store, "i prefer tabs over spaces", ["preference"])
    with pytest.raises(CompileError, match="llm_cmd is not configured"):
        compile_profile(store)


def test_the_pass_is_audited(store: KBStore) -> None:
    pref = _claim(store, "i prefer tabs over spaces", ["preference"])
    compile_profile(store, llm_cmd=_llm([_draft([pref])]), triggered_by="human")

    ev = next(
        e for e in audit.read_events(store.kb_dir) if e.event == "compile.profile"
    )
    assert ev.actor == "human"
    assert ev.data["claims"] == 1
    assert ev.data["proposed"] == 1


# --- cli ------------------------------------------------------------------


def test_cli_profile_flag_compiles_the_profile(store: KBStore) -> None:
    pref = _claim(store, "i prefer tabs over spaces", ["preference"])
    res = CliRunner().invoke(
        cli, ["compile", "--profile", "--llm-cmd", _llm([_draft([pref])])]
    )
    assert res.exit_code == 0, res.output
    assert PROFILE_PAGE_TITLE in res.output
    assert len(_pending_pages(store)) == 1


def test_cli_profile_dry_run(store: KBStore) -> None:
    pref = _claim(store, "i prefer tabs over spaces", ["preference"])
    res = CliRunner().invoke(cli, [
        "compile", "--profile", "--dry-run", "--llm-cmd", _llm([_draft([pref])]),
    ])
    assert res.exit_code == 0, res.output
    assert "would propose" in res.output
    assert _pending_pages(store) == []


def test_cli_profile_json(store: KBStore) -> None:
    pref = _claim(store, "i prefer tabs over spaces", ["preference"])
    res = CliRunner().invoke(cli, [
        "compile", "--profile", "--json", "--llm-cmd", _llm([_draft([pref])]),
    ])
    assert res.exit_code == 0, res.output
    payload = json.loads(res.output)
    assert payload["proposed"][0]["title"] == PROFILE_PAGE_TITLE


def test_cli_profile_with_no_material_is_a_clean_error(store: KBStore) -> None:
    _claim(store, "kubernetes ingress uses nginx", ["infra"])
    res = CliRunner().invoke(
        cli, ["compile", "--profile", "--llm-cmd", "true"]
    )
    assert res.exit_code != 0
    assert "nothing to profile" in res.output
    assert "Traceback" not in res.output


def test_cli_without_profile_still_compiles_topic_pages(store: KBStore) -> None:
    """The default path must be untouched."""
    _claim(store, "kubernetes ingress uses nginx", ["infra"])
    res = CliRunner().invoke(cli, ["compile", "--dry-run", "--llm-cmd", _llm([])])
    assert res.exit_code == 0, res.output
    assert "would propose 0" in res.output


def test_a_draft_the_gate_refuses_is_reported_not_raised(
    store: KBStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A refusal from the gate lands in `dropped`, it never sinks the pass.

    Forced, because the profile pass already rejects out-of-set citations
    before it files — so the remaining ProposalError cases are ones only
    propose_page itself can decide (a stale ref, a model invariant).
    """
    pref = _claim(store, "i prefer tabs over spaces", ["preference"])

    def refuse(*args: object, **kwargs: object):
        raise proposals.ProposalError("unknown claim id: gone")

    monkeypatch.setattr(compile_mod, "propose_page", refuse)
    report = compile_profile(store, llm_cmd=_llm([_draft([pref])]))

    assert report.proposed == []
    assert report.dropped == [
        {"title": PROFILE_PAGE_TITLE, "reason": "unknown claim id: gone"}
    ]
    assert _pending_pages(store) == []
