"""Explicit pins — issue #615."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from vouch import health, lifecycle, pins, proposals
from vouch.cli import cli
from vouch.context import build_context_pack
from vouch.models import ArtifactScope, Page, PageStatus, PageType, Visibility
from vouch.storage import KBStore


@pytest.fixture
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> KBStore:
    kb = KBStore.init(tmp_path)
    monkeypatch.chdir(kb.root)
    return kb


def _claim(store: KBStore, text: str) -> str:
    src = store.put_source(text.encode("utf-8") + b" source bytes")
    pr = proposals.propose_claim(
        store, text=text, evidence=[src.id], proposed_by="agent"
    )
    return proposals.approve(store, pr.id, approved_by="reviewer").id


def _ids(pack: dict) -> list[str]:
    return [i["id"] for i in pack["items"]]


def _write_pins_cfg(store: KBStore, **pins_cfg: object) -> None:
    store.config_path.write_text(
        yaml.safe_dump({"retrieval": {"pins": pins_cfg}}), encoding="utf-8"
    )


# --- the core behaviour ---------------------------------------------------


def test_pinned_artifact_leads_a_pack_it_would_not_have_entered(
    store: KBStore
) -> None:
    """The whole point: a pin does not have to win the query."""
    spec = _claim(store, "the spec says tokens rotate every 24 hours")
    other = _claim(store, "kubernetes ingress uses nginx")
    health.rebuild_index(store)

    before = build_context_pack(store, query="kubernetes", limit=5, max_chars=2000)
    assert spec not in _ids(before)

    pins.add_pin(store, spec, pinned_by="human")
    after = build_context_pack(store, query="kubernetes", limit=5, max_chars=2000)
    assert _ids(after)[0] == spec
    assert other in _ids(after)


def test_pinned_item_is_marked_with_the_pin_backend(store: KBStore) -> None:
    spec = _claim(store, "the spec says tokens rotate every 24 hours")
    health.rebuild_index(store)
    pins.add_pin(store, spec, pinned_by="human")

    pack = build_context_pack(store, query="anything", limit=5, max_chars=2000)
    assert pack["items"][0]["backend"] == "pin"


def test_a_pin_evicts_its_near_duplicate_from_retrieval(store: KBStore) -> None:
    """The same knowledge stored under a second id must not occupy two slots.

    `_dedupe_near_duplicates` runs before pins are injected, so it never gets
    to compare a retrieved item against a pin. Exact `(type, id)` matching
    does not catch it either, because the near-duplicate has a different id.
    """
    # 9 of 10 shared tokens -> Jaccard 0.9, over the 0.85 threshold, while the
    # slugified ids stay distinct so exact matching cannot catch it.
    pinned = _claim(store, "tokens rotate every twenty four hours per the spec")
    twin = _claim(store, "tokens rotate every twenty four hours per the spec doc")
    health.rebuild_index(store)
    pins.add_pin(store, pinned, pinned_by="human")

    ids = _ids(build_context_pack(store, query="tokens rotate",
                                  limit=5, max_chars=2000))
    assert ids[0] == pinned
    assert twin not in ids, "near-duplicate of a pin took a second slot"


def test_near_duplicate_eviction_keeps_the_pin_not_the_higher_score(
    store: KBStore,
) -> None:
    """The pin wins the collision even when retrieval scores its twin higher.

    Deliberately not fixed by moving `_dedupe_near_duplicates` below the pin
    injection: that pass keeps the highest-scored member of a cluster, and
    `pages_first` multiplies a page's score past a pin's flat 1.0 — which
    would evict the pin, the one thing pinning must never allow.
    """
    store.config_path.write_text(
        yaml.safe_dump({
            "retrieval": {
                "pins": {"enabled": True},
                "pages_first": {"enabled": True, "boost": 5.0},
            }
        }),
        encoding="utf-8",
    )
    text = "tokens rotate every twenty four hours per the spec"
    pinned = _claim(store, text)
    store.put_page(Page(
        id="p-twin", title=text, body="same knowledge, page form",
        type=PageType.CONCEPT, status=PageStatus.DRAFT,
    ))
    health.rebuild_index(store)
    pins.add_pin(store, pinned, pinned_by="human")

    ids = _ids(build_context_pack(store, query="tokens rotate",
                                  limit=5, max_chars=2000))
    assert ids[0] == pinned
    assert "p-twin" not in ids


def test_a_pinned_artifact_that_also_ranks_appears_once(store: KBStore) -> None:
    spec = _claim(store, "the spec says tokens rotate every 24 hours")
    health.rebuild_index(store)
    pins.add_pin(store, spec, pinned_by="human")

    ids = _ids(build_context_pack(store, query="tokens", limit=5, max_chars=2000))
    assert ids.count(spec) == 1


def test_pin_order_is_preserved(store: KBStore) -> None:
    first = _claim(store, "the spec says tokens rotate every 24 hours")
    second = _claim(store, "we rejected polling because it doubles cost")
    health.rebuild_index(store)
    pins.add_pin(store, first, pinned_by="human")
    pins.add_pin(store, second, pinned_by="human")

    assert _ids(build_context_pack(
        store, query="unrelated", limit=5, max_chars=2000
    ))[:2] == [first, second]


# --- a pin is not a permission -------------------------------------------


@pytest.mark.parametrize("retire", ["archive", "supersede"])
def test_a_retired_claim_stops_being_injected(store: KBStore, retire: str) -> None:
    """Lifecycle beats a pin — a pin records what to prefer, not a right."""
    spec = _claim(store, "the spec says tokens rotate every 24 hours")
    health.rebuild_index(store)
    pins.add_pin(store, spec, pinned_by="human")
    assert spec in _ids(build_context_pack(store, query="x", limit=5, max_chars=2000))

    if retire == "archive":
        lifecycle.archive(store, claim_id=spec, actor="reviewer")
    else:
        newer = _claim(store, "the spec now says tokens rotate every 12 hours")
        lifecycle.supersede(
            store, old_claim_id=spec, new_claim_id=newer, actor="reviewer"
        )

    assert spec not in _ids(build_context_pack(
        store, query="x", limit=5, max_chars=2000
    ))


def test_an_archived_page_stops_being_injected(store: KBStore) -> None:
    store.put_page(Page(id="p-live", title="design notes", body="b",
                        type=PageType.CONCEPT))
    health.rebuild_index(store)
    pins.add_pin(store, "p-live", pinned_by="human")
    assert "p-live" in _ids(build_context_pack(
        store, query="x", limit=5, max_chars=2000
    ))

    page = store.get_page("p-live")
    page.status = PageStatus.ARCHIVED
    store.update_page(page)

    assert "p-live" not in _ids(build_context_pack(
        store, query="x", limit=5, max_chars=2000
    ))


def test_a_pin_cannot_widen_viewer_scope(store: KBStore) -> None:
    """Pinning must not be a way to see what the scope filter hides."""
    src = store.put_source(b"private source bytes")
    pr = proposals.propose_claim(
        store, text="the private staging key rotates weekly",
        evidence=[src.id], proposed_by="agent",
        scope=ArtifactScope(visibility=Visibility.PRIVATE, project="other-project"),
    )
    private = proposals.approve(store, pr.id, approved_by="reviewer").id
    health.rebuild_index(store)
    pins.add_pin(store, private, pinned_by="human")

    pack = build_context_pack(
        store, query="x", limit=5, max_chars=2000, project="this-project"
    )
    assert private not in _ids(pack)


def test_pinning_an_unknown_artifact_is_refused(store: KBStore) -> None:
    with pytest.raises(pins.PinError, match="unknown artifact"):
        pins.add_pin(store, "never-existed", pinned_by="human")


def test_pinning_an_empty_id_is_refused(store: KBStore) -> None:
    with pytest.raises(pins.PinError, match="needs an artifact id"):
        pins.add_pin(store, "   ", pinned_by="human")


# --- budget share ---------------------------------------------------------


def test_pins_cannot_starve_retrieval(store: KBStore) -> None:
    """The share caps pins even when many are set."""
    pinned = [_claim(store, f"pinned working-set item number {i} " + "x" * 80)
              for i in range(6)]
    health.rebuild_index(store)
    for cid in pinned:
        pins.add_pin(store, cid, pinned_by="human")

    _write_pins_cfg(store, budget_share=0.3)
    items = pins.pinned_items(store, max_chars=1000)
    assert 0 < len(items) < len(pinned)
    assert sum(len(i.summary) for i in items) <= 1000 * 0.3 + max(
        len(i.summary) for i in items
    )


def test_at_least_one_pin_survives_a_tiny_budget(store: KBStore) -> None:
    """The first pin is never dropped — a share of ~0 still honours one."""
    spec = _claim(store, "the spec says tokens rotate every 24 hours")
    health.rebuild_index(store)
    pins.add_pin(store, spec, pinned_by="human")

    _write_pins_cfg(store, budget_share=0.01)
    assert [i.id for i in pins.pinned_items(store, max_chars=100)] == [spec]


def test_pins_can_be_disabled_in_config(store: KBStore) -> None:
    spec = _claim(store, "the spec says tokens rotate every 24 hours")
    health.rebuild_index(store)
    pins.add_pin(store, spec, pinned_by="human")

    _write_pins_cfg(store, enabled=False)
    assert pins.pinned_items(store, max_chars=2000) == []
    assert spec not in _ids(build_context_pack(
        store, query="x", limit=5, max_chars=2000
    ))


@pytest.mark.parametrize(
    ("raw", "expected"),
    [(0.5, 0.5), (2.0, 1.0), (-1.0, 0.0), ("nonsense", pins.DEFAULT_BUDGET_SHARE),
     (True, pins.DEFAULT_BUDGET_SHARE)],
)
def test_budget_share_is_read_defensively(
    store: KBStore, raw: object, expected: float
) -> None:
    _write_pins_cfg(store, budget_share=raw)
    assert pins.load_config(store).budget_share == expected


def test_missing_config_falls_back_to_defaults(store: KBStore) -> None:
    cfg = pins.load_config(store)
    assert cfg.enabled is pins.DEFAULT_ENABLED
    assert cfg.budget_share == pins.DEFAULT_BUDGET_SHARE


# --- storage, expiry, local vs shared -------------------------------------


def test_expired_pins_are_dropped_on_read_without_rewriting(store: KBStore) -> None:
    """Reading the KB must not mutate it."""
    spec = _claim(store, "the spec says tokens rotate every 24 hours")
    pins.add_pin(
        store, spec, pinned_by="human",
        expires_at=datetime.now(UTC) - timedelta(days=1),
    )
    path = store.kb_dir / pins.SHARED_FILENAME
    before = path.read_text(encoding="utf-8")

    assert pins.load_pins(store) == []
    assert path.read_text(encoding="utf-8") == before


def test_a_future_expiry_stays_live(store: KBStore) -> None:
    spec = _claim(store, "the spec says tokens rotate every 24 hours")
    pins.add_pin(
        store, spec, pinned_by="human",
        expires_at=datetime.now(UTC) + timedelta(days=7),
    )
    assert [p.artifact_id for p in pins.load_pins(store)] == [spec]


def test_local_pins_are_gitignored_and_excludable(store: KBStore) -> None:
    spec = _claim(store, "the spec says tokens rotate every 24 hours")
    pins.add_pin(store, spec, pinned_by="human", local=True)

    assert pins.LOCAL_FILENAME in (store.kb_dir / ".gitignore").read_text()
    assert [p.artifact_id for p in pins.load_pins(store)] == [spec]
    assert pins.load_pins(store, include_local=False) == []


def test_shared_wins_when_an_artifact_is_pinned_in_both_sets(
    store: KBStore
) -> None:
    spec = _claim(store, "the spec says tokens rotate every 24 hours")
    pins.add_pin(store, spec, pinned_by="human", note="shared")
    pins.add_pin(store, spec, pinned_by="human", local=True, note="local")

    live = pins.load_pins(store)
    assert len(live) == 1
    assert live[0].local is False
    assert live[0].note == "shared"


def test_repinning_replaces_rather_than_duplicates(store: KBStore) -> None:
    spec = _claim(store, "the spec says tokens rotate every 24 hours")
    pins.add_pin(store, spec, pinned_by="human", note="first")
    pins.add_pin(store, spec, pinned_by="human", note="second")

    live = pins.load_pins(store)
    assert len(live) == 1
    assert live[0].note == "second"


def test_removing_the_last_pin_removes_the_file(store: KBStore) -> None:
    spec = _claim(store, "the spec says tokens rotate every 24 hours")
    pins.add_pin(store, spec, pinned_by="human")
    assert pins.remove_pin(store, spec) is True
    assert not (store.kb_dir / pins.SHARED_FILENAME).exists()
    assert pins.remove_pin(store, spec) is False


def test_malformed_pin_rows_are_skipped_not_fatal(store: KBStore) -> None:
    spec = _claim(store, "the spec says tokens rotate every 24 hours")
    (store.kb_dir / pins.SHARED_FILENAME).write_text(
        yaml.safe_dump({"pins": [
            "not-a-mapping",
            {"kind": "claim"},                      # no id
            {"id": "x"},                            # no kind
            {"id": "y", "kind": "entity"},          # unpinnable kind
            {"id": spec, "kind": "claim", "expires_at": "not-a-date"},
        ]}),
        encoding="utf-8",
    )
    live = pins.load_pins(store)
    assert [p.artifact_id for p in live] == [spec]
    assert live[0].expires_at is None


def test_unreadable_pin_file_is_not_fatal(store: KBStore) -> None:
    (store.kb_dir / pins.SHARED_FILENAME).write_text("{{ not yaml", encoding="utf-8")
    assert pins.load_pins(store) == []


def test_a_page_pin_resolves_to_its_title(store: KBStore) -> None:
    store.put_page(Page(id="p1", title="design notes", body="body",
                        type=PageType.CONCEPT))
    pin = pins.add_pin(store, "p1", pinned_by="human")
    assert pin.kind == "page"
    assert pins.pinned_items(store, max_chars=2000)[0].summary == "design notes"


# --- cli ------------------------------------------------------------------


def test_cli_pin_list_unpin_roundtrip(store: KBStore) -> None:
    spec = _claim(store, "the spec says tokens rotate every 24 hours")
    runner = CliRunner()

    empty = runner.invoke(cli, ["pins", "list"])
    assert empty.exit_code == 0
    assert "no pins" in empty.output

    added = runner.invoke(cli, ["pin", spec, "--note", "the constraint"])
    assert added.exit_code == 0, added.output
    assert "pinned claim/" in added.output

    listed = runner.invoke(cli, ["pins", "list"])
    assert listed.exit_code == 0
    assert spec in listed.output
    assert "the constraint" in listed.output

    removed = runner.invoke(cli, ["unpin", spec])
    assert removed.exit_code == 0, removed.output
    assert "no pins" in runner.invoke(cli, ["pins", "list"]).output


def test_cli_pin_json_output(store: KBStore) -> None:
    spec = _claim(store, "the spec says tokens rotate every 24 hours")
    CliRunner().invoke(cli, ["pin", spec])
    res = CliRunner().invoke(cli, ["pins", "list", "--json"])
    assert res.exit_code == 0, res.output
    payload = json.loads(res.output)
    assert payload["pins"][0]["id"] == spec
    assert payload["pins"][0]["local"] is False


def test_cli_local_pin_is_labelled(store: KBStore) -> None:
    spec = _claim(store, "the spec says tokens rotate every 24 hours")
    res = CliRunner().invoke(cli, ["pin", spec, "--local"])
    assert res.exit_code == 0, res.output
    assert "(local)" in res.output
    assert "local " in CliRunner().invoke(cli, ["pins", "list"]).output


def test_cli_expiry_is_accepted_and_shown(store: KBStore) -> None:
    spec = _claim(store, "the spec says tokens rotate every 24 hours")
    assert CliRunner().invoke(cli, ["pin", spec, "--expires", "7d"]).exit_code == 0
    listed = CliRunner().invoke(cli, ["pins", "list"])
    assert "expires" in listed.output
    assert pins.load_pins(store)[0].expires_at is not None


def test_cli_expiry_accepts_an_iso_date_in_the_future(store: KBStore) -> None:
    """An ISO date must expire in the future, not be mirrored into the past.

    `parse_since` returns ISO input unchanged, so mirroring it around now (the
    handling durations need) turned a future date into a past one and created
    the pin already expired — silently, since nothing rejects it.
    """
    spec = _claim(store, "the spec says tokens rotate every 24 hours")
    tomorrow = (datetime.now(UTC) + timedelta(days=1)).date().isoformat()

    res = CliRunner().invoke(cli, ["pin", spec, "--expires", tomorrow])
    assert res.exit_code == 0, res.output

    # load_pins drops expired pins on read, so the old inversion made the pin
    # disappear outright — this list is empty before the fix.
    live = pins.load_pins(store)
    assert [p.artifact_id for p in live] == [spec]
    assert live[0].expires_at is not None
    assert live[0].expires_at > datetime.now(UTC), f"{tomorrow} stored wrong"


def test_cli_expiry_accepts_a_full_iso_timestamp(store: KBStore) -> None:
    spec = _claim(store, "the spec says tokens rotate every 24 hours")
    when = (datetime.now(UTC) + timedelta(days=3)).replace(microsecond=0)

    res = CliRunner().invoke(cli, ["pin", spec, "--expires", when.isoformat()])
    assert res.exit_code == 0, res.output
    assert pins.load_pins(store)[0].expires_at == when


@pytest.mark.parametrize("spec_text", ["all", "", "not-a-real-spec"])
def test_cli_expiry_rejects_specs_that_mean_no_bound(
    store: KBStore, spec_text: str
) -> None:
    """`all` / `""` resolve to "no lower bound", which as an expiry means never.

    Never-expires is already what omitting the flag does, so accepting these
    silently would hide a typo rather than honour a request.
    """
    spec = _claim(store, "the spec says tokens rotate every 24 hours")
    res = CliRunner().invoke(cli, ["pin", spec, "--expires", spec_text])

    assert res.exit_code != 0
    assert "Error:" in res.output
    assert not pins.load_pins(store), "no pin should be written on a bad expiry"


def test_cli_unknown_artifact_is_a_clean_error(store: KBStore) -> None:
    """A domain error must render as `Error: ...`, never a traceback."""
    res = CliRunner().invoke(cli, ["pin", "never-existed"])
    assert res.exit_code != 0
    assert "Error: unknown artifact" in res.output
    assert "Traceback" not in res.output


def test_cli_unpinning_something_unpinned_is_a_clean_error(store: KBStore) -> None:
    res = CliRunner().invoke(cli, ["unpin", "never-pinned"])
    assert res.exit_code != 0
    assert "not in the shared pin set" in res.output
    assert "Traceback" not in res.output


# --- defensive paths ------------------------------------------------------


def test_unreadable_config_falls_back_to_defaults(store: KBStore) -> None:
    store.config_path.write_text("{{ not yaml", encoding="utf-8")
    assert pins.load_config(store).budget_share == pins.DEFAULT_BUDGET_SHARE


def test_non_mapping_config_falls_back_to_defaults(store: KBStore) -> None:
    store.config_path.write_text("- a list, not a mapping\n", encoding="utf-8")
    assert pins.load_config(store).enabled is pins.DEFAULT_ENABLED


def test_config_without_a_retrieval_block_falls_back(store: KBStore) -> None:
    store.config_path.write_text(
        yaml.safe_dump({"review": {"auto_approve_on_receipt": True}}),
        encoding="utf-8",
    )
    assert pins.load_config(store).budget_share == pins.DEFAULT_BUDGET_SHARE


def test_pin_file_that_is_not_a_list_yields_no_pins(store: KBStore) -> None:
    (store.kb_dir / pins.SHARED_FILENAME).write_text(
        yaml.safe_dump({"pins": {"not": "a list"}}), encoding="utf-8"
    )
    assert pins.load_pins(store) == []


def test_gitignore_is_only_appended_once(store: KBStore) -> None:
    first = _claim(store, "the spec says tokens rotate every 24 hours")
    second = _claim(store, "we rejected polling because it doubles cost")
    pins.add_pin(store, first, pinned_by="human", local=True)
    pins.add_pin(store, second, pinned_by="human", local=True)

    text = (store.kb_dir / ".gitignore").read_text(encoding="utf-8")
    assert text.count(pins.LOCAL_FILENAME) == 1


def test_gitignore_without_a_trailing_newline_is_extended_cleanly(
    store: KBStore
) -> None:
    gi = store.kb_dir / ".gitignore"
    gi.write_text("state.db", encoding="utf-8")  # no trailing newline
    spec = _claim(store, "the spec says tokens rotate every 24 hours")
    pins.add_pin(store, spec, pinned_by="human", local=True)

    lines = gi.read_text(encoding="utf-8").splitlines()
    assert "state.db" in lines
    assert pins.LOCAL_FILENAME in lines


def test_a_pinned_artifact_deleted_from_disk_is_skipped(store: KBStore) -> None:
    """The yaml can go while the pin file survives; that must not raise."""
    spec = _claim(store, "the spec says tokens rotate every 24 hours")
    store.put_page(Page(id="p1", title="design notes", body="b",
                        type=PageType.CONCEPT))
    pins.add_pin(store, spec, pinned_by="human")
    pins.add_pin(store, "p1", pinned_by="human")
    assert len(pins.pinned_items(store, max_chars=4000)) == 2

    store._claim_path(spec).unlink()
    store._page_path("p1").unlink()
    assert pins.pinned_items(store, max_chars=4000) == []
