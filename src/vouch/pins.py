"""Explicit pins — a working set that always enters the context pack (#615).

Retrieval ranks for the *query*, so the three or four artifacts that should be
in front of an agent on every turn of a long task drop out the moment the
conversation moves. ``hot_memory`` and ``salience`` are the implicit,
recency-driven version of this and decay exactly when a long task needs them
not to. A pin is the explicit version: this specific artifact, until I say
otherwise.

**Not a gate bypass.** A pin is a pointer to an artifact that is *already*
approved — it asserts nothing new and creates no durable knowledge, so there is
nothing for a reviewer to review. Pinning cannot introduce a claim, and an
artifact that was never approved cannot be pinned in the first place.

**Pins never resurrect retired knowledge.** A pinned claim that is later
superseded, archived or redacted, or a pinned page that is archived, stops
being injected — the pin does not override lifecycle. The same applies to
viewer scope: a pin is not a way to see something the scope filter hides.
Pins are a ranking instruction, never a permission.

**Pins cannot starve retrieval.** Injected pins are capped at
``retrieval.pins.budget_share`` of the pack's character budget (default 0.3),
so the ranker always keeps the majority of the pack.

Two files, both under ``.vouch/``:

* ``pins.yaml`` — committed, so a team shares one working set and changes to
  it show up in review like everything else.
* ``pins.local.yaml`` — gitignored, for a personal working set.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from .config_coerce import coerce_bool

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .storage import KBStore

logger = logging.getLogger(__name__)

SHARED_FILENAME = "pins.yaml"
LOCAL_FILENAME = "pins.local.yaml"

DEFAULT_ENABLED = True
DEFAULT_BUDGET_SHARE = 0.3
DEFAULT_MAX_CHARS = 2000

# Kinds a pin may point at. Claims and pages are what a working set is made of;
# the rest of the artifact kinds are graph structure, not reading material.
PINNABLE_KINDS = ("claim", "page")


class PinError(RuntimeError):
    """A pin could not be created or removed."""


@dataclass(frozen=True)
class Pin:
    """One pinned artifact."""

    artifact_id: str
    kind: str
    pinned_at: datetime
    pinned_by: str
    expires_at: datetime | None = None
    note: str | None = None
    local: bool = False

    def expired(self, *, now: datetime | None = None) -> bool:
        if self.expires_at is None:
            return False
        return (now or datetime.now(UTC)) >= self.expires_at

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "id": self.artifact_id,
            "kind": self.kind,
            "pinned_at": self.pinned_at.isoformat(timespec="seconds"),
            "pinned_by": self.pinned_by,
        }
        if self.expires_at is not None:
            out["expires_at"] = self.expires_at.isoformat(timespec="seconds")
        if self.note:
            out["note"] = self.note
        return out


@dataclass(frozen=True)
class PinsConfig:
    enabled: bool = DEFAULT_ENABLED
    budget_share: float = DEFAULT_BUDGET_SHARE


def load_config(store: KBStore) -> PinsConfig:
    """Read ``retrieval.pins`` from config.yaml; fall back to defaults."""
    try:
        loaded = yaml.safe_load(store.config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return PinsConfig()
    if not isinstance(loaded, dict):
        return PinsConfig()
    retrieval = loaded.get("retrieval")
    raw = retrieval.get("pins") if isinstance(retrieval, dict) else None
    if not isinstance(raw, dict):
        return PinsConfig()
    share = raw.get("budget_share", DEFAULT_BUDGET_SHARE)
    if not isinstance(share, int | float) or isinstance(share, bool):
        share = DEFAULT_BUDGET_SHARE
    # A share outside (0, 1] is a config typo, not an instruction to disable
    # retrieval or to let pins take the whole pack.
    share = min(1.0, max(0.0, float(share)))
    return PinsConfig(
        enabled=coerce_bool(raw.get("enabled", DEFAULT_ENABLED), DEFAULT_ENABLED),
        budget_share=share,
    )


def _path(store: KBStore, *, local: bool) -> Path:
    return store.kb_dir / (LOCAL_FILENAME if local else SHARED_FILENAME)


def _ensure_ignored(store: KBStore) -> None:
    """Keep the local pin set out of git, the way retrieval telemetry is.

    Best-effort: an unwritable .gitignore must not stop someone pinning.
    """
    gi = store.kb_dir / ".gitignore"
    try:
        text = gi.read_text(encoding="utf-8") if gi.exists() else ""
        if LOCAL_FILENAME in text:
            return
        if text and not text.endswith("\n"):
            text += "\n"
        gi.write_text(f"{text}{LOCAL_FILENAME}\n", encoding="utf-8")
    except OSError as e:  # pragma: no cover - filesystem edge
        logger.debug("pins: could not update .gitignore (%s)", e)


def _parse_dt(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed


def _read_file(store: KBStore, *, local: bool) -> list[Pin]:
    """Parse one pin file. A malformed entry is skipped, never fatal."""
    path = _path(store, local=local)
    if not path.exists():
        return []
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return []
    rows = loaded.get("pins") if isinstance(loaded, dict) else loaded
    if not isinstance(rows, list):
        return []
    out: list[Pin] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        artifact_id = str(row.get("id", "")).strip()
        kind = str(row.get("kind", "")).strip()
        if not artifact_id or kind not in PINNABLE_KINDS:
            continue
        out.append(Pin(
            artifact_id=artifact_id,
            kind=kind,
            pinned_at=_parse_dt(row.get("pinned_at")) or datetime.now(UTC),
            pinned_by=str(row.get("pinned_by", "unknown")),
            expires_at=_parse_dt(row.get("expires_at")),
            note=row.get("note") if isinstance(row.get("note"), str) else None,
            local=local,
        ))
    return out


def _write_file(store: KBStore, pins: list[Pin], *, local: bool) -> None:
    if local:
        _ensure_ignored(store)
    path = _path(store, local=local)
    if not pins:
        path.unlink(missing_ok=True)
        return
    path.write_text(
        yaml.safe_dump({"pins": [p.to_dict() for p in pins]}, sort_keys=False),
        encoding="utf-8",
    )


def load_pins(
    store: KBStore, *, include_local: bool = True, now: datetime | None = None
) -> list[Pin]:
    """Every live pin, shared first then local, expired ones dropped.

    Expiry is applied on read rather than by rewriting the file: reading the
    KB must not mutate it, and a pin set that quietly rewrote itself during a
    context build would be a write on the read path.
    """
    pins = _read_file(store, local=False)
    if include_local:
        pins += _read_file(store, local=True)
    seen: set[tuple[str, str]] = set()
    live: list[Pin] = []
    for pin in pins:
        key = (pin.kind, pin.artifact_id)
        if key in seen or pin.expired(now=now):
            continue
        seen.add(key)
        live.append(pin)
    return live


def _resolve_kind(store: KBStore, artifact_id: str) -> str | None:
    """``"claim"`` / ``"page"`` / ``None`` — what this id actually is."""
    from .storage import ArtifactNotFoundError

    try:
        store.get_claim(artifact_id)
        return "claim"
    except ArtifactNotFoundError:
        pass
    try:
        store.get_page(artifact_id)
        return "page"
    except ArtifactNotFoundError:
        return None


def add_pin(
    store: KBStore,
    artifact_id: str,
    *,
    pinned_by: str,
    local: bool = False,
    expires_at: datetime | None = None,
    note: str | None = None,
) -> Pin:
    """Pin an existing claim or page. Raises :class:`PinError` otherwise."""
    artifact_id = artifact_id.strip()
    if not artifact_id:
        raise PinError("pin needs an artifact id")
    kind = _resolve_kind(store, artifact_id)
    if kind is None:
        raise PinError(
            f"unknown artifact {artifact_id!r}: a pin points at an approved "
            f"claim or page, so there is nothing to pin until it exists"
        )
    existing = _read_file(store, local=local)
    kept = [p for p in existing if p.artifact_id != artifact_id]
    pin = Pin(
        artifact_id=artifact_id,
        kind=kind,
        pinned_at=datetime.now(UTC),
        pinned_by=pinned_by,
        expires_at=expires_at,
        note=note,
        local=local,
    )
    _write_file(store, [*kept, pin], local=local)
    return pin


def remove_pin(store: KBStore, artifact_id: str, *, local: bool = False) -> bool:
    """Unpin. Returns False when it was not pinned in that set."""
    artifact_id = artifact_id.strip()
    existing = _read_file(store, local=local)
    kept = [p for p in existing if p.artifact_id != artifact_id]
    if len(kept) == len(existing):
        return False
    _write_file(store, kept, local=local)
    return True


def pinned_items(
    store: KBStore,
    *,
    viewer: Any = None,
    max_chars: int | None = None,
    config: PinsConfig | None = None,
) -> list[Any]:
    """Live pinned artifacts as ``ContextItem``s, within the pin budget.

    Lifecycle and scope are re-checked on every build: a pin records *what* to
    prefer, never a right to see it. Retired artifacts and out-of-scope ones
    are dropped exactly as retrieval would drop them.
    """
    from .context import _RETRACTED_CLAIM_STATUSES, _page_is_live
    from .models import ContextItem
    from .scoping import artifact_scope_for_hit, is_visible
    from .storage import ArtifactNotFoundError

    cfg = config or load_config(store)
    if not cfg.enabled:
        return []

    budget = int((max_chars or DEFAULT_MAX_CHARS) * cfg.budget_share)
    items: list[Any] = []
    used = 0
    for pin in load_pins(store):
        summary: str | None = None
        citations: list[str] = []
        if pin.kind == "claim":
            try:
                claim = store.get_claim(pin.artifact_id)
            except ArtifactNotFoundError:
                continue
            if claim.status in _RETRACTED_CLAIM_STATUSES:
                continue
            summary = claim.text
            citations = list(claim.evidence)
        else:
            # No missing-page guard: `_page_is_live` returns False for a page
            # whose yaml is gone as well as for an archived one, so anything
            # reaching the read here has already been proved readable.
            if not _page_is_live(store, pin.artifact_id):
                continue
            summary = store.get_page(pin.artifact_id).title

        if viewer is not None:
            scope = artifact_scope_for_hit(store, pin.kind, pin.artifact_id)
            if scope is not None and not is_visible(scope, viewer):
                continue

        summary = summary or ""
        # Stop at the first pin that would breach the share rather than
        # skipping it and taking a later, smaller one — pin order is the
        # user's stated priority and silently reordering it would be worse
        # than dropping the tail.
        if used + len(summary) > budget and items:
            break
        used += len(summary)
        items.append(ContextItem(
            id=pin.artifact_id,
            type=pin.kind,  # type: ignore[arg-type]
            summary=summary,
            score=1.0,
            backend="pin",
            citations=citations,
            freshness="unknown",
        ))
    return items
