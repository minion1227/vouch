"""Correction capture — the highest-signal event in a session (#430).

The adapter already captures tool *outcomes* passively (`PostToolUse`). The
one thing it never captured is the user pushing back — "no, we deploy from
`main` not `release`" — which is simultaneously the knowledge most worth
keeping and the knowledge most reliably lost, because keeping it required
someone to remember to propose a claim afterwards.

This turns a detected correction into a **proposal**. Never a write.

The whole design constraint is in that sentence. This module routes
exclusively through `proposals.propose_quoted_claim`, it has no import of
`proposals.approve`, and the pending queue *is* the draft state — a human
still drains it. Three guards keep an over-eager heuristic from becoming a
reviewer's problem:

* **a cheap trigger** — regex pushback detection on the turn boundary, no
  LLM call, so this costs nothing per turn and stays deterministic.
* **dedup** — a repeated correction does not re-file. Checked against
  approved claims and against this session's own pending queue.
* **a per-session cap** — `capture.correction.max_per_session` bounds how
  much one run can put in front of a reviewer, counted from the queue itself
  so it survives a process restart.

The claim cites a receipt: the user's message is registered as a `message`
source and the corrective sentence is quoted verbatim out of it, so what
lands in the queue is mechanically verifiable rather than a paraphrase the
reviewer has to take on faith.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

import yaml

from .config_coerce import coerce_bool
from .models import ProposalStatus
from .secrets import mask_secrets
from .storage import KBStore

_log = logging.getLogger(__name__)

CORRECTION_ACTOR = "auto:correction"
CORRECTION_TAG = "auto:correction"
CORRECTION_RATIONALE = "captured from user correction"

DEFAULT_ENABLED = True
DEFAULT_MAX_PER_SESSION = 3
DEFAULT_MIN_CHARS = 12
# Above this token overlap with an existing approved claim or a pending
# correction, the correction is treated as already known and dropped.
DEFAULT_DEDUP_THRESHOLD = 0.6

# A correction is longer than a token of disagreement but shorter than a
# fresh instruction; past this it is a new task, not a fix to the last one.
MAX_CORRECTION_CHARS = 400

# Openers that mark the turn as a correction of what just happened. Anchored
# at the start of the prompt (or of a sentence in it) so "no" inside ordinary
# prose — "there is no config file" — does not trip the heuristic.
_PUSHBACK = re.compile(
    r"""^\s*(?:
        no[,.\s!]+ |
        nope[,.\s!]+ |
        wrong[,.\s!]+ |
        that'?s\s+(?:not\s+right|wrong|incorrect) |
        not\s+(?:quite|right|correct) |
        actually[,\s] |
        incorrect[,.\s!]+ |
        don'?t\s+do\s+that |
        i\s+(?:said|meant|told\s+you)
    )""",
    re.IGNORECASE | re.VERBOSE,
)

# The correction is only meaningful if it also *asserts* something. A bare
# "no." is disagreement without content and is not worth a reviewer's time.
_ASSERTION = re.compile(
    r"\b(?:is|are|was|were|use|uses|should|must|always|never|it'?s|we|the)\b",
    re.IGNORECASE,
)


# Words too common to say anything about whether two corrections match.
_STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "do", "dont",
    "for", "from", "in", "is", "it", "its", "must", "not", "of", "on", "or",
    "should", "that", "the", "this", "to", "use", "was", "we", "were", "you",
})

_TOKEN_RE = re.compile(r"[a-z0-9]+")


class CorrectionError(RuntimeError):
    pass


def _tokens(text: str) -> set[str]:
    return {
        t for t in _TOKEN_RE.findall(text.lower())
        if t not in _STOPWORDS and len(t) > 2
    }


def overlap(a: str, b: str) -> float:
    """Jaccard overlap of the two texts' significant tokens, 0.0 to 1.0."""
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


@dataclass(frozen=True)
class CorrectionConfig:
    enabled: bool = DEFAULT_ENABLED
    max_per_session: int = DEFAULT_MAX_PER_SESSION
    min_chars: int = DEFAULT_MIN_CHARS
    dedup_threshold: float = DEFAULT_DEDUP_THRESHOLD


def load_config(store: KBStore) -> CorrectionConfig:
    """Read ``capture.correction:`` from config.yaml; fall back to defaults."""
    try:
        loaded = yaml.safe_load(store.config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError):
        return CorrectionConfig()
    if not isinstance(loaded, dict):
        return CorrectionConfig()
    capture = loaded.get("capture")
    if not isinstance(capture, dict):
        return CorrectionConfig()
    raw = capture.get("correction")
    if not isinstance(raw, dict):
        return CorrectionConfig()
    try:
        max_per_session = int(raw.get("max_per_session", DEFAULT_MAX_PER_SESSION))
    except (TypeError, ValueError):
        max_per_session = DEFAULT_MAX_PER_SESSION
    try:
        min_chars = int(raw.get("min_chars", DEFAULT_MIN_CHARS))
    except (TypeError, ValueError):
        min_chars = DEFAULT_MIN_CHARS
    try:
        threshold = float(raw.get("dedup_threshold", DEFAULT_DEDUP_THRESHOLD))
    except (TypeError, ValueError):
        threshold = DEFAULT_DEDUP_THRESHOLD
    return CorrectionConfig(
        enabled=coerce_bool(raw.get("enabled", DEFAULT_ENABLED), DEFAULT_ENABLED),
        max_per_session=max(0, max_per_session),
        min_chars=max(0, min_chars),
        dedup_threshold=threshold,
    )


def detect(prompt: str, *, min_chars: int = DEFAULT_MIN_CHARS) -> str | None:
    """The corrective statement inside `prompt`, or None if it isn't one.

    Deliberately cheap and deliberately conservative: a false negative costs
    one lost correction, a false positive costs a reviewer's attention, and
    the second is the one that makes an ambient feature get turned off.
    """
    text = (prompt or "").strip()
    if not text or len(text) > MAX_CORRECTION_CHARS:
        return None
    if not _PUSHBACK.match(text):
        return None
    if len(text) < min_chars:
        return None
    # Strip the opener itself: "no, we deploy from main" is worth keeping as
    # "we deploy from main" — the disagreement is context, not knowledge.
    stripped = _PUSHBACK.sub("", text, count=1).strip(" ,.;:!-—")
    if not stripped or len(stripped) < min_chars:
        return None
    if not _ASSERTION.search(stripped):
        return None
    return stripped


def _session_pending(store: KBStore, session_id: str | None) -> list[Any]:
    return [
        p for p in store.list_proposals(ProposalStatus.PENDING)
        if p.proposed_by == CORRECTION_ACTOR
        and (session_id is None or p.session_id == session_id)
    ]


def _already_known(store: KBStore, text: str, *, threshold: float) -> str | None:
    """The id of an approved claim or pending correction that already says this.

    Lexical on purpose, matching the lesson repeat guard: the embedding path
    (#147) needs the `[embeddings]` extra, and dedup that silently stops
    working on a base install is precisely how an unattended capture floods a
    queue. The embedding hits are folded in on top when available.
    """
    for claim in store.list_claims():
        if overlap(text, claim.text) >= threshold:
            return claim.id
    for proposal in _session_pending(store, None):
        existing = str(proposal.payload.get("text", ""))
        if overlap(text, existing) >= threshold:
            return proposal.id
    try:
        from .embeddings.similarity import find_similar_on_propose

        for warning in find_similar_on_propose(store, text):
            artifact_id = warning.get("artifact_id")
            if isinstance(artifact_id, str):
                return artifact_id
    except ImportError:
        pass
    return None


def _skip(reason: str, **extra: Any) -> dict[str, Any]:
    return {"captured": False, "reason": reason, **extra}


def capture(
    store: KBStore,
    *,
    prompt: str,
    session_id: str | None = None,
    agent: str | None = None,
    context: str | None = None,
) -> dict[str, Any]:
    """File a detected correction as a pending claim proposal.

    Returns a small report either way — `{"captured": False, "reason": ...}`
    when a guard declined, so a caller (or a test) can see *why* nothing was
    filed instead of inferring it from silence. Never raises on a normal
    decline; the only errors are a broken KB.
    """
    cfg = load_config(store)
    if not cfg.enabled:
        return _skip("disabled")

    corrective = detect(prompt, min_chars=cfg.min_chars)
    if corrective is None:
        return _skip("not_a_correction")

    # Mask before anything durable is written. A correction is free-form user
    # text typed in a hurry — exactly where a pasted credential shows up.
    corrective = mask_secrets(corrective)

    filed = len(_session_pending(store, session_id))
    if filed >= cfg.max_per_session:
        return _skip("session_cap", cap=cfg.max_per_session, filed=filed)

    duplicate = _already_known(store, corrective, threshold=cfg.dedup_threshold)
    if duplicate is not None:
        return _skip("duplicate", duplicate_of=duplicate)

    # The user's own message is the source, and the corrective sentence is
    # quoted verbatim out of it, so the proposal carries a receipt the gate
    # can verify by string comparison rather than a paraphrase.
    source = store.put_source(
        corrective.encode("utf-8"),
        title="user correction",
        source_type="message",
        tags=[CORRECTION_TAG],
        metadata={
            "session_id": session_id,
            "agent": agent,
            "context": context,
        },
    )

    # Imported here rather than at module scope: proposals imports lessons,
    # and a module-scope import back into proposals would be circular.
    from .proposals import propose_quoted_claim

    result = propose_quoted_claim(
        store,
        text=corrective,
        source_id=source.id,
        quote=corrective,
        proposed_by=CORRECTION_ACTOR,
        claim_type="preference",
        confidence=0.5,
        tags=[CORRECTION_TAG],
        rationale=CORRECTION_RATIONALE,
        session_id=session_id,
    )
    if result is None:  # pragma: no cover - quote is the source, always found
        return _skip("no_receipt")
    return {
        "captured": True,
        "proposal_id": result.proposal.id,
        "status": result.proposal.status.value,
        "text": corrective,
        "source_id": source.id,
        "warnings": result.warnings,
    }


def maybe_capture(
    store: KBStore,
    *,
    prompt: str,
    session_id: str | None = None,
    agent: str | None = None,
) -> dict[str, Any] | None:
    """`capture` for hook callers: swallows every failure, returns None on one.

    The UserPromptSubmit hook's contract is that it must never break a turn,
    so an unwritable KB or a malformed config drops the correction rather
    than raising into the host.
    """
    try:
        report = capture(store, prompt=prompt, session_id=session_id, agent=agent)
    except Exception:
        _log.warning("correction capture failed", exc_info=True)
        return None
    return report if report.get("captured") else None
