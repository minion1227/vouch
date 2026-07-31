---
vep: 0007
title: cascade option for propose_delete
author: devmixa702
status: draft
created: 2026-07-30
landed-in: ""
supersedes: []
superseded-by: ""
---

# VEP-0007: cascade option for propose_delete

## Summary

Add an optional `cascade` flag to `kb.propose_delete`. With it set, the
proposal carries the referrer edits needed to release the target — pages and
claims get their pointer unlinked, relations get deleted — so the reviewer
approves the target and everything that had to move as a single reviewed
unit. Default is `false`, so today's refusal is unchanged.

## Motivation

`referenced_by()` refuses a delete while anything still points at the target
(`src/vouch/proposals.py:1150`). the rule is correct, but in a compiled kb
pages cite claims in bulk, so a large fraction of claims end up permanently
undeletable. the kb this was reported from has 1638 claims across 107 pages.

two cases are worse than "just referenced":

- **supersede pairs are mutually locked.** `lifecycle.supersede`
  (`src/vouch/lifecycle.py:41`) writes three pointers for one logical edge:
  `new.supersedes` gains `old`, `old.superseded_by` becomes `new`, and a
  `{new}--supersedes--{old}` relation is minted. deleting `old` is blocked by
  `new`, deleting `new` is blocked by `old`, and the relation blocks both. no
  delete ordering escapes it, and `contradicts` behaves the same way.
- **the relation endpoint check is kind-blind.** `referenced_by` compares the
  target id against bare `rel.source` / `rel.target`, which carry no kind tag
  (`src/vouch/models.py:310`), so a relation about a same-slug entity or page
  blocks an unrelated claim. see "open questions" — this is not fixed here,
  and it interacts badly with cascade.

the workarounds are all dead ends. `kb.supersede` is the right move when a
replacement exists, but not when the claim is simply wrong noise, and it mints
another undeletable pair. hand-editing the page's `claims:` list works and is
exactly the parallel data path that bypasses `proposals.approve()` that the
project forbids.

## Proposal

one new optional parameter on an existing method:

```
kb.propose_delete(target_kind, target_id, cascade=false, rationale=?, session_id=?, dry_run=?)
```

with `cascade=true`, the filed proposal's payload gains a `cascade` key
alongside the existing `target_kind` / `id` / `snapshot`:

```yaml
payload:
  target_kind: claim
  id: postgres-tuning-matters
  snapshot: {...}
  cascade:
    - kind: page
      id: db-notes
      unlink_claims: [postgres-tuning-matters]
    - kind: claim
      id: newer-claim
      unlink_supersedes: [postgres-tuning-matters]
    - kind: relation
      id: newer-claim--supersedes--postgres-tuning-matters
      action: delete
```

surface mirrors, all four registration sites:

- `src/vouch/server.py` — `kb_propose_delete` gains `cascade: bool = False`.
- `src/vouch/jsonl_server.py` — `_h_propose_delete` reads `p.get("cascade", False)`.
- `src/vouch/capabilities.py` — `METHODS` is unchanged; no new method name.
- `src/vouch/cli.py` — `vouch propose-delete <kind> <id> --cascade`.

the refusal message at `proposals.py:523-530` gains a trailing hint naming the
flag, so the dead end is discoverable from the error alone:

```
cannot delete claim c1: referenced by page 'p1' (supersede it instead? or retry with cascade)
```

no new `kb.*` method, no on-disk layout change, no bundle format change. two
new audit event *names* are added — see Compatibility.

## Design

**building the cascade at propose time.** `propose_delete` already calls
`referenced_by`; when `cascade=true` it instead calls a new
`_cascade_plan(store, target_kind, target_id)` that walks the same referrers
and returns typed edit records rather than display strings. `referenced_by`
keeps its current signature and behaviour — it is the display path, reached
from the propose-time refusal and the approve-time re-check, and it is pinned
by six tests in `tests/test_delete.py`.

the plan is derived, never trusted from the caller. a caller cannot hand-craft
a cascade payload that touches an artifact the walk did not find, because the
payload is rebuilt from scratch at approve time and compared.

**per-referrer actions:**

| referrer holds | action |
|---|---|
| `page.claims` / `page.entities` contains target | unlink — drop the id from the list |
| `claim.entities` contains target | unlink |
| `claim.supersedes` contains target | unlink |
| `claim.superseded_by == target` | clear to `None`, and reset `status` from `SUPERSEDED` back to the pre-supersede value |
| relation with an endpoint at target | delete the relation |

the `superseded_by` case is the subtle one. `ClaimStatus.SUPERSEDED` is a
member of `_RETRACTED_CLAIM_STATUSES` (`src/vouch/context.py:47`), so clearing
the pointer without resetting the status would leave the claim permanently
suppressed from recall, context, and search with nothing superseding it —
live knowledge that no surface will ever show again. note that `vouch doctor`
would *not* catch it: `health.py:409` flags `dangling_superseded_by` only when
the pointer is non-null and points at a missing claim. the plan therefore
records the status transition explicitly rather than inferring it at apply
time.

**applying at approve time.** `_approve_delete` (`proposals.py:1218`)
re-derives the plan before applying it, the same way it re-checks
`referenced_by` today at `:1248`:

```
plan_at_approve = _cascade_plan(store, target_kind, target_id)
if plan_at_approve != proposal.payload["cascade"]:
    raise ProposalError("cascade changed since the proposal was filed; re-propose")
apply each edit (put_page / update_claim / delete_relation)
assert referenced_by(...) == []          # belt and braces
delete the target, deindex, audit
```

refusing on drift rather than silently re-planning is deliberate: the reviewer
approved a specific set of edits, and a page that gained a citation since the
proposal was filed is a new decision, not the one that was approved.

**audit.** the log names events with domain verbs (`claim.archive`,
`claim.redact`, `page.dead_refs_wipe`), not a generic update, so the cascade
adds two names in that style: `claim.cascade_unlink` and
`page.cascade_unlink`, carrying the target id and the field that was cleared.
relation removals reuse the existing `relation.delete` that `_approve_delete`
already emits. every cascade event is written before the terminal
`{target_kind}.delete` with the same `actor`, so the log reads as one chain
and `vouch audit` on any touched artifact shows why it moved. the existing
`proposal.delete.approve` tail is unchanged.

**pre-flight.** the `DELETE` branch of `_payload_block_reason`
(`proposals.py:758`, branch at `:816-831`) must not report a cascade proposal
as blocked — it currently returns the refusal whenever `referenced_by` is
non-empty. it gains the same cascade awareness so `vouch pending` does not
show an approvable proposal as stuck.

**console.** `webapp/src/components/DeleteArtifactButton.tsx` already has a
`confirming` step and already surfaces a referenced-block refusal verbatim as
a toast. that confirm step gains the referrer list and a "delete with cascade"
action, which re-issues `kb.propose_delete` with `cascade: true`.

## Compatibility

- **`.vouch/` layout:** unchanged. no migration.
- **existing proposals:** a decided delete proposal without a `cascade` key is
  read as an empty plan. `payload` is already free-form.
- **bundle format:** unchanged.
- **audit-log shape:** the `AuditEvent` *shape* is unchanged, but two new
  event *names* are introduced (`claim.cascade_unlink`,
  `page.cascade_unlink`). `AuditEvent.event` is an open `str` ("dotted verb"),
  and `vouch audit` does not switch on a closed set, so in-tree readers need
  no change; any downstream consumer that does enumerate event names would.
  flagged explicitly because the audit-log surface is one of the things this
  VEP exists to get accepted.
- **`kb.capabilities`:** unchanged — no method added, renamed, or removed.
  note that `tests/test_capabilities.py` will therefore *not* catch a missing
  cli flag; the parameter needs its own cross-surface test.
- **default behaviour:** unchanged. `cascade` defaults to `false`, so every
  existing caller and every existing kb behaves exactly as it does today.

## Security implications

the review gate is untouched. `propose_delete` still files to `proposed/`,
approval still requires `approved_by != proposed_by`, and nothing new is
auto-approved.

the change does widen the blast radius of a single approval: approving one
proposal can now edit several artifacts instead of one. three mitigations:

1. **the plan is reviewer-visible.** the cascade is in the payload, so
   `vouch show <id>` and the console confirm step both name every artifact
   that will move before anyone approves.
2. **the plan is re-derived, not replayed.** a payload edited in `proposed/`
   between filing and approval is caught by the drift check, which compares
   against a freshly computed plan.
3. **drift refuses rather than re-plans.** a referrer added after filing
   invalidates the proposal instead of being silently swept up.

the residual risk is a reviewer approving a large cascade without reading it —
the same risk the gate always carries, recorded the same way in the audit log.

## Performance implications

not a hot path. `_cascade_plan` walks the same `list_pages` / `list_claims` /
`list_relations` that `referenced_by` already walks, once at propose and once
at approve. on the reported 1638-claim / 107-page kb that is milliseconds, and
delete is an interactive, one-at-a-time operation.

## Open questions

1. **the kind-blind endpoint match is not fixed here, and cascade makes it
   dangerous.** the suggestion in the source issue was to "compare endpoints
   only when the endpoint's kind resolves to the target's kind". that does not
   work. resolving a bare id by probing store paths — the `graph._node_kind`
   idiom at `src/vouch/graph.py:33` — is ambiguous in exactly the failing case,
   because the artifact being deleted always occupies its own id. reproduced
   against a store holding a claim and an entity both id'd `postgres` plus a
   relation joining two entities:

   ```
   referenced_by(claim, 'postgres') = ["relation 'postgres--uses--redis'"]
   claim_path exists : True
   entity_path exists: True
   graph._node_kind('postgres') = claim      # matches target_kind, still blocks
   ```

   under `cascade=true` this stops being a nuisance and becomes data loss: the
   cascade would *delete* a legitimate relation between two entities in order
   to release an unrelated claim. **proposed interim rule: `_cascade_plan`
   refuses when a relation endpoint id resolves to more than one kind**, naming
   the ambiguity, rather than guessing. a real fix needs `source_kind` /
   `target_kind` on `Relation` — an object-model change with a migration, and
   its own VEP. should that VEP land first?

2. **should cascade be allowed to unlink the last claim off a page?** a page
   whose `claims:` list empties is still a valid page but may now be
   uncited prose. leaving it is the conservative default; flagging it in the
   plan is cheap.

3. **`--cascade` on the cli mirror only, or also a `vouch review` affordance?**
   the reviewer currently approves by id; offering cascade at approve time
   would mean mutating a filed proposal, which this VEP deliberately avoids.

## Alternatives considered

- **an `unlink` proposal kind** — `kb.propose_unlink(page, claim)` as its own
  reviewable write, then delete separately. more granular and more auditable,
  but it turns a one-claim cleanup into n+1 review rounds, and the intermediate
  states are pages that have already lost a citation for a claim that still
  exists.
- **soft delete / `status: retired`** — hide the claim from search and context
  without removing the file, leaving referrers intact. cheapest to build and
  never breaks a reference, but it does not answer "get this out of my kb" and
  storage keeps growing. `archive` already covers the hide-it case.
- **allow deleting a page-cited claim and let the page dangle** — the console
  already renders a broken reference for a missing claim, so it would not
  crash. rejected: silently corrupting pages is worse than the refusal.
- **relax `referenced_by` for supersede pairs specifically** — treat
  `superseded_by` as a mirror of the holder's own `supersedes` and stop
  counting it. rejected: it still dangles the pointer and the minted
  `--supersedes--` relation, so it trades a clean refusal for quiet corruption.

## References

- [issue #600](https://github.com/vouchdev/vouch/issues/600) — the report this VEP answers
- [VEP-0001: review gate](VEP-0001-review-gate.md) — the invariant this preserves
- `src/vouch/proposals.py:496` `propose_delete`, `:1150` `referenced_by`, `:1218` `_approve_delete`
- `src/vouch/lifecycle.py:41` `supersede` — the three-pointer bookkeeping
