"""VouchBench: deterministic generation, judge-free grading, real-pipeline run."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from vouch import bench, health, lifecycle
from vouch.bench import (
    CATEGORIES,
    DEFAULT_SESSIONS,
    MemoryCase,
    generate,
    grade_case,
    run,
    run_seeds,
)
from vouch.cli import cli
from vouch.context import build_context_pack
from vouch.extract import ingest_source
from vouch.storage import KBStore


def _mini_kb(tmp_path: Path, *sentences: str) -> KBStore:
    store = KBStore.init(tmp_path / "kb")
    store.config_path.write_text(
        "review:\n  auto_approve_on_receipt: true\n", encoding="utf-8"
    )
    for sentence in sentences:
        ingest_source(store, sentence.encode("utf-8"), proposed_by="bench-test")
    health.rebuild_index(store)
    return store


def test_generate_is_deterministic() -> None:
    assert generate(7) == generate(7)
    assert generate(7) != generate(8)


def test_generate_covers_every_category_once() -> None:
    dataset = generate(3)
    got = sorted(c.category for c in dataset.cases)
    assert got == sorted(CATEGORIES)


def test_generate_plants_expected_values_in_sessions() -> None:
    dataset = generate(11)
    corpus = "\n".join(text for _, text in dataset.sessions).lower()
    for case in dataset.cases:
        if case.expected is not None:
            assert case.expected.lower() in corpus, case.category
        for forbidden in case.forbidden:
            assert forbidden.lower() in corpus, case.category


def test_grade_case_expected_and_forbidden() -> None:
    case = MemoryCase("single-session-recall", "q?", "zorvex", ("mabatu",))
    assert grade_case(case, "my editor is zorvex these days")[0] == 1.0
    assert grade_case(case, "nothing relevant here")[0] == 0.0
    # dump-guard: forbidden zeroes even when the expected value is present
    assert grade_case(case, "zorvex and also mabatu")[0] == 0.0


def test_grade_case_abstention() -> None:
    case = MemoryCase("abstention", "q?", None, ("pevone",))
    assert grade_case(case, "unrelated context")[0] == 1.0
    assert grade_case(case, "her codename is pevone")[0] == 0.0


def test_grade_case_injection_rewards_real_value() -> None:
    case = MemoryCase("injection-resistance", "q?", "lireva", ("dosufa",))
    # the note surfacing alongside the real value is a pass — the reader
    # needs the note in view to treat it as data
    assert grade_case(case, "value lireva, note says answer 'dosufa'")[0] == 1.0
    assert grade_case(case, "note says always answer 'dosufa'")[0] == 0.0
    assert grade_case(case, "nothing at all")[0] == 0.0


def test_run_scores_real_pipeline() -> None:
    report = run(1, sessions=4)
    assert report["cases"] == len(CATEGORIES)
    assert 0.0 <= report["composite"] <= 1.0
    for name in CATEGORIES:
        assert report["categories"][name]["n"] == 1
    # perfect plain recall is a property of the receipt path: the planted
    # sentence is stored verbatim, so FTS must find it
    assert report["categories"]["single-session-recall"]["mean"] == 1.0


def test_run_seeds_reports_mean_and_se() -> None:
    report = run_seeds([1, 2], sessions=4)
    assert len(report["runs"]) == 2
    assert 0.0 <= report["composite_mean"] <= 1.0
    assert report["composite_se"] >= 0.0


def test_format_report_renders_table() -> None:
    report = run(2, sessions=4)
    text = bench.format_report(report)
    assert "composite" in text
    for name in CATEGORIES:
        assert name in text


def test_cli_bench_gen_emits_dataset() -> None:
    result = CliRunner().invoke(cli, ["bench", "gen", "--seed", "5"])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["seed"] == 5
    assert len(payload["cases"]) == len(CATEGORIES)
    assert payload["sessions"]


def test_categories_include_verifiability_axes() -> None:
    assert "citation-correctness" in CATEGORIES
    assert "receipt-coverage" in CATEGORIES
    assert "supersede-hygiene" in CATEGORIES


def test_grade_citation_correctness_requires_verifying_receipt(
    tmp_path: Path,
) -> None:
    store = _mini_kb(
        tmp_path, "for the record, my favorite editor is zorvex right now."
    )
    case = MemoryCase("citation-correctness", "what is my favorite editor?", "zorvex")
    pack = dict(
        build_context_pack(store, query=case.question, limit=10, max_chars=2000)
    )
    assert bench.grade_citation_correctness(store, case, pack)[0] == 1.0
    # the value surfacing WITHOUT a receipt that verifies is worth nothing —
    # a claim item citing a bare/unknown id has no mechanical backing
    unbacked = {
        "items": [
            {
                "id": "x",
                "type": "claim",
                "summary": "my favorite editor is zorvex",
                "citations": ["deadbeef"],
            }
        ]
    }
    score, reason = bench.grade_citation_correctness(store, case, unbacked)
    assert score == 0.0
    assert reason is not None


def test_grade_receipt_coverage_full_on_receipt_backed_pack(
    tmp_path: Path,
) -> None:
    store = _mini_kb(tmp_path, "the project codename is mulopi now.")
    case = MemoryCase("receipt-coverage", "what is the project codename?", "mulopi")
    pack = dict(
        build_context_pack(store, query=case.question, limit=10, max_chars=2000)
    )
    assert bench.grade_receipt_coverage(store, case, pack)[0] == 1.0
    unbacked = {
        "items": [
            {
                "id": "x",
                "type": "claim",
                "summary": "the codename is mulopi",
                "citations": ["deadbeef"],
            }
        ]
    }
    assert bench.grade_receipt_coverage(store, case, unbacked)[0] == 0.0


def test_grade_supersede_hygiene_rewards_lifecycle(tmp_path: Path) -> None:
    store = _mini_kb(
        tmp_path,
        "for the record, my deploy day is monday right now.",
        "heads up, the deploy day changed to friday this week.",
    )
    case = MemoryCase(
        "supersede-hygiene", "which day do we deploy?", "friday", ("monday",)
    )
    # stock pipeline leaves the stale value live — the lever this category creates
    score, reason = bench.grade_supersede_hygiene(store, case)
    assert score == 0.0
    assert reason is not None
    old = next(c for c in store.list_claims() if "monday" in c.text)
    new = next(c for c in store.list_claims() if "friday" in c.text)
    lifecycle.supersede(store, old_claim_id=old.id, new_claim_id=new.id, actor="t")
    assert bench.grade_supersede_hygiene(store, case)[0] == 1.0


def test_run_verifiability_stock_scores() -> None:
    report = run(1, sessions=4)
    # guard categories: the receipt path makes these perfect on the stock
    # engine; a change that surfaces unbacked content loses points here
    assert report["categories"]["receipt-coverage"]["mean"] == 1.0
    assert report["categories"]["citation-correctness"]["mean"] == 1.0
    # lever category: nothing in the no-model ingest path supersedes yet
    assert report["categories"]["supersede-hygiene"]["mean"] == 0.0


def test_pack_text_strips_highlight_markers() -> None:
    # retrieval wraps query-matched terms in guillemets; grading must see
    # the text, not the markup, or substring checks fail exactly on the
    # query-relevant claims (and highlighted forbidden values sneak past)
    pack = {"items": [{"summary": "my limit is 340 «requests» «per» «minute»"}]}
    text = bench._pack_text(pack)
    assert "340 requests per minute" in text
    case = MemoryCase("single-session-recall", "q", "340 requests per minute")
    assert grade_case(case, text)[0] == 1.0


def test_paired_verdict_floor_gatekeeps_zero_se() -> None:
    # identical per-seed diffs collapse the SE; the floor still gates
    verdict = bench.paired_verdict([0.5, 0.5, 0.5], [0.505, 0.505, 0.505])
    assert verdict["se"] == 0.0
    assert verdict["band"] == bench.DETHRONE_FLOOR
    assert not verdict["dethroned"]
    clear = bench.paired_verdict([0.5, 0.5, 0.5], [0.51, 0.51, 0.51])
    assert clear["dethroned"]


def test_paired_verdict_requires_same_seed_count() -> None:
    with pytest.raises(ValueError):
        bench.paired_verdict([0.5, 0.5], [0.5])


def test_run_seeds_threads_strategy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[object] = []

    def fake_run(seed: int, **kwargs: object) -> dict:
        seen.append(kwargs.get("strategy"))
        return {
            "seed": seed,
            "composite": 0.5,
            "categories": {name: {"mean": 0.5} for name in CATEGORIES},
        }

    monkeypatch.setattr(bench, "run", fake_run)
    sentinel = object()
    bench.run_seeds([1, 2], strategy=sentinel)
    assert seen == [sentinel, sentinel]


def test_cli_bench_run_against_requires_strategy(tmp_path: Path) -> None:
    champion = tmp_path / "champ.py"
    champion.write_text("def rank(query, candidates, *, limit):\n    return []\n")
    result = CliRunner().invoke(cli, ["bench", "run", "--against", str(champion)])
    assert result.exit_code != 0
    assert "--against requires --strategy" in result.output


# --- derivation categories (#617) -----------------------------------------

_DERIVATION_CATEGORIES = (
    "passive-consolidation",
    "multi-hop-relational",
    "temporal-depth",
    "aggregation",
)


def test_categories_include_derivation_axes() -> None:
    for name in _DERIVATION_CATEGORIES:
        assert name in CATEGORIES


def test_derivation_cases_state_no_answer_only_parts() -> None:
    """The graded fact must be derivable and never stated.

    Each derivation case carries `required` parts instead of an `expected`
    answer — if it carried an expected string the generator would have had to
    write that string into a session, which is the degeneration into
    single-session recall the category exists to avoid.
    """
    cases = [c for c in generate(11).cases if c.category in _DERIVATION_CATEGORIES]
    assert len(cases) == len(_DERIVATION_CATEGORIES)
    for case in cases:
        assert case.expected is None, case.category
        assert len(case.required) >= 3, case.category
        assert len(set(case.required)) == len(case.required), case.category


def test_derivation_parts_are_never_answerable_from_one_session() -> None:
    """Every required part is in the corpus, and no session holds them all.

    Deliberately not "one part per session": ``sub_spread`` only keeps the
    parts in distinct sessions while the session count allows, and a five-part
    case under ``sessions=4`` shares. What the category actually needs is the
    weaker property asserted here — that no single session answers the
    question on its own.
    """
    # sessions=3 crowds five parts into fewer sessions than parts, which is
    # where sub_spread stops being able to keep them distinct — the invariant
    # has to survive there, not just at the roomy default.
    for sessions in (DEFAULT_SESSIONS, 4, 3):
        dataset = generate(4, sessions=sessions)
        bodies = [text for _title, text in dataset.sessions]
        corpus = "\n".join(bodies)
        for case in dataset.cases:
            if case.category not in _DERIVATION_CATEGORIES:
                continue
            for part in case.required:
                assert part in corpus, (sessions, case.category, part)
            assert not any(
                all(part in body for part in case.required) for body in bodies
            ), f"{case.category} is answerable from one session at {sessions=}"


def test_derivation_answer_key_is_a_pure_function_of_the_seed() -> None:
    for seed in (1, 9, 23):
        first = [c for c in generate(seed).cases if c.required]
        second = [c for c in generate(seed).cases if c.required]
        assert first == second
    assert [c.required for c in generate(1).cases if c.required] != [
        c.required for c in generate(2).cases if c.required
    ]


def test_existing_categories_are_unchanged_by_derivation_cases() -> None:
    """The ten original categories must generate identically.

    The derivation block draws from a derived rng and its own pools precisely
    so recorded per-category scores stay comparable across this change. If a
    future edit advances the main rng before category 10, this fails.
    """
    original = [c for c in generate(1).cases if not c.required]
    assert [c.category for c in original] == [
        "single-session-recall",
        "multi-session",
        "knowledge-update",
        "point-in-time",
        "decoy-discrimination",
        "injection-resistance",
        "abstention",
        "citation-correctness",
        "receipt-coverage",
        "supersede-hygiene",
    ]
    # pinned against the pre-#617 generator for seed 1 — these are the values
    # the ten categories produced before the derivation block existed, so a
    # change here means the main rng stream moved and recorded scores shifted
    assert original[0].expected == "zedo-2"
    assert original[2].expected == "tenare"
    assert original[2].forbidden == ("babubo",)
    assert original[4].expected == "410 requests per minute"
    assert original[9].expected == "9:05"


def test_grade_required_all_parts_present() -> None:
    case = MemoryCase("aggregation", "list all", None, required=("alpha", "beta"))
    assert grade_case(case, "alpha and beta both surfaced") == (1.0, None)


def test_grade_required_reports_the_shortfall() -> None:
    case = MemoryCase(
        "passive-consolidation", "what is on it", None,
        required=("alpha", "beta", "gamma"),
    )
    score, reason = grade_case(case, "only alpha here")
    assert score == 0.0
    assert reason is not None
    assert "1/3 parts" in reason
    assert "'beta'" in reason and "'gamma'" in reason


def test_grade_required_is_case_insensitive() -> None:
    case = MemoryCase("temporal-depth", "history", None, required=("Alpha", "BETA"))
    assert grade_case(case, "alpha then beta") == (1.0, None)


def test_forbidden_still_zeroes_a_derivation_case() -> None:
    """The dump-guard outranks derivation: a leak zeroes before parts count."""
    case = MemoryCase(
        "aggregation", "list all", None,
        forbidden=("leaked",), required=("alpha",),
    )
    score, reason = grade_case(case, "alpha and leaked")
    assert score == 0.0
    assert reason is not None
    assert "forbidden" in reason


def test_run_scores_every_derivation_category() -> None:
    """The real pipeline produces a graded number for each new axis."""
    report = run(1, sessions=4)
    for name in _DERIVATION_CATEGORIES:
        entry = report["categories"][name]
        assert entry["n"] == 1
        assert entry["mean"] is not None
        assert 0.0 <= entry["mean"] <= 1.0


# --- composite guards (#616) ----------------------------------------------


def test_the_report_carries_a_bench_version() -> None:
    """The whole compatibility story: a guarded score must not be silently
    compared to a legacy one."""
    report = run(1, sessions=3)
    assert report["bench_version"] == bench.BENCH_VERSION
    assert bench.BENCH_VERSION >= 2


def test_composite_keeps_its_original_meaning() -> None:
    """`composite` is still the plain mean over category means.

    Every recorded score and ladder entry depends on this. The guards ride
    beside it in `composite_guarded`, never folded into it.
    """
    report = run(1, sessions=3)
    means = [
        entry["mean"] for entry in report["categories"].values()
        if entry["mean"] is not None
    ]
    assert report["composite"] == pytest.approx(
        round(sum(means) / len(means), 4), abs=1e-4
    )


def test_guarded_composite_is_the_product_of_the_guards() -> None:
    report = run(1, sessions=3)
    g = report["guards"]
    assert report["composite_guarded"] == pytest.approx(
        round(report["composite"] * g["efficiency"] * g["consistency"]
              * g["canary"], 4),
        abs=1e-4,
    )


def test_guards_are_reported_separately_not_folded_in() -> None:
    """An opaque composite is worse than no composite."""
    report = run(1, sessions=3)
    for key in ("efficiency", "consistency", "canary", "canary_leak_rate"):
        assert key in report["guards"]


# --- efficiency -----------------------------------------------------------


def test_efficiency_penalises_a_fuller_pack() -> None:
    lean = bench.efficiency_multiplier([100, 120], 2000)
    fat = bench.efficiency_multiplier([1900, 2000], 2000)
    assert lean > fat


def test_efficiency_is_bounded_below() -> None:
    """A wasteful run is penalised, never zeroed — this is a tiebreak."""
    assert bench.efficiency_multiplier([5000], 2000) == bench.GUARD_MIN_EFFICIENCY
    assert bench.efficiency_multiplier([0], 2000) == 1.0


@pytest.mark.parametrize(("used", "budget"), [([], 2000), ([100], 0)])
def test_efficiency_degrades_to_one_without_data(
    used: list[int], budget: int
) -> None:
    assert bench.efficiency_multiplier(used, budget) == 1.0


# --- consistency ----------------------------------------------------------


def test_paraphrase_zero_is_the_original() -> None:
    assert bench.paraphrase("what is the deploy day?", 0) == (
        "what is the deploy day?"
    )


def test_paraphrases_are_deterministic_and_rewrite_the_question() -> None:
    """Template-driven: an LLM rewrite would break seed reproducibility."""
    for i in range(1, 4):
        first = bench.paraphrase("what is the deploy day?", i)
        assert first == bench.paraphrase("what is the deploy day?", i)
        assert first != "what is the deploy day?"
        assert "what is the deploy day?" in first


def test_consistency_counts_agreeing_cases() -> None:
    assert bench.consistency_multiplier([True, True]) == 1.0
    assert bench.consistency_multiplier([True, False]) == 0.5
    assert bench.consistency_multiplier([False]) == 0.0
    assert bench.consistency_multiplier([]) == 1.0


# --- canary ---------------------------------------------------------------


def test_canary_halves_only_when_tripped() -> None:
    assert bench.canary_multiplier(False) == 1.0
    assert bench.canary_multiplier(True) == bench.CANARY_PENALTY


def test_every_seed_plants_a_distinct_canary() -> None:
    values = {generate(seed).canary for seed in range(1, 6)}
    assert all(values)
    assert len(values) == 5


def test_the_canary_is_in_the_corpus_but_answers_nothing() -> None:
    dataset = generate(3)
    corpus = "\n".join(text for _title, text in dataset.sessions)
    assert dataset.canary in corpus
    for case in dataset.cases:
        assert case.expected != dataset.canary
        assert dataset.canary not in (case.required or ())


def test_canary_shares_no_vocabulary_with_any_question() -> None:
    """Otherwise the guard measures the generator's word choice, not the ranker.

    A first draft read "the retired access code was ..." and tripped on
    "what was the project codename before it changed?" — lexical overlap on
    code/retired, not a dump.
    """
    import re

    dataset = generate(1)
    sentence = bench._CANARY_TEMPLATE.format(value=dataset.canary)
    bait_words = {
        w for w in re.findall(r"[a-z]{4,}", sentence.lower())
    } - {dataset.canary}
    for case in dataset.cases:
        asked = set(re.findall(r"[a-z]{4,}", case.question.lower()))
        assert not (bait_words & asked), (
            f"canary shares {bait_words & asked} with {case.question!r}"
        )


def test_canary_leak_rate_accompanies_the_binary_trip() -> None:
    """0.5 cannot distinguish one stray pack from half of them; the rate can."""
    report = run(1, sessions=3)
    g = report["guards"]
    assert 0.0 <= g["canary_leak_rate"] <= 1.0
    assert (g["canary_leak_rate"] > 0) is g["canary_tripped"]


# --- run_seeds ------------------------------------------------------------


def test_run_seeds_reports_both_composites_and_the_guards() -> None:
    report = run_seeds([1, 2], sessions=3)
    assert report["bench_version"] == bench.BENCH_VERSION
    assert "composite_mean" in report
    assert "composite_guarded_mean" in report
    assert set(report["guards"]) == {"efficiency", "consistency", "canary"}
    assert report["composite_guarded_mean"] <= report["composite_mean"]


def test_run_seeds_tolerates_a_report_without_guards(monkeypatch) -> None:
    """A bench_version 1 report has no guard block; the aggregate must not crash.

    This is the compatibility case the version stamp exists for — an old
    recorded run still aggregates, and its guards read as neutral.
    """
    def legacy_run(seed: int, **kwargs: object) -> dict:
        return {
            "seed": seed, "composite": 0.5,
            "categories": {name: {"mean": 0.5} for name in CATEGORIES},
        }

    monkeypatch.setattr(bench, "run", legacy_run)
    report = run_seeds([1, 2])
    assert report["composite_mean"] == 0.5
    assert report["composite_guarded_mean"] == 0.5
    assert report["guards"] == {
        "efficiency": 1.0, "consistency": 1.0, "canary": 1.0,
    }


def test_consistency_drops_when_a_rephrasing_changes_the_answer(
    monkeypatch
) -> None:
    """The guard's failure path.

    Stock retrieval agrees with its own paraphrases, so this forces a
    disagreement: a rewrite that retrieves nothing must cost consistency,
    which is the brittleness a category mean cannot see.
    """
    def destructive(question: str, index: int) -> str:
        return question if index <= 0 else "zzzz nonexistent query zzzz"

    monkeypatch.setattr(bench, "paraphrase", destructive)
    report = run(1, sessions=3)

    assert report["guards"]["consistency"] < 1.0
    assert report["composite_guarded"] < report["composite"]
