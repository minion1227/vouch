"""PDF and audio sources — page and timestamp receipts (#613).

The property under test throughout: a media source stores *extracted text*, so
the byte-offset receipt keeps verifying by ``==`` exactly as it does for a text
file, while a coordinate map resolves that same span back to a page number or a
point in the recording.

No test needs pypdf installed or a speech model on PATH. The pdf reader is
injected as a fake module and the transcription command is a shell one-liner,
which is the point of both being optional: the pipeline is exercised without
either dependency existing.
"""

from __future__ import annotations

import hashlib
import sys
import types
from pathlib import Path

import pytest
from click.testing import CliRunner

from vouch import media
from vouch.cli import cli
from vouch.media import (
    CoordinateKind,
    Cue,
    MediaError,
    MediaKind,
    Segment,
    assemble_cues,
    assemble_pages,
    coordinate_map,
    extract_pdf_pages,
    format_timestamp,
    load_config,
    locator_for_span,
    media_kind,
    parse_cues,
    register_media_source,
    resolve_coordinate,
    source_coordinates,
    transcribe,
)
from vouch.receipts import ReceiptStatus, receipt_for_quote, verify_receipt
from vouch.storage import KBStore
from vouch.verify import verify_source

VTT = """WEBVTT

00:00:00.000 --> 00:00:04.000
the migration ran clean

00:01:03.500 --> 00:01:07.000
rollback took eleven minutes
"""

SRT = """1
00:00:02,000 --> 00:00:05,000
first cue

2
01:00:00,000 --> 01:00:04,000
an hour in
"""


@pytest.fixture
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> KBStore:
    s = KBStore.init(tmp_path)
    monkeypatch.chdir(s.root)
    return s


def _fake_pypdf(monkeypatch: pytest.MonkeyPatch, pages: list[str] | None, *, boom: bool = False):
    """Install a stand-in ``pypdf`` so the pdf path runs without the extra."""
    module = types.ModuleType("pypdf")

    class _Page:
        def __init__(self, text: str) -> None:
            self._text = text

        def extract_text(self) -> str | None:
            return self._text

    class _Reader:
        def __init__(self, _stream: object) -> None:
            if boom:
                raise RuntimeError("not a pdf")
            self.pages = [_Page(p) for p in (pages or [])]

    module.PdfReader = _Reader  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pypdf", module)


# --- kind detection -------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("spec.pdf", MediaKind.PDF),
        ("SPEC.PDF", MediaKind.PDF),
        ("call.mp3", MediaKind.AUDIO),
        ("call.M4A", MediaKind.AUDIO),
        ("notes.md", None),
        ("noext", None),
    ],
)
def test_media_kind_detection(name: str, expected: MediaKind | None) -> None:
    assert media_kind(Path(name)) is expected


# --- assembly -------------------------------------------------------------


def test_assemble_pages_records_byte_spans() -> None:
    content, segments = assemble_pages(["alpha", "beta"])
    assert content == b"alpha\n\nbeta"
    assert segments == [Segment(0, 5, "1"), Segment(7, 11, "2")]
    assert content[segments[1].byte_start : segments[1].byte_end] == b"beta"


def test_assemble_pages_offsets_are_bytes_not_characters() -> None:
    """An em-dash is 3 bytes; a character index would put page 2 in the wrong place."""
    content, segments = assemble_pages(["a—b", "second"])
    assert segments[0].byte_end == 5
    assert content[segments[1].byte_start : segments[1].byte_end].decode() == "second"


def test_assemble_pages_empty() -> None:
    assert assemble_pages([]) == (b"", [])


def test_assemble_cues_records_spans_and_labels() -> None:
    content, segments = assemble_cues([Cue(0.0, "one"), Cue(63.5, "two")])
    assert content == b"one\ntwo"
    assert [s.label for s in segments] == ["0", "63.5"]
    assert content[segments[1].byte_start : segments[1].byte_end] == b"two"


# --- coordinate map -------------------------------------------------------


def test_coordinate_map_is_plain_scalars() -> None:
    cmap = coordinate_map(CoordinateKind.PAGE, [Segment(0, 5, "1")])
    assert cmap == {"kind": "page", "segments": [{"start": 0, "end": 5, "label": "1"}]}


def test_resolve_coordinate_pages_and_gap() -> None:
    _content, segments = assemble_pages(["alpha", "beta"])
    cmap = coordinate_map(CoordinateKind.PAGE, segments)
    assert resolve_coordinate(cmap, 0) == "p1"
    assert resolve_coordinate(cmap, 8) == "p2"
    # the separator belongs to no page
    assert resolve_coordinate(cmap, 5) is None
    assert resolve_coordinate(cmap, 999) is None


def test_resolve_coordinate_timestamps() -> None:
    _content, segments = assemble_cues([Cue(0.0, "one"), Cue(3723.0, "two")])
    cmap = coordinate_map(CoordinateKind.TIMESTAMP, segments)
    assert resolve_coordinate(cmap, 0) == "t=00:00:00"
    assert resolve_coordinate(cmap, 4) == "t=01:02:03"


@pytest.mark.parametrize(
    "cmap",
    [
        {"kind": "page", "segments": "not-a-list"},
        {"kind": "page", "segments": ["not-a-dict"]},
        {"kind": "page", "segments": [{"start": "x", "end": 5, "label": "1"}]},
        {"kind": "page", "segments": [{"start": 0, "end": 5}]},
        {},
    ],
)
def test_resolve_coordinate_survives_malformed_maps(cmap: dict[str, object]) -> None:
    """A corrupt map degrades to 'no coordinate', never to an exception."""
    assert resolve_coordinate(cmap, 0) is None


def test_format_timestamp() -> None:
    assert format_timestamp(0) == "00:00:00"
    assert format_timestamp(863.9) == "00:14:23"
    assert format_timestamp(3661) == "01:01:01"


def test_locator_for_span_with_and_without_coordinate() -> None:
    _content, segments = assemble_pages(["alpha", "beta"])
    cmap = coordinate_map(CoordinateKind.PAGE, segments)
    assert locator_for_span(cmap, 7, 11) == "p2@b7-11"
    # offset in the separator: still a valid receipt, just no page to name
    assert locator_for_span(cmap, 5, 6) == "b5-6"


# --- cue parsing ----------------------------------------------------------


def test_parse_cues_webvtt() -> None:
    cues = parse_cues(VTT)
    assert [c.start_seconds for c in cues] == [0.0, 63.5]
    assert cues[1].text == "rollback took eleven minutes"


def test_parse_cues_srt_with_hours() -> None:
    cues = parse_cues(SRT)
    assert [c.start_seconds for c in cues] == [2.0, 3600.0]
    assert cues[0].text == "first cue"


def test_parse_cues_joins_wrapped_lines() -> None:
    cues = parse_cues("00:00:01.000 --> 00:00:02.000\nwrapped\nover two lines\n")
    assert cues[0].text == "wrapped over two lines"


def test_parse_cues_drops_empty_cue() -> None:
    raw = "00:00:01.000 --> 00:00:02.000\n\n00:00:03.000 --> 00:00:04.000\nreal\n"
    cues = parse_cues(raw)
    assert [c.text for c in cues] == ["real"]


def test_parse_cues_without_timings_is_an_error() -> None:
    with pytest.raises(MediaError, match="no timed cues"):
        parse_cues("just a plain transcript with no timings\n")


# --- pdf extraction -------------------------------------------------------


def test_extract_pdf_pages_reads_text_layer(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_pypdf(monkeypatch, ["  page one  ", "page two"])
    assert extract_pdf_pages(b"%PDF-fake") == ["page one", "page two"]


def test_extract_pdf_pages_without_the_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "pypdf", None)
    with pytest.raises(MediaError, match=r"vouch-kb\[pdf\]"):
        extract_pdf_pages(b"%PDF-fake")


def test_extract_pdf_pages_unreadable_pdf(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_pypdf(monkeypatch, None, boom=True)
    with pytest.raises(MediaError, match="could not read pdf"):
        extract_pdf_pages(b"not-a-pdf")


def test_extract_pdf_pages_refuses_scanned_pdf(monkeypatch: pytest.MonkeyPatch) -> None:
    """No text layer must fail loudly — never silently register an empty source."""
    _fake_pypdf(monkeypatch, ["", "   "])
    with pytest.raises(MediaError, match="no text layer"):
        extract_pdf_pages(b"%PDF-scan")


# --- transcription command ------------------------------------------------


def test_transcribe_substitutes_path_placeholder(tmp_path: Path) -> None:
    audio = tmp_path / "call.mp3"
    audio.write_bytes(b"\x00")
    out = transcribe(audio, "printf %s {path}", timeout_seconds=30)
    assert out == str(audio)


def test_transcribe_appends_path_when_no_placeholder(tmp_path: Path) -> None:
    audio = tmp_path / "call.mp3"
    audio.write_bytes(b"\x00")
    assert transcribe(audio, "printf %s", timeout_seconds=30).endswith("call.mp3")


def test_transcribe_reports_command_failure(tmp_path: Path) -> None:
    audio = tmp_path / "call.mp3"
    audio.write_bytes(b"\x00")
    with pytest.raises(MediaError, match="transcribe_cmd failed"):
        transcribe(audio, "echo boom >&2; exit 3", timeout_seconds=30)


def test_transcribe_times_out(tmp_path: Path) -> None:
    audio = tmp_path / "call.mp3"
    audio.write_bytes(b"\x00")
    with pytest.raises(MediaError, match="timed out"):
        transcribe(audio, "sleep 5; : {path}", timeout_seconds=0.2)


# --- config ---------------------------------------------------------------


def test_load_config_defaults_when_unset(store: KBStore) -> None:
    cfg = load_config(store)
    assert cfg.transcribe_cmd is None
    assert cfg.timeout_seconds == media.DEFAULT_TRANSCRIBE_TIMEOUT_SECONDS


def test_load_config_reads_sources_block(store: KBStore) -> None:
    store.config_path.write_text(
        "sources:\n  transcribe_cmd: whisper {path}\n  transcribe_timeout_seconds: 30\n",
        encoding="utf-8",
    )
    cfg = load_config(store)
    assert cfg.transcribe_cmd == "whisper {path}"
    assert cfg.timeout_seconds == 30.0


def test_load_config_coerces_a_bad_timeout(store: KBStore) -> None:
    """A string timeout must not crash registration — it falls back to the default."""
    store.config_path.write_text(
        "sources:\n  transcribe_cmd: whisper\n  transcribe_timeout_seconds: soon\n",
        encoding="utf-8",
    )
    assert load_config(store).timeout_seconds == media.DEFAULT_TRANSCRIBE_TIMEOUT_SECONDS


@pytest.mark.parametrize("body", ["", "just a string\n", "sources: nope\n"])
def test_load_config_survives_odd_config(store: KBStore, body: str) -> None:
    store.config_path.write_text(body, encoding="utf-8")
    assert load_config(store).transcribe_cmd is None


def test_load_config_survives_unreadable_config(store: KBStore) -> None:
    store.config_path.unlink()
    assert load_config(store).transcribe_cmd is None


# --- registration ---------------------------------------------------------


def test_register_pdf_source_end_to_end(
    store: KBStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_pypdf(monkeypatch, ["cover page", "the rollback took eleven minutes"])
    pdf = tmp_path / "postmortem.pdf"
    pdf.write_bytes(b"%PDF-fake")

    src = register_media_source(store, pdf)

    assert src.type.value == "pdf"
    assert src.media_type == "text/plain"
    assert store.read_source_content(src.id) == b"cover page\n\nthe rollback took eleven minutes"
    assert src.metadata["origin_sha256"] == hashlib.sha256(b"%PDF-fake").hexdigest()
    assert src.metadata["origin_filename"] == "postmortem.pdf"
    assert src.metadata["coordinates"]["kind"] == "page"


def test_pdf_quote_earns_a_receipt_carrying_its_page(
    store: KBStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The whole point: a verified receipt that also says which page."""
    _fake_pypdf(monkeypatch, ["cover page", "the rollback took eleven minutes"])
    pdf = tmp_path / "postmortem.pdf"
    pdf.write_bytes(b"%PDF-fake")
    src = register_media_source(store, pdf)
    content = store.read_source_content(src.id)

    ev = receipt_for_quote(
        source_id=src.id,
        source_bytes=content,
        quote="rollback took eleven minutes",
        coordinates=source_coordinates(src),
    )

    assert ev is not None
    assert ev.locator.startswith("p2@b")
    assert verify_receipt(ev, content).status is ReceiptStatus.VERIFIED


def test_register_audio_source_end_to_end(store: KBStore, tmp_path: Path) -> None:
    audio = tmp_path / "standup.mp3"
    audio.write_bytes(b"\x00\x01")
    vtt = VTT.replace("\n", "\\n")

    src = register_media_source(store, audio, transcribe_cmd=f"printf '{vtt}'")

    assert src.type.value == "audio"
    content = store.read_source_content(src.id)
    assert content == b"the migration ran clean\nrollback took eleven minutes"
    ev = receipt_for_quote(
        source_id=src.id,
        source_bytes=content,
        quote="rollback took eleven minutes",
        coordinates=source_coordinates(src),
    )
    assert ev is not None
    assert ev.locator.startswith("t=00:01:03@b")
    assert verify_receipt(ev, content).status is ReceiptStatus.VERIFIED


def test_register_audio_uses_configured_command(store: KBStore, tmp_path: Path) -> None:
    audio = tmp_path / "standup.mp3"
    audio.write_bytes(b"\x00")
    store.config_path.write_text(
        "sources:\n  transcribe_cmd: \"printf '00:00:01.000 --> 00:00:02.000\\\\nhello'\"\n",
        encoding="utf-8",
    )
    src = register_media_source(store, audio)
    assert store.read_source_content(src.id) == b"hello"


def test_register_audio_without_a_configured_command(store: KBStore, tmp_path: Path) -> None:
    audio = tmp_path / "standup.mp3"
    audio.write_bytes(b"\x00")
    with pytest.raises(MediaError, match=r"sources\.transcribe_cmd is not configured"):
        register_media_source(store, audio)


def test_register_rejects_unsupported_file(store: KBStore, tmp_path: Path) -> None:
    notes = tmp_path / "notes.md"
    notes.write_text("plain", encoding="utf-8")
    with pytest.raises(MediaError, match="not a supported media file"):
        register_media_source(store, notes)


def test_register_reports_unreadable_file(store: KBStore, tmp_path: Path) -> None:
    with pytest.raises(MediaError, match="could not read"):
        register_media_source(store, tmp_path / "missing.pdf", kind=MediaKind.PDF)


def test_source_coordinates_absent_on_ordinary_source(store: KBStore) -> None:
    src = store.put_source(b"plain text", title="notes.txt")
    assert source_coordinates(src) is None


# --- drift detection ------------------------------------------------------


def test_verify_rechecks_the_original_pdf(
    store: KBStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Extraction must not sever the link back to the file it came from."""
    _fake_pypdf(monkeypatch, ["contract text"])
    pdf = store.root / "contract.pdf"
    pdf.write_bytes(b"%PDF-one")
    src = register_media_source(store, pdf)

    result = verify_source(store, src)
    assert result.stored_ok is True
    assert result.external_status == "match"

    pdf.write_bytes(b"%PDF-two")
    assert verify_source(store, src).external_status == "drift"


def test_verify_reports_a_missing_original(
    store: KBStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_pypdf(monkeypatch, ["contract text"])
    pdf = store.root / "contract.pdf"
    pdf.write_bytes(b"%PDF-one")
    src = register_media_source(store, pdf)
    pdf.unlink()
    assert verify_source(store, src).external_status == "missing"


# --- cli surface ----------------------------------------------------------


def test_cli_source_add_extracts_a_pdf(
    store: KBStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_pypdf(monkeypatch, ["cover", "body text"])
    pdf = store.root / "spec.pdf"
    pdf.write_bytes(b"%PDF-fake")

    result = CliRunner().invoke(cli, ["source", "add", str(pdf)])

    assert result.exit_code == 0, result.output
    src = store.get_source(result.output.strip())
    assert src.type.value == "pdf"
    assert store.read_source_content(src.id) == b"cover\n\nbody text"


def test_cli_source_add_raw_keeps_the_bytes(store: KBStore) -> None:
    pdf = store.root / "spec.pdf"
    pdf.write_bytes(b"%PDF-fake")

    result = CliRunner().invoke(cli, ["source", "add", str(pdf), "--raw"])

    assert result.exit_code == 0, result.output
    src = store.get_source(result.output.strip())
    assert src.type.value == "file"
    assert store.read_source_content(src.id) == b"%PDF-fake"


def test_cli_source_add_surfaces_a_media_error(store: KBStore) -> None:
    audio = store.root / "call.mp3"
    audio.write_bytes(b"\x00")
    result = CliRunner().invoke(cli, ["source", "add", str(audio)])
    assert result.exit_code != 0


def test_cli_source_locate_prints_the_page(
    store: KBStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_pypdf(monkeypatch, ["cover", "body text"])
    pdf = store.root / "spec.pdf"
    pdf.write_bytes(b"%PDF-fake")
    sid = CliRunner().invoke(cli, ["source", "add", str(pdf)]).output.strip()

    result = CliRunner().invoke(cli, ["source", "locate", sid, "body text"])

    assert result.exit_code == 0, result.output
    assert result.output.strip() == "p2@b7-16"


def test_cli_source_locate_on_a_plain_source(store: KBStore) -> None:
    src = store.put_source(b"plain text", title="notes.txt")
    result = CliRunner().invoke(cli, ["source", "locate", src.id, "text"])
    assert result.exit_code == 0, result.output
    assert result.output.strip() == "b6-10"


def test_cli_source_locate_rejects_a_paraphrase(store: KBStore) -> None:
    src = store.put_source(b"plain text", title="notes.txt")
    result = CliRunner().invoke(cli, ["source", "locate", src.id, "something else"])
    assert result.exit_code == 1
    assert "not found verbatim" in result.output
