"""PDF and audio sources — page and timestamp receipts.

A lot of knowledge worth keeping arrives as a pdf (a spec, a paper, a contract)
or as audio (a recorded call, a voice note). Neither can be cited directly: a
receipt is the byte span ``[byte_start, byte_end)`` into a source's stored bytes
(see :mod:`vouch.receipts`), and the bytes of a pdf or an mp3 do not spell the
sentence anyone wants to quote.

The shape here keeps the receipt primitive untouched. The *extracted text* is
what gets stored and content-addressed, so byte-offset receipts keep verifying
by ``==`` with no new code path. Alongside it, a **coordinate map** records
which byte range came from which page or which point in the recording, so a
verified receipt also resolves to a real location in the original file —
"page 7" or "t=00:14:23" rather than "somewhere in the derived text".

Two constraints from the issue this implements (#613) shape the module:

* **No new hard dependency.** ``pypdf`` is an optional extra imported lazily,
  and transcription is a *configured command* (``sources.transcribe_cmd``), the
  same deployment-config pattern as ``compile.llm_cmd``. Neither is imported or
  invoked unless a caller actually registers that kind of file.
* **Fail loudly, never silently degrade.** A scanned pdf with no text layer is
  out of scope; it raises rather than returning an empty transcript or reaching
  for OCR. Knowledge that silently became empty is worse than knowledge that
  refused to enter.

The original binary is not thrown away conceptually: its sha256 is recorded in
``metadata['origin_sha256']`` so :mod:`vouch.verify` can re-check the pdf or the
recording for drift long after extraction. Extracting to text by hand loses
exactly that link, which is why doing it inside vouch is worth the module.
"""

from __future__ import annotations

import hashlib
import io
import mimetypes
import re
import shlex
import subprocess
import tempfile
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from .config_coerce import coerce_numeric
from .models import Source

if TYPE_CHECKING:
    from .storage import KBStore

DEFAULT_TRANSCRIBE_TIMEOUT_SECONDS = 600.0

# Pages are joined by a blank line so the stored text reads as a document
# rather than one run-on paragraph. The separator sits *between* page spans and
# belongs to no page, which is why a byte offset landing in it resolves to no
# coordinate rather than to an arbitrary neighbour.
PAGE_SEPARATOR = "\n\n"
CUE_SEPARATOR = "\n"

PDF_EXTENSIONS = frozenset({".pdf"})
AUDIO_EXTENSIONS = frozenset(
    {".mp3", ".m4a", ".wav", ".flac", ".ogg", ".opus", ".aac", ".wma", ".aiff"}
)


class MediaError(ValueError):
    """Media could not become a citable source (extraction, config, or shape)."""


class CoordinateKind(StrEnum):
    PAGE = "page"
    TIMESTAMP = "timestamp"


class MediaKind(StrEnum):
    PDF = "pdf"
    AUDIO = "audio"


@dataclass(frozen=True)
class Segment:
    """One page's, or one cue's, byte span in the extracted text.

    ``label`` is the coordinate in the *original* file: a 1-indexed page number
    for pdfs, a start offset in seconds for audio. It is carried as a string so
    the stored map round-trips through yaml without float formatting surprises.
    """

    byte_start: int
    byte_end: int
    label: str


@dataclass(frozen=True)
class Cue:
    """A transcript cue: text plus where in the recording it starts."""

    start_seconds: float
    text: str


@dataclass(frozen=True)
class MediaConfig:
    transcribe_cmd: str | None = None
    timeout_seconds: float = DEFAULT_TRANSCRIBE_TIMEOUT_SECONDS


def load_config(store: KBStore) -> MediaConfig:
    """Read ``sources:`` from config.yaml; fall back to defaults.

    Same defensive shape as ``compile.load_config``: an unreadable or
    non-mapping config is not an error here, it just means nothing is
    configured, and the transcribe path reports that when it is actually used.
    """
    try:
        loaded = yaml.safe_load(store.config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return MediaConfig()
    if not isinstance(loaded, dict):
        return MediaConfig()
    raw = loaded.get("sources")
    if not isinstance(raw, dict):
        return MediaConfig()
    cmd = raw.get("transcribe_cmd")
    return MediaConfig(
        transcribe_cmd=str(cmd) if cmd else None,
        timeout_seconds=coerce_numeric(
            raw.get("transcribe_timeout_seconds", DEFAULT_TRANSCRIBE_TIMEOUT_SECONDS),
            DEFAULT_TRANSCRIBE_TIMEOUT_SECONDS,
            float,
        ),
    )


def media_kind(path: Path) -> MediaKind | None:
    """Which media pipeline ``path`` belongs to, or None for ordinary files."""
    ext = path.suffix.lower()
    if ext in PDF_EXTENSIONS:
        return MediaKind.PDF
    if ext in AUDIO_EXTENSIONS:
        return MediaKind.AUDIO
    return None


# --- assembly -------------------------------------------------------------


def assemble_pages(pages: list[str]) -> tuple[bytes, list[Segment]]:
    """Join page texts into the stored artifact, recording each page's span.

    Offsets are byte offsets, not character offsets, because that is the unit
    the receipt indexes — under utf-8 the two diverge at the first multi-byte
    codepoint, and a pdf is exactly where an em-dash shows up.
    """
    chunks: list[str] = []
    segments: list[Segment] = []
    cursor = 0
    for number, text in enumerate(pages, start=1):
        if chunks:
            cursor += len(PAGE_SEPARATOR.encode("utf-8"))
            chunks.append(PAGE_SEPARATOR)
        size = len(text.encode("utf-8"))
        segments.append(Segment(cursor, cursor + size, str(number)))
        chunks.append(text)
        cursor += size
    return "".join(chunks).encode("utf-8"), segments


def assemble_cues(cues: list[Cue]) -> tuple[bytes, list[Segment]]:
    """Join cue texts into the stored transcript, recording each cue's span."""
    chunks: list[str] = []
    segments: list[Segment] = []
    cursor = 0
    for cue in cues:
        if chunks:
            cursor += len(CUE_SEPARATOR.encode("utf-8"))
            chunks.append(CUE_SEPARATOR)
        size = len(cue.text.encode("utf-8"))
        segments.append(Segment(cursor, cursor + size, _format_seconds(cue.start_seconds)))
        chunks.append(cue.text)
        cursor += size
    return "".join(chunks).encode("utf-8"), segments


def _format_seconds(seconds: float) -> str:
    """Seconds as a plain decimal string — no trailing ``.0`` noise."""
    rounded = round(seconds, 3)
    return str(int(rounded)) if rounded == int(rounded) else str(rounded)


# --- coordinate map -------------------------------------------------------


def coordinate_map(kind: CoordinateKind, segments: list[Segment]) -> dict[str, Any]:
    """The map stored on ``Source.metadata['coordinates']``.

    Plain lists of scalars, so it diffs readably in a PR like everything else
    under ``.vouch/`` — the same reason claims are yaml and not a binary index.
    """
    return {
        "kind": kind.value,
        "segments": [
            {"start": s.byte_start, "end": s.byte_end, "label": s.label} for s in segments
        ],
    }


def _parse_segments(coordinates: dict[str, Any]) -> list[tuple[int, int, str]]:
    """Read a stored map back defensively — a malformed row is skipped, not fatal."""
    rows = coordinates.get("segments")
    if not isinstance(rows, list):
        return []
    out: list[tuple[int, int, str]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        start, end, label = row.get("start"), row.get("end"), row.get("label")
        if isinstance(start, int) and isinstance(end, int) and label is not None:
            out.append((start, end, str(label)))
    return out


def format_timestamp(seconds: float) -> str:
    """``HH:MM:SS`` — the form ``Evidence.locator`` already documents for audio."""
    total = int(seconds)
    return f"{total // 3600:02d}:{(total % 3600) // 60:02d}:{total % 60:02d}"


def resolve_coordinate(coordinates: dict[str, Any], byte_offset: int) -> str | None:
    """Where in the original file ``byte_offset`` of the stored text came from.

    Returns ``p7`` for pdfs and ``t=00:14:23`` for audio, or None when the
    offset falls outside every recorded span (a page separator, or a map that
    does not cover the text).
    """
    kind = coordinates.get("kind")
    for start, end, label in _parse_segments(coordinates):
        if start <= byte_offset < end:
            if kind == CoordinateKind.TIMESTAMP.value:
                return f"t={format_timestamp(float(label))}"
            return f"p{label}"
    return None


def locator_for_span(coordinates: dict[str, Any], start: int, end: int) -> str:
    """The ``Evidence.locator`` for a receipt span, enriched with its coordinate.

    Always carries the byte span, because that is the part that verifies; the
    page or timestamp prefix is what makes it resolvable by a human holding the
    original pdf or recording.
    """
    span = f"b{start}-{end}"
    coordinate = resolve_coordinate(coordinates, start)
    return f"{coordinate}@{span}" if coordinate else span


# --- extraction -----------------------------------------------------------


def extract_pdf_pages(data: bytes) -> list[str]:
    """Text layer of ``data``, one string per page.

    Raises rather than reaching for OCR when there is no text layer: a scanned
    pdf is explicitly out of scope, and silently registering an empty source
    would produce a citable artifact that cites nothing.
    """
    try:
        from pypdf import PdfReader
    except ImportError as e:
        raise MediaError(
            "pdf support needs the optional extra — pip install 'vouch-kb[pdf]'"
        ) from e

    try:
        reader = PdfReader(io.BytesIO(data))
        pages = [(page.extract_text() or "").strip() for page in reader.pages]
    # pypdf raises a wide, version-dependent set (its own errors plus whatever
    # the underlying stream raises), so the catch is deliberately broad: a
    # malformed pdf must surface as a MediaError, never as a stray traceback.
    except Exception as e:
        raise MediaError(f"could not read pdf: {e}") from e
    if not any(pages):
        raise MediaError(
            "pdf has no text layer (scanned?) — out of scope, vouch does not ocr"
        )
    return pages


_CUE_TIME_RE = re.compile(
    r"(?P<h>\d{1,2}:)?(?P<m>\d{1,2}):(?P<s>\d{1,2})(?:[.,](?P<ms>\d{1,3}))?\s*-->"
)


def _cue_start_seconds(match: re.Match[str]) -> float:
    hours = int((match.group("h") or "0:")[:-1])
    seconds = hours * 3600 + int(match.group("m")) * 60 + int(match.group("s"))
    return seconds + int((match.group("ms") or "0").ljust(3, "0")) / 1000


def parse_cues(raw: str) -> list[Cue]:
    """Parse WebVTT or SubRip output into cues.

    Both formats are a timing line (``00:00:04.000 --> 00:00:07.000``) followed
    by text lines, which is all this needs; accepting both means the configured
    command can be whisper, whisper.cpp, or anything else that speaks either.
    Cues with no text are dropped — they would contribute an empty span that no
    quote can ever land in.
    """
    cues: list[Cue] = []
    start: float | None = None
    lines: list[str] = []

    def flush() -> None:
        if start is not None and lines:
            cues.append(Cue(start, " ".join(lines)))

    for line in raw.splitlines():
        stripped = line.strip()
        match = _CUE_TIME_RE.match(stripped)
        if match:
            flush()
            start, lines = _cue_start_seconds(match), []
            continue
        if not stripped:
            flush()
            start, lines = None, []
            continue
        if start is not None:
            lines.append(stripped)
    flush()

    if not cues:
        raise MediaError(
            "transcription produced no timed cues — expected webvtt or srt output"
        )
    return cues


def transcribe(path: Path, cmd: str, *, timeout_seconds: float) -> str:
    """Run the configured transcription command over ``path`` and return stdout.

    Deployment config, not a baked model dependency: vouch never chooses a
    speech model. ``{path}`` in the command is substituted with the shell-quoted
    absolute path, and appended when the placeholder is absent. Runs in a
    throwaway cwd for the same reason ``llm_draft.run_llm`` does — a CLI that
    discovers per-project hooks from its cwd should not fire this project's own
    pipeline while transcribing for it.
    """
    quoted = shlex.quote(str(path))
    line = cmd.replace("{path}", quoted) if "{path}" in cmd else f"{cmd} {quoted}"
    with tempfile.TemporaryDirectory(prefix="vouch-transcribe-") as tmp:
        try:
            proc = subprocess.run(
                line, shell=True, cwd=tmp, capture_output=True, text=True,
                encoding="utf-8", errors="replace", timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired as e:
            raise MediaError(
                f"sources.transcribe_cmd timed out after {timeout_seconds:.0f}s"
            ) from e
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()[:400]
        raise MediaError(f"sources.transcribe_cmd failed ({proc.returncode}): {detail}")
    return proc.stdout


# --- registration ---------------------------------------------------------


def _origin_metadata(path: Path, data: bytes, kind: MediaKind) -> dict[str, Any]:
    """Provenance back to the binary the text was extracted from.

    ``origin_sha256`` is what lets ``vouch source verify`` re-check the pdf or
    the recording later. Without it the extracted text is unmoored from the
    thing it came from, which is the whole failure mode of extracting by hand.
    """
    guessed, _ = mimetypes.guess_type(path.name)
    return {
        "origin_sha256": hashlib.sha256(data).hexdigest(),
        "origin_bytes": len(data),
        "origin_media_type": guessed or f"application/{kind.value}",
        "origin_filename": path.name,
    }


def register_media_source(
    store: KBStore,
    path: Path,
    *,
    kind: MediaKind | None = None,
    title: str | None = None,
    transcribe_cmd: str | None = None,
    timeout_seconds: float | None = None,
    tags: list[str] | None = None,
) -> Source:
    """Extract ``path`` to text and register it as an ordinary Source.

    The returned Source is content-addressed on the *extracted text*, so every
    existing path — ingest, the receipt gate, ``kb.source_verify``, receipt
    coverage — works on it unchanged. What is new is ``metadata['coordinates']``,
    which maps the stored bytes back to pages or timestamps.

    The caller is trusted with the path: this is reached from the CLI, where the
    human already has filesystem access. It is deliberately not wired to the
    remote MCP/JSONL ``register_source_from_path`` surface, which confines reads
    to the project root for exactly that reason.
    """
    resolved = path.resolve()
    detected = kind or media_kind(resolved)
    if detected is None:
        raise MediaError(f"not a supported media file: {resolved.name}")
    try:
        data = resolved.read_bytes()
    except OSError as e:
        raise MediaError(f"could not read {resolved}: {e}") from e

    if detected is MediaKind.PDF:
        content, segments = assemble_pages(extract_pdf_pages(data))
        coordinates = coordinate_map(CoordinateKind.PAGE, segments)
    else:
        cfg = load_config(store)
        cmd = transcribe_cmd or cfg.transcribe_cmd
        if not cmd:
            raise MediaError(
                "sources.transcribe_cmd is not configured — set it in "
                ".vouch/config.yaml, e.g.\nsources:\n  transcribe_cmd: "
                '"whisper --output_format vtt --output_dir - {path}"'
            )
        raw = transcribe(
            resolved, cmd,
            timeout_seconds=timeout_seconds or cfg.timeout_seconds,
        )
        content, segments = assemble_cues(parse_cues(raw))
        coordinates = coordinate_map(CoordinateKind.TIMESTAMP, segments)

    metadata = _origin_metadata(resolved, data, detected)
    metadata["coordinates"] = coordinates
    return store.put_source(
        content,
        title=title or resolved.name,
        locator=str(resolved),
        source_type=detected.value,
        media_type="text/plain",
        tags=tags,
        metadata=metadata,
    )


def source_coordinates(source: Source) -> dict[str, Any] | None:
    """The coordinate map on ``source``, or None when it carries none."""
    coordinates = source.metadata.get("coordinates")
    return coordinates if isinstance(coordinates, dict) else None
