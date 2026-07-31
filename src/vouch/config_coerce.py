"""shared config-value coercion: booleans and numbers.

`bool(x)` on an arbitrary config value is a trap: `yaml.safe_load` resolves
an *unquoted* `true`/`false` to a real Python bool, but a mistakenly-quoted
`"false"` in config.yaml stays a non-empty string, and `bool("false")` is
True. A dozen `load_config()` functions across this codebase each read a
boolean flag out of raw YAML; before this module they either repeated the
same `bool(raw.get(...))` bug (compile.py's `two_phase`, and the `enabled`
flag in session_split/admission/inbox/recall/capture/volunteer_context) or
raised hard on a bad value (install_adapter.py's install.yaml flags, which
is the right call for a one-time CLI install but not for config read on
every session/render).

`coerce_bool()` is the fail-soft half of that spectrum: recognized string
spellings are parsed case-insensitively, anything else (wrong type, typo,
unrecognized spelling) degrades to the caller's default rather than taking
down every caller on a config mistake.

`coerce_numeric()` is the same posture for `int`/`float` fields: a bare
`int(raw.get(...))` raises outright on a typo'd value (`max_chars: "12,000"`,
`min_observations: "three"`), and several `load_config()` functions are
called from hook-driven paths (SessionStart recall injection, session-split
capture) where that exception takes down the hook on every session until a
human edits config.yaml by hand.
"""

from __future__ import annotations

from typing import Any, TypeVar

_TRUE_STRINGS = frozenset({"true", "yes", "on", "1"})
_FALSE_STRINGS = frozenset({"false", "no", "off", "0"})

_N = TypeVar("_N", int, float)


def coerce_bool(value: Any, default: bool) -> bool:
    """parse a YAML-sourced config boolean, defaulting on anything unrecognized."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in _TRUE_STRINGS:
            return True
        if lowered in _FALSE_STRINGS:
            return False
    return default


def coerce_numeric(value: Any, default: _N, cast: type[_N]) -> _N:
    """parse a YAML-sourced config number, defaulting on anything uncastable."""
    try:
        return cast(value)
    except (TypeError, ValueError):
        return default
