"""Name cleaning and validation helpers shared by loaders and persistence."""

import re

MIN_NAME_LENGTH = 2

_HEADER_NAMES = {
    "name",
    "navn",
    "fornavn",
    "firstname",
    "køn",
    "gender",
    "kjønn",
}

_PLACEHOLDER_PATTERNS = (
    re.compile(r"^name\s*\d+$", re.IGNORECASE),
    re.compile(r"^navn\s*\d+$", re.IGNORECASE),
    re.compile(r"^fornavn\s*\d+$", re.IGNORECASE),
)


def strip_name_notes(name: str) -> str:
    """Strip note suffixes from raw name strings."""
    if not isinstance(name, str):
        return ""
    return name.split(" - ", 1)[0].strip()


def is_valid_name(name: str) -> bool:
    """Return whether a value is a real name rather than a header or placeholder."""
    if not name or not isinstance(name, str):
        return False

    name_lower = name.strip().lower()
    if name_lower in _HEADER_NAMES:
        return False

    if any(pattern.match(name_lower) for pattern in _PLACEHOLDER_PATTERNS):
        return False

    return len(name_lower) >= MIN_NAME_LENGTH
