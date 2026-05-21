"""Pure helpers for the binary name-filter workflow."""

import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class FilterCounts:
    """Counts for the three filter states shown in the UI."""

    not_decided: int
    included: int
    excluded: int


def load_name_inclusions_json(inclusions_json: str) -> dict[str, bool]:
    """Decode a stored include/exclude map only when every entry has the expected shape."""
    try:
        decoded: Any = json.loads(inclusions_json)
    except (json.JSONDecodeError, TypeError):
        return {}

    if not isinstance(decoded, dict):
        return {}

    inclusions: dict[str, bool] = {}
    for name, status in decoded.items():
        if not isinstance(name, str) or not isinstance(status, bool):
            return {}
        inclusions[name] = status
    return inclusions


def count_filter_statuses(names: list[str], inclusions: dict[str, bool]) -> FilterCounts:
    """Count names by explicit include/exclude state."""
    not_decided = included = excluded = 0
    for name in names:
        status = inclusions.get(name)
        if status is None:
            not_decided += 1
        elif status is True:
            included += 1
        else:
            excluded += 1
    return FilterCounts(not_decided=not_decided, included=included, excluded=excluded)


def apply_filter_count_transition(
    counts: FilterCounts,
    *,
    old_status: bool | None,
    new_status: bool | None,
) -> FilterCounts:
    """Return updated counts after one name changes filter state."""
    not_decided = counts.not_decided
    included = counts.included
    excluded = counts.excluded

    if old_status is None:
        not_decided -= 1
    elif old_status is True:
        included -= 1
    else:
        excluded -= 1

    if new_status is None:
        not_decided += 1
    elif new_status is True:
        included += 1
    else:
        excluded += 1

    return FilterCounts(not_decided=not_decided, included=included, excluded=excluded)


def get_undecided_names(names: list[str], inclusions: dict[str, bool]) -> list[str]:
    """Return names without an explicit include/exclude decision."""
    return [name for name in names if name not in inclusions]


def get_included_names(names: list[str], inclusions: dict[str, bool]) -> list[str]:
    """Return names explicitly included for the tournament."""
    return [name for name in names if inclusions.get(name) is True]


def get_excluded_names(names: list[str], inclusions: dict[str, bool]) -> list[str]:
    """Return names explicitly excluded from the tournament."""
    return [name for name in names if inclusions.get(name) is False]


def set_many_filter_statuses(
    inclusions: dict[str, bool],
    names: list[str],
    *,
    status: bool,
) -> int:
    """Apply one filter status to many names and return the number changed."""
    changed = 0
    for name in names:
        if inclusions.get(name) is not status:
            inclusions[name] = status
            changed += 1
    return changed
