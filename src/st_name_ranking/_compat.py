"""Compatibility helpers for deprecated top-level module paths."""

from __future__ import annotations

import importlib
import sys
import warnings
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import ModuleType


def install_deprecated_module_alias(alias_name: str, target_name: str, *, remove_in: str) -> ModuleType:
    """Resolve a deprecated module path to its canonical implementation."""
    warnings.warn(
        f"{alias_name} is deprecated; import {target_name} instead. The alias is planned for removal in {remove_in}.",
        DeprecationWarning,
        stacklevel=2,
    )
    target_module = importlib.import_module(target_name)
    sys.modules[alias_name] = target_module
    return target_module
