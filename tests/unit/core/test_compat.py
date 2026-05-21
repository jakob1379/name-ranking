"""Tests for deprecated module alias compatibility helpers."""

from types import ModuleType

import pytest

from st_name_ranking import _compat


def test_install_deprecated_module_alias_returns_and_registers_target(monkeypatch):
    target = ModuleType("canonical.module")
    imported: list[str] = []

    def import_module(name: str) -> ModuleType:
        imported.append(name)
        return target

    monkeypatch.setattr(_compat.importlib, "import_module", import_module)

    with pytest.warns(DeprecationWarning, match="legacy.module is deprecated"):
        result = _compat.install_deprecated_module_alias(
            "legacy.module",
            "canonical.module",
            remove_in="0.3.0",
        )

    assert result is target
    assert imported == ["canonical.module"]
    assert _compat.sys.modules["legacy.module"] is target
    _compat.sys.modules.pop("legacy.module", None)


def test_install_deprecated_module_alias_warning_names_replacement(monkeypatch):
    target = ModuleType("canonical.other")
    monkeypatch.setattr(_compat.importlib, "import_module", lambda _name: target)

    with pytest.warns(DeprecationWarning, match="legacy.other is deprecated") as warnings:
        _compat.install_deprecated_module_alias("legacy.other", "canonical.other", remove_in="1.0")

    assert "import canonical.other instead" in str(warnings[0].message)
    assert "planned for removal in 1.0" in str(warnings[0].message)
    _compat.sys.modules.pop("legacy.other", None)
