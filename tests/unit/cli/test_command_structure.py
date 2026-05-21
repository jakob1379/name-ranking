"""CLI command structure tests."""

from unittest.mock import patch

from typer.testing import CliRunner

from st_name_ranking.commands.cli import app


def test_top_level_commands_are_serve_and_db() -> None:
    """Top-level CLI should expose focused serve|db commands."""
    runner = CliRunner()
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "serve" in result.output
    assert "db" in result.output
    assert "init" not in result.output
    assert "features" not in result.output
    assert "model" not in result.output


def test_db_group_contains_database_commands() -> None:
    """db group should contain operational database subcommands."""
    runner = CliRunner()
    result = runner.invoke(app, ["db", "--help"])

    assert result.exit_code == 0
    assert "init" in result.output
    assert "stats" in result.output
    assert "import" in result.output
    assert "features" in result.output
    assert "model" in result.output
    assert "origins" in result.output


def test_no_command_shows_help_and_error() -> None:
    """No-arg CLI should print help and missing command error."""
    runner = CliRunner()
    result = runner.invoke(app, [])

    assert result.exit_code == 2
    assert "Name Ranking Database Management CLI" in result.output
    assert "serve" in result.output
    assert "db" in result.output
    assert "Missing command." in result.output
    assert result.output.count("Usage:") == 1


def test_db_origins_classify_is_canonical_classification_entrypoint() -> None:
    """Origin classification should be exposed under the db maintenance group."""
    runner = CliRunner()

    with patch("st_name_ranking.commands.cli.classify_all_names", return_value=7) as classify_all:
        result = runner.invoke(app, ["db", "origins", "classify", "--limit", "50", "--batch-size", "25"])

    assert result.exit_code == 0
    assert "Processing Data Enrichment" in result.output
    assert "Classified 7 names" in result.output
    classify_all.assert_called_once_with(50, 25)
