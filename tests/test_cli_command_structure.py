"""CLI command structure tests."""

from typer.testing import CliRunner

from st_name_ranking.cli import app


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
