#!/usr/bin/env python3
"""Typer CLI for Name Ranking Database Management.

Provides commands for database initialization, data processing,
and statistics.
"""

import datetime as dt
import importlib
import shutil
import sqlite3
import sys
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

# Import model functions
from st_name_ranking.active_learning.selection import (
    get_or_initialize_active_learning_model,
)

# Import maintenance workflow implementations
from st_name_ranking.classification.classify_origins import classify_all_names

# Import database functions
from st_name_ranking.persistence import database
from st_name_ranking.persistence.database import (
    get_connection,
    get_stats,
    init_database,
    sync_names_with_submodule,
)
from st_name_ranking.persistence.feature_store import get_feature_stats, has_feature_cache, rebuild_feature_cache

app = typer.Typer(
    help="Name Ranking Database Management CLI - Uses SQLite database",
    add_completion=True,
)
console = Console()


@app.callback(invoke_without_command=True)
def root_callback(ctx: typer.Context) -> None:
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())
        console.print(Panel("Missing command.", title="Error", border_style="red"))
        raise typer.Exit(code=2)


FORCE_DEFAULT = False
SERVER_HEADLESS_DEFAULT = False

IMPORT_SOURCE_ARG = typer.Argument(
    help="Path to the exported database file to import",
)
SERVE_TARGET_ARG = typer.Argument(
    help="Path to the Streamlit app",
)
FORCE_OPTION = typer.Option(
    "--force",
    "-f",
    help="Skip confirmation prompt",
)
CLASSIFY_OPTION = typer.Option(
    "--classify",
    help="Classify unclassified origins after initialization",
)
ORIGIN_LIMIT_OPTION = typer.Option(
    "--limit",
    "-l",
    help="Maximum number of unclassified names to classify",
)
ORIGIN_BATCH_SIZE_OPTION = typer.Option(
    "--batch-size",
    "-b",
    min=1,
    help="Number of names to classify per batch",
)
ORIGIN_STATS_ONLY_OPTION = typer.Option(
    "--stats-only",
    help="Print origin classification stats without running classification",
)
ORIGIN_SHOW_STATS_OPTION = typer.Option(
    "--show-stats",
    help="Print origin classification stats after classification",
)
SERVER_PORT_OPTION = typer.Option(
    "--server.port",
    help="Port for the Streamlit server",
)
SERVER_HEADLESS_OPTION = typer.Option(
    "--server.headless",
    help="Run without opening browser",
)

db_app = typer.Typer(
    help="Database and maintenance commands",
    name="db",
)

# Create subcommand groups
features_app = typer.Typer(
    help="Feature cache management commands",
    name="features",
)
model_app = typer.Typer(
    help="Active learning model management commands",
    name="model",
)
origins_app = typer.Typer(
    help="Origin classification maintenance commands",
    name="origins",
)

# Register subcommand groups
db_app.add_typer(features_app)
db_app.add_typer(model_app)
db_app.add_typer(origins_app)
app.add_typer(db_app)

# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------


def print_success(message: str) -> None:
    """Print a success message.

    Args:
        message: The message to display.
    """
    console.print(f"[green]✓[/green] {message}")


def print_error(message: str) -> None:
    """Print an error message.

    Args:
        message: The message to display.
    """
    console.print(f"[red]✗[/red] {message}")


def print_info(message: str) -> None:
    """Print an info message."""
    console.print(f"[blue]→[/blue] {message}")


def print_warning(message: str) -> None:
    """Print a warning message."""
    console.print(f"[yellow]⚠[/yellow] {message}")


def is_database_initialized() -> bool:
    """Check if the database exists and has the core schema."""
    if not database.get_db_path().exists():
        return False

    try:
        with get_connection() as conn:
            return table_exists(conn, "names")
    except sqlite3.OperationalError:
        return False


def table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    """Check if a table exists in the database.

    Args:
        conn: Database connection
        table_name: Name of the table to check

    Returns:
        True if table exists, False otherwise
    """
    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
    return cursor.fetchone() is not None


# ----------------------------------------------------------------------
# CLI Commands
# ----------------------------------------------------------------------


@db_app.command("init")
def init(
    *,
    classify: Annotated[bool, CLASSIFY_OPTION] = False,
) -> None:
    """Initialize the name ranking database.

    This command:
    1. Creates the database schema (if not exists)
    2. Syncs names from godkendtefornavne submodule
    3. Computes and caches features for all names
    """
    console.print("[bold blue]Initializing Name Ranking Database[/bold blue]")
    console.print()

    # 1. Initialize database schema
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Creating database schema...", total=None)
        init_database()
        progress.update(task, completed=True)
    print_success("Database schema created")

    # 2. Sync names from submodule
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Syncing names from submodule...", total=None)
        try:
            inserted = sync_names_with_submodule()
            progress.update(task, completed=True)
            print_success(f"Synced {inserted} new names from submodule")
        except (RuntimeError, ValueError) as e:
            print_error(f"Failed to sync names: {e}")
            raise typer.Exit(code=1) from e

    # 3. Compute and cache features (always done during init)
    console.print()
    console.print("[bold blue]Computing Features[/bold blue]")
    console.print()

    with Progress(console=console, transient=True) as progress:
        task = progress.add_task("Extracting features...", total=None)

        def update_progress(current: int, total: int) -> None:
            progress.update(task, total=total, completed=current)

        rebuild_result = rebuild_feature_cache(clear_existing=False, batch_size=100, progress_callback=update_progress)

    print_success(f"Created feature set version: {rebuild_result.version}")
    print_success(f"Computed features for {rebuild_result.processed} names")
    print_info(f"Feature dimension: {len(rebuild_result.feature_names)}")

    if classify:
        console.print()
        _run_origin_classification(limit=None, batch_size=100)

    # Show final statistics
    console.print()
    stats_command()


@db_app.command("stats")
def stats() -> None:
    """Show database statistics.

    Displays counts of names, comparisons, feature cache status,
    and model status in one comprehensive view.
    """
    stats_command()


def stats_command() -> None:
    """Internal statistics function with rich output."""
    console.print("[bold blue]Database Statistics[/bold blue]")
    console.print()

    stats = get_stats()
    feature_stats = get_feature_stats()

    # Create summary table
    summary_table = Table(title="Database Summary", show_header=False, box=None)
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Value", style="bold")

    total_names = stats.total_names
    summary_table.add_row("Total names", str(total_names))

    if total_names > 0:
        summary_table.add_row(
            "Classified names",
            f"{stats.classified_names} ({stats.classified_names / total_names * 100:.1f}%)",
        )
        summary_table.add_row(
            "Unclassified names",
            f"{stats.unclassified_names} ({stats.unclassified_names / total_names * 100:.1f}%)",
        )
        summary_table.add_row(
            "Rated names",
            f"{stats.rated_names} ({stats.rated_names / total_names * 100:.1f}%)",
        )
    else:
        summary_table.add_row("Classified names", "0 (0.0%)")
        summary_table.add_row("Unclassified names", "0 (0.0%)")
        summary_table.add_row("Rated names", "0 (0.0%)")

    console.print(summary_table)
    console.print()

    _print_feature_table(feature_stats, total_names)
    _print_model_table()
    _print_origin_distribution(stats)


def _print_feature_table(feature_stats: dict[str, Any], total_names: int) -> None:
    """Print feature cache statistics."""

    # Feature cache status table
    feature_table = Table(title="Feature Cache", show_header=False, box=None)
    feature_table.add_column("Metric", style="cyan")
    feature_table.add_column("Value", style="bold")

    names_with_features = feature_stats["names_with_features"]
    feature_coverage = names_with_features / max(total_names, 1) * 100
    feature_table.add_row("Names with features", f"{names_with_features} ({feature_coverage:.1f}%)")
    feature_table.add_row("Feature sets", str(feature_stats["feature_sets_count"]))
    feature_table.add_row("Active version", feature_stats["active_version"] or "None")

    console.print(feature_table)
    console.print()


def _print_model_table() -> None:
    """Print active learning model status table."""
    try:
        model = get_or_initialize_active_learning_model()
        state = model.state

        model_table = Table(title="Model Status", show_header=False, box=None)
        model_table.add_column("Metric", style="cyan")
        model_table.add_column("Value", style="bold")

        model_table.add_row("Feature dimension", str(state.feature_dim))
        model_table.add_row("Training samples", str(state.training_samples))

        console.print(model_table)
        console.print()
    except (RuntimeError, ValueError):
        print_warning("Model not initialized yet")
        console.print()


def _print_origin_distribution(stats: Any) -> None:
    """Print origin distribution statistics."""
    origin_distribution = _stat(stats, "origin_distribution")
    total_names = int(_stat(stats, "total_names"))
    if origin_distribution:
        dist_table = Table(
            title="Origin Distribution",
            show_header=True,
            header_style="bold",
        )
        dist_table.add_column("Region", style="cyan")
        dist_table.add_column("Count", justify="right")
        dist_table.add_column("Percentage", justify="right")

        for region, count in sorted(
            origin_distribution.items(),
            key=lambda x: x[1],
            reverse=True,
        ):
            percentage = count / total_names * 100
            dist_table.add_row(
                region,
                str(count),
                f"{percentage:.1f}%",
            )

        console.print(dist_table)
    else:
        print_info("No origin classification data available.")


# ----------------------------------------------------------------------
# Origin Classification Subcommands
# ----------------------------------------------------------------------


def _stat(stats: Any, name: str) -> Any:
    """Read a statistic from either the current dataclass or legacy dict shape."""
    if isinstance(stats, dict):
        return stats[name]
    return getattr(stats, name)


def _print_origin_classification_stats(label: str = "Origin Classification Statistics") -> None:
    """Print focused origin classification statistics."""
    stats = get_stats()
    total = int(_stat(stats, "total_names"))
    classified = int(_stat(stats, "classified_names"))
    unclassified = int(_stat(stats, "unclassified_names"))

    table = Table(title=label, show_header=False, box=None)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="bold")
    table.add_row("Total names", str(total))
    if total > 0:
        table.add_row("Classified names", f"{classified} ({classified / total * 100:.1f}%)")
        table.add_row("Unclassified names", f"{unclassified} ({unclassified / total * 100:.1f}%)")
    else:
        table.add_row("Classified names", "0 (0.0%)")
        table.add_row("Unclassified names", "0 (0.0%)")
    console.print(table)

    _print_origin_distribution(stats)


@origins_app.command("classify")
def _run_origin_classification(
    *,
    limit: Annotated[int | None, ORIGIN_LIMIT_OPTION] = None,
    batch_size: Annotated[int, ORIGIN_BATCH_SIZE_OPTION] = 100,
    show_stats: Annotated[bool, ORIGIN_SHOW_STATS_OPTION] = False,
    stats_only: Annotated[bool, ORIGIN_STATS_ONLY_OPTION] = False,
) -> int:
    """Run the canonical origin-classification maintenance workflow."""
    console.print("[bold blue]Processing Data Enrichment[/bold blue]")
    console.print()

    if stats_only:
        _print_origin_classification_stats()
        return 0

    try:
        classified = classify_all_names(limit, batch_size)
    except ImportError as err:
        print_error(f"Origin classification dependency unavailable: {err}")
        raise typer.Exit(code=1) from err
    except (RuntimeError, ValueError) as err:
        print_error(f"Origin classification failed: {err}")
        raise typer.Exit(code=1) from err

    print_success(f"Classified {classified} names")

    if show_stats:
        console.print()
        _print_origin_classification_stats("Origin Classification Statistics")

    return classified


@app.command("process", hidden=True)
def process(
    *,
    limit: Annotated[int | None, ORIGIN_LIMIT_OPTION] = None,
    batch_size: Annotated[int, ORIGIN_BATCH_SIZE_OPTION] = 100,
) -> None:
    """Compatibility alias for `st-name-ranking db origins classify`."""
    _run_origin_classification(limit=limit, batch_size=batch_size)


# ----------------------------------------------------------------------
# Features Subcommands
# ----------------------------------------------------------------------


@features_app.command("rebuild")
def features_rebuild(
    *,
    force: Annotated[bool, FORCE_OPTION] = FORCE_DEFAULT,
) -> None:
    """Recompute all features (useful after feature set update).

    This command:
    1. Clears existing cached features
    2. Creates a new feature set version
    3. Re-extracts features for all names
    4. Shows progress during extraction
    """
    console.print("[bold blue]Rebuilding Feature Cache[/bold blue]")
    console.print()

    # Check if database is initialized
    if not is_database_initialized():
        print_error("Database not initialized. Run 'st-name-ranking db init' first.")
        raise typer.Exit(1)

    # Check if features exist
    feature_stats = get_feature_stats()
    if feature_stats["names_with_features"] > 0 and not force:
        confirm = typer.confirm(
            f"This will clear {feature_stats['names_with_features']} cached features and recompute them. Continue?",
        )
        if not confirm:
            console.print("Rebuild cancelled.")
            raise typer.Abort

    console.print()
    console.print("[bold blue]Extracting Features[/bold blue]")

    with Progress(console=console, transient=True) as progress:
        task = progress.add_task("Extracting features...", total=None)

        def update_progress(current: int, total: int) -> None:
            progress.update(task, total=total, completed=current)

        rebuild_result = rebuild_feature_cache(clear_existing=True, batch_size=100, progress_callback=update_progress)

    print_success(f"Cleared {rebuild_result.deleted} cached features")
    print_success(f"Created new feature set version: {rebuild_result.version}")
    print_success(f"Computed features for {rebuild_result.processed} names")
    print_info(f"Feature dimension: {len(rebuild_result.feature_names)}")
    print_info(f"Active feature set: {rebuild_result.version}")


@features_app.command("status")
def features_status() -> None:
    """Show feature cache status."""
    console.print("[bold blue]Feature Cache Status[/bold blue]")
    console.print()

    # Check if database is initialized
    try:
        with get_connection() as conn:
            if not table_exists(conn, "names"):
                print_error("Database not initialized. Run 'st-name-ranking db init' first.")
                raise typer.Exit(1)
    except sqlite3.OperationalError as err:
        print_error("Database not initialized. Run 'st-name-ranking db init' first.")
        raise typer.Exit(1) from err

    feature_stats = get_feature_stats()
    db_stats = get_stats()

    # Create status table
    status_table = Table(
        show_header=True,
        header_style="bold",
    )
    status_table.add_column("Metric", style="cyan")
    status_table.add_column("Value", justify="right")

    status_table.add_row("Total names", str(db_stats.total_names))
    status_table.add_row("Names with features", str(feature_stats["names_with_features"]))
    status_table.add_row("Feature sets", str(feature_stats["feature_sets_count"]))
    status_table.add_row("Active version", feature_stats["active_version"] or "None")

    # Coverage percentage
    if db_stats.total_names > 0:
        coverage = feature_stats["names_with_features"] / db_stats.total_names * 100
        status_table.add_row("Coverage", f"{coverage:.1f}%")
    else:
        status_table.add_row("Coverage", "0%")

    console.print(status_table)
    console.print()

    # Check if features need recomputation
    if feature_stats["names_with_features"] == 0:
        print_warning("No features computed. Run: st-name-ranking db features rebuild")
    elif feature_stats["names_with_features"] < db_stats.total_names:
        print_warning(
            f"Missing features for {db_stats.total_names - feature_stats['names_with_features']} names. "
            "Run: st-name-ranking db features rebuild",
        )
    else:
        print_success("All names have cached features")


# ----------------------------------------------------------------------
# Model Subcommands
# ----------------------------------------------------------------------


@model_app.command("status")
def model_status() -> None:
    """Show active learning model status."""
    console.print("[bold blue]Active Learning Model Status[/bold blue]")
    console.print()

    # Check if features exist
    if not has_feature_cache():
        print_warning("No features computed yet. Some model operations may fail.")
        print_info("Run 'st-name-ranking db features rebuild' to compute features.")
        console.print()

    try:
        model = get_or_initialize_active_learning_model()

        # Get model state
        state = model.state
        training_samples = state.training_samples
        feature_dim = state.feature_dim

        # Create status table
        status_table = Table(
            show_header=True,
            header_style="bold",
        )
        status_table.add_column("Metric", style="cyan")
        status_table.add_column("Value", justify="right")

        status_table.add_row("Feature dimension", str(feature_dim))
        status_table.add_row("Training samples", str(training_samples))
        status_table.add_row("Last updated", "From database")

        console.print(status_table)
        console.print()

        # Show feature names (truncated)
        print_info(f"Features: {', '.join(state.feature_names[:5])}...")
        print_info(f"Total features: {len(state.feature_names)}")

    except (RuntimeError, ValueError) as e:
        print_error(f"Failed to get model status: {e}")


@model_app.command("reset")
def model_reset() -> None:
    """Reset active learning model (reinitialize)."""
    console.print("[bold blue]Resetting Active Learning Model[/bold blue]")
    console.print()

    confirm = typer.confirm(
        "Are you sure you want to reset the model? All learned preferences will be lost.",
    )
    if not confirm:
        console.print("Model reset cancelled.")
        raise typer.Abort

    try:
        # Delete model state from database
        with get_connection() as conn:
            conn.execute("DELETE FROM model_state WHERE id = 1")

        # Reinitialize model
        model = get_or_initialize_active_learning_model()
        model.save_to_db()

        print_success("Model reset successfully. New model initialized.")

    except (RuntimeError, ValueError) as e:
        print_error(f"Failed to reset model: {e}")


def _run_serve_preflight_checks() -> None:
    """Validate db and feature prerequisites for serve command."""
    if not is_database_initialized():
        print_warning("Database not initialized yet.")
        run_init = typer.confirm("Run 'st-name-ranking db init' now?", default=True)
        if not run_init:
            print_info("Exiting. Run 'st-name-ranking db init' and try again.")
            raise typer.Exit(code=1)

        init()
        if not is_database_initialized():
            print_error("Database is still not initialized. Run 'st-name-ranking db init' first.")
            raise typer.Exit(code=1)

    if not has_feature_cache():
        print_warning("Features not computed yet.")
        rebuild = typer.confirm("Run 'st-name-ranking db features rebuild' now?", default=True)
        if not rebuild:
            print_info("Exiting. Run 'st-name-ranking db features rebuild' and try again.")
            raise typer.Exit(code=1)

        features_rebuild(force=True)
        if not has_feature_cache():
            print_error("Features are still unavailable. Run 'st-name-ranking db features rebuild' first.")
            raise typer.Exit(code=1)


def _run_streamlit(cmd: list[str]) -> None:
    """Run streamlit CLI with provided arguments."""
    original_argv = sys.argv.copy()
    try:
        streamlit_cli = importlib.import_module("streamlit.web.cli")
        sys.argv = cmd
        streamlit_cli.main()
    except SystemExit as err:
        if err.code in (None, 0):
            raise typer.Exit(code=0) from err
        print_error(f"Streamlit failed to start (exit code: {err.code})")
        raise typer.Exit(code=1) from err
    except KeyboardInterrupt as err:
        console.print("\n[bold blue]Streamlit stopped[/bold blue]")
        raise typer.Exit(code=0) from err
    finally:
        sys.argv = original_argv


# ----------------------------------------------------------------------
# Import Command
# ----------------------------------------------------------------------


@db_app.command("import")
def import_db(
    source: Annotated[Path, IMPORT_SOURCE_ARG],
    *,
    force: Annotated[bool, FORCE_OPTION] = FORCE_DEFAULT,
) -> None:
    """Import a database from an exported file.

    Backs up the current database and replaces it with the imported one.
    """
    console.print("[bold blue]Importing Database[/bold blue]")
    console.print()

    # Verify source file exists
    if not source.exists():
        print_error(f"Source file not found: {source}")
        raise typer.Exit(code=1)

    # Confirm unless --force
    if not force:
        confirm = typer.confirm(
            f"This will replace your current database with {source.name}. A backup will be created. Continue?",
        )
        if not confirm:
            console.print("Import cancelled.")
            raise typer.Abort

    try:
        # Create backup of current database if it exists
        db_path = database.get_db_path()
        if db_path.exists():
            backup_timestamp = dt.datetime.now(dt.UTC).strftime("%Y%m%d_%H%M%S")
            backup_path = db_path.with_suffix(f".db.backup.{backup_timestamp}")
            shutil.copy2(db_path, backup_path)
            print_success(f"Created backup: {backup_path.name}")

        # Copy new database
        db_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, db_path)
        print_success(f"Imported database from {source.name}")
        print_info(f"Database location: {db_path}")

        # Show stats
        console.print()
        stats_command()

    except (OSError, RuntimeError) as e:
        print_error(f"Failed to import database: {e}")
        raise typer.Exit(code=1) from None


@app.command("serve")
def serve(
    target: Annotated[Path, SERVE_TARGET_ARG] = Path("src/st_name_ranking/interface/main.py"),
    *,
    server_port: Annotated[int, SERVER_PORT_OPTION] = 8501,
    server_headless: Annotated[bool, SERVER_HEADLESS_OPTION] = SERVER_HEADLESS_DEFAULT,
) -> None:
    """Serve the Streamlit web interface.

    This command launches the name ranking Streamlit application.
    Run 'st-name-ranking db init' first if you haven't initialized the database.
    """
    _run_serve_preflight_checks()

    print_success("Pre-flight checks passed")
    print_info(f"Starting Streamlit on port {server_port}")

    cmd = ["streamlit", "run", str(target), "--server.port", str(server_port)]

    if server_headless:
        cmd.extend(["--server.headless", "true"])

    _run_streamlit(cmd)


if __name__ == "__main__":
    app()
