#!/usr/bin/env python3
"""Typer CLI for Name Ranking Database Management.

Provides commands for database initialization, data processing,
and statistics.
"""

import datetime as dt
import json
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

# Import database functions
from st_name_ranking.database import (
    DB_PATH,
    get_connection,
    get_stats,
    init_database,
    sync_names_with_submodule,
)

# Import features for extraction
from st_name_ranking.features import FeatureExtractor

# Import model functions
from st_name_ranking.utils import (
    get_active_learning_model,
)

app = typer.Typer(
    help="Name Ranking Database Management CLI - Uses SQLite database",
    add_completion=False,
)
console = Console()

# Create subcommand groups
features_app = typer.Typer(
    help="Feature cache management commands",
    name="features",
)
model_app = typer.Typer(
    help="Active learning model management commands",
    name="model",
)

# Register subcommand groups
app.add_typer(features_app)
app.add_typer(model_app)

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


# ----------------------------------------------------------------------
# Feature System Helpers
# ----------------------------------------------------------------------


def ensure_features_computed() -> bool:
    """Check if features exist in the database.

    Returns:
        True if features exist, False otherwise.
    """
    try:
        with get_connection() as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM name_features LIMIT 1")
            count = cursor.fetchone()[0]
            return count > 0
    except sqlite3.OperationalError:
        # Table doesn't exist yet
        return False


def get_feature_stats() -> dict[str, Any]:
    """Get feature system statistics.

    Returns:
        Dictionary with feature statistics.
    """
    with get_connection() as conn:
        # Check if tables exist
        cursor = conn.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table' AND name IN ('feature_sets', 'name_features')
        """)
        existing_tables = {row[0] for row in cursor.fetchall()}

        if "feature_sets" not in existing_tables or "name_features" not in existing_tables:
            return {
                "feature_sets_count": 0,
                "names_with_features": 0,
                "active_version": None,
            }

        # Get feature set count
        cursor = conn.execute("SELECT COUNT(*) FROM feature_sets")
        feature_sets_count = cursor.fetchone()[0]

        # Get names with features
        cursor = conn.execute("SELECT COUNT(DISTINCT name_id) FROM name_features")
        names_with_features = cursor.fetchone()[0]

        # Get active version
        cursor = conn.execute("""
            SELECT version FROM feature_sets
            WHERE is_active = 1
            ORDER BY created_at DESC
            LIMIT 1
        """)
        row = cursor.fetchone()
        active_version = row[0] if row else None

        return {
            "feature_sets_count": feature_sets_count,
            "names_with_features": names_with_features,
            "active_version": active_version,
        }


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


def create_feature_set(version: str, feature_names: list[str]) -> int:
    """Create a new feature set and return its ID.

    Args:
        version: Feature set version identifier
        feature_names: List of feature names

    Returns:
        The ID of the created feature set

    Raises:
        RuntimeError: If database tables don't exist (init not run)
    """
    with get_connection() as conn:
        # Check if tables exist
        if not table_exists(conn, "feature_sets"):
            raise RuntimeError("Database not initialized. Run 'st-name-ranking init' first.")

        # Deactivate all existing feature sets
        conn.execute("UPDATE feature_sets SET is_active = 0")

        # Insert new feature set
        cursor = conn.execute(
            """
            INSERT INTO feature_sets (version, feature_names_json, is_active)
            VALUES (?, ?, 1)
            """,
            (version, json.dumps(feature_names)),
        )
        return cursor.lastrowid


def extract_and_cache_features(
    feature_set_id: int,
    batch_size: int = 100,
    progress_callback: Any | None = None,
) -> int:
    """Extract features for all names and cache them in the database.

    Args:
        feature_set_id: The ID of the feature set to use
        batch_size: Number of names to process per batch
        progress_callback: Optional callback function(current, total)

    Returns:
        Number of names processed
    """
    extractor = FeatureExtractor()
    feature_names = extractor.get_feature_names()

    with get_connection() as conn:
        # Get all names with their metadata
        cursor = conn.execute("""
            SELECT id, name, gender, origin_region
            FROM names
            ORDER BY id
        """)
        names_data = cursor.fetchall()

    total = len(names_data)
    processed = 0

    # Process in batches
    for i in range(0, total, batch_size):
        batch = names_data[i : i + batch_size]

        with get_connection() as conn:
            for name_id, name, gender, origin_region in batch:
                # Extract features
                features = extractor.extract(name, gender, origin_region)
                features_dict = {name: float(value) for name, value in zip(feature_names, features.tolist())}

                # Insert into database
                conn.execute(
                    """
                    INSERT OR REPLACE INTO name_features
                    (name_id, feature_set_id, features_json, computed_at)
                    VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                    """,
                    (name_id, feature_set_id, json.dumps(features_dict)),
                )

        processed += len(batch)
        if progress_callback:
            progress_callback(processed, total)

    return processed


def clear_all_features() -> int:
    """Clear all cached features from the database.

    Returns:
        Number of rows deleted
    """
    with get_connection() as conn:
        # Check if table exists first
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='name_features'")
        if not cursor.fetchone():
            return 0
        cursor = conn.execute("DELETE FROM name_features")
        return cursor.rowcount


# ----------------------------------------------------------------------
# CLI Commands
# ----------------------------------------------------------------------


@app.command()
def init() -> None:
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

    # Get feature set version based on current time
    version = dt.datetime.now(dt.UTC).strftime("%Y%m%d_%H%M%S")

    # Create feature set
    extractor = FeatureExtractor()
    feature_names = extractor.get_feature_names()
    feature_set_id = create_feature_set(version, feature_names)
    print_success(f"Created feature set version: {version}")

    # Extract and cache features
    with Progress(console=console, transient=True) as progress:
        task = progress.add_task("Extracting features...", total=None)

        def update_progress(current: int, total: int) -> None:
            progress.update(task, total=total, completed=current)

        processed = extract_and_cache_features(feature_set_id, batch_size=100, progress_callback=update_progress)

    print_success(f"Computed features for {processed} names")
    print_info(f"Feature dimension: {len(feature_names)}")

    # Show final statistics
    console.print()
    stats_command()


@app.command()
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

    # Feature cache status table
    feature_table = Table(title="Feature Cache", show_header=False, box=None)
    feature_table.add_column("Metric", style="cyan")
    feature_table.add_column("Value", style="bold")

    feature_table.add_row(
        "Names with features",
        f"{feature_stats['names_with_features']} ({feature_stats['names_with_features'] / max(total_names, 1) * 100:.1f}%)",
    )
    feature_table.add_row("Feature sets", str(feature_stats["feature_sets_count"]))
    feature_table.add_row("Active version", feature_stats["active_version"] or "None")

    console.print(feature_table)
    console.print()

    # Model status
    try:
        model = get_active_learning_model()
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

    # Origin distribution table
    if stats.origin_distribution:
        dist_table = Table(
            title="Origin Distribution",
            show_header=True,
            header_style="bold",
        )
        dist_table.add_column("Region", style="cyan")
        dist_table.add_column("Count", justify="right")
        dist_table.add_column("Percentage", justify="right")

        for region, count in sorted(
            stats.origin_distribution.items(),
            key=lambda x: x[1],
            reverse=True,
        ):
            percentage = count / stats.total_names * 100
            dist_table.add_row(
                region,
                str(count),
                f"{percentage:.1f}%",
            )

        console.print(dist_table)
    else:
        print_info("No origin classification data available.")


# ----------------------------------------------------------------------
# Features Subcommands
# ----------------------------------------------------------------------


@features_app.command("rebuild")
def features_rebuild(
    *,
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Skip confirmation prompt",
    ),
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
    try:
        with get_connection() as conn:
            if not table_exists(conn, "names"):
                print_error("Database not initialized. Run 'name-db init' first.")
                raise typer.Exit(1)
    except sqlite3.OperationalError:
        print_error("Database not initialized. Run 'name-db init' first.")
        raise typer.Exit(1)

    # Check if features exist
    feature_stats = get_feature_stats()
    if feature_stats["names_with_features"] > 0 and not force:
        confirm = typer.confirm(
            f"This will clear {feature_stats['names_with_features']} cached features and recompute them. Continue?",
        )
        if not confirm:
            console.print("Rebuild cancelled.")
            raise typer.Abort()

    # Clear existing features
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Clearing existing features...", total=None)
        deleted = clear_all_features()
        progress.update(task, completed=True)
    print_success(f"Cleared {deleted} cached features")

    # Create new feature set
    version = dt.datetime.now(dt.UTC).strftime("%Y%m%d_%H%M%S")
    extractor = FeatureExtractor()
    feature_names = extractor.get_feature_names()
    feature_set_id = create_feature_set(version, feature_names)
    print_success(f"Created new feature set version: {version}")

    # Extract and cache features
    console.print()
    console.print("[bold blue]Extracting Features[/bold blue]")

    with Progress(console=console, transient=True) as progress:
        task = progress.add_task("Extracting features...", total=None)

        def update_progress(current: int, total: int) -> None:
            progress.update(task, total=total, completed=current)

        processed = extract_and_cache_features(feature_set_id, batch_size=100, progress_callback=update_progress)

    print_success(f"Computed features for {processed} names")
    print_info(f"Feature dimension: {len(feature_names)}")
    print_info(f"Active feature set: {version}")


@features_app.command("status")
def features_status() -> None:
    """Show feature cache status."""
    console.print("[bold blue]Feature Cache Status[/bold blue]")
    console.print()

    # Check if database is initialized
    try:
        with get_connection() as conn:
            if not table_exists(conn, "names"):
                print_error("Database not initialized. Run 'st-name-ranking init' first.")
                raise typer.Exit(1)
    except sqlite3.OperationalError:
        print_error("Database not initialized. Run 'st-name-ranking init' first.")
        raise typer.Exit(1)

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
        print_warning("No features computed. Run: st-name-ranking features rebuild")
    elif feature_stats["names_with_features"] < db_stats.total_names:
        print_warning(
            f"Missing features for {db_stats.total_names - feature_stats['names_with_features']} names. "
            "Run: st-name-ranking features rebuild",
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
    if not ensure_features_computed():
        print_warning("No features computed yet. Some model operations may fail.")
        print_info("Run 'st-name-ranking features rebuild' to compute features.")
        console.print()

    try:
        model = get_active_learning_model()

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
        raise typer.Abort()

    try:
        # Delete model state from database
        with get_connection() as conn:
            conn.execute("DELETE FROM model_state WHERE id = 1")

        # Reinitialize model
        model = get_active_learning_model()
        model.save_to_db()

        print_success("Model reset successfully. New model initialized.")

    except (RuntimeError, ValueError) as e:
        print_error(f"Failed to reset model: {e}")


# ----------------------------------------------------------------------
# Import Command
# ----------------------------------------------------------------------


@app.command("import")
def import_db(
    source: Path = typer.Argument(..., help="Path to the exported database file to import"),
    *,
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Skip confirmation prompt",
    ),
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
            raise typer.Abort()

    try:
        # Create backup of current database if it exists
        if DB_PATH.exists():
            backup_path = DB_PATH.with_suffix(f".db.backup.{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}")
            shutil.copy2(DB_PATH, backup_path)
            print_success(f"Created backup: {backup_path.name}")

        # Copy new database
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, DB_PATH)
        print_success(f"Imported database from {source.name}")
        print_info(f"Database location: {DB_PATH}")

        # Show stats
        console.print()
        stats_command()

    except (OSError, RuntimeError) as e:
        print_error(f"Failed to import database: {e}")
        raise typer.Exit(code=1) from None


@app.command()
def start(
    target: Path = typer.Argument("src/st_name_ranking/main.py", help="Path to the Streamlit app"),
    server_port: int = typer.Option(8501, "--server.port", help="Port for the Streamlit server"),
    server_headless: bool = typer.Option(False, "--server.headless", help="Run without opening browser"),
) -> None:
    """Start the Streamlit web interface.

    This command launches the name ranking Streamlit application.
    Run 'st-name-ranking init' first if you haven't initialized the database.
    """
    # Pre-flight checks
    if not DB_PATH.exists():
        print_error("Database not initialized. Run 'st-name-ranking init' first.")
        raise typer.Exit(code=1)

    # Check if feature tables exist
    try:
        with get_connection() as conn:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name IN ('feature_sets', 'name_features')"
            )
            existing_tables = {row[0] for row in cursor.fetchall()}

            if "feature_sets" not in existing_tables or "name_features" not in existing_tables:
                print_error("Features not computed. Run 'st-name-ranking features rebuild' first.")
                raise typer.Exit(code=1)

            # Check if features have been computed for at least some names
            cursor = conn.execute("SELECT COUNT(*) FROM name_features LIMIT 1")
            count = cursor.fetchone()[0]
            if count == 0:
                print_error("Features not computed. Run 'st-name-ranking features rebuild' first.")
                raise typer.Exit(code=1)
    except sqlite3.OperationalError:
        print_error("Features not computed. Run 'st-name-ranking features rebuild' first.")
        raise typer.Exit(code=1)

    print_success("Pre-flight checks passed")
    print_info(f"Starting Streamlit on port {server_port}")

    # Build the command
    cmd = [
        sys.executable,
        "-m",
        "streamlit.web.cli",
        "run",
        str(target),
        "--server.port",
        str(server_port),
    ]

    if server_headless:
        cmd.extend(["--server.headless", "true"])

    # Launch streamlit
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print_error(f"Streamlit failed to start: {e}")
        raise typer.Exit(code=1) from None
    except KeyboardInterrupt:
        console.print("\n[bold blue]Streamlit stopped[/bold blue]")
        raise typer.Exit(code=0)


if __name__ == "__main__":
    app()
