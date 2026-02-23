#!/usr/bin/env python3
"""Typer CLI for Name Ranking Database Management.

Provides commands for database initialization, data processing,
and statistics.
"""

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

# Import database functions
from st_name_ranking.classify_origins import classify_all_names
from st_name_ranking.database import (
    get_stats,
    init_database,
    sync_names_with_submodule,
)

# Import model functions
from st_name_ranking.utils import (
    get_active_learning_model,
)

app = typer.Typer(
    help="Name Ranking Database Management CLI - Uses SQLite database",
    add_completion=False,
)
console = Console()

# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------


def print_success(message: str) -> None:
    """Print a success message."""
    console.print(f"[green]✓[/green] {message}")


def print_error(message: str) -> None:
    """Print an error message."""
    console.print(f"[red]✗[/red] {message}")


def print_info(message: str) -> None:
    """Print an info message."""
    console.print(f"[blue]→[/blue] {message}")


# ----------------------------------------------------------------------
# CLI Commands
# ----------------------------------------------------------------------


@app.command()
def init(
    classify: bool = typer.Option(
        False,
        "--classify",
        "-c",
        help="Run initial origin classification after initialization",
    ),
) -> None:
    """Initialize the name ranking database.

    This command:
    1. Creates the database schema (if not exists)
    2. Syncs names from godkendtefornavne submodule
    3. Optionally runs initial origin classification
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
        except Exception as e:
            print_error(f"Failed to sync names: {e}")
            raise typer.Exit(code=1)

    # 3. Optional origin classification
    if classify:
        process_command(limit=None, batch_size=100)

    # Show final statistics
    stats_command()


@app.command()
def process(
    limit: int | None = typer.Option(
        None,
        "--limit",
        "-l",
        help="Limit number of names to classify (for testing)",
    ),
    batch_size: int = typer.Option(
        100,
        "--batch-size",
        "-b",
        help="Batch size for processing",
    ),
) -> None:
    """Process data enrichment tasks (origin classification).

    This command processes unclassified names in batches,
    predicting their nationality and mapping to geographic regions.
    """
    process_command(limit, batch_size)


def process_command(limit: int | None = None, batch_size: int = 100) -> None:
    """Internal classification function with rich output."""
    console.print("[bold blue]Processing Data Enrichment[/bold blue]")
    console.print()

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task(
                "Processing data enrichment...",
                total=None,
            )
            classified = classify_all_names(limit, batch_size)
            progress.update(task, completed=True)

        if classified == 0:
            print_info("No unclassified names found.")
        else:
            print_success(f"Classified {classified} names")

    except ImportError:
        print_error("ethnidata is not installed.")
        console.print()
        console.print("Install it with:")
        console.print("  [bold]pip install ethnidata[/bold]")
        console.print()
        console.print("Or add to pyproject.toml dependencies:")
        console.print("  dependencies = [")
        console.print("    ...")
        console.print('    "ethnidata>=4.1.1",')
        console.print("    ...")
        console.print("  ]")
        raise typer.Exit(code=1)
    except Exception as e:
        print_error(f"Classification failed: {e}")
        raise typer.Exit(code=1)


@app.command()
def stats() -> None:
    """Show database statistics.

    Displays counts of names, classified names, rated names,
    and origin distribution.
    """
    stats_command()


def stats_command() -> None:
    """Internal statistics function with rich output."""
    console.print("[bold blue]Database Statistics[/bold blue]")
    console.print()

    stats = get_stats()

    # Create summary table
    summary_table = Table(title="Summary", show_header=False, box=None)
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


@app.command()
def model_status() -> None:
    """Show active learning model status."""
    console.print("[bold blue]Active Learning Model Status[/bold blue]")
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

    except Exception as e:
        print_error(f"Failed to get model status: {e}")


@app.command()
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
        from st_name_ranking.database import get_connection

        with get_connection() as conn:
            conn.execute("DELETE FROM model_state WHERE id = 1")

        # Reinitialize model
        model = get_active_learning_model()
        model.save_to_db()

        print_success("Model reset successfully. New model initialized.")

    except Exception as e:
        print_error(f"Failed to reset model: {e}")


if __name__ == "__main__":
    app()
