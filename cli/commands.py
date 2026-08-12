from __future__ import annotations

import subprocess
import sys
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.prompt import IntPrompt, Prompt
from rich.table import Table

from cli.http_client import CLIConnectionError, JournalAPIClient
from shared.models.enums import DateRangeEnum

app = typer.Typer(help="Journaling Chatbot CLI")
console = Console()

RANGE_MAP = {
    "today": DateRangeEnum.TODAY.value,
    "week": DateRangeEnum.THIS_WEEK.value,
    "month": DateRangeEnum.THIS_MONTH.value,
    "30d": DateRangeEnum.LAST_30D.value,
}


@app.command(name="log")
def log_cmd(message: str = typer.Argument(..., help="Text to log or query via chat")):
    """Log an expense, checkin, or query via one-shot chat."""
    client = JournalAPIClient()
    try:
        res = client.chat(message)
        reply = res.get("reply", "")
        structured = res.get("structured_data")

        console.print(Panel(reply, title="Assistant", border_style="blue"))
        if structured:
            console.print("[dim]Structured data created:[/dim]")
            console.print(structured)
    except CLIConnectionError as e:
        console.print(f"[bold red]Error:[/bold red] {e}")


@app.command(name="mood")
def mood_cmd():
    """Interactive mood check-in prompt."""
    client = JournalAPIClient()
    console.print("[bold green]Mood Check-in[/bold green]")
    try:
        rating = IntPrompt.ask("How are you feeling overall? (1-5)", choices=["1", "2", "3", "4", "5"])

        sleep_str = Prompt.ask("Sleep hours? (Press Enter to skip)", default="")
        sleep_hours = float(sleep_str) if sleep_str.strip() else None

        energy_str = Prompt.ask("Energy level? (1-5, Press Enter to skip)", default="")
        energy = int(energy_str) if energy_str.strip() in ("1", "2", "3", "4", "5") else None

        social_str = Prompt.ask("Did you have social contact today? (y/n/Press Enter to skip)", default="")
        social_contact = None
        if social_str.lower().startswith("y"):
            social_contact = True
        elif social_str.lower().startswith("n"):
            social_contact = False

        note_str = Prompt.ask("Any note or context? (Press Enter to skip)", default="")
        note = note_str.strip() if note_str.strip() else None

        payload = {
            "self_report": rating,
            "sleep_hours": sleep_hours,
            "energy": energy,
            "social_contact": social_contact,
            "note": note,
        }

        res = client.mood_checkin(payload)
        console.print("[bold green]Mood check-in recorded successfully![/bold green]")
        console.print(res)
    except CLIConnectionError as e:
        console.print(f"[bold red]Error:[/bold red] {e}")


@app.command(name="summary")
def summary_cmd(
    category: Optional[str] = typer.Option(None, "--category", "-c", help="Filter by category"),
    range: str = typer.Option("month", "--range", "-r", help="Date range: today, week, month, 30d"),
):
    """View financial summary table."""
    client = JournalAPIClient()
    mapped_range = RANGE_MAP.get(range.lower(), DateRangeEnum.THIS_MONTH.value)
    try:
        summary = client.get_summary(category=category, date_range=mapped_range)

        table = Table(title=f"Financial Summary ({summary.get('date_range')})")
        table.add_column("Category", style="cyan")
        table.add_column("Total Spent", justify="right", style="green")
        table.add_column("Count", justify="right", style="magenta")

        by_cat = summary.get("by_category", [])
        for item in by_cat:
            cat_name = item.get("category")
            amt = item.get("total_minor", 0) / 100
            cnt = item.get("count", 0)
            table.add_row(cat_name, f"{amt:.2f}", str(cnt))

        tot_debit = summary.get("total_debit_minor", 0) / 100
        tot_cnt = summary.get("transaction_count", 0)
        table.add_section()
        table.add_row("[bold]TOTAL[/bold]", f"[bold]{tot_debit:.2f}[/bold]", f"[bold]{tot_cnt}[/bold]")

        console.print(table)
    except CLIConnectionError as e:
        console.print(f"[bold red]Error:[/bold red] {e}")


@app.command(name="serve")
def serve_cmd():
    """Run Uvicorn dev server for orchestrator backend."""
    console.print("[bold green]Starting Uvicorn server...[/bold green]")
    cmd = [sys.executable, "-m", "uvicorn", "orchestrator.main:app", "--reload"]
    subprocess.run(cmd)
