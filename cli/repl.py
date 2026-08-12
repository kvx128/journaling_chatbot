from __future__ import annotations

import sys
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

from cli.http_client import CLIConnectionError, JournalAPIClient

console = Console()


def run_repl() -> None:
    client = JournalAPIClient()

    console.print(
        Panel.fit(
            "[bold green]Journaling Chatbot REPL[/bold green]\n"
            "Type your message to log expenses, check summaries, or log your mood.\n"
            "Type [bold cyan]/exit[/bold cyan] or [bold cyan]/quit[/bold cyan] to leave.",
            title="Welcome",
            border_style="green",
        )
    )

    try:
        client.health()
    except CLIConnectionError as e:
        console.print(f"[bold red]Connection Error:[/bold red] {e}")
        return

    while True:
        try:
            user_input = Prompt.ask("\n[bold yellow]You[/bold yellow]").strip()
            if not user_input:
                continue

            if user_input.lower() in ("/exit", "/quit"):
                console.print("[dim]Goodbye![/dim]")
                break

            response = client.chat(user_input)
            reply = response.get("reply", "")
            crisis = response.get("crisis_flagged", False)

            border_style = "red" if crisis else "blue"
            title = "[bold red]CRISIS NOTICE[/bold red]" if crisis else "Assistant"

            console.print(
                Panel(
                    reply,
                    title=title,
                    border_style=border_style,
                )
            )

        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Goodbye![/dim]")
            break
        except CLIConnectionError as e:
            console.print(f"[bold red]Connection Error:[/bold red] {e}")
        except Exception as e:
            console.print(f"[bold red]Error:[/bold red] {e}")
