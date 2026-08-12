from __future__ import annotations

import typer
from cli.commands import app as commands_app
from cli.repl import run_repl

app = commands_app


@app.callback(invoke_without_command=True)
def main_callback(ctx: typer.Context):
    """Journaling Chatbot CLI."""
    if ctx.invoked_subcommand is None:
        run_repl()


if __name__ == "__main__":
    app()
