import json
import os
from datetime import datetime, timezone
import typer
from rich.console import Console
from rich.prompt import Prompt, Confirm
from rich.panel import Panel

from shared.models.enums import IntentEnum, Category

app = typer.Typer(help="Labeling CLI for ground-truth evaluation data.")
console = Console()

@app.command()
def main():
    pool_path = "ml/data/candidates/pool.jsonl"
    if not os.path.exists(pool_path):
        console.print("[red]no candidate pool found — run `python -m ml.data.prepare_pool` first[/red]")
        raise typer.Exit(1)

    candidates = []
    with open(pool_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                candidates.append(json.loads(line))

    labeled_path = "ml/data/gold/labeled.jsonl"
    skipped_path = "ml/data/gold/skipped.jsonl"

    os.makedirs(os.path.dirname(labeled_path), exist_ok=True)

    labeled_ids = set()
    if os.path.exists(labeled_path):
        with open(labeled_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    labeled_ids.add(json.loads(line)["id"])

    total_pool = len(candidates)

    unlabeled = [c for c in candidates if c["id"] not in labeled_ids]

    labeled_count_start = len(labeled_ids)
    current_labeled = 0
    current_skipped = 0

    categories = list(Category)

    for candidate in unlabeled:
        while True:
            idx = labeled_count_start + current_labeled + current_skipped + 1
            console.print(f"\n[cyan][{idx}/{total_pool}][/cyan]")
            console.print(Panel(candidate["text"], title=candidate["id"], subtitle=candidate.get("source", ""), expand=False))

            intent_choices = ["1", "2", "3", "4", "5", "6", "7", "/skip", "/quit"]
            intent_str = Prompt.ask(
                "Intent? [1]FIN_LOG [2]FIN_QUERY [3]MOOD [4]JOURNAL [5]SMALLTALK [6]UNKNOWN [7]BOTH",
                choices=intent_choices
            )

            if intent_str == "/quit":
                console.print(f"[green]Progress saved. {labeled_count_start + current_labeled}/{total_pool} labeled. Resume anytime with the same command.[/green]")
                raise typer.Exit()

            if intent_str == "/skip":
                with open(skipped_path, "a", encoding="utf-8") as sf:
                    sf.write(json.dumps({
                        "id": candidate["id"],
                        "text": candidate["text"],
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    }) + "\n")
                current_skipped += 1
                break

            intent_map = {
                "1": "FINANCE_LOG",
                "2": "FINANCE_QUERY",
                "3": "MOOD_CHECKIN",
                "4": "JOURNAL_FREE",
                "5": "SMALLTALK",
                "6": "UNKNOWN",
                "7": "BOTH_FINANCE_MOOD"
            }
            intent_val = intent_map[intent_str]

            amount_minor = None
            category = None
            merchant = None

            if intent_str in ["1", "2", "7"]:
                amt_input = Prompt.ask("Amount (plain rupees, e.g. 450 or 450.50) [dim](blank for null)[/dim]", default="")
                if amt_input.strip():
                    try:
                        amount_minor = int(float(amt_input.strip()) * 100)
                    except ValueError:
                        console.print("[yellow]Invalid amount, setting to null[/yellow]")

                cat_prompt = "Category?\n"
                for ci, cat in enumerate(categories, 1):
                    cat_prompt += f"[{ci}] {cat.value}  "
                    if ci % 5 == 0: cat_prompt += "\n"
                cat_prompt += "\n[0] skip"

                cat_input = Prompt.ask(cat_prompt, default="0")
                if cat_input.isdigit():
                    ci = int(cat_input)
                    if 1 <= ci <= len(categories):
                        category = categories[ci - 1].value

                merch_input = Prompt.ask("Merchant [dim](blank for null)[/dim]", default="")
                if merch_input.strip():
                    merchant = merch_input.strip()

            valence = None
            arousal = None
            emotion_tags = None
            self_report = None

            do_mood = False
            if intent_str in ["3", "7"]:
                do_mood = True
            elif intent_str in ["4", "6"]:
                add_mood = Confirm.ask("Add mood signal too?", default=False)
                if add_mood:
                    do_mood = True

            if do_mood:
                val_input = Prompt.ask("Valence (-1.0 to 1.0) [dim](blank for null)[/dim]", default="")
                if val_input.strip():
                    try:
                        valence = float(val_input.strip())
                    except ValueError:
                        pass

                aro_input = Prompt.ask("Arousal (-1.0 to 1.0) [dim](blank for null)[/dim]", default="")
                if aro_input.strip():
                    try:
                        arousal = float(aro_input.strip())
                    except ValueError:
                        pass

                tags_input = Prompt.ask("Emotion tags (comma-separated) [dim](blank for null)[/dim]", default="")
                if tags_input.strip():
                    emotion_tags = [t.strip() for t in tags_input.split(",") if t.strip()]

                self_input = Prompt.ask("Self-report estimate (1-5) [dim](blank for null)[/dim]", default="")
                if self_input.strip().isdigit() and 1 <= int(self_input) <= 5:
                    self_report = int(self_input.strip())

            summary = f"intent: {intent_val}"
            if amount_minor is not None:
                summary += f"   amount: {amount_minor/100:.2f}"
            if category:
                summary += f"   category: [{category}]"
            if merchant:
                summary += f"   merchant: {merchant}"
            if valence is not None:
                summary += f"   valence: {valence}"
            if arousal is not None:
                summary += f"   arousal: {arousal}"
            if emotion_tags:
                summary += f"   tags: {emotion_tags}"
            if self_report is not None:
                summary += f"   self_report: {self_report}"

            console.print(f"\n  [bold green]{summary}[/bold green]")

            correct = Prompt.ask("correct? [y/n/edit]", choices=["y", "n", "edit"], default="y")
            if correct == "y":
                record = {
                    "id": candidate["id"],
                    "text": candidate["text"],
                    "intent": intent_val,
                    "amount_minor": amount_minor,
                    "category": category,
                    "merchant": merchant,
                    "valence": valence,
                    "arousal": arousal,
                    "emotion_tags": emotion_tags,
                    "self_report_estimate": self_report,
                    "labeler_note": None,
                    "labeled_at": datetime.now(timezone.utc).isoformat()
                }
                with open(labeled_path, "a", encoding="utf-8") as lf:
                    lf.write(json.dumps(record) + "\n")
                current_labeled += 1
                break
            elif correct == "edit" or correct == "n":
                console.print("[yellow]Restarting this candidate...[/yellow]")
                continue

    console.print(f"\n[green]Pool exhausted! Total labeled this run: {current_labeled}. Total skipped: {current_skipped}. Path: {labeled_path}[/green]")

if __name__ == "__main__":
    app()
