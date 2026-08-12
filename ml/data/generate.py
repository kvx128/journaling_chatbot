import os
import json
import random
import subprocess
import logging
from pathlib import Path
from typing import Optional
import typer

from ml.data.models import validate_batch
from ml.data.dedupe import SeenSet

app = typer.Typer()
logging.basicConfig(level=logging.INFO, format="%(message)s")

BASE_DIR = Path(__file__).parent
SEEDS_DIR = BASE_DIR / "seeds"
SCHEMAS_DIR = BASE_DIR / "schemas"
GENERATED_DIR = BASE_DIR / "generated"


def build_prompt(task: str, seeds: list, batch_size: int) -> str:
    prompt = f"Generate exactly {batch_size} NEW, DIVERSE examples for the '{task}' task.\n"
    prompt += (
        "Use English only (no Hinglish or code-switching). "
        "Include a mix of terse and verbose phrasings, and include realistic mistakes "
        "or typos in some examples. 2026 India context is fine but not required in every example.\n\n"
    )

    if task == "finance":
        prompt += (
            "CRITICAL: Money amounts should represent Indian Rupees (INR), but the output "
            "`amount_minor` field must be the integer number of paise (rupees * 100). "
            "E.g. Rs 5.50 -> 550, 1.2k -> 120000.\n\n"
        )

    prompt += "Here are some inspiration examples (do not copy these verbatim):\n"
    for s in seeds:
        prompt += json.dumps(s) + "\n"

    return prompt


@app.command()
def generate(
    task: str = typer.Option(..., "--task", help="router, finance, or journal"),
    batches: int = typer.Option(3, "--batches", help="Number of batches to generate"),
    batch_size: int = typer.Option(20, "--batch-size", help="Examples per batch"),
    model: str = typer.Option("gemini-3.6-flash-medium", "--model", help="Model ID"),
    agy_path: Optional[str] = typer.Option(None, "--agy-path", envvar="AGY_PATH", help="Path to agy binary")
):
    if not agy_path:
        logging.error(
            "No agy binary path provided. Pass --agy-path or set the AGY_PATH env var "
            "to the agy executable (e.g. C:\\Users\\<you>\\AppData\\Local\\agy\\bin\\agy.exe)."
        )
        raise typer.Exit(1)

    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    generated_file = GENERATED_DIR / f"{task}.jsonl"
    seed_file = SEEDS_DIR / f"{task}_seeds.jsonl"
    schema_file = SCHEMAS_DIR / f"{task}_batch.schema.json"

    if not seed_file.exists():
        logging.error(f"Seed file not found: {seed_file}")
        raise typer.Exit(1)

    if not schema_file.exists():
        logging.error(f"Schema file not found: {schema_file}")
        raise typer.Exit(1)

    seen_set = SeenSet()
    total_existing = 0

    if generated_file.exists():
        with open(generated_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                    if "text" in obj:
                        seen_set.add_and_check(obj["text"])
                        total_existing += 1
                except Exception:
                    pass

    all_seeds = []
    with open(seed_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    all_seeds.append(json.loads(line))
                except Exception:
                    pass

    for b in range(1, batches + 1):
        sample_seeds = random.sample(all_seeds, min(6, len(all_seeds)))
        prompt = build_prompt(task, sample_seeds, batch_size)

        cmd = [
            agy_path,
            "-p", prompt,
            "--model", model,
            "--json-schema", str(schema_file.resolve()),
            "--output-format", "json",
            "--print-timeout", "5m"
        ]

        try:
            res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", timeout=360)
        except subprocess.TimeoutExpired:
            logging.error(f"[{task}] batch {b}/{batches}: timeout after 360s, skipping.")
            continue
        except FileNotFoundError:
            logging.error(f"[{task}] batch {b}/{batches}: agy binary not found at '{agy_path}'. Aborting.")
            raise typer.Exit(1)

        if res.returncode != 0:
            logging.error(f"[{task}] batch {b}/{batches}: agy failed with code {res.returncode}. Skipping. stderr: {res.stderr.strip()[:200]}")
            continue

        try:
            out_obj = json.loads(res.stdout)
        except json.JSONDecodeError:
            logging.error(f"[{task}] batch {b}/{batches}: failed to parse JSON from stdout. Skipping.")
            continue

        if out_obj.get("status") != "SUCCESS":
            logging.error(f"[{task}] batch {b}/{batches}: status != SUCCESS. Skipping.")
            continue

        structured_data = out_obj.get("structured_output")
        if not structured_data:
            logging.error(f"[{task}] batch {b}/{batches}: missing structured_output. Skipping.")
            continue

        valid_models = validate_batch(task, structured_data)

        generated_count = len(structured_data.get("examples", []))
        valid_count = len(valid_models)

        new_items = []
        for vm in valid_models:
            if seen_set.add_and_check(vm.text):
                new_items.append(vm)

        new_count = len(new_items)

        if new_items:
            with open(generated_file, "a", encoding="utf-8") as f:
                for vm in new_items:
                    try:
                        f.write(vm.model_dump_json() + "\n")
                    except AttributeError:
                        f.write(vm.json() + "\n")
                    total_existing += 1

        logging.info(
            f"[{task}] batch {b}/{batches}: {generated_count} generated, "
            f"{valid_count} valid, {new_count} new, written {new_count} (total so far: {total_existing})"
        )

    logging.info(f"Finished {task}. Total examples in {generated_file.name}: {total_existing}")


if __name__ == "__main__":
    app()
