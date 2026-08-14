"""Build the SFT training set for FT-JRN (mood valence/arousal/tags).

Combines two sources, deliberately weighted toward the real one:
  - ml/data/public/va_combined.jsonl   12,956 records, human-annotated (EmoBank +
    Facebook posts VA study). No emotion_tags in the source, so tags is [].
  - ml/data/generated/journal.jsonl    125 records, Gemini-synthetic, has emotion_tags.

Output is TRL's standard "conversational" SFT format: one JSON object per line with a
"messages" list (system/user/assistant), so the training script can hand it straight to
SFTTrainer and let the tokenizer's own chat template handle formatting.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from ml.data.dedupe import SeenSet

ROOT = Path(__file__).parent.parent.parent
PUBLIC_VA = ROOT / "ml" / "data" / "public" / "va_combined.jsonl"
GENERATED_JOURNAL = ROOT / "ml" / "data" / "generated" / "journal_qa_clean.jsonl"
OUT_DIR = Path(__file__).parent / "data"

VAL_SIZE = 500
SEED = 42

SYSTEM_PROMPT = (
    "You analyze the emotional content of a short piece of text. "
    "Respond with a single JSON object containing exactly these fields: "
    '"valence" (float, -1.0 to 1.0, negative to positive feeling), '
    '"arousal" (float, -1.0 to 1.0, calm to activated), and '
    '"emotion_tags" (a list of short lowercase words naming the emotion, '
    "empty list if none apply). No other text, just the JSON object."
)


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        print(f"  (missing, skipped: {path})")
        return []
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def to_example(text: str, valence: float, arousal: float, tags: list[str]) -> dict:
    completion = json.dumps(
        {"valence": round(valence, 2), "arousal": round(arousal, 2), "emotion_tags": tags},
        ensure_ascii=False,
    )
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
            {"role": "assistant", "content": completion},
        ]
    }


def main() -> None:
    random.seed(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    seen = SeenSet()
    examples: list[dict] = []
    counts = {"public": 0, "generated": 0, "duplicate": 0}

    for rec in _load_jsonl(PUBLIC_VA):
        text = rec.get("text", "").strip()
        if not text or not seen.add_and_check(text):
            counts["duplicate"] += 1
            continue
        examples.append(to_example(text, rec["valence"], rec["arousal"], []))
        counts["public"] += 1

    for rec in _load_jsonl(GENERATED_JOURNAL):
        text = rec.get("text", "").strip()
        if not text or not seen.add_and_check(text):
            counts["duplicate"] += 1
            continue
        examples.append(
            to_example(text, rec["valence"], rec["arousal"], rec.get("emotion_tags") or [])
        )
        counts["generated"] += 1

    random.shuffle(examples)

    val_size = min(VAL_SIZE, len(examples) // 10)
    val_set = examples[:val_size]
    train_set = examples[val_size:]

    train_path = OUT_DIR / "journal_train.jsonl"
    val_path = OUT_DIR / "journal_val.jsonl"

    with open(train_path, "w", encoding="utf-8") as f:
        for ex in train_set:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    with open(val_path, "w", encoding="utf-8") as f:
        for ex in val_set:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"  public (human-annotated): {counts['public']}")
    print(f"  generated (synthetic):    {counts['generated']}")
    print(f"  dropped as duplicates:    {counts['duplicate']}")
    print(f"\nTrain: {len(train_set)} -> {train_path}")
    print(f"Val:   {len(val_set)} -> {val_path}")


if __name__ == "__main__":
    main()
