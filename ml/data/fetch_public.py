"""Download and normalize public valence/arousal datasets.

Both sources are human-annotated (not model-generated), which is why they are
worth using: they are independent of the Gemini-generated synthetic data.

Scales differ per source and are normalized to a shared -1..1 range:
  EmoBank  : 1..5 (single mean per dimension)
  FB VA    : 1..9 (two annotators, averaged)
"""
from __future__ import annotations

import csv
import io
import json
from pathlib import Path

import httpx

OUT_DIR = Path(__file__).parent / "public"

EMOBANK_URL = "https://github.com/JULIELab/EmoBank/raw/master/corpus/emobank.csv"
FB_VA_URL = (
    "https://github.com/wwbp/additional_data_sets/raw/master/valence_arousal/"
    "dataset-fb-valence-arousal-anon.csv"
)


def _rescale(value: float, lo: float, hi: float) -> float:
    """Map a value on [lo, hi] onto [-1, 1]."""
    return round((2.0 * (value - lo) / (hi - lo)) - 1.0, 4)


def _fetch(url: str) -> str:
    resp = httpx.get(url, timeout=120.0, follow_redirects=True)
    resp.raise_for_status()
    return resp.text


def load_emobank() -> list[dict]:
    rows = csv.DictReader(io.StringIO(_fetch(EMOBANK_URL)))
    out = []
    for r in rows:
        text = (r.get("text") or "").strip()
        if not text:
            continue
        try:
            valence = _rescale(float(r["V"]), 1.0, 5.0)
            arousal = _rescale(float(r["A"]), 1.0, 5.0)
        except (KeyError, ValueError):
            continue
        out.append(
            {"text": text, "valence": valence, "arousal": arousal, "source": "emobank"}
        )
    return out


def load_fb_va() -> list[dict]:
    rows = csv.DictReader(io.StringIO(_fetch(FB_VA_URL)))
    out = []
    for r in rows:
        text = (r.get("Anonymized Message") or "").strip()
        if not text:
            continue
        try:
            valence = _rescale((float(r["Valence1"]) + float(r["Valence2"])) / 2, 1.0, 9.0)
            arousal = _rescale((float(r["Arousal1"]) + float(r["Arousal2"])) / 2, 1.0, 9.0)
        except (KeyError, ValueError):
            continue
        out.append(
            {"text": text, "valence": valence, "arousal": arousal, "source": "fb_va"}
        )
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    combined: list[dict] = []

    for name, loader in (("emobank", load_emobank), ("fb_va", load_fb_va)):
        try:
            records = loader()
        except Exception as exc:
            print(f"  {name}: FAILED ({exc})")
            continue
        path = OUT_DIR / f"{name}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"  {name}: {len(records)} records -> {path.name}")
        combined.extend(records)

    combined_path = OUT_DIR / "va_combined.jsonl"
    with open(combined_path, "w", encoding="utf-8") as f:
        for rec in combined:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\nCombined: {len(combined)} records -> {combined_path}")
    if combined:
        vals = [r["valence"] for r in combined]
        aros = [r["arousal"] for r in combined]
        print(f"  valence range: {min(vals):+.2f} .. {max(vals):+.2f}")
        print(f"  arousal range: {min(aros):+.2f} .. {max(aros):+.2f}")


if __name__ == "__main__":
    main()
