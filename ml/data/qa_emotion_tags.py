"""Cross-check Gemini-generated (text, valence, arousal, emotion_tags) examples
for internal consistency using the same circumplex-model lookup used as the
runtime fallback (shared/extraction/va_tags.py).

This does NOT require Gemini's tag words to literally match the fallback's
8-word octant vocabulary (Gemini legitimately uses richer words like "grief"
or "burnout" that aren't in that small set). Instead it checks SIGN agreement:
does the example's own valence/arousal (also assigned by Gemini, in the same
call) agree in polarity with the well-known emotional connotation of the tag
words it chose? A text scored valence=+0.7 but tagged "grief" is an internal
contradiction regardless of vocabulary — that's what this catches.

Unknown tag words are skipped (no polarity asserted), not rejected — this is
a precision-over-recall filter for confident mislabels, not a vocabulary gate.
"""
from __future__ import annotations

import json
from pathlib import Path

import typer

app = typer.Typer(help="QA-check generated journal examples against VA/tag consistency")

# (valence_sign, arousal_sign): +1, -1, or 0 (no strong prior on that axis).
# Sign is None to mean the axis isn't checked for that word (rare, only used
# for a couple of intentionally ambiguous words).
#
# Arousal is only asserted for words whose PRIMARY dictionary meaning is
# itself an energy level (calm/relaxed vs. excited/panicked) — the rest are
# left at 0 (unchecked). An earlier version asserted arousal broadly and
# flagged ~19% of a real generated batch; inspecting the flags showed the
# valence calls were all correct and only the hardcoded arousal sign was
# wrong (e.g. "relief" and "gratitude" can spike high-arousal or settle
# low-arousal depending on context — the word alone doesn't determine it).
# Valence, by contrast, is stable per word regardless of context.
_TAG_POLARITY: dict[str, tuple[int, int]] = {
    # positive valence, high arousal (energy level is the word's core meaning)
    "excitement": (1, 1), "elation": (1, 1), "thrill": (1, 1),
    # positive valence, low arousal (ditto)
    "contentment": (1, -1), "calm": (1, -1), "relaxation": (1, -1),
    # positive valence, arousal context-dependent
    "joy": (1, 0), "enthusiasm": (1, 0), "pride": (1, 0),
    "satisfaction": (1, 0), "relief": (1, 0), "gratitude": (1, 0),
    "hope": (1, 0), "love": (1, 0), "optimism": (1, 0),
    # negative valence, high arousal (energy level is the word's core meaning)
    "anger": (-1, 1), "frustration": (-1, 1), "anxiety": (-1, 1),
    "fear": (-1, 1), "panic": (-1, 1), "irritation": (-1, 1),
    "overwhelm": (-1, 1), "embarrassment": (-1, 1),
    # negative valence, low arousal (ditto — genuinely about low energy)
    "tiredness": (-1, -1), "sleepiness": (-1, -1),
    # negative valence, arousal context-dependent
    "stress": (-1, 0), "worry": (-1, 0), "jealousy": (-1, 0),
    "disgust": (-1, 0), "contempt": (-1, 0), "resentment": (-1, 0),
    "sadness": (-1, 0), "grief": (-1, 0), "exhaustion": (-1, 0),
    "burnout": (-1, 0), "loneliness": (-1, 0), "boredom": (-1, 0),
    "hopelessness": (-1, 0), "shame": (-1, 0), "guilt": (-1, 0),
    "disappointment": (-1, 0), "numbness": (-1, 0), "apathy": (-1, 0),
    # ambiguous on both axes — not asserted
    "surprise": (0, 0), "confusion": (0, 0), "nostalgia": (0, 0),
    "curiosity": (0, 0),
}


def _normalize(tag: str) -> str:
    return tag.strip().lower()


_ALIASES = {
    "happy": "joy", "happiness": "joy", "excited": "excitement",
    "sad": "sadness", "angry": "anger", "frustrated": "frustration",
    "anxious": "anxiety", "afraid": "fear", "scared": "fear",
    "tired": "tiredness", "exhausted": "exhaustion", "drained": "exhaustion",
    "lonely": "loneliness", "bored": "boredom", "worried": "worry",
    "stressed": "stress", "grateful": "gratitude", "content": "contentment",
    "relaxed": "relaxation", "relieved": "relief", "proud": "pride",
    "ashamed": "shame", "guilty": "guilt", "disappointed": "disappointment",
    "hopeless": "hopelessness", "hopeful": "hope", "irritated": "irritation",
    "overwhelmed": "overwhelm", "confused": "confusion", "curious": "curiosity",
    "embarrassed": "embarrassment", "numb": "numbness", "apathetic": "apathy",
    "sleepy": "sleepiness", "jealous": "jealousy",
}


def _polarity_for_tag(tag: str) -> tuple[int, int] | None:
    t = _normalize(tag)
    t = _ALIASES.get(t, t)
    return _TAG_POLARITY.get(t)


def check_example(valence: float, arousal: float, emotion_tags: list[str]) -> list[str]:
    """Return a list of human-readable mismatch reasons; empty means consistent."""
    reasons = []
    v_sign = 1 if valence > 0.1 else (-1 if valence < -0.1 else 0)
    a_sign = 1 if arousal > 0.1 else (-1 if arousal < -0.1 else 0)

    for tag in emotion_tags:
        polarity = _polarity_for_tag(tag)
        if polarity is None:
            continue
        exp_v, exp_a = polarity
        if exp_v != 0 and v_sign != 0 and exp_v != v_sign:
            reasons.append(f"tag '{tag}' implies valence {exp_v:+d} but example has {valence:+.2f}")
        if exp_a != 0 and a_sign != 0 and exp_a != a_sign:
            reasons.append(f"tag '{tag}' implies arousal {exp_a:+d} but example has {arousal:+.2f}")

    return reasons


@app.command()
def qa(
    input_file: str = typer.Option("ml/data/generated/journal.jsonl", help="Generated examples to check"),
    clean_out: str = typer.Option("ml/data/generated/journal_qa_clean.jsonl", help="Examples that passed"),
    flagged_out: str = typer.Option("ml/data/generated/journal_qa_flagged.jsonl", help="Examples that failed, with reasons"),
):
    examples = []
    with open(input_file, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))

    clean = []
    flagged = []
    for ex in examples:
        reasons = check_example(ex["valence"], ex["arousal"], ex.get("emotion_tags") or [])
        if reasons:
            flagged.append({**ex, "_qa_reasons": reasons})
        else:
            clean.append(ex)

    Path(clean_out).parent.mkdir(parents=True, exist_ok=True)
    with open(clean_out, "w", encoding="utf-8") as f:
        for ex in clean:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    with open(flagged_out, "w", encoding="utf-8") as f:
        for ex in flagged:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    tagged = [e for e in examples if e.get("emotion_tags")]
    typer.echo(f"Total examples:        {len(examples)}")
    typer.echo(f"With non-empty tags:   {len(tagged)}")
    typer.echo(f"Passed QA:             {len(clean)}")
    typer.echo(f"Flagged (mismatch):    {len(flagged)}")
    typer.echo(f"Clean output:          {clean_out}")
    typer.echo(f"Flagged output:        {flagged_out}")


if __name__ == "__main__":
    app()
