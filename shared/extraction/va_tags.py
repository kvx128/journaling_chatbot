"""Deterministic emotion-tag derivation from valence/arousal.

The FT-JRN model's own `emotion_tags` output is unreliable: only ~1% of its
training examples carried a non-empty tag (the two big human-annotated data
sources, EmoBank and the Facebook VA study, supply valence/arousal scores but
no categorical labels), so the model learned to emit `[]` almost regardless
of input. This module fills that gap deterministically using the circumplex
model of affect (Russell, 1980), which maps a point in valence/arousal space
to a named emotion by its angle around the circle. It's a fallback/stopgap,
not a replacement for a model that actually learned to tag — the real fix is
more tagged training data, with this same lookup used as a consistency check
on that data during generation.
"""
from __future__ import annotations

import math

# Octant labels at 45-degree increments, starting at 0 degrees (pure positive
# valence, zero arousal) and going counterclockwise, matching a standard
# circumplex diagram layout.
_OCTANT_TAGS = [
    "content",    # 0   deg: V+, A0
    "excited",    # 45  deg: V+, A+
    "tense",      # 90  deg: V0, A+
    "angry",      # 135 deg: V-, A+
    "sad",        # 180 deg: V-, A0
    "depressed",  # 225 deg: V-, A-
    "tired",      # 270 deg: V0, A- (low energy, not quite positive — "calm"
                  # belongs on the positive-valence side, see 315 deg)
    "relaxed",    # 315 deg: V+, A-
]

# Below this distance from the origin, valence/arousal are both close enough
# to zero that no octant label is meaningfully distinct.
NEUTRAL_MAGNITUDE = 0.15


def tags_from_va(valence: float, arousal: float) -> list[str]:
    magnitude = math.hypot(valence, arousal)
    if magnitude < NEUTRAL_MAGNITUDE:
        return ["neutral"]

    angle_deg = math.degrees(math.atan2(arousal, valence)) % 360
    octant = round(angle_deg / 45) % 8
    return [_OCTANT_TAGS[octant]]
