from __future__ import annotations

from shared.extraction.va_tags import tags_from_va


def test_neutral_near_origin():
    assert tags_from_va(0.0, 0.0) == ["neutral"]
    assert tags_from_va(0.1, -0.1) == ["neutral"]


def test_high_valence_high_arousal_is_excited():
    assert tags_from_va(0.8, 0.8) == ["excited"]


def test_low_valence_high_arousal_is_angry():
    assert tags_from_va(-0.8, 0.8) == ["angry"]


def test_low_valence_low_arousal_is_depressed():
    assert tags_from_va(-0.8, -0.8) == ["depressed"]


def test_high_valence_low_arousal_is_relaxed():
    assert tags_from_va(0.8, -0.8) == ["relaxed"]


def test_pure_positive_valence_is_content():
    assert tags_from_va(1.0, 0.0) == ["content"]


def test_pure_negative_valence_is_sad():
    assert tags_from_va(-1.0, 0.0) == ["sad"]


def test_pure_negative_arousal_is_tired():
    assert tags_from_va(0.0, -1.0) == ["tired"]


def test_mildly_negative_valence_strongly_negative_arousal_is_tired():
    # Regression case: this should read as low-energy/disappointed, not "calm"
    # (which belongs on the positive-valence side of the low-arousal axis).
    assert tags_from_va(-0.25, -0.88) == ["tired"]


def test_always_returns_non_empty_list():
    for v in [-1.0, -0.5, 0.0, 0.5, 1.0]:
        for a in [-1.0, -0.5, 0.0, 0.5, 1.0]:
            tags = tags_from_va(v, a)
            assert isinstance(tags, list)
            assert len(tags) >= 1
