from __future__ import annotations

from fastapi.testclient import TestClient


def test_crisis_triggers(client: TestClient):
    trigger_phrases = [
        "I just want to kill myself",
        "feeling like I want to die",
        "ready to end it all tonight",
    ]

    for phrase in trigger_phrases:
        res = client.post("/chat", json={"message": phrase})
        assert res.status_code == 200
        data = res.json()
        assert data["crisis_flagged"] is True
        assert "1800-599-0019" in data["reply"]
        assert data["intent"] == "UNKNOWN"


def test_crisis_false_positives_avoided(client: TestClient):
    safe_phrases = [
        "this deadline is killing me",
        "dying to see that new movie",
        "I could kill for a coffee right now",
    ]

    for phrase in safe_phrases:
        res = client.post("/chat", json={"message": phrase})
        assert res.status_code == 200
        data = res.json()
        assert data["crisis_flagged"] is False


def test_smalltalk_routing(client: TestClient):
    res = client.post("/chat", json={"message": "hey"})
    assert res.status_code == 200
    data = res.json()
    assert data["intent"] == "SMALLTALK"
    assert data["crisis_flagged"] is False


def test_mood_checkin_routing(client: TestClient):
    res = client.post("/chat", json={"message": "feeling really anxious and drained today"})
    assert res.status_code == 200
    data = res.json()
    assert data["intent"] == "MOOD_CHECKIN"
    assert data["crisis_flagged"] is False
    assert "mood_entry_id" in data.get("structured_data", {})


def test_mood_checkin_structured_crisis(client: TestClient):
    payload = {
        "self_report": 1,
        "note": "I want to end it all",
    }
    res = client.post("/mood/checkin", json=payload)
    assert res.status_code == 400
    assert "1800-599-0019" in res.json()["detail"]
