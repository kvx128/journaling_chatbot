from __future__ import annotations

from fastapi.testclient import TestClient


def test_create_transaction_manual(client: TestClient):
    payload = {
        "amount_minor": 15000,
        "category": "TRANSPORT",
        "direction": "debit",
        "merchant": "Uber",
        "source": "api",
    }
    response = client.post("/finance/transactions", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["amount_minor"] == 15000
    assert data["category"] == "TRANSPORT"
    assert data["confirmed"] is True


def test_finance_summary_aggregation(client: TestClient):
    txns = [
        {"amount_minor": 10000, "category": "GROCERIES", "direction": "debit"},
        {"amount_minor": 20000, "category": "GROCERIES", "direction": "debit"},
        {"amount_minor": 15000, "category": "TRANSPORT", "direction": "debit"},
        {"amount_minor": 50000, "category": "OTHER", "direction": "credit"},
    ]
    for txn in txns:
        res = client.post("/finance/transactions", json=txn)
        assert res.status_code == 200

    response = client.get("/finance/summary?date_range=THIS_MONTH")
    assert response.status_code == 200
    data = response.json()

    assert data["total_debit_minor"] == 45000
    assert data["total_credit_minor"] == 50000
    assert data["net_minor"] == 5000
    assert data["transaction_count"] == 4

    by_cat = {item["category"]: item for item in data["by_category"]}
    assert by_cat["GROCERIES"]["total_minor"] == 30000
    assert by_cat["GROCERIES"]["count"] == 2
    assert by_cat["TRANSPORT"]["total_minor"] == 15000
    assert by_cat["TRANSPORT"]["count"] == 1


def test_chat_finance_log_and_verify(client: TestClient):
    chat_payload = {"message": "spent 350 on food delivery"}
    response = client.post("/chat", json=chat_payload)
    assert response.status_code == 200
    chat_data = response.json()

    assert chat_data["intent"] == "FINANCE_LOG"
    structured = chat_data.get("structured_data", {})
    assert "transaction_id" in structured
    assert structured["amount_minor"] == 35000
    assert structured["category"] == "FOOD_DELIVERY"

    summary_res = client.get("/finance/summary?category=FOOD_DELIVERY")
    assert summary_res.status_code == 200
    summary_data = summary_res.json()

    found = False
    for cat_item in summary_data["by_category"]:
        if cat_item["category"] == "FOOD_DELIVERY":
            assert cat_item["total_minor"] >= 35000
            found = True
            break
    assert found


def test_unauthorized_missing_api_key(client: TestClient):
    from fastapi.testclient import TestClient
    from orchestrator.main import app

    unauth_client = TestClient(app)
    response = unauth_client.get("/finance/summary")
    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid or missing X-API-Key header"
