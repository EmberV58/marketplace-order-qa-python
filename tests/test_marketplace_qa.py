from src.marketplace_qa import answer_order_question


def test_buyer_update_wins_for_delivery_question(monkeypatch):
    calls = []

    def fake_post(path, payload):
        calls.append((path, payload))
        if path == "/vector/query":
            return {"matches": [{"metadata": {"text": "Buyer update: parcel two is due Friday."}}, {"metadata": {"text": "Seller asset: ships in two parcels."}}]}
        return {"results": [{"text": "Buyer update: parcel two is due Friday."}]}

    monkeypatch.setattr("src.marketplace_qa._post", fake_post)
    monkeypatch.setattr("src.marketplace_qa.embed", lambda text: [0.1, 0.2])
    result = answer_order_question("ord-17", "When does the second parcel arrive?")
    assert result["answer"] == "Buyer update: parcel two is due Friday."
    assert calls[0][1]["filter"] == {"order_id": "ord-17"}
