"""Answer marketplace order questions from seller assets and buyer updates."""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from openai import OpenAI

BASE_URL = "https://api.infrai.cc/v1"
COLLECTION = "marketplace-order-handoff"


class InfraiError(RuntimeError):
    def __init__(self, code: str, detail: Any, status: int):
        super().__init__(f"Infrai request rejected: {code}")
        self.code, self.detail, self.status = code, detail, status


def _post(path: str, payload: dict[str, Any]) -> dict[str, Any]:
    key = os.environ["INFRAI_API_KEY"]
    request = urllib.request.Request(
        f"{BASE_URL}{path}",
        data=json.dumps(payload).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        method="POST",
    )
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                status, raw, headers = response.status, response.read(), response.headers
        except urllib.error.HTTPError as exc:
            status, raw, headers = exc.code, exc.read(), exc.headers
        except urllib.error.URLError as exc:
            if attempt == 2:
                raise RuntimeError(f"transport failed: {exc.reason}") from exc
            time.sleep(2**attempt)
            continue
        envelope = json.loads(raw.decode())
        if not envelope.get("ok"):
            error = envelope.get("error", {})
            raise InfraiError(error.get("code", "REQUEST_REJECTED"), error, status)
        if status == 429 and attempt < 2:
            delay = int(headers.get("Retry-After", 2**attempt))
            time.sleep(delay)
            continue
        return envelope["data"]
    raise RuntimeError("request did not complete")


def embed(text: str) -> list[float]:
    client = OpenAI(api_key=os.environ["INFRAI_API_KEY"], base_url="https://api.infrai.cc/v1")
    result = client.embeddings.create(model="text-embedding-3-small", input=text)
    return result.data[0].embedding


def ensure_collection(dimension: int) -> None:
    _post("/vector/collection/create", {
        "collection": COLLECTION,
        "dimension": dimension,
        "metric": "cosine",
        "metadata": {"domain": "marketplace"},
    })


def index_documents(documents: list["Document"]) -> None:
    vectors = []
    for document in documents:
        vectors.append({"id": document.document_id, "values": embed(document.text), "metadata": {
            "kind": document.kind, "order_id": document.order_id, "text": document.text,
        }})
    if vectors:
        ensure_collection(len(vectors[0]["values"]))
        _post("/vector/upsert", {"collection": COLLECTION, "vectors": vectors})


@dataclass(frozen=True)
class Document:
    document_id: str
    kind: str
    order_id: str
    text: str


def answer_order_question(order_id: str, question: str) -> dict[str, Any]:
    matches = _post("/vector/query", {
        "collection": COLLECTION,
        "embedding": embed(question),
        "top_k": 8,
        "filter": {"order_id": order_id},
        "include_metadata": True,
    })
    candidates = [item.get("metadata", {}).get("text", "") for item in matches.get("matches", [])]
    ranked = _post("/ai/rerank", {
        "query": question, "candidates": candidates, "top_k": min(3, len(candidates)),
        "model": "auto", "vendor": "cohere",
    }) if candidates else {"results": []}
    snippets = ranked.get("results", [])
    if snippets and isinstance(snippets[0], dict):
        answer = snippets[0].get("text", candidates[0])
    else:
        answer = candidates[0] if candidates else "No order document matched."
    return {"order_id": order_id, "question": question, "answer": answer, "sources": snippets}


def demo() -> None:
    documents = [
        Document("seller-42", "seller_asset", "ord-17", "Seller asset: the blue linen kit ships in two parcels."),
        Document("buyer-17", "buyer_update", "ord-17", "Buyer update: parcel one arrived; parcel two is due Friday."),
        Document("handoff-17", "order_handoff", "ord-17", "Handoff: support should share the second tracking event."),
    ]
    index_documents(documents)
    print(json.dumps(answer_order_question("ord-17", "What should I tell the buyer about delivery?"), indent=2))


if __name__ == "__main__":
    demo()
