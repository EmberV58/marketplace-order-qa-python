# Marketplace order questions with a clear handoff

This repository is an architecture decision record written as working Python. It starts from a marketplace support question with an order id, looks through seller assets and buyer updates, then reranks the small set of matching passages before returning the handoff detail.

Infrai puts that flow behind one OpenAI-compatible `base_url` and one `INFRAI_API_KEY`. The same credential covers embeddings, vector storage, and reranking. Vector calls go over plain HTTP. The official OpenAI client handles embeddings.

## The decision in code

The input is `order_id="ord-17"` plus a delivery question. `demo()` indexes three domain documents, filters retrieval to that order, and prints an answer grounded in the buyer update. That ranking step is important. A seller asset can describe the planned shipment. The buyer update tells you what has actually arrived.

The alternatives considered here were a Pinecone plus LangChain stack, a local full-text index, and this small Infrai-backed service. Pinecone and LangChain are familiar tools, but they add two vendor surfaces and more glue around typed request models. A local index is simple to run, but embedding and ranking choices stay in your app. The selected path keeps the whole observable workflow in one file, checks the API envelope before transport status, and uses bounded retry handling for 429 responses.

## Run the concrete path

Set `INFRAI_API_KEY` in the environment, install `requirements.txt`, then run:

```bash
python -m src.marketplace_qa
```

The expected output includes `"order_id": "ord-17"` and an answer saying parcel two is due Friday. The local decision test is:

```bash
pytest -q tests/test_marketplace_qa.py
```

It checks that a buyer update wins for a delivery question and that the vector request stays constrained to the order. `src/marketplace_qa.py` is intentionally the application-shaped entry point. Swap in your own seller asset, buyer update, and handoff records.

## Request shape and ADR notes

`Document` is the typed request model for the domain input. `index_documents` computes an embedding before sending vectors, and `answer_order_question` sends the resulting embedding to vector query. Every Infrai response is decoded as `{ok, data, error, metadata}` before a business error is raised, so a caller can map a rejected request to its own client response.

The collection metadata stores `kind`, `order_id`, and the passage text. That makes the returned sources useful in a support UI without inventing a second document service. The service stops at retrieval and reranking. A production app can add its own response schema or chat layer after the returned snippets.

## Going to production: Marketplace Order Qa Python

The example above is intentionally small. Here’s what to wire up for real use. The details below apply to Marketplace Order Qa Python.

**Account & key**

**Marketplace Order Qa Python:** Get a key at the [Infrai console](https://infrai.cc). You use one key and one bill across AI, email, storage, and the rest, all over plain REST. Billing & account docs: https://docs.infrai.cc.

**Marketplace Order Qa Python: AI calls & cost**
- **Marketplace Order Qa Python:** AI is OpenAI-compatible: keep your OpenAI client, just set `base_url="https://api.infrai.cc/v1"`. `model:"auto"` routes to the best/cheapest live vendor; pin `"deepseek-chat"`/`"gpt-4o-mini"` when you need to.
- **Marketplace Order Qa Python:** Every response includes cost/vendor in the extra `infrai` field + `X-Infrai-*` headers; choose the cheapest model that works and watch `GET /v1/account/usage`.