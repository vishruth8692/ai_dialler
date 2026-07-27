"""Local Chroma-backed store for the delivery-partner Q&A content.

A CSV upload fully replaces the previously loaded set (reset_store + add_qa_pairs) so the admin
UI always reflects exactly what's in the most recently uploaded file.
"""

from pathlib import Path
from typing import Optional

import chromadb
from sentence_transformers import SentenceTransformer

from app.config import CHROMA_PERSIST_DIR

_EMBEDDING_MODEL_NAME = "paraphrase-multilingual-mpnet-base-v2"
_COLLECTION_NAME = "qa_pairs"

_embedder: Optional[SentenceTransformer] = None
_collection = None


def _get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(_EMBEDDING_MODEL_NAME)
    return _embedder


def warm_up() -> None:
    """Loads the embedding model eagerly (including its Hugging Face Hub metadata checks) so the
    first retrieve() call during a live call doesn't pay that cost - confirmed on a real call to
    add 11+ seconds to the very first reply. Call this once at server startup, not on the request
    path."""
    _get_embedder()


def _get_client() -> chromadb.ClientAPI:
    Path(CHROMA_PERSIST_DIR).mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)


def _get_collection():
    global _collection
    if _collection is None:
        _collection = _get_client().get_or_create_collection(_COLLECTION_NAME)
    return _collection


def reset_store() -> None:
    global _collection
    client = _get_client()
    try:
        client.delete_collection(_COLLECTION_NAME)
    except Exception:
        pass
    _collection = client.get_or_create_collection(_COLLECTION_NAME)


def add_qa_pairs(pairs: list[dict]) -> int:
    """pairs: [{"question": str, "answer": str, "language": str, "type": str}, ...]. type is
    "survey" (asked aloud, in order, as the call script) or "faq" (searchable via retrieve() for
    grounding side questions, never asked aloud) - defaults to "survey" if omitted."""
    if not pairs:
        return 0
    collection = _get_collection()
    embedder = _get_embedder()
    questions = [p["question"] for p in pairs]
    embeddings = embedder.encode(questions, convert_to_numpy=True).tolist()
    ids = [f"qa-{i}" for i in range(len(pairs))]
    metadatas = [
        {
            "answer": p["answer"],
            "language": p.get("language") or "",
            "order": i,
            "type": p.get("type") or "survey",
        }
        for i, p in enumerate(pairs)
    ]
    collection.add(ids=ids, embeddings=embeddings, documents=questions, metadatas=metadatas)
    return len(pairs)


def _row_from(doc: str, meta: dict) -> dict:
    return {
        "question": doc,
        "answer": meta.get("answer", ""),
        "language": meta.get("language", ""),
        "type": meta.get("type") or "survey",
    }


def get_all_pairs() -> list[dict]:
    """Returns ALL pairs (survey + faq) in original CSV upload order - used for the admin table.
    Callers building the spoken call script should filter to type == "survey" (see
    get_survey_questions())."""
    collection = _get_collection()
    if collection.count() == 0:
        return []
    data = collection.get()
    rows = list(zip(data["documents"], data["metadatas"]))
    rows.sort(key=lambda r: r[1].get("order", 0))
    return [_row_from(doc, meta) for doc, meta in rows]


def get_survey_questions() -> list[dict]:
    """Just the type=="survey" rows, in order - the ordered script CallSession asks through."""
    return [p for p in get_all_pairs() if p["type"] == "survey"]


def retrieve(query: str, top_k: int = 3) -> list[dict]:
    """Searches ALL pairs (survey + faq) - a rider can ask about anything in the knowledge base,
    not just the "faq" rows, so this deliberately doesn't filter by type."""
    return retrieve_multi([query], top_k=top_k)[0]


def retrieve_multi(queries: list[str], top_k: int = 3) -> list[list[dict]]:
    """Like retrieve(), but embeds and queries multiple texts in ONE batched call instead of one
    per query. Confirmed on a real production call that each separate embedder.encode() call took
    2-4s on Railway's CPU-only inference (no MPS/GPU there, unlike local dev) - call_session's
    _retrieve_context() was making two such calls per turn, adding up to ~4-8s of pure embedding
    latency to every single reply. Batching amortizes the model's per-call overhead across both
    queries instead of paying it twice."""
    collection = _get_collection()
    count = collection.count()
    if count == 0:
        return [[] for _ in queries]
    embedder = _get_embedder()
    query_embeddings = embedder.encode(queries, convert_to_numpy=True).tolist()
    results = collection.query(query_embeddings=query_embeddings, n_results=min(top_k, count))
    return [
        [_row_from(doc, meta) for doc, meta in zip(docs, metas)]
        for docs, metas in zip(results["documents"], results["metadatas"])
    ]
