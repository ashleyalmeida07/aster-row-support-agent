"""
retrieve_policy — RAG retrieval tool over the ChromaDB knowledge base.

Retrieves relevant policy chunks with metadata-based precedence:
- Filters/boosts status:active over status:superseded
- Flags non-authoritative sources (draft, internal-only, no policy_authority)
- Returns source citations (filename + heading) with every result
"""

import chromadb
from typing import Optional

from app.config import CHROMA_PERSIST_DIR, CHROMA_COLLECTION_NAME, RETRIEVAL_TOP_K


def _get_collection() -> chromadb.Collection:
    """Get the existing ChromaDB collection."""
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    return client.get_collection(name=CHROMA_COLLECTION_NAME)


def retrieve_policy(query: str, top_k: int = RETRIEVAL_TOP_K) -> dict:
    """
    Retrieve relevant policy passages from the knowledge base.

    Args:
        query: The user's question or search query about Aster & Row policies.
        top_k: Maximum number of results to return (default 8).

    Returns:
        A dict with:
          - "passages": list of passage dicts, each with:
              - "text": the passage content
              - "source_file": filename (e.g. "01-returns-policy-current.md")
              - "heading": section heading
              - "title": document title
              - "is_authoritative": bool — True if active + official
              - "status": active / superseded / draft
              - "policy_authority": official / none
              - "audience": customer / internal
              - "document_id": the front-matter document_id
              - "relevance_score": similarity score (lower distance = more relevant)
              - "precedence_note": human-readable note about document authority
          - "retrieval_metadata": summary info about the retrieval
    """
    collection = _get_collection()

    # Retrieve more than top_k so we can re-rank after metadata filtering
    fetch_k = min(top_k * 2, 20)

    results = collection.query(
        query_texts=[query],
        n_results=fetch_k,
        include=["documents", "metadatas", "distances"],
    )

    passages = []
    for i in range(len(results["ids"][0])):
        meta = results["metadatas"][0][i]
        distance = results["distances"][0][i]
        text = results["documents"][0][i]

        # Build precedence note
        precedence_note = _build_precedence_note(meta)

        passages.append({
            "text": text,
            "source_file": meta.get("source_file", ""),
            "heading": meta.get("heading", ""),
            "title": meta.get("title", ""),
            "is_authoritative": meta.get("is_authoritative", False),
            "status": meta.get("status", "unknown"),
            "policy_authority": meta.get("policy_authority", "unknown"),
            "audience": meta.get("audience", "unknown"),
            "document_id": meta.get("document_id", ""),
            "relevance_score": round(1.0 - distance, 4),  # Convert distance to similarity
            "precedence_note": precedence_note,
        })

    # Re-rank: authoritative first, then by relevance score
    passages.sort(key=_ranking_key, reverse=True)

    # Trim to requested top_k
    passages = passages[:top_k]

    # Count stats
    auth_count = sum(1 for p in passages if p["is_authoritative"])
    non_auth_count = len(passages) - auth_count

    return {
        "passages": passages,
        "retrieval_metadata": {
            "query": query,
            "total_results": len(passages),
            "authoritative_results": auth_count,
            "non_authoritative_results": non_auth_count,
        },
    }


def _ranking_key(passage: dict) -> tuple:
    """
    Ranking key for re-sorting passages.
    
    Priority order:
    1. is_authoritative (True > False) — active+official docs first
    2. status weighting (active > superseded > draft)
    3. relevance_score (higher is better)
    """
    status_weight = {
        "active": 3,
        "superseded": 1,
        "draft": 0,
    }
    return (
        int(passage["is_authoritative"]),
        status_weight.get(passage["status"], 0),
        passage["relevance_score"],
    )


def _build_precedence_note(meta: dict) -> str:
    """Build a human-readable note about this document's authority level."""
    status = meta.get("status", "unknown")
    authority = meta.get("policy_authority", "unknown")
    audience = meta.get("audience", "unknown")
    superseded_by = meta.get("superseded_by", "")

    if status == "superseded":
        note = f"SUPERSEDED document — replaced by {superseded_by}. Do NOT use as current authority."
        return note
    
    if status == "draft" or authority == "none":
        return "DRAFT / NON-AUTHORITATIVE — not approved for customer answers. Do not cite as policy."
    
    if audience == "internal":
        return "INTERNAL document — contains operational rules for the agent, not customer-facing policy text."
    
    if status == "active" and authority == "official":
        return "AUTHORITATIVE — active official policy. Safe to cite for customer answers."
    
    return f"Status: {status}, Authority: {authority} — verify before citing."
