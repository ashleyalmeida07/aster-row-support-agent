"""
Heading-based Markdown chunker + ChromaDB indexer.

Reads all .md files from knowledge-base/, parses YAML front matter,
splits by ## headings, and stores chunks in ChromaDB with rich metadata
for document-precedence filtering.

Usage:
    python -m app.indexer          # Build/rebuild the index
    python -m app.indexer --query "return window"  # Quick test query
"""

import re
import sys
import yaml
import chromadb
from pathlib import Path
from typing import Optional

from app.config import KNOWLEDGE_BASE_DIR, CHROMA_PERSIST_DIR, CHROMA_COLLECTION_NAME


# ---------------------------------------------------------------------------
# Front-matter parsing
# ---------------------------------------------------------------------------

FRONT_MATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


def parse_front_matter(text: str) -> tuple[dict, str]:
    """Extract YAML front matter and return (metadata_dict, body_text)."""
    match = FRONT_MATTER_RE.match(text)
    if not match:
        return {}, text
    raw_yaml = match.group(1)
    metadata = yaml.safe_load(raw_yaml) or {}
    body = text[match.end():]
    return metadata, body


# ---------------------------------------------------------------------------
# Heading-based chunking
# ---------------------------------------------------------------------------

HEADING_RE = re.compile(r"^(#{1,3})\s+(.+)$", re.MULTILINE)


def chunk_by_headings(body: str, source_file: str, doc_meta: dict) -> list[dict]:
    """
    Split markdown body by ## headings.
    
    Each chunk contains:
      - id: unique chunk identifier
      - text: the chunk content (heading + body until next heading)
      - metadata: merged doc-level + chunk-level metadata
    """
    chunks = []
    
    # Find all heading positions
    headings = list(HEADING_RE.finditer(body))
    
    if not headings:
        # No headings — treat the whole body as one chunk
        chunk_text = body.strip()
        if chunk_text:
            chunks.append(_make_chunk(
                source_file=source_file,
                doc_meta=doc_meta,
                heading="(full document)",
                heading_level=0,
                text=chunk_text,
                chunk_index=0,
            ))
        return chunks
    
    # Text before the first heading (if any) — include with doc title
    preamble = body[:headings[0].start()].strip()
    if preamble:
        chunks.append(_make_chunk(
            source_file=source_file,
            doc_meta=doc_meta,
            heading="(preamble)",
            heading_level=0,
            text=preamble,
            chunk_index=0,
        ))
    
    # Each heading section
    for i, match in enumerate(headings):
        heading_level = len(match.group(1))
        heading_text = match.group(2).strip()
        
        # Content is from this heading to the next heading (or end of body)
        start = match.start()
        end = headings[i + 1].start() if i + 1 < len(headings) else len(body)
        chunk_text = body[start:end].strip()
        
        if chunk_text:
            chunks.append(_make_chunk(
                source_file=source_file,
                doc_meta=doc_meta,
                heading=heading_text,
                heading_level=heading_level,
                text=chunk_text,
                chunk_index=len(chunks),
            ))
    
    return chunks


def _make_chunk(
    source_file: str,
    doc_meta: dict,
    heading: str,
    heading_level: int,
    text: str,
    chunk_index: int,
) -> dict:
    """Build a chunk dict with merged metadata."""
    
    status = doc_meta.get("status", "unknown")
    policy_authority = doc_meta.get("policy_authority", "unknown")
    
    # Determine authoritativeness
    is_authoritative = (
        status == "active"
        and policy_authority == "official"
    )
    
    doc_id = doc_meta.get("document_id", source_file)
    chunk_id = f"{doc_id}::{chunk_index}"
    
    return {
        "id": chunk_id,
        "text": text,
        "metadata": {
            "source_file": source_file,
            "document_id": doc_id,
            "title": doc_meta.get("title", ""),
            "heading": heading,
            "heading_level": heading_level,
            "status": status,
            "audience": doc_meta.get("audience", "unknown"),
            "policy_authority": policy_authority,
            "is_authoritative": is_authoritative,
            "effective_date": doc_meta.get("effective_date", ""),
            "supersedes": doc_meta.get("supersedes", ""),
            "superseded_by": doc_meta.get("superseded_by", ""),
        },
    }


# ---------------------------------------------------------------------------
# Index all knowledge-base documents
# ---------------------------------------------------------------------------

def load_and_chunk_all(kb_dir: Optional[Path] = None) -> list[dict]:
    """Load all .md files from knowledge-base and chunk them."""
    kb_dir = kb_dir or KNOWLEDGE_BASE_DIR
    all_chunks = []
    
    md_files = sorted(kb_dir.glob("*.md"))
    if not md_files:
        print(f"WARNING: No .md files found in {kb_dir}")
        return all_chunks
    
    for md_path in md_files:
        raw_text = md_path.read_text(encoding="utf-8")
        doc_meta, body = parse_front_matter(raw_text)
        chunks = chunk_by_headings(body, md_path.name, doc_meta)
        all_chunks.extend(chunks)
        print(f"  [{doc_meta.get('status', '?'):>10}] {md_path.name} -> {len(chunks)} chunks")
    
    return all_chunks


def build_index(kb_dir: Optional[Path] = None, persist_dir: Optional[str] = None) -> chromadb.Collection:
    """
    Build (or rebuild) the ChromaDB collection from knowledge-base docs.
    
    Returns the ChromaDB collection.
    """
    persist_dir = persist_dir or CHROMA_PERSIST_DIR
    
    print(f"Building ChromaDB index at: {persist_dir}")
    print(f"Reading knowledge base from: {kb_dir or KNOWLEDGE_BASE_DIR}")
    print()
    
    # Load and chunk
    all_chunks = load_and_chunk_all(kb_dir)
    print(f"\nTotal chunks: {len(all_chunks)}")
    
    # Initialize ChromaDB
    client = chromadb.PersistentClient(path=persist_dir)
    
    # Delete existing collection if it exists (rebuild)
    try:
        client.delete_collection(CHROMA_COLLECTION_NAME)
        print(f"Deleted existing collection '{CHROMA_COLLECTION_NAME}'")
    except Exception:
        pass
    
    collection = client.get_or_create_collection(
        name=CHROMA_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    
    # Prepare batch data
    ids = [c["id"] for c in all_chunks]
    documents = [c["text"] for c in all_chunks]
    metadatas = []
    for c in all_chunks:
        # ChromaDB metadata values must be str, int, float, or bool
        meta = {}
        for k, v in c["metadata"].items():
            if v is None:
                meta[k] = ""
            elif isinstance(v, bool):
                meta[k] = v
            elif isinstance(v, (int, float)):
                meta[k] = v
            else:
                meta[k] = str(v)
        metadatas.append(meta)
    
    # Add all chunks in one batch
    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
    )
    
    print(f"Indexed {collection.count()} chunks into '{CHROMA_COLLECTION_NAME}'")
    return collection


def get_collection(persist_dir: Optional[str] = None) -> chromadb.Collection:
    """Get the existing ChromaDB collection (for retrieval)."""
    persist_dir = persist_dir or CHROMA_PERSIST_DIR
    client = chromadb.PersistentClient(path=persist_dir)
    return client.get_collection(name=CHROMA_COLLECTION_NAME)


# ---------------------------------------------------------------------------
# Quick test / CLI
# ---------------------------------------------------------------------------

def test_query(query: str, n_results: int = 5):
    """Run a test query against the index and print results."""
    collection = get_collection()
    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )
    
    print(f"\nQuery: '{query}'")
    print(f"{'-' * 80}")
    
    for i in range(len(results["ids"][0])):
        doc_id = results["ids"][0][i]
        meta = results["metadatas"][0][i]
        distance = results["distances"][0][i]
        doc_preview = results["documents"][0][i][:150].replace("\n", " ")
        
        auth_tag = "[AUTH]" if meta.get("is_authoritative") else "[NON-AUTH]"
        print(f"\n  [{i+1}] {auth_tag}  dist={distance:.4f}")
        print(f"      Source: {meta.get('source_file')} | {meta.get('heading')}")
        print(f"      Status: {meta.get('status')} | Authority: {meta.get('policy_authority')}")
        print(f"      Text:   {doc_preview}...")


if __name__ == "__main__":
    args = sys.argv[1:]
    
    if "--query" in args:
        query_idx = args.index("--query")
        query = args[query_idx + 1] if query_idx + 1 < len(args) else "return window"
        test_query(query)
    else:
        build_index()
        print("\n--- Quick verification queries ---")
        test_query("return window")
        test_query("dishwasher tumbler")
        test_query("ship to Germany")
