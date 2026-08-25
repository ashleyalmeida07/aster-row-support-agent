"""
Main entry point — FastAPI server + CLI for the Aster & Row support agent.

Run as CLI:
    python -m app.main
    python -m app.main --debug        # verbose debug output per turn

Run as API server:
    uvicorn app.main:app --reload
    # POST /chat  {"message": "...", "session_id": "optional", "debug": false}

Observability:
    Turn logs are written to logs/turns.jsonl
    python -m app.main --tail         # print last 5 turn logs
"""

import json
import logging
import sys
from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel

from app.agent import chat, clear_session, list_sessions
from app.turn_logger import get_log_path, tail_log

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-22s | %(levelname)-5s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("aster_row")


# ---------------------------------------------------------------------------
# FastAPI app — Phase 8 interface
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Aster & Row Support Agent",
    description="RAG + order-lookup support agent. POST /chat to interact.",
    version="1.0.0",
)


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    debug: bool = False


class ChatResponse(BaseModel):
    response: str
    sources: list[str] = []
    human_handoff: bool = False
    session_id: str
    debug_log: Optional[dict] = None


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Send a message to the Aster & Row support agent.

    Returns:
        response: The agent's answer
        sources: List of cited sources as "filename | heading"
        human_handoff: True when the agent recommends human assistance
        session_id: Use this in subsequent requests for multi-turn conversation
    """
    result = chat(
        message=request.message,
        session_id=request.session_id,
        debug=request.debug,
    )
    return ChatResponse(
        response=result["response"],
        sources=result.get("sources", []),
        human_handoff=result.get("handoff_recommended", False),
        session_id=result["session_id"],
        debug_log=result.get("debug_log"),
    )


@app.delete("/session/{session_id}")
async def clear_session_endpoint(session_id: str):
    """Clear a conversation session."""
    clear_session(session_id)
    return {"status": "cleared", "session_id": session_id}


@app.get("/sessions")
async def list_sessions_endpoint():
    """List active session IDs."""
    return {"sessions": list_sessions()}


@app.get("/health")
async def health():
    """Health check."""
    return {"status": "ok", "turn_log": get_log_path()}


# ---------------------------------------------------------------------------
# CLI — Phase 8 interface
# ---------------------------------------------------------------------------

def run_cli(debug_mode: bool = False):
    """Run the agent in interactive CLI mode."""
    print("=" * 62)
    print("  Aster & Row Support Agent")
    print("  Type your message and press Enter.")
    print("  Commands: /debug, /new, /log, /quit")
    print(f"  Turn log: {get_log_path()}")
    print("=" * 62)
    print()

    if debug_mode:
        print("  [Debug mode: ON]\n")

    session_id = None

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        # CLI commands
        if user_input.lower() == "/quit":
            print("Goodbye!")
            break
        elif user_input.lower() == "/debug":
            debug_mode = not debug_mode
            print(f"  [Debug mode: {'ON' if debug_mode else 'OFF'}]\n")
            continue
        elif user_input.lower() == "/new":
            session_id = None
            print("  [New session started]\n")
            continue
        elif user_input.lower() == "/log":
            entries = tail_log(3)
            if not entries:
                print("  [No turn logs yet]\n")
            else:
                for e in entries:
                    print(json.dumps(e, indent=2, default=str))
            continue

        # Send to agent
        try:
            result = chat(
                message=user_input,
                session_id=session_id,
                debug=debug_mode,
            )
        except Exception as e:
            print(f"\n  [ERROR] {e}\n")
            logger.exception("Chat error")
            continue

        session_id = result["session_id"]

        # --- Print response ---
        print()
        print(f"Agent: {result['response']}")

        # Sources
        if result.get("sources"):
            print(f"\n  Sources:")
            for s in result["sources"]:
                print(f"    - {s}")

        # Human handoff flag
        if result.get("handoff_recommended"):
            print("\n  [!] Human handoff recommended")

        # Debug log
        if debug_mode and result.get("debug_log"):
            _print_debug(result["debug_log"])

        print()


def _print_debug(debug: dict):
    """Pretty-print the debug log for CLI."""
    print("\n  --- DEBUG ---")

    router = debug.get("router", {})
    print(f"  Route:     {router.get('route', '?')}")
    print(f"  Reasoning: {router.get('reasoning', '?')[:120]}")

    if debug.get("rewritten_query"):
        print(f"  Rewritten: {debug['rewritten_query']}")

    retrieval = debug.get("retrieval", {})
    if retrieval:
        auth = retrieval.get("authoritative_results", 0)
        total = retrieval.get("total_results", 0)
        print(f"  Retrieved: {total} passages ({auth} authoritative)")
        for p in retrieval.get("passages", [])[:4]:
            tag = "[AUTH]    " if p["is_authoritative"] else "[NON-AUTH]"
            print(f"    {tag} {p['source_file']} | {p['heading']} (score={p['relevance_score']})")

    tool = debug.get("tool", {})
    if tool and tool.get("action"):
        print(f"  Tool:      lookup_order → {tool.get('action')} "
              f"(id={tool.get('normalized_id', '?')}, found={tool.get('found', '?')})")

    gen = debug.get("generation", {})
    if gen:
        grounded = gen.get("grounded", True)
        notes = gen.get("groundedness_notes", [])
        print(f"  Grounded:  {grounded}")
        for note in notes:
            print(f"    WARN: {note}")

    print("  --- END DEBUG ---")


if __name__ == "__main__":
    args = sys.argv[1:]

    if "--tail" in args:
        entries = tail_log(5)
        if not entries:
            print("No turn logs found.")
        else:
            for e in entries:
                print(json.dumps(e, indent=2, default=str))
    else:
        debug = "--debug" in args
        run_cli(debug_mode=debug)
