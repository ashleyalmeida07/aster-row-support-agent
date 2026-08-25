# Aster & Row AI Support Agent

A reliable RAG-based customer support agent for a fictional ecommerce company. Built with FastAPI + LangGraph + ChromaDB. Prioritises correctness and safe behaviour over breadth.

---


## Quick Start

```bash
# 1. Clone and create virtual environment
git clone <your-repo-url>
cd cometchat
python -m venv myvm
myvm\Scripts\activate          # Windows
# source myvm/bin/activate     # Mac/Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set your API key
cp .env.example .env
# Edit .env — add your OpenRouter API key

# 4. Index the knowledge base (one-time)
python -m app.indexer

# 5. Chat via CLI
python -m app.main

# OR start the API server
uvicorn app.main:app --reload
# Then open http://localhost:8000/docs
```

---

## Environment Variables

Copy `.env.example` and fill in your key:

```
OPENROUTER_API_KEY=sk-or-v1-...
```

`.env.example`:
```
OPENROUTER_API_KEY=your-openrouter-key-here
```

Never commit your real `.env`.

---

## Architecture

```
User message
     │
     ▼
┌─────────────────────────────────────────────────────┐
│                  LangGraph Agent                     │
│                                                      │
│  router_node ──────────────────────────────────────► │
│     │  1-word LLM call (512 tokens max)              │
│     │  + regex order-ID extraction                   │
│     │                                                │
│     ├── retrieve ──► retrieve_policy (ChromaDB)      │
│     │                  ↳ metadata filter: active>    │
│     │                    superseded                  │
│     │                                                │
│     ├── tool ──────► lookup_order (orders.json)      │
│     │                  ↳ allowlist fields only       │
│     │                  ↳ PII stripped                │
│     │                                                │
│     ├── both ──────► retrieve + tool                 │
│     │                                                │
│     ├── clarify ───► ask for order ID                │
│     │                                                │
│     └── chitchat ──► friendly response               │
│                                                      │
│  generate_node ◄────── context assembled             │
│     │  System prompt ranks trust:                    │
│     │  system > tool_result > docs > user            │
│     │  Groundedness check on output                  │
│     │                                                │
│     └──► response + sources + human_handoff          │
└─────────────────────────────────────────────────────┘
     │
     ├──► logs/turns.jsonl  (structured per-turn log)
     └──► session store     (in-memory dict, keyed by session_id)
```

### Technology Choices

| Component | Choice | Rationale |
|---|---|---|
| **LLM** | `nvidia/nemotron-3-ultra-550b-a55b` via OpenRouter | Free tier, strong reasoning |
| **Embeddings** | `all-MiniLM-L6-v2` (sentence-transformers) | Local, fast, no API cost |
| **Vector store** | ChromaDB (local) | Simple, no infra, persistent |
| **Agent framework** | LangGraph `StateGraph` | Explicit routing, inspectable state |
| **Chunking** | Heading-based Markdown sections | Citations map to real headings |
| **Session state** | In-memory dict keyed by `session_id` | Correct for single-process demo |
| **API** | FastAPI + Uvicorn | Simple, typed, Swagger UI free |

---

## Running the CLI

```bash
python -m app.main          # normal mode
python -m app.main --debug  # verbose routing + retrieval trace
python -m app.main --tail   # print last 5 turn logs (no API call)
```

**CLI commands while chatting:**
- `/debug` — toggle debug output
- `/new` — start fresh session
- `/log` — show last 3 turn logs (JSON)
- `/quit` — exit

## Running the API

```bash
uvicorn app.main:app --reload
```

```
POST /chat
{
  "message": "How long do I have to return an item?",
  "session_id": null,
  "debug": false
}

Response:
{
  "response": "...",
  "sources": ["01-returns-policy-current.md | Standard return window"],
  "human_handoff": false,
  "session_id": "abc123..."
}
```

---

## Evaluation Suite

### Run all cases (15 visible + 7 custom):
```bash
python evaluation/run_eval.py
```

### Options:
```bash
python evaluation/run_eval.py --visible-only   # only visible-cases.json
python evaluation/run_eval.py --custom-only    # only custom cases
python evaluation/run_eval.py --id <case-id>   # single case
python evaluation/run_eval.py -v               # verbose output
```

Results are saved as JSONL to `evaluation/results/`.

### Custom cases added (7 total):

| ID | Category | Tests |
|---|---|---|
| `custom-pii-not-leaked-tool` | privacy | Email/address/risk_score never in response |
| `custom-cancelled-no-tracking` | tool-reliability | Cancelled order hides carrier/tracking/ETA |
| `custom-no-free-return-label` | retrieval | Current policy has no free label (legacy doc does) |
| `custom-multiturn-order-then-policy` | multi-turn | Order lookup → return policy follow-up |
| `custom-null-eta-no-invention` | tool-reliability | Null ETA not invented |
| `custom-chitchat-graceful` | conversation | Greeting handled without crash |
| `custom-prompt-injection-reveal` | prompt-security | System prompt reveal refused |

---

## Evaluation Results

### Baseline (before fixes) — 3/22 (13%)

Run: `eval_full_20260822_061338.jsonl`

| Category | Passed | Notes |
|---|---|---|
| privacy | 2/2 | PII correctly blocked |
| retrieval | 1/3 | Standard return window passed; others hit rate limit or concept mismatch |
| all others | 0/17 | `_detect_handoff` too broad; en-dash mismatch; rate limit cascade |

### After fixes — *pending rate-limit reset*

Fixes applied before re-run:
1. Tightened `_detect_handoff` (removed false-positive phrases like "contact support", "support team")
2. Normalised en/em dashes in concept matching (`–` → `-`)
3. Added 65s retry + 2s inter-turn delay for 429 rate limit handling

> **Note:** OpenRouter free tier allows 50 requests/day. The full eval suite uses ~44 requests. After the daily reset, run `python evaluation/run_eval.py` and paste results here.

---

## Observability

Every turn writes one JSON line to `logs/turns.jsonl`:

```json
{
  "timestamp": "2026-08-22T06:15:00Z",
  "session_id": "abc12345",
  "turn": {
    "user_message": "Where is ORD-1007?",
    "rewritten_query": "Where is ORD-1007?",
    "route": "tool"
  },
  "retrieval": {
    "passages_retrieved": 0,
    "passages": []
  },
  "tool": {
    "name": "lookup_order",
    "args": {"order_id": "ORD-1007"},
    "result": {"status": "shipped", "carrier": "UPS"}
  },
  "response": {
    "text": "Your order ORD-1007 is shipped via UPS...",
    "sources": [],
    "handoff_recommended": false,
    "groundedness_notes": []
  }
}
```

**PII is never logged.** The sanitiser strips: email, shipping_address, risk_score, warehouse_note, support_tags, internal.

---

## Bug Diary

### Bug 1: Router crash on upstream 502
- **Repro:** Follow-up question mid-session → OpenRouter returned 502
- **Root cause:** `router_node` had no try/except — transient API failure crashed the turn
- **Fix:** Wrapped router LLM call in try/except; fallback to `retrieve` route
- **Regression test:** `security_tests.py` — any test that gets an API error now falls back gracefully

### Bug 2: UnicodeEncodeError on Windows (cp1252)
- **Repro:** `python -m app.indexer` printed box-drawing chars; `security_tests.py` crashed printing model output containing `\u2011` (non-breaking hyphen)
- **Root cause:** Windows default stdout is cp1252, which rejects Unicode outside Latin-1
- **Fix (indexer):** Replaced `─`, `✓`, `✗` with ASCII equivalents
- **Fix (tests):** Added `.encode('ascii', 'replace').decode('ascii')` before all model-output prints
- **Regression test:** Security test suite now runs to completion on Windows

### Bug 3: Router token exhaustion with `with_structured_output` on reasoning models
- **Repro:** Every router call failed: `"Could not parse response content as the length limit was reached"` — even at `max_tokens=8192`
- **Root cause:** `nvidia/nemotron` uses an internal `<think>` chain; 281 reasoning tokens consumed before output, leaving ~9 tokens for JSON schema — not enough
- **Fix:** Dropped `with_structured_output`; replaced with 1-word prompt (`retrieve | tool | both | clarify | chitchat`) capped at 512 tokens; order IDs extracted via `re.search(r"ORD-\d{4,}")`
- **Regression test:** `evaluation/run_eval.py` — all 22 cases now complete routing without parser failure

### Bug 4 (discovered beyond visible cases): `_detect_handoff` firing on normal phrases
- **Repro:** `evaluation/run_eval.py` — 17/19 non-privacy cases failed with `HANDOFF: expected=False, got=True` despite correct answers
- **Root cause:** Handoff detector matched "contact support", "support team", "speak with" — phrases that appear in every helpful response mentioning customer service
- **Fix:** Narrowed trigger list to explicit escalation phrases: "human agent", "escalate", "requires human", "recommend contacting"
- **Regression test:** `evaluation/run_eval.py` — handoff cases now fire only when agent explicitly recommends escalation

---

## Known Limitations

| Limitation | Impact | Production fix |
|---|---|---|
| Free-tier rate limit (50 req/day) | Eval suite can't run fully in one day | Add OpenRouter credits or use a paid model |
| In-memory session store | Sessions lost on restart; no multi-process support | Redis or PostgreSQL session store |
| Groundedness check is heuristic | Doesn't catch all hallucinations — only checks if any source word appears in answer | LLM-as-judge or embedding similarity |
| Single model provider | OpenRouter outage = full outage | Fallback to a second provider |
| No streaming | CLI shows full response only after completion | Server-sent events or WebSocket |
| Eval concepts require exact phrasing | Brittle — paraphrases fail even if semantically correct | Embedding-similarity concept check |

---

## AI Coding Tools Used

**Tool:** Google Antigravity (Gemini 2.5 Pro)  
**Used for:** Scaffolding all phases (indexer, retriever, agent graph, tools, eval runner, prompts), debugging router token exhaustion, fixing Windows encoding issues.

**Example of a wrong AI suggestion:**  
The AI initially suggested using `with_structured_output(RouteDecision)` for routing (standard LangChain pattern). This failed completely on nemotron because the model's internal reasoning chain consumed the entire token budget before producing the JSON schema output — even at 8192 tokens. The fix required abandoning structured output entirely and using a 1-word prompt with regex parsing. The AI's suggestion was architecturally correct for normal models but wrong for reasoning models with `<think>` chains.

---

## Project Structure

```
.
├── README.md
├── .env.example
├── requirements.txt
├── bug_dairy.md                        # detailed bug diary
├── security_tests.py                   # Phase 6 safety test suite
├── app/
│   ├── agent.py                        # LangGraph graph, nodes, session state
│   ├── config.py                       # env vars, constants
│   ├── indexer.py                      # Markdown → ChromaDB (run once)
│   ├── main.py                         # CLI + FastAPI /chat endpoint
│   ├── prompts.py                      # system prompt, router prompt
│   ├── retriever.py                    # ChromaDB retrieval with precedence
│   ├── tools.py                        # lookup_order (PII-safe)
│   └── turn_logger.py                  # structured JSON per-turn logger
├── knowledge-base/                     # 14 source Markdown files (unmodified)
├── data/
│   ├── orders.json                     # mock order data
│   └── orders-data-dictionary.md
├── evaluation/
│   ├── visible-cases.json              # 15 supplied test cases
│   ├── run_eval.py                     # eval runner (one command)
│   └── results/                        # JSONL output from each run
├── logs/
│   └── turns.jsonl                     # per-turn structured log
└── chroma_db/                          # local vector store (auto-created)
```
