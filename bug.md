## Bug Diary

### Bug 1: Router crash on upstream 502
- **Repro:** Asked "where is canada?" as a follow-up; OpenRouter/Nemotron 
  returned a 502 mid-conversation.
- **Root cause:** router_node had no try/except — a transient API failure 
  would crash the whole turn instead of degrading gracefully.
- **Fix:** Wrapped the router LLM call in try/except; on failure, route to 
  an explicit "error" state that returns a handoff message instead of 
  guessing a route.
- **Regression test:** `test_router_handles_llm_failure()` — mocks the LLM 
  call to raise an exception, asserts response has handoff=True and doesn't 
  crash.

### Bug 2: UnicodeEncodeError on Windows (cp1252) crashing indexer and tests
- **Repro:** Running `python -m app.indexer` on Windows printed box-drawing 
  characters (`─`, `✓`, `✗`) that Windows cp1252 codec cannot encode, 
  crashing the script. Same crash appeared in security_tests.py when the 
  model output a non-breaking hyphen (`\u2011`) inside an agent response.
- **Root cause:** Python's default stdout encoding on Windows is cp1252, 
  which does not support Unicode characters outside the Latin-1 range. The 
  indexer and test scripts used f-strings with emoji/box chars directly.
- **Fix (indexer):** Replaced all Unicode box-drawing characters (`─`, `✓`, 
  `✗`) with ASCII equivalents (`-`, `[OK]`, `[NON-AUTH]`).
- **Fix (tests):** Added `.encode('ascii', 'replace').decode('ascii')` to all 
  print statements that render model output, replacing unmappable chars with 
  `?` instead of raising.
- **Lesson:** Always set `PYTHONIOENCODING=utf-8` in the run environment, or 
  defensively encode model output before printing on Windows.

### Bug 3: Router token exhaustion with `with_structured_output` on reasoning models
- **Repro:** Every router call failed with `"Could not parse response content 
  as the length limit was reached"` — even after raising `max_tokens` from 
  1024 → 4096 → 8192, the structured output still could not be parsed.
- **Root cause:** `nvidia/nemotron-3-ultra-550b-a55b` is a reasoning model 
  that spends tokens on an internal `<think>` chain before emitting output. 
  With 277 prompt tokens + 281 reasoning tokens consumed, only ~9 tokens 
  remained for the actual JSON schema response — not enough. At 8192 tokens 
  the model consumed the full budget on reasoning, leaving zero for output.
- **Fix:** Dropped `with_structured_output` entirely for routing. Replaced 
  with a 1-word prompt ("Reply with ONE word: retrieve | tool | both | 
  clarify | chitchat") capped at 512 tokens. The route word is extracted 
  via regex. Order IDs are extracted directly from the user message with 
  `re.search(r"ORD-\d{4,}")` — no LLM budget needed.
- **Lesson:** Structured output / JSON schema modes are expensive for 
  reasoning models. For high-frequency, low-complexity decisions (routing), 
  prefer free-text + regex over schema-constrained output.
