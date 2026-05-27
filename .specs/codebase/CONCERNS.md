# Concerns

## HIGH — Missing Error Handling at Startup

**Area:** Reliability  
**Files:** `src/agenticlog/agent.py` lines 44-58  
**Issue:** LLM client and ChromaDB are initialized at module import time. If LMStudio is not running or `data/vectordb/` doesn't exist, the import fails and the entire app crashes before the user sees any UI.  
**Fix:** Lazy initialization with explicit error messages; health-check before first invocation.  

## HIGH — LMStudio Is a Single Point of Failure

**Area:** Reliability  
**Files:** `src/agenticlog/agent.py`, `src/agenticlog/config.py`  
**Issue:** No retry logic, no circuit breaker, no fallback model. Any LMStudio crash kills all 3 routing paths simultaneously.  
**Fix:** Add retry with exponential backoff on LLM calls; catch `httpx.ConnectError` and surface a user-friendly message in Streamlit.  

## MEDIUM — Hardcoded LLM Credentials in Source Code

**Area:** Security / Portability  
**File:** `src/agenticlog/config.py` lines 18-19  
**Issue:** `OPENAI_API_BASE` and `OPENAI_API_KEY` hardcoded. CLAUDE.md says to use `.env` but code ignores `.env`.  
**Fix:** Load from `os.environ` with fallback to localhost defaults; validate at startup.  

## MEDIUM — Incomplete Test Coverage

**Area:** Quality  
**Missing:**
- `agent_workflow.invoke()` end-to-end (requires LMStudio → integration test, currently absent)
- `app.py` entirely untested
- `config.py` constants untested
- Edge case: LMStudio returns malformed JSON
- Edge case: ChromaDB returns 0 documents (test exists in test_agentic_rag.py but verify coverage)
**Fix:** Add integration test with LMStudio mock at HTTP level; add Streamlit test with `streamlit.testing`.  

## MEDIUM — No Logging Module

**Area:** Observability  
**Files:** `src/agenticlog/rag.py` (print statements lines 140, 176), `src/agenticlog/agent.py`  
**Issue:** `print()` used for operational feedback. No log levels, no structured output, no file logging.  
**Fix:** Replace with `logging.getLogger(__name__)` at each module; add log level config in `config.py`.  

## LOW — Decision Routing Keyword Lists Not Configurable

**Area:** Maintainability  
**File:** `src/agenticlog/agent.py` lines 127-142  
**Issue:** Keyword lists for routing ("explain", "summarize", "search the web", etc.) are hardcoded inline in `passo_decisao_agente`. Adding new triggers requires editing agent logic.  
**Fix:** Move to `config.py` as `KEYWORDS_GERAR`, `KEYWORDS_WEB` lists.  

## LOW — Type Hints Incomplete

**Area:** Maintainability  
**Files:** `src/agenticlog/rag.py`, `src/agenticlog/config.py`, `app.py`  
**Issue:** Functions lack return type annotations; config constants untyped; session_state keys are untyped strings.  
**Fix:** Add return types progressively; annotate config constants with `Final[str]` etc.  

## LOW — No Rate Limiting on DuckDuckGo

**Area:** Reliability  
**File:** `src/agenticlog/agent.py` lines 90-98  
**Issue:** Unlimited requests to unofficial DuckDuckGo API. High query volume could trigger rate limiting or IP blocks.  
**Fix:** Add cooldown between web searches; cache recent results.  

## LOW — FastAPI/Uvicorn Installed but Unused

**Area:** Dependency bloat  
**File:** `requirements.txt`  
**Issue:** `fastapi==0.115.8` and `uvicorn==0.34.0` in requirements but no API server exists.  
**Fix:** Remove if no REST API is planned; or use them if API surface is a roadmap item.  

## INFO — No CI/CD Gate for Coverage

**Area:** Process  
**Issue:** Coverage is measured manually. No automated enforcement of 80% target.  
**Fix:** Add GitHub Actions workflow running `pytest --cov=agenticlog --cov-fail-under=80`.  
