# TradingAgents — Copilot Instructions

## Project Overview

TradingAgents is a multi-agent LLM framework that simulates a trading firm. Agents collaborate to produce BUY/SELL/HOLD decisions for a given stock ticker and date.

## Commands

```bash
pip install -e .                  # install
python -m cli.main                # interactive CLI
python main.py                    # programmatic run (NVDA example)
pytest tests/                     # run all tests
pytest tests/agents/analysts/     # run a single test module
python test.py                    # benchmark technical indicators (not pytest)
```

## Architecture

### Agent Pipeline (sequential)

```
Analyst Team (parallel) → Researcher Debate → Trader → Risk Debate → Portfolio Manager
```

1. **Analyst Team** — four specialists run concurrently via `asyncio.gather` in `tradingagents/agents/analysts/analyst_team.py`. Each calls data tools, produces a text report, and writes it into `AgentState`.
2. **Researcher Team** — Bull and Bear researchers debate for `max_debate_rounds` cycles; Research Manager synthesizes into a recommendation.
3. **Trader** — converts the recommendation into a trade plan.
4. **Risk Team** — Risky, Safe, and Neutral analysts debate for `max_risk_discuss_rounds` cycles; Risk Judge issues final approval.
5. **Portfolio Manager** — optionally executes via Alpaca.

### Entry Point

```python
ta = TradingAgentsGraph(debug=True, config=config)
state, decision = ta.propagate("NVDA", "2024-05-10")
```

`TradingAgentsGraph` (`tradingagents/graph/trading_graph.py`) initializes LLMs, wires the LangGraph `StateGraph` via `GraphSetup`, and exposes `propagate(ticker, date)`.

### State Objects

Defined in `tradingagents/agents/utils/agent_states.py`:
- `AgentState` (extends `MessagesState`) — accumulates all analyst reports, debate histories, and final decisions as the graph traverses.
- `InvestDebateState` — tracks Bull/Bear debate history and round count.
- `RiskDebateState` — tracks Risky/Safe/Neutral debate history and round count.

### Graph Wiring

- `tradingagents/graph/setup.py` — `GraphSetup.setup_graph()` builds and compiles the `StateGraph`, wires nodes and edges.
- `tradingagents/graph/conditional_logic.py` — `ConditionalLogic` controls debate cycling (checks `state["investment_debate_state"]["count"]` and `state["risk_debate_state"]["count"]`).
- `tradingagents/graph/propagation.py` — initializes state dicts before graph execution.
- `tradingagents/graph/reflection.py` — post-trade: `reflect_and_remember()` stores lessons in ChromaDB.

### Data Vendor Abstraction

`tradingagents/dataflows/interface.py` is a routing layer. Each data function (e.g., `get_stock_data`, `get_news`) is dispatched to the correct vendor implementation based on `config["data_vendors"]` and optional per-tool overrides in `config["tool_vendors"]`.

Vendors: `yfinance`, `alpha_vantage`, `openai`, `google`, `local` (offline CSV cache).

`local` vendor reads pre-downloaded CSVs from `config["data_dir"]`; useful for reproducible backtests without API calls.

### Configuration

All settings live in `tradingagents/default_config.py`. Pass a modified copy to `TradingAgentsGraph`:

```python
config = DEFAULT_CONFIG.copy()
config["llm_provider"] = "anthropic"   # openai | anthropic | google | openrouter | ollama
config["deep_think_llm"] = "claude-opus-4-5"
config["quick_think_llm"] = "claude-haiku-4-5"
config["max_debate_rounds"] = 2
config["data_vendors"]["news_data"] = "google"
ta = TradingAgentsGraph(config=config)
```

`tool_vendors` dict overrides `data_vendors` at the individual tool level.

## Key Conventions

### Analyst field naming
Each analyst type maps to a fixed `AgentState` field: `market → market_report`, `social → sentiment_report`, `news → news_report`, `fundamentals → fundamentals_report`. This mapping is defined inside `_run_analyst()` in `analyst_team.py` and must stay consistent with `AgentState`.

### Tests use extensive stubs
`tests/agents/analysts/conftest.py` stubs out all heavy dependencies (pandas, yfinance, LangChain, ChromaDB, etc.) before any import. When adding tests, register new modules there if they transitively import heavy packages.

### `pytest.ini` sets `asyncio_mode = auto`
All `async def` test functions are automatically treated as async tests — no `@pytest.mark.asyncio` decorator needed (though it's harmless to include).

### Tool wrappers are LangChain tools
Functions in `tradingagents/agents/utils/agent_utils.py` are decorated as LangChain tools and passed as `tools` lists to agent chains. Agents call them by name during their reasoning loop.

## Environment Variables

Copy `.env.example` → `.env`:

| Variable | Required for |
|---|---|
| `OPENAI_API_KEY` | Default LLM config |
| `ALPHA_VANTAGE_API_KEY` | Fundamental and news data |
| `ANTHROPIC_API_KEY` | Anthropic provider |
| `GOOGLE_API_KEY` | Google provider |
| `ALPACA_API_KEY` / `ALPACA_SECRET_KEY` | Live/paper trading via `trade.py` |
