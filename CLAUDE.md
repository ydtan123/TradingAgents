# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

TradingAgents is a multi-agent LLM framework that simulates a trading firm's research and risk management workflow. Specialized AI agents collaborate to analyze stocks and produce BUY/SELL/HOLD decisions.

## Commands

### Installation
```bash
pip install -e .
```

### Run the interactive CLI
```bash
python -m cli.main
```

### Run a programmatic analysis (see `main.py` for reference)
```bash
python main.py
```

### Run tests
```bash
pytest                                         # run all tests
pytest tests/agents/analysts/                 # run a specific directory
pytest tests/agents/analysts/test_analyst_team.py  # run a single file
pytest tests/agents/analysts/test_analyst_team.py::TestRunAnalyst::test_no_tool_calls_returns_report  # run a single test
```

Tests live under `tests/` and use pytest with `asyncio_mode = auto` (see `pytest.ini`). Heavy third-party packages (pandas, yfinance, chromadb, etc.) are stubbed in `tests/agents/analysts/conftest.py` so the analyst-team tests run without real dependencies. `test.py` (root) is a separate ad-hoc timing benchmark for `get_stock_stats_indicators_window()`.

## Architecture

### Agent Pipeline (sequential)

```
Analyst Team → Researcher Team (debate) → Trader → Risk Team (debate) → Portfolio Manager
```

1. **Analyst Team** — Four specialists run **concurrently via `asyncio.gather`** inside a single LangGraph node (`analyst_team.py`), each running an independent tool-call loop (up to 10 iterations). Results are written directly to state fields; messages are not shared between analysts.
   - Market Analyst → `market_report`: technical indicators (MACD, RSI, SMA, etc.)
   - Social Media Analyst → `sentiment_report`: company news/Reddit sentiment
   - News Analyst → `news_report`: macro/global news
   - Fundamentals Analyst → `fundamentals_report`: balance sheet, income statement, cash flow

2. **Researcher Team** — Bull and Bear researchers debate over analyst reports for `max_debate_rounds` cycles. A Research Manager synthesizes into an investment recommendation.

3. **Trader** — Generates a trading plan from the research synthesis.

4. **Risk Management Team** — Risky, Neutral, and Safe analysts debate the trader's plan for `max_risk_discuss_rounds` cycles. A Risk Manager issues final approval/rejection.

5. **Portfolio Manager** — Executes via Alpaca API if configured.

### Key Source Files

| File | Purpose |
|------|---------|
| `tradingagents/graph/trading_graph.py` | Main class `TradingAgentsGraph`; initializes LLMs and orchestrates everything. Entry point: `propagate(ticker, date)` |
| `tradingagents/graph/setup.py` | `GraphSetup` — builds the LangGraph `StateGraph`, wires nodes and edges |
| `tradingagents/graph/conditional_logic.py` | Routing logic: controls analyst continuation and debate round cycling |
| `tradingagents/graph/propagation.py` | Initializes `AgentState`, `InvestDebateState`, `RiskDebateState` |
| `tradingagents/graph/reflection.py` | Post-trade learning: `reflect_and_remember()` updates ChromaDB memory |
| `tradingagents/agents/utils/agent_states.py` | TypedDict definitions for all state objects passed through the graph |
| `tradingagents/agents/utils/agent_utils.py` | Tool wrappers that agents call (get_stock_data, get_indicators, etc.) |
| `tradingagents/default_config.py` | `DEFAULT_CONFIG` dict — the single source of truth for all runtime settings |
| `tradingagents/dataflows/interface.py` | Vendor selection: routes data requests to yfinance, Alpha Vantage, etc. |
| `cli/main.py` | Interactive CLI using Rich + questionary |

### Configuration

All behavior is controlled via `DEFAULT_CONFIG` in `tradingagents/default_config.py`:

```python
DEFAULT_CONFIG = {
    "llm_provider": "openai",          # openai | anthropic | google | openrouter | ollama
    "deep_think_llm": "o4-mini",       # Used for complex analysis tasks
    "quick_think_llm": "gpt-4o-mini",  # Used for fast/simple tasks
    "backend_url": "https://api.openai.com/v1",
    "max_debate_rounds": 1,            # Bull vs Bear research debate cycles
    "max_risk_discuss_rounds": 1,      # Risk team discussion cycles
    "max_recur_limit": 100,            # LangGraph recursion limit
    "data_vendors": {
        "core_stock_apis": "yfinance",       # Options: yfinance, alpha_vantage, local
        "technical_indicators": "yfinance",  # Options: yfinance, alpha_vantage, local
        "fundamental_data": "alpha_vantage", # Options: openai, alpha_vantage, local
        "news_data": "alpha_vantage",        # Options: openai, alpha_vantage, google, local
    },
    "tool_vendors": {
        # Per-tool overrides; take precedence over data_vendors category defaults
        # e.g. "get_news": "openai"
    },
}
```

Pass a modified copy to `TradingAgentsGraph(config=...)` to override defaults.

### State Flow

The LangGraph state dict (`AgentState`) accumulates reports as the graph traverses nodes. Sub-states (`InvestDebateState`, `RiskDebateState`) track debate history and round counts — defined in `tradingagents/agents/utils/agent_states.py`. `ConditionalLogic` routes debate cycles based on `count` fields in these sub-states. `Propagator` initializes the state; `SignalProcessor` extracts `BUY/SELL/HOLD` from the final verbose decision using a quick LLM call.

After each run, `propagate()` writes a full state JSON log to `eval_results/<ticker>/TradingAgentsStrategy_logs/`. Call `reflect_and_remember(returns)` after observing real returns to update ChromaDB memories for each agent role.

### Data Vendors

`tradingagents/dataflows/` abstracts multiple data sources. `interface.py` routes calls to the correct vendor based on `data_vendors` (category-level) and `tool_vendors` (per-tool override) config keys. `local.py` provides offline caching. Most data tools accept a ticker + date and return structured strings consumed directly by LLM agents. Tool wrapper functions in `agent_utils.py` are the single call site for all data — agents never import from `dataflows` directly.

### Environment Variables

Copy `.env.example` to `.env` and populate:
- `OPENAI_API_KEY` — required for default config
- `ALPHA_VANTAGE_API_KEY` — required for fundamental and news data
- `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY` — if using those providers
- `ALPACA_API_KEY`, `ALPACA_SECRET_KEY` — for live/paper trading via `trade.py`
