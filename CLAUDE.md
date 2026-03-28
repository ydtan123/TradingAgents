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
python test.py  # ad-hoc timing test for technical indicators
```

There is no formal test suite. `test.py` benchmarks `get_stock_stats_indicators_window()` execution time.

## Architecture

### Agent Pipeline (sequential)

```
Analyst Team → Researcher Team (debate) → Trader → Risk Team (debate) → Portfolio Manager
```

1. **Analyst Team** — Four specialists run in parallel, each producing a report:
   - Market Analyst: technical indicators (MACD, RSI, SMA, etc.)
   - Social Media Analyst: Reddit sentiment
   - News Analyst: recent news events
   - Fundamentals Analyst: balance sheet, income statement, cash flow

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
    "data_vendors": {
        "core_stock_apis": "yfinance",
        "technical_indicators": "yfinance",
        "fundamental_data": "alpha_vantage",
        "news_data": "alpha_vantage",
    },
}
```

Pass a modified copy to `TradingAgentsGraph(config=...)` to override defaults.

### State Flow

The LangGraph state dict (`AgentState`) accumulates reports as the graph traverses nodes. Sub-states (`InvestDebateState`, `RiskDebateState`) track debate history and round counts. These are defined in `tradingagents/agents/utils/agent_states.py`.

### Data Vendors

`tradingagents/dataflows/` abstracts multiple data sources. `interface.py` selects the implementation based on config. `local.py` provides offline caching. Most data tools accept a ticker + date and return structured strings consumed directly by LLM agents.

### Environment Variables

Copy `.env.example` to `.env` and populate:
- `OPENAI_API_KEY` — required for default config
- `ALPHA_VANTAGE_API_KEY` — required for fundamental and news data
- `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY` — if using those providers
- `ALPACA_API_KEY`, `ALPACA_SECRET_KEY` — for live/paper trading via `trade.py`
