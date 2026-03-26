# TradingAgents Documentation Design

**Date:** 2026-03-26
**Status:** Approved

## Goal

Produce two documentation files for new developers onboarding to the TradingAgents codebase:

1. `docs/quickstart.md` — 1-page narrative overview with a single end-to-end Mermaid flowchart
2. `docs/architecture.md` — Full reference covering all major subsystems

## Audience

New developers joining the project. Documents should build mental models progressively, explain "why" not just "what", and define LangGraph-specific terms (StateGraph, ToolNode, conditional_edges) on first use. Code references use `file:line` format.

## Document 1: `docs/quickstart.md`

A concise entry point — readable in under 5 minutes.

### Contents
- "What is TradingAgents?" — 2–3 sentence summary of the project
- One end-to-end Mermaid flowchart (`graph TD`) tracing the full execution path:
  `propagate()` → Analyst Team → Researcher Debate → Research Manager → Trader → Risk Debate → Risk Judge → BUY/SELL/HOLD
  - Includes debate loop notation (Bull↔Bear cycle, Risky↔Safe↔Neutral cycle)
- Brief plain-English explanation of each stage (1 short paragraph)
- Pointer to `docs/architecture.md` for depth

## Document 2: `docs/architecture.md`

Full reference organized into 6 sections.

### Section 1: Agent Pipeline
- Each agent's role, inputs, outputs, and which LLM (deep vs quick) it uses
- Agents: Market Analyst, Social Media Analyst, News Analyst, Fundamentals Analyst, Bull Researcher, Bear Researcher, Research Manager, Trader, Risky/Safe/Neutral Analyst, Risk Judge
- Key file pointers per agent

### Section 2: LangGraph State Flow
- Mermaid diagram (`graph TD`) of the LangGraph StateGraph node-and-edge structure
  - Analyst nodes with tool-call loops and message-clear nodes
  - Conditional edges for debate cycling
- Explanation of state objects:
  - `AgentState` — accumulates all reports across the full pipeline
  - `InvestDebateState` — tracks bull/bear debate history and round count
  - `RiskDebateState` — tracks risky/safe/neutral debate history and round count
- File: `tradingagents/agents/utils/agent_states.py`

### Section 3: Data Layer
- Mermaid diagram (`graph LR`) of vendor routing:
  tool call → `agent_utils.py` → `interface.py` → vendor selection → fallback chain → result
- Tool categories: `core_stock_apis`, `technical_indicators`, `fundamental_data`, `news_data`
- Vendor options per category (yfinance, alpha_vantage, openai, google, local)
- Fallback behavior: primary vendor attempted first; on failure, remaining vendors tried in order
- File: `tradingagents/dataflows/interface.py`

### Section 4: Configuration
- `DEFAULT_CONFIG` fields explained:
  - LLM settings: `llm_provider`, `deep_think_llm`, `quick_think_llm`, `backend_url`
  - Debate settings: `max_debate_rounds`, `max_risk_discuss_rounds`
  - Data vendors: `data_vendors` (category-level), `tool_vendors` (tool-level override)
- How to override: pass modified copy to `TradingAgentsGraph(config=...)`
- File: `tradingagents/default_config.py`

### Section 5: Memory & Reflection
- ChromaDB-backed `FinancialSituationMemory` — one instance per role (bull, bear, trader, invest judge, risk manager)
- When `reflect_and_remember()` is called: after trade completes, with `returns_losses` as feedback
- What is stored: role-specific lessons from past decisions, retrieved on future runs
- File: `tradingagents/graph/reflection.py`, `tradingagents/agents/utils/memory.py`

### Section 6: Key File Map
Table format: File | Purpose | Entry point / key class or function

| File | Purpose | Key symbol |
|------|---------|------------|
| `tradingagents/graph/trading_graph.py` | Main orchestrator | `TradingAgentsGraph`, `propagate()` |
| `tradingagents/graph/setup.py` | LangGraph wiring | `GraphSetup.setup_graph()` |
| `tradingagents/graph/conditional_logic.py` | Routing decisions | `ConditionalLogic` |
| `tradingagents/graph/propagation.py` | Initial state creation | `Propagator` |
| `tradingagents/graph/reflection.py` | Post-trade learning | `Reflector` |
| `tradingagents/graph/signal_processing.py` | Extract BUY/SELL/HOLD | `SignalProcessor` |
| `tradingagents/agents/utils/agent_states.py` | State TypedDicts | `AgentState`, `InvestDebateState`, `RiskDebateState` |
| `tradingagents/agents/utils/agent_utils.py` | Tool wrappers | `get_stock_data`, `get_news`, etc. |
| `tradingagents/dataflows/interface.py` | Vendor routing | `route_to_vendor()` |
| `tradingagents/default_config.py` | All runtime settings | `DEFAULT_CONFIG` |
| `cli/main.py` | Interactive CLI | `main()` |

## Diagrams

Three Mermaid diagrams total:

| # | Location | Type | Shows |
|---|----------|------|-------|
| 1 | `quickstart.md` | `graph TD` | End-to-end pipeline with debate loops |
| 2 | `architecture.md` §2 | `graph TD` | LangGraph StateGraph nodes/edges/conditionals |
| 3 | `architecture.md` §3 | `graph LR` | Data layer vendor routing and fallback |

## Out of Scope

- `trade.py` / Alpaca live trading integration
- `test.py` benchmarking details
- Per-vendor API implementation details (Alpha Vantage internals, yfinance internals, etc.)

## Conventions

- Code references: `path/to/file.py:line` format
- Config keys: inline code (e.g., `"llm_provider"`)
- Mermaid: `graph TD` for pipelines, `graph LR` for data routing
- LangGraph terms defined on first use
