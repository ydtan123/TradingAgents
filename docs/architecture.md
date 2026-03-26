# TradingAgents — Architecture Reference

Full reference for new developers. See [docs/quickstart.md](quickstart.md) for the one-page overview.

---

## 1. Agent Pipeline

Each agent is a LangGraph node — a Python function that receives the current state, does work, and returns state updates. Agents are wired together by `GraphSetup.setup_graph()` in `tradingagents/graph/setup.py`.

### LLMs

Two LLM instances are created in `TradingAgentsGraph.__init__()` (`tradingagents/graph/trading_graph.py:76`):

| Name | Config key | Default | Used for |
|------|-----------|---------|----------|
| `quick_thinking_llm` | `"quick_think_llm"` | `gpt-4o-mini` | All analysts, researchers, trader, risk analysts |
| `deep_thinking_llm` | `"deep_think_llm"` | `o4-mini` | Research Manager, Risk Judge |

### Analyst Team

Analysts run **sequentially** (Market → Social → News → Fundamentals). Each analyst loops on tool calls until it has all the data it needs, then writes its report and clears its messages from the shared state.

| Agent | Node name | LLM | Tools called | Output field |
|-------|-----------|-----|-------------|--------------|
| Market Analyst | `"Market Analyst"` | quick | `get_stock_data`, `get_all_indicators` | `market_report` |
| Social Media Analyst | `"Social Analyst"` | quick | `get_news` | `sentiment_report` |
| News Analyst | `"News Analyst"` | quick | `get_news`, `get_global_news`, `get_insider_sentiment`, `get_insider_transactions` | `news_report` |
| Fundamentals Analyst | `"Fundamentals Analyst"` | quick | `get_fundamentals`, `get_balance_sheet`, `get_cashflow`, `get_income_statement` | `fundamentals_report` |

Source files: `tradingagents/agents/analysts/`

**Tool call loop:** Each analyst uses a conditional edge (`ConditionalLogic.should_continue_<type>` in `tradingagents/graph/conditional_logic.py`). If the last message has `tool_calls`, execution goes to `tools_<type>` (a `ToolNode`) and back to the analyst. Otherwise it goes to `Msg Clear <Type>`, which deletes the analyst's messages from the shared `messages` list, then passes to the next analyst.

### Researcher Team (Debate)

After all analysts finish, the Bull and Bear Researchers debate over the reports.

| Agent | Node name | LLM | Memory | Output fields |
|-------|-----------|-----|--------|---------------|
| Bull Researcher | `"Bull Researcher"` | quick | `bull_memory` | `investment_debate_state.bull_history` |
| Bear Researcher | `"Bear Researcher"` | quick | `bear_memory` | `investment_debate_state.bear_history` |
| Research Manager | `"Research Manager"` | deep | `invest_judge_memory` | `investment_debate_state.judge_decision`, `investment_plan` |

**Debate routing** (`ConditionalLogic.should_continue_debate`):
- If `investment_debate_state["count"] >= 2 × max_debate_rounds` → go to Research Manager
- Else if last response starts with `"Bull"` → go to Bear Researcher
- Else → go to Bull Researcher

Source files: `tradingagents/agents/researchers/`, `tradingagents/agents/managers/research_manager.py`

### Trader

| Agent | Node name | LLM | Memory | Output field |
|-------|-----------|-----|--------|--------------|
| Trader | `"Trader"` | quick | `trader_memory` | `trader_investment_plan` |

Reads `investment_plan` and generates a concrete trading proposal with position sizing rationale.

Source file: `tradingagents/agents/trader/trader.py`

### Risk Management Team (Debate)

Three risk analysts debate the trader's plan in rotation: Risky → Safe → Neutral → Risky → ...

| Agent | Node name | LLM | Output fields |
|-------|-----------|-----|---------------|
| Risky Analyst | `"Risky Analyst"` | quick | `risk_debate_state.risky_history` |
| Safe Analyst | `"Safe Analyst"` | quick | `risk_debate_state.safe_history` |
| Neutral Analyst | `"Neutral Analyst"` | quick | `risk_debate_state.neutral_history` |
| Risk Judge | `"Risk Judge"` | deep | `risk_debate_state.judge_decision`, `final_trade_decision` |

**Risk debate routing** (`ConditionalLogic.should_continue_risk_analysis`):
- If `risk_debate_state["count"] >= 3 × max_risk_discuss_rounds` → go to Risk Judge
- If `latest_speaker` starts with `"Risky"` → go to Safe Analyst
- If `latest_speaker` starts with `"Safe"` → go to Neutral Analyst
- Else → go to Risky Analyst

Source files: `tradingagents/agents/risk_mgmt/`, `tradingagents/agents/managers/risk_manager.py`
