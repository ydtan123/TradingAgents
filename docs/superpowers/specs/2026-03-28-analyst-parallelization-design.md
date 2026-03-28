# Analyst Parallelization Design

**Date:** 2026-03-28
**Status:** Implemented

## Overview

Replace the four sequential analyst nodes (Market, Social Media, News, Fundamentals) and their associated tool nodes and message-clear nodes with a single `Analyst Team` node that runs all selected analysts concurrently using `asyncio.gather`.

## Motivation

The four analysts are fully independent — they read the same inputs (`trade_date`, `company_of_interest`) and write to separate output fields (`market_report`, `sentiment_report`, `news_report`, `fundamentals_report`). Running them sequentially wastes wall-clock time. Parallelizing them cuts analyst phase latency from ~4x single-analyst time to ~1x.

## Architecture

### Before

```
START → Market Analyst ⇄ tools_market → Msg Clear Market
      → Social Analyst ⇄ tools_social → Msg Clear Social
      → News Analyst ⇄ tools_news → Msg Clear News
      → Fundamentals Analyst ⇄ tools_fundamentals → Msg Clear Fundamentals
      → Bull Researcher
```

12 nodes total for the analyst phase.

### After

```
START → Analyst Team → Bull Researcher
```

1 node for the analyst phase.

## Components

### New file: `tradingagents/agents/analysts/analyst_team.py`

Implements `create_analyst_team(llm, selected_analysts)`.

Note: `tool_nodes` was dropped from the signature — tools are inlined directly in `_ANALYST_TOOLS` inside the module. `GraphSetup` no longer stores or receives `tool_nodes`.

Returns an async LangGraph node function that:

1. Extracts `trade_date`, `company_of_interest`, and `messages` from state.
2. Builds one async coroutine per selected analyst using the existing `create_X_analyst(llm)` prompt + chain logic.
3. Each coroutine:
   - Maintains its own local message list (isolated from other analysts).
   - Runs the tool-call loop: `ainvoke` LLM → if tool calls present, execute tools via `run_in_executor` (tools are sync) → repeat until no tool calls remain.
   - Returns `(report_field, report_string)` on success.
   - Catches all exceptions and returns `(report_field, "")` for fault tolerance — failed analysts produce an empty report and the pipeline continues.
4. Runs all coroutines with `asyncio.gather`.
5. Returns a state update dict with all four report fields.

### Tool-call loop (per analyst)

The loop is bounded by `_MAX_TOOL_ITERATIONS = 10` to prevent infinite loops if the LLM keeps requesting tools.

```python
_MAX_TOOL_ITERATIONS = 10

async def _run_analyst(analyst_type, chain, tools_by_name, initial_messages):
    try:
        messages = list(initial_messages)
        loop = asyncio.get_running_loop()
        for _ in range(_MAX_TOOL_ITERATIONS):
            result = await chain.ainvoke({"messages": messages})
            messages.append(result)
            if not result.tool_calls:
                return _REPORT_FIELDS[analyst_type], result.content
            for tool_call in result.tool_calls:
                tool_result = await loop.run_in_executor(
                    None, tools_by_name[tool_call["name"]].invoke, tool_call["args"]
                )
                messages.append(
                    ToolMessage(content=str(tool_result), tool_call_id=tool_call["id"])
                )
        return _REPORT_FIELDS[analyst_type], ""  # max iterations hit
    except Exception:
        return _REPORT_FIELDS[analyst_type], ""
```

Tools are sync (yfinance, Alpha Vantage calls), so they run in a thread executor to avoid blocking the event loop.

## Changes to Existing Files

### `tradingagents/graph/setup.py`

- Remove: analyst node creation loop (lines 60–86), all tool node and delete node registration, sequential edge wiring between analysts (lines 130–153).
- Add: single `create_analyst_team(...)` call, `workflow.add_node("Analyst Team", ...)`, `workflow.add_edge(START, "Analyst Team")`, `workflow.add_edge("Analyst Team", "Bull Researcher")`.

### `tradingagents/graph/conditional_logic.py`

- Remove: `should_continue_market`, `should_continue_social`, `should_continue_news`, `should_continue_fundamentals` — no longer needed.

### `tradingagents/agents/utils/agent_states.py`

- No changes. The four report fields remain as-is.

### `tradingagents/agents/__init__.py`

- Export `create_analyst_team` alongside existing analyst exports.

## Fault Tolerance

- Each analyst coroutine catches all exceptions internally.
- A failed analyst writes an empty string to its report field.
- Downstream agents (Bull/Bear Researcher) receive whatever reports succeeded.
- No changes to downstream state handling are needed.

## Out of Scope

- Parallelizing the researcher debate or risk management team.
- Retry logic for failed analysts.
- Timeouts per analyst.
