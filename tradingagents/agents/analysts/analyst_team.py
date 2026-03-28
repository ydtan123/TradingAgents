import asyncio
import logging

from langchain_core.messages import ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from tradingagents.agents.utils.agent_utils import (
    get_stock_data,
    get_all_indicators,
    get_news,
    get_global_news,
    get_insider_sentiment,
    get_insider_transactions,
    get_fundamentals,
    get_balance_sheet,
    get_cashflow,
    get_income_statement,
)

logger = logging.getLogger(__name__)

# ── per-analyst configuration ─────────────────────────────────────────────────

_ANALYST_TOOLS = {
    "market": [get_stock_data, get_all_indicators],
    "social": [get_news],
    "news": [get_news, get_global_news, get_insider_sentiment, get_insider_transactions],
    "fundamentals": [get_fundamentals, get_balance_sheet, get_cashflow, get_income_statement],
}

_REPORT_FIELDS = {
    "market": "market_report",
    "social": "sentiment_report",
    "news": "news_report",
    "fundamentals": "fundamentals_report",
}

_SYSTEM_MESSAGES = {
    "market": (
        "You are a trading assistant tasked with analyzing financial markets. Your role is to select the"
        " **most relevant indicators** for a given market condition or trading strategy from the following"
        " list. The goal is to choose up to **8 indicators** that provide complementary insights without"
        " redundancy. Categories and each category's indicators are:\n\n"
        "Moving Averages:\n"
        "- close_50_sma: 50 SMA: A medium-term trend indicator. Usage: Identify trend direction and serve"
        " as dynamic support/resistance. Tips: It lags price; combine with faster indicators for timely signals.\n"
        "- close_200_sma: 200 SMA: A long-term trend benchmark. Usage: Confirm overall market trend and"
        " identify golden/death cross setups. Tips: It reacts slowly; best for strategic trend confirmation"
        " rather than frequent trading entries.\n"
        "- close_10_ema: 10 EMA: A responsive short-term average. Usage: Capture quick shifts in momentum"
        " and potential entry points. Tips: Prone to noise in choppy markets; use alongside longer averages"
        " for filtering false signals.\n\n"
        "MACD Related:\n"
        "- macd: MACD: Computes momentum via differences of EMAs. Usage: Look for crossovers and divergence"
        " as signals of trend changes. Tips: Confirm with other indicators in low-volatility or sideways markets.\n"
        "- macds: MACD Signal: An EMA smoothing of the MACD line. Usage: Use crossovers with the MACD line"
        " to trigger trades. Tips: Should be part of a broader strategy to avoid false positives.\n"
        "- macdh: MACD Histogram: Shows the gap between the MACD line and its signal. Usage: Visualize"
        " momentum strength and spot divergence early. Tips: Can be volatile; complement with additional"
        " filters in fast-moving markets.\n\n"
        "Momentum Indicators:\n"
        "- rsi: RSI: Measures momentum to flag overbought/oversold conditions. Usage: Apply 70/30 thresholds"
        " and watch for divergence to signal reversals. Tips: In strong trends, RSI may remain extreme;"
        " always cross-check with trend analysis.\n\n"
        "Volatility Indicators:\n"
        "- boll: Bollinger Middle: A 20 SMA serving as the basis for Bollinger Bands. Usage: Acts as a"
        " dynamic benchmark for price movement. Tips: Combine with the upper and lower bands to effectively"
        " spot breakouts or reversals.\n"
        "- boll_ub: Bollinger Upper Band: Typically 2 standard deviations above the middle line. Usage:"
        " Signals potential overbought conditions and breakout zones. Tips: Confirm signals with other"
        " tools; prices may ride the band in strong trends.\n"
        "- boll_lb: Bollinger Lower Band: Typically 2 standard deviations below the middle line. Usage:"
        " Indicates potential oversold conditions. Tips: Use additional analysis to avoid false reversal signals.\n"
        "- atr: ATR: Averages true range to measure volatility. Usage: Set stop-loss levels and adjust"
        " position sizes based on current market volatility. Tips: It's a reactive measure, so use it as"
        " part of a broader risk management strategy.\n\n"
        "Volume-Based Indicators:\n"
        "- vwma: VWMA: A moving average weighted by volume. Usage: Confirm trends by integrating price"
        " action with volume data. Tips: Watch for skewed results from volume spikes; use in combination"
        " with other volume analyses.\n\n"
        "Select indicators that provide diverse and complementary information. Avoid redundancy (e.g., do"
        " not select both rsi and stochrsi). Also briefly explain why they are suitable for the given"
        " market context. When you tool call, please use the exact name of the indicators provided above"
        " as they are defined parameters, otherwise your call will fail. Please make sure to call"
        " get_stock_data first to retrieve the CSV that is needed to generate indicators. Then use"
        " get_all_indicators with ALL chosen indicator names in a single call (pass them as a list) —"
        " do NOT call it once per indicator. Write a very detailed and nuanced report of the trends you"
        " observe. Do not simply state the trends are mixed, provide detailed and finegrained analysis"
        " and insights that may help traders make decisions."
        " Make sure to append a Markdown table at the end of the report to organize key points in the"
        " report, organized and easy to read."
    ),
    "social": (
        "You are a social media and company specific news researcher/analyst tasked with analyzing social"
        " media posts, recent company news, and public sentiment for a specific company over the past week."
        " You will be given a company's name your objective is to write a comprehensive long report"
        " detailing your analysis, insights, and implications for traders and investors on this company's"
        " current state after looking at social media and what people are saying about that company,"
        " analyzing sentiment data of what people feel each day about the company, and looking at recent"
        " company news. Use the get_news(query, start_date, end_date) tool to search for company-specific"
        " news and social media discussions. Try to look at all sources possible from social media to"
        " sentiment to news. Do not simply state the trends are mixed, provide detailed and finegrained"
        " analysis and insights that may help traders make decisions."
        " Make sure to append a Markdown table at the end of the report to organize key points in the"
        " report, organized and easy to read."
    ),
    "news": (
        "You are a news researcher tasked with analyzing recent news and trends over the past week."
        " Please write a comprehensive report of the current state of the world that is relevant for"
        " trading and macroeconomics. Use the available tools: get_news(query, start_date, end_date)"
        " for company-specific or targeted news searches, and get_global_news(curr_date, look_back_days,"
        " limit) for broader macroeconomic news. Do not simply state the trends are mixed, provide"
        " detailed and finegrained analysis and insights that may help traders make decisions."
        " Make sure to append a Markdown table at the end of the report to organize key points in the"
        " report, organized and easy to read."
    ),
    "fundamentals": (
        "You are a researcher tasked with analyzing fundamental information over the past week about a"
        " company. Please write a comprehensive report of the company's fundamental information such as"
        " financial documents, company profile, basic company financials, and company financial history"
        " to gain a full view of the company's fundamental information to inform traders. Make sure to"
        " include as much detail as possible. Do not simply state the trends are mixed, provide detailed"
        " and finegrained analysis and insights that may help traders make decisions."
        " Make sure to append a Markdown table at the end of the report to organize key points in the"
        " report, organized and easy to read."
        " Use the available tools: `get_fundamentals` for comprehensive company analysis,"
        " `get_balance_sheet`, `get_cashflow`, and `get_income_statement` for specific financial statements."
    ),
}

_BASE_PROMPT_TEMPLATE = (
    "You are a helpful AI assistant, collaborating with other assistants."
    " Use the provided tools to progress towards answering the question."
    " If you are unable to fully answer, that's OK; another assistant with different tools"
    " will help where you left off. Execute what you can to make progress."
    " If you or any other assistant has the FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** or deliverable,"
    " prefix your response with FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** so the team knows to stop."
    " You have access to the following tools: {tool_names}.\n{system_message}"
    "For your reference, the current date is {current_date}. The company we want to look at is {ticker}"
)

_MAX_TOOL_ITERATIONS = 10


# ── internal helpers ──────────────────────────────────────────────────────────

def _build_chain(analyst_type, llm, current_date, ticker):
    """Build an LLM chain for the given analyst type. Returns (chain, tools_by_name)."""
    tools = _ANALYST_TOOLS[analyst_type]
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", _BASE_PROMPT_TEMPLATE),
            MessagesPlaceholder(variable_name="messages"),
        ]
    ).partial(
        system_message=_SYSTEM_MESSAGES[analyst_type],
        tool_names=", ".join([t.name for t in tools]),
        current_date=current_date,
        ticker=ticker,
    )
    chain = prompt | llm.bind_tools(tools)
    tools_by_name = {t.name: t for t in tools}
    return chain, tools_by_name


async def _run_analyst(analyst_type, chain, tools_by_name, initial_messages):
    """Run a single analyst's tool-call loop asynchronously.

    Returns (report_field_name, report_string). On any exception, returns (field, "").
    """
    try:
        messages = list(initial_messages)
        loop = asyncio.get_running_loop()
        for _ in range(_MAX_TOOL_ITERATIONS):
            result = await chain.ainvoke({"messages": messages})
            messages.append(result)
            if not result.tool_calls:
                logger.info(f"{analyst_type} analyst completed report.")
                return _REPORT_FIELDS[analyst_type], result.content
            for tool_call in result.tool_calls:
                tool_output = await loop.run_in_executor(
                    None,
                    tools_by_name[tool_call["name"]].invoke,
                    tool_call["args"],
                )
                messages.append(
                    ToolMessage(
                        content=str(tool_output),
                        tool_call_id=tool_call["id"],
                    )
                )
        logger.warning(f"{analyst_type} analyst hit max iterations ({_MAX_TOOL_ITERATIONS}), returning empty report.")
        return _REPORT_FIELDS[analyst_type], ""
    except Exception as exc:
        logger.warning(f"{analyst_type} analyst failed: {exc}")
        return _REPORT_FIELDS[analyst_type], ""


# ── public factory ────────────────────────────────────────────────────────────

def create_analyst_team(llm, selected_analysts):
    """Return an async LangGraph node that runs all selected analysts in parallel."""

    async def analyst_team_node(state):
        current_date = state["trade_date"]
        ticker = state["company_of_interest"]
        initial_messages = state["messages"]

        coroutines = []
        for analyst_type in selected_analysts:
            chain, tools_by_name = _build_chain(analyst_type, llm, current_date, ticker)
            coroutines.append(
                _run_analyst(analyst_type, chain, tools_by_name, initial_messages)
            )

        results = await asyncio.gather(*coroutines)

        # Each analyst ran its tool-call loop against a local copy of messages.
        # We do not write back to the shared messages field — the original sequential
        # design cleared messages between analysts (Msg Clear nodes) so Bull Researcher
        # never saw analyst messages anyway. Report fields carry all analyst output.
        state_update = {}
        for report_field, report in results:
            state_update[report_field] = report
        return state_update

    return analyst_team_node
