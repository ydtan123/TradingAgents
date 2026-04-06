from langchain_core.tools import tool
from typing import Annotated, List
from tradingagents.dataflows.interface import route_to_vendor

@tool
def get_all_indicators(
    symbol: Annotated[str, "ticker symbol of the company"],
    indicators: Annotated[List[str], "list of technical indicator names to retrieve in one batch call"],
    curr_date: Annotated[str, "The current trading date you are trading on, YYYY-mm-dd"],
    look_back_days: Annotated[int, "how many days to look back"] = 30,
) -> str:
    """
    Retrieve multiple technical indicators for a ticker in a single batched call.
    Preferred over calling get_indicators repeatedly — pass all chosen indicator
    names as a list and receive all results concatenated.
    Args:
        symbol (str): Ticker symbol, e.g. AAPL
        indicators (list[str]): List of indicator names, e.g. ['rsi', 'macd', 'close_50_sma']
        curr_date (str): Current trading date YYYY-mm-dd
        look_back_days (int): How many days to look back, default 30
    Returns:
        str: Concatenated indicator data for all requested indicators.
    """
    indicators_normalized = [i.strip().lower() for i in indicators if i.strip()]
    return route_to_vendor("get_all_indicators", symbol, indicators_normalized, curr_date, look_back_days)


@tool
def get_indicators(
    symbol: Annotated[str, "ticker symbol of the company"],
    indicator: Annotated[str, "technical indicator to get the analysis and report of"],
    curr_date: Annotated[str, "The current trading date you are trading on, YYYY-mm-dd"],
    look_back_days: Annotated[int, "how many days to look back"] = 30,
) -> str:
    """
    Retrieve a single technical indicator for a given ticker symbol.
    Uses the configured technical_indicators vendor.
    Args:
        symbol (str): Ticker symbol of the company, e.g. AAPL, TSM
        indicator (str): A single technical indicator name, e.g. 'rsi', 'macd'. Call this tool once per indicator.
        curr_date (str): The current trading date you are trading on, YYYY-mm-dd
        look_back_days (int): How many days to look back, default is 30
    Returns:
        str: A formatted dataframe containing the technical indicators for the specified ticker symbol and indicator.
    """
    # LLMs sometimes pass multiple indicators as a comma-separated string;
    # split and process each individually.
    indicators = [i.strip().lower() for i in indicator.split(",") if i.strip()]
    results = []
    for ind in indicators:
        try:
            results.append(route_to_vendor("get_indicators", symbol, ind, curr_date, look_back_days))
        except ValueError as e:
            results.append(str(e))
    return "\n\n".join(results)