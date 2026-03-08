import logging

from langchain_core.tools import tool
from typing import Annotated, List
from tradingagents.dataflows.interface import route_to_vendor

logger = logging.getLogger(__name__)

@tool
def get_all_indicators(
    symbol: Annotated[str, "ticker symbol of the company"],
    indicators: Annotated[List[str], "list of technical indicator names to retrieve, e.g. ['close_50_sma', 'rsi', 'macd']"],
    curr_date: Annotated[str, "The current trading date you are trading on, YYYY-mm-dd"],
    look_back_days: Annotated[int, "how many days to look back"] = 30,
) -> str:
    """
    Retrieve multiple technical indicators for a given ticker in a single API call.
    Much more efficient than calling get_indicators repeatedly for each indicator.
    Available indicators: close_50_sma, close_200_sma, close_10_ema, macd, macds, macdh,
    rsi, boll, boll_ub, boll_lb, atr, vwma, mfi.
    Args:
        symbol (str): Ticker symbol of the company, e.g. AAPL, AMZN
        indicators (List[str]): List of indicator names to retrieve
        curr_date (str): The current trading date, YYYY-mm-dd
        look_back_days (int): How many days to look back, default is 30
    Returns:
        str: Formatted report for all requested indicators.
    """
    logger.info(f"Getting all technical indicators for {symbol}, indicators: {indicators}, current date: {curr_date}, look back days: {look_back_days}")
    return route_to_vendor("get_all_indicators", symbol, indicators, curr_date, look_back_days)


@tool
def get_indicators(
    symbol: Annotated[str, "ticker symbol of the company"],
    indicator: Annotated[str, "technical indicator to get the analysis and report of"],
    curr_date: Annotated[str, "The current trading date you are trading on, YYYY-mm-dd"],
    look_back_days: Annotated[int, "how many days to look back"] = 30,
) -> str:
    """
    Retrieve technical indicators for a given ticker symbol.
    Uses the configured technical_indicators vendor.
    Args:
        symbol (str): Ticker symbol of the company, e.g. AAPL, TSM
        indicator (str): Technical indicator to get the analysis and report of
        curr_date (str): The current trading date you are trading on, YYYY-mm-dd
        look_back_days (int): How many days to look back, default is 30
    Returns:
        str: A formatted dataframe containing the technical indicators for the specified ticker symbol and indicator.
    """
    logger.info(f"Getting technical indicators for {symbol}, indicator: {indicator}, current date: {curr_date}, look back days: {look_back_days}")
    #import ipdb; ipdb.set_trace()
    #print(f"Getting technical indicators for {symbol}, indicator: {indicator}, current date: {curr_date}, look back days: {look_back_days}")
    return route_to_vendor("get_indicators", symbol, indicator, curr_date, look_back_days)