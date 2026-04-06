# Import functions from specialized modules
from .alpha_vantage_stock import get_stock
from .alpha_vantage_indicator import get_indicator
from .alpha_vantage_fundamentals import get_fundamentals, get_balance_sheet, get_cashflow, get_income_statement
from .alpha_vantage_news import get_news, get_global_news, get_insider_transactions


def get_all_indicators(
    symbol: str,
    indicators: list,
    curr_date: str,
    look_back_days: int = 30,
) -> str:
    """Batch wrapper: call get_indicator once per item and concatenate results."""
    parts = []
    for ind in indicators:
        try:
            parts.append(get_indicator(symbol, ind, curr_date, look_back_days))
        except Exception as exc:
            parts.append(f"## {ind}\nError: {exc}")
    return "\n\n".join(parts) if parts else "No indicator data returned."