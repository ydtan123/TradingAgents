"""
Stub out heavy third-party and internal dependencies so that
tradingagents.agents.analysts.analyst_team can be imported in a
lightweight test environment (no pandas, yfinance, tqdm, etc.).
"""
import sys
import types
from unittest.mock import MagicMock


def _stub(name):
    """Create and register a stub module."""
    mod = types.ModuleType(name)
    sys.modules[name] = mod
    return mod


# ── heavy third-party packages ────────────────────────────────────────────────
for _pkg in [
    "pandas",
    "tqdm", "tqdm.auto",
    "yfinance",
    "requests",
    "finnhub",
    "praw",
    "stockstats",
    "redis",
    "chromadb",
    "langchain_openai",
    "langchain_anthropic",
    "langchain_google_genai",
    "langchain_experimental",
    "langsmith",
]:
    if _pkg not in sys.modules:
        _stub(_pkg)

# ── tradingagents.dataflows ───────────────────────────────────────────────────
# Prevent the real dataflows from loading (they need pandas, yfinance, etc.)
for _m in [
    "tradingagents.dataflows",
    "tradingagents.dataflows.local",
    "tradingagents.dataflows.interface",
    "tradingagents.dataflows.yfin_utils",
]:
    if _m not in sys.modules:
        sys.modules[_m] = MagicMock()

# ── tradingagents.agents.utils sub-modules ────────────────────────────────────
# Stub the tool modules that agent_utils imports from.
_tool_names = [
    "get_stock_data",
    "get_indicators",
    "get_all_indicators",
    "get_fundamentals",
    "get_balance_sheet",
    "get_cashflow",
    "get_income_statement",
    "get_news",
    "get_insider_sentiment",
    "get_insider_transactions",
    "get_global_news",
    "route_to_vendor",
]

for _submod in [
    "tradingagents.agents.utils.core_stock_tools",
    "tradingagents.agents.utils.technical_indicators_tools",
    "tradingagents.agents.utils.fundamental_data_tools",
    "tradingagents.agents.utils.news_data_tools",
]:
    mod = MagicMock()
    for _fn in _tool_names:
        tool = MagicMock()
        tool.name = _fn
        setattr(mod, _fn, tool)
    sys.modules[_submod] = mod

# Stub agent_utils itself so the __init__.py import of create_msg_delete works
_agent_utils_mod = MagicMock()
_agent_utils_mod.create_msg_delete = MagicMock(return_value=MagicMock())
for _fn in _tool_names:
    tool = MagicMock()
    tool.name = _fn
    setattr(_agent_utils_mod, _fn, tool)
sys.modules["tradingagents.agents.utils.agent_utils"] = _agent_utils_mod

# Stub the memory module
sys.modules["tradingagents.agents.utils.memory"] = MagicMock()

# Stub the agent_states module with real TypedDicts isn't needed — MagicMock is fine
sys.modules["tradingagents.agents.utils.agent_states"] = MagicMock()

# Stub all analyst modules imported by agents/__init__.py
for _analyst in [
    "tradingagents.agents.analysts.fundamentals_analyst",
    "tradingagents.agents.analysts.market_analyst",
    "tradingagents.agents.analysts.news_analyst",
    "tradingagents.agents.analysts.social_media_analyst",
    "tradingagents.agents.researchers.bear_researcher",
    "tradingagents.agents.researchers.bull_researcher",
    "tradingagents.agents.researchers",
    "tradingagents.agents.risk_mgmt.aggresive_debator",
    "tradingagents.agents.risk_mgmt.conservative_debator",
    "tradingagents.agents.risk_mgmt.neutral_debator",
    "tradingagents.agents.risk_mgmt",
    "tradingagents.agents.managers.research_manager",
    "tradingagents.agents.managers.risk_manager",
    "tradingagents.agents.managers",
    "tradingagents.agents.trader.trader",
    "tradingagents.agents.trader",
    "tradingagents.agents.utils",
]:
    if _analyst not in sys.modules:
        sys.modules[_analyst] = MagicMock()

# Stub tradingagents.agents itself so its __init__.py is bypassed
sys.modules["tradingagents.agents"] = MagicMock()

# Stub tradingagents.agents.analysts so the sub-package resolves correctly
# but leave analyst_team itself un-stubbed so tests import the real code.
import os as _os
_analysts_pkg = types.ModuleType("tradingagents.agents.analysts")
_analysts_real_path = _os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)),
    "..", "..", "..",
    "tradingagents", "agents", "analysts",
)
_analysts_pkg.__path__ = [_os.path.normpath(_analysts_real_path)]
_analysts_pkg.__package__ = "tradingagents.agents.analysts"
sys.modules["tradingagents.agents.analysts"] = _analysts_pkg
