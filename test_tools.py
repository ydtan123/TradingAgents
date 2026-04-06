"""
Standalone tool smoke-test.

Calls every data tool directly (no LLM) and reports whether each returned
meaningful content.  Run with:

    python test_tools.py
    python test_tools.py --verbose   # also print vendor name + first 400 chars
"""

import sys
import textwrap

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Initialise the vendor config so route_to_vendor uses DEFAULT_CONFIG
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.dataflows.config import set_config, get_config

set_config(DEFAULT_CONFIG)

from tradingagents.dataflows.interface import (
    route_to_vendor,
    get_vendor,
    get_category_for_method,
)

TICKER = "NVDA"
DATE = "2026-04-02"          # analysis date
START_DATE = "2026-03-26"    # one week before analysis date
END_DATE = DATE

VERBOSE = "--verbose" in sys.argv or "-v" in sys.argv

# ── colour helpers ────────────────────────────────────────────────────────────

GREEN  = "\033[32m"
RED    = "\033[31m"
YELLOW = "\033[33m"
CYAN   = "\033[36m"
RESET  = "\033[0m"
BOLD   = "\033[1m"
DIM    = "\033[2m"

def ok(msg):   print(f"  {GREEN}✓{RESET} {msg}")
def fail(msg): print(f"  {RED}✗{RESET} {msg}")


def _resolved_vendor(tool_name: str) -> str:
    """Return the vendor that will actually handle this tool call."""
    try:
        cat = get_category_for_method(tool_name)
        return get_vendor(cat, tool_name)
    except Exception:
        return "unknown"


def _with_vendor(tool_name: str, vendor: str, fn):
    """Run fn() with a per-tool vendor override, then restore the original."""
    cfg = get_config()
    original = cfg.get("tool_vendors", {}).get(tool_name)
    cfg.setdefault("tool_vendors", {})[tool_name] = vendor
    set_config(cfg)
    try:
        return fn()
    finally:
        if original is None:
            cfg.get("tool_vendors", {}).pop(tool_name, None)
        else:
            cfg["tool_vendors"][tool_name] = original
        set_config(cfg)


def check(label, tool_name, result, min_chars=50):
    """Validate a tool result and print status + optional verbose preview."""
    if not isinstance(result, str):
        result = str(result)

    vendor = _resolved_vendor(tool_name)

    no_data_phrases = [
        "no data found", "no news found", "no global news found",
        "error fetching", "no fundamentals", "error:",
    ]
    is_empty    = len(result.strip()) < min_chars
    looks_empty = any(p in result.lower() for p in no_data_phrases)

    vendor_tag = f"{CYAN}[{vendor}]{RESET}"

    if is_empty or looks_empty:
        fail(f"{vendor_tag} {label}: thin/empty result ({len(result)} chars)")
        if result.strip():
            print(f"    → {result.strip()[:200]}")
        return False

    ok(f"{vendor_tag} {label}: {len(result):,} chars")
    if VERBOSE:
        preview = textwrap.indent(result[:400].replace("\n", " "), "    ")
        print(f"{DIM}{preview}{'…' if len(result) > 400 else ''}{RESET}")
    return True


# ── test runner ───────────────────────────────────────────────────────────────

results = {}

def run(label, tool_name, fn, forced_vendor=None):
    """
    forced_vendor: if set, temporarily override the tool's vendor for this call
                   so the vendor tag shown in output reflects what we're testing.
    """
    print(f"\n{BOLD}{label}{RESET}")
    try:
        if forced_vendor:
            result = _with_vendor(tool_name, forced_vendor, fn)
            # Show the forced vendor explicitly in the output tag
            vendor_tag = f"{CYAN}[{forced_vendor}]{RESET}"
            if not isinstance(result, str):
                result = str(result)
            no_data_phrases = [
                "no data found", "no news found", "no global news found",
                "error fetching", "no fundamentals", "error:",
            ]
            is_empty    = len(result.strip()) < 50
            looks_empty = any(p in result.lower() for p in no_data_phrases)
            if is_empty or looks_empty:
                fail(f"{vendor_tag} {label}: thin/empty result ({len(result)} chars)")
                if result.strip():
                    print(f"    → {result.strip()[:200]}")
                passed = False
            else:
                ok(f"{vendor_tag} {label}: {len(result):,} chars")
                if VERBOSE:
                    preview = textwrap.indent(result[:400].replace("\n", " "), "    ")
                    print(f"{DIM}{preview}{'…' if len(result) > 400 else ''}{RESET}")
                passed = True
        else:
            result = fn()
            passed = check(label, tool_name, result)
    except Exception as exc:
        vendor = forced_vendor or _resolved_vendor(tool_name)
        fail(f"{CYAN}[{vendor}]{RESET} {label}: raised {type(exc).__name__}: {exc}")
        passed = False
    results[label] = passed


# ── default-vendor tests (uses DEFAULT_CONFIG as-is) ─────────────────────────

print(f"\n{BOLD}{'═'*52}{RESET}")
print(f"{BOLD}Default vendor tests{RESET}")
print(f"{BOLD}{'═'*52}{RESET}")

run("get_stock_data",
    "get_stock_data",
    lambda: route_to_vendor("get_stock_data", TICKER, START_DATE, END_DATE))

run("get_all_indicators",
    "get_all_indicators",
    lambda: route_to_vendor("get_all_indicators", TICKER,
                            ["rsi", "macd", "close_50_sma", "close_200_sma"], DATE, 30))

run("get_fundamentals",
    "get_fundamentals",
    lambda: route_to_vendor("get_fundamentals", TICKER))

run("get_balance_sheet",
    "get_balance_sheet",
    lambda: route_to_vendor("get_balance_sheet", TICKER))

run("get_cashflow",
    "get_cashflow",
    lambda: route_to_vendor("get_cashflow", TICKER))

run("get_income_statement",
    "get_income_statement",
    lambda: route_to_vendor("get_income_statement", TICKER))

run("get_news",
    "get_news",
    lambda: route_to_vendor("get_news", TICKER, START_DATE, END_DATE))

run("get_global_news",
    "get_global_news",
    lambda: route_to_vendor("get_global_news", DATE, 7, 10))

# ── Google vendor tests ───────────────────────────────────────────────────────

print(f"\n{BOLD}{'═'*52}{RESET}")
print(f"{BOLD}Google vendor tests{RESET}")
print(f"{BOLD}{'═'*52}{RESET}")

run("get_news [google]",
    "get_news",
    lambda: route_to_vendor("get_news", TICKER, START_DATE, END_DATE),
    forced_vendor="google")

run("get_global_news [google]",
    "get_global_news",
    lambda: route_to_vendor("get_global_news", DATE, 7, 10),
    forced_vendor="google")

run("get_fundamentals [google]",
    "get_fundamentals",
    lambda: route_to_vendor("get_fundamentals", TICKER, DATE),
    forced_vendor="google")

# ── summary ───────────────────────────────────────────────────────────────────

passed = sum(1 for v in results.values() if v)
total  = len(results)

print(f"\n{'─'*52}")
print(f"{BOLD}Results: {passed}/{total} tools returning data{RESET}")
for label, ok_flag in results.items():
    # Derive canonical tool name for vendor lookup
    base = label.split(" [")[0]  # strip " [google]" suffix if present
    tool_key = {"get_news": "get_news", "get_global_news": "get_global_news"}.get(base, base)
    forced = "google" if "[google]" in label else None
    vendor = forced or _resolved_vendor(tool_key)
    status = f"{GREEN}PASS{RESET}" if ok_flag else f"{RED}FAIL{RESET}"
    print(f"  [{status}] {CYAN}[{vendor}]{RESET} {label}")

if passed < total:
    sys.exit(1)
