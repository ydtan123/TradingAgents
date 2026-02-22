#!/usr/bin/env python3
import argparse
from typing import Optional

from dotenv import load_dotenv

from tradingagents.agents.trader.alpaca_trader import AlpacaTrader


def main() -> int:
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Execute Alpaca actions (buy, sell, or view account).",
    )
    parser.add_argument(
        "--action",
        choices=["buy", "sell", "account", "details"],
        required=True,
        help="Action to execute (buy, sell, account, or details).",
    )
    parser.add_argument(
        "--ticker",
        default=None,
        help="Stock ticker symbol (e.g., AAPL). Required for buy/sell.",
    )
    parser.add_argument(
        "--shares",
        type=int,
        default=None,
        help="Number of shares to trade. Required for buy/sell.",
    )
    parser.add_argument(
        "--paper",
        default=None,
        help="Use paper trading (true/false). Defaults to ALPACA_PAPER.",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="Override Alpaca base URL. Defaults to ALPACA_BASE_URL.",
    )

    args = parser.parse_args()

    if args.action in ["buy", "sell"]:
        if not args.ticker or not args.shares:
            parser.error("--ticker and --shares are required for buy/sell actions")

    paper_value: Optional[bool] = None
    if args.paper is not None:
        paper_value = str(args.paper).lower() == "true"

    trader = AlpacaTrader(paper=paper_value, base_url=args.base_url)

    if args.action == "account":
        account = trader.get_account()
        print(f"Account ID: {account.account_number}")
        print(f"Buying Power: ${account.buying_power}")
        print(f"Cash: ${account.cash}")
        print(f"Portfolio Value: ${account.portfolio_value}")
        print(f"Equity: ${account.equity}")
        print(f"Status: {account.status}")
    elif args.action == "details":
        print("\n=== POSITIONS ===")
        positions = trader.get_positions()
        if positions:
            for position in positions:
                print(f"Symbol: {position.symbol}")
                print(f"  Qty: {position.qty}")
                print(f"  Avg Fill Price: ${position.avg_entry_price}")
                print(f"  Market Value: ${position.market_value}")
                print(f"  Unrealized PnL: ${position.unrealized_pl}")
                print(f"  Unrealized PnL %: {position.unrealized_plpc}%")
                print()
        else:
            print("No positions found.")
        
        print("\n=== ORDERS ===")
        orders = trader.get_orders()
        if orders:
            for order in orders:
                print(f"Order ID: {order.id}")
                print(f"  Symbol: {order.symbol}")
                print(f"  Side: {order.side}")
                print(f"  Qty: {order.qty}")
                print(f"  Status: {order.status}")
                print(f"  Type: {order.order_type}")
                print()
        else:
            print("No orders found.")
    elif args.action == "buy":
        order = trader.buy(args.ticker, args.shares)
        order_id = getattr(order, "id", None) or getattr(order, "order_id", None)
        if order_id:
            print(
                f"Order submitted: BUY {args.shares} {args.ticker.upper()} (ID: {order_id})"
            )
        else:
            print(
                f"Order submitted: BUY {args.shares} {args.ticker.upper()}"
            )
    else:  # sell
        order = trader.sell(args.ticker, args.shares)
        order_id = getattr(order, "id", None) or getattr(order, "order_id", None)
        if order_id:
            print(
                f"Order submitted: SELL {args.shares} {args.ticker.upper()} (ID: {order_id})"
            )
        else:
            print(
                f"Order submitted: SELL {args.shares} {args.ticker.upper()}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
