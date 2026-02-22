import argparse
import os
from typing import Optional

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest


class AlpacaTrader:
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        paper: Optional[bool] = None,
        base_url: Optional[str] = None,
    ) -> None:
        resolved_key = api_key or os.getenv("ALPACA_API_KEY")
        resolved_secret = api_secret or os.getenv("ALPACA_API_SECRET")
        resolved_paper = paper if paper is not None else os.getenv("ALPACA_PAPER", "true").lower() == "true"
        resolved_base_url = base_url or os.getenv("ALPACA_BASE_URL")

        if not resolved_key or not resolved_secret:
            raise ValueError("Missing Alpaca API credentials. Set ALPACA_API_KEY and ALPACA_API_SECRET.")
        self.client = TradingClient(
            resolved_key,
            resolved_secret,
            paper=resolved_paper,
            url_override=resolved_base_url,
        )
        print(f"Initialized AlpacaTrader with paper={resolved_paper}, key={resolved_key}, secret={resolved_secret}, base_url={resolved_base_url}")

    def buy(self, ticker: str, shares: int):
        if not ticker or not ticker.strip():
            raise ValueError("Ticker symbol is required.")
        if shares <= 0:
            raise ValueError("Shares must be greater than zero.")

        order = MarketOrderRequest(
            symbol=ticker.strip().upper(),
            qty=shares,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY,
        )
        return self.client.submit_order(order_data=order)

    def sell(self, ticker: str, shares: int):
        if not ticker or not ticker.strip():
            raise ValueError("Ticker symbol is required.")
        if shares <= 0:
            raise ValueError("Shares must be greater than zero.")

        order = MarketOrderRequest(
            symbol=ticker.strip().upper(),
            qty=shares,
            side=OrderSide.SELL,
            time_in_force=TimeInForce.DAY,
        )
        return self.client.submit_order(order_data=order)

    def get_account(self):
        return self.client.get_account()
    def get_position(self, ticker: str):
        if not ticker or not ticker.strip():
            raise ValueError("Ticker symbol is required.")
        return self.client.get_position(ticker.strip().upper())
    def get_positions(self):
        return self.client.get_all_positions()
    
    def get_orders(self, status: Optional[str] = None):
        return self.client.get_orders()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Submit Alpaca buy or sell orders.",
    )
    parser.add_argument(
        "--action",
        choices=["buy", "sell", "account"],
        required=True,
        help="Order action to execute (buy, sell, or account).",
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
        "--api-key",
        default=None,
        help="Alpaca API key. Defaults to ALPACA_API_KEY.",
    )
    parser.add_argument(
        "--api-secret",
        default=None,
        help="Alpaca API secret. Defaults to ALPACA_API_SECRET.",
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

    trader = AlpacaTrader(
        api_key=args.api_key,
        api_secret=args.api_secret,
        paper=paper_value,
        base_url=args.base_url,
    )

    if args.action == "account":
        account = trader.get_account()
        print(f"Account ID: {account.account_number}")
        print(f"Buying Power: ${account.buying_power}")
        print(f"Cash: ${account.cash}")
        print(f"Portfolio Value: ${account.portfolio_value}")
        print(f"Equity: ${account.equity}")
        print(f"Status: {account.status}")
    elif args.action == "buy":
        order = trader.buy(args.ticker, args.shares)
        order_id = getattr(order, "id", None) or getattr(order, "order_id", None)
        if order_id:
            print(f"Order submitted: BUY {args.shares} {args.ticker.upper()} (ID: {order_id})")
        else:
            print(f"Order submitted: BUY {args.shares} {args.ticker.upper()}")
    else:  # sell
        order = trader.sell(args.ticker, args.shares)
        order_id = getattr(order, "id", None) or getattr(order, "order_id", None)
        if order_id:
            print(f"Order submitted: SELL {args.shares} {args.ticker.upper()} (ID: {order_id})")
        else:
            print(f"Order submitted: SELL {args.shares} {args.ticker.upper()}")


