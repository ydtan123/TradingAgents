import logging
import re


class KeywordColorFormatter(logging.Formatter):
    def __init__(self, fmt: str, keyword_colors: dict[str, str] | None = None):
        super().__init__(fmt)
        self.keyword_colors = keyword_colors or {
            "Market Analyst": "\033[31m",
            "News Analyst": "\033[32m",
            "Social Media Analyst": "\033[34m",
            "BUY": "\033[32m",
            "HOLD": "\033[33m",
            "SELL": "\033[31m",
        }
        self.reset = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        message = super().format(record)
        for keyword, color in self.keyword_colors.items():
            pattern = re.compile(rf"(?<!\w){re.escape(keyword)}(?!\w)")
            message = pattern.sub(f"{color}{keyword}{self.reset}", message)
        return message


def setup_tradingagents_logger(verbose: bool) -> logging.Logger:
    tradingagents_logger = logging.getLogger("tradingagents")

    if not tradingagents_logger.handlers:
        handler = logging.StreamHandler()
        formatter = KeywordColorFormatter("%(filename)s:%(lineno)d - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        tradingagents_logger.addHandler(handler)

    tradingagents_logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    return tradingagents_logger
    