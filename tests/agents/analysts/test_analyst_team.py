import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


# ── helpers ──────────────────────────────────────────────────────────────────

def _make_ai_message(content="", tool_calls=None):
    msg = MagicMock()
    msg.content = content
    msg.tool_calls = tool_calls or []
    return msg


def _make_tool_call(name, args, call_id="id1"):
    return {"name": name, "args": args, "id": call_id}


# ── _run_analyst tests ────────────────────────────────────────────────────────

class TestRunAnalyst:
    @pytest.mark.asyncio
    async def test_no_tool_calls_returns_report(self):
        """Analyst returns final report immediately without any tool calls."""
        from tradingagents.agents.analysts.analyst_team import _run_analyst

        final_msg = _make_ai_message(content="Market is bullish.")
        mock_chain = MagicMock()
        mock_chain.ainvoke = AsyncMock(return_value=final_msg)

        field, report = await _run_analyst("market", mock_chain, {}, [])

        assert field == "market_report"
        assert report == "Market is bullish."

    @pytest.mark.asyncio
    async def test_one_tool_call_then_final_report(self):
        """Analyst executes one tool call, then produces final report."""
        from tradingagents.agents.analysts.analyst_team import _run_analyst

        tool_call_msg = _make_ai_message(
            tool_calls=[_make_tool_call("get_stock_data", {"ticker": "AAPL"}, "c1")]
        )
        final_msg = _make_ai_message(content="Final analysis.")

        mock_chain = MagicMock()
        mock_chain.ainvoke = AsyncMock(side_effect=[tool_call_msg, final_msg])

        mock_tool = MagicMock()
        mock_tool.invoke = MagicMock(return_value="stock,price\nAAPL,150")

        field, report = await _run_analyst(
            "market", mock_chain, {"get_stock_data": mock_tool}, []
        )

        assert field == "market_report"
        assert report == "Final analysis."
        mock_tool.invoke.assert_called_once_with({"ticker": "AAPL"})

    @pytest.mark.asyncio
    async def test_exception_returns_empty_string(self):
        """Failed analyst returns empty report; does not raise."""
        from tradingagents.agents.analysts.analyst_team import _run_analyst

        mock_chain = MagicMock()
        mock_chain.ainvoke = AsyncMock(side_effect=Exception("LLM timeout"))

        field, report = await _run_analyst("market", mock_chain, {}, [])

        assert field == "market_report"
        assert report == ""

    @pytest.mark.asyncio
    async def test_report_field_mapping(self):
        """Each analyst type maps to its correct state field."""
        from tradingagents.agents.analysts.analyst_team import _run_analyst

        expected_fields = {
            "market": "market_report",
            "social": "sentiment_report",
            "news": "news_report",
            "fundamentals": "fundamentals_report",
        }

        for analyst_type, expected_field in expected_fields.items():
            final_msg = _make_ai_message(content="report")
            mock_chain = MagicMock()
            mock_chain.ainvoke = AsyncMock(return_value=final_msg)

            field, _ = await _run_analyst(analyst_type, mock_chain, {}, [])
            assert field == expected_field, f"{analyst_type} mapped to wrong field"

    @pytest.mark.asyncio
    async def test_initial_messages_are_not_mutated(self):
        """The caller's original message list is not modified."""
        from tradingagents.agents.analysts.analyst_team import _run_analyst

        original = [MagicMock()]
        final_msg = _make_ai_message(content="done")
        mock_chain = MagicMock()
        mock_chain.ainvoke = AsyncMock(return_value=final_msg)

        await _run_analyst("market", mock_chain, {}, original)
        assert len(original) == 1  # unchanged


# ── create_analyst_team node tests ───────────────────────────────────────────

class TestCreateAnalystTeam:
    def _make_state(self, trade_date="2024-01-01", ticker="AAPL", messages=None):
        return {
            "trade_date": trade_date,
            "company_of_interest": ticker,
            "messages": messages or [],
        }

    @pytest.mark.asyncio
    async def test_all_four_analysts_populate_reports(self):
        """All four selected analysts run and write to their report fields."""
        from tradingagents.agents.analysts.analyst_team import create_analyst_team

        def _mock_chain_for(report_text):
            chain = MagicMock()
            chain.ainvoke = AsyncMock(return_value=_make_ai_message(content=report_text))
            return chain

        with patch(
            "tradingagents.agents.analysts.analyst_team._build_chain"
        ) as mock_build:
            mock_build.side_effect = lambda analyst_type, llm, date, ticker: (
                _mock_chain_for(f"{analyst_type} report"),
                {},
            )

            node = create_analyst_team(MagicMock(), ["market", "social", "news", "fundamentals"])
            result = await node(self._make_state())

        assert result["market_report"] == "market report"
        assert result["sentiment_report"] == "social report"
        assert result["news_report"] == "news report"
        assert result["fundamentals_report"] == "fundamentals report"

    @pytest.mark.asyncio
    async def test_failed_analyst_does_not_block_others(self):
        """One failing analyst produces empty report; others succeed."""
        from tradingagents.agents.analysts.analyst_team import create_analyst_team

        call_count = 0

        def _mock_build(analyst_type, llm, date, ticker):
            nonlocal call_count
            call_count += 1
            chain = MagicMock()
            if analyst_type == "social":
                chain.ainvoke = AsyncMock(side_effect=Exception("API down"))
            else:
                chain.ainvoke = AsyncMock(
                    return_value=_make_ai_message(content=f"{analyst_type} ok")
                )
            return chain, {}

        with patch(
            "tradingagents.agents.analysts.analyst_team._build_chain",
            side_effect=_mock_build,
        ):
            node = create_analyst_team(MagicMock(), ["market", "social", "news", "fundamentals"])
            result = await node(self._make_state())

        assert result["market_report"] == "market ok"
        assert result["sentiment_report"] == ""        # failed
        assert result["news_report"] == "news ok"
        assert result["fundamentals_report"] == "fundamentals ok"

    @pytest.mark.asyncio
    async def test_subset_of_analysts(self):
        """Only selected analysts write to state; unselected fields absent."""
        from tradingagents.agents.analysts.analyst_team import create_analyst_team

        with patch(
            "tradingagents.agents.analysts.analyst_team._build_chain"
        ) as mock_build:
            mock_build.return_value = (
                MagicMock(ainvoke=AsyncMock(return_value=_make_ai_message(content="r"))),
                {},
            )

            node = create_analyst_team(MagicMock(), ["market"])
            result = await node(self._make_state())

        assert "market_report" in result
        assert "sentiment_report" not in result
        assert "news_report" not in result
        assert "fundamentals_report" not in result
