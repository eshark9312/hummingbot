import asyncio
import threading
import time
from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING, List, Optional, Set, Tuple, Dict, Any

import pandas as pd

from hummingbot.client.command.gateway_command import GatewayCommand
from hummingbot.client.performance import PerformanceMetrics
from hummingbot.client.settings import MAXIMUM_TRADE_FILLS_DISPLAY_OUTPUT, AllConnectorSettings
from hummingbot.client.ui.interface_utils import format_df_for_printout
from hummingbot.core.utils.async_utils import safe_ensure_future
from hummingbot.model.trade_fill import TradeFill
from hummingbot.user.user_balances import UserBalances

s_float_0 = float(0)
s_decimal_0 = Decimal("0")


if TYPE_CHECKING:
    from hummingbot.client.hummingbot_application import HummingbotApplication  # noqa: F401


def get_timestamp(days_ago: float = 0.) -> float:
    return time.time() - (60. * 60. * 24. * days_ago)


class HistarbCommand:
    def histarb(self,  # type: HummingbotApplication
                days: float = 0,
                verbose: bool = False,
                precision: Optional[int] = None
                ):
        if threading.current_thread() != threading.main_thread():
            self.ev_loop.call_soon_threadsafe(self.histarb, days, verbose, precision)
            return

        if self.strategy_file_name is None:
            self.notify("\n  Please first import a strategy config file of which to show historical performance.")
            return
        start_time = get_timestamp(days) if days > 0 else self.init_time
        with self.trade_fill_db.get_new_session() as session:
            trades: List[TradeFill] = self._get_trades_from_session(
                int(start_time * 1e3),
                session=session,
                config_file_path=self.strategy_file_name)
            if not trades:
                self.notify("\n  No past trades to report.")
                return
            if verbose:
                self.list_trades(start_time)
            return safe_ensure_future(self.histarb_report(start_time, trades, precision))

    async def histarb_report(self,  # type: HummingbotApplication
                             start_time: float,
                             trades: List[TradeFill],
                             precision: Optional[int] = None,
                             display_report: bool = True) -> Decimal:
        market_info: Set[Tuple[str, str]] = set((t.market, t.symbol) for t in trades)
        if display_report:
            self.report_header(start_time)
        perf_metrics = {}
        for market, symbol in market_info:
            if symbol not in perf_metrics:
                perf_metrics[symbol] = {}
            cur_trades = [t for t in trades if t.market == market and t.symbol == symbol]
            network_timeout = float(self.client_config_map.commands_timeout.other_commands_timeout)
            try:
                cur_balances = await asyncio.wait_for(self.get_current_balances(market), network_timeout)
            except asyncio.TimeoutError:
                self.notify(
                    "\nA network error prevented the balances retrieval to complete. See logs for more details."
                )
                raise
            perf = await PerformanceMetrics.create(symbol, cur_trades, cur_balances)
            perf_metrics[symbol][market] = self.get_trades_perf_summary(perf = perf, precision = precision)

        # summarize the trading performance
        perf_summary_str = self.report_perf_summary(perf_metrics)
        if display_report:
            self.notify(perf_summary_str)
