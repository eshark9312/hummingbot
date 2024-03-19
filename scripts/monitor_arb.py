import logging
from decimal import Decimal
from typing import Any, Dict
import time
from datetime import datetime
import sqlite3

import pandas as pd

from hummingbot.core.data_type.common import OrderType, TradeType
from hummingbot.core.data_type.order_candidate import OrderCandidate
from hummingbot.core.event.events import OrderFilledEvent
from hummingbot.strategy.script_strategy_base import ScriptStrategyBase

def no_format(x):
    return x

def format_3digits(x):
    return f"{x:,.3f}"

def format_4digits(x):
    return f"{x:,.4f}"    

class SimpleArbitrage(ScriptStrategyBase):
    """
    BotCamp Cohort: Sept 2022
    Design Template: https://hummingbot-foundation.notion.site/Simple-Arbitrage-51b2af6e54b6493dab12e5d537798c07
    Video: TBD
    Description:
    A simplified version of Hummingbot arbitrage strategy, this bot checks the Volume Weighted Average Price for
    bid and ask in two exchanges and if it finds a profitable opportunity, it will trade the tokens.
    """
    order_amount = Decimal("1000")  # in base asset
    min_profitability = Decimal("0.8")  # in percentage
    exchange_A = "mexc"
    exchange_B = "kucoin"
    arb_threshold = 0.3                 # threshold for take profit in percentage
    duration_threshold = 0.3            # threshold for duration of arb_opportunity to capture

    quote_amount = 100
    base_assets = ["EGO", "MYRO"]
    markets_set = {f"{base}-USDT" for base in base_assets}
    markets = {exchange_A: markets_set,
               exchange_B: markets_set}

    opportunity_ts = {base: {"buy_a_sell_b": 0, "profit_a_b" : 0,
                             "buy_b_sell_a": 0, "profit_b_a" : 0,
                             "buy_price": 0, "sell_price": 0} for base in base_assets}
    
    sqlite_conn = sqlite3.connect('opportunity_log.db')
    tb_name = "bbl_sui_rite_insp_clore_log"
    # initialize the db
    cursor = sqlite_conn.cursor()
    cursor.execute(f'''CREATE TABLE IF NOT EXISTS {tb_name} (
                        datetime DATETIME,
                        coin TEXT,
                        buy TEXT,
                        sell TEXT,
                        buy_price REAL,
                        sell_price REAL,
                        profit REAL,
                        duration REAL
                    )''')
    sqlite_conn.commit()

    def on_tick(self):
        pass
        # vwap_prices = self.get_vwap_prices_for_amount(self.order_amount)
        # proposal = self.check_profitability_and_create_proposal(vwap_prices)
        # if len(proposal) > 0:
        #     proposal_adjusted: Dict[str, OrderCandidate] = self.adjust_proposal_to_budget(proposal)
        #     # self.place_orders(proposal_adjusted)

    def get_vwap_prices_for_amount(self, amount: Decimal, pair: str = ""):
        bid_ex_a = self.connectors[self.exchange_A].get_vwap_for_volume(pair, False, amount)
        ask_ex_a = self.connectors[self.exchange_A].get_vwap_for_volume(pair, True, amount)
        bid_ex_b = self.connectors[self.exchange_B].get_vwap_for_volume(pair, False, amount)
        ask_ex_b = self.connectors[self.exchange_B].get_vwap_for_volume(pair, True, amount)
        vwap_prices = {
            self.exchange_A: {
                "bid": bid_ex_a.result_price,
                "ask": ask_ex_a.result_price
            },
            self.exchange_B: {
                "bid": bid_ex_b.result_price,
                "ask": ask_ex_b.result_price
            }
        }
        return vwap_prices

    def get_fees_percentages(self, vwap_prices: Dict[str, Any], base: str = "", quote: str = "") -> Dict:
        # We assume that the fee percentage for buying or selling is the same
        base_currency=self.base if base == "" else base
        quote_currency="USDT" if quote == "" else quote
        order_amount = self.quote_amount / self.connectors[self.exchange_A].get_mid_price(trading_pair = f"{base_currency}-{quote_currency}")
        a_fee = self.connectors[self.exchange_A].get_fee(
            base_currency=base_currency,
            quote_currency=quote_currency,
            order_type=OrderType.MARKET,
            order_side=TradeType.BUY,
            amount=order_amount,
            price=vwap_prices[self.exchange_A]["ask"],
            is_maker=False
        ).percent

        b_fee = self.connectors[self.exchange_B].get_fee(
            base_currency=base_currency,
            quote_currency=quote_currency,
            order_type=OrderType.MARKET,
            order_side=TradeType.BUY,
            amount=order_amount,
            price=vwap_prices[self.exchange_B]["ask"],
            is_maker=False
        ).percent

        return {
            self.exchange_A: a_fee,
            self.exchange_B: b_fee
        }

    def get_profitability_analysis(self, vwap_prices: Dict[str, Any], base: str = "", quote: str = "") -> Dict:
        quote_currency="USDT" if quote == "" else quote
        fees = self.get_fees_percentages(vwap_prices = vwap_prices, base = base)
        base_order_amount = self.quote_amount / self.connectors[self.exchange_A].get_mid_price(trading_pair = f"{base}-{quote_currency}")
        buy_a_sell_b_quote = vwap_prices[self.exchange_B]["bid"] * (1 - fees[self.exchange_B]) * base_order_amount - \
            vwap_prices[self.exchange_A]["ask"] * (1 + fees[self.exchange_A]) * base_order_amount
        buy_a_sell_b_base = buy_a_sell_b_quote / (
            (vwap_prices[self.exchange_A]["ask"] + vwap_prices[self.exchange_B]["bid"]) / 2)

        buy_b_sell_a_quote = vwap_prices[self.exchange_A]["bid"] * (1 - fees[self.exchange_A]) * base_order_amount - \
            vwap_prices[self.exchange_B]["ask"] * (1 + fees[self.exchange_B]) * base_order_amount

        buy_b_sell_a_base = buy_b_sell_a_quote / (
            (vwap_prices[self.exchange_B]["ask"] + vwap_prices[self.exchange_A]["bid"]) / 2)

        return {
            "buy_a_sell_b":
                {
                    "quote_diff": buy_a_sell_b_quote,
                    "base_diff": buy_a_sell_b_base,
                    "profitability_pct": buy_a_sell_b_base / base_order_amount
                },
            "buy_b_sell_a":
                {
                    "quote_diff": buy_b_sell_a_quote,
                    "base_diff": buy_b_sell_a_base,
                    "profitability_pct": buy_b_sell_a_base / base_order_amount
                },
        }

    def _update_opportunity_ts(self, base: str, profit: Decimal, is_buy_a:bool, vwap_prices: Dict[str, Any]) -> str:
        if self.opportunity_ts[base]["buy_a_sell_b" if is_buy_a else "buy_b_sell_a"] == 0:
            if profit >= self.arb_threshold:             # start_ts of the arbitrage opportunity
                self.opportunity_ts[base]["buy_a_sell_b" if is_buy_a else "buy_b_sell_a"] = time.time()
                self.opportunity_ts[base]["profit_a_b" if is_buy_a else "profit_b_a"] = profit
                self.opportunity_ts[base]["buy_price"] = vwap_prices[self.exchange_A]["ask"] if is_buy_a else vwap_prices[self.exchange_B]["ask"]
                self.opportunity_ts[base]["sell_price"] = vwap_prices[self.exchange_B]["bid"] if is_buy_a else vwap_prices[self.exchange_A]["bid"]
            return ""
        else:
            duration = time.time() - self.opportunity_ts[base]["buy_a_sell_b" if is_buy_a else "buy_b_sell_a"]
            if profit > self.opportunity_ts[base]["profit_a_b" if is_buy_a else "profit_b_a"]:    # update max_profit and buy/sell_prices
                self.opportunity_ts[base]["profit_a_b" if is_buy_a else "profit_b_a"] = profit
                self.opportunity_ts[base]["buy_price"] = vwap_prices[self.exchange_A]["ask"] if is_buy_a else vwap_prices[self.exchange_B]["ask"]
                self.opportunity_ts[base]["sell_price"] = vwap_prices[self.exchange_B]["bid"] if is_buy_a else vwap_prices[self.exchange_A]["bid"]
            if profit < self.arb_threshold:             # end_ts of the arbitrage opportunity
                self.opportunity_ts[base]["buy_a_sell_b" if is_buy_a else "buy_b_sell_a"] = 0
                max_profit = self.opportunity_ts[base]["profit_a_b" if is_buy_a else "profit_b_a"]
                self.opportunity_ts[base]["profit_a_b" if is_buy_a else "profit_b_a"] = 0
                if duration > 0.3:                      # log opportunity into sqlite3db and hb_logs
                    data = {"datetime": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            "coin": base,
                            "buy": self.exchange_A if is_buy_a else self.exchange_B,
                            "sell": self.exchange_B if is_buy_a else self.exchange_A,
                            "buy_price": f"{self.opportunity_ts[base]['buy_price']:.4f}",
                            "sell_price": f"{self.opportunity_ts[base]['sell_price']:.4f}",
                            "profit": f"{max_profit:.2f}",
                            "duration": f"{duration:.1f}"}
                    self._log_orb_opportunity_sqlite(data)
                    info_msg = f"{base} : {self.exchange_A} -> {self.exchange_B} {max_profit:.2f} % ({duration:.1f})" if is_buy_a else \
                               f"{base} : {self.exchange_B} -> {self.exchange_A} {max_profit:.2f} % ({duration:.1f})"
                    self.logger().info(info_msg)
            return f"({duration:.1f})"

    def _log_orb_opportunity_sqlite(self, data: Dict[str, Any]):
        self.cursor.execute(f'''INSERT INTO {self.tb_name} (
                            datetime, coin, buy, sell, buy_price, sell_price, profit, duration
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)''', (
                            data['datetime'],
                            data['coin'],
                            data['buy'],
                            data['sell'],
                            data['buy_price'],
                            data['sell_price'],
                            data['profit'],
                            data['duration']))
        self.sqlite_conn.commit()

    def batch_arb_profit_analysis_df(self):
        columns = self.base_assets.copy()
        columns.insert(0,"BUY -> SELL")
        buy_a_sell_b = [f"{self.exchange_A} -> {self.exchange_B}"]
        buy_a_sell_b_price = [f"{self.exchange_A} -> {self.exchange_B}"]
        buy_b_sell_a = [f"{self.exchange_B} -> {self.exchange_A}"]
        buy_b_sell_a_price = [f"{self.exchange_B} -> {self.exchange_A}"]
        for base in self.base_assets:
            trading_pair = f"{base}-USDT"
            order_amount = self.quote_amount / self.connectors[self.exchange_A].get_mid_price(trading_pair = trading_pair)
            vwap_prices = self.get_vwap_prices_for_amount(amount = self.quote_amount, pair = trading_pair)
            profitability_analysis = self.get_profitability_analysis(vwap_prices = vwap_prices, base = base)
            buy_a_sell_b_profit = profitability_analysis['buy_a_sell_b']['profitability_pct'] * 100
            buy_a_sell_b_duration = self._update_opportunity_ts(base = base,
                                                                profit = buy_a_sell_b_profit,
                                                                vwap_prices = vwap_prices,
                                                                is_buy_a = True)
            buy_b_sell_a_profit = profitability_analysis['buy_b_sell_a']['profitability_pct'] * 100
            buy_b_sell_a_duration = self._update_opportunity_ts(base = base,
                                                                profit = buy_b_sell_a_profit,
                                                                vwap_prices = vwap_prices,
                                                                is_buy_a = False)
            buy_a_sell_b.append(f"{buy_a_sell_b_profit:.2f}{buy_a_sell_b_duration}")
            buy_b_sell_a.append(f"{buy_b_sell_a_profit:.2f}{buy_b_sell_a_duration}")
            buy_a_sell_b_price.append(f"{vwap_prices[self.exchange_A]['ask']:.4f}->{vwap_prices[self.exchange_B]['bid']:.4f}")
            buy_b_sell_a_price.append(f"{vwap_prices[self.exchange_B]['ask']:.4f}->{vwap_prices[self.exchange_A]['bid']:.4f}")
        df = pd.DataFrame(data = [buy_a_sell_b, buy_b_sell_a, buy_a_sell_b_price, buy_b_sell_a_price], columns = columns)
        return df

    def format_status(self) -> str:
        """
        Returns status of the current strategy on user balances and current active orders. This function is called
        when status command is issued. Override this function to create custom status display output.
        """
        if not self.ready_to_trade:
            return "Market connectors are not ready."
        lines = []
        warning_lines = []
        warning_lines.extend(self.network_warning(self.get_market_trading_pair_tuples()))

        # balance_df = self.get_balance_df()
        # lines.extend(["", "  Balances:"] + ["    " + line for line in balance_df.to_string(
        #                                                             index = False,
        #                                                             formatters = [no_format, no_format, format_3digits, format_3digits]).split("\n")])
        batch_analysis_df = self.batch_arb_profit_analysis_df()
        lines.extend(["", "  Arbitrage Opportunity:"] + \
                     ["    " + line for line in batch_analysis_df.to_string(index = False).split("\n")])
        

        warning_lines.extend(self.balance_warning(self.get_market_trading_pair_tuples()))
        # if len(warning_lines) > 0:
        #     lines.extend(["", "*** WARNINGS ***"] + warning_lines)
        return "\n".join(lines)
