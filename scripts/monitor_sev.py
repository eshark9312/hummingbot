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
    exchanges = ["mexc", "gate_io", "kucoin", "binance", "htx"]
    arb_threshold = {"off_balance" : 0.7, "in_balance" : 0.5}           # threshold for take profit in percentage
    duration_threshold = 0.3            # threshold for duration of arb_opportunity to capture
    is_trade_on = True

    quote_amount = 50
    # base_assets = ["FTM", "NIBI", "AEG", "DECHAT"]
    # markets = {"mexc": {"NIBI-USDT", "DECHAT-USDT"},
    #            "kucoin": {"FTM-USDT", "NIBI-USDT", "AEG-USDT", "DECHAT-USDT"},
    #            "gate_io": {"FTM-USDT", "NIBI-USDT", "AEG-USDT", "DECHAT-USDT"},
    #            "binance": {"FTM-USDT"},
    #            "htx": {"FTM-USDT", "DECHAT-USDT"}}
    base_assets = ["JOYSTREAM", "FTM", "DECHAT", "TARA", "EGO"]
    markets = {"mexc": {"DECHAT-USDT", "JOYSTREAM-USDT", "EGO-USDT", "TARA-USDT"},
               "kucoin": {"FTM-USDT", "EGO-USDT", "DECHAT-USDT"},
               "gate_io": {"FTM-USDT", "DECHAT-USDT", "JOYSTREAM-USDT", "TARA-USDT"},
               "binance": {"FTM-USDT"},
               "htx": {"FTM-USDT", "DECHAT-USDT"}}
    opportunity_ts = {f"{base}-USDT": {} for base in base_assets}
    init_tot_balance = {f"{base}-USDT": {base:0, "USDT":0} for base in base_assets}
    init_tot_balance["initialized"] = False
    sqlite_conn = sqlite3.connect('opportunity_log_0329.db')
    tb_name = "log_04_01"
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
        if not self.ready_to_trade:
            pass
        if self.init_tot_balance["initialized"]:
            pass
        else:
            self._init_tot_balance()   # initialize the total balance of assets
        
        # vwap_prices = self.get_vwap_prices_for_amount(self.order_amount)
        # proposal = self.check_profitability_and_create_proposal(vwap_prices)
        # if len(proposal) > 0:
        #     proposal_adjusted: Dict[str, OrderCandidate] = self.adjust_proposal_to_budget(proposal)
        #     # self.place_orders(proposal_adjusted)

    def _init_tot_balance(self):
        for base in self.base_assets:
            pair = f"{base}-USDT"
            avail_exchanges = [exchange for exchange in self.exchanges if pair in self.markets[exchange]]
            num_exchanges = len(avail_exchanges)
            tot_base_balance = 0
            tot_quote_balance = 0
            for i in range(num_exchanges):
                exchange = avail_exchanges[i]
                tot_base_balance += self.connectors[exchange].get_balance(base)
                tot_quote_balance += self.connectors[exchange].get_balance('USDT')
            self.init_tot_balance[pair][base] = tot_base_balance
            self.init_tot_balance[pair]["USDT"] = tot_quote_balance
        self.init_tot_balance["initialized"] = True

    def get_vwap_prices_for_amount(self, pair: str, exchanges: list, amount: Decimal) -> Dict:
        """
        get vwap prices for certain amount of buy/sell
        """
        vwap_prices = {"order_amount":amount}
        for exchange in exchanges:
            vwap_ask_price = self.connectors[exchange].get_vwap_for_volume(pair, True, amount).result_price
            vwap_bid_price = self.connectors[exchange].get_vwap_for_volume(pair, False, amount).result_price
            vwap_prices[exchange] = {"ask": vwap_ask_price, "bid": vwap_bid_price}
        return vwap_prices

    def get_fees_percentages(self, pair: str, exchanges: list, amount: float = 1000) -> Dict:
        # We assume that the fee percentage for buying or selling is the same
        base_currency=pair.split('-')[0]
        quote_currency=pair.split('-')[1]
        fee_rates = {}
        for exchange in exchanges:
            fee_rates[exchange] = self.connectors[exchange].get_fee(
                            base_currency=base_currency,
                            quote_currency=quote_currency,
                            order_type=OrderType.MARKET,
                            order_side=TradeType.BUY,
                            amount=amount,
                            is_maker=False
                        ).percent
        # Correct fee rates for some exchanges manually
        if 'mexc' in fee_rates:
            fee_rates['mexc'] = Decimal(0.001)
        if 'gate_io' in fee_rates:
            fee_rates['gate_io'] = Decimal(0.001)
        return fee_rates

    def get_profitability_analysis(self, 
                                   vwap_prices: Dict[str, Any], 
                                   fee_rates: Dict[str, Any],
                                   exchange_A: str,
                                   exchange_B: str) -> Dict:
        """
        perform profitability analysis for arbitraging a pair between A & B
        """
        base_order_amount = vwap_prices["order_amount"]
        buy_a_sell_b_quote = vwap_prices[exchange_B]["bid"] * (1 - fee_rates[exchange_B]) * base_order_amount - \
            vwap_prices[exchange_A]["ask"] * (1 + fee_rates[exchange_A]) * base_order_amount
        buy_a_sell_b_base = buy_a_sell_b_quote / (
            (vwap_prices[exchange_A]["ask"] + vwap_prices[exchange_B]["bid"]) / 2)

        buy_b_sell_a_quote = vwap_prices[exchange_A]["bid"] * (1 - fee_rates[exchange_A]) * base_order_amount - \
            vwap_prices[exchange_B]["ask"] * (1 + fee_rates[exchange_B]) * base_order_amount

        buy_b_sell_a_base = buy_b_sell_a_quote / (
            (vwap_prices[exchange_B]["ask"] + vwap_prices[exchange_A]["bid"]) / 2)

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

    def _take_opportunity_ts(self, pair: str, exchange_A: str, exchange_B: str, is_buy_A:bool, 
                                   est_profit: float, amount: Decimal, price: Decimal):
        if is_buy_A:
            exchange_buy = exchange_A
            exchange_sell = exchange_B
        else:
            exchange_buy = exchange_B
            exchange_sell = exchange_A
        base_currency = pair.split('-')[0]
        # balance check
        if self.connectors[exchange_buy].get_balance("USDT") < self.quote_amount * 1.1:
            self.logger().info(f"Insufficient USDT balance to buy {amount} {base_currency} on {exchange_buy}")
            return
        if self.connectors[exchange_sell].get_balance(base_currency) < float(amount) * 1.1:
            self.logger().info(f"Insufficient {base_currency} balance to sell {amount} {base_currency} on {exchange_sell}")
            return
        # check whether the trade is in-balance / off-balance
        if self.connectors[exchange_sell].get_balance(base_currency) < self.connectors[exchange_buy].get_balance(base_currency):
            # base_asset off_balance and est_profit is less than off_balance_threshold -> ignore opportunity
            if est_profit < self.arb_threshold["off_balance"]:
                self.logger().info(f"Not high enough profit for off_balance trading")
                return
        # run market orders
        self.buy(amount = amount, price = price,
                 connector_name = exchange_buy,
                 order_type = OrderType.MARKET,
                 trading_pair = pair)
        self.sell(amount = amount, price = price,
                 connector_name = exchange_sell,
                 order_type = OrderType.MARKET,
                 trading_pair = pair)
        # update time stamp
        ex_pair = f"{exchange_A}-{exchange_B}"
        if self.opportunity_ts[pair][ex_pair]["buy_a_sell_b" if is_buy_A else "buy_b_sell_a"] != 0:
            self.opportunity_ts[pair][ex_pair]["buy_a_sell_b" if is_buy_A else "buy_b_sell_a"] = time.time()

    def _update_opportunity_ts(self,
                               pair: str, 
                               profit: Decimal, vwap_prices: Dict[str, Any],
                               exchange_A: str, exchange_B: str, is_buy_A:bool) -> str:
        ex_pair = f"{exchange_A}-{exchange_B}"
        if ex_pair not in self.opportunity_ts[pair]:
            self.opportunity_ts[pair][ex_pair] = {"buy_a_sell_b":0, "profit_a_b":0,
                                                  "buy_b_sell_a":0, "profit_b_a":0,
                                                  "buy_price":0, "sell_price":0}
        if self.opportunity_ts[pair][ex_pair]["buy_a_sell_b" if is_buy_A else "buy_b_sell_a"] == 0:
            if profit >= self.arb_threshold["in_balance"]:             # start_ts of the arbitrage opportunity
                self.opportunity_ts[pair][ex_pair]["buy_a_sell_b" if is_buy_A else "buy_b_sell_a"] = time.time()
                self.opportunity_ts[pair][ex_pair]["profit_a_b" if is_buy_A else "profit_b_a"] = profit
                self.opportunity_ts[pair][ex_pair]["buy_price"] = vwap_prices[exchange_A]["ask"] if is_buy_A else vwap_prices[exchange_B]["ask"]
                self.opportunity_ts[pair][ex_pair]["sell_price"] = vwap_prices[exchange_B]["bid"] if is_buy_A else vwap_prices[exchange_A]["bid"]
            return ""
        else:
            duration = time.time() - self.opportunity_ts[pair][ex_pair]["buy_a_sell_b" if is_buy_A else "buy_b_sell_a"]
            if profit > self.arb_threshold["in_balance"] and duration > self.duration_threshold:
                if self.is_trade_on :
                    # take opportunity
                    base_amount = vwap_prices["order_amount"]
                    buy_price = vwap_prices[exchange_A if is_buy_A else exchange_B]["ask"]
                    self._take_opportunity_ts(pair = pair, est_profit = profit,
                                            exchange_A = exchange_A, exchange_B = exchange_B, is_buy_A = is_buy_A, 
                                            amount = Decimal(base_amount), price = Decimal(buy_price))
            if profit > self.opportunity_ts[pair][ex_pair]["profit_a_b" if is_buy_A else "profit_b_a"]:    # update max_profit and buy/sell_prices
                self.opportunity_ts[pair][ex_pair]["profit_a_b" if is_buy_A else "profit_b_a"] = profit
                self.opportunity_ts[pair][ex_pair]["buy_price"] = vwap_prices[exchange_A]["ask"] if is_buy_A else vwap_prices[exchange_B]["ask"]
                self.opportunity_ts[pair][ex_pair]["sell_price"] = vwap_prices[exchange_B]["bid"] if is_buy_A else vwap_prices[exchange_A]["bid"]
            if profit < self.arb_threshold["in_balance"]:             # end_ts of the arbitrage opportunity
                self.opportunity_ts[pair][ex_pair]["buy_a_sell_b" if is_buy_A else "buy_b_sell_a"] = 0
                max_profit = self.opportunity_ts[pair][ex_pair]["profit_a_b" if is_buy_A else "profit_b_a"]
                self.opportunity_ts[pair][ex_pair]["profit_a_b" if is_buy_A else "profit_b_a"] = 0
                if duration > self.duration_threshold:          # log opportunity into sqlite3db and hb_logs
                    data = {"datetime": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            "coin": pair,
                            "buy": exchange_A if is_buy_A else exchange_B,
                            "sell": exchange_B if is_buy_A else exchange_A,
                            "buy_price": f"{self.opportunity_ts[pair][ex_pair]['buy_price']:.4f}",
                            "sell_price": f"{self.opportunity_ts[pair][ex_pair]['sell_price']:.4f}",
                            "profit": f"{max_profit:.2f}",
                            "duration": f"{duration:.1f}"}
                    self._log_orb_opportunity_sqlite(data)
                    info_msg = f"{pair} : {exchange_A} -> {exchange_B} {max_profit:.2f} % ({duration:.1f})" if is_buy_A else \
                               f"{pair} : {exchange_B} -> {exchange_A} {max_profit:.2f} % ({duration:.1f})"
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

    def pair_profit_analysis_df(self, pair: str):
        avail_exchanges = [exchange for exchange in self.exchanges if pair in self.markets[exchange]]
        # initialize df for displaying profit_analysis
        columns = avail_exchanges.copy()
        columns = ["BUY -> SELL (Bal) |"]
        num_exchanges = len(avail_exchanges)
        data = [[' ' for _ in range(num_exchanges)] for _ in range(num_exchanges)]
        base_currency=pair.split('-')[0]
        # get analysis data 
        order_amount = self.quote_amount / self.connectors[avail_exchanges[0]].get_mid_price(trading_pair = pair)
        vwap_prices = self.get_vwap_prices_for_amount(amount = order_amount, pair = pair, exchanges = avail_exchanges)
        fee_rates = self.get_fees_percentages(amount = order_amount, pair = pair, exchanges = avail_exchanges)
        for i in range(num_exchanges):
            for j in range(i + 1, num_exchanges):
                exchange_A = avail_exchanges[i]
                exchange_B = avail_exchanges[j]
                profitability_analysis = self.get_profitability_analysis(
                                                vwap_prices = vwap_prices,
                                                fee_rates = fee_rates,
                                                exchange_A = exchange_A,
                                                exchange_B = exchange_B)
                buy_a_sell_b_profit = f"{profitability_analysis['buy_a_sell_b']['profitability_pct'] * 100:.2f}"
                buy_b_sell_a_profit = f"{profitability_analysis['buy_b_sell_a']['profitability_pct'] * 100:.2f}"
                buy_a_sell_b_price = f"{vwap_prices[exchange_A]['ask']:.4f}->{vwap_prices[exchange_B]['bid']:.4f}"
                buy_b_sell_a_price = f"{vwap_prices[exchange_B]['ask']:.4f}->{vwap_prices[exchange_A]['bid']:.4f}"
                buy_a_sell_b_duration = self._update_opportunity_ts(pair = pair,
                                                                exchange_A = exchange_A, exchange_B = exchange_B,
                                                                profit = Decimal(buy_a_sell_b_profit),
                                                                vwap_prices = vwap_prices,
                                                                is_buy_A = True)
                buy_b_sell_a_duration = self._update_opportunity_ts(pair = pair,
                                                                exchange_A = exchange_A, exchange_B = exchange_B,
                                                                profit = Decimal(buy_b_sell_a_profit),
                                                                vwap_prices = vwap_prices,
                                                                is_buy_A = False)
                data[j][i] = f"{buy_b_sell_a_profit}({buy_b_sell_a_price if buy_b_sell_a_duration == '' else buy_b_sell_a_duration}) |"
                data[i][j] = f"{buy_a_sell_b_profit}({buy_a_sell_b_price if buy_a_sell_b_duration == '' else buy_a_sell_b_duration}) |"
        tot_base_balance = 0
        tot_quote_balance = 0
        for i in range(num_exchanges):
            exchange = avail_exchanges[i]
            base_balance = self.connectors[exchange].get_balance(base_currency)
            quote_balance = self.connectors[exchange].get_balance('USDT')
            tot_base_balance += base_balance
            tot_quote_balance += quote_balance
            columns.append(exchange.rjust(12) + f" ( {base_balance:8,.1f} ) |")
            data[i].insert(0, exchange.rjust(9) + f"| {quote_balance:8,.1f} |")
        df = pd.DataFrame(data = data, columns = columns)
        return {"pd.DataFrame": df,
                "tot_base_balance": tot_base_balance,
                "tot_quote_balance": tot_quote_balance}

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
        for base in self.base_assets:
            pair_profit_analysis_data = self.pair_profit_analysis_df(f"{base}-USDT")
            pair_profit_analysis_df = pair_profit_analysis_data["pd.DataFrame"]
            tot_base_balance = pair_profit_analysis_data["tot_base_balance"]
            tot_quote_balance = pair_profit_analysis_data["tot_quote_balance"]
            init_base_balance = self.init_tot_balance[f"{base}-USDT"][base]
            init_quote_balance = self.init_tot_balance[f"{base}-USDT"]["USDT"]
            diff_base_balance = tot_base_balance - init_base_balance
            diff_quote_balance = tot_quote_balance - init_quote_balance
            header = f"{base}-USDT  |".rjust(25) + base.rjust(9) + f" : {tot_base_balance:9,.1f}({diff_base_balance:9,.1f})  || " + \
                                                                   f" {tot_quote_balance:9,.1f}({diff_quote_balance:9,.1f})"
            lines.extend(["", f"{header}", "-"*24 + "+" + "-"*27 + "+" + "-"*27 + "+" + "-"*27 ] + \
                     ["    " + line for line in pair_profit_analysis_df.to_string(index = False).split("\n")])
        return "\n".join(lines)
