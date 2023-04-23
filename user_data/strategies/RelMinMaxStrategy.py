"""Implement relative min-max local strategy."""
from typing import Dict, List, Optional, Tuple, Union
from functools import reduce
from datetime import datetime, timedelta, timezone
import talib.abstract as ta
from technical import qtpylib
import pandas_ta as pta
from pandas import DataFrame
import numpy as np
from freqtrade.strategy.interface import IStrategy
from freqtrade.strategy import DecimalParameter, IntParameter, merge_informative_pair
from freqtrade.persistence import Trade
from freqtrade.exchange import timeframe_to_prev_date
from user_data.strategies.utils import get_max_labels, get_min_labels
from technical import qtpylib


class RelMinMaxStrategy(IStrategy):
    
    minimal_roi = {"0": 0.1, "240": -1}

    plot_config = {
        "main_plot": {},
        "subplots": {
            "do_predict": {
                "do_predict": {
                    "color": "brown"
                }
            },
            "DI_values": {
                "DI_values": {
                    "color": "#8115a9",
                    "type": "line"
                }
            },
            "GTs": {
                "tp_max": {
                    "color": "#69796a",
                    "type": "bar"
                },
                "tp_min": {
                    "color": "#e2517f",
                    "type": "bar"
                },
                 "max": {
                     "color": "#69796a",
                     "type": "line"
                 },
                 "min": {
                    "color": "#e2517f",
                    "type": "line"
                },
                 "neutral": {
                     "color": "#ffffff",
                    "type": "line"
                 }
            },
            "di+-": {
                "di-": {
                    "color": "#e5e95b",
                    "type": "line"
                },
                "di+": {
                    "color": "#cda122",
                    "type": "line"
                }
            }
        }
    }

    position_adjustment_enable = False

    process_only_new_candles = True
    stoploss = -0.05
    use_exit_signal = True
    startup_candle_count: int = 300
    can_short = True
    panic_market = False

    linear_roi_offset = DecimalParameter(
        0.00, 0.02, default=0.005, space="sell", optimize=False, load=True
    )
    # max roi time long in minutes
    # max_roi_time_long = IntParameter(0, 800, default=341, space="sell", optimize=True, load=True)  # BTC
    max_roi_time_long = IntParameter(0, 800, default=60 * 4, space="sell", optimize=True, load=False)
    entry_thr = DecimalParameter(0.5, 1.0, default=0.7, space="buy", optimize=True, load=False)
    exit_thr = DecimalParameter(0.5, 1.0, default=0.7, space="sell", optimize=True, load=False)

    def feature_engineering_expand_all(self, dataframe, period, metadata, **kwargs):
        dataframe["%-rsi-period"] = ta.RSI(dataframe, timeperiod=period)
        dataframe["%-mfi-period"] = ta.MFI(dataframe, timeperiod=period)
        # CTI
        # dataframe["%-CTI-period"] = pta.cti(dataframe["close"], length=period)
        # TODO: ROCR
        dataframe["%-adx-period"] = ta.ADX(dataframe, timeperiod=period)
        dataframe["%-relative_volume-period"] = (
            dataframe["volume"] / dataframe["volume"].rolling(period).mean()
        )
        return dataframe

    def feature_engineering_expand_basic(self, dataframe: DataFrame, metadata: Dict, **kwargs):
        """
        *Only functional with FreqAI enabled strategies*
        This function will automatically expand the defined features on the config defined
        `include_timeframes`, `include_shifted_candles`, and `include_corr_pairs`.
        In other words, a single feature defined in this function
        will automatically expand to a total of
        `include_timeframes` * `include_shifted_candles` * `include_corr_pairs`
        numbers of features added to the model.

        Features defined here will *not* be automatically duplicated on user defined
        `indicator_periods_candles`

        All features must be prepended with `%` to be recognized by FreqAI internals.

        Access metadata such as the current pair/timeframe with:

        `metadata["pair"]` `metadata["tf"]`

        More details on how these config defined parameters accelerate feature engineering
        in the documentation at:

        https://www.freqtrade.io/en/latest/freqai-parameter-table/#feature-parameters

        https://www.freqtrade.io/en/latest/freqai-feature-engineering/#defining-the-features

        :param dataframe: strategy dataframe which will receive the features
        :param metadata: metadata of current pair
        dataframe["%-pct-change"] = dataframe["close"].pct_change()
        dataframe["%-ema-200"] = ta.EMA(dataframe, timeperiod=200)
        """
        dataframe["%-pct-change"] = dataframe["close"].pct_change()
        # dataframe = dataframe.set_index(dataframe.date)
        #dataframe["%-vwap"] = pta.core.vwap(dataframe.high, dataframe.low, dataframe.close, dataframe.volume)
        dataframe["%-vwap"] = qtpylib.rolling_vwap(dataframe, window=12)
        return dataframe
    
    def set_freqai_targets(self, dataframe, **kwargs):
        minmax = np.array(["neutral"] * len(dataframe), dtype=np.object0)
        min_labels = get_min_labels(df=dataframe, window=12, alpha=0.5)
        max_labels = get_max_labels(df=dataframe, window=12, alpha=0.5)
        minmax[min_labels == 1] = "min"
        minmax[max_labels == 1] = "max"
        dataframe["&s-minmax"] = np.array([str(x) for x in minmax]).astype(np.object0)
        return dataframe
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        # All indicators must be populated by feature_engineering_*() functions

        # the model will return all labels created by user in `feature_engineering_*`
        # (& appended targets), an indication of whether or not the prediction should be accepted,
        # the target mean/std values for each of the labels created by user in
        # `set_freqai_targets()` for each training period.

        dataframe = self.freqai.start(dataframe, metadata, self)

        # dataframe["&s-minima"] = dataframe["&s-minima"].astype(np.float32)
        # dataframe["&s-maxima"] = dataframe["&s-maxima"].astype(np.float32)
        min_labels = get_min_labels(df=dataframe, window=12, alpha=0.5)
        max_labels = get_max_labels(df=dataframe, window=12, alpha=0.5)

        self.maxima_threhsold = 0.7 # dataframe["max"][dataframe["&s-minmax"] == "max"].mean()
        self.minima_threhsold = 0.7 # dataframe["min"][dataframe["&s-minmax"] == "min"].mean()

        dataframe["tp_max"] = max_labels.astype(np.float32)
        dataframe["tp_min"] = min_labels.astype(np.float32)
        dataframe["di-"] = ta.MINUS_DI(dataframe, window=12)
        dataframe["di+"] = ta.PLUS_DI(dataframe, window=12)
        return dataframe

    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:

        last_candles_are_stable = True
        if self.panic_market:
            hours_candle_stability = 4
            # enter the market if last `hours_candle_stability` are stable
            last_candles_are_stable = df["do_predict"].rolling(12 * hours_candle_stability).sum().iloc[-1] == 12 * hours_candle_stability

        if last_candles_are_stable:
            self.panic_market = False
            enter_long_conditions = [df["do_predict"] == 1, df["min"] >= self.entry_thr.value]

            if enter_long_conditions:
                df.loc[
                    reduce(lambda x, y: x & y, enter_long_conditions), ["enter_long", "enter_tag"]
                ] = (1, "long")

            if self.can_short:
                enter_short_conditions = [df["do_predict"] == 1, df["max"] >= self.exit_thr.value]

                if enter_short_conditions:
                    df.loc[
                        reduce(lambda x, y: x & y, enter_short_conditions), ["enter_short", "enter_tag"]
                    ] = (1, "short")
        else:
            df["enter_long"] = np.zeros(df.shape[0])
        return df

    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        exit_long_conditions = [df["do_predict"] == 1, df["max"] >= self.exit_thr.value]
        if exit_long_conditions:
            df.loc[reduce(lambda x, y: x & y, exit_long_conditions),
                   ["exit_long", "exit_tag"]] = (1, "exit signal")

        exit_long_missed_conditions = [df["do_predict"] == 1, df["tp_max"].shift(1) == 1]
        if exit_long_missed_conditions:
            df.loc[reduce(lambda x, y: x & y, exit_long_missed_conditions),
                   ["exit_long", "exit_tag"]] = (1, "exit_long_missed")

        if self.can_short:
            exit_short_conditions = [df["do_predict"] == 1, df["min"] >= self.entry_thr.value]
            if exit_short_conditions:
                df.loc[reduce(lambda x, y: x & y, exit_short_conditions),
                       ["exit_short", "exit_tag"]] = (1, "exit signal")
        return df

    def confirm_trade_exit(
        self,
        pair: str,
        trade: Trade,
        order_type: str,
        amount: float,
        rate: float,
        time_in_force: str,
        exit_reason: str,
        current_time,
        **kwargs,
    ) -> bool:

        entry_tag = trade.enter_tag
        follow_mode = self.config.get("freqai", {}).get("follow_mode", False)
        if not follow_mode:
            pair_dict = self.freqai.dd.pair_dict
        else:
            pair_dict = self.freqai.dd.follower_dict

        pair_dict[pair]["prediction" + entry_tag] = 0
        if not follow_mode:
            self.freqai.dd.save_drawer_to_disk()
        else:
            self.freqai.dd.save_follower_dict_to_disk()

        return True

    def confirm_trade_entry(
        self,
        pair: str,
        order_type: str,
        amount: float,
        rate: float,
        time_in_force: str,
        current_time,
        entry_tag,
        side: str,
        **kwargs,
    ) -> bool:

        df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = df.iloc[-1].squeeze()

        if side == "long":
            if rate > (last_candle["close"] * (1 + 0.0025)):
                return False
        else:
            if rate < (last_candle["close"] * (1 - 0.0025)):
                return False

        return True

    def custom_exit(
        self, pair: str, trade: Trade, current_time, current_rate, current_profit, **kwargs
    ):
        return None
        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)

        if dataframe.empty:
            return None

        trade_date = timeframe_to_prev_date(self.config["timeframe"], trade.open_date_utc)
        trade_candle = dataframe.loc[(dataframe["date"] == trade_date)]

        if trade_candle.empty:
            return None
        trade_candle = trade_candle.squeeze()

        # OOD TRACKING to minimize or maximize profit
        # if dataframe["do_predict"].iloc[-1] != 1:
        #     
        #     if trade.is_short:
        #         avg_chg = (dataframe["close"] / dataframe["close"].shift(1) - 1) \
        #                   .abs().mean()
        #         std_chg = (dataframe["close"] / dataframe["close"].shift(1) - 1) \
        #                   .abs().std()
        #         direction = dataframe["close"].iloc[-12:].mean() / current_rate - 1
        #         if direction >= avg_chg + std_chg:
        #             self.ood_profit_track.append(current_profit)
        #             return None
        #         else:
        #             self.ood_profit_track = []
        #             return f"OOD_{trade.enter_tag}_End_Swim_Exit"
        #     else:
        #         avg_chg = (dataframe["close"].shift(1) / dataframe["close"] - 1)\
        #                   .abs().mean()
        #         std_chg = (dataframe["close"].shift(1) / dataframe["close"] - 1)\
        #                   .abs().std()
        #         direction = current_rate / dataframe["close"].iloc[-12:].mean() - 1
        #         if direction >= avg_chg + std_chg:
        #             self.ood_profit_track.append(current_profit)
        #             return None
        #         else:
        #             self.ood_profit_track = []
        #             return f"OOD_{trade.enter_tag}_End_Swim_Exit"
        #     self.ood_profit_track = []
        #     return f"OOD_{trade.enter_tag}_Exit"

        time_alpha = (1 + current_profit / (self.minimal_roi[0] - self.stoploss))
        time_alpha = 1.
        if (current_time - trade.open_date_utc).seconds > (self.max_roi_time_long.value * 60 * time_alpha):
            return f"{trade.enter_tag}_Expired"

    def adjust_trade_position(self, trade: Trade, current_time: datetime,
                              current_rate: float, current_profit: float,
                              min_stake: Optional[float], max_stake: float,
                              **kwargs) -> Optional[float]:
        """
        Custom trade adjustment logic, returning the stake amount that a trade should be increased.
        This means extra buy orders with additional fees.
        Only called when `position_adjustment_enable` is set to True.

        For full documentation please go to https://www.freqtrade.io/en/latest/strategy-advanced/

        When not implemented by a strategy, returns None

        :param trade: trade object.
        :param current_time: datetime object, containing the current datetime
        :param current_rate: Current buy rate.
        :param current_profit: Current profit (as ratio), calculated based on current_rate.
        :param min_stake: Minimal stake size allowed by exchange.
        :param max_stake: Balance available for trading.
        :param **kwargs: Ensure to keep this here so updates to this won't break your strategy.
        :return float: Stake amount to adjust your trade
        """
        return None
        if not trade.is_short:
            if current_profit < -0.02:
                df, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
                try:
                    new_local_minima = [df["&s-minima"] > self.minima_threhsold,
                                        (df["close"] / current_rate - 1) < 1e-3]
                    if df.shape[0] - df.loc[reduce(lambda x, y: x & y, new_local_minima)].index[-1] <= 10:
                        return 20
                except:
                    pass
        return None