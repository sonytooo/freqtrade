import numpy as np
import pandas as pd
from pandas import DataFrame
from freqtrade.strategy import IStrategy
import talib.abstract as ta

class MidTF_MACD_RSI_Strategy(IStrategy):
    """
    MidTF_MACD_RSI_Strategy
    Target Timeframes: 4h, 1d
    Description: A classic trend-following strategy that triggers when the MACD line crosses
    above the signal line while the RSI indicates there is room to grow (not overbought).
    Highly effective for capturing long swing trades on 4h/1d charts.
    """
    INTERFACE_VERSION = 3
    timeframe = '4h'
    can_short = False

    minimal_roi = {
        "0": 0.15,     # 15% profit target
        "2880": 0.05   # Reduce to 5% after 2 days
    }
    stoploss = -0.15   # 15% stop loss for high volatility mid-timeframes
    trailing_stop = True
    trailing_stop_positive = 0.03
    trailing_stop_positive_offset = 0.08
    trailing_only_offset_is_reached = True

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # MACD
        macd = ta.MACD(dataframe)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macdhist'] = macd['macdhist']

        # RSI
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)

        # EMA 200 for trend filter
        dataframe['ema_200'] = ta.EMA(dataframe, timeperiod=200)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['macd'] > dataframe['macdsignal']) &              # MACD crossover
                (dataframe['macd'].shift(1) <= dataframe['macdsignal'].shift(1)) & # Just crossed
                (dataframe['rsi'] < 60) &                                    # Not overbought
                (dataframe['close'] > dataframe['ema_200']) &                # Overall bullish trend
                (dataframe['volume'] > 0)
            ),
            'enter_long'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['macd'] < dataframe['macdsignal']) &              # MACD cross under
                (dataframe['macd'].shift(1) >= dataframe['macdsignal'].shift(1)) &
                (dataframe['rsi'] > 70) &                                    # RSI overbought
                (dataframe['volume'] > 0)
            ),
            'exit_long'] = 1

        return dataframe
