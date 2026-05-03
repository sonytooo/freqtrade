import numpy as np
import pandas as pd
from pandas import DataFrame
from freqtrade.strategy import IStrategy
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib

class MidTF_Bollinger_Breakout_Strategy(IStrategy):
    """
    MidTF_Bollinger_Breakout_Strategy
    Target Timeframes: 4h, 1d
    Description: Trades breakouts of the upper Bollinger Band, accompanied by
    increasing volume. This is highly effective in crypto markets during strong bullish impulses.
    """
    INTERFACE_VERSION = 3
    timeframe = '4h'
    can_short = False

    minimal_roi = {
        "0": 0.20,     # Breakouts can run far, 20% target
        "1440": 0.10,
        "2880": 0.05
    }
    stoploss = -0.12
    trailing_stop = True
    trailing_stop_positive = 0.05
    trailing_stop_positive_offset = 0.10
    trailing_only_offset_is_reached = True

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Bollinger Bands
        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_upperband'] = bollinger['upper']
        dataframe['bb_middleband'] = bollinger['mid']
        dataframe['bb_lowerband'] = bollinger['lower']

        # SMA for Volume
        dataframe['volume_mean'] = dataframe['volume'].rolling(window=20).mean()

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['close'] > dataframe['bb_upperband']) &               # Breakout above upper band
                (dataframe['close'].shift(1) <= dataframe['bb_upperband'].shift(1)) & # Just happened
                (dataframe['volume'] > dataframe['volume_mean'] * 1.5) &         # 50% more volume than average
                (dataframe['volume'] > 0)
            ),
            'enter_long'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['close'] < dataframe['bb_middleband']) &              # Reverts back to mean
                (dataframe['volume'] > 0)
            ),
            'exit_long'] = 1

        return dataframe
