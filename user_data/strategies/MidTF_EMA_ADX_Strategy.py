import numpy as np
import pandas as pd
from pandas import DataFrame
from freqtrade.strategy import IStrategy
import talib.abstract as ta

class MidTF_EMA_ADX_Strategy(IStrategy):
    """
    MidTF_EMA_ADX_Strategy
    Target Timeframes: 4h, 1d
    Description: Combines fast and slow EMAs to detect trend direction, and filters entries
    using ADX to ensure the trend is strong before entering. A powerful combo to avoid ranging markets.
    """
    INTERFACE_VERSION = 3
    timeframe = '4h'
    can_short = False

    minimal_roi = {
        "0": 0.12,
        "1440": 0.06
    }
    stoploss = -0.10
    trailing_stop = True
    trailing_stop_positive = 0.02
    trailing_stop_positive_offset = 0.06
    trailing_only_offset_is_reached = True

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # EMA
        dataframe['ema_9'] = ta.EMA(dataframe, timeperiod=9)
        dataframe['ema_21'] = ta.EMA(dataframe, timeperiod=21)
        dataframe['ema_50'] = ta.EMA(dataframe, timeperiod=50)

        # ADX
        dataframe['adx'] = ta.ADX(dataframe, timeperiod=14)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['ema_9'] > dataframe['ema_21']) &             # Fast EMA above Medium EMA
                (dataframe['ema_21'] > dataframe['ema_50']) &            # Medium EMA above Slow EMA (Bullish alignment)
                (dataframe['ema_9'].shift(1) <= dataframe['ema_21'].shift(1)) & # Crossover trigger
                (dataframe['adx'] > 25) &                                # Strong trend strength
                (dataframe['volume'] > 0)
            ),
            'enter_long'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['ema_9'] < dataframe['ema_21']) &             # Fast EMA crosses below Medium EMA
                (dataframe['volume'] > 0)
            ),
            'exit_long'] = 1

        return dataframe
