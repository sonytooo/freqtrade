import numpy as np
import pandas as pd
from pandas import DataFrame
from freqtrade.strategy import IStrategy
import talib.abstract as ta
import pandas_ta as pta

class MidTF_VWAP_Strategy(IStrategy):
    """
    MidTF_VWAP_Strategy
    Target Timeframes: 4h, 1d
    Description: A momentum-based VWAP strategy. It triggers a long position when the price
    breaks above the VWAP line, demonstrating strong institutional buying pressure. It filters entries
    using the SMA 50 to ensure an overall uptrend and RSI > 50 to confirm positive momentum.
    """
    INTERFACE_VERSION = 3
    timeframe = '4h'
    can_short = False

    minimal_roi = {
        "0": 0.08,     # 8% profit target
        "1440": 0.04,  # After 1 day, reduce to 4%
        "2880": 0.02   # After 2 days, reduce to 2%
    }
    stoploss = -0.10   # 10% hard stop
    trailing_stop = True
    trailing_stop_positive = 0.02
    trailing_stop_positive_offset = 0.05
    trailing_only_offset_is_reached = True

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Calculate RSI
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)

        # Calculate SMA 50
        dataframe['sma_50'] = ta.SMA(dataframe, timeperiod=50)

        # VWAP using pandas_ta
        # Freqtrade provides volume, close, high, low
        if 'vwap' not in dataframe.columns:
            # pandas_ta requires a datetime index to calculate VWAP correctly
            dataframe.index = pd.to_datetime(dataframe['date'])
            dataframe['vwap'] = pta.vwap(dataframe['high'], dataframe['low'], dataframe['close'], dataframe['volume'])
            dataframe.reset_index(drop=True, inplace=True) # Reset index
            # Fill NaNs for VWAP
            dataframe['vwap'] = dataframe['vwap'].ffill()

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['close'] > dataframe['vwap']) &          # Price above VWAP shows buying pressure
                (dataframe['close'] > dataframe['sma_50']) &        # Price above SMA 50 (Uptrend)
                (dataframe['rsi'] > 50) &                           # Momentum is positive
                (dataframe['volume'] > 0)
            ),
            'enter_long'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['close'] < dataframe['vwap']) &          # Price falls below VWAP
                (dataframe['rsi'] > 75) &                           # Overbought
                (dataframe['volume'] > 0)
            ),
            'exit_long'] = 1

        return dataframe
