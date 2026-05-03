import logging
from pandas import DataFrame
from freqtrade.strategy import IStrategy
import talib.abstract as ta

logger = logging.getLogger(__name__)

class MidTF_FreqAI_Strategy(IStrategy):
    """
    MidTF_FreqAI_Strategy
    Target Timeframes: 4h, 1d
    Description: Integrates with FreqAI to predict market direction. We feed technical indicators
    into the AI model (configured via FreqAI settings in config.json). The AI returns
    a prediction (up or down), and we trade based on that prediction combined with basic risk management.
    """
    INTERFACE_VERSION = 3
    timeframe = '4h'
    can_short = False

    minimal_roi = {
        "0": 0.15,
        "1440": 0.05
    }
    stoploss = -0.15
    trailing_stop = True
    trailing_stop_positive = 0.02
    trailing_stop_positive_offset = 0.05
    trailing_only_offset_is_reached = True

    def feature_engineering_expand_all(self, dataframe: DataFrame, period: int, metadata: dict, **kwargs) -> DataFrame:
        """
        *Only for FreqAI*. These are the features FreqAI will use to train the model.
        """
        dataframe[f"%-rsi-{period}"] = ta.RSI(dataframe, timeperiod=period)
        dataframe[f"%-mfi-{period}"] = ta.MFI(dataframe, timeperiod=period)
        dataframe[f"%-adx-{period}"] = ta.ADX(dataframe, timeperiod=period)

        macd = ta.MACD(dataframe, fastperiod=int(period/2), slowperiod=period, signalperiod=int(period/3))
        dataframe[f"%-macd-{period}"] = macd['macd']
        dataframe[f"%-macdsignal-{period}"] = macd['macdsignal']

        dataframe[f"%-ema-{period}"] = ta.EMA(dataframe, timeperiod=period)
        return dataframe

    def feature_engineering_expand_basic(self, dataframe: DataFrame, metadata: dict, **kwargs) -> DataFrame:
        """
        *Only for FreqAI*. Basic features to feed to the AI.
        """
        dataframe["%-pct-change"] = dataframe["close"].pct_change()
        dataframe["%-raw_volume"] = dataframe["volume"]
        return dataframe

    def feature_engineering_standard(self, dataframe: DataFrame, metadata: dict, **kwargs) -> DataFrame:
        """
        *Only for FreqAI*.
        """
        dataframe["%-day_of_week"] = dataframe["date"].dt.dayofweek
        return dataframe

    def set_freqai_targets(self, dataframe: DataFrame, metadata: dict, **kwargs) -> DataFrame:
        """
        *Only for FreqAI*. This defines what the AI is trying to predict.
        Here we predict if the price will go up by at least 2% in the next few candles.
        """
        # We classify as '1' (Up) if the close price X candles in the future is > 2% higher than now.
        # Note: The actual prediction target shifts and evaluation is handled automatically by FreqAI
        # using `label_period_candles` from the config.
        label_period_candles = self.freqai_info["feature_parameters"]["label_period_candles"]

        import numpy as np
        # Use string labels instead of integers
        dataframe["&-target"] = np.where(
            dataframe["close"].shift(-label_period_candles) > dataframe["close"] * 1.02,
            'up',
            'down'
        )

        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # FreqAI handles standard indicator population internally based on the feature engineering functions
        dataframe = self.freqai.start(dataframe, metadata, self)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # The AI creates a 'do_predict' column. If 'do_predict' is 1 and the target prediction is 'up', we enter.
        dataframe.loc[
            (
                (dataframe['do_predict'] == 1) &
                (dataframe['&-target'] == 'up') & # FreqAI predicts the market is going UP
                (dataframe['volume'] > 0)
            ),
            'enter_long'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Exit if FreqAI predicts the market will not go up (target prediction 'down')
        dataframe.loc[
            (
                (dataframe['do_predict'] == 1) &
                (dataframe['&-target'] == 'down') & # FreqAI predicts the market is NOT going up
                (dataframe['volume'] > 0)
            ),
            'exit_long'] = 1

        return dataframe
