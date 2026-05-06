from datetime import datetime
import json
import logging
from typing import Dict, Optional

from pandas import DataFrame
import pandas as pd
import redis

from freqtrade.strategy import IStrategy
from freqtrade.strategy import IntParameter, DecimalParameter, BooleanParameter
from freqtrade.strategy import stoploss_from_open

import smartmoneyconcepts as smc

def test_imports():
    print("All imports successful!")

if __name__ == '__main__':
    test_imports()
