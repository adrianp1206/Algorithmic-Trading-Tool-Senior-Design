import sys
import os

# Get the absolute path to the `src` directory
src_path = os.path.abspath(os.path.join('..', 'src'))

# Add the `src` directory to the Python path
if src_path not in sys.path:
    sys.path.append(src_path)

import unittest
import pandas as pd
import numpy as np
from data_processing import calculate_technical_indicators 

class TestTechnicalIndicators(unittest.TestCase):

   def setUp(self):
    # Sample data with at least 20 rows for indicators with look-back periods
    data = {
        'Open': [100 + i for i in range(20)],
        'High': [101 + i for i in range(20)],
        'Low': [99 + i for i in range(20)],
        'Close': [100 + i for i in range(20)],
        'Volume': [1000 + 10 * i for i in range(20)]
    }
    self.df = pd.DataFrame(data)

    def test_calculate_technical_indicators(self):
        # Apply the technical indicator calculations
        result_df = calculate_technical_indicators(self.df)

        # Define the expected indicators to test
        expected_indicators = [
            'BB_upper', 'BB_middle', 'BB_lower', 'DEMA', 'MIDPOINT', 'MIDPRICE', 'SMA', 'T3', 
            'TEMA', 'TRIMA', 'WMA', 'ADX', 'ADXR', 'APO', 'AROON_DOWN', 'AROON_UP', 'AROONOSC', 
            'CCI', 'CMO', 'MACD', 'MACD_signal', 'MACD_hist', 'MFI', 'MINUS_DI', 'MINUS_DM', 'MOM', 
            'PLUS_DI', 'PLUS_DM', 'ROC', 'RSI', 'STOCH_slowk', 'STOCH_slowd', 'STOCH_fastk', 
            'STOCH_fastd', 'ATR', 'TRANGE', 'AD', 'OBV', 'AVGPRICE', 'MEDPRICE', 
            'TYPPRICE', 'WCLPRICE', 'NATR', 'ADOSC', 'HT_DCPERIOD', 'HT_DCPHASE', 'HT_PHASOR_inphase',
            'HT_PHASOR_quadrature', 'HT_SINE', 'HT_LEADSINE', 'HT_TRENDMODE'
        ]

        # Check if each expected indicator column exists and contains valid values
        for indicator in expected_indicators:
            with self.subTest(indicator=indicator):
                self.assertIn(indicator, result_df.columns, f"{indicator} not found in the DataFrame")
                self.assertFalse(result_df[indicator].isnull().all(), f"{indicator} has all null values")
                self.assertTrue(np.issubdtype(result_df[indicator].dtype, np.number), f"{indicator} is not numeric")

if __name__ == '__main__':
    unittest.main()
