"""
Unit tests for portfolio metrics calculations.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from metrics import (
    calculate_returns,
    calculate_portfolio_returns,
    calculate_cumulative_returns,
    calculate_annualized_return,
    calculate_volatility,
    calculate_sharpe_ratio,
    calculate_max_drawdown,
    calculate_correlation_matrix
)


class TestMetricsCalculations(unittest.TestCase):
    """Test cases for portfolio metrics calculations."""
    
    def setUp(self):
        """Set up test data."""
        # Create sample price data
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        np.random.seed(42)
        
        # Generate synthetic price data
        self.price_data = pd.DataFrame({
            'AAPL': 100 * np.exp(np.cumsum(np.random.normal(0.001, 0.02, len(dates)))),
            'GOOGL': 200 * np.exp(np.cumsum(np.random.normal(0.0008, 0.025, len(dates)))),
            'MSFT': 150 * np.exp(np.cumsum(np.random.normal(0.0012, 0.022, len(dates))))
        }, index=dates)
        
        self.returns_data = calculate_returns(self.price_data)
        self.weights = {'AAPL': 0.4, 'GOOGL': 0.35, 'MSFT': 0.25}
    
    def test_calculate_returns(self):
        """Test daily returns calculation."""
        returns = calculate_returns(self.price_data)
        
        # Check that returns have one less row than prices
        self.assertEqual(len(returns), len(self.price_data) - 1)
        
        # Check that first return matches manual calculation
        expected_first_return = (self.price_data.iloc[1] / self.price_data.iloc[0]) - 1
        pd.testing.assert_series_equal(returns.iloc[0], expected_first_return, rtol=1e-10)
        
        # Test empty DataFrame
        empty_returns = calculate_returns(pd.DataFrame())
        self.assertTrue(empty_returns.empty)
    
    def test_calculate_portfolio_returns(self):
        """Test portfolio returns calculation."""
        portfolio_returns = calculate_portfolio_returns(self.returns_data, self.weights)
        
        # Check that portfolio returns have correct length
        self.assertEqual(len(portfolio_returns), len(self.returns_data))
        
        # Manually calculate first portfolio return
        expected_first_return = (
            self.returns_data.iloc[0]['AAPL'] * 0.4 +
            self.returns_data.iloc[0]['GOOGL'] * 0.35 +
            self.returns_data.iloc[0]['MSFT'] * 0.25
        )
        self.assertAlmostEqual(portfolio_returns.iloc[0], expected_first_return, places=10)
        
        # Test with empty data
        empty_returns = calculate_portfolio_returns(pd.DataFrame(), self.weights)
        self.assertTrue(empty_returns.empty)
    
    def test_calculate_cumulative_returns(self):
        """Test cumulative returns calculation."""
        simple_returns = pd.Series([0.1, 0.05, -0.03, 0.02])
        cum_returns = calculate_cumulative_returns(simple_returns)
        
        # Manually calculate expected values
        expected = pd.Series([0.1, 0.155, 0.10035, 0.122357])
        pd.testing.assert_series_equal(cum_returns, expected, rtol=1e-5)
        
        # Test empty series
        empty_cum = calculate_cumulative_returns(pd.Series(dtype=float))
        self.assertTrue(empty_cum.empty)
    
    def test_calculate_annualized_return(self):
        """Test annualized return calculation."""
        # Create returns for exactly one year (252 trading days)
        annual_returns = pd.Series(np.random.normal(0.001, 0.02, 252))
        ann_return = calculate_annualized_return(annual_returns)
        
        # Should be a reasonable percentage
        self.assertIsInstance(ann_return, float)
        self.assertGreater(ann_return, -100)  # Should be greater than -100%
        self.assertLess(ann_return, 1000)     # Should be less than 1000%
        
        # Test empty series
        empty_return = calculate_annualized_return(pd.Series(dtype=float))
        self.assertEqual(empty_return, 0.0)
    
    def test_calculate_volatility(self):
        """Test volatility calculation."""
        # Test with known standard deviation
        returns = pd.Series([0.01, -0.01, 0.02, -0.015, 0.005])
        volatility = calculate_volatility(returns)
        
        # Should be positive
        self.assertGreater(volatility, 0)
        
        # Test empty series
        empty_vol = calculate_volatility(pd.Series(dtype=float))
        self.assertEqual(empty_vol, 0.0)
    
    def test_calculate_sharpe_ratio(self):
        """Test Sharpe ratio calculation."""
        # Create returns with positive expected return
        good_returns = pd.Series(np.random.normal(0.002, 0.01, 252))
        sharpe = calculate_sharpe_ratio(good_returns)
        
        self.assertIsInstance(sharpe, float)
        
        # Test with zero volatility (constant returns)
        constant_returns = pd.Series([0.001] * 100)
        zero_vol_sharpe = calculate_sharpe_ratio(constant_returns)
        self.assertEqual(zero_vol_sharpe, 0.0)
        
        # Test empty series
        empty_sharpe = calculate_sharpe_ratio(pd.Series(dtype=float))
        self.assertEqual(empty_sharpe, 0.0)
    
    def test_calculate_max_drawdown(self):
        """Test maximum drawdown calculation."""
        # Create returns that should have a known drawdown
        returns = pd.Series([0.1, 0.05, -0.2, -0.1, 0.15, 0.08])
        max_dd, dd_series = calculate_max_drawdown(returns)
        
        # Max drawdown should be negative
        self.assertLess(max_dd, 0)
        
        # Drawdown series should have same length as returns
        self.assertEqual(len(dd_series), len(returns))
        
        # Test empty series
        empty_dd, empty_series = calculate_max_drawdown(pd.Series(dtype=float))
        self.assertEqual(empty_dd, 0.0)
        self.assertTrue(empty_series.empty)
    
    def test_calculate_correlation_matrix(self):
        """Test correlation matrix calculation."""
        corr_matrix = calculate_correlation_matrix(self.returns_data)
        
        # Should be square matrix
        self.assertEqual(corr_matrix.shape[0], corr_matrix.shape[1])
        
        # Diagonal should be 1.0
        for i in range(len(corr_matrix)):
            self.assertAlmostEqual(corr_matrix.iloc[i, i], 1.0, places=5)
        
        # Should be symmetric
        pd.testing.assert_frame_equal(corr_matrix, corr_matrix.T, rtol=1e-10)
        
        # Test empty DataFrame
        empty_corr = calculate_correlation_matrix(pd.DataFrame())
        self.assertTrue(empty_corr.empty)


if __name__ == '__main__':
    unittest.main()
    