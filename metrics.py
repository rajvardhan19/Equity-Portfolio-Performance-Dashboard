"""
Portfolio metrics calculation module.
Handles return calculations, risk metrics, and portfolio analytics.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


def calculate_returns(price_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate daily returns from price data.
    
    Args:
        price_df: DataFrame with price data
        
    Returns:
        DataFrame with daily returns
    """
    if price_df.empty:
        return pd.DataFrame()
    
    return price_df.pct_change().dropna()


def calculate_portfolio_returns(
    returns_df: pd.DataFrame, 
    weights: Dict[str, float]
) -> pd.Series:
    """
    Calculate weighted portfolio returns.
    
    Args:
        returns_df: DataFrame with individual asset returns
        weights: Dictionary mapping tickers to weights
        
    Returns:
        Series with portfolio returns
    """
    if returns_df.empty or not weights:
        return pd.Series(dtype=float)
    
    # Ensure weights sum to 1
    total_weight = sum(weights.values())
    normalized_weights = {k: v/total_weight for k, v in weights.items()}
    
    # Filter returns for available tickers
    available_tickers = [ticker for ticker in normalized_weights.keys() 
                        if ticker in returns_df.columns]
    
    if not available_tickers:
        return pd.Series(dtype=float)
    
    portfolio_returns = pd.Series(0.0, index=returns_df.index)
    
    for ticker in available_tickers:
        portfolio_returns += returns_df[ticker] * normalized_weights[ticker]
    
    return portfolio_returns


def calculate_cumulative_returns(returns: pd.Series) -> pd.Series:
    """
    Calculate cumulative returns from daily returns.
    
    Args:
        returns: Series with daily returns
        
    Returns:
        Series with cumulative returns
    """
    if returns.empty:
        return pd.Series(dtype=float)
    
    return (1 + returns).cumprod() - 1


def calculate_annualized_return(returns: pd.Series) -> float:
    """
    Calculate annualized return from daily returns.
    
    Args:
        returns: Series with daily returns
        
    Returns:
        Annualized return as percentage
    """
    if returns.empty or len(returns) < 2:
        return 0.0
    
    total_return = (1 + returns).prod() - 1
    years = len(returns) / 252  # Assuming 252 trading days per year
    
    if years <= 0:
        return 0.0
    
    try:
        annualized_return = (1 + total_return) ** (1/years) - 1
        return annualized_return * 100
    except:
        return 0.0


def calculate_volatility(returns: pd.Series) -> float:
    """
    Calculate annualized volatility from daily returns.
    
    Args:
        returns: Series with daily returns
        
    Returns:
        Annualized volatility as percentage
    """
    if returns.empty:
        return 0.0
    
    daily_vol = returns.std()
    annualized_vol = daily_vol * np.sqrt(252)
    return annualized_vol * 100


def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
    """
    Calculate Sharpe ratio.
    
    Args:
        returns: Series with daily returns
        risk_free_rate: Risk-free rate (annual)
        
    Returns:
        Sharpe ratio
    """
    if returns.empty:
        return 0.0
    
    annualized_return = calculate_annualized_return(returns) / 100
    volatility = calculate_volatility(returns) / 100
    
    if volatility == 0:
        return 0.0
    
    return (annualized_return - risk_free_rate) / volatility


def calculate_max_drawdown(returns: pd.Series) -> Tuple[float, pd.Series]:
    """
    Calculate maximum drawdown and drawdown series.
    
    Args:
        returns: Series with daily returns
        
    Returns:
        Tuple of (max_drawdown_percentage, drawdown_series)
    """
    if returns.empty:
        return 0.0, pd.Series(dtype=float)
    
    cumulative = calculate_cumulative_returns(returns)
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / (1 + running_max)
    max_drawdown = drawdown.min()
    
    return max_drawdown * 100, drawdown * 100


def calculate_rolling_metrics(
    returns: pd.Series, 
    window: int = 60
) -> Dict[str, pd.Series]:
    """
    Calculate rolling metrics over specified window.
    
    Args:
        returns: Series with daily returns
        window: Rolling window size in days
        
    Returns:
        Dictionary with rolling metrics
    """
    if returns.empty or len(returns) < window:
        return {}
    
    rolling_metrics = {
        'volatility': returns.rolling(window).std() * np.sqrt(252) * 100,
        'return': returns.rolling(window).mean() * 252 * 100
    }
    
    return rolling_metrics


def calculate_correlation_matrix(returns_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate correlation matrix for portfolio assets.
    
    Args:
        returns_df: DataFrame with asset returns
        
    Returns:
        Correlation matrix DataFrame
    """
    if returns_df.empty:
        return pd.DataFrame()
    
    return returns_df.corr()


def calculate_var(returns: pd.Series, confidence_level: float = 0.05) -> float:
    """
    Calculate Value at Risk (VaR) at specified confidence level.
    
    Args:
        returns: Series with daily returns
        confidence_level: Confidence level (e.g., 0.05 for 95% VaR)
        
    Returns:
        VaR as percentage
    """
    if returns.empty:
        return 0.0
    
    return np.percentile(returns, confidence_level * 100) * 100


def calculate_portfolio_metrics(
    returns: pd.Series, 
    benchmark_returns: Optional[pd.Series] = None
) -> Dict[str, float]:
    """
    Calculate comprehensive portfolio metrics.
    
    Args:
        returns: Portfolio returns series
        benchmark_returns: Optional benchmark returns for comparison
        
    Returns:
        Dictionary with portfolio metrics
    """
    if returns.empty:
        return {}
    
    metrics = {
        'Total Return (%)': (calculate_cumulative_returns(returns).iloc[-1] * 100) if len(returns) > 0 else 0,
        'Annualized Return (%)': calculate_annualized_return(returns),
        'Volatility (%)': calculate_volatility(returns),
        'Sharpe Ratio': calculate_sharpe_ratio(returns),
        'Max Drawdown (%)': calculate_max_drawdown(returns)[0],
        'VaR 95% (%)': calculate_var(returns, 0.05),
        'VaR 99% (%)': calculate_var(returns, 0.01),
        'Best Day (%)': returns.max() * 100 if not returns.empty else 0,
        'Worst Day (%)': returns.min() * 100 if not returns.empty else 0,
    }
    
    # Add benchmark comparison metrics if available
    if benchmark_returns is not None and not benchmark_returns.empty:
        # Align dates
        aligned_returns, aligned_benchmark = returns.align(benchmark_returns, join='inner')
        
        if not aligned_returns.empty and not aligned_benchmark.empty:
            excess_returns = aligned_returns - aligned_benchmark
            tracking_error = excess_returns.std() * np.sqrt(252) * 100
            information_ratio = (excess_returns.mean() * 252 * 100) / tracking_error if tracking_error != 0 else 0
            
            # Beta calculation
            covariance = np.cov(aligned_returns, aligned_benchmark)[0][1]
            benchmark_variance = aligned_benchmark.var()
            beta = covariance / benchmark_variance if benchmark_variance != 0 else 1
            
            metrics.update({
                'Alpha (%)': calculate_annualized_return(excess_returns),
                'Beta': beta,
                'Tracking Error (%)': tracking_error,
                'Information Ratio': information_ratio,
                'Benchmark Return (%)': calculate_annualized_return(aligned_benchmark)
            })
    
    return metrics


def calculate_sector_allocation(
    weights: Dict[str, float], 
    stock_info: Dict[str, Dict]
) -> Dict[str, float]:
    """
    Calculate portfolio allocation by sector.
    
    Args:
        weights: Dictionary of stock weights
        stock_info: Dictionary of stock information including sectors
        
    Returns:
        Dictionary of sector allocations
    """
    sector_weights = {}
    
    for ticker, weight in weights.items():
        sector = stock_info.get(ticker, {}).get('sector', 'Unknown')
        sector_weights[sector] = sector_weights.get(sector, 0) + weight
    
    return sector_weights