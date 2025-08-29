"""
Data fetching module for equity portfolio dashboard.
Handles Yahoo Finance API integration and data caching.
"""

import yfinance as yf
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import logging


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_stock_data(
    tickers: List[str], 
    start_date: datetime, 
    end_date: datetime
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Fetch historical stock price data from Yahoo Finance.
    
    Args:
        tickers: List of stock ticker symbols
        start_date: Start date for historical data
        end_date: End date for historical data
        
    Returns:
        Tuple of (price_data_df, failed_tickers)
    """
    if not tickers:
        return pd.DataFrame(), []
    
    failed_tickers = []
    successful_data = {}
    
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(start=start_date, end=end_date)
            
            if hist.empty:
                failed_tickers.append(ticker)
                logger.warning(f"No data found for ticker: {ticker}")
            else:
                successful_data[ticker] = hist['Close']
                
        except Exception as e:
            failed_tickers.append(ticker)
            logger.error(f"Error fetching data for {ticker}: {str(e)}")
    
    if successful_data:
        price_df = pd.DataFrame(successful_data)
        price_df = price_df.dropna()  # Remove rows with any NaN values
    else:
        price_df = pd.DataFrame()
    
    return price_df, failed_tickers


@st.cache_data(ttl=3600)
def fetch_benchmark_data(
    benchmark_ticker: str = "^GSPC",  # S&P 500
    start_date: datetime = None,
    end_date: datetime = None
) -> pd.DataFrame:
    """
    Fetch benchmark index data.
    
    Args:
        benchmark_ticker: Ticker symbol for benchmark (default: S&P 500)
        start_date: Start date for data
        end_date: End date for data
        
    Returns:
        DataFrame with benchmark price data
    """
    try:
        benchmark = yf.Ticker(benchmark_ticker)
        hist = benchmark.history(start=start_date, end=end_date)
        
        if hist.empty:
            logger.warning(f"No benchmark data found for {benchmark_ticker}")
            return pd.DataFrame()
            
        return hist[['Close']].rename(columns={'Close': 'Benchmark'})
        
    except Exception as e:
        logger.error(f"Error fetching benchmark data: {str(e)}")
        return pd.DataFrame()


@st.cache_data(ttl=86400)  # Cache for 24 hours
def get_stock_info(ticker: str) -> Dict:
    """
    Get basic stock information including sector and industry.
    
    Args:
        ticker: Stock ticker symbol
        
    Returns:
        Dictionary with stock information
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        return {
            'sector': info.get('sector', 'Unknown'),
            'industry': info.get('industry', 'Unknown'),
            'market_cap': info.get('marketCap', 0),
            'name': info.get('longName', ticker)
        }
    except Exception as e:
        logger.error(f"Error fetching info for {ticker}: {str(e)}")
        return {
            'sector': 'Unknown',
            'industry': 'Unknown',
            'market_cap': 0,
            'name': ticker
        }


def validate_tickers(tickers: List[str]) -> List[str]:
    """
    Validate ticker symbols by checking if they exist.
    
    Args:
        tickers: List of ticker symbols to validate
        
    Returns:
        List of valid ticker symbols
    """
    valid_tickers = []
    
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker.upper())
            # Try to get basic info to validate ticker exists
            info = stock.info
            if info and 'regularMarketPrice' in info:
                valid_tickers.append(ticker.upper())
        except:
            continue
    
    return valid_tickers


def get_popular_stocks() -> List[str]:
    """
    Return a list of popular stock tickers for easy selection.
    
    Returns:
        List of popular stock ticker symbols
    """
    return [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA',
        'META', 'NVDA', 'JPM', 'JNJ', 'V',
        'PG', 'UNH', 'HD', 'MA', 'DIS',
        'ADBE', 'NFLX', 'CRM', 'BAC', 'WMT'
    ]


def preprocess_price_data(price_df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess price data by handling missing values and ensuring proper datetime index.
    
    Args:
        price_df: Raw price data DataFrame
        
    Returns:
        Preprocessed price DataFrame
    """
    if price_df.empty:
        return price_df
    
    # Ensure datetime index
    if not isinstance(price_df.index, pd.DatetimeIndex):
        price_df.index = pd.to_datetime(price_df.index)
    
    # Forward fill missing values (max 5 consecutive days)
    price_df = price_df.ffill(limit=5)
    
    # Drop any remaining rows with NaN values
    price_df = price_df.dropna()
    
    # Sort by date
    price_df = price_df.sort_index()
    
    return price_df