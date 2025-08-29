"""
Main Streamlit application for Equity Portfolio Performance Dashboard.
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List
import io

# Import custom modules
from data_fetch import (
    fetch_stock_data,
    fetch_benchmark_data,
    get_stock_info,
    get_popular_stocks,
    validate_tickers
)
from metrics import (
    calculate_returns,
    calculate_portfolio_returns,
    calculate_portfolio_metrics,
    calculate_rolling_metrics,
    calculate_correlation_matrix,
    calculate_sector_allocation,
    calculate_cumulative_returns
)
from visualize import (
    plot_cumulative_returns,
    plot_drawdown,
    plot_allocation_pie,
    plot_sector_allocation,
    plot_correlation_heatmap,
    plot_rolling_metrics,
    create_metrics_table,
    plot_performance_comparison
)


def main():
    """Main application function."""
    st.set_page_config(
        page_title="Equity Portfolio Performance Dashboard",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("📈 Equity Portfolio Performance Dashboard")
    st.markdown("---")
    
    # Initialize session state
    if 'portfolio_data' not in st.session_state:
        st.session_state.portfolio_data = None
    if 'benchmark_data' not in st.session_state:
        st.session_state.benchmark_data = None
    
    # Sidebar for inputs
    with st.sidebar:
        st.header("Portfolio Configuration")
        
        # Date range selection
        st.subheader("📅 Date Range")
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "Start Date",
                value=datetime.now() - timedelta(days=365*2),
                max_value=datetime.now() - timedelta(days=1)
            )
        with col2:
            end_date = st.date_input(
                "End Date",
                value=datetime.now() - timedelta(days=1),
                max_value=datetime.now() - timedelta(days=1)
            )
        
        if start_date >= end_date:
            st.error("Start date must be before end date!")
            return
        
        # Stock selection
        st.subheader("🏢 Stock Selection")
        
        # Popular stocks for quick selection
        popular_stocks = get_popular_stocks()
        
        # Multi-select for tickers
        selected_tickers = st.multiselect(
            "Select stocks (or type custom tickers):",
            options=popular_stocks,
            default=['AAPL', 'GOOGL', 'MSFT', 'AMZN'],
            help="Select from popular stocks or type custom ticker symbols"
        )
        
        # Custom ticker input
        custom_tickers = st.text_input(
            "Add custom tickers (comma-separated):",
            placeholder="e.g., TSLA, NVDA, META",
            help="Enter additional ticker symbols separated by commas"
        )
        
        # Combine selected and custom tickers
        all_tickers = list(selected_tickers)
        if custom_tickers:
            custom_list = [ticker.strip().upper() for ticker in custom_tickers.split(',')]
            all_tickers.extend([ticker for ticker in custom_list if ticker not in all_tickers])
        
        if not all_tickers:
            st.warning("Please select at least one stock!")
            return
        
        # Validate tickers
        with st.spinner("Validating tickers..."):
            valid_tickers = validate_tickers(all_tickers)
        
        invalid_tickers = [t for t in all_tickers if t not in valid_tickers]
        if invalid_tickers:
            st.warning(f"Invalid tickers removed: {', '.join(invalid_tickers)}")
        
        if not valid_tickers:
            st.error("No valid tickers found!")
            return
        
        # Portfolio weights
        st.subheader("⚖️ Portfolio Weights")
        weights = {}
        
        # Auto-equal weights option
        equal_weights = st.checkbox(
            "Equal weights",
            value=True,
            help="Automatically assign equal weights to all stocks"
        )
        
        if equal_weights:
            weight_per_stock = 100.0 / len(valid_tickers)
            for ticker in valid_tickers:
                weights[ticker] = weight_per_stock / 100.0
        else:
            total_weight = 0
            for ticker in valid_tickers:
                weight = st.slider(
                    f"{ticker} weight (%)",
                    min_value=0.0,
                    max_value=100.0,
                    value=100.0/len(valid_tickers),
                    step=0.5
                )
                weights[ticker] = weight / 100.0
                total_weight += weight
            
            # Check if weights sum to 100%
            if abs(total_weight - 100.0) > 0.1:
                st.warning(f"Total weight: {total_weight:.1f}% (should be 100%)")
        
        # Load data button
        load_data = st.button("📊 Load Portfolio Data", type="primary")
    
    # Main content area
    if load_data or st.session_state.portfolio_data is not None:
        
        if load_data:
            # Fetch data
            with st.spinner("Fetching stock data..."):
                price_data, failed_tickers = fetch_stock_data(
                    valid_tickers, start_date, end_date
                )
                
                if failed_tickers:
                    st.warning(f"Failed to fetch data for: {', '.join(failed_tickers)}")
                
                if price_data.empty:
                    st.error("No data could be fetched for the selected stocks and date range!")
                    return
                
                # Remove failed tickers from weights
                weights = {k: v for k, v in weights.items() if k in price_data.columns}
                
                # Fetch benchmark data
                benchmark_data = fetch_benchmark_data("^GSPC", start_date, end_date)
                
                # Store in session state
                st.session_state.portfolio_data = {
                    'prices': price_data,
                    'weights': weights,
                    'tickers': list(weights.keys())
                }
                st.session_state.benchmark_data = benchmark_data
        
        # Use data from session state
        portfolio_data = st.session_state.portfolio_data
        benchmark_data = st.session_state.benchmark_data
        
        if portfolio_data is None:
            return
        
        price_data = portfolio_data['prices']
        weights = portfolio_data['weights']
        tickers = portfolio_data['tickers']
        
        # Calculate returns with error handling
        try:
            returns_data = calculate_returns(price_data)
            portfolio_returns = calculate_portfolio_returns(returns_data, weights)
            
            if portfolio_returns.empty:
                st.error("Unable to calculate portfolio returns. Please check your data.")
                return
                
        except Exception as e:
            st.error(f"Error calculating portfolio returns: {str(e)}")
            return
        
        if benchmark_data is not None and not benchmark_data.empty:
            benchmark_returns = calculate_returns(benchmark_data)['Benchmark']
        else:
            benchmark_returns = pd.Series(dtype=float)
        
        # Create tabs for different views
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Performance", "📈 Charts", "🔍 Risk Analysis", 
            "🏢 Holdings", "📋 Data Export"
        ])
        
        with tab1:
            st.header("Portfolio Performance Overview")
            
            # Calculate portfolio metrics
            portfolio_metrics = calculate_portfolio_metrics(
                portfolio_returns, benchmark_returns
            )
            
            # Display key metrics in columns
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Total Return",
                    f"{portfolio_metrics.get('Total Return (%)', 0):.2f}%"
                )
                st.metric(
                    "Annualized Return",
                    f"{portfolio_metrics.get('Annualized Return (%)', 0):.2f}%"
                )
            
            with col2:
                st.metric(
                    "Volatility",
                    f"{portfolio_metrics.get('Volatility (%)', 0):.2f}%"
                )
                st.metric(
                    "Sharpe Ratio",
                    f"{portfolio_metrics.get('Sharpe Ratio', 0):.3f}"
                )
            
            with col3:
                st.metric(
                    "Max Drawdown",
                    f"{portfolio_metrics.get('Max Drawdown (%)', 0):.2f}%"
                )
                st.metric(
                    "VaR (95%)",
                    f"{portfolio_metrics.get('VaR 95% (%)', 0):.2f}%"
                )
            
            with col4:
                if 'Beta' in portfolio_metrics:
                    st.metric("Beta", f"{portfolio_metrics['Beta']:.3f}")
                if 'Alpha (%)' in portfolio_metrics:
                    st.metric("Alpha", f"{portfolio_metrics['Alpha (%)']:.2f}%")
            
            # Detailed metrics table
            st.subheader("Detailed Metrics")
            metrics_df = create_metrics_table(portfolio_metrics)
            if not metrics_df.empty:
                st.dataframe(metrics_df, use_container_width=True)
        
        with tab2:
            st.header("Performance Charts")
            
            # Cumulative returns chart
            st.subheader("Cumulative Returns")
            cum_returns_fig = plot_cumulative_returns(
                portfolio_returns, benchmark_returns
            )
            st.plotly_chart(cum_returns_fig, use_container_width=True)
            
            # Drawdown chart
            st.subheader("Drawdown Analysis")
            drawdown_fig = plot_drawdown(portfolio_returns)
            st.plotly_chart(drawdown_fig, use_container_width=True)
            
            # Rolling metrics
            st.subheader("Rolling Metrics (60-day)")
            rolling_metrics = calculate_rolling_metrics(portfolio_returns, 60)
            if rolling_metrics:
                rolling_fig = plot_rolling_metrics(rolling_metrics)
                st.plotly_chart(rolling_fig, use_container_width=True)
        
        with tab3:
            st.header("Risk Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Correlation heatmap
                st.subheader("Asset Correlations")
                correlation_matrix = calculate_correlation_matrix(returns_data)
                if not correlation_matrix.empty:
                    corr_fig = plot_correlation_heatmap(correlation_matrix)
                    st.plotly_chart(corr_fig, use_container_width=True)
            
            with col2:
                # Risk metrics summary
                st.subheader("Risk Metrics")
                risk_metrics = {
                    'Portfolio Volatility': f"{portfolio_metrics.get('Volatility (%)', 0):.2f}%",
                    'Max Drawdown': f"{portfolio_metrics.get('Max Drawdown (%)', 0):.2f}%",
                    'VaR (95%)': f"{portfolio_metrics.get('VaR 95% (%)', 0):.2f}%",
                    'VaR (99%)': f"{portfolio_metrics.get('VaR 99% (%)', 0):.2f}%",
                    'Best Day': f"{portfolio_metrics.get('Best Day (%)', 0):.2f}%",
                    'Worst Day': f"{portfolio_metrics.get('Worst Day (%)', 0):.2f}%"
                }
                
                for metric, value in risk_metrics.items():
                    st.metric(metric, value)
        
        with tab4:
            st.header("Portfolio Holdings")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Asset allocation pie chart
                st.subheader("Asset Allocation")
                allocation_fig = plot_allocation_pie(weights)
                st.plotly_chart(allocation_fig, use_container_width=True)
            
            with col2:
                # Holdings table
                st.subheader("Holdings Details")
                holdings_data = []
                
                with st.spinner("Loading stock information..."):
                    for ticker, weight in weights.items():
                        try:
                            stock_info = get_stock_info(ticker)
                            holdings_data.append({
                                'Ticker': ticker,
                                'Weight (%)': f"{weight * 100:.2f}%",
                                'Sector': stock_info.get('sector', 'Unknown'),
                                'Name': stock_info.get('name', ticker)
                            })
                        except Exception as e:
                            # Fallback for failed stock info
                            holdings_data.append({
                                'Ticker': ticker,
                                'Weight (%)': f"{weight * 100:.2f}%",
                                'Sector': 'Unknown',
                                'Name': ticker
                            })
                
                if holdings_data:
                    holdings_df = pd.DataFrame(holdings_data)
                    st.dataframe(holdings_df, use_container_width=True)
                else:
                    st.warning("No holdings data available.")
            
            # Sector allocation
            st.subheader("Sector Allocation")
            stock_info_dict = {
                ticker: get_stock_info(ticker) for ticker in tickers
            }
            sector_weights = calculate_sector_allocation(weights, stock_info_dict)
            sector_fig = plot_sector_allocation(sector_weights)
            st.plotly_chart(sector_fig, use_container_width=True)
        
        with tab5:
            st.header("Data Export")
            
            # Prepare data for export
            export_data = pd.DataFrame({
                'Date': portfolio_returns.index,
                'Portfolio_Returns': portfolio_returns.values,
                'Portfolio_Cumulative': (1 + portfolio_returns).cumprod().values - 1
            })
            
            if not benchmark_returns.empty:
                # Align benchmark data
                aligned_benchmark = benchmark_returns.reindex(portfolio_returns.index)
                export_data['Benchmark_Returns'] = aligned_benchmark.values
                export_data['Benchmark_Cumulative'] = (1 + aligned_benchmark).cumprod().values - 1
            
            # Add individual asset returns
            for ticker in tickers:
                if ticker in returns_data.columns:
                    aligned_returns = returns_data[ticker].reindex(portfolio_returns.index)
                    export_data[f'{ticker}_Returns'] = aligned_returns.values
            
            st.subheader("Portfolio Returns Data")
            st.dataframe(export_data.head(), use_container_width=True)
            
            # Download button
            csv_buffer = io.StringIO()
            export_data.to_csv(csv_buffer, index=False)
            
            st.download_button(
                label="📥 Download Portfolio Data (CSV)",
                data=csv_buffer.getvalue(),
                file_name=f"portfolio_data_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                help="Download complete portfolio performance data"
            )
            
            # Portfolio summary for download
            summary_data = pd.DataFrame([portfolio_metrics]).T
            summary_data.columns = ['Value']
            
            summary_csv = io.StringIO()
            summary_data.to_csv(summary_csv)
            
            st.download_button(
                label="📊 Download Portfolio Metrics (CSV)",
                data=summary_csv.getvalue(),
                file_name=f"portfolio_metrics_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                help="Download portfolio performance metrics"
            )
    
    else:
        # Welcome message and instructions
        st.info("""
        👋 Welcome to the Equity Portfolio Performance Dashboard!
        
        **Getting Started:**
        1. Select your desired date range in the sidebar
        2. Choose stocks from popular options or add custom tickers
        3. Set portfolio weights (or use equal weights)
        4. Click "Load Portfolio Data" to analyze your portfolio
        
        **Features:**
        - 📊 Comprehensive performance metrics
        - 📈 Interactive charts and visualizations  
        - 🔍 Risk analysis and correlations
        - 🏢 Holdings and sector allocation
        - 📋 Data export capabilities
        """)
        
        # Sample portfolio suggestion
        with st.expander("💡 Try a Sample Portfolio"):
            st.write("""
            **Suggested Sample Portfolio:**
            - **AAPL** (Apple Inc.) - 25%
            - **GOOGL** (Alphabet Inc.) - 25%
            - **MSFT** (Microsoft Corp.) - 25%
            - **AMZN** (Amazon.com Inc.) - 25%
            
            **Date Range:** Last 2 years
            
            This diversified tech portfolio will demonstrate all dashboard features.
            """)


if __name__ == "__main__":
    main()