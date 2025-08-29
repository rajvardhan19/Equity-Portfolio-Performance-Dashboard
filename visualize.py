"""
Visualization module for portfolio dashboard.
Handles all chart creation and plotting functions.
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import streamlit as st


def plot_cumulative_returns(
    portfolio_returns: pd.Series,
    benchmark_returns: Optional[pd.Series] = None,
    title: str = "Portfolio vs Benchmark Performance"
) -> go.Figure:
    """
    Create cumulative returns comparison chart.
    
    Args:
        portfolio_returns: Portfolio returns series
        benchmark_returns: Optional benchmark returns series
        title: Chart title
        
    Returns:
        Plotly figure object
    """
    fig = go.Figure()
    
    if not portfolio_returns.empty:
        # Calculate cumulative returns inline to avoid circular import
        portfolio_cum = (1 + portfolio_returns).cumprod() - 1
        portfolio_cum = portfolio_cum * 100
        
        fig.add_trace(go.Scatter(
            x=portfolio_cum.index,
            y=portfolio_cum.values,
            mode='lines',
            name='Portfolio',
            line=dict(color='#1f77b4', width=2),
            hovertemplate='<b>Portfolio</b><br>Date: %{x}<br>Return: %{y:.2f}%<extra></extra>'
        ))
    
    if benchmark_returns is not None and not benchmark_returns.empty:
        # Calculate benchmark cumulative returns inline
        benchmark_cum = (1 + benchmark_returns).cumprod() - 1
        benchmark_cum = benchmark_cum * 100
        
        fig.add_trace(go.Scatter(
            x=benchmark_cum.index,
            y=benchmark_cum.values,
            mode='lines',
            name='Benchmark (S&P 500)',
            line=dict(color='#ff7f0e', width=2),
            hovertemplate='<b>Benchmark</b><br>Date: %{x}<br>Return: %{y:.2f}%<extra></extra>'
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Cumulative Return (%)",
        hovermode='x unified',
        showlegend=True,
        height=500,
        template='plotly_white'
    )
    
    return fig


def plot_drawdown(
    returns: pd.Series,
    title: str = "Portfolio Drawdown"
) -> go.Figure:
    """
    Create drawdown chart.
    
    Args:
        returns: Portfolio returns series
        title: Chart title
        
    Returns:
        Plotly figure object
    """
    if returns.empty:
        fig = go.Figure()
        fig.add_annotation(text="No data available", 
                          xref="paper", yref="paper", 
                          x=0.5, y=0.5, showarrow=False)
        return fig
    
    # Calculate drawdown inline to avoid circular import
    if returns.empty:
        drawdown_series = pd.Series(dtype=float)
    else:
        cumulative = (1 + returns).cumprod() - 1
        running_max = cumulative.expanding().max()
        drawdown_series = (cumulative - running_max) / (1 + running_max) * 100
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=drawdown_series.index,
        y=drawdown_series.values,
        mode='lines',
        name='Drawdown',
        fill='tonexty',
        line=dict(color='red', width=1),
        fillcolor='rgba(255, 0, 0, 0.3)',
        hovertemplate='<b>Drawdown</b><br>Date: %{x}<br>Drawdown: %{y:.2f}%<extra></extra>'
    ))
    
    # Add zero line
    fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5)
    
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Drawdown (%)",
        hovermode='x unified',
        height=400,
        template='plotly_white'
    )
    
    return fig


def plot_allocation_pie(
    weights: Dict[str, float],
    title: str = "Portfolio Allocation"
) -> go.Figure:
    """
    Create portfolio allocation pie chart.
    
    Args:
        weights: Dictionary of asset weights
        title: Chart title
        
    Returns:
        Plotly figure object
    """
    if not weights:
        fig = go.Figure()
        fig.add_annotation(text="No allocation data", 
                          xref="paper", yref="paper", 
                          x=0.5, y=0.5, showarrow=False)
        return fig
    
    labels = list(weights.keys())
    values = [w * 100 for w in weights.values()]
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.3,
        hovertemplate='<b>%{label}</b><br>Weight: %{value:.1f}%<extra></extra>'
    )])
    
    fig.update_layout(
        title=title,
        height=400,
        template='plotly_white'
    )
    
    return fig


def plot_sector_allocation(
    sector_weights: Dict[str, float],
    title: str = "Sector Allocation"
) -> go.Figure:
    """
    Create sector allocation bar chart.
    
    Args:
        sector_weights: Dictionary of sector weights
        title: Chart title
        
    Returns:
        Plotly figure object
    """
    if not sector_weights:
        fig = go.Figure()
        fig.add_annotation(text="No sector data", 
                          xref="paper", yref="paper", 
                          x=0.5, y=0.5, showarrow=False)
        return fig
    
    sectors = list(sector_weights.keys())
    weights = [w * 100 for w in sector_weights.values()]
    
    fig = go.Figure(data=[go.Bar(
        x=sectors,
        y=weights,
        marker_color='lightblue',
        hovertemplate='<b>%{x}</b><br>Weight: %{y:.1f}%<extra></extra>'
    )])
    
    fig.update_layout(
        title=title,
        xaxis_title="Sector",
        yaxis_title="Weight (%)",
        height=400,
        template='plotly_white'
    )
    
    return fig


def plot_correlation_heatmap(
    correlation_matrix: pd.DataFrame,
    title: str = "Asset Correlation Matrix"
) -> go.Figure:
    """
    Create correlation heatmap.
    
    Args:
        correlation_matrix: Correlation matrix DataFrame
        title: Chart title
        
    Returns:
        Plotly figure object
    """
    if correlation_matrix.empty:
        fig = go.Figure()
        fig.add_annotation(text="No correlation data", 
                          xref="paper", yref="paper", 
                          x=0.5, y=0.5, showarrow=False)
        return fig
    
    fig = go.Figure(data=go.Heatmap(
        z=correlation_matrix.values,
        x=correlation_matrix.columns,
        y=correlation_matrix.index,
        colorscale='RdBu',
        zmid=0,
        text=np.round(correlation_matrix.values, 2),
        texttemplate='%{text}',
        textfont={"size": 10},
        hovertemplate='<b>%{y} vs %{x}</b><br>Correlation: %{z:.3f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=title,
        height=500,
        template='plotly_white'
    )
    
    return fig


def plot_rolling_metrics(
    rolling_metrics: Dict[str, pd.Series],
    title: str = "Rolling Metrics"
) -> go.Figure:
    """
    Create rolling metrics chart.
    
    Args:
        rolling_metrics: Dictionary of rolling metric series
        title: Chart title
        
    Returns:
        Plotly figure object
    """
    if not rolling_metrics:
        fig = go.Figure()
        fig.add_annotation(text="No rolling metrics data", 
                          xref="paper", yref="paper", 
                          x=0.5, y=0.5, showarrow=False)
        return fig
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Rolling Return (60-day)", "Rolling Volatility (60-day)"),
        vertical_spacing=0.1
    )
    
    if 'return' in rolling_metrics:
        returns_data = rolling_metrics['return'].dropna()
        fig.add_trace(
            go.Scatter(
                x=returns_data.index,
                y=returns_data.values,
                mode='lines',
                name='Rolling Return',
                line=dict(color='blue'),
                hovertemplate='<b>Rolling Return</b><br>Date: %{x}<br>Return: %{y:.2f}%<extra></extra>'
            ),
            row=1, col=1
        )
    
    if 'volatility' in rolling_metrics:
        vol_data = rolling_metrics['volatility'].dropna()
        fig.add_trace(
            go.Scatter(
                x=vol_data.index,
                y=vol_data.values,
                mode='lines',
                name='Rolling Volatility',
                line=dict(color='red'),
                hovertemplate='<b>Rolling Volatility</b><br>Date: %{x}<br>Volatility: %{y:.2f}%<extra></extra>'
            ),
            row=2, col=1
        )
    
    fig.update_layout(
        title=title,
        height=600,
        template='plotly_white',
        showlegend=False
    )
    
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Return (%)", row=1, col=1)
    fig.update_yaxes(title_text="Volatility (%)", row=2, col=1)
    
    return fig


def create_metrics_table(metrics: Dict[str, float]) -> pd.DataFrame:
    """
    Create a formatted metrics table for display.
    
    Args:
        metrics: Dictionary of metric values
        
    Returns:
        Formatted DataFrame for display
    """
    if not metrics:
        return pd.DataFrame()
    
    df = pd.DataFrame(list(metrics.items()), columns=['Metric', 'Value'])
    
    # Format values based on metric type
    for idx, row in df.iterrows():
        metric = row['Metric']
        value = row['Value']
        
        if pd.isna(value):
            df.at[idx, 'Value'] = 'N/A'
        elif 'Ratio' in metric or metric == 'Beta':
            df.at[idx, 'Value'] = f"{value:.3f}"
        elif '%' in metric:
            df.at[idx, 'Value'] = f"{value:.2f}%"
        else:
            df.at[idx, 'Value'] = f"{value:.2f}"
    
    return df


def plot_performance_comparison(
    portfolio_metrics: Dict[str, float],
    benchmark_metrics: Dict[str, float],
    title: str = "Portfolio vs Benchmark Metrics"
) -> go.Figure:
    """
    Create comparison bar chart of key metrics.
    
    Args:
        portfolio_metrics: Portfolio metrics dictionary
        benchmark_metrics: Benchmark metrics dictionary
        title: Chart title
        
    Returns:
        Plotly figure object
    """
    # Select key metrics for comparison
    key_metrics = [
        'Annualized Return (%)',
        'Volatility (%)',
        'Sharpe Ratio',
        'Max Drawdown (%)'
    ]
    
    portfolio_values = [portfolio_metrics.get(metric, 0) for metric in key_metrics]
    benchmark_values = [benchmark_metrics.get(metric, 0) for metric in key_metrics]
    
    fig = go.Figure(data=[
        go.Bar(name='Portfolio', x=key_metrics, y=portfolio_values, marker_color='lightblue'),
        go.Bar(name='Benchmark', x=key_metrics, y=benchmark_values, marker_color='lightcoral')
    ])
    
    fig.update_layout(
        title=title,
        barmode='group',
        height=400,
        template='plotly_white'
    )
    
    return fig