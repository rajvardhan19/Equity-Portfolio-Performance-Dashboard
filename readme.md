# Equity Portfolio Performance Dashboard

A comprehensive Streamlit web application for analyzing and visualizing equity portfolio performance with advanced risk metrics and interactive charts.

## 🚀 Features

### Portfolio Analysis
- **Multi-asset portfolio construction** with customizable weights
- **Historical performance analysis** with configurable date ranges
- **Benchmark comparison** against S&P 500 index
- **Real-time data fetching** from Yahoo Finance

### Performance Metrics
- Total and annualized returns
- Volatility and Sharpe ratio
- Maximum drawdown analysis
- Value at Risk (VaR) calculations
- Alpha and Beta vs benchmark
- Rolling metrics (60-day windows)

### Interactive Visualizations
- **Cumulative returns chart** - Portfolio vs benchmark comparison
- **Drawdown analysis** - Peak-to-trough visualizations
- **Correlation heatmap** - Asset correlation matrix
- **Allocation charts** - Asset and sector breakdowns
- **Rolling metrics** - Time-series risk analysis

### Advanced Features
- **Sector allocation analysis** with automatic categorization
- **Risk metrics dashboard** with comprehensive statistics
- **Data export functionality** for further analysis
- **Caching system** for improved performance
- **Error handling** for missing data and API failures

## 📋 Requirements

```
streamlit==1.28.1
pandas==2.1.1
numpy==1.24.3
yfinance==0.2.21
plotly==5.17.0
seaborn==0.12.2
matplotlib==3.7.2
scipy==1.11.3
requests==2.31.0
python-dateutil==2.8.2
```

## 🛠️ Installation

1. **Clone the repository:**
```bash
git clone <repository-url>
cd equity-portfolio-dashboard
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Run the application:**
```bash
streamlit run app.py
```

## 📁 Project Structure

```
equity-portfolio-dashboard/
├── app.py                 # Main Streamlit application
├── data_fetch.py          # Data fetching and API integration
├── metrics.py             # Portfolio calculations and analytics
├── visualize.py           # Chart creation and plotting functions
├── test_metrics.py        # Unit tests for metrics calculations
├── requirements.txt       # Python dependencies
└── README.md             # Project documentation
```

## 🎯 Usage Guide

### Getting Started

1. **Launch the application** and navigate to the sidebar
2. **Select date range** for your analysis period
3. **Choose stocks** from popular options or enter custom tickers
4. **Set portfolio weights** manually or use equal weighting
5. **Click "Load Portfolio Data"** to begin analysis

### Dashboard Tabs

#### 📊 Performance Tab
- Key performance metrics overview
- Detailed metrics table with all calculations
- Risk-adjusted returns analysis

#### 📈 Charts Tab
- Cumulative returns visualization
- Drawdown analysis charts
- Rolling metrics time series

#### 🔍 Risk Analysis Tab
- Asset correlation heatmap
- Risk metrics summary
- Value at Risk calculations

#### 🏢 Holdings Tab
- Portfolio allocation pie chart
- Holdings details table
- Sector allocation analysis

#### 📋 Data Export Tab
- Download portfolio performance data
- Export metrics and calculations
- CSV format for further analysis

## 🧪 Testing

Run the test suite to verify calculations:

```bash
python test_metrics.py
```

The test suite covers:
- Return calculations accuracy
- Portfolio weighting logic
- Risk metrics computations
- Edge cases and error handling

## 📊 Key Calculations

### Portfolio Returns
```python
portfolio_return = Σ(weight_i × return_i)
```

### Sharpe Ratio
```python
sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
```

### Maximum Drawdown
```python
drawdown = (trough_value - peak_value) / peak_value
max_drawdown = min(all_drawdowns)
```

### Value at Risk (VaR)
```python
VaR = percentile(returns, confidence_level)
```

## 🔧 Configuration

### Data Sources
- **Primary:** Yahoo Finance via yfinance library
- **Benchmark:** S&P 500 index (^GSPC)
- **Caching:** 1 hour for price data, 24 hours for stock info

### Performance Settings
- **Trading days per year:** 252
- **Default risk-free rate:** 2%
- **Rolling window:** 60 days
- **VaR confidence levels:** 95% and 99%

## 🚨 Error Handling

The application includes comprehensive error handling for:

- **Invalid ticker symbols** - Automatic validation and removal
- **Missing data points** - Forward filling with limits
- **API failures** - Graceful degradation and user feedback
- **Date range issues** - Input validation and warnings
- **Weight normalization** - Automatic adjustment to 100%

## 🎨 Customization

### Adding New Metrics
1. Implement calculation in `metrics.py`
2. Add visualization in `visualize.py`
3. Integrate into main dashboard in `app.py`
4. Add unit tests in `test_metrics.py`

### Custom Data Sources
Modify `data_fetch.py` to integrate additional data providers:
- Alpha Vantage API
- Quandl
- Custom CSV uploads

## 📈 Performance Optimization

- **Streamlit caching** for expensive API calls
- **Data preprocessing** to handle missing values
- **Efficient pandas operations** for calculations
- **Lazy loading** of stock information

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License. See LICENSE file for details.

## 📞 Support

For issues and questions:
- Create GitHub issues for bugs
- Check documentation for usage questions
- Review test files for implementation examples

## 🔮 Future Enhancements

- **Multi-currency support** for international portfolios
- **Options and derivatives** analysis
- **Monte Carlo simulations** for scenario analysis
- **Machine learning** for return predictions
- **Real-time streaming** data integration
- **Portfolio optimization** algorithms
- **ESG scoring** and sustainability metrics

---

**Built with ❤️ using Streamlit, Pandas, and Plotly**