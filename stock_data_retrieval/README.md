# Stock Market Analysis Agent 📊

This is the main stock market analysis agent that provides comprehensive investment insights and recommendations. Built with Google ADK and powered by advanced financial analysis algorithms, it delivers professional-grade stock analysis using free APIs.

## 🎯 Purpose

The Stock Market Analysis Agent provides complete investment analysis by combining real-time market data with AI-powered insights. It analyzes stocks from multiple perspectives - technical, fundamental, and risk assessment - to generate actionable investment recommendations.

## 🚀 Features

### Core Capabilities
- **🤖 AI-Powered Analysis**: Google ADK agent with LiteLlm for intelligent insights
- **📊 Real-time Data**: Yahoo Finance integration for live market data
- **📈 Technical Analysis**: RSI, Moving Averages, Beta calculations
- **💰 Financial Analysis**: P/E ratios, Market Cap, Financial health metrics
- **🎯 Investment Recommendations**: Automated buy/sell/hold signals with reasoning
- **🔍 Risk Assessment**: Volatility analysis and risk factor identification

### Analysis Types
- **Price Analysis**: Current price, daily/weekly/monthly/yearly changes
- **Technical Indicators**: RSI (14-day), Moving Averages (50, 200-day), Beta
- **Financial Metrics**: P/E ratio, Market Cap, Dividend Yield, Debt-to-Equity
- **Volume Analysis**: Current vs average volume, volume ratio analysis
- **Risk Metrics**: Volatility, Beta calculations, 52-week positioning
- **Investment Insights**: Market sentiment, investment signals, risk factors

## 📈 Supported Analysis Timeframes

| Period | Usage | Analysis Purpose |
|--------|-------|------------------|
| 1 Day | Intraday analysis | Real-time price movements |
| 1 Week | Short-term trends | Weekly performance tracking |
| 1 Month | Monthly analysis | Recent performance evaluation |
| 3 Months | Quarterly trends | Quarterly performance review |
| 6 Months | Semi-annual | Medium-term trend analysis |
| 1 Year | Annual performance | Long-term trend evaluation |
| 2 Years | Extended view | Historical performance |
| 5 Years | Long-term analysis | Long-term investment perspective |

## 🔧 Core Functions

### 1. `get_comprehensive_stock_analysis(symbol: str)`

**Main Analysis Function** - Provides complete investment analysis with AI-powered insights.

**Parameters:**
- `symbol` (str): Stock ticker symbol (e.g., 'AAPL', 'GOOGL', 'TSLA')

**Returns:**
Complete analysis dictionary with investment insights:

```python
{
    "symbol": "AAPL",
    "company_name": "Apple Inc.",
    "current_price": 150.25,
    "daily_change": "+2.50 (+1.69%)",
    "overall_sentiment": "BULLISH",
    "recommendation": "Consider buying - multiple positive indicators",
    "market_insights": [
        "Trading in middle range (65.2% of 52-week range)",
        "RSI at 58.3 shows neutral momentum",
        "Price above both 50-day and 200-day moving averages - bullish trend"
    ],
    "investment_signals": [
        "Strong upward trend confirmed by moving averages"
    ],
    "risk_factors": [],
    "pe_ratio": 25.4,
    "dividend_yield": "0.50%",
    "beta": 1.2,
    "rsi_14": 58.3,
    "target_price": 165.0
}
```

### 2. `stock_data(symbol: str)`

**Raw Data Function** - Retrieves comprehensive financial data without AI analysis.

**Parameters:**
- `symbol` (str): Stock ticker symbol

**Returns:**
Raw financial data dictionary with all metrics and calculations.

## 🛠️ Technical Implementation

### Dependencies
- **`google-adk`**: Agent framework for AI-powered analysis
- **`litellm`**: LLM integration via OpenRouter
- **`yfinance`**: Yahoo Finance API for stock data
- **`pandas`**: Data manipulation and analysis
- **`numpy`**: Numerical computations
- **`python-dotenv`**: Environment variable management

### Agent Configuration
```python
from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm

# Model configuration using free OpenRouter model
stock_data_retrieval_agent = LlmAgent(
    name="stock_data_retrieval_agent",
    model=LiteLlm(
        model="openrouter/moonshotai/kimi-k2:free",
        api_key=os.getenv("OPENROUTER_API_KEY")
    ),
    description="Agent for comprehensive stock market analysis and investment insights",
    tools=[get_comprehensive_stock_analysis],
    output_key="stock_analysis_results"
)
```

### Key Technical Features
- **Error Handling**: Robust error handling for invalid symbols and API failures
- **Data Validation**: Comprehensive data validation and type conversion
- **Beta Calculation**: Market correlation analysis using SPY as benchmark
- **RSI Calculation**: 14-day Relative Strength Index implementation
- **JSON Serialization**: Safe conversion of numpy types for JSON output

## 📊 Analysis Features

### Investment Intelligence
- **Sentiment Analysis**: BULLISH/BEARISH/NEUTRAL market sentiment
- **Investment Signals**: Automated buy/sell/hold recommendations
- **Risk Assessment**: Comprehensive risk factor identification
- **Market Insights**: Human-readable analysis explanations

### Price Analysis
- **Current Price**: Real-time market price with daily changes
- **Historical Performance**: 1-day to 5-year performance tracking
- **52-Week Analysis**: High/low ranges and current positioning
- **Price Changes**: Absolute and percentage changes across timeframes

### Technical Indicators
- **RSI (14-day)**: Relative Strength Index for momentum analysis
- **Moving Averages**: 50-day and 200-day simple moving averages
- **Beta Calculation**: Market correlation using SPY as benchmark
- **Volume Analysis**: Current vs average volume with ratio analysis

### Financial Health Metrics
- **Valuation Ratios**: P/E ratio, Price-to-Book, Forward P/E
- **Profitability**: ROE, ROA, Profit Margins, Operating Margins
- **Financial Strength**: Debt-to-Equity, Book Value
- **Income**: Dividend Yield, Earnings Per Share

## 🔄 Agent Integration

### Input Processing
- **User Queries**: Natural language stock analysis requests
- **Stock Symbols**: Ticker symbols for analysis (e.g., 'AAPL', 'GOOGL')
- **Analysis Requests**: Specific investment recommendation requests

### Output Generation
- **Comprehensive Analysis**: Complete investment insights with recommendations
- **JSON Responses**: Structured data for programmatic use
- **Investment Recommendations**: Clear buy/sell/hold signals with reasoning
- **Risk Assessment**: Detailed risk factors and market insights

### AI Agent Workflow
1. **Query Processing**: LiteLlm processes natural language requests
2. **Data Retrieval**: Yahoo Finance API fetches real-time stock data
3. **Analysis Engine**: Comprehensive financial and technical analysis
4. **Insight Generation**: AI-powered investment insights and recommendations
5. **Response Formatting**: Structured JSON output with human-readable insights

## 🚦 Error Handling

The agent includes robust error handling for:
- **Invalid Stock Symbols**: Returns error message with symbol validation
- **Market Data Unavailable**: Handles missing or incomplete data gracefully
- **API Failures**: Network connectivity and API rate limiting protection
- **Calculation Errors**: Safe handling of division by zero and missing data
- **Data Type Issues**: Automatic conversion of numpy types for JSON compatibility

## 🧪 Testing & Usage

### Quick Test - AI Agent
```python
from stock_data_retrieval.agent import stock_data_retrieval_agent

# Test with natural language query
response = stock_data_retrieval_agent.run("Analyze Apple stock and tell me if I should buy it")
print(response.output_key)
```

### Quick Test - Direct Function
```python
from stock_data_retrieval.agent import get_comprehensive_stock_analysis

# Test comprehensive analysis
result = get_comprehensive_stock_analysis("AAPL")
print(f"Current price: ${result['current_price']:.2f}")
print(f"Recommendation: {result['recommendation']}")
print(f"Sentiment: {result['overall_sentiment']}")
```

### Raw Data Function Test
```python
from stock_data_retrieval.agent import stock_data

# Test raw data retrieval
data = stock_data("AAPL")
print(f"Current price: ${data['current_price']:.2f}")
print(f"P/E ratio: {data['pe_ratio']}")
```

## 📝 Usage Examples

### AI-Powered Investment Analysis
```python
from stock_data_retrieval.agent import get_comprehensive_stock_analysis

# Get comprehensive investment analysis
analysis = get_comprehensive_stock_analysis("AAPL")

print(f"Company: {analysis['company_name']}")
print(f"Current Price: ${analysis['current_price']:.2f}")
print(f"Daily Change: {analysis['daily_change']}")
print(f"Overall Sentiment: {analysis['overall_sentiment']}")
print(f"Recommendation: {analysis['recommendation']}")

print("\nMarket Insights:")
for insight in analysis['market_insights']:
    print(f"• {insight}")

print("\nInvestment Signals:")
for signal in analysis['investment_signals']:
    print(f"• {signal}")

print("\nRisk Factors:")
for risk in analysis['risk_factors']:
    print(f"• {risk}")
```

### Portfolio Analysis
```python
# Analyze multiple stocks for portfolio
symbols = ["AAPL", "GOOGL", "MSFT", "TSLA"]
portfolio_analysis = {}

for symbol in symbols:
    analysis = get_comprehensive_stock_analysis(symbol)
    portfolio_analysis[symbol] = {
        'price': analysis['current_price'],
        'sentiment': analysis['overall_sentiment'],
        'recommendation': analysis['recommendation']
    }
    print(f"{symbol}: {analysis['overall_sentiment']} - {analysis['recommendation']}")
```

### Raw Data Access
```python
from stock_data_retrieval.agent import stock_data

# Get raw financial data
data = stock_data("AAPL")

print(f"Current Price: ${data['current_price']:.2f}")
print(f"P/E Ratio: {data['pe_ratio']}")
print(f"Market Cap: ${data['market_cap']:,}")
print(f"52-Week High: ${data['52_week_high']:.2f}")
print(f"RSI: {data['rsi_14_days']:.1f}")
```

## 🔧 Configuration

### Environment Setup
Create a `.env` file in the project root:
```bash
OPENROUTER_API_KEY=your_openrouter_api_key_here
```

Get your free OpenRouter API key from [OpenRouter](https://openrouter.ai/).

### Customization Options
- **Analysis Timeframes**: Modify period calculations in `stock_data()` function
- **Technical Indicators**: Adjust RSI window, moving average periods
- **Investment Logic**: Customize sentiment analysis rules in `get_comprehensive_stock_analysis()`
- **Risk Thresholds**: Modify beta, RSI, and P/E ratio thresholds for recommendations

## 🐛 Troubleshooting

### Common Issues
1. **"Invalid symbol provided"**: Check ticker symbol format (e.g., 'AAPL', not 'Apple')
2. **"No data available"**: Verify symbol exists and market is open
3. **API Key Error**: Ensure OPENROUTER_API_KEY is set in `.env` file
4. **Import Errors**: Run `pip install -r requirements.txt`

### Debug Tips
- Test with well-known symbols first (AAPL, GOOGL, MSFT)
- Check internet connection for Yahoo Finance API
- Verify `.env` file is in project root directory

## 📚 Additional Resources

- [Yahoo Finance API (yfinance)](https://python-yahoofinance.readthedocs.io/)
- [Google ADK Documentation](https://developers.google.com/agent-development-kit)
- [OpenRouter API](https://openrouter.ai/docs)
- [LiteLLM Documentation](https://docs.litellm.ai/)

---

## ⚠️ Disclaimer

> **Educational Purpose Only**
> 
> This agent provides educational stock analysis and should not be considered financial advice. All investment decisions should be based on your own research and risk tolerance. Past performance does not guarantee future results.

**Investment Risk Warning:**
- Stock investments carry inherent risks
- Market conditions can change rapidly
- Always diversify your portfolio
- Consider consulting with a qualified financial advisor