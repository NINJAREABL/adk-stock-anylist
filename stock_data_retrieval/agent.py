"""
This Agent is responsible for retrieving stock market data.
"""

from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import os
from dotenv import load_dotenv
from typing import Optional, List

# Load environment variables from .env file
load_dotenv()

# Get API key from environment variable (optional at import time)
openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
if openrouter_api_key:
    # Set the environment variable for the library to use
    os.environ["OPENROUTER_API_KEY"] = openrouter_api_key


def safe_convert(value):
    """
    Safely convert numpy types to Python native types for JSON serialization.
    """
    if value is None:
        return None
    if isinstance(value, (np.integer, np.signedinteger)):
        return int(value)
    if isinstance(value, (np.floating, np.complexfloating)):
        return float(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def stock_data(symbol: str):
    """
    Comprehensive stock data retrieval function for financial experts.
    
    Args:
        symbol (str): Stock symbol (e.g., 'AAPL', 'GOOGL', 'TSLA')
    
    Returns:
        dict: Comprehensive financial data including price metrics, technical indicators, and analysis
    """
    try:
        # Validate input
        if not symbol or not isinstance(symbol, str):
            return {
                "error": "Invalid symbol provided. Please provide a valid stock symbol.",
                "symbol": symbol,
                "data_retrieved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        
        # Clean and uppercase the symbol
        symbol = symbol.strip().upper()
        
        # Create ticker object
        ticker = yf.Ticker(symbol)
        
        # Get current date
        today = datetime.now()
        
        # Define time periods
        periods = {
            '1d': today - timedelta(days=1),
            '1w': today - timedelta(weeks=1),
            '1m': today - timedelta(days=30),
            '3m': today - timedelta(days=90),
            '6m': today - timedelta(days=180),
            '1y': today - timedelta(days=365),
            '2y': today - timedelta(days=730),
            '5y': today - timedelta(days=1825)
        }
        
        # Get historical data for different periods
        hist_1d = ticker.history(period="1d", interval="1m")
        hist_1w = ticker.history(period="5d")
        hist_1m = ticker.history(period="1mo")
        hist_3m = ticker.history(period="3mo")
        hist_6m = ticker.history(period="6mo")
        hist_1y = ticker.history(period="1y")
        hist_2y = ticker.history(period="2y")
        hist_5y = ticker.history(period="5y")
        
        # Check if we got any data
        if hist_1d.empty and hist_1w.empty and hist_1m.empty:
            return {
                "error": f"No data available for symbol {symbol}. Please check the symbol and try again.",
                "symbol": symbol,
                "data_retrieved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        
        # Get company info
        info = ticker.info
        
        # Current price data
        current_price = hist_1d['Close'].iloc[-1] if not hist_1d.empty else None
        open_price = hist_1d['Open'].iloc[0] if not hist_1d.empty else None
        
        # Calculate means for different periods
        mean_1m = hist_1m['Close'].mean() if not hist_1m.empty else None
        mean_3m = hist_3m['Close'].mean() if not hist_3m.empty else None
        mean_6m = hist_6m['Close'].mean() if not hist_6m.empty else None
        mean_1y = hist_1y['Close'].mean() if not hist_1y.empty else None
        mean_2y = hist_2y['Close'].mean() if not hist_2y.empty else None
        
        # Calculate volatility (standard deviation)
        volatility_1m = hist_1m['Close'].std() if not hist_1m.empty else None
        volatility_6m = hist_6m['Close'].std() if not hist_6m.empty else None
        volatility_1y = hist_1y['Close'].std() if not hist_1y.empty else None
        
        # Calculate price changes and percentages
        price_change_1d = (current_price - hist_1d['Close'].iloc[0]) if len(hist_1d) > 1 else 0
        price_change_1w = (current_price - hist_1w['Close'].iloc[0]) if len(hist_1w) > 1 else 0
        price_change_1m = (current_price - hist_1m['Close'].iloc[0]) if len(hist_1m) > 1 else 0
        price_change_1y = (current_price - hist_1y['Close'].iloc[0]) if len(hist_1y) > 1 else 0
        
        # Calculate percentage changes
        pct_change_1d = (price_change_1d / hist_1d['Close'].iloc[0] * 100) if len(hist_1d) > 1 and hist_1d['Close'].iloc[0] != 0 else 0
        pct_change_1w = (price_change_1w / hist_1w['Close'].iloc[0] * 100) if len(hist_1w) > 1 and hist_1w['Close'].iloc[0] != 0 else 0
        pct_change_1m = (price_change_1m / hist_1m['Close'].iloc[0] * 100) if len(hist_1m) > 1 and hist_1m['Close'].iloc[0] != 0 else 0
        pct_change_1y = (price_change_1y / hist_1y['Close'].iloc[0] * 100) if len(hist_1y) > 1 and hist_1y['Close'].iloc[0] != 0 else 0
        
        # Calculate 52-week high and low
        week_52_high = hist_1y['High'].max() if not hist_1y.empty else None
        week_52_low = hist_1y['Low'].min() if not hist_1y.empty else None
        
        # Calculate moving averages
        ma_50 = hist_1y['Close'].tail(50).mean() if len(hist_1y) >= 50 else None
        ma_200 = hist_1y['Close'].tail(200).mean() if len(hist_1y) >= 200 else None
        
        # Volume analysis
        avg_volume_1m = hist_1m['Volume'].mean() if not hist_1m.empty else None
        current_volume = hist_1d['Volume'].iloc[-1] if not hist_1d.empty else None
        
        # RSI calculation (14-day)
        def calculate_rsi(prices, window=14):
            if len(prices) < window:
                return None
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi.iloc[-1]
        
        rsi = calculate_rsi(hist_1m['Close']) if not hist_1m.empty else None
        
        # Beta calculation (correlation with market - using SPY as proxy)
        try:
            spy = yf.Ticker("SPY")
            spy_hist = spy.history(period="1y")
            if not hist_1y.empty and not spy_hist.empty:
                # Align dates
                common_dates = hist_1y.index.intersection(spy_hist.index)
                if len(common_dates) > 10:
                    stock_returns = hist_1y.loc[common_dates]['Close'].pct_change().dropna()
                    market_returns = spy_hist.loc[common_dates]['Close'].pct_change().dropna()
                    beta = np.cov(stock_returns, market_returns)[0][1] / np.var(market_returns)
                else:
                    beta = None
            else:
                beta = None
        except Exception:
            beta = None
        
        # Comprehensive stock data dictionary
        stock_analysis = {
            # Basic Information
            "symbol": symbol.upper(),
            "company_name": info.get('longName', 'N/A'),
            "sector": info.get('sector', 'N/A'),
            "industry": info.get('industry', 'N/A'),
            "market_cap": safe_convert(info.get('marketCap', 'N/A')),
            
            # Current Price Data
            "todays_close_price": safe_convert(round(current_price, 2)) if current_price else None,
            "open_price": safe_convert(round(open_price, 2)) if open_price else None,
            "current_price": safe_convert(round(current_price, 2)) if current_price else None,
            
            # Historical Averages
            "mean_price_1_month": safe_convert(round(mean_1m, 2)) if mean_1m else None,
            "mean_price_3_months": safe_convert(round(mean_3m, 2)) if mean_3m else None,
            "mean_price_6_months": safe_convert(round(mean_6m, 2)) if mean_6m else None,
            "mean_price_1_year": safe_convert(round(mean_1y, 2)) if mean_1y else None,
            "mean_price_2_years": safe_convert(round(mean_2y, 2)) if mean_2y else None,
            
            # Price Changes
            "price_change_1_day": safe_convert(round(price_change_1d, 2)) if price_change_1d else None,
            "price_change_1_week": safe_convert(round(price_change_1w, 2)) if price_change_1w else None,
            "price_change_1_month": safe_convert(round(price_change_1m, 2)) if price_change_1m else None,
            "price_change_1_year": safe_convert(round(price_change_1y, 2)) if price_change_1y else None,
            
            # Percentage Changes
            "percentage_change_1_day": safe_convert(round(pct_change_1d, 2)) if pct_change_1d else None,
            "percentage_change_1_week": safe_convert(round(pct_change_1w, 2)) if pct_change_1w else None,
            "percentage_change_1_month": safe_convert(round(pct_change_1m, 2)) if pct_change_1m else None,
            "percentage_change_1_year": safe_convert(round(pct_change_1y, 2)) if pct_change_1y else None,
            
            # 52-Week Range
            "52_week_high": safe_convert(round(week_52_high, 2)) if week_52_high else None,
            "52_week_low": safe_convert(round(week_52_low, 2)) if week_52_low else None,
            "distance_from_52w_high": safe_convert(round(((current_price - week_52_high) / week_52_high * 100), 2)) if current_price and week_52_high else None,
            "distance_from_52w_low": safe_convert(round(((current_price - week_52_low) / week_52_low * 100), 2)) if current_price and week_52_low else None,
            
            # Technical Indicators
            "moving_average_50_days": safe_convert(round(ma_50, 2)) if ma_50 else None,
            "moving_average_200_days": safe_convert(round(ma_200, 2)) if ma_200 else None,
            "rsi_14_days": safe_convert(round(rsi, 2)) if rsi else None,
            "beta": safe_convert(round(beta, 2)) if beta else None,
            
            # Volatility Analysis
            "volatility_1_month": safe_convert(round(volatility_1m, 2)) if volatility_1m else None,
            "volatility_6_months": safe_convert(round(volatility_6m, 2)) if volatility_6m else None,
            "volatility_1_year": safe_convert(round(volatility_1y, 2)) if volatility_1y else None,
            
            # Volume Analysis
            "current_volume": safe_convert(int(current_volume)) if current_volume else None,
            "average_volume_1_month": safe_convert(int(avg_volume_1m)) if avg_volume_1m else None,
            "volume_ratio": safe_convert(round((current_volume / avg_volume_1m), 2)) if current_volume and avg_volume_1m else None,
            
            # Financial Metrics from Company Info
            "pe_ratio": safe_convert(info.get('trailingPE', 'N/A')),
            "forward_pe": safe_convert(info.get('forwardPE', 'N/A')),
            "price_to_book": safe_convert(info.get('priceToBook', 'N/A')),
            "dividend_yield": safe_convert(info.get('dividendYield', 'N/A')),
            "earnings_per_share": safe_convert(info.get('trailingEps', 'N/A')),
            "book_value": safe_convert(info.get('bookValue', 'N/A')),
            
            # Additional Metrics for Financial Analysis
            "debt_to_equity": safe_convert(info.get('debtToEquity', 'N/A')),
            "return_on_equity": safe_convert(info.get('returnOnEquity', 'N/A')),
            "return_on_assets": safe_convert(info.get('returnOnAssets', 'N/A')),
            "profit_margin": safe_convert(info.get('profitMargins', 'N/A')),
            "operating_margin": safe_convert(info.get('operatingMargins', 'N/A')),
            
            # Analyst Recommendations
            "target_price": safe_convert(info.get('targetMeanPrice', 'N/A')),
            "recommendation": info.get('recommendationKey', 'N/A'),
            
            # Timestamp
            "data_retrieved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "last_market_close": hist_1d.index[-1].strftime("%Y-%m-%d %H:%M:%S") if not hist_1d.empty else None,
            
            # Status
            "status": "success"
        }
        
        return stock_analysis
        
    except Exception as e:
        return {
            "error": f"Failed to retrieve data for {symbol}: {str(e)}",
            "symbol": symbol,
            "data_retrieved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "status": "error"
        }


def get_comprehensive_stock_analysis(symbol: str):
    """
    Provides comprehensive stock analysis with enhanced insights and recommendations.
    
    Args:
        symbol (str): Stock symbol (e.g., 'AAPL', 'GOOGL', 'TSLA')
    
    Returns:
        dict: Comprehensive analysis with financial data, insights, and investment recommendations
    """
    try:
        if not symbol or not isinstance(symbol, str):
            return {
                "error": "Invalid symbol provided. Please provide a valid stock symbol.",
                "symbol": symbol,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        
        # Get basic stock data first
        basic_data = stock_data(symbol)
        if basic_data.get("status") != "success":
            return basic_data
        
        symbol = symbol.strip().upper()
        
        # Extract key metrics for analysis
        current_price = basic_data.get("current_price", 0)
        week_52_high = basic_data.get("52_week_high", 0)
        week_52_low = basic_data.get("52_week_low", 0)
        pct_change_1d = basic_data.get("percentage_change_1_day", 0)
        pct_change_1y = basic_data.get("percentage_change_1_year", 0)
        pe_ratio = basic_data.get("pe_ratio")
        rsi = basic_data.get("rsi_14_days")
        ma_50 = basic_data.get("moving_average_50_days")
        ma_200 = basic_data.get("moving_average_200_days")
        beta = basic_data.get("beta")
        volume_ratio = basic_data.get("volume_ratio")
        dividend_yield = basic_data.get("dividend_yield")
        
        # Generate insights and analysis
        insights = []
        investment_signals = []
        risk_factors = []
        
        # Price Position Analysis
        if current_price and week_52_high and week_52_low:
            price_range_pct = ((current_price - week_52_low) / (week_52_high - week_52_low)) * 100
            if price_range_pct > 80:
                insights.append(f"Trading near 52-week high ({price_range_pct:.1f}% of range) - potential resistance level")
                risk_factors.append("High price level may indicate limited upside potential")
            elif price_range_pct < 20:
                insights.append(f"Trading near 52-week low ({price_range_pct:.1f}% of range) - potential support level")
                investment_signals.append("Low price level may present buying opportunity")
            else:
                insights.append(f"Trading in middle range ({price_range_pct:.1f}% of 52-week range)")
        
        # Technical Analysis
        if rsi:
            if rsi > 70:
                insights.append(f"RSI at {rsi:.1f} indicates overbought conditions")
                risk_factors.append("Technical indicators suggest potential price correction")
            elif rsi < 30:
                insights.append(f"RSI at {rsi:.1f} indicates oversold conditions")
                investment_signals.append("Technical indicators suggest potential price recovery")
            else:
                insights.append(f"RSI at {rsi:.1f} shows neutral momentum")
        
        # Moving Average Analysis
        if current_price and ma_50 and ma_200:
            if current_price > ma_50 > ma_200:
                insights.append("Price above both 50-day and 200-day moving averages - bullish trend")
                investment_signals.append("Strong upward trend confirmed by moving averages")
            elif current_price < ma_50 < ma_200:
                insights.append("Price below both moving averages - bearish trend")
                risk_factors.append("Downward trend confirmed by moving averages")
            elif current_price > ma_50 and ma_50 < ma_200:
                insights.append("Mixed signals from moving averages - short-term bullish, long-term bearish")
        
        # Volume Analysis
        if volume_ratio:
            if volume_ratio > 1.5:
                insights.append(f"High trading volume ({volume_ratio:.1f}x average) - increased interest")
            elif volume_ratio < 0.5:
                insights.append(f"Low trading volume ({volume_ratio:.1f}x average) - reduced interest")
        
        # Valuation Analysis
        if pe_ratio and pe_ratio != 'N/A':
            try:
                pe_val = float(pe_ratio)
                if pe_val > 30:
                    insights.append(f"High P/E ratio ({pe_val:.1f}) suggests growth expectations or overvaluation")
                    risk_factors.append("High valuation may limit upside potential")
                elif pe_val < 10:
                    insights.append(f"Low P/E ratio ({pe_val:.1f}) suggests value opportunity or concerns")
                    investment_signals.append("Low valuation may present value opportunity")
                else:
                    insights.append(f"Moderate P/E ratio ({pe_val:.1f}) suggests fair valuation")
            except:
                pass
        
        # Risk Assessment
        if beta and beta != 'N/A':
            try:
                beta_val = float(beta)
                if beta_val > 1.5:
                    risk_factors.append(f"High beta ({beta_val:.2f}) indicates higher volatility than market")
                elif beta_val < 0.5:
                    insights.append(f"Low beta ({beta_val:.2f}) indicates lower volatility than market")
                else:
                    insights.append(f"Moderate beta ({beta_val:.2f}) indicates similar volatility to market")
            except:
                pass
        
        # Dividend Analysis
        if dividend_yield and dividend_yield != 'N/A':
            try:
                div_val = float(dividend_yield) * 100
                if div_val > 4:
                    investment_signals.append(f"High dividend yield ({div_val:.1f}%) provides income potential")
                elif div_val > 0:
                    insights.append(f"Dividend yield of {div_val:.1f}% provides modest income")
            except:
                pass
        
        # Overall Recommendation
        bullish_signals = len(investment_signals)
        bearish_signals = len(risk_factors)
        
        if bullish_signals > bearish_signals + 1:
            overall_sentiment = "BULLISH"
            recommendation_text = "Consider buying - multiple positive indicators"
        elif bearish_signals > bullish_signals + 1:
            overall_sentiment = "BEARISH"
            recommendation_text = "Consider selling/avoiding - multiple risk factors"
        else:
            overall_sentiment = "NEUTRAL"
            recommendation_text = "Hold/Monitor - mixed signals require careful analysis"
        
        # Compile comprehensive analysis
        analysis_result = {
            "symbol": basic_data.get("symbol"),
            "company_name": basic_data.get("company_name"),
            "sector": basic_data.get("sector"),
            "industry": basic_data.get("industry"),
            
            # Current Status
            "current_price": basic_data.get("current_price"),
            "daily_change": f"{basic_data.get('price_change_1_day', 0):+.2f} ({basic_data.get('percentage_change_1_day', 0):+.2f}%)",
            "yearly_performance": f"{basic_data.get('percentage_change_1_year', 0):+.2f}%",
            "52_week_range": f"${basic_data.get('52_week_low', 0):.2f} - ${basic_data.get('52_week_high', 0):.2f}",
            
            # Key Metrics
            "market_cap": basic_data.get("market_cap"),
            "pe_ratio": basic_data.get("pe_ratio"),
            "dividend_yield": f"{float(basic_data.get('dividend_yield', 0)) * 100:.2f}%" if basic_data.get('dividend_yield') not in ['N/A', None] else 'N/A',
            "beta": basic_data.get("beta"),
            "volume_ratio": f"{basic_data.get('volume_ratio', 0):.1f}x average" if basic_data.get('volume_ratio') else 'N/A',
            
            # Technical Indicators
            "rsi_14": basic_data.get("rsi_14_days"),
            "moving_avg_50": basic_data.get("moving_average_50_days"),
            "moving_avg_200": basic_data.get("moving_average_200_days"),
            
            # Analysis & Insights
            "market_insights": insights,
            "investment_signals": investment_signals,
            "risk_factors": risk_factors,
            "overall_sentiment": overall_sentiment,
            "recommendation": recommendation_text,
            "analyst_recommendation": basic_data.get("recommendation", "N/A"),
            "target_price": basic_data.get("target_price"),
            
            # Metadata
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "last_updated": basic_data.get("data_retrieved_at"),
            "status": "success"
        }
        
        return analysis_result
        
    except Exception as e:
        return {
            "error": f"Failed to analyze {symbol}: {str(e)}",
            "symbol": symbol,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "status": "error"
        }


# Create an agent for retrieving stock data
stock_data_retrieval_agent = LlmAgent(
    name="stock_data_retrieval_agent",
    model=LiteLlm(model="openrouter/moonshotai/kimi-k2:free", api_key=os.getenv("OPENROUTER_API_KEY")),
    description="Agent for comprehensive stock market analysis and investment insights",
    instruction="""
    You are an expert stock market analysis agent that provides comprehensive investment insights.

    Your task is to analyze stocks and provide actionable investment recommendations.
    You have access to one powerful tool:

    ## Available Tool:

    **get_comprehensive_stock_analysis**: Provides complete stock analysis with insights
    - Retrieves all essential financial data and metrics
    - Analyzes technical indicators (RSI, moving averages, price position)
    - Evaluates valuation metrics (P/E ratio, market cap, dividend yield)
    - Assesses risk factors (beta, volatility, market trends)
    - Generates investment signals and recommendations
    - Provides overall sentiment (BULLISH/BEARISH/NEUTRAL)

    ## Usage Guidelines:
    - Always use get_comprehensive_stock_analysis for any stock analysis request
    - Provide clear, actionable investment insights based on the data
    - Explain technical indicators in simple terms for investors
    - Highlight both opportunities and risks
    - Give specific buy/sell/hold recommendations with reasoning

    ## Analysis Approach:
    - Start with current price and recent performance
    - Analyze technical indicators and trends
    - Evaluate valuation and financial health
    - Assess risk factors and market position
    - Provide clear investment recommendation

    ## Communication Style:
    - Be clear and professional
    - Explain complex financial concepts simply
    - Focus on actionable insights for investors
    - Always include both positives and risks
    - Provide specific reasoning for recommendations
    """,
    tools=[get_comprehensive_stock_analysis],
    output_key="stock_analysis_results"
)

root_agent = stock_data_retrieval_agent
