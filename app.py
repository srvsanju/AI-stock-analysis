import os
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from google import genai

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="AI Stock Analyzer",
    page_icon="📈",
    layout="wide"
)

# --- API KEY HANDLING ---
# Checks Streamlit Cloud Secrets first, then environment variables, then sidebar input
def get_api_key():
    if "GEMINI_API_KEY" in st.secrets:
        return st.secrets["GEMINI_API_KEY"]
    if os.environ.get("GEMINI_API_KEY"):
        return os.environ.get("GEMINI_API_KEY")
    return None

# --- CACHED DATA FETCHING ---
@st.cache_data(ttl=3600)  # Cache data for 1 hour to respect free rate limits
def fetch_stock_data(ticker_symbol: str, period: str = "1y"):
    stock = yf.Ticker(ticker_symbol)
    hist = stock.history(period=period)
    info = stock.info
    return hist, info

@st.cache_data(ttl=3600)
def generate_ai_analysis(api_key: str, ticker: str, metrics_summary: str):
    client = genai.Client(api_key=api_key)
    prompt = f"""
    You are an objective equity research analyst. Analyze the following stock data for {ticker}.
    
    Data Summary:
    {metrics_summary}
    
    Provide a structured, concise analysis covering:
    1. **Executive Summary**: Current valuation and market positioning.
    2. **Bull Case (Key Strengths)**: 2-3 bullet points based on fundamentals and momentum.
    3. **Bear Case (Key Risks)**: 2-3 bullet points highlighting valuation, debt, or growth concerns.
    4. **Key Metrics to Watch**: What investors should monitor next quarter.
    
    Keep the tone neutral, analytical, and strictly grounded in the provided numbers. Do not give direct buy/sell financial advice.
    """
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )
    return response.text

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Settings")
    saved_key = get_api_key()
    if saved_key:
        api_key = saved_key
        st.success("Gemini API Key loaded from secrets/environment.")
    else:
        api_key = st.text_input(
            "Enter Gemini API Key",
            type="password",
            help="Get a free key at aistudio.google.com"
        )
    
    st.divider()
    ticker = st.text_input("Stock Ticker", value="AAPL").upper().strip()
    period = st.selectbox("Chart Timeframe", ["3mo", "6mo", "1y", "2y", "5y"], index=2)
    run_btn = st.button("Analyze Stock", type="primary", use_container_width=True)

# --- MAIN DASHBOARD ---
st.title("📈 AI-Powered Stock Analysis Dashboard")
st.caption("Powered by yfinance, Plotly, and Google Gemini")

if run_btn or ticker:
    if not ticker:
        st.warning("Please enter a valid stock ticker symbol.")
        st.stop()

    with st.spinner(f"Fetching market data for {ticker}..."):
        try:
            hist, info = fetch_stock_data(ticker, period)
            if hist.empty:
                st.error(f"No historical price data found for '{ticker}'. Check the symbol and try again.")
                st.stop()
        except Exception as e:
            st.error(f"Error fetching stock data: {e}")
            st.stop()

    # Company Header
    company_name = info.get("longName", ticker)
    sector = info.get("sector", "N/A")
    industry = info.get("industry", "N/A")
    st.subheader(f"{company_name} ({ticker})")
    st.markdown(f"**Sector:** {sector} | **Industry:** {industry}")

    # Key Metrics Row
    current_price = hist["Close"].iloc[-1]
    prev_close = hist["Close"].iloc[-2] if len(hist) > 1 else current_price
    price_delta = current_price - prev_close
    pct_delta = (price_delta / prev_close) * 100

    market_cap = info.get("marketCap")
    pe_ratio = info.get("trailingPE")
    forward_pe = info.get("forwardPE")
    div_yield = info.get("dividendYield")

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Current Price", f"${current_price:,.2f}", f"{price_delta:+.2f} ({pct_delta:+.2f}%)")
    col2.metric("Market Cap", f"${market_cap / 1e9:,.2f}B" if market_cap else "N/A")
    col3.metric("Trailing P/E", f"{pe_ratio:.2f}" if pe_ratio else "N/A")
    col4.metric("Forward P/E", f"{forward_pe:.2f}" if forward_pe else "N/A")
    col5.metric("Dividend Yield", f"{div_yield * 100:.2f}%" if div_yield else "0.00%")

    # Interactive Candlestick Chart with Moving Averages
    hist["SMA_20"] = hist["Close"].rolling(window=20).mean()
    hist["SMA_50"] = hist["Close"].rolling(window=50).mean()

    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=hist.index,
        open=hist["Open"],
        high=hist["High"],
        low=hist["Low"],
        close=hist["Close"],
        name="Price"
    ))
    fig.add_trace(go.Scatter(
        x=hist.index, y=hist["SMA_20"],
        line=dict(color="orange", width=1.5),
        name="20-Day SMA"
    ))
    fig.add_trace(go.Scatter(
        x=hist.index, y=hist["SMA_50"],
        line=dict(color="cyan", width=1.5),
        name="50-Day SMA"
    ))
    fig.update_layout(
        title=f"{ticker} Price Action & Moving Averages ({period})",
        yaxis_title="Price (USD)",
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)

    # AI Analysis Section
    st.divider()
    st.subheader("🤖 Gemini AI Research Brief")

    if not api_key:
        st.info("👈 Enter your free **Gemini API Key** in the sidebar to generate the AI analysis.")
    else:
        # Prepare clean financial context for the LLM
        period_return = ((hist["Close"].iloc[-1] - hist["Close"].iloc[0]) / hist["Close"].iloc[0]) * 100
        high_52w = info.get("fiftyTwoWeekHigh", "N/A")
        low_52w = info.get("fiftyTwoWeekLow", "N/A")
        revenue_growth = info.get("revenueGrowth")
        profit_margins = info.get("profitMargins")
        debt_to_equity = info.get("debtToEquity")

        metrics_summary = f"""
        - Company: {company_name} ({ticker})
        - Sector/Industry: {sector} / {industry}
        - Current Price: ${current_price:.2f} (52W Range: ${low_52w} - ${high_52w})
        - {period} Price Return: {period_return:.2f}%
        - Market Cap: {f"${market_cap:,.0f}" if market_cap else "N/A"}
        - Trailing P/E: {pe_ratio} | Forward P/E: {forward_pe}
        - Revenue Growth (YoY): {f"{revenue_growth * 100:.2f}%" if revenue_growth else "N/A"}
        - Profit Margin: {f"{profit_margins * 100:.2f}%" if profit_margins else "N/A"}
        - Debt-to-Equity: {debt_to_equity}
        - Business Summary: {info.get("longBusinessSummary", "N/A")[:600]}...
        """

        with st.spinner("Generating AI analysis..."):
            try:
                analysis = generate_ai_analysis(api_key, ticker, metrics_summary)
                st.markdown(analysis)
            except Exception as e:
                st.error(f"Gemini API Error: {e}")

# --- DISCLAIMER ---
st.divider()
st.caption("⚠️ **Disclaimer:** This tool is for educational and informational purposes only. AI-generated insights may contain inaccuracies and do not constitute financial or investment advice.")
