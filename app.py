import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.linear_model import LinearRegression
from streamlit_autorefresh import st_autorefresh

# 🔄 Auto Refresh
st_autorefresh(interval=5000, key="refresh")

# 🔥 Page Config
st.set_page_config(page_title="AI Stock Advisor", layout="wide")

# 🎨 UI
st.markdown("""
<style>
.stApp {
    background: linear-gradient(-45deg, #0f2027, #203a43, #2c5364);
    background-size: 400% 400%;
    animation: gradient 10s ease infinite;
}
.title {
    text-align:center;
    color:white;
    font-size:45px;
    font-weight:bold;
}
.glass {
    background: rgba(255,255,255,0.08);
    padding:20px;
    border-radius:15px;
    backdrop-filter: blur(10px);
    border:1px solid rgba(255,255,255,0.2);
    margin-bottom:10px;
}
.metric {
    font-size:22px;
    color:#00FFAA;
}
</style>
""", unsafe_allow_html=True)

# 🏷️ Title
st.markdown('<p class="title">🤖 AI FINANCIAL ADVISOR</p>', unsafe_allow_html=True)

# 🎛️ Sidebar
st.sidebar.title("⚙️ Controls")

popular_stocks = {
    "Apple (USA)": "AAPL",
    "Tesla (USA)": "TSLA",
    "Amazon (USA)": "AMZN",
    "Google (USA)": "GOOGL",
    "Reliance (India)": "RELIANCE.NS",
    "TCS (India)": "TCS.NS",
    "Infosys (India)": "INFY.NS",
    "HDFC Bank (India)": "HDFCBANK.NS"
}

selected_stock = st.sidebar.selectbox("Select Stock", list(popular_stocks.keys()))
stock = popular_stocks[selected_stock]

custom_stock = st.sidebar.text_input("Or Enter Custom Symbol")
if custom_stock:
    stock = custom_stock.strip().upper()

compare = st.sidebar.multiselect(
    "Compare Stocks",
    ["AAPL", "TSLA", "AMZN", "RELIANCE.NS", "INFY.NS", "TCS.NS"]
)

# 📥 Fetch Data
data = yf.download(stock, period="1d", interval="5m", progress=False)

# 🔥 Clean Data
data.dropna(inplace=True)

# ❌ Handle Empty
if data.empty:
    st.error("⚠️ Invalid Stock Symbol or No Data Available")
    st.stop()

# 🔥 SAFE CURRENT PRICE
try:
    if isinstance(data["Close"], pd.DataFrame):
        current_price = float(data["Close"].iloc[-1].values[0])
    else:
        current_price = float(data["Close"].iloc[-1])
except:
    st.error("⚠️ Error reading stock data")
    st.stop()

# 🤖 ML Prediction
data["Prediction"] = data["Close"].shift(-1)
data.dropna(inplace=True)

X = data["Close"].values.reshape(-1, 1)
y = data["Prediction"].values

model = LinearRegression()
model.fit(X, y)

pred = model.predict(np.array(current_price).reshape(-1,1))[0]

# 💱 Currency
currency = "₹" if ".NS" in stock else "$"

# 📊 KPI CARDS
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"""
    <div class="glass">
    <h4>💰 Current Price</h4>
    <p class="metric">{currency} {round(current_price,2)}</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="glass">
    <h4>🔮 Predicted Price</h4>
    <p class="metric">{currency} {round(pred,2)}</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="glass">
    <h4>📊 Change</h4>
    <p class="metric">{currency} {round(pred-current_price,2)}</p>
    </div>
    """, unsafe_allow_html=True)

# 🚦 Signal
st.subheader("🚦 Trading Signal")
if pred > current_price:
    st.success("🟢 BUY Signal")
else:
    st.error("🔴 SELL Signal")

# 📈 Chart
st.subheader("📊 Live Chart")
st.line_chart(data["Close"])

# =========================
# 🤖 AI FINANCIAL ADVISOR
# =========================
if compare:
    st.subheader("📊 Stock Comparison")

    try:
        compare_data = yf.download(compare, period="1d", interval="5m", progress=False)["Close"]
        compare_data.dropna(inplace=True)
        st.line_chart(compare_data)
    except:
        st.warning("⚠️ Comparison data not available")

    st.subheader("🤖 AI Advisor")

    best_stock = None
    best_score = -999

    for stk in compare:
        try:
            d = yf.download(stk, period="1d", interval="5m", progress=False)
            d.dropna(inplace=True)

            if d.empty:
                continue

            current = float(d["Close"].iloc[-1])

            d["Prediction"] = d["Close"].shift(-1)
            d.dropna(inplace=True)

            Xc = d["Close"].values.reshape(-1,1)
            yc = d["Prediction"].values

            m = LinearRegression()
            m.fit(Xc, yc)

            pred_c = m.predict(np.array(current).reshape(-1,1))[0]

            growth = pred_c - current
            growth_percent = (growth / current) * 100

            if growth_percent > 1:
                decision = "BUY 📈"
                risk = "Low 🟢"
            elif growth_percent > 0:
                decision = "HOLD ⚖️"
                risk = "Medium 🟡"
            else:
                decision = "SELL 📉"
                risk = "High 🔴"

            confidence = min(abs(growth_percent)*10,100)
            curr = "₹" if ".NS" in stk else "$"

            st.markdown(f"""
            <div class="glass">
            <b>{stk}</b><br>
            Current: {curr}{round(current,2)}<br>
            Predicted: {curr}{round(pred_c,2)}<br>
            Growth: {round(growth_percent,2)}%<br>
            Decision: {decision}<br>
            Risk: {risk}<br>
            Confidence: {round(confidence,2)}%
            </div>
            """, unsafe_allow_html=True)

            if growth_percent > best_score:
                best_score = growth_percent
                best_stock = stk

        except:
            continue

    if best_stock:
        st.success(f"🚀 Recommended: {best_stock}")
    else:
        st.warning("⚠️ No recommendation available")
