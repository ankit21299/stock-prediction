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
@keyframes gradient {
    0% {background-position: 0%}
    100% {background-position: 100%}
}
.title {
    text-align:center;
    color:white;
    font-size:45px;
    font-weight:bold;
}
</style>
""", unsafe_allow_html=True)

# 🏷️ Title
st.markdown('<p class="title">🤖 AI FINANCIAL ADVISOR DASHBOARD</p>', unsafe_allow_html=True)

# 🎛️ Sidebar
st.sidebar.title("⚙️ Controls")

# 🔽 Dropdown
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

# ✍️ Custom Input
custom_stock = st.sidebar.text_input("Or Enter Custom Symbol")
if custom_stock:
    stock = custom_stock.strip().upper()

# 📊 Compare
compare = st.sidebar.multiselect(
    "Compare Stocks",
    ["AAPL", "TSLA", "AMZN", "RELIANCE.NS", "INFY.NS", "TCS.NS"]
)

# 📥 Fetch Data
with st.spinner("Fetching Live Data..."):
    data = yf.download(stock, period="1d", interval="5m")

# ❌ Error handle
if data.empty:
    st.error("Invalid Stock Symbol or No Data")

else:
    # ✅ Ensure float
    current_price = float(data["Close"].iloc[-1])

    # 🤖 ML Prediction
    data["Prediction"] = data["Close"].shift(-1)
    data.dropna(inplace=True)

    X = data["Close"].values.reshape(-1, 1)
    y = data["Prediction"].values

    model = LinearRegression()
    model.fit(X, y)

    # ✅ FIXED (No 3D array issue)
    pred = model.predict(np.array([[current_price]]))[0]

    # 💱 Currency detect
    currency = "₹" if ".NS" in stock else "$"

    # 📊 KPI
    col1, col2, col3 = st.columns(3)
    col1.metric("💰 Current Price", f"{currency} {round(current_price,2)}")
    col2.metric("🔮 Predicted Price", f"{currency} {round(pred,2)}")
    col3.metric("📊 Change", f"{currency} {round(pred-current_price,2)}")

    # 🚦 Signal
    st.subheader("🚦 Trading Signal")

    if pred > current_price:
        st.success("🟢 BUY Signal")
    else:
        st.error("🔴 SELL Signal")

    # 📈 Chart
    st.subheader("📊 Live Price Chart")
    st.line_chart(data["Close"])

    # ===============================
    # 🤖 AI FINANCIAL ADVISOR SYSTEM
    # ===============================
    if compare:
        st.subheader("📊 Stock Comparison")

        compare_data = yf.download(compare, period="1d", interval="5m")["Close"]
        st.line_chart(compare_data)

        st.subheader("🤖 AI Financial Advisor")

        best_stock = None
        best_score = -999

        for stk in compare:
            try:
                data_cmp = yf.download(stk, period="1d", interval="5m")

                if data_cmp.empty:
                    continue

                current = float(data_cmp["Close"].iloc[-1])

                data_cmp["Prediction"] = data_cmp["Close"].shift(-1)
                data_cmp.dropna(inplace=True)

                X_cmp = data_cmp["Close"].values.reshape(-1, 1)
                y_cmp = data_cmp["Prediction"].values

                model_cmp = LinearRegression()
                model_cmp.fit(X_cmp, y_cmp)

                # ✅ FIXED HERE ALSO
                pred_cmp = model_cmp.predict(np.array([[current]]))[0]

                growth = pred_cmp - current
                growth_percent = (growth / current) * 100

                # 🚦 Decision Logic
                if growth_percent > 1:
                    decision = "BUY 📈"
                    risk = "Low Risk 🟢"
                elif growth_percent > 0:
                    decision = "HOLD ⚖️"
                    risk = "Medium Risk 🟡"
                else:
                    decision = "SELL 📉"
                    risk = "High Risk 🔴"

                confidence = min(abs(growth_percent) * 10, 100)
                curr_symbol = "₹" if ".NS" in stk else "$"

                st.markdown(f"""
                **📌 {stk}**
                - Current: {curr_symbol}{round(current,2)}
                - Predicted: {curr_symbol}{round(pred_cmp,2)}
                - Growth: {round(growth_percent,2)}%
                - Decision: {decision}
                - Risk: {risk}
                - Confidence: {round(confidence,2)}%
                """)

                if growth_percent > best_score:
                    best_score = growth_percent
                    best_stock = stk

            except Exception as e:
                continue

        # 🔥 Final Recommendation
        if best_stock:
            st.success(f"🚀 Recommended Stock: {best_stock}")
            st.info("💡 Based on AI analysis, this stock shows highest growth potential.")
        else:
            st.warning("⚠️ Unable to generate recommendation.")