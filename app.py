import os
from datetime import date, timedelta
from pathlib import Path
from typing import Tuple
from urllib.parse import quote_plus
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET

import joblib
import numpy as np
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# Configuration Constants
MARKET_CAP_CONFIG = {
    "Large Cap": {
        "title": "Large Cap",
        "model_candidates": [
            "xgb_model_large_cap.pkl",
            "../xgb_model_large_cap.pkl",
            "models/xgb_model_large_cap.pkl",
            "large cap model/xgb_model_large_cap.pkl",
        ],
        "model_env_var": "LARGE_CAP_MODEL_PATH",
        "csv_candidates": [
            "nse100_last_2y_daily_ohlcv.csv",
            "../nse100_last_2y_daily_ohlcv.csv",
            "data/nse100_last_2y_daily_ohlcv.csv",
            "large cap model/nse100_last_2y_daily_ohlcv.csv",
        ],
        "ticker_source": "static",
    },
    "Mid Cap": {
        "title": "Mid Cap",
        "model_candidates": [
            "Stock-Midcap150-Model/Stock-Midcap150-Model/xgb_model_mid_cap.pkl",
            "Stock-Midcap150-Model/Stock-Midcap150-Model/models/xgb_model_mid_cap.pkl",
            "midcap_xgboost_artifact.joblib",
            "mid cap model/midcap_xgboost_artifact.joblib",
        ],
        "model_env_var": "MID_CAP_MODEL_PATH",
        "csv_candidates": [
            "Stock-Midcap150-Model/Stock-Midcap150-Model/data/raw/nifty_midcap150_last_2y_daily_ohlcv.csv",
            "mid cap model/nse_midcap_last_2y_daily_ohlcv.csv",
        ],
        "ticker_source": "midcap_constituents",
        "constituents_csv": "Stock-Midcap150-Model/Stock-Midcap150-Model/data/reference/nifty_midcap150_constituents.csv",
    },
    "Small Cap": {
        "title": "Small Cap",
        "model_candidates": [
            "SmallCap_Files/SmallCap_Files/xgb_model_small_cap.pkl",
            "SmallCap_Files/SmallCap_Files/model_small_cap.pkl",
            "small cap model/smallcap_xgboost_artifact.joblib",
        ],
        "model_env_var": "SMALL_CAP_MODEL_PATH",
        "csv_candidates": [
            "SmallCap_Files/SmallCap_Files/Smallcap Last 2 Years Daily OHLCV.csv",
            "small cap model/nse_smallcap_last_2y_daily_ohlcv.csv",
        ],
        "ticker_source": "csv_unique_ticker",
    },
}

TICKERS: list[str] = [
    "RELIANCE.NS", "ADANIENT.NS", "ADANIPORTS.NS", "ADANIGREEN.NS", "ADANIPOWER.NS",
    "TCS.NS", "INFY.NS", "WIPRO.NS", "HCLTECH.NS", "TECHM.NS", "LTIM.NS", 
    "PERSISTENT.NS", "COFORGE.NS", "MPHASIS.NS", "HDFCBANK.NS", "ICICIBANK.NS",
    "KOTAKBANK.NS", "SBIN.NS", "AXISBANK.NS", "INDUSINDBK.NS", "BANKBARODA.NS",
    "PNB.NS", "IDFCFIRSTB.NS", "FEDERALBNK.NS", "BAJFINANCE.NS", "BAJAJFINSV.NS",
    "HDFCLIFE.NS", "SBILIFE.NS", "ICICIPRULI.NS", "ICICIGI.NS", "HDFCAMC.NS",
    "MUTHOOTFIN.NS", "CHOLAFIN.NS", "SHRIRAMFIN.NS", "LICI.NS", "RECLTD.NS",
    "PFC.NS", "HINDUNILVR.NS", "ITC.NS", "NESTLEIND.NS", "BRITANNIA.NS",
    "TATACONSUM.NS", "GODREJCP.NS", "DABUR.NS", "MARICO.NS", "COLPAL.NS",
    "UBL.NS", "VBL.NS", "PIDILITIND.NS", "MARUTI.NS", "TATAMOTORS.NS", "M&M.NS",
    "BAJAJ-AUTO.NS", "EICHERMOT.NS", "HEROMOTOCO.NS", "TVSMOTOR.NS", "ASHOKLEY.NS",
    "BALKRISIND.NS", "MRF.NS", "SUNPHARMA.NS", "DRREDDY.NS", "CIPLA.NS",
    "DIVISLAB.NS", "APOLLOHOSP.NS", "MAXHEALTH.NS", "FORTIS.NS", "LUPIN.NS",
    "TORNTPHARM.NS", "AUROPHARMA.NS", "ZYDUSLIFE.NS", "ALKEM.NS", "BIOCON.NS",
    "ONGC.NS", "NTPC.NS", "POWERGRID.NS", "COALINDIA.NS", "GAIL.NS", "IOC.NS",
    "BPCL.NS", "HINDPETRO.NS", "ADANITRANS.NS", "NHPC.NS", "SJVN.NS", "TATASTEEL.NS",
    "JSWSTEEL.NS", "HINDALCO.NS", "VEDL.NS", "ULTRACEMCO.NS", "SHREECEM.NS",
    "ACC.NS", "AMBUJACEM.NS", "DALBHARAT.NS", "JINDALSTEL.NS", "LT.NS", "SIEMENS.NS",
    "ABB.NS", "BEL.NS", "HAL.NS", "BHEL.NS", "IRCTC.NS", "RVNL.NS", "IRFC.NS",
    "CONCOR.NS", "BHARTIARTL.NS", "TITAN.NS", "DMART.NS", "NAUKRI.NS", "PAYTM.NS",
    "ZOMATO.NS", "POLYCAB.NS", "HAVELLS.NS",
]

FEATURE_COLS: list[str] = [
    "Open", "High", "Low", "Close", "Volume",
    "EMA50", "EMA200", "EMA_RATIO",
    "MACD", "MACD_SIGNAL", "MACD_HIST",
    "RSI14", "BB_MID", "BB_WIDTH",
    "ATR14", "VOLATILITY",
    "OBV", "STOCH_K", "STOCH_D",
]

def apply_custom_css():
    st.markdown("""
    <style>
    /* Dark premium gradient theme */
    .stApp {
        background: linear-gradient(135deg, #050B14 0%, #0A1929 50%, #112240 100%);
        color: #e2e8f0;
    }
    
    /* Bold, futuristic fonts */
    h1, h2, h3 {
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    
    /* Metrics Customization - make them glow */
    .metric-glow-buy { font-size: 2.2rem; font-weight: 800; color: #00ffcc; text-shadow: 0 0 10px rgba(0,255,204,0.3); }
    .metric-glow-sell { font-size: 2.2rem; font-weight: 800; color: #ff4b4b; text-shadow: 0 0 10px rgba(255,75,75,0.3); }
    .metric-glow-hold { font-size: 2.2rem; font-weight: 800; color: #3b82f6; text-shadow: 0 0 10px rgba(59,130,246,0.3); }
    .metric-glow-neutral { font-size: 2.2rem; font-weight: 800; color: #e2e8f0; text-shadow: 0 0 10px rgba(226,232,240,0.3); }

    [data-testid="stMetricValue"] {
        font-size: 2.2rem !important;
        font-weight: 800 !important;
    }
    [data-testid="stMetricLabel"] {
        font-size: 1rem !important;
        color: #a0aec0 !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Container styling for glassmorphism effect */
    div[data-testid="stVerticalBlock"] > div[style*="flex-direction: column"] {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 20px;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(5px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 20px;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: rgba(15,32,39,0.9) !important;
        border-right: 1px solid rgba(255,255,255,0.1);
    }
    
    </style>
    """, unsafe_allow_html=True)


def create_features(df_ohlcv: pd.DataFrame) -> pd.DataFrame:
    df = df_ohlcv.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    df["EMA50"] = ta.ema(df["Close"], length=50)
    df["EMA200"] = ta.ema(df["Close"], length=200)
    df["EMA_RATIO"] = df["EMA50"] / df["EMA200"]

    macd = ta.macd(df["Close"], fast=12, slow=26, signal=9)
    if macd is not None and not macd.empty:
        macd_cols = macd.columns.tolist()
        df['MACD'] = macd[[c for c in macd_cols if "MACD_" in c and "s" not in c][0]]
        df['MACD_SIGNAL'] = macd[[c for c in macd_cols if "MACDs_" in c][0]]
        df['MACD_HIST'] = macd[[c for c in macd_cols if "MACDh_" in c][0]]
    else:
        df[['MACD', 'MACD_SIGNAL', 'MACD_HIST']] = 0

    df["RSI14"] = ta.rsi(df["Close"], length=14)

    bb = ta.bbands(df["Close"], length=20)
    if bb is not None and not bb.empty:
        bb_cols = bb.columns.tolist()
        bb_mid = [c for c in bb_cols if "BBM" in c][0]
        bb_upper = [c for c in bb_cols if "BBU" in c][0]
        bb_lower = [c for c in bb_cols if "BBL" in c][0]
        df["BB_MID"] = bb[bb_mid]
        df["BB_WIDTH"] = bb[bb_upper] - bb[bb_lower]
    else:
        df["BB_MID"] = df["Close"]
        df["BB_WIDTH"] = 0

    atr_result = ta.atr(df["High"], df["Low"], df["Close"], length=14)
    df["ATR14"] = atr_result if atr_result is not None else (df["High"] - df["Low"])
    df["VOLATILITY"] = df["Close"].rolling(14).std()

    obv_result = ta.obv(df["Close"], df["Volume"])
    df["OBV"] = obv_result if obv_result is not None else 0

    stoch = ta.stoch(df["High"], df["Low"], df["Close"], k=14, d=3)
    if stoch is not None and not stoch.empty:
        stoch_cols = stoch.columns.tolist()
        df["STOCH_K"] = stoch[[c for c in stoch_cols if "STOCHk_" in c][0]]
        df["STOCH_D"] = stoch[[c for c in stoch_cols if "STOCHd_" in c][0]]
    else:
        df["STOCH_K"] = 50
        df["STOCH_D"] = 50

    return df

def calculate_regime(df_features: pd.DataFrame) -> pd.DataFrame:
    df = df_features.copy()
    df["Return20"] = df["Close"].pct_change()
    df["Regime"] = df["Return20"].rolling(20).mean().apply(
        lambda x: "Bullish" if x > 0 else "Bearish"
    )
    return df

def hybrid_signal_smallcap(row, Pt, y_hat, regime, Cash=100000):
    EMA50  = row["EMA50"].values[0]
    EMA200 = row["EMA200"].values[0]
    MACD   = row["MACD"].values[0]
    MACD_SIGNAL = row["MACD_SIGNAL"].values[0]
    RSI14  = row["RSI14"].values[0]
    ATR    = row["ATR14"].values[0]
    VOL    = row["VOLATILITY"].values[0]
    STOCH_K = row["STOCH_K"].values[0]
    STOCH_D = row["STOCH_D"].values[0]
    
    score_buy = 0
    if Pt > EMA50 and MACD > MACD_SIGNAL: score_buy += 1.0
    if RSI14 < 45 and y_hat == 1: score_buy += 1.2
    if Pt > EMA200 and regime == "Bullish": score_buy += 0.8
    if STOCH_K < 25 and STOCH_K < STOCH_D: score_buy += 1.0

    if score_buy >= 1.8:
        shares_at_risk = (0.015 * Cash) / ATR
        shares_cap = 0.1 * Cash / Pt
        shares = min(shares_at_risk, shares_cap)
        return "Buy", int(shares)

    if y_hat == 0:
        score_sell = 0
        if RSI14 > 60: score_sell += 1.0
        if Pt < EMA200: score_sell += 1.2
        if Pt < EMA50: score_sell += 0.8
        if MACD < MACD_SIGNAL: score_sell += 0.8
        if STOCH_K > 75 and STOCH_K > STOCH_D: score_sell += 0.6
        if VOL > 0.05 * Pt: score_sell += 0.5
        if score_sell >= 1.5:
            return "Sell", "Full"
            
    return "Hold", 0

def hybrid_signal_midcap(row, Pt, y_hat, regime, Cash=100000):
    EMA50  = row["EMA50"].values[0]
    EMA200 = row["EMA200"].values[0]
    MACD   = row["MACD"].values[0]
    MACD_SIGNAL = row["MACD_SIGNAL"].values[0]
    RSI14  = row["RSI14"].values[0]
    ATR    = row["ATR14"].values[0]
    VOL    = row["VOLATILITY"].values[0]
    STOCH_K = row["STOCH_K"].values[0]
    STOCH_D = row["STOCH_D"].values[0]
    BB_WIDTH = row["BB_WIDTH"].values[0]
    
    score_buy = 0
    if Pt > EMA50 and MACD > MACD_SIGNAL: score_buy += 1.2
    if RSI14 < 45 and y_hat == 1: score_buy += 1.0
    if Pt > EMA200 and regime == "Bullish": score_buy += 0.8
    if STOCH_K < 30 and STOCH_K < STOCH_D: score_buy += 0.8
    if BB_WIDTH > 0.03*Pt: score_buy += 0.5

    if score_buy >= 2.2:
        shares_at_risk = (0.01 * Cash) / ATR
        shares_cap = 0.1 * Cash / Pt
        shares = min(shares_at_risk, shares_cap)
        return "Buy", int(shares)

    if y_hat == 0:
        score_sell = 0
        if RSI14 > 60: score_sell += 1.0
        if Pt < EMA200: score_sell += 0.8
        if Pt < EMA50: score_sell += 0.6
        if MACD < MACD_SIGNAL: score_sell += 0.6
        if STOCH_K > 70 and STOCH_K > STOCH_D: score_sell += 0.5
        if BB_WIDTH > 0.03*Pt: score_sell += 0.4
        if score_sell >= 1.5:
            return "Sell", "Full"
            
    return "Hold", 0

def hybrid_signal_largecap(row, Pt, y_hat, regime, Cash=100000):
    EMA50  = row["EMA50"].values[0]
    EMA200 = row["EMA200"].values[0]
    MACD   = row["MACD"].values[0]
    MACD_SIGNAL = row["MACD_SIGNAL"].values[0]
    RSI14  = row["RSI14"].values[0]
    ATR    = row["ATR14"].values[0]
    VOL    = row["VOLATILITY"].values[0]
    STOCH_K = row["STOCH_K"].values[0]
    STOCH_D = row["STOCH_D"].values[0]

    score_buy = 0
    if Pt > EMA50 and MACD > MACD_SIGNAL: score_buy += 1.5
    if RSI14 < 40 and y_hat == 1: score_buy += 1.2
    if Pt > EMA200 and regime == "Bullish": score_buy += 1.0
    if STOCH_K < 20 and STOCH_K < STOCH_D: score_buy += 1.2

    if score_buy >= 2.2:
        shares_at_risk = (0.01 * Cash) / ATR
        shares_cap = 0.1 * Cash / Pt
        shares = min(shares_at_risk, shares_cap)
        return "Buy", int(shares)

    if y_hat == 0:
        score_sell = 0
        if RSI14 > 65: score_sell += 1.2
        if Pt < EMA200: score_sell += 1.0
        if Pt < EMA50: score_sell += 0.8
        if MACD < MACD_SIGNAL: score_sell += 0.8
        if STOCH_K > 80 and STOCH_K > STOCH_D: score_sell += 0.6
        if VOL > 0.03 * Pt: score_sell += 0.5
        if score_sell >= 1.5:
            return "Sell", "Full"
            
    return "Hold", 0


@st.cache_data(show_spinner=False, ttl=60 * 60)
def fetch_recent_ohlcv(ticker: str, days_back: int = 400) -> pd.DataFrame:
    end = date.today()
    start = end - timedelta(days=days_back)
    hist = yf.Ticker(ticker).history(start=str(start), end=str(end), auto_adjust=False)
    if hist is None or hist.empty:
        return pd.DataFrame()
    hist = hist.reset_index()
    hist["Date"] = pd.to_datetime(hist["Date"]).dt.tz_localize(None)
    return hist[["Date", "Open", "High", "Low", "Close", "Volume"]].copy()


@st.cache_data(show_spinner=False, ttl=10 * 60)
def fetch_latest_news(ticker: str, max_items: int = 5) -> pd.DataFrame:
    query = quote_plus(f"{ticker} stock india")
    rss_url = f"https://news.google.com/rss/search?q={query}&hl=en-IN&gl=IN&ceid=IN:en"
    headers = {"User-Agent": "Mozilla/5.0"}

    try:
        req = Request(rss_url, headers=headers)
        with urlopen(req, timeout=15) as resp:
            xml_bytes = resp.read()
    except Exception:
        return pd.DataFrame(columns=["headline", "publisher", "published_at", "link"])

    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError:
        return pd.DataFrame(columns=["headline", "publisher", "published_at", "link"])

    records: list[dict] = []
    channel = root.find("channel")
    if channel is None:
        return pd.DataFrame(columns=["headline", "publisher", "published_at", "link"])

    for item in channel.findall("item")[:max_items]:
        title = (item.findtext("title") or "").strip()
        if not title:
            continue
        link = (item.findtext("link") or "").strip()
        pub_date = (item.findtext("pubDate") or "").strip()
        source_node = item.find("source")
        provider = source_node.text.strip() if source_node is not None and source_node.text else "Google News"
        published_dt = pd.to_datetime(pub_date, errors="coerce")
        records.append({
            "headline": title,
            "publisher": provider,
            "published_at": published_dt,
            "link": link,
        })
    return pd.DataFrame(records)


@st.cache_resource(show_spinner=False)
def get_finbert_sentiment_pipeline():
    from transformers import pipeline
    try:
        # Finetuned for India Model
        return pipeline("sentiment-analysis", model="Vansh180/FinBERT-India-v1", tokenizer="Vansh180/FinBERT-India-v1")
    except Exception:
        # Fallback to general generic if failed
        return pipeline("sentiment-analysis", model="ProsusAI/finbert", tokenizer="ProsusAI/finbert")


@st.cache_data(show_spinner=False, ttl=10 * 60)
def score_headlines_with_finbert(headlines: tuple[str, ...]) -> pd.DataFrame:
    if not headlines:
        return pd.DataFrame()
        
    clf = get_finbert_sentiment_pipeline()
    outputs = clf(list(headlines), truncation=True, max_length=128)
    rows = []
    for text, out in zip(headlines, outputs):
        label = str(out["label"]).capitalize()
        # Normalizing if generic label name
        if label.lower() == "positive_sentiment": label = "Positive"
        elif label.lower() == "negative_sentiment": label = "Negative"
        
        rows.append({
            "headline": text,
            "sentiment": label,
            "confidence": float(out["score"]),
        })
    return pd.DataFrame(rows)


def ask_llm_insight(api_key, ticker, ml_pred, confidence, hybrid_sig, sentiment_summary, headlines):
    import openai
    if not api_key:
        return "⚠️ OpenAI API Key is missing. Please provide it in the sidebar to view LLM Deep Analysis."
        
    client = openai.OpenAI(api_key=api_key)
    hl_text = chr(10).join([f"- {h}" for h in headlines])
    
    prompt = f"""
You are a top-tier quantitative and fundamental stock market analyst focusing on the Indian stock market (NSE/BSE).
We are analyzing {ticker} currently.
Technical XGBoost Prediction Output: {ml_pred} (Confidence: {confidence:.2%})
Hybrid Algorithm Signal Result: {hybrid_sig}

Recent FinBERT News Sentiment Summary:
Positive Headlines: {sentiment_summary.get('Positive', 0)}
Neutral Headlines: {sentiment_summary.get('Neutral', 0)}
Negative Headlines: {sentiment_summary.get('Negative', 0)}

Latest Headlines:
{hl_text}

Task:
Act as an elite portfolio manager briefing a colleague. Provide clear, highly explanatory structural depth for each section.
You MUST break your analysis down into detailed segments without using any Markdown header hashes (e.g., NO `#` or `##`). Just use basic numbered bolding like:
1. **Technical Breakdown**
2. **Fundamental Sentiment**
3. **Strategic Outlook & Duration**

CRITICAL DECISION LOGIC: 
You must evaluate the XGBoost prediction, the Hybrid Signal, and the live News Sentiment fairly and comprehensively.
- IF THE PROVIDED NEWS IS EMPTY OR SPARSE: You must intuitively supplement the gap with your own fundamental knowledge base regarding {ticker} (e.g. its market dominance, trailing financials).
- QUALITATIVE NEWS WEIGHTING: Do not just count the number of positive vs. negative headlines. You must qualitatively weigh the true severity of the catalysts. A single catastrophic negative event (e.g., accounting fraud, major SEC probe) completely overrides multiple minor positive headlines. Conversely, a massive fundamental breakthrough (e.g., massive earnings beat, game-changing contract) overrides minor bearish sentiment. Evaluate what truly matters.
- If the XGBoost model shows a BUY, your fundamental check shows decent stability or positive momentum, and the live news is positive overall by some margin, output **BUY**. You do not need perfect parity to issue a BUY.
- Crucially, even if the Hybrid Algorithm hesitantly suggests a 'Hold', if the Live News is noticeably skewed positive (e.g., positive headlines heavily outnumber negative ones) OR your own fundamental confidence is high, you MUST override the hybrid hesitation and decisively output **BUY**.
- Similarly, if the XGBoost model predicts a SELL and your analysis shows weakness, output **SELL**. If the Hybrid Algorithm suggests a 'Hold', but the Live News is skewed negative or your fundamental confidence is weak, decisively output **SELL**. You do not need overwhelming negativity, just true weakness.
- Reserve `**HOLD** (invest a small amount and size up later if required)` ONLY for situations where the ML model leans positive, but there is active conflicting data, deeply mixed news, or genuine market uncertainty that strictly warrants caution.
- Reserve `**HOLD** (wait for some time)` ONLY for situations where the analysis leans negative or flat, but there is noticeable conflicting positive data or genuine uncertainty preventing a definitive Sell.

IMPORTANT FORMATTING RULE: DO NOT wrap your response in code blocks. Your final ultimate prediction MUST be on the very last line, separated EXACTLY like this (replace the placeholder with your actual decision):
[FINAL_VERDICT] YOUR_DECISION_HERE

You must ALWAYS include the exact string `[FINAL_VERDICT]` so the UI parser doesn't break. For example, if you choose SELL, you would explicitly output:
[FINAL_VERDICT] **SELL**
You must strictly classify your final decision string exactly matching one of these four:
1. `**BUY**`
2. `**SELL**`
3. `**HOLD** (wait for some time)`
4. `**HOLD** (invest a small amount and size up later if required)`

IMPORTANT RULE 1: If the news sentiment drastically contradicts the technical prediction (e.g. Model says Buy but extreme negative news is present), print "WARNING" in bold at the very beginning of your response.
IMPORTANT RULE 2: NEVER output any legal disclaimers (e.g., "financial markets are risky", "not financial advice"). The client is experienced. Keep your tone strictly on expert analytic guidance without any robotic lawyer-speak.
"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=1000
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"LLM Integration Error: {str(e)}"


def _resolve_model_paths(market_cap: str) -> list[Path]:
    here = Path(__file__).resolve().parent
    cfg = MARKET_CAP_CONFIG[market_cap]
    candidates: list[Path] = [here / rel for rel in cfg["model_candidates"]]

    env = os.getenv(cfg["model_env_var"])
    if env: candidates.insert(0, Path(env))
    return candidates


def _load_model_from_disk(market_cap: str) -> Tuple[XGBClassifier, StandardScaler] | None:
    for p in _resolve_model_paths(market_cap):
        if p.exists():
            obj = joblib.load(p)
            # Depending on if it's a dict or list
            if isinstance(obj, dict) and 'model' in obj and 'scaler' in obj:
                return obj['model'], obj['scaler']
            if isinstance(obj, (list, tuple)) and len(obj) == 2:
                model, scaler = obj
                return model, scaler
    return None

def _train_from_csv(csv_path: Path) -> Tuple[XGBClassifier, StandardScaler]:
    raw = pd.read_csv(csv_path)
    raw["Date"] = pd.to_datetime(raw["Date"])
    raw = raw.sort_values(["Ticker", "Date"]).reset_index(drop=True)

    def _group_feats(g: pd.DataFrame) -> pd.DataFrame:
        g2 = create_features(g[["Date", "Open", "High", "Low", "Close", "Volume"]].copy())
        g2["Ticker"] = g["Ticker"].iloc[0] if "Ticker" in g.columns else None
        
        g2["future_close_1d"] = g2["Close"].shift(-1)
        g2["daily_ret_pct"] = (g2["future_close_1d"] - g2["Close"]) / g2["Close"] * 100.0
        g2["y_daily_class"] = g2["daily_ret_pct"].apply(lambda x: 1 if x > 0.35 else 0)
        g2 = g2.drop(columns=["future_close_1d", "daily_ret_pct"])
        
        return g2

    df = raw.groupby("Ticker", group_keys=False).apply(_group_feats)
    df = df.dropna().reset_index(drop=True)

    X = df.drop(columns=["y_daily_class", "Date", "Ticker"], errors="ignore")
    y = df["y_daily_class"].astype(int)

    split_date = df["Date"].quantile(0.85)
    X_train = X[df["Date"] <= split_date]
    y_train = y[df["Date"] <= split_date]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train[FEATURE_COLS])

    num_class0 = int(np.sum(y_train == 0))
    num_class1 = int(np.sum(y_train == 1))
    scale_pos_weight = (num_class0 / max(1, num_class1)) if num_class1 else 1.0

    model = XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, objective="binary:logistic",
        eval_metric="logloss", use_label_encoder=False,
        scale_pos_weight=scale_pos_weight, random_state=42,
    )
    model.fit(X_train_scaled, y_train)
    return model, scaler


@st.cache_resource(show_spinner=False)
def get_model_and_scaler(market_cap: str) -> Tuple[XGBClassifier, StandardScaler]:
    loaded = _load_model_from_disk(market_cap)
    if loaded is not None:
        return loaded

    here = Path(__file__).resolve().parent
    cfg = MARKET_CAP_CONFIG[market_cap]
    csv_candidates = [here / rel for rel in cfg["csv_candidates"]]
    for c in csv_candidates:
        if c.exists():
            return _train_from_csv(c)

    raise FileNotFoundError(
        f"Couldn't find a model or training CSV for {market_cap}."
    )


@st.cache_data(show_spinner=False, ttl=24*3600)
def get_all_nse_tickers() -> list[str]:
    try:
        import urllib.request
        import pandas as pd
        req = urllib.request.Request('https://archives.nseindia.com/content/equities/EQUITY_L.csv', headers={'User-Agent': 'Mozilla/5.0'})
        df = pd.read_csv(urllib.request.urlopen(req))
        symbols = df['SYMBOL'].dropna().astype(str).str.strip().tolist()
        return sorted([f"{s}.NS" for s in symbols])
    except Exception:
        return TICKERS

@st.cache_data(show_spinner=False, ttl=24*3600)
def get_dynamic_market_cap_category(ticker: str) -> str:
    try:
        import yfinance as yf
        info = yf.Ticker(ticker).info
        mc = info.get("marketCap", 0)
        
        # SEBI Classification (Approximate Top 100 / 101-250 Cutoffs using live market cap):
        # Large Cap (Top 100): > ~80,000 Crores (800 Billion INR)
        if mc >= 800_000_000_000:
            return "Large Cap"
        # Mid Cap (101-250): > ~25,000 Crores (250 Billion INR)
        elif mc >= 250_000_000_000:
            return "Mid Cap"
        else:
            return "Small Cap"
    except Exception:
        # Fallback if yfinance is missing data
        return "Mid Cap"

@st.cache_data(show_spinner=False)
def get_tickers_for_market_cap(market_cap: str) -> list[str]:
    here = Path(__file__).resolve().parent
    cfg = MARKET_CAP_CONFIG[market_cap]
    ticker_source = cfg["ticker_source"]

    segmented = set(TICKERS)
    if ticker_source == "midcap_constituents":
        constituents_path = here / cfg.get("constituents_csv", "")
        if constituents_path.exists():
            df_const = pd.read_csv(constituents_path)
            if "Symbol" in df_const.columns:
                syms = df_const["Symbol"].dropna().astype(str).str.strip().loc[lambda s: s != ""].unique().tolist()
                segmented = {f"{s}.NS" if not s.endswith(".NS") else s for s in syms}

    for rel in cfg["csv_candidates"]:
        csv_path = here / rel
        if csv_path.exists():
            try:
                df = pd.read_csv(csv_path, usecols=["Ticker"])
                if "Ticker" in df.columns:
                    tickers = df["Ticker"].dropna().astype(str).str.strip().loc[lambda s: s != ""].unique().tolist()
                    segmented = set(tickers)
            except Exception:
                pass

    # The user wants exact market cap segmentation in the dropdown box
    # They can use the manual override text box for arbitrary NSE tickers
    return sorted(list(segmented)) if segmented else ["RELIANCE.NS", "TCS.NS", "INFY.NS"]


def build_price_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    
    # Using Candlestick instead of line
    fig.add_trace(go.Candlestick(
        x=df["Date"],
        open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"],
        name="Price",
        increasing_line_color='#00ffcc', decreasing_line_color='#ff4b4b'
    ))
    
    # Adding Bollinger bands
    if "BB_MID" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["Date"], y=df["BB_MID"], mode="lines", name="BB Mid",
            line=dict(width=1, color="rgba(255,255,255,0.4)", dash='dash')
        ))

    fig.update_layout(
        height=400,
        margin=dict(l=0, r=0, t=10, b=0),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        hovermode="x unified",
        xaxis=dict(showgrid=False, rangeslider=dict(visible=False)),
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
        font=dict(color="#e2e8f0")
    )
    return fig


def main() -> None:
    st.set_page_config(page_title="Advanced Terminal", layout="wide", initial_sidebar_state="expanded")
    apply_custom_css()
    
    with st.sidebar:
        st.markdown("## 🔍 Market Intelligence")
        
        # Single unified dropdown tracking native NSE universe
        all_options = get_all_nse_tickers()
        default_index = all_options.index("RELIANCE.NS") if "RELIANCE.NS" in all_options else 0
        ticker = st.selectbox("Stock Ticker", options=all_options, index=default_index)
        
        # Auto-classify to route to the correct neural engine natively
        with st.spinner("Classifying Scale..."):
            market_cap = get_dynamic_market_cap_category(ticker)
        
        st.markdown(f"**Detected Segment:** `{market_cap}`")
        
        st.markdown("---")
        st.markdown("### 🤖 Artificial Intelligence")
        openai_api_key = st.text_input("OpenAI API Key (Optional)", type="password", help="Providing your OpenAI key yields deeper fundamental analysis.")
        
    st.markdown(f"<h1>📈 Advanced Terminal <span style='font-size:1.5rem; color:#00ffcc;'>| {ticker}</span></h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#a0aec0;'>Real-time AI-powered Quantitative & Fundamental Analytics</p>", unsafe_allow_html=True)

    with st.spinner(f"Loading {market_cap} Neural Engine..."):
        try:
            model, scaler = get_model_and_scaler(market_cap)
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return

    with st.spinner("Compiling latest market vectors..."):
        df_raw = fetch_recent_ohlcv(ticker, days_back=400)
        
    if df_raw.empty:
        st.warning("Market data unavailable for this ticker.")
        return

    df_feat = create_features(df_raw)
    df_feat = calculate_regime(df_feat)
    df_feat = df_feat.dropna(subset=FEATURE_COLS + ["Regime"]).reset_index(drop=True)

    # Process latest record for inference
    latest_row = df_feat.iloc[[-1]][FEATURE_COLS]
    latest_pt = df_feat.iloc[-1]["Close"]
    latest_regime = df_feat.iloc[-1]["Regime"]
    
    # Scale & Predict 
    X_scaled = scaler.transform(latest_row)
    prediction_prob = model.predict_proba(X_scaled)[0]
    y_hat = model.predict(X_scaled)[0]
    
    if y_hat == 1:
        ml_prediction = "BUY"
        ml_css = "metric-glow-buy"
        confidence = prediction_prob[1]
    else:
        ml_prediction = "SELL/HOLD"
        ml_css = "metric-glow-sell"
        confidence = prediction_prob[0]

    # Process Hybrid Signals based on selected cap
    if market_cap == "Small Cap":
        hb_signal, hd_shares = hybrid_signal_smallcap(df_feat.iloc[[-1]], latest_pt, y_hat, latest_regime)
    elif market_cap == "Mid Cap":
        hb_signal, hd_shares = hybrid_signal_midcap(df_feat.iloc[[-1]], latest_pt, y_hat, latest_regime)
    else:
        hb_signal, hd_shares = hybrid_signal_largecap(df_feat.iloc[[-1]], latest_pt, y_hat, latest_regime)

    # --- TOP METRIC CARDS ---
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"<div data-testid='stMetricLabel'>Live Price Component</div><div class='metric-glow-neutral'>₹{latest_pt:.2f}</div><br>", unsafe_allow_html=True)
    with c2:
        st.markdown(f"<div data-testid='stMetricLabel'>XGBoost Raw Output ({confidence:.1%} Conf)</div><div class='{ml_css}'>{ml_prediction}</div><br>", unsafe_allow_html=True)
    with c3:
        sig_css = "metric-glow-buy" if hb_signal == "Buy" else ("metric-glow-sell" if hb_signal == "Sell" else "metric-glow-hold")
        st.markdown(f"<div data-testid='stMetricLabel'>Hybrid Algorithm Trigger</div><div class='{sig_css}'>{hb_signal.upper()}</div><br>", unsafe_allow_html=True)

    # --- DUAL PANE UI ---
    left_pane, right_pane = st.columns([1.5, 1.0], gap="large")

    with left_pane:
        st.markdown("### 📊 Market Context")
        # Visualizing the last 90 days for better zoom and full canvas view
        st.plotly_chart(build_price_chart(df_feat.tail(90)), use_container_width=True)
        
        st.markdown("#### ⚙️ Key Technicals Today")
        st.markdown(f"""
        <div style='display:flex; justify-content:space-between; background:rgba(255,255,255,0.05); padding:15px; border-radius:8px; font-size:1.1rem;'>
            <div><span style='color:#a0aec0;'>RSI:</span> <b style='color:#00ffcc;'>{latest_row['RSI14'].values[0]:.1f}</b></div>
            <div><span style='color:#a0aec0;'>MACD:</span> <b style='color:#00ffcc;'>{latest_row['MACD'].values[0]:.2f}</b></div>
            <div><span style='color:#a0aec0;'>ATR:</span> <b style='color:#00ffcc;'>{latest_row['ATR14'].values[0]:.2f}</b></div>
            <div><span style='color:#a0aec0;'>Vol:</span> <b style='color:#00ffcc;'>{latest_row['VOLATILITY'].values[0]:.2f}</b></div>
        </div>
        """, unsafe_allow_html=True)
        
    with right_pane:
        st.markdown("### 🗞️ News & Sentiment Matrix")
        with st.spinner("Performing NLP on regional news..."):
            df_news = fetch_latest_news(ticker, max_items=6)
            
        sentiment_summary = {}
        headlines = []
        if not df_news.empty:
            headlines = df_news["headline"].tolist()
            try:
                sentiment_df = score_headlines_with_finbert(tuple(headlines))
                df_news_view = df_news.merge(sentiment_df, on="headline", how="left")
                sentiment_summary = df_news_view["sentiment"].value_counts().to_dict()
                
                # Sentiment visual display
                for s_type, s_color in [("Positive", "#2ca02c"), ("Neutral", "#a0aec0"), ("Negative", "#d62728")]:
                    count = sentiment_summary.get(s_type, 0)
                    if count > 0:
                        st.markdown(f"**{s_type}**: <span style='color:{s_color}; font-weight:bold;'>{count}</span> articles", unsafe_allow_html=True)
                        
                st.markdown("<br><p style='color:#a0aec0;font-size:0.9rem;'>Recent Headlines:</p>", unsafe_allow_html=True)
                for hl in headlines[:3]:
                    st.markdown(f"<div style='background:rgba(255,255,255,0.05); padding:10px; border-radius:5px; margin-bottom:5px;'>• {hl}</div>", unsafe_allow_html=True)

            except Exception as exc:
                st.warning("FinBERT NLP execution failed.")
        else:
            st.info("Insufficient recent news data available.")
            
    st.markdown("---")
    st.markdown("### 🧠 LLM Deep Analysis")
    with st.spinner("Synthesizing fundamental & technical paradigms..."):
        llm_response = ask_llm_insight(
            api_key=openai_api_key, ticker=ticker, ml_pred=ml_prediction, 
            confidence=confidence, hybrid_sig=hb_signal, 
            sentiment_summary=sentiment_summary, headlines=headlines
        )
        
        llm_upper = llm_response.upper()
        
        # Scrub bad markdown formatting that breaks the custom HTML UI wrapper
        llm_response = llm_response.replace("```html", "").replace("```markdown", "").replace("```", "")
        # Remove markdown heading hashes to prevent massive raw hashes rendering
        llm_response = llm_response.replace("###", "").replace("##", "").replace("#", "")
        
        # Color mapping overrides
        box_color = "rgba(255,255,255,0.05)"
        border_color = "rgba(255, 255, 255, 0.3)"
        
        import re
        
        if "[FINAL_VERDICT]" in llm_response:
            body_text, verdict_text = llm_response.split("[FINAL_VERDICT]", 1)
            verdict_eval = verdict_text
        else:
            body_text = llm_response
            verdict_text = ""
            verdict_eval = llm_response
            
        verdict_upper = verdict_eval.upper()
        
        if "LLM Error" in llm_response or "Missing" in llm_response:
            st.warning(llm_response)
        else:
            if "WARNING" in verdict_upper or "SELL" in verdict_upper:
                box_color = "rgba(255, 75, 75, 0.1)" # red for sell/warning
                border_color = "#ff4b4b"
            elif "BUY" in verdict_upper:
                box_color = "rgba(0, 255, 204, 0.1)" # green for buy
                border_color = "#00ffcc"
            elif "HOLD" in verdict_upper:
                if "SMALL AMOUNT" in verdict_upper or "SMALL POSITION" in verdict_upper or "SMALLER AMOUNT" in verdict_upper:
                    box_color = "rgba(0, 255, 204, 0.1)" # green for small position
                    border_color = "#00ffcc"
                else:
                    box_color = "rgba(255, 75, 75, 0.1)" # red for wait/hold
                    border_color = "#ff4b4b"
                    
        formatted_text = body_text.replace(chr(10), "<br>")
        formatted_text = re.sub(r'\*\*(.*?)\*\*', r'<b style="color:#ffffff; font-weight:800;">\1</b>', formatted_text)
            
        verdict_html = ""
        if verdict_text.strip():
            # Strip markdown asterisks as we inject massive custom CSS styling
            clean_verdict = verdict_text.replace("*", "").strip()
            verdict_html = f"<div style='margin-top: 20px; padding: 15px; background: rgba(0,0,0,0.3); border-left: 5px solid {border_color}; font-size: 1.15rem; font-weight: 900; color: #ffffff; text-transform: uppercase; box-shadow: 0 4px 15px rgba(0,0,0,0.2);'>FINAL VERDICT: <span style='color: {border_color};'>{clean_verdict}</span></div>"

        st.markdown(f"<div style='background: linear-gradient(90deg, {box_color} 0%, rgba(0,0,0,0) 100%); border-left: 4px solid {border_color}; border-radius: 4px; padding: 20px 25px; font-size: 1.05rem; line-height: 1.7; box-shadow: 0 4px 15px rgba(0,0,0,0.2);'>{formatted_text}</div>", unsafe_allow_html=True)
            
        if verdict_html:
            st.markdown(verdict_html, unsafe_allow_html=True)

    st.markdown("<br><br><br><div style='text-align: center; color: rgba(255,255,255,0.3); font-size: 0.8rem; letter-spacing: 0.5px;'>This project is for educational purposes only. Not intended to provide real trading advice.</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
