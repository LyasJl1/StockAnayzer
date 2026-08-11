# -*- coding: utf-8 -*-
"""Tableau de bord Streamlit d'analyse multifactorielle d'actions."""

from __future__ import annotations

from html import escape
from math import isfinite, sqrt
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf


PERIODS = {"6 mois": "6mo", "1 an": "1y", "2 ans": "2y", "5 ans": "5y"}
WEIGHTS = {
    "Investisseur": {"Fondamentaux": 25, "Valorisation": 20, "Croissance": 20, "Technique": 20, "Risque": 15},
    "Trader / Swing": {"Fondamentaux": 15, "Valorisation": 10, "Croissance": 15, "Technique": 40, "Risque": 20},
}
CURRENCY_SYMBOLS = {
    "USD": "$", "EUR": "€", "GBP": "£", "JPY": "¥", "CHF": "CHF ",
    "CAD": "C$", "CNY": "¥", "INR": "₹", "KRW": "₩",
}


st.set_page_config(
    page_title="Stock Analyzer & Trading Assistant",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


def apply_styles() -> None:
    """Applique le thème sombre et les composants visuels du tableau de bord."""
    st.markdown(
        """
        <style>
        .stApp { background:#0b1220; color:#e5e7eb; }
        [data-testid="stSidebar"] { background:#111827; }
        [data-testid="stMetric"] { background:#111c2f; border:1px solid #25334a;
          border-radius:12px; padding:14px; }
        .meta,.muted,.disclaimer { color:#94a3b8; }
        .meta { margin-top:-.7rem; margin-bottom:1rem; }
        .decision,.card { background:#111c2f; border:1px solid #25334a;
          border-radius:14px; padding:18px; box-sizing:border-box; }
        .decision { background:linear-gradient(135deg,#12233c,#111827); }
        .score { font-size:clamp(2.3rem,5vw,3.6rem); font-weight:800; line-height:1; }
        .verdict { font-size:1.1rem; font-weight:800; margin:.6rem 0; }
        .good { color:#34d399; } .bad { color:#f87171; } .warn { color:#fb923c; }
        .info { color:#93c5fd; }
        .card h3 { margin:0 0 12px; } .card ul { padding-left:1.2rem; }
        .card li { margin:.55rem 0; line-height:1.4; }
        .bar-row { display:grid; grid-template-columns:120px 1fr 60px; gap:10px;
          align-items:center; margin:10px 0; }
        .bar-track { height:8px; background:#25334a; border-radius:99px; overflow:hidden; }
        .bar-fill { height:100%; border-radius:99px; }
        .trade-row { display:flex; justify-content:space-between; gap:12px;
          border-bottom:1px solid #25334a; padding:9px 0; }
        .trade-row:last-child { border:0; } .value { font-weight:750; text-align:right; }
        .disclaimer { font-size:.8rem; line-height:1.45; }
        @media(max-width:768px) { .bar-row { grid-template-columns:95px 1fr 52px; } }
        </style>
        """,
        unsafe_allow_html=True,
    )


def valid_number(value: Any) -> bool:
    """Retourne vrai si ``value`` est un nombre fini."""
    try:
        return value is not None and bool(pd.notna(value)) and isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return False


def format_price(value: Any, symbol: str) -> str:
    return f"{symbol}{float(value):,.2f}" if valid_number(value) else "N/D"


def format_percentage(value: Any, *, decimal: bool = True) -> str:
    if not valid_number(value):
        return "N/D"
    number = float(value) * (100 if decimal else 1)
    return f"{number:+.1f} %"


@st.cache_data(ttl=900, show_spinner=False)
def load_stock_data(ticker: str, period: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Télécharge les cours et métadonnées Yahoo Finance d'un symbole."""
    stock = yf.Ticker(ticker)
    history = stock.history(period=period, auto_adjust=False)
    if history.empty or "Close" not in history:
        raise ValueError("Aucun historique disponible pour ce ticker.")
    history = history.dropna(subset=["Close"]).copy()
    try:
        info = stock.get_info()
    except Exception:  # yfinance peut fournir les cours même si les métadonnées échouent
        info = {}
    return history, info or {}


def calculate_indicators(history: pd.DataFrame) -> pd.DataFrame:
    data = history.copy()
    close = data["Close"]
    data["MM50"] = close.rolling(50).mean()
    data["MM200"] = close.rolling(200).mean()
    delta = close.diff()
    gains = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    losses = -delta.clip(upper=0).ewm(alpha=1 / 14, adjust=False).mean()
    rs = gains / losses.replace(0, float("nan"))
    data["RSI"] = 100 - (100 / (1 + rs))
    data["MACD"] = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    data["Signal"] = data["MACD"].ewm(span=9, adjust=False).mean()
    return data


def score_analysis(data: pd.DataFrame, info: dict[str, Any], mode: str) -> tuple[dict[str, float], list[str], list[str]]:
    """Calcule cinq sous-scores explicables, puis les arguments associés."""
    close = float(data["Close"].iloc[-1])
    mm50, mm200 = data["MM50"].iloc[-1], data["MM200"].iloc[-1]
    rsi, macd, signal = data["RSI"].iloc[-1], data["MACD"].iloc[-1], data["Signal"].iloc[-1]
    pe, margin = info.get("trailingPE"), info.get("profitMargins")
    revenue_growth, earnings_growth = info.get("revenueGrowth"), info.get("earningsGrowth")
    roe, debt = info.get("returnOnEquity"), info.get("debtToEquity")
    volatility = data["Close"].pct_change().std() * sqrt(252)

    fundamentals = 50.0
    fundamentals += 20 if valid_number(margin) and margin > 0.10 else (-15 if valid_number(margin) and margin < 0 else 0)
    fundamentals += 15 if valid_number(roe) and roe > 0.12 else 0
    fundamentals += 15 if valid_number(info.get("freeCashflow")) and info["freeCashflow"] > 0 else 0
    valuation = 50.0
    valuation += 30 if valid_number(pe) and 0 < pe < 20 else (-20 if valid_number(pe) and pe > 40 else 0)
    valuation += 20 if valid_number(info.get("priceToBook")) and 0 < info["priceToBook"] < 3 else 0
    growth = 50.0
    growth += 25 if valid_number(revenue_growth) and revenue_growth > 0.08 else (-20 if valid_number(revenue_growth) and revenue_growth < 0 else 0)
    growth += 25 if valid_number(earnings_growth) and earnings_growth > 0.08 else (-20 if valid_number(earnings_growth) and earnings_growth < 0 else 0)
    technical = 50.0
    technical += 15 if valid_number(mm50) and close > mm50 else -15
    technical += 15 if valid_number(mm200) and close > mm200 else -15
    technical += 10 if valid_number(macd) and valid_number(signal) and macd > signal else -10
    technical += 10 if valid_number(rsi) and 40 <= rsi <= 65 else (-10 if valid_number(rsi) and rsi > 75 else 0)
    risk = 75.0
    risk -= 25 if valid_number(volatility) and volatility > 0.45 else (10 if valid_number(volatility) and volatility > 0.30 else 0)
    risk -= 20 if valid_number(debt) and debt > 150 else 0
    risk -= 10 if valid_number(info.get("beta")) and info["beta"] > 1.5 else 0

    scores = {
        "Fondamentaux": max(0, min(100, fundamentals)),
        "Valorisation": max(0, min(100, valuation)),
        "Croissance": max(0, min(100, growth)),
        "Technique": max(0, min(100, technical)),
        "Risque": max(0, min(100, risk)),
    }
    favorable, unfavorable = [], []
    (favorable if valid_number(margin) and margin > 0.10 else unfavorable).append("Marge nette solide" if valid_number(margin) and margin > 0.10 else "Rentabilité à surveiller")
    (favorable if valid_number(revenue_growth) and revenue_growth > 0 else unfavorable).append("Chiffre d'affaires en croissance" if valid_number(revenue_growth) and revenue_growth > 0 else "Croissance du CA faible ou négative")
    (favorable if valid_number(mm200) and close > mm200 else unfavorable).append("Cours au-dessus de la MM200" if valid_number(mm200) and close > mm200 else "Cours sous la MM200")
    (favorable if valid_number(macd) and valid_number(signal) and macd > signal else unfavorable).append("MACD au-dessus de son signal" if valid_number(macd) and valid_number(signal) and macd > signal else "Momentum MACD défavorable")
    if valid_number(pe):
        (favorable if 0 < pe < 25 else unfavorable).append("Valorisation P/E raisonnable" if 0 < pe < 25 else "P/E exigeant ou atypique")
    return scores, favorable, unfavorable


def render_chart(data: pd.DataFrame, ticker: str) -> None:
    figure = go.Figure()
    figure.add_trace(go.Candlestick(x=data.index, open=data["Open"], high=data["High"], low=data["Low"], close=data["Close"], name=ticker))
    figure.add_trace(go.Scatter(x=data.index, y=data["MM50"], name="MM50", line={"color": "#60a5fa", "width": 1.5}))
    figure.add_trace(go.Scatter(x=data.index, y=data["MM200"], name="MM200", line={"color": "#f59e0b", "width": 1.5}))
    figure.update_layout(template="plotly_dark", height=520, margin={"l": 10, "r": 10, "t": 35, "b": 10}, xaxis_rangeslider_visible=False, paper_bgcolor="#0b1220", plot_bgcolor="#0b1220", legend={"orientation": "h"})
    st.plotly_chart(figure, use_container_width=True)


def render_dashboard(ticker: str, period: str, mode: str) -> None:
    """Affiche l'unique version du tableau de bord d'analyse."""
    with st.spinner(f"Analyse de {ticker}…"):
        history, info = load_stock_data(ticker, period)
    data = calculate_indicators(history)
    current = float(data["Close"].iloc[-1])
    previous = float(data["Close"].iloc[-2]) if len(data) > 1 else current
    daily_change = current / previous - 1 if previous else 0
    currency = CURRENCY_SYMBOLS.get(str(info.get("currency", "USD")), f"{info.get('currency', 'USD')} ")
    name = escape(str(info.get("longName") or info.get("shortName") or ticker))
    scores, favorable, unfavorable = score_analysis(data, info, mode)
    weights = WEIGHTS[mode]
    global_score = sum(scores[key] * weights[key] for key in scores) / 100
    known = sum(valid_number(info.get(key)) for key in ("trailingPE", "profitMargins", "revenueGrowth", "returnOnEquity", "beta", "debtToEquity"))
    confidence = "Élevé" if known >= 5 and len(data) >= 200 else ("Modéré" if known >= 3 else "Limité")
    verdict = "Configuration favorable" if global_score >= 70 else ("Profil équilibré / à surveiller" if global_score >= 50 else "Prudence recommandée")

    st.title(f"{name} · {escape(ticker)}")
    st.markdown(f"<div class='meta'>{escape(str(info.get('sector', 'Secteur non renseigné')))} · Mode {escape(mode)}</div>", unsafe_allow_html=True)
    metrics = st.columns(6)
    metrics[0].metric("Prix actuel", format_price(current, currency), format_percentage(daily_change))
    metrics[1].metric("P/E", f"{float(info['trailingPE']):.1f}x" if valid_number(info.get("trailingPE")) else "N/D")
    metrics[2].metric("Marge nette", format_percentage(info.get("profitMargins")))
    metrics[3].metric("Croissance CA", format_percentage(info.get("revenueGrowth")))
    metrics[4].metric("MM50", format_price(data["MM50"].iloc[-1], currency))
    metrics[5].metric("MM200", format_price(data["MM200"].iloc[-1], currency))

    left, right = st.columns([1, 1.4])
    with left:
        color = "good" if global_score >= 70 else ("warn" if global_score >= 50 else "bad")
        st.markdown(f"""
        <div class="decision">
          <div class="muted">SCORE GLOBAL</div>
          <div class="score {color}">{global_score:.0f}<span style="font-size:1rem">/100</span></div>
          <div class="verdict">{escape(verdict)}</div>
          <div class="muted">Niveau de confiance : <b>{confidence}</b></div>
        </div>
        """, unsafe_allow_html=True)
    with right:
        bars = "".join(
            f'<div class="bar-row"><span>{escape(label)}</span><div class="bar-track"><div class="bar-fill" style="width:{score:.0f}%;background:{"#34d399" if score >= 65 else "#fb923c" if score >= 45 else "#f87171"}"></div></div><b>{score:.0f}/100</b></div>'
            for label, score in scores.items()
        )
        st.markdown(f'<div class="card"><h3>Analyse multifactorielle</h3>{bars}</div>', unsafe_allow_html=True)

    st.subheader("Graphique & moyennes mobiles")
    render_chart(data, ticker)

    returns = data["Close"].pct_change().dropna()
    volatility = returns.std() * sqrt(252)
    high_52 = data["High"].tail(252).max()
    low_52 = data["Low"].tail(252).min()
    rsi, macd, signal = data["RSI"].iloc[-1], data["MACD"].iloc[-1], data["Signal"].iloc[-1]
    stop_loss = current * (0.92 if mode == "Investisseur" else 0.96)
    take_profit = current * (1.16 if mode == "Investisseur" else 1.08)
    risk_reward = (take_profit - current) / (current - stop_loss)
    fundamentals = [
        ("ROE", format_percentage(info.get("returnOnEquity"))),
        ("Free Cash Flow", format_price(info.get("freeCashflow"), currency)),
        ("Dette / capitaux propres", f"{float(info['debtToEquity']):.1f} %" if valid_number(info.get("debtToEquity")) else "N/D"),
        ("Price / Book", f"{float(info['priceToBook']):.1f}x" if valid_number(info.get("priceToBook")) else "N/D"),
    ]
    technical = [
        ("RSI (14)", f"{float(rsi):.1f}" if valid_number(rsi) else "N/D"),
        ("MACD", f"{float(macd):.2f}" if valid_number(macd) else "N/D"),
        ("Signal MACD", f"{float(signal):.2f}" if valid_number(signal) else "N/D"),
        ("Volatilité annualisée", format_percentage(volatility)),
        ("Plus haut 52 semaines", format_price(high_52, currency)),
        ("Plus bas 52 semaines", format_price(low_52, currency)),
    ]

    col1, col2 = st.columns(2)
    with col1:
        rows = "".join(f'<div class="trade-row"><span>{escape(label)}</span><span class="value">{escape(value)}</span></div>' for label, value in fundamentals)
        st.markdown(f'<div class="card"><h3>Fondamentaux · valorisation · croissance</h3>{rows}</div>', unsafe_allow_html=True)
    with col2:
        rows = "".join(f'<div class="trade-row"><span>{escape(label)}</span><span class="value">{escape(value)}</span></div>' for label, value in technical)
        st.markdown(f'<div class="card"><h3>Analyse technique & risque</h3>{rows}</div>', unsafe_allow_html=True)

    st.subheader("Arguments & aide à la décision")
    pro, con = st.columns(2)
    with pro:
        items = "".join(f'<li>{escape(item)}</li>' for item in favorable) or "<li>Données positives insuffisantes</li>"
        st.markdown(f'<div class="card"><h3 class="good">Arguments favorables</h3><ul>{items}</ul></div>', unsafe_allow_html=True)
    with con:
        items = "".join(f'<li>{escape(item)}</li>' for item in unfavorable) or "<li>Aucun signal défavorable majeur détecté</li>"
        st.markdown(f'<div class="card"><h3 class="bad">Arguments défavorables</h3><ul>{items}</ul></div>', unsafe_allow_html=True)

    st.subheader("Synthèse d'aide à la décision")
    summary = f"Le profil {mode.lower()} obtient {global_score:.0f}/100. {verdict}. Les niveaux ci-dessous sont indicatifs et doivent être adaptés à votre horizon, votre taille de position et votre tolérance au risque."
    trade_rows = f"""
      <p>{escape(summary)}</p>
      <div class="trade-row"><span>Stop-Loss indicatif</span><span class="value bad">{format_price(stop_loss, currency)} ({format_percentage(stop_loss/current-1, decimal=True)})</span></div>
      <div class="trade-row"><span>Take-Profit indicatif</span><span class="value good">{format_price(take_profit, currency)} ({format_percentage(take_profit/current-1, decimal=True)})</span></div>
      <div class="trade-row"><span>Ratio risque / récompense</span><span class="value">1 : {risk_reward:.1f}</span></div>
    """
    st.markdown(f'<div class="decision">{trade_rows}</div>', unsafe_allow_html=True)
    st.markdown("<p class='disclaimer'>Information éducative uniquement : cette analyse automatisée ne constitue ni un conseil en investissement, ni une recommandation d'achat ou de vente. Les données peuvent être incomplètes ou différées.</p>", unsafe_allow_html=True)


apply_styles()
with st.sidebar:
    st.header("Paramètres d'analyse")
    ticker_input = st.text_input("Ticker", value="AAPL", placeholder="AAPL, MSFT, MC.PA…").strip().upper()
    selected_period = st.selectbox("Période", options=list(PERIODS))
    selected_mode = st.radio("Profil", options=list(WEIGHTS))
    analyze = st.button("Analyser", type="primary", use_container_width=True)
    st.caption("Les tickers internationaux nécessitent leur suffixe de place (ex. MC.PA).")

if not ticker_input:
    st.info("Saisissez un ticker dans la barre latérale.")
elif analyze or ticker_input:
    try:
        render_dashboard(ticker_input, PERIODS[selected_period], selected_mode)
    except Exception as error:
        st.error(f"Impossible d'analyser {ticker_input} : {error}")
