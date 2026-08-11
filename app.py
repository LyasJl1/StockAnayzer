# -*- coding: utf-8 -*-
"""Assistant Streamlit d'analyse multifactorielle d'actions."""
"""Assistant Streamlit d'analyse multifactorielle d'actions."""
"""Tableau de bord Streamlit d'analyse fondamentale et technique d'actions."""

from __future__ import annotations

from html import escape
from math import isfinite, sqrt
from math import isfinite
from datetime import datetime, timezone
from html import escape
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf


PERIODS = {"6 mois": "6m", "1 an": "1y", "2 ans": "2y", "5 ans": "5y"}
CURRENCY_SYMBOLS = {"USD": "$", "EUR": "€", "GBP": "£", "JPY": "¥", "CHF": "CHF", "CAD": "C$"}
WEIGHTS = {
    "Investisseur": {"Fondamentaux": 25, "Valorisation": 20, "Croissance": 20, "Technique": 20, "Risque": 15},
    "Trader / Swing": {"Fondamentaux": 15, "Valorisation": 10, "Croissance": 15, "Technique": 40, "Risque": 20},
}
HELP = {
    "P/E": "Prix rapporté au bénéfice par action. À comparer avec le secteur et la croissance.",
    "Marge nette": "Part du chiffre d'affaires conservée en bénéfice net.",
    "MM50": "Moyenne des 50 dernières clôtures, indicateur de tendance intermédiaire.",
    "MM200": "Moyenne des 200 dernières clôtures, indicateur de tendance longue.",
    "RSI": "Oscillateur de momentum sur 14 séances : surachat au-dessus de 70, survente sous 30.",
    "MACD": "Écart entre les moyennes exponentielles 12 et 26 séances, comparé à son signal.",
    "Beta": "Sensibilité historique du titre aux mouvements du marché.",
    "Volatilité": "Amplitude annualisée des variations quotidiennes passées.",
    "Free Cash Flow": "Trésorerie générée après les dépenses d'investissement.",
    "ROE": "Rentabilité des capitaux propres engagés par les actionnaires.",
}

st.set_page_config(page_title="Stock Analyzer & Trading Assistant", page_icon="📈", layout="wide")


def apply_styles() -> None:
    st.markdown("""
    <style>
    .stApp { background:#0b1220; color:#e5e7eb; } [data-testid="stSidebar"] { background:#111827; }
    [data-testid="stMetric"] { background:#111c2f; border:1px solid #25334a; border-radius:12px; padding:14px; }
    .meta,.muted { color:#94a3b8; } .meta { margin-top:-.7rem; margin-bottom:1rem; }
    .decision { background:linear-gradient(135deg,#12233c,#111827); border:1px solid #315078; border-radius:16px; padding:20px; }
    .score { font-size:clamp(2.2rem,5vw,3.5rem); font-weight:800; line-height:1; color:#f8fafc; }
    .verdict { font-size:1.1rem; font-weight:800; letter-spacing:.04em; margin:.5rem 0; }
    .cards { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:12px; align-items:stretch; }
    .card,.analysis-card { background:#111c2f; border:1px solid #25334a; border-radius:14px; padding:16px; }
    .analysis-card { min-height:100%; box-sizing:border-box; }
    .card h3,.analysis-card h3 { margin:0 0 12px; font-size:1.05rem; }
    .card ul { padding-left:1.15rem; margin:.3rem 0; }
    .card { background:#111c2f; border:1px solid #25334a; border-radius:14px; padding:16px; }
    .card h3 { margin:0 0 12px; font-size:1.05rem; } .card ul { padding-left:1.15rem; margin:.3rem 0; }
    .card li { margin:.55rem 0; line-height:1.4; } .good { color:#34d399; } .warn { color:#fb923c; }
    .bad { color:#f87171; } .info { color:#93c5fd; }
    .bar-row { display:grid; grid-template-columns:120px 1fr 62px; gap:10px; align-items:center; margin:9px 0; }
    .bar-track { height:8px; background:#25334a; border-radius:99px; overflow:hidden; }
    .bar-fill { height:100%; border-radius:99px; }
    .trade-row { display:flex; justify-content:space-between; gap:12px; border-bottom:1px solid #25334a; padding:8px 0; }
    .trade-row:last-child { border:0; } .value { font-weight:750; text-align:right; }
    .badge { display:inline-block; padding:7px 10px; border-radius:8px; font-weight:800; background:#17243a; }
    .compact { font-size:.78rem; line-height:1.4; } .disclaimer { color:#94a3b8; font-size:.8rem; }
    @media(max-width:768px){ .cards{grid-template-columns:1fr}.bar-row{grid-template-columns:95px 1fr 55px}.decision{padding:15px} }
    </style>""", unsafe_allow_html=True)


def valid_number(value: Any) -> bool:
    try:
        return value is not None and bool(pd.notna(value)) and isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
CURRENCY_SYMBOLS = {"USD": "$", "EUR": "€", "GBP": "£", "JPY": "¥", "CNY": "¥", "INR": "₹", "KRW": "₩"}


st.set_page_config(
    page_title="Stock Analyzer & Trading Assistant",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


def apply_styles() -> None:
    """Applique une identité visuelle sombre sans dépendance externe."""
    st.markdown(
        """
        <style>
        .stApp { background: #0b1220; color: #e5e7eb; }
        [data-testid="stSidebar"] { background: #111827; }
        [data-testid="stMetric"] {
            background: #111c2f; border: 1px solid #25334a; border-radius: 12px;
            padding: 16px 18px; min-height: 125px;
        }
        [data-testid="stMetricValue"] { color: #f8fafc; }
        .company-meta { color: #94a3b8; margin-top: -0.7rem; margin-bottom: 1.3rem; }
        .analysis-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr));
                         gap: 12px; align-items: stretch; }
        .analysis-card {
            background: #111c2f; border: 1px solid #25334a; border-radius: 14px;
            padding: 18px; min-height: 370px; box-sizing: border-box;
            display: flex; flex-direction: column; gap: 12px;
        }
        .analysis-card h3 { color: #f8fafc; font-size: 1.1rem; margin: 0; }
        .section-label { color: #94a3b8; font-size: .72rem; font-weight: 700;
                         letter-spacing: .09em; margin-bottom: 3px; text-transform: uppercase; }
        .headline-price { color: #f8fafc; font-size: clamp(1.75rem, 3vw, 2.35rem);
                          font-weight: 750; line-height: 1.05; }
        .badge { border-radius: 8px; font-weight: 800; letter-spacing: .04em;
                 padding: 9px 11px; display: block; width: fit-content; }
        .bull, .positive { color: #34d399; background: rgba(52, 211, 153, .10); }
        .bear, .warning { color: #fb923c; background: rgba(251, 146, 60, .10); }
        .neutral { color: #93c5fd; background: rgba(147, 197, 253, .10); }
        .detail-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr));
                       gap: 8px; }
        .detail-item { background: #0d1728; border-radius: 9px; padding: 10px 11px; }
        .detail-label { color: #94a3b8; display: block; font-size: .73rem; margin-bottom: 3px; }
        .detail-value { color: #e5e7eb; font-size: .98rem; font-weight: 700; }
        .fundamental-card { background: #0d1728; border: 1px solid #25334a;
                            border-radius: 10px; padding: 12px; }
        .fundamental-card .badge { margin-top: 7px; }
        .trade-row { align-items: center; border-bottom: 1px solid #25334a;
                     display: flex; justify-content: space-between; gap: 12px; padding: 8px 0; }
        .trade-row:last-of-type { border-bottom: 0; }
        .trade-label { color: #94a3b8; font-size: .86rem; }
        .trade-value { color: #f8fafc; font-weight: 750; text-align: right; }
        .stop { color: #f87171; } .target { color: #34d399; }
        .compact-note { color: #94a3b8; font-size: .73rem; line-height: 1.35;
                        margin: auto 0 0; }
        .disclaimer { color: #94a3b8; font-size: .82rem; }
        @media (max-width: 768px) {
            .analysis-grid { grid-template-columns: 1fr; }
            .analysis-card { min-height: auto; padding: 15px; }
            .detail-grid { grid-template-columns: 1fr; }
            .headline-price { font-size: 1.8rem; }
        }
        .panel { background: #111c2f; border: 1px solid #25334a; border-radius: 14px;
                 padding: 20px; margin-bottom: 14px; }
        .badge { border-radius: 8px; font-weight: 800; letter-spacing: .04em;
                 padding: 10px 12px; margin: 8px 0 15px; display: inline-block; }
        .bull, .positive { color: #34d399; background: rgba(52, 211, 153, .10); }
        .bear, .warning { color: #fb923c; background: rgba(251, 146, 60, .10); }
        .neutral { color: #93c5fd; background: rgba(147, 197, 253, .10); }
        .stop { color: #f87171; } .target { color: #34d399; }
        .disclaimer { color: #94a3b8; font-size: .82rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def valid_number(value: Any) -> bool:
    """Indique si une valeur peut être affichée comme nombre fini."""
    try:
        return value is not None and bool(pd.notna(value)) and isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return value is not None and pd.notna(value) and float(value) not in (float("inf"), float("-inf"))
    except (TypeError, ValueError):
        return False


def format_price(value: Any, currency: str | None) -> str:
    if not valid_number(value): return "N/A"
    code = (currency or "").upper(); symbol = CURRENCY_SYMBOLS.get(code)
    return f"{symbol}{float(value):,.2f}" if symbol else f"{float(value):,.2f} {code}".strip()


def format_percentage(value: Any, signed: bool = False) -> str:
    if not valid_number(value): return "N/A"
    return f"{float(value):+,.1%}" if signed else f"{float(value):,.1%}"
    if not valid_number(value):
        return "N/A"
    code = (currency or "").upper()
    symbol = CURRENCY_SYMBOLS.get(code)
    return f"{symbol}{float(value):,.2f}" if symbol else f"{float(value):,.2f} {code}".strip()


def format_percentage(value: Any, *, signed: bool = False) -> str:
    """Formate une valeur décimale (0,10 = 10 %)."""
    if not valid_number(value):
        return "N/A"
    return f"{float(value):+,.2%}" if signed else f"{float(value):,.2%}"


def format_ratio(value: Any) -> str:
    return f"{float(value):,.2f}x" if valid_number(value) else "N/A"


def format_number(value: Any, decimals: int = 1) -> str:
    """Formate un indicateur brut sans jamais exposer NaN ou une valeur infinie."""
    return f"{float(value):,.{decimals}f}" if valid_number(value) else "N/A"


def average_available(values: list[float | None]) -> tuple[float | None, int, int]:
    available = [max(0.0, min(100.0, float(v))) for v in values if valid_number(v)]
    return (sum(available) / len(available) if available else None, len(available), len(values))


def tier(value: float | None) -> tuple[str, str, str]:
    if not valid_number(value): return "N/A", "#64748b", "⚪"
    if value >= 75: return "Très favorable", "#34d399", "🟢"
    if value >= 55: return "Favorable", "#86efac", "🟢"
    if value >= 40: return "Neutre", "#fb923c", "🟠"
    if value >= 25: return "Prudence", "#f87171", "🔴"
    return "Défavorable", "#ef4444", "🔴"


def _annual_growth(statement: pd.DataFrame, labels: tuple[str, ...]) -> float | None:
    if statement is None or statement.empty: return None
    for label in labels:
        if label in statement.index:
            series = pd.to_numeric(statement.loc[label], errors="coerce").dropna().sort_index(ascending=False)
            if len(series) >= 2 and float(series.iloc[1]) != 0:
                return float(series.iloc[0]) / float(series.iloc[1]) - 1
    return None


@st.cache_data(ttl=900, show_spinner=False)
def load_stock_data(symbol: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    stock = yf.Ticker(symbol)
    history = stock.history(period="5y", interval="1d", auto_adjust=False, actions=False)
    if history is None or history.empty or "Close" not in history: raise ValueError("Historique indisponible")
    history = history.copy(); history["Close"] = pd.to_numeric(history["Close"], errors="coerce")
    history = history[history["Close"].map(valid_number)].sort_index()
    if history.empty: raise ValueError("Clôtures indisponibles")
    try: info = stock.info or {}
    except Exception: info = {}
    try: financials = stock.financials
    except Exception: financials = pd.DataFrame()
    annual_revenue = _annual_growth(financials, ("Total Revenue", "Operating Revenue"))
    annual_earnings = _annual_growth(financials, ("Net Income", "Net Income Common Stockholders"))
    data = {
        "name": info.get("longName") or info.get("shortName") or symbol,
        "sector": info.get("sector") or "N/A", "country": info.get("country") or "N/A",
        "currency": info.get("currency") or "", "pe": info.get("trailingPE"),
        "forward_pe": info.get("forwardPE"), "price_to_book": info.get("priceToBook"),
        "peg": info.get("trailingPegRatio") or info.get("pegRatio"), "net_margin": info.get("profitMargins"),
        "roe": info.get("returnOnEquity"), "free_cash_flow": info.get("freeCashflow"),
        "operating_cash_flow": info.get("operatingCashflow"), "debt_to_equity": info.get("debtToEquity"),
        "revenue_growth": annual_revenue if valid_number(annual_revenue) else info.get("revenueGrowth"),
        "earnings_growth": annual_earnings if valid_number(annual_earnings) else info.get("earningsGrowth"),
        "beta": info.get("beta"),
    }
    return history, data


def calculate_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    change = close.diff(); gain = change.clip(lower=0); loss = -change.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, pd.NA)
    rsi = 100 - 100 / (1 + rs)
    return rsi.mask((avg_loss == 0) & (avg_gain > 0), 100).mask((avg_loss == 0) & (avg_gain == 0), 50)


def calculate_macd(close: pd.Series) -> tuple[pd.Series, pd.Series]:
    macd = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    return macd, macd.ewm(span=9, adjust=False).mean()


def calculate_volatility(close: pd.Series) -> float | None:
    returns = close.pct_change(fill_method=None).replace([float("inf"), float("-inf")], pd.NA).dropna()
    value = returns.std() * sqrt(252) if len(returns) > 1 else None
    return float(value) if valid_number(value) else None


def enrich_history(history: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = history.copy(); close = frame["Close"]
    frame["MM50"] = close.rolling(50, min_periods=50).mean(); frame["MM200"] = close.rolling(200, min_periods=200).mean()
    frame["RSI"] = calculate_rsi(close); frame["MACD"], frame["MACD_SIGNAL"] = calculate_macd(close)
    year = close.tail(252); current = float(close.iloc[-1]); peak = close.cummax()
    values = {
        "price": current, "mm50": frame["MM50"].dropna().iloc[-1] if frame["MM50"].notna().any() else None,
        "mm200": frame["MM200"].dropna().iloc[-1] if frame["MM200"].notna().any() else None,
        "rsi": frame["RSI"].dropna().iloc[-1] if frame["RSI"].notna().any() else None,
        "macd": frame["MACD"].iloc[-1], "macd_signal": frame["MACD_SIGNAL"].iloc[-1],
        "high52": year.max() if not year.empty else None, "low52": year.min() if not year.empty else None,
        "volatility": calculate_volatility(close), "drawdown": (close / peak - 1).tail(252).min(),
    }
    for ma in ("mm50", "mm200"):
        values[f"gap_{ma}"] = current / values[ma] - 1 if valid_number(values[ma]) and values[ma] != 0 else None
    values["gap_high"] = current / values["high52"] - 1 if valid_number(values["high52"]) and values["high52"] else None
    values["gap_low"] = current / values["low52"] - 1 if valid_number(values["low52"]) and values["low52"] else None
    return frame, values


def calculate_fundamental_score(f: dict[str, Any]) -> tuple[float | None, int, int]:
    margin = f["net_margin"]; roe = f["roe"]; debt = f["debt_to_equity"]
    return average_available([
        (90 if margin > .20 else 72 if margin > .10 else 52 if margin > .05 else 25) if valid_number(margin) else None,
        (90 if roe > .20 else 70 if roe > .10 else 35) if valid_number(roe) else None,
        (80 if f["free_cash_flow"] > 0 else 20) if valid_number(f["free_cash_flow"]) else None,
        (80 if f["operating_cash_flow"] > 0 else 20) if valid_number(f["operating_cash_flow"]) else None,
        (85 if debt < 50 else 65 if debt < 100 else 40 if debt < 200 else 20) if valid_number(debt) else None,
    ])


def calculate_valuation_score(f: dict[str, Any]) -> tuple[float | None, int, int]:
    def pe_score(x: Any) -> float | None:
        if not valid_number(x) or x <= 0: return None
        return 65 if x < 10 else 80 if x < 20 else 65 if x < 30 else 40 if x < 50 else 20
    pb, peg = f["price_to_book"], f["peg"]
    return average_available([pe_score(f["pe"]), pe_score(f["forward_pe"]),
        (80 if 0 < pb < 2 else 60 if pb < 4 else 30) if valid_number(pb) and pb > 0 else None,
        (80 if 0 < peg <= 1.5 else 55 if peg <= 2.5 else 30) if valid_number(peg) and peg > 0 else None])


def calculate_growth_score(f: dict[str, Any]) -> tuple[float | None, int, int]:
    def growth(x: Any) -> float | None:
        if not valid_number(x): return None
        return 90 if x > .20 else 75 if x > .10 else 58 if x > 0 else 30 if x > -.10 else 15
    return average_available([growth(f["revenue_growth"]), growth(f["earnings_growth"])])


def calculate_technical_score(t: dict[str, Any]) -> tuple[float | None, int, int]:
    price = t["price"]; rsi = t["rsi"]
    return average_available([
        (80 if price > t["mm50"] else 30) if valid_number(t["mm50"]) else None,
        (85 if price > t["mm200"] else 25) if valid_number(t["mm200"]) else None,
        (75 if 45 <= rsi <= 65 else 55 if 30 <= rsi <= 70 else 30) if valid_number(rsi) else None,
        (80 if t["macd"] > 0 and t["macd"] > t["macd_signal"] else 55 if t["macd"] > t["macd_signal"] else 25)
        if valid_number(t["macd"]) and valid_number(t["macd_signal"]) else None])


def calculate_risk_score(f: dict[str, Any], t: dict[str, Any]) -> tuple[float | None, int, int]:
    vol, beta, dd, debt = t["volatility"], f["beta"], t["drawdown"], f["debt_to_equity"]
    return average_available([
        (85 if vol < .20 else 65 if vol < .35 else 40 if vol < .55 else 20) if valid_number(vol) else None,
        (80 if beta < .8 else 65 if beta < 1.2 else 40 if beta < 1.6 else 20) if valid_number(beta) and beta >= 0 else None,
        (85 if dd > -.10 else 65 if dd > -.20 else 40 if dd > -.35 else 20) if valid_number(dd) else None,
        (85 if debt < 50 else 60 if debt < 120 else 30) if valid_number(debt) else None,
        (65 if t["gap_high"] < -.05 else 45) if valid_number(t["gap_high"]) else None])


def calculate_global_score(scores: dict[str, tuple[float | None, int, int]], mode: str) -> tuple[float | None, float]:
    present = [(scores[name][0], WEIGHTS[mode][name]) for name in WEIGHTS[mode] if valid_number(scores[name][0])]
    global_score = sum(score * weight for score, weight in present) / sum(weight for _, weight in present) if present else None
    available = sum(item[1] for item in scores.values()); total = sum(item[2] for item in scores.values())
    return global_score, available / total * 100 if total else 0


def generate_bull_points(f: dict[str, Any], t: dict[str, Any]) -> list[str]:
    points: list[str] = []
    if valid_number(t["gap_mm200"]) and t["gap_mm200"] > 0: points.append(f"Cours {format_percentage(t['gap_mm200'])} au-dessus de la MM200 : tendance longue positive.")
    if valid_number(f["net_margin"]) and f["net_margin"] > .10: points.append(f"Marge nette de {format_percentage(f['net_margin'])}, supérieure au seuil de 10 %.")
    if valid_number(f["revenue_growth"]) and f["revenue_growth"] > 0: points.append(f"Croissance du CA de {format_percentage(f['revenue_growth'], True)} : activité en progression.")
    if valid_number(f["free_cash_flow"]) and f["free_cash_flow"] > 0: points.append(f"Free Cash Flow positif de {f['free_cash_flow']:,.0f} {f['currency']}.")
    if valid_number(f["pe"]) and 0 < f["pe"] < 25: points.append(f"P/E de {f['pe']:.1f}x : niveau modéré selon cette seule métrique.")
    if valid_number(t["rsi"]) and 50 <= t["rsi"] <= 70: points.append(f"RSI à {t['rsi']:.1f} : momentum positif hors zone de surachat.")
    return points[:5]


def generate_bear_points(f: dict[str, Any], t: dict[str, Any]) -> list[str]:
    points: list[str] = []
    if valid_number(t["gap_mm200"]) and t["gap_mm200"] < 0: points.append(f"Cours {format_percentage(abs(t['gap_mm200']))} sous la MM200 : tendance longue dégradée.")
    if valid_number(f["revenue_growth"]) and f["revenue_growth"] < 0: points.append(f"Croissance du CA de {format_percentage(f['revenue_growth'], True)} : activité en contraction.")
    if valid_number(f["net_margin"]) and f["net_margin"] <= .10: points.append(f"Marge nette de {format_percentage(f['net_margin'])}, sous le seuil de 10 %.")
    if valid_number(t["rsi"]) and t["rsi"] > 70: points.append(f"RSI à {t['rsi']:.1f} : titre en zone de surachat.")
    if valid_number(t["volatility"]) and t["volatility"] > .35: points.append(f"Volatilité annualisée de {format_percentage(t['volatility'])} : fluctuations historiques élevées.")
    if valid_number(f["debt_to_equity"]) and f["debt_to_equity"] > 150: points.append(f"Dette/capitaux propres de {f['debt_to_equity']:.0f} %, niveau élevé.")
    return points[:5]


def generate_summary(name: str, mode: str, scores: dict[str, tuple[float | None, int, int]], t: dict[str, Any]) -> str:
    parts = [f"En mode {mode}, {name} présente"]
    parts.append("une qualité fondamentale favorable" if valid_number(scores["Fondamentaux"][0]) and scores["Fondamentaux"][0] >= 55 else "des fondamentaux à examiner")
    if valid_number(scores["Croissance"][0]): parts.append("une croissance dynamique" if scores["Croissance"][0] >= 55 else "une croissance à surveiller")
    if valid_number(t["gap_mm200"]): parts.append("une tendance longue positive" if t["gap_mm200"] > 0 else "une tendance longue dégradée")
    return ", ".join(parts[:-1]) + (" et " + parts[-1] if len(parts) > 1 else "") + ". Ce profil déterministe décrit les données disponibles, sans constituer une recommandation."


def filter_period(history: pd.DataFrame, period: str) -> pd.DataFrame:
    offsets = {"6m": pd.DateOffset(months=6), "1y": pd.DateOffset(years=1), "2y": pd.DateOffset(years=2), "5y": pd.DateOffset(years=5)}
    return history.loc[history.index >= history.index.max() - offsets[period]].copy()


def build_chart(data: pd.DataFrame, symbol: str, currency: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data["Close"], name="Clôture", line={"color":"#60a5fa","width":2}))
    fig.add_trace(go.Scatter(x=data.index, y=data["MM50"], name="MM50", line={"color":"#34d399","width":1.5}))
    fig.add_trace(go.Scatter(x=data.index, y=data["MM200"], name="MM200", line={"color":"#f59e0b","width":2,"dash":"dash"}))
    fig.update_layout(template="plotly_dark", title=f"Évolution de {symbol}", hovermode="x unified", xaxis_title="Date", yaxis_title=f"Prix ({currency or 'devise locale'})", paper_bgcolor="#0b1220", plot_bgcolor="#0b1220", height=500, margin={"l":30,"r":20,"t":55,"b":30}, legend={"orientation":"h","y":1.08})
    return fig


def render_score_rows(scores: dict[str, tuple[float | None, int, int]]) -> None:
    rows = []
    for name, (score, _, _) in scores.items():
        label, color, icon = tier(score); width = score if valid_number(score) else 0
        rows.append(f'<div class="bar-row"><span>{name}</span><div class="bar-track"><div class="bar-fill" style="width:{width:.0f}%;background:{color}"></div></div><span>{icon} {score:.0f}</span></div>' if valid_number(score) else f'<div class="bar-row"><span>{name}</span><div class="bar-track"></div><span>⚪ N/A</span></div>')
    st.markdown('<div class="card">' + ''.join(rows) + '</div>', unsafe_allow_html=True)


def render_dashboard(symbol: str, period: str, mode: str) -> None:
    with st.spinner(f"Analyse de {symbol}…"): history, f = load_stock_data(symbol)
    history, t = enrich_history(history); f["currency"] = f.get("currency") or ""
    previous = float(history["Close"].iloc[-2]) if len(history) > 1 else None
    daily = t["price"] / previous - 1 if valid_number(previous) and previous != 0 else None
    scores = {"Fondamentaux": calculate_fundamental_score(f), "Valorisation": calculate_valuation_score(f), "Croissance": calculate_growth_score(f), "Technique": calculate_technical_score(t), "Risque": calculate_risk_score(f, t)}
    global_score, confidence = calculate_global_score(scores, mode); verdict, verdict_color, verdict_icon = tier(global_score)
    bulls, bears = generate_bull_points(f, t), generate_bear_points(f, t)

    st.header(str(f["name"])); st.markdown(f'<div class="meta">{escape(symbol)} • {escape(str(f["sector"]))} • {escape(str(f["country"]))}</div>', unsafe_allow_html=True)
    cols = st.columns(4); cols[0].metric("Prix actuel", format_price(t["price"], f["currency"]), format_percentage(daily, True), help="Dernière clôture disponible.")
    cols[1].metric("P/E", format_ratio(f["pe"]), help=HELP["P/E"]); cols[2].metric("Marge nette", format_percentage(f["net_margin"]), help=HELP["Marge nette"]); cols[3].metric("Croissance du CA", format_percentage(f["revenue_growth"], True))

    st.subheader("Analyse & Aide à la décision")
    decision_html = (
        f"""<div class="decision">
        <div class="muted">Score global pondéré • Mode {escape(mode)}</div>
        <div class="score">{global_score:.0f} / 100</div>
        <div class="verdict" style="color:{verdict_color}">
            {verdict_icon} Profil global : {verdict.upper()}
        </div>
        <p>{escape(generate_summary(str(f["name"]), mode, scores, t))}</p>
        <div class="muted">Confiance de l'analyse : <b>{confidence:.0f} %</b>
            ({sum(x[1] for x in scores.values())}/{sum(x[2] for x in scores.values())}
            indicateurs disponibles)
        </div></div>"""
        if valid_number(global_score)
        else '<div class="decision">Données insuffisantes pour établir un score.</div>'
    )
    st.markdown(decision_html, unsafe_allow_html=True)
    st.markdown(f'<div class="decision"><div class="muted">Score global pondéré • Mode {escape(mode)}</div><div class="score">{global_score:.0f} / 100</div><div class="verdict" style="color:{verdict_color}">{verdict_icon} Profil global : {verdict.upper()}</div><p>{escape(generate_summary(str(f["name"]), mode, scores, t))}</p><div class="muted">Confiance de l’analyse : <b>{confidence:.0f} %</b> ({sum(x[1] for x in scores.values())}/{sum(x[2] for x in scores.values())} indicateurs disponibles)</div></div>' if valid_number(global_score) else '<div class="decision">Données insuffisantes pour établir un score.</div>', unsafe_allow_html=True)
    acols = st.columns(2)
    with acols[0]: st.markdown('<div class="card"><h3 class="good">✅ Arguments favorables</h3><ul>' + ''.join(f'<li>{escape(x)}</li>' for x in bulls) + ('<li class="muted">Aucun argument favorable vérifiable détecté.</li>' if not bulls else '') + '</ul></div>', unsafe_allow_html=True)
    with acols[1]: st.markdown('<div class="card"><h3 class="warn">⚠️ Risques / Vigilances</h3><ul>' + ''.join(f'<li>{escape(x)}</li>' for x in bears) + ('<li class="muted">Aucun risque spécifique détecté parmi les données disponibles.</li>' if not bears else '') + '</ul></div>', unsafe_allow_html=True)

    st.plotly_chart(build_chart(filter_period(history, period), symbol, f["currency"]), use_container_width=True, config={"displaylogo":False,"scrollZoom":True})
    st.subheader("Matrice de décision"); render_score_rows(scores)
    st.subheader("Diagnostic technique & Plan de Trading")
    stop = t["price"] * .93; target = t["price"] * 1.15; ratio = (target - t["price"]) / (t["price"] - stop)
    tech_text = "Le cours évolue au-dessus de ses MM50 et MM200." if valid_number(t["mm50"]) and valid_number(t["mm200"]) and t["price"] > t["mm50"] > t["mm200"] else "La structure MM50/MM200 ne confirme pas une tendance haussière complète."
    if valid_number(t["rsi"]) and t["rsi"] > 70: tech_text += " Le RSI élevé signale une zone de surachat."
    left, right = st.columns(2, gap="small")
    with left:
        st.markdown(
            f"""
            <div class="analysis-card">
            <div class="card">
                <h3>Lecture technique</h3>
                <p>{tech_text}</p>
                <div class="trade-row"><span>MM50 ⓘ</span><span class="value">
                    {format_price(t["mm50"], f["currency"])}
                    ({format_percentage(t["gap_mm50"], True)})</span></div>
                <div class="trade-row"><span>MM200 ⓘ</span><span class="value warn">
                    {format_price(t["mm200"], f["currency"])}
                    ({format_percentage(t["gap_mm200"], True)})</span></div>
                <div class="trade-row"><span>RSI 14 ⓘ</span>
                    <span class="value">{format_number(t["rsi"])}</span></div>
                <div class="trade-row"><span>MACD / signal ⓘ</span><span class="value">
                    {format_number(t["macd"], 2)} / {format_number(t["macd_signal"], 2)}</span></div>
                <div class="trade-row"><span>Plus haut / bas 52 sem.</span><span class="value">
                    {format_price(t["high52"], f["currency"])} /
                    {format_price(t["low52"], f["currency"])}</span></div>
                <div class="trade-row"><span>Volatilité annualisée ⓘ</span>
                    <span class="value">{format_percentage(t["volatility"])}</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        st.markdown(
            f"""
            <div class="analysis-card">
            <div class="card">
                <h3>Plan de Trading Suggéré</h3>
                <div class="trade-row"><span>Entrée théorique</span>
                    <span class="value">{format_price(t["price"], f["currency"])}</span></div>
                <div class="trade-row"><span>Stop-Loss (-7 %)</span>
                    <span class="value bad">{format_price(stop, f["currency"])}</span></div>
                <div class="trade-row"><span>Take-Profit (+15 %)</span>
                    <span class="value good">{format_price(target, f["currency"])}</span></div>
                <div class="trade-row"><span>Risque / récompense</span>
                    <span class="value good">1 : {ratio:.2f}</span></div>
                <p class="compact muted">Niveaux mécaniques indicatifs, sans prise en compte
                    de votre profil ni des frais.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    cards = st.columns(2)
    with cards[0]:
        st.markdown(f'<div class="card"><h3>Lecture technique</h3><p>{tech_text}</p><div class="trade-row"><span>MM50 ⓘ</span><span class="value">{format_price(t["mm50"],f["currency"])} ({format_percentage(t["gap_mm50"],True)})</span></div><div class="trade-row"><span>MM200 ⓘ</span><span class="value warn">{format_price(t["mm200"],f["currency"])} ({format_percentage(t["gap_mm200"],True)})</span></div><div class="trade-row"><span>RSI 14 ⓘ</span><span class="value">{format_number(t["rsi"])}</span></div><div class="trade-row"><span>MACD / signal ⓘ</span><span class="value">{format_number(t["macd"],2)} / {format_number(t["macd_signal"],2)}</span></div><div class="trade-row"><span>Plus haut / bas 52 sem.</span><span class="value">{format_price(t["high52"],f["currency"])} / {format_price(t["low52"],f["currency"])}</span></div><div class="trade-row"><span>Volatilité annualisée ⓘ</span><span class="value">{format_percentage(t["volatility"])}</span></div></div>', unsafe_allow_html=True)
    with cards[1]:
        st.markdown(f'<div class="card"><h3>Plan de Trading Suggéré</h3><div class="trade-row"><span>Entrée théorique</span><span class="value">{format_price(t["price"],f["currency"])}</span></div><div class="trade-row"><span>Stop-Loss (-7 %)</span><span class="value bad">{format_price(stop,f["currency"])}</span></div><div class="trade-row"><span>Take-Profit (+15 %)</span><span class="value good">{format_price(target,f["currency"])}</span></div><div class="trade-row"><span>Risque / récompense</span><span class="value good">1 : {ratio:.2f}</span></div><p class="compact muted">Niveaux mécaniques indicatifs, sans prise en compte de votre profil ni des frais.</p></div>', unsafe_allow_html=True)
    with st.expander("Méthodologie et définitions"):
        st.write("Les scores utilisent uniquement les indicateurs disponibles ; les pondérations restantes sont automatiquement renormalisées. Un score de risque élevé signifie un risque historiquement mieux maîtrisé.")
        for key, explanation in HELP.items(): st.markdown(f"**{key}** — {explanation}")
        st.caption(f"Pondérations {mode} : " + ", ".join(f"{k} {v} %" for k, v in WEIGHTS[mode].items()))


def main() -> None:
    apply_styles(); st.title("📈 Stock Analyzer & Trading Assistant")
    st.markdown('<p class="disclaimer">⚠️ Analyse informative et déterministe — elle ne constitue pas un conseil financier.</p>', unsafe_allow_html=True)
    with st.sidebar:
        st.header("Paramètres"); raw = st.text_input("Ticker", "AAPL"); period_label = st.selectbox("Période affichée", list(PERIODS), 1)
        mode = st.selectbox("Type d'analyse", ["Investisseur", "Trader / Swing"], help="Adapte les pondérations du score global.")
        analyze = st.button("Lancer l'analyse", type="primary", use_container_width=True)
        mode = st.selectbox("Type d’analyse", ["Investisseur", "Trader / Swing"], help="Adapte les pondérations du score global.")
        analyze = st.button("Lancer l’analyse", type="primary", use_container_width=True)
    symbol = "".join(raw.split()).upper()
    if analyze:
        if not symbol: st.error("Veuillez saisir un ticker valide.")
        else: load_stock_data.clear(); st.session_state["analysis"] = {"symbol":symbol,"period":PERIODS[period_label],"mode":mode}
    analysis = st.session_state.get("analysis")
    if not analysis: st.info("Saisissez un ticker puis cliquez sur **Lancer l'analyse**."); return
    try: render_dashboard(analysis["symbol"], analysis["period"], analysis["mode"])
    except (ValueError, KeyError, IndexError, TypeError, ZeroDivisionError): st.error("Impossible d'analyser ce ticker. Vérifiez le symbole et les données disponibles.")
    except Exception: st.error("Impossible d'analyser ce ticker. Vérifiez votre connexion Internet.")


if __name__ == "__main__": main()
    if not analysis: st.info("Saisissez un ticker puis cliquez sur **Lancer l’analyse**."); return
    try: render_dashboard(analysis["symbol"], analysis["period"], analysis["mode"])
    except (ValueError, KeyError, IndexError, TypeError, ZeroDivisionError): st.error("Impossible d’analyser ce ticker. Vérifiez le symbole et les données disponibles.")
    except Exception: st.error("Impossible d’analyser ce ticker. Vérifiez votre connexion Internet.")


if __name__ == "__main__": main()
def _annual_revenue_growth(financials: pd.DataFrame) -> float | None:
    """Calcule la croissance entre les deux exercices annuels les plus récents."""
    if financials is None or financials.empty:
        return None
    revenue = None
    for label in ("Total Revenue", "Operating Revenue"):
        if label in financials.index:
            revenue = pd.to_numeric(financials.loc[label], errors="coerce").dropna()
            break
    if revenue is None or len(revenue) < 2:
        return None
    revenue = revenue.sort_index(ascending=False)
    latest, previous = float(revenue.iloc[0]), float(revenue.iloc[1])
    return (latest / previous) - 1 if previous != 0 else None


@st.cache_data(ttl=900, show_spinner=False)
def load_stock_data(ticker_symbol: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Charge cinq ans de cours et les fondamentaux, avec défaillances isolées."""
    stock = yf.Ticker(ticker_symbol)
    history = stock.history(period="5y", interval="1d", auto_adjust=False, actions=False)
    if history is None or history.empty or "Close" not in history.columns:
        raise ValueError("Aucun historique de prix disponible.")

    history = history.copy()
    history["Close"] = pd.to_numeric(history["Close"], errors="coerce")
    history.loc[~history["Close"].map(isfinite), "Close"] = pd.NA
    history = history.dropna(subset=["Close"]).sort_index()
    if history.empty:
        raise ValueError("Les cours de clôture sont indisponibles.")

    try:
        info = stock.info or {}
    except Exception:
        info = {}
    try:
        annual_growth = _annual_revenue_growth(stock.financials)
    except Exception:
        annual_growth = None

    fundamentals = {
        "name": info.get("longName") or info.get("shortName") or ticker_symbol,
        "sector": info.get("sector") or "N/A",
        "country": info.get("country") or "N/A",
        "currency": info.get("currency") or history.attrs.get("currency") or "",
        "pe": info.get("trailingPE"),
        "net_margin": info.get("profitMargins"),
        "revenue_growth": annual_growth if valid_number(annual_growth) else info.get("revenueGrowth"),
    }
    return history, fundamentals


def filter_period(history: pd.DataFrame, period: str) -> pd.DataFrame:
    """Filtre uniquement l'affichage, après calcul des indicateurs longs."""
    offsets = {"6m": pd.DateOffset(months=6), "1y": pd.DateOffset(years=1),
               "2y": pd.DateOffset(years=2), "5y": pd.DateOffset(years=5)}
    cutoff = history.index.max() - offsets[period]
    return history.loc[history.index >= cutoff].copy()


def get_fundamental_diagnostic(margin: Any, pe: Any) -> tuple[str, str]:
    """Retourne un diagnostic pédagogique fondé sur deux règles simples."""
    if not valid_number(margin) or not valid_number(pe):
        return "DONNÉES FONDAMENTALES INCOMPLÈTES", "neutral"
    score = int(float(margin) > 0.10) + int(0 < float(pe) < 25)
    if score == 2:
        return "FONDAMENTAUX FAVORABLES", "positive"
    if score == 1:
        return "FONDAMENTAUX MITIGÉS", "warning"
    return "FONDAMENTAUX À SURVEILLER", "warning"


def build_chart(data: pd.DataFrame, ticker_symbol: str, currency: str) -> go.Figure:
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=data.index, y=data["Close"], name="Cours de clôture",
                                line={"color": "#60a5fa", "width": 2}, hovertemplate="%{x|%d/%m/%Y}<br>%{y:,.2f}<extra></extra>"))
    figure.add_trace(go.Scatter(x=data.index, y=data["MM200"], name="MM200",
                                line={"color": "#f59e0b", "width": 2, "dash": "dash"},
                                hovertemplate="%{x|%d/%m/%Y}<br>%{y:,.2f}<extra></extra>"))
    figure.update_layout(
        template="plotly_dark", title=f"Évolution de {ticker_symbol}", hovermode="x unified",
        xaxis_title="Date", yaxis_title=f"Prix ({currency or 'devise locale'})",
        paper_bgcolor="#0b1220", plot_bgcolor="#0b1220", height=520,
        margin={"l": 30, "r": 20, "t": 60, "b": 30}, legend={"orientation": "h", "y": 1.08},
    )
    figure.update_xaxes(rangeslider_visible=False, showgrid=True, gridcolor="#1f2937")
    figure.update_yaxes(showgrid=True, gridcolor="#1f2937")
    return figure


def render_dashboard(ticker_symbol: str, period: str) -> None:
    """Orchestre les calculs et le rendu du tableau de bord."""
    with st.spinner(f"Analyse de {ticker_symbol} en cours…"):
        history, fundamentals = load_stock_data(ticker_symbol)

    # La MM200 est calculée avant le filtre visuel sur cinq ans de cours.
    history = history.copy()
    history["MM200"] = history["Close"].rolling(window=200, min_periods=200).mean()
    current_price = float(history["Close"].iloc[-1])
    previous_close = float(history["Close"].iloc[-2]) if len(history) >= 2 else None
    daily_change = (
        current_price / previous_close - 1
        if valid_number(previous_close) and previous_close != 0
        else None
    )
    daily_change = (current_price / float(history["Close"].iloc[-2]) - 1) if len(history) >= 2 else None
    latest_ma = history["MM200"].dropna()
    ma200 = float(latest_ma.iloc[-1]) if not latest_ma.empty else None
    ma_gap = (current_price / ma200 - 1) if valid_number(ma200) and ma200 != 0 else None
    visible = filter_period(history, period)
    currency = fundamentals["currency"]

    st.header(str(fundamentals["name"]))
    st.markdown(
        f'<div class="company-meta">{escape(ticker_symbol)} • {escape(str(fundamentals["sector"]))} • {escape(str(fundamentals["country"]))}</div>',
        unsafe_allow_html=True,
    )

    metric_cols = st.columns(4)
    metric_cols[0].metric("Prix actuel", format_price(current_price, currency), format_percentage(daily_change, signed=True))
    metric_cols[1].metric("P/E", format_ratio(fundamentals["pe"]))
    metric_cols[2].metric("Marge nette", format_percentage(fundamentals["net_margin"]))
    metric_cols[3].metric("Croissance du CA", format_percentage(fundamentals["revenue_growth"]))

    st.plotly_chart(build_chart(visible, ticker_symbol, currency), use_container_width=True, config={"displaylogo": False, "scrollZoom": True})
    if not valid_number(ma200):
        st.warning("Historique insuffisant pour calculer une MM200 fiable (200 séances requises).")

    st.subheader("Diagnostic & Plan de Trading")
    if valid_number(ma200):
        bullish = current_price > float(ma200)
        trend, trend_class = (
            ("TENDANCE HAUSSIÈRE", "bull")
            if bullish
            else ("TENDANCE BAISSIÈRE", "bear")
        )
    else:
        trend, trend_class = "TENDANCE INDÉTERMINÉE", "neutral"

    diagnostic, diagnostic_class = get_fundamental_diagnostic(
        fundamentals["net_margin"], fundamentals["pe"]
    )
    stop_loss = current_price * 0.93
    take_profit = current_price * 1.15
    risk = current_price - stop_loss
    reward = take_profit - current_price
    risk_reward = reward / risk if risk > 0 else None
    ratio_class = (
        "positive"
        if valid_number(risk_reward) and float(risk_reward) >= 2
        else "neutral"
    )
    ratio_label = f"1 : {risk_reward:.2f}" if valid_number(risk_reward) else "N/A"

    # Un seul appel de rendu contient les deux cartes afin d'éviter qu'un ancien
    # fragment Streamlit ne réaffiche séparément le ratio ou les titres.
    st.markdown(
        f"""
        <div class="analysis-grid">
    left, right = st.columns(2, gap="small")
    with left:
        st.markdown(
            f"""
            <div class="analysis-card">
                <h3>Diagnostic</h3>
                <div>
                    <div class="section-label">Prix actuel</div>
                    <div class="headline-price">{format_price(current_price, currency)}</div>
                </div>
                <div class="badge {trend_class}">{trend}</div>
                <div class="detail-grid">
                    <div class="detail-item">
                        <span class="detail-label">MM200 actuelle</span>
                        <span class="detail-value">{format_price(ma200, currency)}</span>
                    </div>
                    <div class="detail-item">
                        <span class="detail-label">Écart à la MM200</span>
                        <span class="detail-value">{format_percentage(ma_gap, signed=True)}</span>
                    </div>
                </div>
                <div class="fundamental-card">
                    <div class="section-label">Diagnostic fondamental</div>
                    <div class="badge {diagnostic_class}">{diagnostic}</div>
                </div>
                <p class="compact-note">Lecture simplifiée fondée uniquement sur la marge nette
                (&gt; 10 %) et le P/E (entre 0 et 25) ; ce n’est pas une conclusion d’investissement.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right:
        st.markdown(
            f"""
            <div class="analysis-card">
                <h3>Plan de Trading Suggéré</h3>
                <div>
                    <div class="trade-row">
                        <span class="trade-label">Entrée théorique</span>
                        <span class="trade-value">{format_price(current_price, currency)}</span>
                    </div>
                    <div class="trade-row">
                        <span class="trade-label">Stop-Loss (-7 %)</span>
                        <span class="trade-value stop">{format_price(stop_loss, currency)}</span>
                    </div>
                    <div class="trade-row">
                        <span class="trade-label">Take-Profit (+15 %)</span>
                        <span class="trade-value target">{format_price(take_profit, currency)}</span>
                    </div>
                </div>
                <div>
                    <div class="section-label">Ratio risque / récompense</div>
                    <div class="badge {ratio_class}">{ratio_label}</div>
                </div>
                <p class="compact-note">Niveaux mécaniques et indicatifs, sans prise en compte de
                la volatilité, des frais ni de votre profil de risque.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
            """,
            unsafe_allow_html=True,
        )
    left, right = st.columns(2)
    with left:
        st.markdown('<div class="panel"><h3>Diagnostic</h3>', unsafe_allow_html=True)
        if valid_number(ma200):
            bullish = current_price > float(ma200)
            trend, css_class = ("TENDANCE HAUSSIÈRE", "bull") if bullish else ("TENDANCE BAISSIÈRE", "bear")
            st.markdown(f'<div class="badge {css_class}">{trend}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="badge neutral">TENDANCE INDÉTERMINÉE</div>', unsafe_allow_html=True)
        st.write(f"**Prix actuel :** {format_price(current_price, currency)}")
        st.write(f"**MM200 actuelle :** {format_price(ma200, currency)}")
        st.write(f"**Écart à la MM200 :** {format_percentage(ma_gap, signed=True)}")
        diagnostic, diagnostic_class = get_fundamental_diagnostic(fundamentals["net_margin"], fundamentals["pe"])
        st.markdown(f'<div class="badge {diagnostic_class}">{diagnostic}</div>', unsafe_allow_html=True)
        st.caption("Lecture simplifiée fondée uniquement sur la marge nette (> 10 %) et le P/E (entre 0 et 25) ; ce n’est pas une conclusion d’investissement.")
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        stop_loss = current_price * 0.93
        take_profit = current_price * 1.15
        risk = current_price - stop_loss
        reward = take_profit - current_price
        risk_reward = reward / risk if risk > 0 else None
        ratio_class = "positive" if valid_number(risk_reward) and float(risk_reward) >= 2 else "neutral"
        st.markdown('<div class="panel"><h3>Plan de Trading Suggéré</h3>', unsafe_allow_html=True)
        st.write(f"**Entrée théorique :** {format_price(current_price, currency)}")
        st.markdown(f'<p class="stop"><b>Stop-Loss (-7 %) :</b> {format_price(stop_loss, currency)}</p>', unsafe_allow_html=True)
        st.markdown(f'<p class="target"><b>Take-Profit (+15 %) :</b> {format_price(take_profit, currency)}</p>', unsafe_allow_html=True)
        ratio_label = f"1 : {risk_reward:.2f}" if valid_number(risk_reward) else "N/A"
        st.markdown(
            f'<div class="badge {ratio_class}">Risque / récompense&nbsp;: {ratio_label}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(f'<div class="badge {ratio_class}">Risque / récompense&nbsp;: 1 : {risk_reward:.2f}</div>', unsafe_allow_html=True)
        st.caption("Niveaux mécaniques et indicatifs, sans prise en compte de la volatilité, des frais ni de votre profil de risque.")
        st.markdown("</div>", unsafe_allow_html=True)


def main() -> None:
    apply_styles()
    st.title("📈 Stock Analyzer & Trading Assistant")
    st.markdown('<p class="disclaimer">⚠️ Outil fourni à titre informatif uniquement — il ne constitue pas un conseil financier.</p>', unsafe_allow_html=True)

    with st.sidebar:
        st.header("Paramètres")
        ticker_input = st.text_input("Ticker", value="AAPL", help="Exemples : AAPL, MSFT, TTE.PA")
        period_label = st.selectbox("Période affichée", list(PERIODS), index=1)
        analyze = st.button("Lancer l’analyse", type="primary", use_container_width=True)

    ticker_symbol = "".join(ticker_input.split()).upper()
    if analyze:
        if not ticker_symbol:
            st.error("Veuillez saisir un ticker valide.")
        else:
            # Le clic demande explicitement une actualisation, même pendant le TTL du cache.
            load_stock_data.clear()
            st.session_state["analysis"] = {
                "ticker": ticker_symbol,
                "period": PERIODS[period_label],
            }
            st.session_state["analysis"] = {"ticker": ticker_symbol, "period": PERIODS[period_label],
                                             "requested_at": datetime.now(timezone.utc).isoformat()}

    analysis = st.session_state.get("analysis")
    if not analysis:
        st.info("Saisissez un ticker puis cliquez sur **Lancer l’analyse**.")
        return

    try:
        render_dashboard(analysis["ticker"], analysis["period"])
    except (ValueError, KeyError, IndexError, TypeError, ZeroDivisionError):
        st.error("Impossible d’analyser ce ticker. Vérifiez le symbole ou les données disponibles.")
    except Exception:
        st.error("Impossible d’analyser ce ticker. Vérifiez le symbole ou votre connexion Internet.")


if __name__ == "__main__":
    main()
