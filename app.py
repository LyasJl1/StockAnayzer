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


def format_large_amount(value: Any, symbol: str) -> str:
    """Formate les montants financiers sans afficher de longues suites de chiffres."""
    if not valid_number(value):
        return "N/D"
    amount = float(value)
    absolute = abs(amount)
    for divisor, suffix in ((1_000_000_000, "Md"), (1_000_000, "M"), (1_000, "k")):
        if absolute >= divisor:
            return f"{symbol}{amount / divisor:,.2f} {suffix}"
    return format_price(amount, symbol)


def tier_score(value: Any, tiers: tuple[tuple[float, float], ...], default: float) -> float | None:
    """Retourne le score du premier seuil atteint, sans noter une donnée absente."""
    if not valid_number(value):
        return None
    number = float(value)
    for upper_bound, score in tiers:
        if number <= upper_bound:
            return score
    return default


def average_available(*scores: float | None) -> float:
    available = [score for score in scores if score is not None]
    return sum(available) / len(available) if available else 50.0


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
    """Calcule des scores gradués uniquement à partir des données disponibles."""
    del mode  # Les pondérations propres au profil sont appliquées au score global.
    close = float(data["Close"].iloc[-1])
    mm50, mm200 = data["MM50"].iloc[-1], data["MM200"].iloc[-1]
    rsi, macd, signal = data["RSI"].iloc[-1], data["MACD"].iloc[-1], data["Signal"].iloc[-1]
    pe, pb = info.get("trailingPE"), info.get("priceToBook")
    margin, roe = info.get("profitMargins"), info.get("returnOnEquity")
    revenue_growth, earnings_growth = info.get("revenueGrowth"), info.get("earningsGrowth")
    debt, beta = info.get("debtToEquity"), info.get("beta")
    free_cashflow = info.get("freeCashflow")
    volatility = data["Close"].pct_change().std() * sqrt(252)
    distance_mm50 = close / float(mm50) - 1 if valid_number(mm50) and mm50 else None
    distance_mm200 = close / float(mm200) - 1 if valid_number(mm200) and mm200 else None

    # Les maxima usuels restent sous 100 : une note parfaite exige ainsi des
    # métriques exceptionnellement favorables dans toutes les catégories.
    margin_score = tier_score(margin, ((0, 15), (0.05, 35), (0.10, 55), (0.15, 70), (0.25, 82)), 92)
    roe_score = tier_score(roe, ((0, 15), (0.08, 40), (0.12, 58), (0.20, 75), (0.30, 86)), 92)
    fcf_score = None if not valid_number(free_cashflow) else (78 if free_cashflow > 0 else 22)
    pe_score = None if not valid_number(pe) or pe <= 0 else tier_score(pe, ((10, 90), (15, 80), (20, 70), (25, 60), (35, 45), (50, 30)), 15)
    pb_score = None if not valid_number(pb) or pb <= 0 else tier_score(pb, ((1, 82), (2, 76), (3, 66), (5, 52), (8, 35)), 20)
    revenue_score = tier_score(revenue_growth, ((-0.10, 15), (0, 30), (0.05, 52), (0.10, 66), (0.20, 80), (0.35, 88)), 92)
    earnings_score = tier_score(earnings_growth, ((-0.15, 15), (0, 30), (0.05, 50), (0.10, 64), (0.20, 78), (0.40, 88)), 92)
    mm50_score = tier_score(distance_mm50, ((-0.15, 20), (-0.05, 35), (0, 48), (0.05, 65), (0.15, 78), (0.30, 70)), 55)
    mm200_score = tier_score(distance_mm200, ((-0.20, 18), (-0.08, 32), (0, 48), (0.10, 68), (0.25, 82), (0.45, 72)), 55)
    rsi_score = tier_score(rsi, ((25, 38), (35, 55), (50, 72), (65, 82), (70, 68), (80, 42)), 22)
    macd_score = None if not valid_number(macd) or not valid_number(signal) else (72 if macd > signal else 38)
    debt_score = tier_score(debt, ((20, 88), (50, 78), (100, 62), (150, 45), (250, 28)), 15)
    volatility_score = tier_score(volatility, ((0.15, 88), (0.25, 78), (0.35, 64), (0.50, 45), (0.70, 28)), 15)
    beta_score = tier_score(beta, ((0.8, 82), (1.1, 74), (1.4, 58), (1.8, 40), (2.5, 25)), 15)

    scores = {
        "Fondamentaux": average_available(margin_score, roe_score, fcf_score),
        "Valorisation": average_available(pe_score, pb_score),
        "Croissance": average_available(revenue_score, earnings_score),
        "Technique": average_available(mm50_score, mm200_score, rsi_score, macd_score),
        "Risque": average_available(debt_score, volatility_score, beta_score),
    }
    favorable: list[str] = []
    unfavorable: list[str] = []
    if valid_number(margin):
        text = f"Marge nette TTM de {format_percentage(margin)} → " + ("rentabilité supérieure au seuil de qualité de 10 %." if margin >= 0.10 else "rentabilité sous le seuil de qualité de 10 % retenu par le modèle.")
        (favorable if margin >= 0.10 else unfavorable).append(text)
    if valid_number(revenue_growth):
        text = f"Croissance du CA YoY de {format_percentage(revenue_growth)} → " + ("dynamique récente positive." if revenue_growth > 0 else "contraction récente de l'activité.")
        (favorable if revenue_growth > 0 else unfavorable).append(text)
    if valid_number(earnings_growth):
        text = f"Croissance des bénéfices YoY de {format_percentage(earnings_growth)} → " + ("progression bénéficiaire positive." if earnings_growth > 0 else "recul bénéficiaire à surveiller.")
        (favorable if earnings_growth > 0 else unfavorable).append(text)
    if distance_mm200 is not None:
        relation = "au-dessus" if distance_mm200 >= 0 else "sous"
        text = f"Cours {format_percentage(distance_mm200)} {relation} de la MM200 → " + ("tendance long terme positive." if distance_mm200 >= 0 else "tendance long terme dégradée.")
        (favorable if distance_mm200 >= 0 else unfavorable).append(text)
    if valid_number(macd) and valid_number(signal):
        text = f"MACD {float(macd):.2f} contre signal {float(signal):.2f} → " + ("momentum positif." if macd > signal else "momentum défavorable.")
        (favorable if macd > signal else unfavorable).append(text)
    if valid_number(pe):
        if pe <= 0:
            unfavorable.append(f"P/E de {float(pe):.1f}x → bénéfices négatifs ou ratio non interprétable, donc non scoré.")
        else:
            text = f"P/E TTM de {float(pe):.1f}x → " + ("valorisation modérée selon les seuils du modèle." if pe <= 25 else "valorisation exigeante selon les seuils du modèle.")
            (favorable if pe <= 25 else unfavorable).append(text)
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


def score_verdict(score: float) -> str:
    if score >= 85:
        return "Exceptionnel / Très favorable"
    if score >= 70:
        return "Favorable"
    if score >= 55:
        return "Plutôt favorable / équilibré"
    if score >= 40:
        return "Neutre / prudence"
    if score >= 25:
        return "Défavorable"
    return "Très défavorable"


def calculate_trade_levels(current: float, mode: str, method: str = "mechanical") -> tuple[float, float]:
    """Calcule le plan de sortie ; ``method`` permettra l'ajout futur de l'ATR."""
    if method != "mechanical":
        raise ValueError("Méthode de plan de trading non prise en charge.")
    stop_factor, target_factor = ((0.92, 1.16) if mode == "Investisseur" else (0.96, 1.08))
    return current * stop_factor, current * target_factor


def build_summary(
    mode: str,
    verdict: str,
    pe: Any,
    margin: Any,
    revenue_growth: Any,
    distance_mm50: float | None,
    distance_mm200: float | None,
    position_52w: float | None,
) -> str:
    """Construit une synthèse factuelle comprenant toujours force et vigilance."""
    strengths: list[str] = []
    cautions: list[str] = []
    if valid_number(pe) and 0 < pe <= 25:
        strengths.append(f"la valorisation reste modérée avec un P/E TTM de {float(pe):.1f}x")
    elif valid_number(pe):
        cautions.append(f"le P/E TTM de {float(pe):.1f}x demande une interprétation prudente")
    if distance_mm50 is not None and distance_mm200 is not None:
        if distance_mm50 > 0 and distance_mm200 > 0:
            strengths.append("la tendance technique est positive, le cours évoluant au-dessus de ses MM50 et MM200")
        elif distance_mm50 < 0 and distance_mm200 < 0:
            cautions.append("le cours reste sous ses MM50 et MM200")
    if valid_number(revenue_growth):
        if revenue_growth > 0:
            strengths.append(f"la croissance du CA YoY atteint {format_percentage(revenue_growth)}")
        else:
            cautions.append(f"la croissance du CA YoY ressort à {format_percentage(revenue_growth)}")
    if valid_number(margin) and margin < 0.10:
        cautions.append(f"la marge nette TTM de {format_percentage(margin)} reste sous le seuil de qualité de 10 %")
    if position_52w is not None and position_52w >= 0.80:
        cautions.append(f"le cours se situe à {position_52w:.0%} de sa fourchette 52 semaines, proche de son sommet annuel")
    elif position_52w is not None and position_52w <= 0.20:
        cautions.append(f"le cours se situe à seulement {position_52w:.0%} de sa fourchette 52 semaines")
    if not strengths:
        strengths.append("au moins une composante du modèle conserve un score relatif positif")
    if not cautions:
        cautions.append("les données historiques et fondamentales restent incomplètes et ne préjugent pas des performances futures")
    return (
        f"Le profil {mode.lower()} ressort « {verdict.lower()} ». "
        f"Parmi les forces, {strengths[0]}. "
        f"En revanche, {cautions[0]}. "
        "Cette lecture est descriptive et ne constitue aucune promesse de performance future."
    )


def render_dashboard(ticker: str, period: str, mode: str) -> None:
    """Affiche l'unique version du tableau de bord d'analyse."""
    with st.spinner(f"Analyse de {ticker}…"):
        history, info = load_stock_data(ticker, period)
    data = calculate_indicators(history)
    current = float(data["Close"].iloc[-1])
    previous = float(data["Close"].iloc[-2]) if len(data) > 1 else current
    daily_change = current / previous - 1 if previous else 0
    currency = CURRENCY_SYMBOLS.get(str(info.get("currency", "USD")), f"{info.get('currency', 'USD')} ")
    currency_code = str(info.get("currency") or "Devise N/D")
    exchange = str(info.get("fullExchangeName") or info.get("exchange") or "Place N/D")
    name = escape(str(info.get("longName") or info.get("shortName") or ticker))
    mm50, mm200 = data["MM50"].iloc[-1], data["MM200"].iloc[-1]
    distance_mm50 = current / float(mm50) - 1 if valid_number(mm50) and mm50 else None
    distance_mm200 = current / float(mm200) - 1 if valid_number(mm200) and mm200 else None
    returns = data["Close"].pct_change().dropna()
    volatility = returns.std() * sqrt(252)
    high_52 = data["High"].tail(252).max()
    low_52 = data["Low"].tail(252).min()
    range_52 = high_52 - low_52
    position_52w = (current - low_52) / range_52 if valid_number(range_52) and range_52 > 0 else None
    rsi, macd, signal = data["RSI"].iloc[-1], data["MACD"].iloc[-1], data["Signal"].iloc[-1]
    scores, favorable, unfavorable = score_analysis(data, info, mode)
    weights = WEIGHTS[mode]
    global_score = sum(scores[key] * weights[key] for key in scores) / 100
    confidence_metrics = (
        info.get("trailingPE"), info.get("profitMargins"), info.get("revenueGrowth"),
        info.get("earningsGrowth"), info.get("returnOnEquity"), info.get("freeCashflow"),
        info.get("priceToBook"), info.get("debtToEquity"), info.get("beta"), volatility,
        rsi, mm50, mm200,
    )
    available_count = sum(valid_number(metric) for metric in confidence_metrics)
    confidence_percent = round(available_count / len(confidence_metrics) * 100)
    verdict = score_verdict(global_score)

    st.title(name)
    st.markdown(
        f"<div class='meta'><b>{escape(ticker)} · {escape(exchange)} · {escape(currency_code)}</b><br>"
        f"{escape(str(info.get('sector', 'Secteur non renseigné')))} · Mode {escape(mode)}</div>",
        unsafe_allow_html=True,
    )
    metrics = st.columns(6)
    metrics[0].metric("Prix actuel", format_price(current, currency), format_percentage(daily_change))
    metrics[1].metric("P/E", f"{float(info['trailingPE']):.1f}x" if valid_number(info.get("trailingPE")) else "N/D")
    metrics[2].metric("Marge nette TTM", format_percentage(info.get("profitMargins")))
    metrics[3].metric("Croissance CA YoY", format_percentage(info.get("revenueGrowth")))
    metrics[4].metric("MM50", format_price(mm50, currency), f"Cours {format_percentage(distance_mm50)}")
    metrics[5].metric("MM200", format_price(mm200, currency), f"Cours {format_percentage(distance_mm200)}")
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
          <div class="muted">Confiance : <b>{confidence_percent} % — {available_count}/{len(confidence_metrics)} indicateurs disponibles</b></div>
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

    stop_loss, take_profit = calculate_trade_levels(current, mode)
    risk_reward = (take_profit - current) / (current - stop_loss)
    fundamentals = [
        ("ROE", format_percentage(info.get("returnOnEquity"))),
        ("Free Cash Flow (TTM)", format_large_amount(info.get("freeCashflow"), currency)),
        ("Croissance CA (YoY)", format_percentage(info.get("revenueGrowth"))),
        ("Croissance bénéfices (YoY)", format_percentage(info.get("earningsGrowth"))),
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
        ("Position dans la fourchette 52 sem.", f"{position_52w:.0%}" if position_52w is not None else "N/D"),
        ("Distance du cours à la MM50", format_percentage(distance_mm50)),
        ("Distance du cours à la MM200", format_percentage(distance_mm200)),
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
    summary = build_summary(
        mode, verdict, info.get("trailingPE"), info.get("profitMargins"),
        info.get("revenueGrowth"), distance_mm50, distance_mm200, position_52w,
    )
    trade_rows = f"""
      <p>{escape(summary)}</p>
      <h3>Plan mécanique</h3>
      <div class="trade-row"><span>Stop-Loss indicatif</span><span class="value bad">{format_price(stop_loss, currency)} ({format_percentage(stop_loss/current-1, decimal=True)})</span></div>
      <div class="trade-row"><span>Take-Profit indicatif</span><span class="value good">{format_price(take_profit, currency)} ({format_percentage(take_profit/current-1, decimal=True)})</span></div>
      <div class="trade-row"><span>Ratio risque / récompense</span><span class="value">1 : {risk_reward:.1f}</span></div>
      <p class="disclaimer">Ces niveaux ne tiennent pas encore compte des supports techniques ni de l'ATR.</p>
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
