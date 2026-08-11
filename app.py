# -*- coding: utf-8 -*-
"""Tableau de bord Streamlit d'analyse multifactorielle d'actions."""

from __future__ import annotations

from datetime import date, datetime, timezone
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


def _normalise_event_dates(value: Any) -> list[date]:
    """Extrait récursivement des dates, malgré les formats variables de Yahoo."""
    dates: list[date] = []
    if isinstance(value, (datetime, pd.Timestamp)):
        timestamp = pd.Timestamp(value)
        if timestamp.tzinfo is not None:
            timestamp = timestamp.tz_convert("UTC").tz_localize(None)
        dates.append(timestamp.date())
    elif isinstance(value, date):
        dates.append(value)
    elif isinstance(value, (list, tuple, set)):
        for item in value:
            dates.extend(_normalise_event_dates(item))
    elif isinstance(value, dict):
        for item in value.values():
            dates.extend(_normalise_event_dates(item))
    elif isinstance(value, pd.Series):
        dates.extend(_normalise_event_dates(value.tolist()))
    elif isinstance(value, pd.DataFrame):
        dates.extend(_normalise_event_dates(value.to_dict()))
    elif isinstance(value, str):
        try:
            parsed = pd.to_datetime(value, errors="raise")
        except (TypeError, ValueError, OverflowError):
            pass
        else:
            dates.extend(_normalise_event_dates(parsed))
    return sorted(set(dates))


def _calendar_earnings_dates(calendar: Any) -> list[date]:
    """Repère uniquement le champ de résultats d'un calendrier Yahoo."""
    aliases = {"earningsdate", "earningsdates", "earnings date", "earnings dates"}
    matches: list[date] = []
    if isinstance(calendar, dict):
        for key, value in calendar.items():
            if str(key).strip().lower() in aliases:
                matches.extend(_normalise_event_dates(value))
            elif isinstance(value, (dict, pd.DataFrame)):
                matches.extend(_calendar_earnings_dates(value))
    elif isinstance(calendar, pd.DataFrame):
        for label in calendar.columns:
            if str(label).strip().lower() in aliases:
                matches.extend(_normalise_event_dates(calendar[label]))
        for label in calendar.index:
            if str(label).strip().lower() in aliases:
                matches.extend(_normalise_event_dates(calendar.loc[label]))
    return sorted(set(matches))


def _info_date(value: Any) -> date | None:
    """Convertit une date Yahoo (notamment un timestamp Unix) sans l'inventer."""
    if valid_number(value):
        try:
            return datetime.fromtimestamp(float(value), tz=timezone.utc).date()
        except (ValueError, OverflowError, OSError):
            return None
    dates = _normalise_event_dates(value)
    return dates[0] if dates else None


def get_key_events(ticker: str, info: dict[str, Any]) -> dict[str, Any]:
    """Charge les résultats et dividendes Yahoo sans bloquer l'analyse principale."""
    try:
        stock = yf.Ticker(ticker)
    except Exception:
        stock = None
    calendar: Any = None
    if stock is not None:
        try:
            calendar = stock.get_calendar()
        except Exception:
            calendar = None
    calendar_empty = calendar is None or (hasattr(calendar, "empty") and bool(calendar.empty))
    calendar_empty = calendar_empty or (isinstance(calendar, dict) and not calendar)
    if calendar_empty and stock is not None:
        try:
            calendar = stock.calendar
        except Exception:
            calendar = None

    today = datetime.now(timezone.utc).date()
    future_earnings = [item for item in _calendar_earnings_dates(calendar) if item >= today]
    # Une fenêtre Yahoo contient généralement deux bornes proches ; on la conserve.
    earnings_dates = future_earnings[:2]

    last_dividend_amount: float | None = None
    last_dividend_date: date | None = None
    if stock is not None:
        try:
            dividends = stock.get_dividends(period="1y")
            if isinstance(dividends, pd.Series) and not dividends.dropna().empty:
                last_index = dividends.dropna().index[-1]
                last_value = dividends.dropna().iloc[-1]
                if valid_number(last_value):
                    last_dividend_amount = float(last_value)
                    normalised = _normalise_event_dates(last_index)
                    last_dividend_date = normalised[0] if normalised else None
        except Exception:
            pass

    annual_dividend = info.get("dividendRate") if isinstance(info, dict) else None
    dividend_yield = info.get("dividendYield") if isinstance(info, dict) else None
    if not valid_number(dividend_yield) and valid_number(annual_dividend):
        current_price = info.get("currentPrice")
        if not valid_number(current_price):
            current_price = info.get("regularMarketPrice")
        if valid_number(current_price) and float(current_price) > 0:
            dividend_yield = float(annual_dividend) / float(current_price)
    return {
        "earnings_dates": earnings_dates,
        "days_until_earnings": (earnings_dates[0] - today).days if earnings_dates else None,
        "annual_dividend": float(annual_dividend) if valid_number(annual_dividend) else None,
        "dividend_yield": float(dividend_yield) if valid_number(dividend_yield) else None,
        "ex_dividend_date": _info_date(info.get("exDividendDate")) if isinstance(info, dict) else None,
        "last_dividend_amount": last_dividend_amount,
        "last_dividend_date": last_dividend_date,
    }


def _format_french_date(value: date | None) -> str:
    if value is None:
        return "N/D"
    months = ("janvier", "février", "mars", "avril", "mai", "juin", "juillet", "août", "septembre", "octobre", "novembre", "décembre")
    return f"{value.day} {months[value.month - 1]} {value.year}"


def render_key_events(events: dict[str, Any], currency_symbol: str) -> None:
    """Affiche les événements à titre informatif, sans incidence sur le score."""
    st.subheader("📅 Prochains événements clés")
    earnings_dates = events.get("earnings_dates") if isinstance(events, dict) else None
    earnings_dates = earnings_dates if isinstance(earnings_dates, list) else []
    if len(earnings_dates) > 1:
        earnings_text = f"{_format_french_date(earnings_dates[0])} – {_format_french_date(earnings_dates[1])}"
    elif earnings_dates:
        earnings_text = _format_french_date(earnings_dates[0])
    else:
        earnings_text = "N/D"
    days = events.get("days_until_earnings") if isinstance(events, dict) else None
    countdown = ""
    if isinstance(days, int):
        countdown = "Aujourd’hui" if days == 0 else f"Dans {days} jours"

    annual = format_price(events.get("annual_dividend"), currency_symbol)
    yield_value = events.get("dividend_yield")
    dividend_yield = f"{float(yield_value) * 100:.2f} %" if valid_number(yield_value) else "N/D"
    ex_date = _format_french_date(events.get("ex_dividend_date"))
    last_amount = format_price(events.get("last_dividend_amount"), currency_symbol)
    last_date = _format_french_date(events.get("last_dividend_date"))
    left, right = st.columns(2)
    with left:
        st.markdown(
            f'<div class="card"><h3>📊 Résultats</h3><div class="value">{escape(earnings_text)}</div>'
            f'<p class="muted">{escape(countdown)}</p></div>', unsafe_allow_html=True,
        )
    with right:
        st.markdown(
            f'<div class="card"><h3>💰 Dividende</h3>'
            f'<div class="trade-row"><span>Dividende annuel indiqué</span><span class="value">{escape(annual)}</span></div>'
            f'<div class="trade-row"><span>Rendement actuel</span><span class="value">{escape(dividend_yield)}</span></div>'
            f'<div class="trade-row"><span>Date ex-dividende</span><span class="value">{escape(ex_date)}</span></div>'
            f'<div class="trade-row"><span>Dernier dividende versé</span><span class="value">{escape(last_amount)}</span></div>'
            f'<div class="trade-row"><span>Date du dernier versement</span><span class="value">{escape(last_date)}</span></div></div>',
            unsafe_allow_html=True,
        )
    if isinstance(days, int) and 0 <= days <= 7:
        alert = "⚠️ Publication de résultats prévue aujourd'hui" if days == 0 else f"⚠️ Résultats imminents dans {days} jours"
        st.markdown(
            f'<div class="decision"><b class="bad">{escape(alert)}</b><br>'
            '<span class="muted">Une publication de résultats peut entraîner une hausse temporaire de la volatilité.</span></div>',
            unsafe_allow_html=True,
        )
    st.markdown('<p class="disclaimer">Données événementielles fournies par Yahoo Finance et susceptibles d’être modifiées.</p>', unsafe_allow_html=True)


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


def build_comparison_snapshot(ticker: str, period: str, mode: str) -> dict[str, Any]:
    """Construit une vue standardisée en réutilisant le moteur d'analyse."""
    history, info = load_stock_data(ticker, period)
    data = calculate_indicators(history)
    current = float(data["Close"].iloc[-1])
    previous = float(data["Close"].iloc[-2]) if len(data) > 1 else None
    mm50, mm200 = data["MM50"].iloc[-1], data["MM200"].iloc[-1]
    scores, _, _ = score_analysis(data, info, mode)
    weights = WEIGHTS[mode]
    return {
        "ticker": ticker,
        "name": str(info.get("longName") or info.get("shortName") or ticker),
        "exchange": str(info.get("fullExchangeName") or info.get("exchange") or "Place N/D"),
        "currency": str(info.get("currency") or "Devise N/D"),
        "price": current,
        "daily_change": current / previous - 1 if previous else None,
        "pe": info.get("trailingPE"),
        "margin": info.get("profitMargins"),
        "revenue_growth": info.get("revenueGrowth"),
        "roe": info.get("returnOnEquity"),
        "price_to_book": info.get("priceToBook"),
        "rsi": data["RSI"].iloc[-1],
        "mm50": current / float(mm50) - 1 if valid_number(mm50) and mm50 else None,
        "mm200": current / float(mm200) - 1 if valid_number(mm200) and mm200 else None,
        "volatility": data["Close"].pct_change().std() * sqrt(252),
        "global_score": sum(scores[key] * weights[key] for key in scores) / 100,
        "scores": scores,
    }


def comparison_winner(metric: str, left: Any, right: Any, tickers: tuple[str, str]) -> str:
    """Détermine l'avantage selon la nature du critère, sans noter les absences."""
    if metric == "price":
        return "—"
    if not valid_number(left) and not valid_number(right):
        return "—"
    if not valid_number(left):
        return f"🏆 {tickers[1]}"
    if not valid_number(right):
        return f"🏆 {tickers[0]}"
    a, b = float(left), float(right)
    if metric in {"pe", "price_to_book"}:
        if a <= 0 and b <= 0:
            return "—"
        if a <= 0:
            return f"🏆 {tickers[1]}"
        if b <= 0:
            return f"🏆 {tickers[0]}"
        better = a < b
    elif metric == "volatility":
        better = a < b
    elif metric == "rsi":
        a_target, b_target = 40 <= a <= 65, 40 <= b <= 65
        if a_target and b_target:
            return "⚖️ Égalité"
        if a_target != b_target:
            return f"🏆 {tickers[0] if a_target else tickers[1]}"
        better = abs(a - 52.5) < abs(b - 52.5)
    elif metric in {"mm50", "mm200"}:
        # Une hausse modérée au-dessus de la moyenne est privilégiée ;
        # les excès et les tendances négatives sont pénalisés.
        utility = lambda value: -abs(value - 0.08) - (0.15 if value < 0 else 0)
        better = utility(a) > utility(b)
    else:
        better = a > b
    if abs(a - b) < 1e-12:
        return "⚖️ Égalité"
    return f"🏆 {tickers[0] if better else tickers[1]}"


def build_head_to_head_summary(left: dict[str, Any], right: dict[str, Any]) -> str:
    """Produit une synthèse déterministe uniquement avec les mesures disponibles."""
    tickers = (left["ticker"], right["ticker"])
    sentences: list[str] = []
    pe_winner = comparison_winner("pe", left["pe"], right["pe"], tickers)
    if pe_winner.startswith("🏆"):
        sentences.append(f"{pe_winner[2:]} ressort mieux valorisée grâce à un P/E positif inférieur")
    margin_winner = comparison_winner("margin", left["margin"], right["margin"], tickers)
    if margin_winner.startswith("🏆"):
        sentences.append(f"{margin_winner[2:]} présente la meilleure marge nette")
    technical_winner = comparison_winner(
        "score", left["scores"]["Technique"], right["scores"]["Technique"], tickers
    )
    if technical_winner.startswith("🏆"):
        sentences.append(f"sur le plan technique, {technical_winner[2:]} obtient le meilleur score")
    score_gap = left["global_score"] - right["global_score"]
    if abs(score_gap) >= 3:
        winner = tickers[0] if score_gap > 0 else tickers[1]
        sentences.append(f"le score global donne actuellement un avantage à {winner}")
    else:
        sentences.append("les scores globaux restent très équilibrés")
    return ". ".join(sentence[0].upper() + sentence[1:] for sentence in sentences) + "."


def render_comparison(left: dict[str, Any], right: dict[str, Any]) -> None:
    """Affiche les cartes et le tableau du comparateur."""
    card_columns = st.columns([1, 0.18, 1])
    for column, snapshot in ((card_columns[0], left), (card_columns[2], right)):
        score = snapshot["global_score"]
        color = "good" if score >= 70 else ("warn" if score >= 50 else "bad")
        details = (
            f"P/E : {float(snapshot['pe']):.1f}x" if valid_number(snapshot["pe"]) else "P/E : N/D"
        )
        details += f"<br>Marge nette : {format_percentage(snapshot['margin'])}"
        details += f"<br>Croissance CA : {format_percentage(snapshot['revenue_growth'])}"
        details += f"<br>RSI : {float(snapshot['rsi']):.1f}" if valid_number(snapshot["rsi"]) else "<br>RSI : N/D"
        with column:
            st.markdown(
                f'<div class="decision"><h3>{escape(snapshot["name"])}</h3>'
                f'<div class="meta"><b>{escape(snapshot["ticker"])} · {escape(snapshot["exchange"])} · {escape(snapshot["currency"])}</b></div>'
                f'<div class="score {color}">{score:.0f}<span style="font-size:1rem"> / 100</span></div>'
                f'<p>{details}</p></div>', unsafe_allow_html=True,
            )
    card_columns[1].markdown("<h2 style='text-align:center;padding-top:5rem'>VS</h2>", unsafe_allow_html=True)
    gap = left["global_score"] - right["global_score"]
    if abs(gap) >= 3:
        st.subheader(f"🏆 Avantage global : {left['ticker'] if gap > 0 else right['ticker']}")
    else:
        st.subheader("⚖️ Comparaison très équilibrée")
    st.caption("Le score global est un outil de comparaison multifactorielle et ne constitue pas une recommandation d'investissement.")

    def ratio(value: Any) -> str:
        return f"{float(value):.1f}x" if valid_number(value) else "N/D"

    rows = [
        ("Score global", "global_score", left["global_score"], right["global_score"], lambda v: f"{v:.0f}/100"),
        ("Prix actuel", "price", left["price"], right["price"], format_price),
        ("Variation journalière", "daily_change", left["daily_change"], right["daily_change"], format_percentage),
        ("P/E", "pe", left["pe"], right["pe"], ratio),
        ("Marge nette", "margin", left["margin"], right["margin"], format_percentage),
        ("Croissance CA", "revenue_growth", left["revenue_growth"], right["revenue_growth"], format_percentage),
        ("ROE", "roe", left["roe"], right["roe"], format_percentage),
        ("Price / Book", "price_to_book", left["price_to_book"], right["price_to_book"], ratio),
        ("RSI 14", "rsi", left["rsi"], right["rsi"], lambda v: f"{float(v):.1f}" if valid_number(v) else "N/D"),
        ("Volatilité annualisée", "volatility", left["volatility"], right["volatility"], format_percentage),
        ("Distance à MM50", "mm50", left["mm50"], right["mm50"], format_percentage),
        ("Distance à MM200", "mm200", left["mm200"], right["mm200"], format_percentage),
    ]
    for score_name in WEIGHTS[next(iter(WEIGHTS))]:
        rows.append((f"Score {score_name}", "score", left["scores"][score_name], right["scores"][score_name], lambda v: f"{v:.0f}/100"))
    table = []
    for label, key, a, b, formatter in rows:
        if key == "price":
            left_text = formatter(a, CURRENCY_SYMBOLS.get(left["currency"], f"{left['currency']} "))
            right_text = formatter(b, CURRENCY_SYMBOLS.get(right["currency"], f"{right['currency']} "))
        else:
            left_text, right_text = formatter(a), formatter(b)
        table.append({
            "Critère": label, "Action A": left_text, "Action B": right_text,
            "Avantage": comparison_winner(key, a, b, (left["ticker"], right["ticker"])),
        })
    st.dataframe(pd.DataFrame(table), hide_index=True, use_container_width=True)
    st.subheader("Résumé du face-à-face")
    st.markdown(f'<div class="card">{escape(build_head_to_head_summary(left, right))}</div>', unsafe_allow_html=True)


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

    left, right = st.columns([1, 1.4])
    with left:
        color = "good" if global_score >= 70 else ("warn" if global_score >= 50 else "bad")
        st.markdown(f"""
        <div class="decision">
          <div class="muted">SCORE GLOBAL</div>
          <div class="score {color}">{global_score:.0f}<span style="font-size:1rem">/100</span></div>
          <div class="verdict">{escape(verdict)}</div>
          <div class="muted">Confiance : <b>{confidence_percent} % — {available_count}/{len(confidence_metrics)} indicateurs disponibles</b></div>
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

    render_key_events(get_key_events(ticker, info), currency)

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
    """
    st.markdown(
        f'<div class="decision">{trade_rows}</div>',
        unsafe_allow_html=True,
    )
    st.markdown("<p class='disclaimer'>Information éducative uniquement : cette analyse automatisée ne constitue ni un conseil en investissement, ni une recommandation d'achat ou de vente. Les données peuvent être incomplètes ou différées.</p>", unsafe_allow_html=True)


apply_styles()
with st.sidebar:
    navigation = st.radio("Mode", options=["📊 Analyse", "🥊 Comparateur"])
    st.header("Paramètres d'analyse")
    selected_period = st.selectbox("Période", options=list(PERIODS))
    selected_mode = st.radio("Profil", options=list(WEIGHTS))
    if navigation == "📊 Analyse":
        ticker_input = st.text_input("Ticker", value="AAPL", placeholder="AAPL, MSFT, MC.PA…").strip().upper()
        analyze = st.button("Analyser", type="primary", use_container_width=True)
        st.caption("Les tickers internationaux nécessitent leur suffixe de place (ex. MC.PA).")
    else:
        ticker_a = st.text_input("Action A", value="TTE.PA").strip().upper()
        ticker_b = st.text_input("Action B", value="SHEL").strip().upper()
        compare = st.button("Comparer", type="primary", use_container_width=True)

if navigation == "📊 Analyse":
    if not ticker_input:
        st.info("Saisissez un ticker dans la barre latérale.")
    elif analyze or ticker_input:
        try:
            render_dashboard(ticker_input, PERIODS[selected_period], selected_mode)
        except Exception as error:
            st.error(f"Impossible d'analyser {ticker_input} : {error}")
else:
    st.title("🥊 Comparateur d'actions")
    st.caption("Comparez deux entreprises avec les mêmes critères d'analyse.")
    if compare:
        snapshots: list[dict[str, Any] | None] = []
        for label, ticker in (("Action A", ticker_a), ("Action B", ticker_b)):
            if not ticker:
                st.error(f"{label} : saisissez un ticker.")
                snapshots.append(None)
                continue
            try:
                with st.spinner(f"Chargement de {ticker}…"):
                    snapshots.append(build_comparison_snapshot(ticker, PERIODS[selected_period], selected_mode))
            except Exception as error:
                st.error(f"Impossible de charger {label} ({ticker}) : {error}")
                snapshots.append(None)
        if all(snapshots):
            render_comparison(snapshots[0], snapshots[1])
