# -*- coding: utf-8 -*-
"""Tableau de bord Streamlit d'analyse multifactorielle d'actions."""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from html import escape
from math import isfinite, sqrt
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from yfinance import EquityQuery
from google import genai


PERIODS = {"6 mois": "6mo", "1 an": "1y", "2 ans": "2y", "5 ans": "5y"}
GEMINI_MODEL = "gemini-3.5-flash"
WEIGHTS = {
    "Investisseur": {"Fondamentaux": 25, "Valorisation": 20, "Croissance": 20, "Technique": 20, "Risque": 15},
    "Trader / Swing": {"Fondamentaux": 15, "Valorisation": 10, "Croissance": 15, "Technique": 40, "Risque": 20},
}
CURRENCY_SYMBOLS = {
    "USD": "$", "EUR": "€", "GBP": "£", "JPY": "¥", "CHF": "CHF ",
    "CAD": "C$", "CNY": "¥", "INR": "₹", "KRW": "₩",
}
BACKTEST_HORIZONS = (5, 20, 60)
MIN_SIGNAL_GAP = 10


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
        @media(max-width:768px) { .bar-row { grid-template-columns:minmax(90px,auto) 1fr minmax(62px,auto); }
          .bar-row > :last-child { white-space:nowrap; } }
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
    current_price = info.get("currentPrice") if isinstance(info, dict) else None
    if not valid_number(current_price) and isinstance(info, dict):
        current_price = info.get("regularMarketPrice")

    # On normalise le rendement en ratio décimal (ex. 0.0045 = 0,45 %).
    # La priorité va au calcul montant annuel / cours, afin d'éviter les unités
    # parfois ambiguës du champ Yahoo `dividendYield`.
    dividend_yield: float | None = None
    if valid_number(annual_dividend) and valid_number(current_price) and float(current_price) > 0:
        dividend_yield = float(annual_dividend) / float(current_price)
    elif isinstance(info, dict):
        trailing_yield = info.get("trailingAnnualDividendYield")
        if valid_number(trailing_yield) and float(trailing_yield) >= 0:
            dividend_yield = float(trailing_yield)

    return {
        "earnings_dates": earnings_dates,
        "days_until_earnings": (earnings_dates[0] - today).days if earnings_dates else None,
        "annual_dividend": float(annual_dividend) if valid_number(annual_dividend) else None,
        "dividend_yield": dividend_yield,
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


def build_ai_snapshot(
    ticker: str,
    company_name: str,
    mode: str,
    global_score: float,
    verdict: str,
    scores: dict[str, float],
    info: dict[str, Any],
    current: float,
    rsi: Any,
    macd: Any,
    signal: Any,
    volatility: Any,
    distance_mm50: float | None,
    distance_mm200: float | None,
    position_52w: float | None,
    events: dict[str, Any],
    timing: dict[str, Any],
) -> dict[str, Any]:
    """Construit l'instantané IA uniquement avec les métriques déjà calculées."""

    def available(values: dict[str, Any]) -> dict[str, Any]:
        return {
            label: float(value) if valid_number(value) else value
            for label, value in values.items()
            if valid_number(value) or isinstance(value, (str, date)) and bool(value)
        }

    earnings_dates = events.get("earnings_dates") or []
    next_earnings = earnings_dates[0] if earnings_dates else None
    snapshot: dict[str, Any] = {
        "ticker": ticker,
        "nom_entreprise": company_name,
        "profil": mode,
        "score_global_sur_100": float(global_score),
        "verdict": verdict,
        "scores_sur_100": available({
            "Fondamentaux": scores.get("Fondamentaux"),
            "Valorisation": scores.get("Valorisation"),
            "Croissance": scores.get("Croissance"),
            "Technique": scores.get("Technique"),
            "Risque": scores.get("Risque"),
        }),
        "fondamentaux": available({
            "PE": info.get("trailingPE"),
            "marge_nette_ratio": info.get("profitMargins"),
            "croissance_CA_YoY_ratio": info.get("revenueGrowth"),
            "croissance_benefices_YoY_ratio": info.get("earningsGrowth"),
            "ROE_ratio": info.get("returnOnEquity"),
            "free_cash_flow": info.get("freeCashflow"),
            "dette_sur_capitaux_propres": info.get("debtToEquity"),
            "price_to_book": info.get("priceToBook"),
        }),
        "technique": available({
            "prix_actuel": current,
            "RSI": rsi,
            "MACD": macd,
            "signal_MACD": signal,
            "volatilite_annualisee_ratio": volatility,
            "distance_MM50_ratio": distance_mm50,
            "distance_MM200_ratio": distance_mm200,
            "position_fourchette_52_semaines_ratio": position_52w,
        }),
        "evenements": available({
            "prochaine_date_resultats": next_earnings,
            "jours_avant_resultats": events.get("days_until_earnings"),
            "rendement_dividende_ratio": events.get("dividend_yield"),
            "date_ex_dividende": events.get("ex_dividend_date"),
        }),
        "timing_entree": {
            "score_sur_100": timing.get("score") if valid_number(timing.get("score")) else "N/D",
            "confiance_pourcent": timing.get("confidence"),
            "verdict": timing.get("verdict"),
            "sous_scores_sur_100": available(timing.get("categories", {})),
            "conditions_validees": timing.get("confirmed_conditions"),
            "conditions_disponibles": timing.get("available_conditions"),
            "support_recent": timing.get("support"),
            "resistance_recente": timing.get("resistance"),
            "ATR_pourcent_du_cours": timing.get("atr_percent"),
        },
    }
    return {key: value for key, value in snapshot.items() if value not in ({}, None, "")}


def generate_ai_opinion(snapshot: dict[str, Any], api_key: str) -> str:
    """Demande à Gemini d'expliquer l'instantané, sans enrichissement externe."""
    prompt = """Tu es un assistant pédagogique d'analyse financière.

Analyse UNIQUEMENT les métriques fournies ci-dessous. Tu n'as pas le droit :
- d'inventer une information absente
- d'utiliser une information externe
- de modifier les chiffres
- de prédire avec certitude l'évolution du cours
- de dire « achetez », « vendez », « il faut acheter » ou « il faut vendre »
- de présenter cette analyse comme un conseil financier
- de transformer le Timing en conseil financier ou de dire « achète maintenant »
- de promettre une hausse ou d'inventer un niveau de prix

Réponds en français. Produis exactement 3 points courts :
✅ Point rassurant : explique la principale force visible dans les données.
⚠️ Point d'attention : explique le principal risque ou la principale faiblesse visible.
🧭 Lecture du timing : explique la convergence actuelle sans donner d'ordre.

Chaque point doit citer au moins une métrique réelle fournie lorsqu'une métrique
pertinente est disponible. Maximum environ 100 mots au total. Si certaines données
sont manquantes, ignore-les. Ne les invente jamais.

Données :
""" + json.dumps(snapshot, ensure_ascii=False, default=str, allow_nan=False)
    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
        text = getattr(response, "text", None)
        return text.strip() if isinstance(text, str) else ""
    except Exception:
        return ""


def render_ai_opinion(snapshot: dict[str, Any], ticker: str, mode: str, period: str) -> None:
    """Affiche et mémorise l'avis IA propre à l'analyse courante."""
    st.subheader("💡 L'Avis de l'IA")
    state_key = f"ai_opinion_{ticker}_{mode}_{period}"
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
    except Exception:
        api_key = None

    with st.container(border=True):
        st.markdown("### 💡 Analyse IA")
        opinion = st.session_state.get(state_key)
        if opinion:
            st.markdown(opinion)
            st.markdown(
                '<p class="disclaimer">Cette synthèse est générée automatiquement à partir des métriques '
                'affichées. Elle peut contenir des erreurs et ne constitue pas un conseil en investissement.</p>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown("Gemini peut synthétiser les données calculées ci-dessus.")
            if not api_key:
                st.info("Configurez GEMINI_API_KEY dans les secrets Streamlit pour activer l'analyse IA.")
            elif st.button("✨ Générer l'avis IA", key=f"generate_{state_key}"):
                with st.spinner("Analyse IA en cours…"):
                    opinion = generate_ai_opinion(snapshot, str(api_key))
                if opinion:
                    st.session_state[state_key] = opinion
                    st.rerun()
                else:
                    st.error("Impossible de générer l'analyse IA pour le moment.")


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


def calculate_atr(data: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calcule l'ATR de Wilder sans altérer les indicateurs historiques."""
    if not {"High", "Low", "Close"}.issubset(data.columns) or data.empty:
        return pd.Series(index=data.index, dtype=float, name=f"ATR{period}")
    previous_close = data["Close"].shift(1)
    true_range = pd.concat(
        [data["High"] - data["Low"], (data["High"] - previous_close).abs(),
         (data["Low"] - previous_close).abs()], axis=1,
    ).max(axis=1)
    atr = true_range.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    atr.name = f"ATR{period}"
    return atr.where(atr > 0)


def calculate_price_zones(data: pd.DataFrame, atr: float | None) -> dict[str, Any]:
    """Repère les pivots proches sur 60 séances, sans extrapoler de niveau."""
    result: dict[str, Any] = {"support": None, "resistance": None,
                              "support_zone": None, "resistance_zone": None}
    if data.empty or not {"Close", "High", "Low"}.issubset(data.columns):
        return result
    recent = data.tail(60)
    current = recent["Close"].iloc[-1]
    if not valid_number(current) or len(recent) < 5:
        return result
    current = float(current)
    lows, highs = recent["Low"], recent["High"]
    pivot_lows = [float(lows.iloc[i]) for i in range(2, len(recent) - 2)
                  if valid_number(lows.iloc[i]) and lows.iloc[i] < lows.iloc[i-2:i].min()
                  and lows.iloc[i] < lows.iloc[i+1:i+3].min() and lows.iloc[i] < current]
    pivot_highs = [float(highs.iloc[i]) for i in range(2, len(recent) - 2)
                   if valid_number(highs.iloc[i]) and highs.iloc[i] > highs.iloc[i-2:i].max()
                   and highs.iloc[i] > highs.iloc[i+1:i+3].max() and highs.iloc[i] > current]
    support = max(pivot_lows) if pivot_lows else None
    resistance = min(pivot_highs) if pivot_highs else None
    if support is None:
        candidate = recent["Low"].dropna().min()
        support = float(candidate) if valid_number(candidate) and candidate < current else None
    if resistance is None:
        candidate = recent["High"].dropna().max()
        resistance = float(candidate) if valid_number(candidate) and candidate > current else None
    result.update({"support": support, "resistance": resistance})
    width = float(atr) * .25 if valid_number(atr) and float(atr) > 0 else None
    if support is not None:
        result["support_zone"] = (support - width, support + width) if width else (support, support)
    if resistance is not None:
        result["resistance_zone"] = (resistance - width, resistance + width) if width else (resistance, resistance)
    return result


def timing_verdict(score: float | None) -> str:
    if score is None:
        return "⚪ Timing non déterminable"
    if score >= 80:
        return "🟢 Conditions très favorables"
    if score >= 70:
        return "🟢 Conditions favorables"
    if score >= 55:
        return "🟠 Attendre une confirmation"
    if score >= 40:
        return "🟠 Timing fragile / patience"
    return "🔴 Conditions peu favorables"


def build_timing_conditions(current: Any, mm50: Any, mm200: Any, rsi: Any,
                            macd: Any, signal: Any, days: Any) -> list[dict[str, Any]]:
    """Construit les six confirmations binaires, en distinguant N/D et échec."""
    conditions: list[dict[str, Any]] = []
    def add(label: str, available: bool, passed: bool = False, current_value: Any = None,
            target: Any = None, detail: str = "") -> None:
        conditions.append({"label": label, "available": available,
                           "passed": bool(passed) if available else False,
                           "current": current_value if available else None,
                           "target": target if available else None, "detail": detail})
    for average, label in ((mm200, "Cours > MM200"), (mm50, "Cours > MM50")):
        available = valid_number(current) and valid_number(average) and float(current) > 0
        passed = available and float(current) > float(average)
        detail = ""
        if available:
            if passed:
                detail = f"{(float(current) / float(average) - 1) * 100:+.2f} % au-dessus"
            else:
                detail = f"{(float(average) / float(current) - 1) * 100:+.2f} % nécessaire"
        add(label, available, passed, current, average, detail)
    available = valid_number(mm50) and valid_number(mm200)
    add("MM50 > MM200", available, available and float(mm50) > float(mm200), mm50, mm200)
    available = valid_number(rsi)
    add("RSI dans la zone de confirmation (45–70)", available,
        available and 45 <= float(rsi) <= 70, rsi, "45–70",
        f"RSI actuel : {float(rsi):.1f}" if available else "")
    available = valid_number(macd) and valid_number(signal)
    add("MACD > signal", available, available and float(macd) > float(signal), macd, signal,
        f"MACD {float(macd):.2f} vs signal {float(signal):.2f}" if available else "")
    available = isinstance(days, int) and days >= 0
    add("Pas de résultats dans les 7 prochains jours", available,
        available and days > 7, days, "> 7 jours",
        f"Résultats dans {days} jour{'s' if days != 1 else ''}" if available else "")
    return conditions


def calculate_entry_timing(data: pd.DataFrame, info: dict[str, Any],
                           events: dict[str, Any], mode: str) -> dict[str, Any]:
    """Calcule l'indice déterministe de convergence des signaux d'entrée."""
    del mode  # V1 robuste : barème identique pour les deux profils.
    criteria: list[dict[str, Any]] = []
    def criterion(category: str, key: str, label: str, maximum: float,
                  value: Any, earned: float | None, passed: bool | None = None) -> None:
        available = earned is not None and valid_number(value)
        criteria.append({"category": category, "key": key, "label": label,
                         "earned": float(earned) if available else None, "max": maximum,
                         "available": available, "passed": passed if available else None,
                         "current_value": float(value) if available else None})
    row = data.iloc[-1] if not data.empty else pd.Series(dtype=float)
    current, mm50, mm200 = row.get("Close"), row.get("MM50"), row.get("MM200")
    rsi, macd, signal = row.get("RSI"), row.get("MACD"), row.get("Signal")
    criterion("Tendance", "price_above_mm200", "Cours au-dessus de la MM200", 10, mm200,
              10 if valid_number(current) and valid_number(mm200) and current > mm200 else (0 if valid_number(current) and valid_number(mm200) else None), valid_number(current) and valid_number(mm200) and current > mm200)
    criterion("Tendance", "price_above_mm50", "Cours au-dessus de la MM50", 8, mm50,
              8 if valid_number(current) and valid_number(mm50) and current > mm50 else (0 if valid_number(current) and valid_number(mm50) else None), valid_number(current) and valid_number(mm50) and current > mm50)
    criterion("Tendance", "mm50_above_mm200", "MM50 au-dessus de la MM200", 8, mm50 if valid_number(mm200) else None,
              8 if valid_number(mm50) and valid_number(mm200) and mm50 > mm200 else (0 if valid_number(mm50) and valid_number(mm200) else None), valid_number(mm50) and valid_number(mm200) and mm50 > mm200)
    old_mm50 = data["MM50"].iloc[-11] if "MM50" in data and len(data) >= 11 else None
    criterion("Tendance", "mm50_slope", "MM50 orientée à la hausse", 4, old_mm50,
              4 if valid_number(mm50) and valid_number(old_mm50) and mm50 > old_mm50 else (0 if valid_number(mm50) and valid_number(old_mm50) else None), valid_number(mm50) and valid_number(old_mm50) and mm50 > old_mm50)
    rsi_points = None
    if valid_number(rsi):
        rv = float(rsi)
        rsi_points = 10 if 45 <= rv <= 60 else 7 if 60 < rv <= 70 else 6 if 40 <= rv < 45 else 3 if 30 < rv < 40 else 2 if rv > 70 else 4
    criterion("Momentum", "rsi", "RSI 14", 10, rsi, rsi_points, valid_number(rsi) and 45 <= float(rsi) <= 70)
    criterion("Momentum", "macd_cross", "MACD supérieur au signal", 10, macd if valid_number(signal) else None,
              10 if valid_number(macd) and valid_number(signal) and macd > signal else (0 if valid_number(macd) and valid_number(signal) else None), valid_number(macd) and valid_number(signal) and macd > signal)
    old_gap = data["MACD"].iloc[-6] - data["Signal"].iloc[-6] if len(data) >= 6 and {"MACD", "Signal"}.issubset(data.columns) else None
    gap = macd - signal if valid_number(macd) and valid_number(signal) else None
    criterion("Momentum", "macd_improving", "Momentum MACD en amélioration", 5, old_gap,
              5 if valid_number(gap) and valid_number(old_gap) and gap > old_gap else (0 if valid_number(gap) and valid_number(old_gap) else None), valid_number(gap) and valid_number(old_gap) and gap > old_gap)
    distance = float(current) / float(mm50) - 1 if valid_number(current) and valid_number(mm50) and float(mm50) else None
    distance_points = None
    if distance is not None:
        distance_points = 10 if -.02 <= distance <= .05 else 7 if .05 < distance <= .10 else 5 if -.05 <= distance < -.02 else 4 if .10 < distance <= .15 else 2
    criterion("Zone de prix", "distance_mm50", "Distance à la MM50", 10, distance, distance_points, distance is not None and -.02 <= distance <= .05)
    high, low = data["High"].tail(252).max() if "High" in data else None, data["Low"].tail(252).min() if "Low" in data else None
    position = (float(current)-float(low))/(float(high)-float(low)) if all(valid_number(x) for x in (current, high, low)) and high > low else None
    position_points = None
    if position is not None:
        position_points = 5 if .35 <= position <= .80 else 4 if .20 <= position < .35 else 3 if .80 < position <= .92 else 2 if position < .20 else 1
    criterion("Zone de prix", "position_52w", "Position dans la fourchette 52 semaines", 5, position, position_points)
    atr_series = calculate_atr(data)
    atr = atr_series.iloc[-1] if not atr_series.empty else None
    extension = abs(float(current)-float(mm50))/float(atr) if all(valid_number(x) for x in (current, mm50, atr)) and atr > 0 else None
    extension_points = 5 if extension is not None and extension <= 1 else 4 if extension is not None and extension <= 2 else 2 if extension is not None and extension <= 3 else 0 if extension is not None else None
    criterion("Zone de prix", "extension_atr", "Extension par rapport à la MM50", 5, extension, extension_points, extension is not None and extension <= 2)
    volatility = data["Close"].pct_change().dropna().std() * sqrt(252) if "Close" in data else None
    vol_points = None
    if valid_number(volatility):
        vol_points = 10 if volatility < .20 else 8 if volatility <= .30 else 6 if volatility <= .40 else 3 if volatility <= .55 else 1
    criterion("Risque", "volatility", "Volatilité annualisée", 10, volatility, vol_points, valid_number(volatility) and volatility <= .30)
    atr_percent = float(atr)/float(current) if valid_number(atr) and valid_number(current) and current > 0 else None
    atrp_points = 5 if atr_percent is not None and atr_percent < .02 else 4 if atr_percent is not None and atr_percent <= .03 else 3 if atr_percent is not None and atr_percent <= .045 else 1 if atr_percent is not None and atr_percent <= .06 else 0 if atr_percent is not None else None
    criterion("Risque", "atr_percent", "ATR en pourcentage du cours", 5, atr_percent, atrp_points, atr_percent is not None and atr_percent <= .03)
    days = events.get("days_until_earnings") if isinstance(events, dict) else None
    event_points = 0 if isinstance(days, int) and 0 <= days <= 3 else 3 if isinstance(days, int) and 4 <= days <= 7 else 6 if isinstance(days, int) and 8 <= days <= 14 else 10 if isinstance(days, int) and days > 14 else None
    criterion("Événements", "earnings", "Éloignement des prochains résultats", 10, days, event_points, isinstance(days, int) and days > 7)
    available = [item for item in criteria if item["available"]]
    available_max = sum(item["max"] for item in available)
    earned = sum(item["earned"] for item in available)
    score = earned / available_max * 100 if available_max else None
    categories = {}
    for name in ("Tendance", "Momentum", "Zone de prix", "Risque", "Événements"):
        items = [item for item in available if item["category"] == name]
        maximum = sum(item["max"] for item in items)
        categories[name] = sum(item["earned"] for item in items) / maximum * 100 if maximum else None
    positives, warnings = [], []
    lookup = {item["key"]: item for item in criteria}
    if lookup["price_above_mm200"]["passed"]: positives.append(f"Cours au-dessus de la MM200 ({(current/mm200-1)*100:+.1f} %)")
    elif lookup["price_above_mm200"]["available"]: warnings.append(f"Cours sous la MM200 de {(1-current/mm200)*100:.1f} %")
    if lookup["price_above_mm50"]["passed"]: positives.append(f"Cours au-dessus de la MM50 ({(current/mm50-1)*100:+.1f} %)")
    elif lookup["price_above_mm50"]["available"]: warnings.append(f"Cours sous la MM50 de {(1-current/mm50)*100:.1f} %")
    if lookup["mm50_above_mm200"]["passed"]: positives.append("MM50 au-dessus de la MM200")
    elif lookup["mm50_above_mm200"]["available"]: warnings.append("MM50 sous la MM200")
    if lookup["mm50_slope"]["passed"]: positives.append("MM50 orientée à la hausse")
    if valid_number(rsi) and 45 <= rsi <= 60: positives.append(f"RSI {rsi:.1f} : momentum équilibré")
    elif valid_number(rsi) and rsi > 70: warnings.append(f"RSI élevé à {rsi:.1f} : titre potentiellement étendu")
    elif valid_number(rsi) and rsi < 45: warnings.append(f"RSI faible à {rsi:.1f}")
    if lookup["macd_cross"]["passed"]: positives.append("MACD supérieur au signal")
    elif lookup["macd_cross"]["available"]: warnings.append("MACD inférieur au signal")
    if lookup["macd_improving"]["passed"]: positives.append("Momentum MACD en amélioration")
    if (distance is not None and distance > .12) or (extension is not None and extension > 3): warnings.append("Le cours est fortement étendu par rapport à sa tendance récente")
    if valid_number(volatility) and volatility <= .30: positives.append(f"Volatilité contenue à {volatility:.0%}")
    elif valid_number(volatility) and volatility > .40: warnings.append(f"Volatilité élevée : {volatility:.0%}")
    if isinstance(days, int) and days > 14: positives.append("Aucun résultat prévu dans les 14 prochains jours")
    elif isinstance(days, int) and days <= 7: warnings.append(f"Résultats dans {days} jour{'s' if days != 1 else ''}")
    zones = calculate_price_zones(data, float(atr) if valid_number(atr) else None)
    conditions = build_timing_conditions(current, mm50, mm200, rsi, macd, signal, days)
    confirmed = sum(item["passed"] for item in conditions if item["available"])
    condition_max = sum(item["available"] for item in conditions)
    confirmation_level = float(mm50) if valid_number(current) and valid_number(mm50) and current < mm50 else zones["resistance"] if valid_number(zones["resistance"]) and zones["resistance"] > current else None
    return {"score": score, "confidence": round(available_max), "verdict": timing_verdict(score),
            "categories": categories, "criteria": criteria, "positive_signals": positives[:6],
            "warning_signals": warnings[:6], "conditions": conditions,
            "confirmed_conditions": confirmed, "available_conditions": condition_max,
            "support_zone": zones["support_zone"], "resistance_zone": zones["resistance_zone"],
            "support": zones["support"], "resistance": zones["resistance"],
            "confirmation_level": confirmation_level, "atr": float(atr) if valid_number(atr) else None,
            "atr_percent": atr_percent, "current": float(current) if valid_number(current) else None,
            "mm50": float(mm50) if valid_number(mm50) else None, "mm200": float(mm200) if valid_number(mm200) else None,
            "distance_mm50": distance, "extension_atr": extension,
            "available_points": earned, "maximum_available_points": available_max}



@st.cache_data(ttl=1800, show_spinner=False)
def load_backtest_history(ticker: str, period: str) -> pd.DataFrame:
    """Charge une seule série quotidienne ajustée, avec 300 jours de warm-up."""
    years = {"1y": 1, "2y": 2, "3y": 3, "5y": 5}.get(period, 3)
    analysis_start = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize() - pd.DateOffset(years=years)
    download_start = analysis_start - timedelta(days=300)
    # auto_adjust garantit que tous les prix (signal, rendements, excursions) sont
    # ajustés de façon cohérente pour splits et dividendes.
    history = yf.download(ticker, start=download_start.date(), interval="1d",
                          auto_adjust=True, progress=False, threads=False)
    if history.empty:
        return pd.DataFrame()
    if isinstance(history.columns, pd.MultiIndex):
        history.columns = history.columns.get_level_values(0)
    if "Close" not in history:
        return pd.DataFrame()
    history = history.dropna(subset=["Close"]).copy()
    history.attrs["analysis_start"] = analysis_start
    return history


def calculate_historical_timing_series(data: pd.DataFrame, mode: str) -> pd.DataFrame:
    """Rejoue causalement le barème Timing technique, sans événements ni fondamentaux."""
    if data.empty:
        return pd.DataFrame()
    indicators = calculate_indicators(data)
    indicators["ATR14"] = calculate_atr(indicators)
    rows: list[dict[str, Any]] = []
    # Chaque tranche s'arrête à la séance notée : les calculs rolling/EWM et les
    # zones ne peuvent donc observer aucune valeur future. Le barème reste celui
    # de calculate_entry_timing, et Événements est retiré du dénominateur.
    for position in range(len(indicators)):
        prefix = indicators.iloc[:position + 1]
        timing = calculate_entry_timing(prefix, {}, {}, mode)
        criteria = [item for item in timing["criteria"]
                    if item["category"] != "Événements" and item["available"]]
        available = sum(item["max"] for item in criteria)
        earned = sum(item["earned"] for item in criteria)
        categories: dict[str, float | None] = {}
        for category in ("Tendance", "Momentum", "Zone de prix", "Risque"):
            selected = [item for item in criteria if item["category"] == category]
            maximum = sum(item["max"] for item in selected)
            categories[category] = sum(item["earned"] for item in selected) / maximum * 100 if maximum else None
        row = indicators.iloc[position]
        close, mm50, mm200 = row.get("Close"), row.get("MM50"), row.get("MM200")
        rows.append({
            "date": indicators.index[position], "close": close,
            "timing_score": earned / available * 100 if available else None,
            "timing_confidence": available,
            "tendance_score": categories["Tendance"],
            "momentum_score": categories["Momentum"], "zone_score": categories["Zone de prix"],
            "risque_score": categories["Risque"], "RSI": row.get("RSI"),
            "MACD": row.get("MACD"), "Signal MACD": row.get("Signal"),
            "MM50": mm50, "MM200": mm200, "ATR14": row.get("ATR14"),
            "distance_mm50": close / mm50 - 1 if valid_number(close) and valid_number(mm50) and mm50 else None,
            "distance_mm200": close / mm200 - 1 if valid_number(close) and valid_number(mm200) and mm200 else None,
            "_position": position,
        })
    return pd.DataFrame(rows).set_index("date", drop=False)


def extract_backtest_signals(timing: pd.DataFrame, threshold: float,
                             avoid_close: bool = True, *, low: bool = False) -> pd.DataFrame:
    """Conserve les franchissements du seuil et applique éventuellement l'espacement."""
    if timing.empty or "timing_score" not in timing:
        return timing.iloc[0:0].copy()
    score = timing["timing_score"]
    crossing = (score <= threshold) & (score.shift(1) > threshold) if low else \
               (score >= threshold) & (score.shift(1) < threshold)
    candidates = timing.loc[crossing.fillna(False)].copy()
    if not avoid_close or candidates.empty:
        return candidates
    kept: list[int] = []
    last_position: int | None = None
    for position in candidates["_position"].astype(int):
        if last_position is None or position - last_position >= MIN_SIGNAL_GAP:
            kept.append(position)
            last_position = position
    return candidates[candidates["_position"].isin(kept)].copy()


def calculate_forward_returns(observations: pd.DataFrame, timeline: pd.DataFrame) -> pd.DataFrame:
    """Calcule Close[T+h]/Close[T]-1 ; un horizon incomplet reste NaN."""
    result = observations.copy()
    closes = timeline["close"].reset_index(drop=True)
    for horizon in BACKTEST_HORIZONS:
        values: list[float | None] = []
        for position in result["_position"].astype(int):
            if position + horizon < len(closes) and valid_number(closes.iloc[position]):
                future = closes.iloc[position + horizon]
                values.append(float(future) / float(closes.iloc[position]) - 1 if valid_number(future) else None)
            else:
                values.append(None)
        result[f"return_{horizon}"] = values
    return result


def calculate_signal_drawdowns(signals: pd.DataFrame, timeline: pd.DataFrame) -> pd.DataFrame:
    """Ajoute le drawdown et la MFE observés de T à T+h inclus."""
    result = signals.copy()
    closes = timeline["close"].reset_index(drop=True)
    for horizon in BACKTEST_HORIZONS:
        drawdowns, mfes = [], []
        for position in result["_position"].astype(int):
            if position + horizon >= len(closes) or not valid_number(closes.iloc[position]):
                drawdowns.append(None); mfes.append(None); continue
            window = closes.iloc[position:position + horizon + 1].dropna()
            reference = float(closes.iloc[position])
            drawdowns.append(float(window.min()) / reference - 1 if not window.empty else None)
            mfes.append(float(window.max()) / reference - 1 if not window.empty else None)
        result[f"drawdown_{horizon}"] = drawdowns
        result[f"mfe_{horizon}"] = mfes
    return result


def calculate_backtest_statistics(signals: pd.DataFrame) -> dict[str, Any]:
    statistics: dict[str, Any] = {}
    for horizon in BACKTEST_HORIZONS:
        returns = signals.get(f"return_{horizon}", pd.Series(dtype=float)).dropna()
        drawdowns = signals.get(f"drawdown_{horizon}", pd.Series(dtype=float)).dropna()
        mfes = signals.get(f"mfe_{horizon}", pd.Series(dtype=float)).dropna()
        statistics[horizon] = {
            "observations": len(returns), "mean": returns.mean() if len(returns) else None,
            "median": returns.median() if len(returns) else None,
            "positive": (returns > 0).mean() if len(returns) else None,
            "best": returns.max() if len(returns) else None, "worst": returns.min() if len(returns) else None,
            "drawdown_mean": drawdowns.mean() if len(drawdowns) else None,
            "drawdown_median": drawdowns.median() if len(drawdowns) else None,
            "drawdown_worst": drawdowns.min() if len(drawdowns) else None,
            "mfe_mean": mfes.mean() if len(mfes) else None, "mfe_median": mfes.median() if len(mfes) else None,
        }
    return statistics


def calculate_baseline_statistics(timing: pd.DataFrame) -> dict[str, Any]:
    """Baseline de toutes les séances admissibles de la fenêtre analysée."""
    baseline = timing.copy()
    baseline["_position"] = range(len(baseline))
    return calculate_backtest_statistics(calculate_forward_returns(baseline, baseline))


def calculate_pullback_timing_series(data: pd.DataFrame) -> pd.DataFrame:
    """Calcule V2 de façon vectorielle et causale (repli + début de reprise).

    Chaque critère indisponible est retiré du dénominateur. Un rebond dont le
    cours *et* la MM50 restent sous la MM200 est plafonné à 55/100.
    """
    if data.empty or "Close" not in data:
        return pd.DataFrame()
    frame = calculate_indicators(data)
    frame["ATR14"] = calculate_atr(frame)
    close, mm50, mm200 = frame["Close"], frame["MM50"], frame["MM200"]
    rsi, macd, signal, atr = frame["RSI"], frame["MACD"], frame["Signal"], frame["ATR14"]
    distance = close.div(mm50).sub(1).where(mm50.ne(0))
    gap = macd - signal
    components: dict[str, tuple[pd.Series, pd.Series, float]] = {}

    def add(name: str, points: pd.Series, available: pd.Series, maximum: float) -> None:
        components[name] = (points.where(available), available, maximum)

    add("price_above_mm200", (close > mm200).astype(float) * 10, close.notna() & mm200.notna(), 10)
    add("mm50_above_mm200", (mm50 > mm200).astype(float) * 10, mm50.notna() & mm200.notna(), 10)
    add("mm200_rising", (mm200 > mm200.shift(20)).astype(float) * 10,
        mm200.notna() & mm200.shift(20).notna(), 10)
    pullback_points = pd.Series(0.0, index=frame.index)
    pullback_points.mask(distance.between(-.05, -.01, inclusive="both"), 15, inplace=True)
    pullback_points.mask(distance.between(-.01, .02, inclusive="neither"), 12, inplace=True)
    pullback_points.mask(distance.between(-.08, -.05, inclusive="left"), 8, inplace=True)
    pullback_points.mask(distance.between(.02, .06, inclusive="both"), 7, inplace=True)
    pullback_points.mask(distance.between(-.12, -.08, inclusive="left"), 3, inplace=True)
    pullback_points.mask(distance > .06, 2, inplace=True)
    add("distance_mm50", pullback_points, distance.notna(), 15)
    rsi_points = pd.Series(1.0, index=frame.index)
    rsi_points.mask(rsi < 32, 3, inplace=True)
    rsi_points.mask(rsi.between(32, 38, inclusive="left"), 6, inplace=True)
    rsi_points.mask(rsi.between(38, 48, inclusive="both"), 10, inplace=True)
    rsi_points.mask(rsi.between(48, 55, inclusive="right"), 7, inplace=True)
    rsi_points.mask(rsi.between(55, 65, inclusive="right"), 4, inplace=True)
    add("pullback_rsi", rsi_points, rsi.notna(), 10)
    atr_distance = close.sub(mm50).abs().div(atr).where(atr.gt(0))
    atr_distance_points = pd.Series(0.0, index=frame.index)
    atr_distance_points.mask(atr_distance <= 2.5, 2, inplace=True)
    atr_distance_points.mask(atr_distance <= 1.5, 4, inplace=True)
    atr_distance_points.mask(atr_distance <= .75, 5, inplace=True)
    add("support_atr", atr_distance_points, atr_distance.notna(), 5)
    add("macd_gap_improving", (gap > gap.shift(5)).astype(float) * 10,
        gap.notna() & gap.shift(5).notna(), 10)
    add("rsi_improving", ((rsi > rsi.shift(3)) & (rsi >= 40)).astype(float) * 8,
        rsi.notna() & rsi.shift(3).notna(), 8)
    add("price_recovery", (close > close.shift(3)).astype(float) * 7,
        close.notna() & close.shift(3).notna(), 7)
    add("macd_cross", (macd > signal).astype(float) * 5, macd.notna() & signal.notna(), 5)
    atr_percent = atr.div(close).where(close.gt(0))
    atr_risk = pd.Series(0.0, index=frame.index)
    atr_risk.mask(atr_percent <= .06, 1, inplace=True)
    atr_risk.mask(atr_percent <= .045, 3, inplace=True)
    atr_risk.mask(atr_percent <= .03, 4, inplace=True)
    atr_risk.mask(atr_percent < .02, 5, inplace=True)
    add("atr_risk", atr_risk, atr_percent.notna(), 5)
    extension = pd.Series(0.0, index=frame.index)
    extension.mask(distance.between(-.12, -.05, inclusive="left"), 2, inplace=True)
    extension.mask(distance.between(-.05, .05, inclusive="both"), 5, inplace=True)
    extension.mask(distance.between(.05, .10, inclusive="right"), 2, inplace=True)
    add("extension_mm50", extension, distance.notna(), 5)

    earned = sum((item[0].fillna(0) for item in components.values()), pd.Series(0.0, index=frame.index))
    available = sum((item[1].astype(float) * item[2] for item in components.values()),
                    pd.Series(0.0, index=frame.index))
    score = earned.div(available.where(available.gt(0))).mul(100)
    context_valid = ~((close < mm200) & (mm50 < mm200) & close.notna() & mm50.notna() & mm200.notna())
    score = score.where(context_valid, score.clip(upper=55))
    result = pd.DataFrame({
        "date": frame.index, "close": close, "timing_score": score,
        "timing_confidence": available, "MM50": mm50, "MM200": mm200,
        "RSI": rsi, "MACD": macd, "Signal MACD": signal, "ATR14": atr,
        "distance_mm50": distance, "distance_mm200": close.div(mm200).sub(1),
        "distance_from_20d_high": close.div(close.rolling(20).max()).sub(1),
        "long_term_context_valid": context_valid, "_position": range(len(frame)),
    }, index=frame.index)
    for name, (points, _, maximum) in components.items():
        result[f"points_{name}"] = points
        result[f"max_{name}"] = maximum
    return result


def _v3_prepared(data: pd.DataFrame) -> pd.DataFrame:
    """Retourne les indicateurs nécessaires à V3 sans modifier la série source."""
    if data.empty or "Close" not in data:
        return pd.DataFrame()
    required = {"MM50", "MM200", "RSI", "MACD", "Signal"}
    frame = data.copy() if required.issubset(data.columns) else calculate_indicators(data)
    if "ATR14" not in frame:
        frame["ATR14"] = calculate_atr(frame)
    return frame


def _v3_condition(key: str, label: str, available: bool, passed: bool = False,
                  detail: str = "", points: float = 0, maximum: float = 0) -> dict[str, Any]:
    return {"key": key, "label": label, "available": bool(available),
            "passed": bool(passed) if available else False, "detail": detail,
            "points": float(points) if available else None, "max": maximum}


def evaluate_v3_regime(data: pd.DataFrame) -> dict[str, Any]:
    """Évalue le contexte long terme à la dernière date disponible, causalement."""
    frame = _v3_prepared(data)
    if frame.empty:
        return {"status": "N/D", "conditions": [], "score": 0, "available_points": 0,
                "volatility": None, "volatility_class": "N/D", "extended": False}
    row = frame.iloc[-1]
    close, mm50, mm200 = row.get("Close"), row.get("MM50"), row.get("MM200")
    old_mm200 = frame["MM200"].iloc[-21] if len(frame) >= 21 else None
    specifications = (
        ("price_above_mm200", "Cours > MM200", close, mm200, 15),
        ("mm50_above_mm200", "MM50 > MM200", mm50, mm200, 15),
        ("mm200_rising", "MM200 montante sur 20 séances", mm200, old_mm200, 10),
    )
    conditions = []
    for key, label, left, right, maximum in specifications:
        available = valid_number(left) and valid_number(right)
        passed = available and float(left) > float(right)
        conditions.append(_v3_condition(key, label, available, passed,
                                         "validée" if passed else "non validée", maximum if passed else 0, maximum))
    returns = frame["Close"].pct_change().dropna().tail(252)
    volatility = returns.std() * sqrt(252) if len(returns) >= 20 else None
    if valid_number(volatility):
        volatility_class = ("favorable" if volatility < .25 else "acceptable" if volatility <= .40
                            else "prudence" if volatility <= .55 else "défavorable")
    else:
        volatility_class = "N/D"
    conditions.append(_v3_condition("volatility", "Volatilité du régime", valid_number(volatility),
                                     valid_number(volatility) and volatility <= .40,
                                     f"{float(volatility):.1%} — {volatility_class}" if valid_number(volatility) else "N/D"))
    distance_200 = float(close) / float(mm200) - 1 if valid_number(close) and valid_number(mm200) and mm200 else None
    distance_50 = float(close) / float(mm50) - 1 if valid_number(close) and valid_number(mm50) and mm50 else None
    extended = bool((distance_200 is not None and distance_200 > .35) or
                    (distance_50 is not None and distance_50 > .20))
    conditions.append(_v3_condition("extension", "Absence d'extension extrême", distance_50 is not None or distance_200 is not None,
                                     not extended, "⚠️ Tendance haussière mais cours fortement étendu." if extended else "Extension acceptable"))
    core = conditions[:3]
    available_core = [condition for condition in core if condition["available"]]
    ratio = sum(condition["passed"] for condition in available_core) / len(available_core) if available_core else None
    status = "N/D" if ratio is None else "Favorable" if ratio >= 1 else "Mitigé" if ratio >= 2 / 3 else "Défavorable"
    return {"status": status, "conditions": conditions,
            "score": sum(condition["points"] or 0 for condition in core),
            "available_points": sum(condition["max"] for condition in core if condition["available"]),
            "volatility": float(volatility) if valid_number(volatility) else None,
            "volatility_class": volatility_class, "extended": extended,
            "distance_mm50": distance_50, "distance_mm200": distance_200}


def evaluate_v3_setup(data: pd.DataFrame, regime: dict[str, Any] | None = None) -> dict[str, Any]:
    """Cherche un repli surveillable ; ce résultat ne constitue jamais une entrée."""
    frame = _v3_prepared(data)
    if frame.empty:
        return {"status": "N/D", "conditions": [], "score": 0, "available_points": 0}
    regime = regime or evaluate_v3_regime(frame)
    row = frame.iloc[-1]
    close, mm50, rsi, atr = row.get("Close"), row.get("MM50"), row.get("RSI"), row.get("ATR14")
    distance = float(close) / float(mm50) - 1 if valid_number(close) and valid_number(mm50) and mm50 else None
    atr_distance = abs(float(close) - float(mm50)) / float(atr) if all(valid_number(x) for x in (close, mm50, atr)) and atr > 0 else None
    distance_points = 15 if distance is not None and -.05 <= distance <= .03 else 9 if distance is not None and -.08 <= distance < -.05 else 6 if distance is not None and .03 < distance <= .08 else 0
    rsi_points = 10 if valid_number(rsi) and 38 <= rsi <= 50 else 6 if valid_number(rsi) and 32 <= rsi <= 55 else 0
    atr_points = 5 if atr_distance is not None and atr_distance <= 1 else 3 if atr_distance is not None and atr_distance <= 2 else 0
    conditions = [
        _v3_condition("distance_mm50", "Cours proche MM50", distance is not None,
                      distance is not None and -.05 <= distance <= .03,
                      f"{distance:+.1%}" if distance is not None else "N/D", distance_points, 15),
        _v3_condition("rsi_setup", "RSI de détente", valid_number(rsi),
                      valid_number(rsi) and 35 <= rsi <= 52,
                      f"RSI {float(rsi):.1f}" if valid_number(rsi) else "N/D", rsi_points, 10),
        _v3_condition("atr_proximity", "Distance à la MM50 en ATR", atr_distance is not None,
                      atr_distance is not None and atr_distance <= 2,
                      f"{atr_distance:.1f} ATR" if atr_distance is not None else "N/D", atr_points, 5),
        _v3_condition("support", "Support récent", False, detail="Support N/D (ignoré dans le backtest causal)"),
    ]
    context_ok = regime["status"] in {"Favorable", "Mitigé"}
    present = (context_ok and distance is not None and -.05 <= distance <= .03 and
               valid_number(rsi) and 35 <= rsi <= 52 and not regime.get("extended", False))
    convergence = sum(condition["passed"] for condition in conditions[:3] if condition["available"])
    status = "Présent" if present else "Possible" if context_ok and convergence >= 2 and not regime.get("extended", False) else "Absent"
    if not any(condition["available"] for condition in conditions[:3]):
        status = "N/D"
    return {"status": status, "conditions": conditions,
            "score": distance_points + rsi_points + atr_points,
            "available_points": sum(condition["max"] for condition in conditions if condition["available"]),
            "distance_mm50": distance, "atr_distance": atr_distance, "rsi": float(rsi) if valid_number(rsi) else None}


def evaluate_v3_trigger(data: pd.DataFrame) -> dict[str, Any]:
    """Détecte une reprise à T exclusivement à partir de T et de séances passées."""
    frame = _v3_prepared(data)
    if frame.empty:
        return {"status": "N/D", "conditions": [], "score": 0, "available_points": 0}
    row = frame.iloc[-1]
    close, rsi, macd, signal, mm50 = (row.get(key) for key in ("Close", "RSI", "MACD", "Signal", "MM50"))
    previous = lambda column, periods: frame[column].iloc[-periods - 1] if column in frame and len(frame) > periods else None
    gap = macd - signal if valid_number(macd) and valid_number(signal) else None
    gap3_values = (previous("MACD", 3), previous("Signal", 3))
    gap5_values = (previous("MACD", 5), previous("Signal", 5))
    gap3 = gap3_values[0] - gap3_values[1] if all(valid_number(x) for x in gap3_values) else None
    gap5 = gap5_values[0] - gap5_values[1] if all(valid_number(x) for x in gap5_values) else None
    prior_below = None
    if len(frame) >= 2 and {"Close", "MM50"}.issubset(frame.columns):
        prior = frame.iloc[max(0, len(frame) - 6):-1]
        available_prior = prior["Close"].notna() & prior["MM50"].notna()
        if available_prior.any():
            prior_below = bool((prior.loc[available_prior, "Close"] < prior.loc[available_prior, "MM50"]).any())
    definitions = [
        ("price_recovery", "Cours en reprise", all(valid_number(x) for x in (close, previous("Close", 3), previous("Close", 1))),
         valid_number(close) and valid_number(previous("Close", 3)) and valid_number(previous("Close", 1)) and close > previous("Close", 3) and close > previous("Close", 1), 7),
        ("rsi_improving", "RSI en amélioration", valid_number(rsi) and valid_number(previous("RSI", 3)),
         valid_number(rsi) and valid_number(previous("RSI", 3)) and rsi > previous("RSI", 3) and rsi >= 42, 6),
        ("macd_gap_improving", "MACD gap en amélioration", all(valid_number(x) for x in (gap, gap3, gap5)),
         all(valid_number(x) for x in (gap, gap3, gap5)) and gap > gap3 and gap > gap5, 7),
        ("macd_cross", "MACD > signal", valid_number(macd) and valid_number(signal),
         valid_number(macd) and valid_number(signal) and macd > signal, 5),
        ("mm50_reclaim", "Reprise de la MM50", valid_number(close) and valid_number(mm50) and prior_below is not None,
         valid_number(close) and valid_number(mm50) and close > mm50 and prior_below is True, 5),
    ]
    conditions = [_v3_condition(key, label, available, passed,
                                "validé" if passed else "non validé", maximum if passed else 0, maximum)
                  for key, label, available, passed, maximum in definitions]
    available = [condition for condition in conditions if condition["available"]]
    passed = sum(condition["passed"] for condition in available)
    status = "N/D" if not available else "Fort" if passed >= 3 else "Partiel" if passed >= 1 else "Absent"
    return {"status": status, "conditions": conditions, "validated": passed,
            "score": sum(condition["points"] or 0 for condition in conditions),
            "available_points": sum(condition["max"] for condition in available)}


def build_v3_entry_status(regime: dict[str, Any], setup: dict[str, Any],
                          trigger: dict[str, Any]) -> str:
    """Traduit les trois états V3 en une décision lisible, sans score principal."""
    r, s, t = regime.get("status"), setup.get("status"), trigger.get("status")
    if r == "Défavorable": return "🔴 Régime défavorable — aucune entrée V3"
    if r == "Favorable" and s == "Présent" and t == "Fort": return "🟢 Configuration d'entrée techniquement confirmée"
    if r == "Favorable" and s == "Présent" and t == "Partiel": return "🟠 Setup intéressant — confirmation en cours"
    if r == "Favorable" and s == "Présent" and t == "Absent": return "🟠 Setup intéressant — attendre le déclencheur"
    if r == "Favorable" and s == "Absent": return "⚪ Tendance saine — pas de setup d'entrée actuellement"
    if r == "Mitigé" and s == "Présent" and t == "Fort": return "🟡 Reprise détectée dans un contexte encore mitigé"
    if "N/D" in {r, s, t}: return "⚪ V3 non déterminable — données insuffisantes"
    return "🟡 Configuration incomplète — rester sélectif"


def calculate_rigorous_entry_v3(data: pd.DataFrame) -> dict[str, Any]:
    """Calcule V3 à la dernière séance : régime, setup, trigger, puis indice technique."""
    regime = evaluate_v3_regime(data)
    setup = evaluate_v3_setup(data, regime)
    trigger = evaluate_v3_trigger(data)
    strength = regime["score"] + setup["score"] + trigger["score"]
    if regime["status"] == "Défavorable":
        strength = min(strength, 55)
    conditions = [{**condition, "stage": stage} for stage, result in
                  (("Régime", regime), ("Setup", setup), ("Trigger", trigger))
                  for condition in result["conditions"]]
    return {"regime": regime, "setup": setup, "trigger": trigger,
            "status": build_v3_entry_status(regime, setup, trigger), "conditions": conditions,
            "metrics": {"v3_signal_strength": float(strength),
                        "available_points": regime["available_points"] + setup["available_points"] + trigger["available_points"]}}


def calculate_v3_timing_series(data: pd.DataFrame) -> pd.DataFrame:
    """Construit les états V3 causaux et les événements Early / Confirmed."""
    frame = _v3_prepared(data)
    rows = []
    for position in range(len(frame)):
        result = calculate_rigorous_entry_v3(frame.iloc[:position + 1])
        rows.append({"date": frame.index[position], "close": frame["Close"].iloc[position],
                     "regime_status": result["regime"]["status"], "setup_status": result["setup"]["status"],
                     "trigger_status": result["trigger"]["status"],
                     "v3_signal_strength": result["metrics"]["v3_signal_strength"], "_position": position})
    series = pd.DataFrame(rows, index=frame.index)
    eligible = (series["regime_status"] == "Favorable") & (series["setup_status"] == "Présent")
    early_state = eligible & (series["trigger_status"] == "Partiel")
    confirmed_state = eligible & (series["trigger_status"] == "Fort")
    series["v3_early"] = early_state & ~early_state.shift(1, fill_value=False)
    series["v3_confirmed"] = confirmed_state & ~confirmed_state.shift(1, fill_value=False)
    return series


def extract_v3_signals(series: pd.DataFrame, signal_type: str,
                       min_gap: int = MIN_SIGNAL_GAP) -> pd.DataFrame:
    """Extrait uniquement les transitions d'état V3, puis applique l'espacement partagé."""
    column = "v3_early" if signal_type.lower() == "early" else "v3_confirmed"
    if series.empty or column not in series:
        return series.iloc[0:0].copy()
    candidates = series.loc[series[column].fillna(False)].copy()
    kept, last = [], None
    for position in candidates["_position"].astype(int):
        if last is None or position - last >= min_gap:
            kept.append(position); last = position
    return candidates[candidates["_position"].isin(kept)]


def analyze_v3_confirmations(early: pd.DataFrame, confirmed: pd.DataFrame,
                             timeline: pd.DataFrame, window: int = 15) -> dict[str, Any]:
    """Associe chaque Early au premier Confirmed suivant, sans prétention prédictive."""
    confirmed_positions = confirmed.get("_position", pd.Series(dtype=int)).astype(int).tolist()
    closes = timeline["close"].reset_index(drop=True)
    records = []
    for _, row in early.iterrows():
        start = int(row["_position"])
        matches = [position for position in confirmed_positions if start < position <= start + window]
        end = min(matches) if matches else None
        performance = (float(closes.iloc[end]) / float(closes.iloc[start]) - 1
                       if end is not None and end < len(closes) and valid_number(closes.iloc[start]) else None)
        records.append({"early_position": start, "confirmed_position": end,
                        "delay": end - start if end is not None else None,
                        "performance_to_confirmation": performance,
                        "outcome": "Confirmé" if end is not None else "Setup non confirmé"})
    confirmed_records = [record for record in records if record["confirmed_position"] is not None]
    return {"records": records, "early_count": len(records), "confirmed_count": len(confirmed_records),
            "confirmation_rate": len(confirmed_records) / len(records) if records else None,
            "average_delay": pd.Series([record["delay"] for record in confirmed_records]).mean() if confirmed_records else None,
            "median_delay": pd.Series([record["delay"] for record in confirmed_records]).median() if confirmed_records else None,
            "average_performance_to_confirmation": pd.Series([record["performance_to_confirmation"] for record in confirmed_records]).mean() if confirmed_records else None,
            "median_performance_to_confirmation": pd.Series([record["performance_to_confirmation"] for record in confirmed_records]).median() if confirmed_records else None,
            "unconfirmed": early.iloc[[i for i, record in enumerate(records) if record["confirmed_position"] is None]].copy() if records else early.iloc[0:0].copy()}


def extract_threshold_signals(series: pd.DataFrame, threshold: float,
                              min_gap: int = MIN_SIGNAL_GAP) -> pd.DataFrame:
    """Extrait les franchissements haussiers avec le même espacement que V1."""
    if series.empty or "timing_score" not in series:
        return series.iloc[0:0].copy()
    candidates = series.loc[(series["timing_score"] >= threshold) &
                            (series["timing_score"].shift(1) < threshold)].copy()
    kept, last = [], None
    for position in candidates["_position"].astype(int):
        if last is None or position - last >= min_gap:
            kept.append(position); last = position
    return candidates[candidates["_position"].isin(kept)].copy()


@st.cache_data(ttl=1800, show_spinner=False)
def load_timing_lab_histories(tickers: tuple[str, ...], period: str) -> dict[str, pd.DataFrame]:
    """Télécharge en une requête les historiques ajustés du Lab et leur warm-up."""
    years = {"1y": 1, "2y": 2, "3y": 3, "5y": 5}.get(period, 3)
    analysis_start = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize() - pd.DateOffset(years=years)
    start = analysis_start - timedelta(days=300)
    raw = yf.download(list(tickers), start=start.date(), interval="1d", auto_adjust=True,
                      progress=False, threads=True, group_by="ticker")
    histories: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        frame = raw[ticker].copy() if isinstance(raw.columns, pd.MultiIndex) and ticker in raw.columns.levels[0] else raw.copy()
        if "Close" in frame:
            frame = frame.dropna(subset=["Close"])
            frame.attrs["analysis_start"] = analysis_start
            histories[ticker] = frame
    return histories


def sample_size_warning(n: int) -> str | None:
    if n < 10:
        return "⚠️ Échantillon très faible : résultat peu fiable."
    if n < 30:
        return "🟠 Échantillon limité : interpréter avec prudence."
    return None


def calculate_alpha(signal_return: Any, baseline_return: Any) -> float | None:
    """Retourne l'écart absolu signal-baseline (exprimé ensuite en points)."""
    if not valid_number(signal_return) or not valid_number(baseline_return):
        return None
    return float(signal_return) - float(baseline_return)


def format_alpha(value: Any) -> str:
    """Formate un rendement décimal en points de pourcentage, jamais en ratio relatif."""
    return f"{float(value) * 100:+.1f} pt" if valid_number(value) else "N/D"


def split_validation_universes(conception: list[str], validation: list[str],
                               limit: int = 20) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Déduplique les univers et retire systématiquement leur chevauchement."""
    designed = tuple(dict.fromkeys(symbol.upper() for symbol in conception if symbol))
    candidates = tuple(dict.fromkeys(symbol.upper() for symbol in validation if symbol))[:limit]
    designed_set = set(designed)
    return tuple(symbol for symbol in candidates if symbol not in designed_set), \
        tuple(symbol for symbol in candidates if symbol in designed_set)


def aggregate_out_of_sample(raw_stats: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Agrège sans imputation : une action pèse une fois, sauf l'alpha pondéré."""
    rows: list[dict[str, Any]] = []
    for horizon in BACKTEST_HORIZONS:
        for engine in ("V1", "V2", "V3 Early", "V3 Confirmed"):
            available, baselines = [], []
            for item in raw_stats:
                if engine not in item:
                    continue
                stats, baseline = item[engine][horizon], item["baseline"][horizon]
                alpha = calculate_alpha(stats["mean"], baseline["mean"])
                if valid_number(alpha):
                    available.append((float(stats["mean"]), float(alpha), stats["observations"],
                                      stats.get("drawdown_mean"), stats.get("drawdown_median"),
                                      stats.get("drawdown_worst")))
                    baselines.append(float(baseline["mean"]))
            alphas = [x[1] for x in available]
            observations = sum(x[2] for x in available)
            drawdowns = [float(x[3]) for x in available if valid_number(x[3])]
            med_drawdowns = [float(x[4]) for x in available if valid_number(x[4])]
            worst_drawdowns = [float(x[5]) for x in available if valid_number(x[5])]
            positive = sum(alpha > 0 for alpha in alphas)
            rows.append({
                "engine": engine, "horizon": horizon, "assets": len(alphas),
                "performance_mean": pd.Series([x[0] for x in available]).mean() if available else None,
                "performance_median": pd.Series([x[0] for x in available]).median() if available else None,
                "baseline_mean": pd.Series(baselines).mean() if baselines else None,
                "alpha_mean": pd.Series(alphas).mean() if alphas else None,
                "alpha_median": pd.Series(alphas).median() if alphas else None,
                "alpha_weighted": (sum(x[1] * x[2] for x in available) / observations
                                   if observations else None),
                "positive_assets": positive, "nonpositive_assets": len(alphas) - positive,
                "positive_ratio": positive / len(alphas) if alphas else None,
                "observations": observations,
                "drawdown_mean": pd.Series(drawdowns).mean() if drawdowns else None,
                "drawdown_median": pd.Series(med_drawdowns).median() if med_drawdowns else None,
                "drawdown_worst": min(worst_drawdowns) if worst_drawdowns else None,
            })
    return rows


def out_of_sample_robustness(summary_20: dict[str, Any], summary_60: dict[str, Any],
                             baseline_drawdown: Any = None) -> str:
    """Classement prudent et entièrement déterministe de la robustesse historique."""
    mean, median, ratio = (summary_20.get("alpha_mean"), summary_20.get("alpha_median"),
                           summary_20.get("positive_ratio"))
    if not all(valid_number(x) for x in (mean, median, ratio)):
        return "Faible"
    if (mean <= 0 and median <= 0) or ratio < .4:
        return "Faible"
    signals_ok = summary_20.get("observations", 0) >= 30
    dd = summary_20.get("drawdown_mean")
    dd_ok = not (valid_number(dd) and valid_number(baseline_drawdown) and dd < baseline_drawdown - .03)
    sixty_ok = not (valid_number(summary_60.get("alpha_mean")) and summary_60["alpha_mean"] < -.03)
    return "Encourageante" if mean > 0 and median > 0 and ratio > .5 and signals_ok and dd_ok and sixty_ok else "Mitigée"


def build_out_of_sample_interpretation(engine: str, summary_20: dict[str, Any],
                                       robustness: str) -> str:
    """Produit une lecture historique déterministe, sans prétention prédictive."""
    mean, median = summary_20.get("alpha_mean"), summary_20.get("alpha_median")
    positive, assets = summary_20.get("positive_assets", 0), summary_20.get("assets", 0)
    if robustness == "Faible":
        return (f"{engine} ne montre pas d'avantage robuste hors échantillon : l'alpha moyen à 20 séances "
                f"est {format_alpha(mean)} et {positive} actions sur {assets} battent leur baseline. "
                "Ce constat est historique et ne constitue pas une prédiction.")
    if robustness == "Encourageante":
        return (f"{engine} montre un avantage historique encourageant hors échantillon sur cet univers : "
                f"alpha moyen {format_alpha(mean)}, alpha médian {format_alpha(median)}, et baseline battue "
                f"sur {positive} actions sur {assets}. Une validation sur d'autres périodes reste nécessaire.")
    return (f"Les résultats hors échantillon de {engine} sont mitigés : alpha moyen {format_alpha(mean)}, "
            f"alpha médian {format_alpha(median)}, et {positive} actions sur {assets} au-dessus de leur baseline. "
            "Ils doivent être confirmés sur d'autres périodes.")


def build_horizon_stability(engine: str, summaries: dict[int, dict[str, Any]]) -> str:
    values = {h: summaries[h].get("alpha_mean") for h in BACKTEST_HORIZONS}
    detail = ", ".join(f"{h}j {format_alpha(values[h])}" for h in BACKTEST_HORIZONS)
    if valid_number(values[20]) and values[20] > 0 and valid_number(values[60]) and values[60] <= 0:
        conclusion = f"L'avantage {engine} apparaît principalement à moyen terme et ne se maintient pas à 60 séances."
    elif all(valid_number(values[h]) and values[h] > 0 for h in BACKTEST_HORIZONS):
        conclusion = f"L'alpha historique de {engine} reste positif sur les trois horizons étudiés."
    else:
        conclusion = f"L'alpha historique de {engine} varie selon l'horizon."
    return f"{detail}. {conclusion} Il ne s'agit pas d'une prédiction."


def build_backtest_interpretation(high: dict[str, Any], baseline: dict[str, Any],
                                  low: dict[str, Any], threshold: int) -> str:
    """Lecture déterministe de l'horizon 20, favorable ou défavorable sans sélection."""
    hs, bs, ls = high[20], baseline[20], low[20]
    n = hs["observations"]
    if not all(valid_number(item["mean"]) for item in (hs, bs, ls)):
        return "Les observations disponibles ne suffisent pas pour comparer honnêtement les groupes à 20 séances."
    detail = (f"Sur {n} signaux Timing ≥ {threshold}, la performance moyenne après 20 séances "
              f"a été de {format_percentage(hs['mean'])}, contre {format_percentage(bs['mean'])} "
              f"pour l’ensemble des séances admissibles. {hs['positive']:.0%} des signaux historiques "
              "étudiés ont terminé positifs après 20 séances. ")
    if hs["mean"] > bs["mean"] and hs["mean"] > ls["mean"]:
        conclusion = "Les signaux élevés ont historiquement mieux performé dans cet échantillon."
    else:
        conclusion = ("Les signaux Timing élevés n’ont pas historiquement surperformé les séances ordinaires "
                      "sur cet horizon ; le seuil actuel ne montre donc pas d’avantage clair.")
    warning = sample_size_warning(n)
    return detail + conclusion + (f" {warning}" if warning else "") + " Les résultats passés ne garantissent pas les résultats futurs."


def render_backtest_charts(timing: pd.DataFrame, signals: pd.DataFrame, threshold: int) -> None:
    price = go.Figure()
    price.add_trace(go.Scatter(x=timing.index, y=timing["close"], name="Cours", mode="lines"))
    if not signals.empty:
        price.add_trace(go.Scatter(x=signals.index, y=signals["close"], name="Signaux",
                                   mode="markers", marker={"size": 9, "color": "#34d399"}))
    price.update_layout(title="Cours ajusté et signaux", height=330, margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(price, use_container_width=True)
    score = go.Figure(go.Scatter(x=timing.index, y=timing["timing_score"], name="Timing", mode="lines"))
    score.add_hline(y=threshold, line_dash="dash", line_color="#fb923c")
    score.update_yaxes(range=[0, 100], title="Timing /100")
    score.update_layout(height=280, margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(score, use_container_width=True)
    returns = signals.get("return_20", pd.Series(dtype=float)).dropna() * 100
    if not returns.empty:
        histogram = go.Figure(go.Histogram(x=returns, nbinsx=15))
        histogram.add_vline(x=0, line_dash="dash", line_color="#f87171")
        histogram.update_layout(title="Distribution des performances 20 séances après signal",
                                xaxis_title="Performance (%)", height=320,
                                margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(histogram, use_container_width=True)


def _lab_variant(timing: pd.DataFrame, threshold: float,
                 timeline: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    signals = extract_threshold_signals(timing, threshold)
    signals = calculate_signal_drawdowns(calculate_forward_returns(signals, timeline), timeline)
    return signals, calculate_backtest_statistics(signals)


def _v3_variant(series: pd.DataFrame, signal_type: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    signals = extract_v3_signals(series, signal_type)
    signals = calculate_signal_drawdowns(calculate_forward_returns(signals, series), series)
    return signals, calculate_backtest_statistics(signals)


def _render_v3_snapshot(result: dict[str, Any]) -> None:
    """Affichage vertical et mobile des trois étapes, avant l'indice secondaire."""
    st.markdown("### V3 — Régime / Setup / Trigger")
    icons = {"Favorable": "✅", "Mitigé": "🟡", "Défavorable": "❌", "Présent": "✅",
             "Possible": "🟡", "Absent": "❌", "Fort": "✅", "Partiel": "🟠", "N/D": "⚪"}
    for title, key in (("Régime", "regime"), ("Setup", "setup"), ("Trigger", "trigger")):
        stage = result[key]
        st.markdown(f"**{title} : {icons.get(stage['status'], '⚪')} {stage['status']}**")
        for condition in stage["conditions"]:
            marker = "⚪" if not condition["available"] else "✅" if condition["passed"] else "❌"
            st.caption(f"{marker} {condition['label']} — {condition['detail']}")
    st.info(result["status"])
    st.caption("V3 sépare le contexte, le setup et le déclenchement afin d'éviter de confondre tendance saine et moment d'entrée.")
    st.caption(f"Indice expérimental V3 (métrique technique de backtest) : {result['metrics']['v3_signal_strength']:.0f}/100")


def _lab_eligible(history: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aligne strictement V1 et V2 sur la même fenêtre et les mêmes séances."""
    v1 = calculate_historical_timing_series(history, "Investisseur")
    v2 = calculate_pullback_timing_series(history)
    start = history.attrs.get("analysis_start", v1.index.min())
    dates = v1.index.intersection(v2.index)
    mask = (dates >= start) & v1.loc[dates, "MM200"].notna() & v2.loc[dates, "MM200"].notna()
    dates = dates[mask]
    aligned_v1, aligned_v2 = v1.loc[dates].copy(), v2.loc[dates].copy()
    positions = range(len(dates))
    aligned_v1["_position"] = positions
    aligned_v2["_position"] = range(len(dates))
    return aligned_v1, aligned_v2


def _render_lab_chart(v1: pd.DataFrame, v2: pd.DataFrame, s1: pd.DataFrame,
                      s2: pd.DataFrame, threshold1: float, threshold2: float) -> None:
    price = go.Figure(go.Scatter(x=v1.index, y=v1["close"], name="Cours", mode="lines"))
    common = s1.index.intersection(s2.index)
    only1, only2 = s1.index.difference(common), s2.index.difference(common)
    for dates, label, symbol, color, size in ((only1, "V1 uniquement", "triangle-up", "#60a5fa", 9),
                                               (only2, "V2 uniquement", "diamond", "#fb923c", 9),
                                               (common, "V1 + V2", "star", "#34d399", 14)):
        if len(dates):
            price.add_trace(go.Scatter(x=dates, y=v1.loc[dates, "close"], name=label,
                                       mode="markers", marker={"symbol": symbol, "color": color, "size": size}))
    price.update_layout(title="Cours ajusté — signaux distincts et communs", height=360,
                        margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(price, use_container_width=True)
    scores = go.Figure()
    scores.add_trace(go.Scatter(x=v1.index, y=v1["timing_score"], name="V1 Timing actuel"))
    scores.add_trace(go.Scatter(x=v2.index, y=v2["timing_score"], name="V2 Repli + Reprise"))
    scores.add_hline(y=threshold1, line_dash="dash", line_color="#60a5fa", annotation_text="Seuil V1")
    if threshold2 != threshold1:
        scores.add_hline(y=threshold2, line_dash="dot", line_color="#fb923c", annotation_text="Seuil V2")
    scores.update_yaxes(range=[0, 100], title="Convergence /100")
    scores.update_layout(height=300, margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(scores, use_container_width=True)


def _lab_table(stats1: dict[str, Any], stats2: dict[str, Any],
               baseline: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for horizon in BACKTEST_HORIZONS:
        a, b, base = stats1[horizon], stats2[horizon], baseline[horizon]
        rows.append({"Horizon": f"{horizon} séances", "V1 perf moy.": format_percentage(a["mean"]),
                     "V2 perf moy.": format_percentage(b["mean"]), "Baseline": format_percentage(base["mean"]),
                     "V1 médiane": format_percentage(a["median"]), "V2 médiane": format_percentage(b["median"]),
                     "V1 positifs": f"{a['positive']:.0%}" if valid_number(a["positive"]) else "N/D",
                     "V2 positifs": f"{b['positive']:.0%}" if valid_number(b["positive"]) else "N/D",
                     "V1 drawdown": format_percentage(a["drawdown_mean"]),
                     "V2 drawdown": format_percentage(b["drawdown_mean"])})
    return pd.DataFrame(rows)


def _calculate_validation_assets(tickers: tuple[str, ...], histories: dict[str, pd.DataFrame],
                                 threshold1: float, threshold2: float) -> tuple[list[dict[str, Any]], list[str]]:
    """Mesure les quatre signaux sur une fenêtre, une baseline et un warm-up identiques."""
    results, ignored = [], []
    for symbol in tickers:
        history = histories.get(symbol, pd.DataFrame())
        if len(history) < 220:
            ignored.append(symbol)
            continue
        v1, v2 = _lab_eligible(history)
        if v1.empty or v2.empty:
            ignored.append(symbol)
            continue
        s1, stats1 = _lab_variant(v1, threshold1, v1)
        s2, stats2 = _lab_variant(v2, threshold2, v2)
        # La série V3 est la détection existante, calculée sur tout le warm-up puis
        # seulement découpée sur les dates admissibles communes à V1/V2.
        v3 = calculate_v3_timing_series(history).reindex(v1.index).dropna(subset=["close"]).copy()
        v3["_position"] = range(len(v3))
        early, stats_early = _v3_variant(v3, "early")
        confirmed, stats_confirmed = _v3_variant(v3, "confirmed")
        confirmation = analyze_v3_confirmations(early, confirmed, v3)
        unconfirmed_rows = calculate_signal_drawdowns(
            calculate_forward_returns(confirmation["unconfirmed"], v3), v3)
        unconfirmed_stats = calculate_backtest_statistics(unconfirmed_rows)
        baseline_rows = calculate_signal_drawdowns(calculate_forward_returns(v1, v1), v1)
        baseline = calculate_backtest_statistics(baseline_rows)
        results.append({"ticker": symbol, "V1": stats1, "V2": stats2,
                        "V3 Early": stats_early, "V3 Confirmed": stats_confirmed,
                        "baseline": baseline, "signals_V1": len(s1), "signals_V2": len(s2),
                        "signals_V3 Early": len(early), "signals_V3 Confirmed": len(confirmed),
                        "v3_confirmation": confirmation,
                        "v3_unconfirmed": unconfirmed_stats})
    return results, ignored


def _validation_export(raw_stats: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for item in raw_stats:
        row = {"Ticker": item["ticker"], "V1 signals": item["signals_V1"],
               "V2 signals": item["signals_V2"],
               "V3 Early signals": item["signals_V3 Early"],
               "V3 Confirmed signals": item["signals_V3 Confirmed"]}
        for horizon in BACKTEST_HORIZONS:
            baseline = item["baseline"][horizon]["mean"]
            row[f"baseline {horizon}"] = baseline
            for engine in ("V1", "V2", "V3 Early", "V3 Confirmed"):
                stats = item[engine][horizon]
                row[f"{engine} perf {horizon}"] = stats["mean"]
                row[f"{engine} alpha {horizon}"] = calculate_alpha(stats["mean"], baseline)
                row[f"{engine} drawdown mean {horizon}"] = stats["drawdown_mean"]
                row[f"{engine} drawdown median {horizon}"] = stats["drawdown_median"]
                row[f"{engine} drawdown worst {horizon}"] = stats["drawdown_worst"]
        rows.append(row)
    return pd.DataFrame(rows)


def build_v3_oos_interpretation(early: dict[str, Any], confirmed: dict[str, Any]) -> str:
    """Répond à la question Early/Confirmed avec quatre cas déterministes."""
    em, ed, er = early.get("alpha_mean"), early.get("alpha_median"), early.get("positive_ratio")
    cm, cd, cr = confirmed.get("alpha_mean"), confirmed.get("alpha_median"), confirmed.get("positive_ratio")
    if all(valid_number(x) for x in (em, ed, er, cm)) and em > 0 and ed > 0 and er > .5 and em > cm:
        return ("🟢 Le Setup V3 semble apporter davantage d'information que le Trigger actuel. "
                "Le Trigger pourrait être trop tardif ou trop restrictif.")
    if all(valid_number(x) for x in (em, ed, cm, cd)) and em <= 0 and ed <= 0 and cm <= 0 and cd <= 0:
        return ("🔴 V3 ne montre pas d'avantage robuste hors échantillon. Ni le Setup précoce ni "
                "la confirmation actuelle ne produisent un edge historique clair.")
    if all(valid_number(x) for x in (cm, em, cr)) and cm > em and cm > 0 and cr > .5:
        return "🟢 La confirmation V3 semble historiquement améliorer les setups Early sur cet échantillon."
    return "🟠 Les résultats V3 sont mitigés et dépendent de l'horizon ou des actions étudiées."


def aggregate_v3_confirmations(raw_stats: list[dict[str, Any]]) -> dict[str, Any]:
    """Agrège les appariements Early→Confirmed et les setups non confirmés, sans imputation."""
    records = [record for item in raw_stats for record in item["v3_confirmation"]["records"]]
    paired = [record for record in records if record["confirmed_position"] is not None]
    unconfirmed = sum(item["v3_confirmation"]["early_count"] - item["v3_confirmation"]["confirmed_count"]
                      for item in raw_stats)
    delays = pd.Series([record["delay"] for record in paired], dtype=float)
    costs = pd.Series([record["performance_to_confirmation"] for record in paired], dtype=float)
    return {"early": len(records), "confirmed": len(paired), "unconfirmed": unconfirmed,
            "rate": len(paired) / len(records) if records else None,
            "delay_mean": delays.mean() if len(delays) else None,
            "delay_median": delays.median() if len(delays) else None,
            "cost_mean": costs.mean() if len(costs) else None,
            "cost_median": costs.median() if len(costs) else None}


def aggregate_unconfirmed_v3_alpha(raw_stats: list[dict[str, Any]]) -> dict[str, Any]:
    """Agrège à poids égal les alphas 20j non confirmés disponibles par action."""
    alphas = []
    for item in raw_stats:
        unconfirmed = item["v3_unconfirmed"][20]["mean"]
        baseline = item["baseline"][20]["mean"]
        alpha = calculate_alpha(unconfirmed, baseline)
        if valid_number(alpha):
            alphas.append(float(alpha))
    values = pd.Series(alphas, dtype=float)
    return {"alpha_mean": values.mean() if len(values) else None,
            "alpha_median": values.median() if len(values) else None,
            "assets": len(values)}


def _render_out_of_sample_results(result: dict[str, Any]) -> None:
    raw_stats, conception_stats = result["validation"], result["conception"]
    ignored = result["ignored"]
    if ignored:
        st.warning(f"{len(ignored)} ticker(s) ignorés faute de données suffisantes : {', '.join(ignored)}.")
    if not raw_stats:
        st.error("Aucun historique hors échantillon exploitable.")
        return
    valid_assets = len(raw_stats)
    if valid_assets < 5:
        st.warning("⚠️ Univers de validation trop petit pour tirer une conclusion robuste.")
    elif valid_assets < 10:
        st.warning("🟠 Univers de validation limité.")

    engines = ("V1", "V2", "V3 Early", "V3 Confirmed")
    summaries = aggregate_out_of_sample(raw_stats)
    by_engine = {engine: {row["horizon"]: row for row in summaries if row["engine"] == engine}
                 for engine in engines}
    baseline_dd = pd.Series([item["baseline"][20]["drawdown_mean"] for item in raw_stats
                             if valid_number(item["baseline"][20]["drawdown_mean"])]).mean()
    robustness = {engine: out_of_sample_robustness(by_engine[engine][20], by_engine[engine][60], baseline_dd)
                  for engine in engines}

    st.markdown("### 📊 Synthèse principale — quatre moteurs")
    engine_rows = []
    for engine in engines:
        row = {"Moteur": engine}
        for horizon in BACKTEST_HORIZONS:
            row[f"Alpha {horizon}j"] = format_alpha(by_engine[engine][horizon]["alpha_mean"])
        summary20 = by_engine[engine][20]
        row["Alpha médian 20j"] = format_alpha(summary20["alpha_median"])
        row["Actions alpha + 20j"] = f"{summary20['positive_assets']} / {summary20['assets']}"
        engine_rows.append(row)
    st.dataframe(pd.DataFrame(engine_rows), use_container_width=True, hide_index=True)

    st.markdown("### 📊 Validation hors échantillon — 20 séances")
    for engine in engines:
        summary = by_engine[engine][20]
        title = f"{engine} — Validation hors échantillon" if engine.startswith("V3") else engine
        with st.container(border=True):
            st.markdown(f"#### {title}")
            st.write(f"Performance moyenne : **{format_percentage(summary['performance_mean'])}** · "
                     f"Performance médiane : **{format_percentage(summary['performance_median'])}**  \n"
                     f"Baseline : **{format_percentage(summary['baseline_mean'])}** · "
                     f"Alpha moyen : **{format_alpha(summary['alpha_mean'])}** · "
                     f"Alpha médian : **{format_alpha(summary['alpha_median'])}**  \n"
                     f"Alpha pondéré par observations : **{format_alpha(summary['alpha_weighted'])}** · "
                     f"Actions avec alpha positif : **{summary['positive_assets']} / {summary['assets']}** · "
                     f"Nombre total de signaux : **{summary['observations']}**  \n"
                     f"Drawdown moyen : **{format_percentage(summary['drawdown_mean'])}** · "
                     f"Pire drawdown : **{format_percentage(summary['drawdown_worst'])}** · "
                     f"Robustesse : **{robustness[engine]}**")
            st.write(build_out_of_sample_interpretation(engine, summary, robustness[engine]))
            if warning := sample_size_warning(summary["observations"]):
                st.warning(warning)

    table20 = []
    for item in raw_stats:
        base = item["baseline"][20]["mean"]
        table20.append({"Ticker": item["ticker"], "Baseline 20j": format_percentage(base),
                        "V1 Alpha": format_alpha(calculate_alpha(item["V1"][20]["mean"], base)),
                        "V2 Alpha": format_alpha(calculate_alpha(item["V2"][20]["mean"], base)),
                        "V3 Early Alpha": format_alpha(calculate_alpha(item["V3 Early"][20]["mean"], base)),
                        "V3 Confirmed Alpha": format_alpha(calculate_alpha(item["V3 Confirmed"][20]["mean"], base)),
                        "Signaux Early": item["signals_V3 Early"],
                        "Signaux Confirmed": item["signals_V3 Confirmed"]})
    st.dataframe(pd.DataFrame(table20), use_container_width=True, hide_index=True)

    with st.expander("Statistiques par action — 5 / 20 / 60 séances"):
        details = []
        for item in raw_stats:
            for engine in engines:
                for horizon in BACKTEST_HORIZONS:
                    stats, baseline = item[engine][horizon], item["baseline"][horizon]["mean"]
                    details.append({"Ticker": item["ticker"], "Moteur": engine,
                                    "Horizon": f"{horizon} séances", "Signaux": stats["observations"],
                                    "Performance moyenne": format_percentage(stats["mean"]),
                                    "Performance médiane": format_percentage(stats["median"]),
                                    "% positifs": f"{stats['positive']:.1%}" if valid_number(stats["positive"]) else "N/D",
                                    "Baseline": format_percentage(baseline),
                                    "Alpha": format_alpha(calculate_alpha(stats["mean"], baseline)),
                                    "Drawdown moyen": format_percentage(stats["drawdown_mean"]),
                                    "Pire drawdown": format_percentage(stats["drawdown_worst"])})
        st.dataframe(pd.DataFrame(details), use_container_width=True, hide_index=True)

    st.markdown("### ⚡ Early ou Confirmed : que montre l'historique ?")
    comparison = []
    for horizon in BACKTEST_HORIZONS:
        early, confirmed = by_engine["V3 Early"][horizon], by_engine["V3 Confirmed"][horizon]
        comparable = [(calculate_alpha(item["V3 Early"][horizon]["mean"], item["baseline"][horizon]["mean"]),
                       calculate_alpha(item["V3 Confirmed"][horizon]["mean"], item["baseline"][horizon]["mean"]))
                      for item in raw_stats]
        comparable = [(a, b) for a, b in comparable if valid_number(a) and valid_number(b)]
        wins = sum(a > b for a, b in comparable)
        comparison.append({"Horizon": f"{horizon} séances",
                           "Early perf. moyenne": format_percentage(early["performance_mean"]),
                           "Confirmed perf. moyenne": format_percentage(confirmed["performance_mean"]),
                           "Early Alpha": format_alpha(early["alpha_mean"]),
                           "Confirmed Alpha": format_alpha(confirmed["alpha_mean"]),
                           "Early alpha médian": format_alpha(early["alpha_median"]),
                           "Confirmed alpha médian": format_alpha(confirmed["alpha_median"]),
                           "Early actions alpha +": f"{early['positive_assets']} / {early['assets']}",
                           "Confirmed actions alpha +": f"{confirmed['positive_assets']} / {confirmed['assets']}",
                           "Early drawdown moy.": format_percentage(early["drawdown_mean"]),
                           "Confirmed drawdown moy.": format_percentage(confirmed["drawdown_mean"]),
                           "Signaux Early / Confirmed": f"{early['observations']} / {confirmed['observations']}",
                           "Early > Confirmed": f"{wins} / {len(comparable)} actions"})
        st.write(f"Early bat Confirmed sur **{wins}/{len(comparable)} actions** à {horizon} séances.")
    st.dataframe(pd.DataFrame(comparison), use_container_width=True, hide_index=True)
    st.info(build_v3_oos_interpretation(by_engine["V3 Early"][20], by_engine["V3 Confirmed"][20]))

    confirmations = aggregate_v3_confirmations(raw_stats)
    st.markdown("### 🔁 Taux de confirmation observé historiquement")
    c1, c2, c3 = st.columns(3)
    c1.metric("Early total", confirmations["early"])
    c2.metric("Confirmés sous 15 séances", confirmations["confirmed"])
    c3.metric("Taux observé", f"{confirmations['rate']:.1%}" if valid_number(confirmations["rate"]) else "N/D")
    st.write(f"Délai moyen Early → Confirmed : **{confirmations['delay_mean']:.1f} séances** · "
             f"Délai médian : **{confirmations['delay_median']:.1f} séances**" if valid_number(confirmations["delay_mean"]) else
             "Aucune paire Early → Confirmed observée sous 15 séances.")
    st.write(f"Performance déjà passée avant confirmation — moyenne : **{format_percentage(confirmations['cost_mean'])}** · "
             f"médiane : **{format_percentage(confirmations['cost_median'])}**")
    confirmation_rows = []
    for item in raw_stats:
        data = item["v3_confirmation"]
        confirmation_rows.append({"Ticker": item["ticker"], "Early": data["early_count"],
                                  "Confirmés sous 15 séances": data["confirmed_count"],
                                  "Taux observé": f"{data['confirmation_rate']:.1%}" if valid_number(data["confirmation_rate"]) else "N/D"})
    st.dataframe(pd.DataFrame(confirmation_rows), use_container_width=True, hide_index=True)

    st.markdown("### Setups Early non confirmés sous 15 séances")
    unconfirmed_row = {"Nombre": confirmations["unconfirmed"]}
    unconfirmed_alpha = aggregate_unconfirmed_v3_alpha(raw_stats)
    for horizon in BACKTEST_HORIZONS:
        pieces = [(item["v3_unconfirmed"][horizon]["mean"], item["v3_unconfirmed"][horizon]["observations"])
                  for item in raw_stats if valid_number(item["v3_unconfirmed"][horizon]["mean"])]
        observations = sum(n for _, n in pieces)
        mean = sum(value * n for value, n in pieces) / observations if observations else None
        positives = [(item["v3_unconfirmed"][horizon]["positive"], item["v3_unconfirmed"][horizon]["observations"])
                     for item in raw_stats if valid_number(item["v3_unconfirmed"][horizon]["positive"])]
        positive_n = sum(n for _, n in positives)
        positive = sum(value * n for value, n in positives) / positive_n if positive_n else None
        unconfirmed_row[f"Performance moyenne {horizon}j"] = format_percentage(mean)
        unconfirmed_row[f"% positifs {horizon}j"] = f"{positive:.1%}" if valid_number(positive) else "N/D"
        if horizon == 20:
            unconfirmed_row["Alpha moyen 20j"] = format_alpha(unconfirmed_alpha["alpha_mean"])
            unconfirmed_row["Alpha médian 20j"] = format_alpha(unconfirmed_alpha["alpha_median"])
            unconfirmed_row["Actions disponibles"] = unconfirmed_alpha["assets"]
    st.dataframe(pd.DataFrame([unconfirmed_row]), use_container_width=True, hide_index=True)

    st.markdown("#### Synthèse complète")
    synthesis = [{"Moteur": row["engine"], "Horizon": f"{row['horizon']} séances",
                  "Perf moyenne": format_percentage(row["performance_mean"]),
                  "Perf médiane": format_percentage(row["performance_median"]),
                  "Baseline": format_percentage(row["baseline_mean"]), "Alpha moyen": format_alpha(row["alpha_mean"]),
                  "Alpha médian": format_alpha(row["alpha_median"]),
                  "Alpha pondéré / observations": format_alpha(row["alpha_weighted"]),
                  "Alpha positif": f"{row['positive_assets']} / {row['assets']}",
                  "Drawdown moy.": format_percentage(row["drawdown_mean"]),
                  "Pire drawdown": format_percentage(row["drawdown_worst"]), "Signaux": row["observations"]}
                 for row in summaries]
    st.dataframe(pd.DataFrame(synthesis), use_container_width=True, hide_index=True)

    st.markdown("#### Stabilité, meilleur et pire alpha par action")
    for engine in engines:
        st.write(f"**{engine}** — {build_horizon_stability(engine, by_engine[engine])}")
        asset_alphas = [(item["ticker"], calculate_alpha(item[engine][20]["mean"], item["baseline"][20]["mean"]))
                        for item in raw_stats]
        asset_alphas = [(ticker, alpha) for ticker, alpha in asset_alphas if valid_number(alpha)]
        if asset_alphas:
            best, worst = max(asset_alphas, key=lambda x: x[1]), min(asset_alphas, key=lambda x: x[1])
            st.caption(f"Meilleur alpha : {best[0]} {format_alpha(best[1])} — Pire alpha : {worst[0]} {format_alpha(worst[1])}.")

    chart = go.Figure()
    for engine in engines:
        chart.add_trace(go.Bar(name=engine, x=[item["ticker"] for item in raw_stats],
                               y=[calculate_alpha(item[engine][20]["mean"], item["baseline"][20]["mean"]) * 100
                                  if valid_number(calculate_alpha(item[engine][20]["mean"], item["baseline"][20]["mean"])) else None
                                  for item in raw_stats]))
    chart.add_hline(y=0, line_color="#94a3b8")
    chart.update_layout(title="Alpha 20 séances par action — V1, V2, V3 Early et V3 Confirmed", barmode="group",
                        yaxis_title="Alpha (points de pourcentage)", height=420,
                        margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(chart, use_container_width=True)

    st.markdown("### 🔬 Conception vs validation")
    conception_summary = aggregate_out_of_sample(conception_stats)
    conception_20 = {(row["engine"], row["horizon"]): row for row in conception_summary}
    for engine in ("V1", "V2"):
        design_alpha = conception_20.get((engine, 20), {}).get("alpha_mean")
        validation_alpha = by_engine[engine][20]["alpha_mean"]
        st.write(f"**{engine} Alpha moyen 20j** — Conception : {format_alpha(design_alpha)} · Hors échantillon : {format_alpha(validation_alpha)}")
        if valid_number(design_alpha) and design_alpha > 0 and (not valid_number(validation_alpha) or validation_alpha < design_alpha * .5):
            st.warning("⚠️ L'avantage observé pendant la conception diminue fortement hors échantillon.")
        elif valid_number(design_alpha) and valid_number(validation_alpha) and abs(validation_alpha - design_alpha) <= max(.005, abs(design_alpha) * .5):
            st.success("✅ L'avantage observé reste relativement stable hors échantillon.")
        else:
            st.info("Les résultats de conception et de validation diffèrent ; aucune stabilité claire n'est établie.")

    export = _validation_export(raw_stats)
    st.download_button("⬇️ Exporter la validation hors échantillon", export.to_csv(index=False).encode("utf-8"),
                       "timing_out_of_sample_validation.csv", "text/csv")
    st.info("Validation descriptive uniquement : aucune règle, aucun seuil et aucun ticker n'ont été optimisés.")

def render_timing_lab() -> None:
    """Interface expérimentale comparant V1 et V2 inchangées à V3."""
    st.title("🧬 Timing Lab")
    st.caption("Comparez le moteur Timing actuel à une logique expérimentale de repli + reprise.")
    st.warning("Le Lab est un outil expérimental. Il sert à évaluer des règles, pas à sélectionner automatiquement la meilleure stratégie.")
    left, middle, right = st.columns(3)
    left.markdown("### V1 Timing actuel\nConvergence et confirmation de tendance — moteur historique inchangé.")
    middle.markdown("### V2 Repli + Reprise\nMoteur expérimental historique inchangé — **ce score n'est pas une probabilité**.")
    right.markdown("### V3 Régime / Setup / Trigger\nDécision en trois étapes ; l'indice n'est qu'une métrique secondaire de backtest.")
    st.subheader("Qui fait historiquement mieux ?")
    periods = {"1 an": "1y", "2 ans": "2y", "3 ans": "3y", "5 ans": "5y"}
    c1, c2, c3, c4 = st.columns(4)
    ticker = c1.text_input("Ticker", "AAPL", key="lab_ticker").strip().upper()
    period_label = c2.selectbox("Période", list(periods), index=2, key="lab_period")
    threshold1 = c3.slider("Seuil V1", 40, 90, 70, key="lab_v1")
    threshold2 = c4.slider("Seuil V2", 40, 90, 70, key="lab_v2")
    if st.button("🧪 Comparer V1, V2 et V3", type="primary") and ticker:
        with st.spinner("Comparaison causale en cours…"):
            histories = load_timing_lab_histories((ticker,), periods[period_label])
            history = histories.get(ticker, pd.DataFrame())
            if len(history) < 220:
                st.error("Historique insuffisant pour comparer les moteurs.")
            else:
                v1, v2 = _lab_eligible(history)
                s1, stats1 = _lab_variant(v1, threshold1, v1)
                s2, stats2 = _lab_variant(v2, threshold2, v2)
                baseline = calculate_backtest_statistics(calculate_forward_returns(v1, v1))
                v3 = calculate_v3_timing_series(history).loc[v1.index].copy()
                v3["_position"] = range(len(v3))
                early, stats_early = _v3_variant(v3, "early")
                confirmed, stats_confirmed = _v3_variant(v3, "confirmed")
                comparison = []
                for label, stats in (("V1", stats1), ("V2", stats2),
                                     ("V3 Early", stats_early), ("V3 Confirmed", stats_confirmed)):
                    comparison.append({"Moteur": label, "20j": format_percentage(stats[20]["mean"]),
                                       "Alpha 20j": format_alpha(calculate_alpha(stats[20]["mean"], baseline[20]["mean"])),
                                       "% positif": f"{stats[20]['positive']:.0%}" if valid_number(stats[20]["positive"]) else "N/D",
                                       "Drawdown": format_percentage(stats[20]["drawdown_mean"])})
                st.dataframe(pd.DataFrame(comparison), use_container_width=True, hide_index=True)
                _render_v3_snapshot(calculate_rigorous_entry_v3(history))
                confirmation = analyze_v3_confirmations(early, confirmed, v3)
                m1, m2, m3 = st.columns(3)
                m1.metric("V3 Early", confirmation["early_count"]); m2.metric("Confirmés sous 15 séances", confirmation["confirmed_count"])
                m3.metric("Taux de confirmation observé", f"{confirmation['confirmation_rate']:.1%}" if valid_number(confirmation["confirmation_rate"]) else "N/D")
                st.caption(f"Délai Early → Confirmed : {confirmation['average_delay']:.1f} séances · performance déjà passée : {format_percentage(confirmation['average_performance_to_confirmation'])}." if valid_number(confirmation["average_delay"]) else "Aucune paire Early → Confirmed observée sous 15 séances.")
                unconfirmed = calculate_backtest_statistics(calculate_signal_drawdowns(calculate_forward_returns(confirmation["unconfirmed"], v3), v3))
                st.caption("Setups non confirmés — " + " · ".join(f"{h}j {format_percentage(unconfirmed[h]['mean'])}" for h in BACKTEST_HORIZONS))
                m1, m2, m3 = st.columns(3)
                m1.metric("Signaux V1", len(s1)); m2.metric("Signaux V2", len(s2))
                common = s1.index.intersection(s2.index)
                m3.metric("Signaux communs", len(common))
                if warning := sample_size_warning(min(len(s1), len(s2))): st.warning(warning)
                common_rows = v1.loc[common].copy()
                common_stats = calculate_backtest_statistics(calculate_forward_returns(common_rows, v1))
                st.caption("Convergence V1 + V2 — groupe exploratoire, qui ne constitue pas encore un troisième moteur. "
                           f"Performance moyenne à 20 séances : {format_percentage(common_stats[20]['mean'])}.")
                diagnostics = pd.DataFrame({
                    "Groupe": ["V1", "V2"],
                    "Distance moyenne MM50": [format_percentage(s1["distance_mm50"].mean()), format_percentage(s2["distance_mm50"].mean())],
                    "RSI moyen": [f"{s1['RSI'].mean():.1f}" if len(s1) else "N/D", f"{s2['RSI'].mean():.1f}" if len(s2) else "N/D"],
                    "Distance moyenne MM200": [format_percentage(s1["distance_mm200"].mean()), format_percentage(s2["distance_mm200"].mean())],
                    "Distance au plus haut 20j": [format_percentage(v1.loc[s1.index, "close"].div(v1["close"].rolling(20).max().loc[s1.index]).sub(1).mean()),
                                                  format_percentage(s2["distance_from_20d_high"].mean())],
                })
                st.markdown("#### Le signal détecte-t-il réellement un repli ?")
                st.dataframe(diagnostics, use_container_width=True, hide_index=True)
                horizons_won = [h for h in BACKTEST_HORIZONS if valid_number(stats1[h]["mean"]) and valid_number(stats2[h]["mean"]) and stats2[h]["mean"] > stats1[h]["mean"]]
                if horizons_won and len(horizons_won) < len(BACKTEST_HORIZONS):
                    st.info(f"V2 surperforme V1 à {', '.join(map(str, horizons_won))} séances, mais pas sur tous les horizons.")
                else:
                    st.info("Aucune conclusion ne doit reposer sur une seule moyenne ou un seul horizon.")
                _render_lab_chart(v1, v2, s1, s2, threshold1, threshold2)

    st.divider(); st.subheader("🌍 Validation multi-actions")
    tickers_text = st.text_area("Tickers (15 maximum, séparés par espaces, virgules ou lignes)",
                                "AAPL MSFT GOOGL NVDA JPM XOM KO SPY", key="lab_multi")
    tokens = [item.strip().upper() for item in tickers_text.replace(",", " ").split() if item.strip()]
    tickers = tuple(dict.fromkeys(tokens[:15]))
    mc1, mc2, mc3 = st.columns(3)
    multi_period_label = mc1.selectbox("Période commune", list(periods), index=2, key="lab_multi_period")
    multi_v1 = mc2.slider("Seuil V1 commun", 40, 90, 70, key="lab_multi_v1")
    multi_v2 = mc3.slider("Seuil V2 commun", 40, 90, 70, key="lab_multi_v2")
    if st.button("🌍 Lancer la validation multi-actions") and tickers:
        histories = load_timing_lab_histories(tickers, periods[multi_period_label])
        rows, raw_stats = [], []
        for symbol in tickers:
            history = histories.get(symbol, pd.DataFrame())
            if len(history) < 220: continue
            v1, v2 = _lab_eligible(history)
            s1, a = _lab_variant(v1, multi_v1, v1); s2, b = _lab_variant(v2, multi_v2, v2)
            base = calculate_backtest_statistics(calculate_forward_returns(v1, v1))
            rows.append({"Ticker": symbol, "Signaux V1": len(s1), "Signaux V2": len(s2),
                         "V1 20j": format_percentage(a[20]["mean"]), "V2 20j": format_percentage(b[20]["mean"]),
                         "Baseline 20j": format_percentage(base[20]["mean"]),
                         "V1 positifs": f"{a[20]['positive']:.0%}" if valid_number(a[20]["positive"]) else "N/D",
                         "V2 positifs": f"{b[20]['positive']:.0%}" if valid_number(b[20]["positive"]) else "N/D",
                         "V1 60j": format_percentage(a[60]["mean"]), "V2 60j": format_percentage(b[60]["mean"])})
            raw_stats.append((symbol, a, b, base, s1, s2))
        if not rows:
            st.error("Aucun historique exploitable.")
        else:
            st.caption(f"Validation sur {len(rows)} actifs — {multi_period_label}, mêmes seuils, horizons et espacement.")
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            aggregates = []
            for horizon in BACKTEST_HORIZONS:
                for label, index in (("V1", 1), ("V2", 2)):
                    values = [(item[index][horizon]["mean"], item[index][horizon]["observations"]) for item in raw_stats
                              if valid_number(item[index][horizon]["mean"]) and item[index][horizon]["observations"]]
                    per_asset = sum(v for v, _ in values) / len(values) if values else None
                    weighted = sum(v * n for v, n in values) / sum(n for _, n in values) if values else None
                    aggregates.append({"Horizon": f"{horizon} séances", "Moteur": label,
                                       "Moyenne par action": format_percentage(per_asset),
                                       "Moyenne pondérée par signaux": format_percentage(weighted),
                                       "Observations": sum(n for _, n in values)})
            st.markdown("#### Deux agrégations complémentaires")
            st.dataframe(pd.DataFrame(aggregates), use_container_width=True, hide_index=True)
            comparable = [x for x in raw_stats if all(valid_number(y[20]["mean"]) for y in (x[1], x[2], x[3]))]
            wins_v1 = sum(x[2][20]["mean"] > x[1][20]["mean"] for x in comparable)
            wins_base = sum(x[2][20]["mean"] > x[3][20]["mean"] for x in comparable)
            total_v2 = sum(x[2][20]["observations"] for x in raw_stats)
            v2_medians = [x[2][20]["median"] for x in raw_stats if valid_number(x[2][20]["median"])]
            v1_medians = [x[1][20]["median"] for x in raw_stats if valid_number(x[1][20]["median"])]
            dd_ok = all(not (valid_number(x[1][20]["drawdown_mean"]) and valid_number(x[2][20]["drawdown_mean"]) and
                            x[2][20]["drawdown_mean"] < x[1][20]["drawdown_mean"] - .03) for x in raw_stats)
            encouraging = (len(comparable) > 0 and wins_v1 > len(comparable)/2 and wins_base > len(comparable)/2 and
                           v2_medians and v1_medians and pd.Series(v2_medians).median() >= pd.Series(v1_medians).median() and
                           dd_ok and total_v2 >= 30)
            robustness = "Encourageante" if encouraging else "Mitigée" if total_v2 >= 10 else "Non concluante"
            st.metric("Robustesse V2", robustness)
            st.write(f"V2 > V1 sur **{wins_v1} / {len(comparable)}** actions après 20 séances.  \n"
                     f"V2 > baseline sur **{wins_base} / {len(comparable)}** actions.")
            if not encouraging: st.warning("La logique Repli + Reprise n'apporte pas d'amélioration claire et robuste sur cet échantillon.")
            if warning := sample_size_warning(total_v2): st.warning(warning)

    st.divider(); st.subheader("🧪 Validation hors échantillon")
    st.caption("Test séparé sur un univers non utilisé pendant la conception. Les règles, seuils, horizons, "
               "espacement des signaux et cours ajustés restent strictement identiques pour tous les actifs.")
    conception_text = st.text_area("Univers de conception", "AAPL\nMSFT\nGOOGL\nNVDA\nJPM\nXOM\nKO\nSPY",
                                   key="lab_oos_conception", help="Ces actifs sont toujours exclus du test hors échantillon.")
    validation_text = st.text_area("Univers de validation hors échantillon (20 maximum)",
                                   "META\nAMZN\nAVGO\nWMT\nJNJ\nPG\nCVX\nBAC\nCAT\nHD\nDIS\nPEP",
                                   key="lab_oos_validation")
    conception = parse_custom_tickers(conception_text, limit=50)
    validation = parse_custom_tickers(validation_text, limit=20)
    effective, overlap = split_validation_universes(conception, validation)
    info1, info2, info3 = st.columns(3)
    info1.metric("Univers de conception", f"{len(conception)} actions")
    info2.metric("Univers de validation", f"{len(effective)} actions")
    info3.metric("Chevauchement", len(overlap))
    if overlap:
        st.warning(f"{len(overlap)} ticker(s) exclus car déjà présents dans l'univers de conception. "
                   f"Actifs exclus : {', '.join(overlap)}.")
    oc1, oc2, oc3 = st.columns(3)
    oos_period_label = oc1.selectbox("Période figée", list(periods), index=2, key="lab_oos_period")
    oos_v1 = oc2.slider("Seuil V1 figé", 40, 90, 70, key="lab_oos_v1")
    oos_v2 = oc3.slider("Seuil V2 figé", 40, 90, 70, key="lab_oos_v2")
    st.caption(f"Espacement minimum inchangé : {MIN_SIGNAL_GAP} séances · Horizons : 5 / 20 / 60 séances · "
               "Aucun ajustement automatique par ticker.")
    signature = (effective, periods[oos_period_label], oos_v1, oos_v2)
    if st.button("🧪 Lancer la validation hors échantillon", type="primary", disabled=not effective):
        with st.spinner("Validation causale groupée en cours…"):
            all_tickers = tuple(dict.fromkeys((*conception, *effective)))
            histories = load_timing_lab_histories(all_tickers, periods[oos_period_label])
            validation_stats, ignored = _calculate_validation_assets(effective, histories, oos_v1, oos_v2)
            conception_stats, _ = _calculate_validation_assets(tuple(conception), histories, oos_v1, oos_v2)
            st.session_state["timing_oos_result"] = {
                "signature": signature, "validation": validation_stats, "conception": conception_stats,
                "ignored": ignored,
            }
    stored = st.session_state.get("timing_oos_result")
    if stored and stored.get("signature") == signature:
        _render_out_of_sample_results(stored)
    elif stored:
        st.info("Les paramètres ont changé. Relancez la validation pour obtenir des résultats comparables.")
    st.warning("Les règles V2 ont été définies avant observation de ce test. Ne modifiez pas immédiatement les seuils après chaque résultat : cela créerait un risque de sur-optimisation.")
    st.info("Une future étape consistera à figer la meilleure variante puis à la tester sur une période / un univers non utilisé lors de sa conception.")


def render_backtest() -> None:
    st.title("🧪 Backtest du moteur Timing")
    st.caption("Évaluez historiquement le comportement des signaux Timing sur une action.")
    st.info("Le backtest mesure le comportement passé du moteur Timing. Les performances historiques ne garantissent pas les performances futures.")
    st.markdown("**Le backtest porte principalement sur les signaux techniques du moteur Timing. Les fondamentaux actuels ne sont pas réappliqués artificiellement au passé.**")
    periods = {"1 an": "1y", "2 ans": "2y", "3 ans": "3y", "5 ans": "5y"}
    c1, c2 = st.columns(2)
    ticker = c1.text_input("Ticker", value="AAPL", key="backtest_ticker").strip().upper()
    period_label = c2.selectbox("Période historique", list(periods), index=2)
    threshold = st.slider("Signal Timing minimum", 40, 90, 70, format="%d/100")
    avoid_close = st.checkbox("Éviter les signaux trop rapprochés", value=True)
    if not st.button("▶️ Lancer le backtest", type="primary"):
        return
    if not ticker:
        st.error("Aucune donnée historique exploitable trouvée pour ce ticker."); return
    with st.spinner("Calcul historique causal en cours…"):
        try:
            history = load_backtest_history(ticker, periods[period_label])
        except Exception:
            history = pd.DataFrame()
        if history.empty:
            st.error("Aucune donnée historique exploitable trouvée pour ce ticker."); return
        if len(history) < 200:
            st.warning("Historique insuffisant pour réaliser un backtest fiable du moteur Timing."); return
        full_timing = calculate_historical_timing_series(history, "Investisseur")
        start = history.attrs.get("analysis_start", full_timing.index.min())
        eligible = full_timing[(full_timing.index >= start) & full_timing["timing_score"].notna()].copy()
        # MM200 impose réellement le warm-up ; aucune fausse valeur n'est injectée.
        eligible = eligible[eligible["MM200"].notna()].copy()
        if eligible.empty:
            st.warning("Historique insuffisant pour réaliser un backtest fiable du moteur Timing."); return
        high = extract_backtest_signals(eligible, threshold, avoid_close)
        low_group = extract_backtest_signals(eligible, 50, avoid_close, low=True)
        high = calculate_signal_drawdowns(calculate_forward_returns(high, full_timing), full_timing)
        low_group = calculate_signal_drawdowns(calculate_forward_returns(low_group, full_timing), full_timing)
        baseline_rows = calculate_forward_returns(eligible, full_timing)
        high_stats, low_stats = calculate_backtest_statistics(high), calculate_backtest_statistics(low_group)
        baseline_stats = calculate_backtest_statistics(baseline_rows)
    metrics = st.columns(3)
    metrics[0].metric("Nombre de signaux", len(high))
    metrics[1].metric("Timing moyen au signal", f"{high['timing_score'].mean():.0f}/100" if not high.empty else "N/D")
    metrics[2].metric("Période analysée", period_label)
    buy_hold = eligible["close"].iloc[-1] / eligible["close"].iloc[0] - 1
    st.caption(f"Performance de l’action sur la période (Buy & Hold, contexte uniquement) : {format_percentage(buy_hold)}")
    warning = sample_size_warning(len(high))
    if warning: st.warning(warning)
    table = []
    for horizon in BACKTEST_HORIZONS:
        item = high_stats[horizon]
        table.append({"Horizon": f"{horizon} séances", "Signaux": item["observations"],
                      "Perf. moyenne": format_percentage(item["mean"]), "Médiane": format_percentage(item["median"]),
                      "Positifs": f"{item['positive']:.0%}" if valid_number(item["positive"]) else "N/D",
                      "Drawdown moyen": format_percentage(item["drawdown_mean"]),
                      "Pire drawdown": format_percentage(item["drawdown_worst"]),
                      "MFE moyen": format_percentage(item["mfe_mean"])})
    st.dataframe(pd.DataFrame(table), hide_index=True, use_container_width=True)
    st.subheader("📊 Le Timing discrimine-t-il réellement ?")
    for horizon in BACKTEST_HORIZONS:
        st.markdown(f"### Après {horizon} séances")
        cols = st.columns(3)
        groups = ((f"Timing ≥ {threshold}", high_stats[horizon]), ("Toutes les séances", baseline_stats[horizon]),
                  ("Timing ≤ 50", low_stats[horizon]))
        for column, (label, item) in zip(cols, groups):
            column.metric(label, format_percentage(item["mean"]),
                          f"{item['positive']:.0%} positifs" if valid_number(item["positive"]) else "N/D")
        means = [item[1]["mean"] for item in groups]
        if all(valid_number(value) for value in means):
            if means[0] > means[1] and means[0] > means[2]:
                st.success("✅ Historiquement, les signaux Timing élevés ont mieux performé sur cet horizon.")
            elif abs(means[0] - means[1]) < .005:
                st.info("🟠 L’avantage historique du Timing est faible sur cet horizon.")
            else:
                st.warning("⚠️ Le Timing élevé n’a pas apporté d’avantage historique sur cet horizon.")
    render_backtest_charts(eligible, high, threshold)
    with st.expander("Voir tous les signaux"):
        detail = high.sort_index(ascending=False).copy()
        shown = pd.DataFrame({"Date": detail.index.strftime("%d/%m/%Y"),
            "Prix au signal": detail["close"].map(lambda x: f"{x:.2f}"),
            "Timing": detail["timing_score"].map(lambda x: f"{x:.0f}/100"),
            "RSI": detail["RSI"].map(lambda x: f"{x:.1f}" if valid_number(x) else "N/D"),
            "Distance MM50": detail["distance_mm50"].map(format_percentage),
            "Distance MM200": detail["distance_mm200"].map(format_percentage),
            "+5j": detail["return_5"].map(format_percentage), "+20j": detail["return_20"].map(format_percentage),
            "+60j": detail["return_60"].map(format_percentage), "Drawdown 20j": detail["drawdown_20"].map(format_percentage)})
        st.dataframe(shown, hide_index=True, use_container_width=True)
    st.subheader("🧭 Lecture du backtest")
    st.write(build_backtest_interpretation(high_stats, baseline_stats, low_stats, threshold))
    with st.expander("⚠️ Limites du backtest"):
        st.markdown("""- Données historiques Yahoo susceptibles d’être ajustées ou incomplètes.
- Fondamentaux point-in-time absents : les fondamentaux actuels ne sont jamais réappliqués au passé.
- Événements historiques non intégrés et exclus du dénominateur.
- Frais, fiscalité et stratégie de capital non simulés.
- Performances basées sur les cours de clôture ajustés.
- Échantillons parfois faibles ; les performances passées ne sont pas prédictives avec certitude.""")
    st.markdown("*Ce backtest est un outil d’évaluation statistique du modèle. Il ne constitue pas une recommandation d’investissement et les performances passées ne garantissent pas les performances futures.*")

def build_entry_decision(global_score: float, timing_result: dict[str, Any],
                         valuation_score: float | None) -> dict[str, str]:
    timing = timing_result.get("score")
    if not valid_number(timing):
        main = "⚪ Lecture combinée non déterminable faute de données Timing"
    elif global_score >= 70 and timing >= 70:
        main = "🟢 Action intéressante — conditions actuelles favorables"
    elif global_score >= 70 and timing >= 55:
        main = "🟠 Action intéressante — attendre une confirmation"
    elif global_score >= 70:
        main = "🟠 Bonne qualité — timing actuellement fragile"
    elif global_score < 55 and timing >= 70:
        main = "🟡 Momentum intéressant, mais qualité globale insuffisante"
    elif global_score < 55 and timing < 55:
        main = "🔴 Profil global et timing peu favorables"
    else:
        main = "🟠 Profil global intermédiaire — confirmations à surveiller"
    secondary = "🔵 Entreprise solide — valorisation exigeante" if global_score >= 70 and valid_number(valuation_score) and valuation_score < 40 and valid_number(timing) and timing >= 55 else ""
    return {"verdict": main, "detail": secondary}


def render_entry_timing(timing: dict[str, Any], global_score: float,
                        global_verdict: str, valuation_score: float | None,
                        currency: str, info: dict[str, Any]) -> None:
    """Affiche séparément qualité, Timing, confirmations et zones techniques."""
    st.subheader("🧭 Analyse du timing d'entrée")
    score = timing.get("score")
    score_text = f"{score:.0f}/100" if valid_number(score) else "N/D"
    decision = build_entry_decision(global_score, timing, valuation_score)
    decision_detail = f'<p class="info">{escape(decision["detail"])}</p>' if decision["detail"] else ""
    st.markdown(
        f'<div class="decision"><div class="trade-row"><span><b>QUALITÉ / SCORE GLOBAL</b><br>'
        f'<span class="score">{global_score:.0f}<small>/100</small></span><br>{escape(global_verdict)}</span>'
        f'<span><b>TIMING ACTUEL</b><br><span class="score">{escape(score_text)}</span><br>'
        f'{escape(timing["verdict"])}<br><span class="muted">Confiance Timing : {timing["confidence"]} %</span></span></div>'
        f'<p class="verdict">{escape(decision["verdict"])}</p>'
        f'{decision_detail}'
        '<p class="disclaimer">Le score Timing mesure la convergence actuelle de plusieurs signaux. '
        'Il ne représente pas une probabilité de hausse.</p></div>', unsafe_allow_html=True,
    )
    bars = ""
    for label, value in timing["categories"].items():
        shown = f"{value:.0f}/100" if valid_number(value) else "N/D"
        width = float(value) if valid_number(value) else 0
        bars += (f'<div class="bar-row"><span>{escape(label)}</span><div class="bar-track">'
                 f'<div class="bar-fill" style="width:{width:.0f}%;background:#60a5fa"></div></div><b>{shown}</b></div>')
    st.markdown(f'<div class="card"><h3>Sous-scores Timing</h3>{bars}</div>', unsafe_allow_html=True)
    good, warn = st.columns(2)
    with good:
        items = "".join(f"<li>{escape(text)}</li>" for text in timing["positive_signals"]) or "<li>Aucun signal disponible</li>"
        st.markdown(f'<div class="card"><h3 class="good">✅ Encourageant aujourd’hui</h3><ul>{items}</ul></div>', unsafe_allow_html=True)
    with warn:
        warnings = list(timing["warning_signals"])
        if valid_number(valuation_score) and valuation_score < 40:
            warnings.append("Valorisation exigeante selon le score Valorisation")
            if valid_number(info.get("trailingPE")):
                warnings.append(f"P/E actuel : {float(info['trailingPE']):.1f}x")
            if valid_number(info.get("priceToBook")):
                warnings.append(f"Price / Book actuel : {float(info['priceToBook']):.1f}x")
        items = "".join(f"<li>{escape(text)}</li>" for text in warnings[:6]) or "<li>Aucun avertissement disponible</li>"
        st.markdown(f'<div class="card"><h3 class="warn">⚠️ À surveiller / raisons d’attendre</h3><ul>{items}</ul></div>', unsafe_allow_html=True)
    if global_score >= 70 and valid_number(timing.get("distance_mm50")) and -.05 <= timing["distance_mm50"] < 0:
        rsi_item = next((x for x in timing["criteria"] if x["key"] == "rsi"), None)
        above_200 = next((x for x in timing["criteria"] if x["key"] == "price_above_mm200"), None)
        if rsi_item and rsi_item["available"] and 35 <= rsi_item["current_value"] <= 50 and above_200 and above_200["passed"]:
            st.info("🟡 Repli technique à surveiller sur une action dont le score global reste favorable. Une confirmation RSI, MACD ou MM50 reste nécessaire.")
    st.markdown("### 🎯 Conditions à surveiller")
    for condition in timing["conditions"]:
        if not condition["available"]:
            st.markdown(f"⚪ **{condition['label']} — N/D**")
            continue
        icon = "✅" if condition["passed"] else "❌"
        details = condition["detail"] or "Condition validée" if condition["passed"] else condition["detail"]
        st.markdown(f"{icon} **{condition['label']}**  \n{escape(str(details))}")
    available = timing["available_conditions"]
    confirmed = timing["confirmed_conditions"]
    percent = round(confirmed / available * 100) if available else 0
    st.progress(percent / 100, text=f"Confirmation actuelle : {confirmed} / {available} conditions validées — {percent} %" if available else "Confirmation actuelle : N/D")
    st.caption("Ce taux binaire de confirmation est distinct du score Timing pondéré.")
    st.markdown("### 📍 Zones techniques à surveiller")
    def zone_text(zone: Any) -> str:
        return f"{format_price(zone[0], currency)} – {format_price(zone[1], currency)}" if isinstance(zone, tuple) else "N/D"
    rows = [("Cours actuel", format_price(timing.get("current"), currency)),
            ("Support récent", zone_text(timing.get("support_zone"))),
            ("MM50", format_price(timing.get("mm50"), currency)),
            ("MM200", format_price(timing.get("mm200"), currency)),
            ("Résistance récente", zone_text(timing.get("resistance_zone"))),
            ("ATR 14", format_price(timing.get("atr"), currency)),
            ("ATR / cours", f"{timing['atr_percent']:.2%}" if valid_number(timing.get("atr_percent")) else "N/D"),
            ("Premier niveau de confirmation technique à surveiller", format_price(timing.get("confirmation_level"), currency))]
    html = "".join(f'<div class="trade-row"><span>{escape(label)}</span><span class="value">{escape(value)}</span></div>' for label, value in rows)
    current, mm50, mm200 = timing.get("current"), timing.get("mm50"), timing.get("mm200")
    reading = ""
    if all(valid_number(x) for x in (current, mm50, mm200)) and current < mm50 and current < mm200:
        reading = "Le cours reste sous ses deux moyennes mobiles principales."
    elif all(valid_number(x) for x in (current, mm50, mm200)) and current > mm50 and current > mm200:
        reading = "Le cours évolue au-dessus de ses moyennes mobiles principales."
    elif valid_number(timing.get("support")) and valid_number(mm50) and timing["support"] < current < mm50:
        reading = "Le cours évolue actuellement entre son support récent et sa MM50."
    reading_html = f'<p class="muted">{escape(reading)}</p>' if reading else ""
    st.markdown(f'<div class="card">{html}{reading_html}</div>', unsafe_allow_html=True)


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


SCREENER_SECTORS = (
    "Tous", "Technology", "Financial Services", "Healthcare", "Industrials", "Energy",
    "Consumer Cyclical", "Consumer Defensive", "Communication Services", "Basic Materials",
    "Real Estate", "Utilities",
)


def parse_custom_tickers(value: str, limit: int = 30) -> list[str]:
    """Normalise une liste libre de symboles, en conservant leur ordre."""
    symbols: list[str] = []
    for candidate in value.replace(",", " ").split():
        symbol = candidate.strip().upper()
        if symbol and symbol not in symbols:
            symbols.append(symbol)
        if len(symbols) >= limit:
            break
    return symbols


WATCHLIST_LIMIT = 25


def initialize_watchlist() -> None:
    """Initialise une liste ordonnée depuis l'URL, une seule fois par session."""
    if "watchlist" not in st.session_state:
        raw_value = st.query_params.get("watchlist", "")
        if isinstance(raw_value, list):
            raw_value = ",".join(str(item) for item in raw_value)
        st.session_state["watchlist"] = parse_custom_tickers(
            str(raw_value), limit=WATCHLIST_LIMIT,
        )


def sync_watchlist_query_params() -> None:
    """Synchronise uniquement le paramètre d'URL appartenant à la watchlist."""
    value = ",".join(st.session_state.get("watchlist", []))
    if value:
        st.query_params["watchlist"] = value
    elif "watchlist" in st.query_params:
        del st.query_params["watchlist"]


def add_to_watchlist(ticker: str) -> None:
    """Ajoute un ticker normalisé sans dépasser la limite absolue."""
    initialize_watchlist()
    parsed = parse_custom_tickers(ticker, limit=1)
    if parsed and parsed[0] not in st.session_state["watchlist"]:
        if len(st.session_state["watchlist"]) >= WATCHLIST_LIMIT:
            return
        st.session_state["watchlist"].append(parsed[0])
        sync_watchlist_query_params()


def remove_from_watchlist(ticker: str) -> None:
    """Retire un ticker de la liste et de l'URL."""
    initialize_watchlist()
    symbol = ticker.strip().upper()
    st.session_state["watchlist"] = [
        item for item in st.session_state["watchlist"] if item != symbol
    ]
    sync_watchlist_query_params()


def clear_watchlist() -> None:
    """Vide la liste après confirmation par l'interface."""
    st.session_state["watchlist"] = []
    sync_watchlist_query_params()


def _symbols_from_screen_response(response: Any) -> list[str]:
    """Extrait les symboles des différentes enveloppes renvoyées par Yahoo."""
    symbols: list[str] = []

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            symbol = value.get("symbol") or value.get("ticker")
            if isinstance(symbol, str) and symbol.strip():
                normalised = symbol.strip().upper()
                if normalised not in symbols:
                    symbols.append(normalised)
            for key in ("quotes", "results", "finance", "data"):
                if key in value:
                    visit(value[key])
        elif isinstance(value, (list, tuple)):
            for item in value:
                visit(item)

    visit(response)
    return symbols


def discover_screener_candidates(region: str, sector: str | None, max_candidates: int) -> list[str]:
    """Découvre au plus 40 grandes/moyennes capitalisations via Yahoo Finance."""
    size = max(1, min(int(max_candidates), 40))
    try:
        conditions = [
            EquityQuery("eq", ["region", region]),
            EquityQuery("gte", ["intradaymarketcap", 500_000_000]),
        ]
        if sector:
            conditions.append(EquityQuery("eq", ["sector", sector]))
        response = yf.screen(
            EquityQuery("and", conditions), size=size,
            sortField="intradaymarketcap", sortAsc=False,
        )
    except Exception:
        return []
    return _symbols_from_screen_response(response)[:size]


@st.cache_data(ttl=1800, show_spinner=False)
def load_screener_history(tickers: tuple[str, ...]) -> pd.DataFrame:
    """Télécharge en une seule requête un an de cours pour les candidats."""
    if not tickers:
        return pd.DataFrame()
    try:
        history = yf.download(
            list(tickers), period="1y", interval="1d", group_by="ticker",
            auto_adjust=False, threads=True, progress=False,
        )
    except Exception:
        return pd.DataFrame()
    return history if isinstance(history, pd.DataFrame) else pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def load_screener_info(ticker: str) -> dict[str, Any]:
    """Charge les fondamentaux d'un candidat présélectionné sans propager d'erreur."""
    try:
        info = yf.Ticker(ticker).get_info()
    except Exception:
        return {}
    return info if isinstance(info, dict) else {}


def extract_ticker_history(history: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Isole un ticker, quelle que soit l'orientation du MultiIndex yfinance."""
    if not isinstance(history, pd.DataFrame) or history.empty:
        return pd.DataFrame()
    extracted = history
    if isinstance(history.columns, pd.MultiIndex):
        extracted = pd.DataFrame()
        for level in range(history.columns.nlevels):
            values = history.columns.get_level_values(level).astype(str)
            matches = [value for value in values.unique() if value.upper() == ticker.upper()]
            if matches:
                try:
                    extracted = history.xs(matches[0], axis=1, level=level, drop_level=True)
                except (KeyError, TypeError, ValueError):
                    extracted = pd.DataFrame()
                break
    if isinstance(extracted.columns, pd.MultiIndex):
        extracted.columns = extracted.columns.get_level_values(-1)
    if "Close" not in extracted.columns:
        return pd.DataFrame()
    return extracted.dropna(subset=["Close"]).copy()


def build_screener_row(
    ticker: str, history: pd.DataFrame, info: dict[str, Any], mode: str,
) -> dict[str, Any] | None:
    """Construit une ligne en appelant strictement le moteur multifactoriel existant."""
    if history.empty or "Close" not in history:
        return None
    data = calculate_indicators(history)
    if data.empty:
        return None
    current = data["Close"].iloc[-1]
    if not valid_number(current):
        return None
    current = float(current)
    previous = data["Close"].iloc[-2] if len(data) > 1 else None
    mm50, mm200 = data["MM50"].iloc[-1], data["MM200"].iloc[-1]
    rsi = data["RSI"].iloc[-1]
    volatility = data["Close"].pct_change().dropna().std() * sqrt(252)
    distance_mm50 = current / float(mm50) - 1 if valid_number(mm50) and float(mm50) else None
    distance_mm200 = current / float(mm200) - 1 if valid_number(mm200) and float(mm200) else None
    scores, _, _ = score_analysis(data, info, mode)
    global_score = sum(scores[key] * WEIGHTS[mode][key] for key in scores) / 100
    confidence_metrics = (
        info.get("trailingPE"), info.get("profitMargins"), info.get("revenueGrowth"),
        info.get("earningsGrowth"), info.get("returnOnEquity"), info.get("freeCashflow"),
        info.get("priceToBook"), info.get("debtToEquity"), info.get("beta"), volatility,
        rsi, mm50, mm200,
    )
    return {
        "ticker": ticker,
        "name": str(info.get("longName") or info.get("shortName") or ticker),
        "currency": str(info.get("currency") or ""),
        "global_score": global_score,
        "confidence": round(sum(valid_number(item) for item in confidence_metrics) / len(confidence_metrics) * 100),
        "price": current,
        "daily_change": current / float(previous) - 1 if valid_number(previous) and float(previous) else None,
        "pe": info.get("trailingPE"), "margin": info.get("profitMargins"),
        "revenue_growth": info.get("revenueGrowth"), "rsi": rsi,
        "volatility": volatility, "mm50": distance_mm50, "mm200": distance_mm200,
        "above_mm200": distance_mm200 is not None and distance_mm200 > 0,
        "scores": scores,
    }


def filter_screener_results(
    rows: list[dict[str, Any]], minimum_score: float, maximum_pe: float,
    ignore_pe: bool, minimum_growth: float, require_mm200: bool, limit: int,
) -> list[dict[str, Any]]:
    """Applique les filtres sans convertir les fondamentaux absents en zéro."""
    selected = []
    for row in rows:
        if not valid_number(row.get("global_score")) or float(row["global_score"]) < minimum_score:
            continue
        pe = row.get("pe")
        if not ignore_pe and (not valid_number(pe) or float(pe) <= 0 or float(pe) > maximum_pe):
            continue
        growth = row.get("revenue_growth")
        if not valid_number(growth) or float(growth) < minimum_growth:
            continue
        if require_mm200 and (not valid_number(row.get("mm200")) or float(row["mm200"]) <= 0):
            continue
        selected.append(row)
    return sorted(selected, key=lambda item: float(item["global_score"]), reverse=True)[:limit]


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


def open_analysis_ticker(ticker: str) -> None:
    """Prépare la navigation vers l'analyse détaillée avant le rerun Streamlit."""
    st.session_state["analysis_ticker"] = ticker
    st.session_state["navigation_mode"] = "📊 Analyse"


def open_screener_ticker(ticker: str) -> None:
    """Conserve le point d'entrée historique du Screener."""
    open_analysis_ticker(ticker)


def _format_screener_table(
    rows: list[dict[str, Any]], *, include_verdict: bool = False,
) -> pd.DataFrame:
    """Présente uniquement des chaînes lisibles et jamais les valeurs Python brutes."""
    table = []
    for row in rows:
        currency = CURRENCY_SYMBOLS.get(row["currency"], f"{row['currency']} " if row["currency"] else "")
        mm200 = row.get("mm200")
        formatted = {
            "Ticker": row["ticker"], "Société": row["name"],
            "Score": f"{row['global_score']:.0f}", "Confiance": f"{row['confidence']} %",
            "Prix": format_price(row.get("price"), currency),
            "Variation jour": format_percentage(row.get("daily_change")),
            "P/E": f"{float(row['pe']):.1f}x" if valid_number(row.get("pe")) else "N/D",
            "Marge nette": format_percentage(row.get("margin")),
            "Croissance CA": format_percentage(row.get("revenue_growth")),
            "RSI": f"{float(row['rsi']):.1f}" if valid_number(row.get("rsi")) else "N/D",
            "Volatilité": format_percentage(row.get("volatility")),
            "MM50": format_percentage(row.get("mm50")),
            "MM200": (f"{'✅' if float(mm200) > 0 else '❌'} {format_percentage(mm200)}"
                      if valid_number(mm200) else "N/D"),
        }
        if include_verdict:
            formatted = {
                **dict(list(formatted.items())[:3]),
                "Verdict": score_verdict(float(row["global_score"])),
                **dict(list(formatted.items())[3:]),
            }
        table.append(formatted)
    return pd.DataFrame(table)


def build_watchlist_rows(watchlist: list[str], mode: str) -> list[dict[str, Any]]:
    """Calcule les lignes via le téléchargement batch et le moteur du Screener."""
    histories = load_screener_history(tuple(watchlist))
    rows: list[dict[str, Any]] = []
    for ticker in watchlist:
        try:
            row = build_screener_row(
                ticker, extract_ticker_history(histories, ticker),
                load_screener_info(ticker), mode,
            )
            if row is not None:
                rows.append(row)
        except Exception:
            continue
    return rows


def render_watchlist(mode: str) -> None:
    """Affiche, importe, exporte et gère les actions favorites."""
    st.title("⭐ Ma Watchlist")
    st.caption("Suivez rapidement vos actions favorites avec le même modèle multifactoriel.")
    watchlist: list[str] = st.session_state["watchlist"]

    add_col, button_col = st.columns([4, 1])
    with add_col:
        manual_ticker = st.text_input(
            "Ticker à ajouter", placeholder="AAPL, TTE.PA, MSFT…", key="watchlist_manual_ticker",
        )
    with button_col:
        st.write("")
        add_clicked = st.button("+ Ajouter", key="watchlist_add_manual", use_container_width=True)
    if add_clicked and manual_ticker.strip():
        parsed = parse_custom_tickers(manual_ticker, limit=1)
        if parsed and parsed[0] in watchlist:
            st.info("Cette action est déjà dans votre watchlist.")
        elif len(watchlist) >= WATCHLIST_LIMIT:
            st.warning("La watchlist est limitée à 25 actions.")
        elif parsed:
            add_to_watchlist(parsed[0])
            st.rerun()

    with st.expander("Importer une liste de tickers"):
        uploaded = st.file_uploader("Fichier CSV", type=["csv"], key="watchlist_import")
        if uploaded is not None and st.button("Importer", key="watchlist_import_button"):
            try:
                imported = pd.read_csv(uploaded)
                ticker_column = next(
                    (column for column in imported.columns if str(column).strip().lower() == "ticker"), None,
                )
                if ticker_column is None:
                    st.error("Le CSV doit contenir une colonne Ticker.")
                else:
                    candidates = parse_custom_tickers(
                        " ".join(imported[ticker_column].dropna().astype(str)), limit=WATCHLIST_LIMIT,
                    )
                    available = WATCHLIST_LIMIT - len(watchlist)
                    new_candidates = [item for item in candidates if item not in watchlist]
                    additions = new_candidates[:available]
                    for ticker in additions:
                        add_to_watchlist(ticker)
                    if len(new_candidates) > available:
                        st.warning("La watchlist est limitée à 25 actions.")
                    if additions:
                        st.rerun()
                    else:
                        st.info("Aucune nouvelle action à importer.")
            except Exception:
                st.error("Impossible de lire ce CSV. Vérifiez son format.")

    if not watchlist:
        st.markdown(
            '<div class="card"><h3>Votre watchlist est vide.</h3>'
            '<p class="muted">Ajoutez une action depuis l’analyse, le Screener ou avec le champ ci-dessus.</p></div>',
            unsafe_allow_html=True,
        )
        st.caption("Les scores et données sont fournis à titre informatif et peuvent être incomplets ou différés. Ils ne constituent pas une recommandation d'investissement.")
        return

    signature = (tuple(watchlist), mode)
    refresh = st.button("🔄 Actualiser la Watchlist", key="watchlist_refresh")
    if refresh or st.session_state.get("watchlist_signature") != signature:
        with st.spinner("Actualisation de la Watchlist…"):
            st.session_state["watchlist_rows"] = build_watchlist_rows(watchlist, mode)
        st.session_state["watchlist_signature"] = signature
    rows = st.session_state.get("watchlist_rows", [])
    sorted_rows = sorted(
        rows, key=lambda row: float(row["global_score"]) if valid_number(row.get("global_score")) else -1,
        reverse=True,
    )
    scores = [float(row["global_score"]) for row in rows if valid_number(row.get("global_score"))]
    metrics = st.columns(3)
    metrics[0].metric("Actions suivies", len(watchlist))
    metrics[1].metric("Score moyen", f"{sum(scores) / len(scores):.0f}/100" if scores else "N/D")
    metrics[2].metric("Meilleur score", f"{max(scores):.0f}/100" if scores else "N/D")
    if sorted_rows:
        table = _format_screener_table(sorted_rows, include_verdict=True)
        st.dataframe(table, hide_index=True, use_container_width=True)
        st.download_button(
            "⬇️ Exporter en CSV", table.to_csv(index=False).encode("utf-8-sig"),
            file_name="watchlist.csv", mime="text/csv", key="watchlist_export",
        )
    else:
        st.info("Aucune donnée valide n'est actuellement disponible pour les actions suivies.")

    st.subheader("📊 Ouvrir une action")
    selected = st.selectbox("Action à analyser", watchlist, key="watchlist_open_ticker")
    st.button(
        "Analyser cette action", key="watchlist_open_button",
        on_click=open_analysis_ticker, args=(selected,),
    )
    st.subheader("Gérer la Watchlist")
    to_remove = st.selectbox("Action à retirer", watchlist, key="watchlist_remove_ticker")
    if st.button("🗑️ Retirer", key="watchlist_remove_button"):
        remove_from_watchlist(to_remove)
        st.rerun()
    confirm_clear = st.checkbox(
        "Je confirme vouloir vider la watchlist", key="watchlist_confirm_clear",
    )
    if confirm_clear and st.button("🗑️ Vider la Watchlist", key="watchlist_clear_button"):
        clear_watchlist()
        st.rerun()
    st.caption("Les scores et données sont fournis à titre informatif et peuvent être incomplets ou différés. Ils ne constituent pas une recommandation d'investissement.")


def render_screener(mode: str) -> None:
    """Affiche les critères, orchestre l'analyse limitée et restitue le classement."""
    st.title("🔎 Screener d'actions")
    st.caption("Explorez plusieurs actions avec le même modèle multifactoriel.")
    market = st.selectbox("Marché", ("🇫🇷 France", "🇺🇸 États-Unis", "✏️ Liste personnalisée"), key="screener_market")
    custom_value = ""
    if market == "✏️ Liste personnalisée":
        custom_value = st.text_area(
            "Tickers (30 maximum)", placeholder="AAPL, MSFT, NVDA, TTE.PA, SHEL",
            key="screener_custom_tickers",
        )
    sector = st.selectbox("Secteur", SCREENER_SECTORS, key="screener_sector")
    candidate_count = st.slider("Nombre d'actions à examiner", 10, 40, 20, key="screener_candidate_count")
    col1, col2 = st.columns(2)
    with col1:
        minimum_score = st.slider("Score minimum", 0, 100, 65, key="screener_minimum_score")
        maximum_pe = st.slider("P/E maximum", 5, 80, 30, key="screener_maximum_pe")
        ignore_pe = st.checkbox("Ne pas filtrer sur le P/E", key="screener_ignore_pe")
    with col2:
        minimum_growth_percent = st.slider("Croissance CA minimum (%)", -20, 50, 0, key="screener_minimum_growth")
        require_mm200 = st.checkbox("Cours au-dessus de la MM200", value=False, key="screener_require_mm200")
        result_limit = st.slider("Nombre maximum de résultats", 5, 25, 15, key="screener_result_limit")

    if st.button("🔎 Rechercher", type="primary", key="screener_search"):
        if market == "✏️ Liste personnalisée":
            candidates = parse_custom_tickers(custom_value)[:candidate_count]
        else:
            region = "fr" if market == "🇫🇷 France" else "us"
            with st.spinner("Découverte des actions disponibles sur Yahoo Finance…"):
                candidates = discover_screener_candidates(
                    region, None if sector == "Tous" else sector, candidate_count,
                )
        if not candidates:
            st.session_state["screener_run"] = {"examined": 0, "rows": []}
            st.warning("Yahoo Finance n'a retourné aucun candidat. Vérifiez la liste ou réessayez plus tard.")
        else:
            histories = load_screener_history(tuple(candidates))
            progress = st.progress(0)
            status = st.empty()
            rows: list[dict[str, Any]] = []
            for index, ticker in enumerate(candidates, start=1):
                status.text(f"Analyse de {index} / {len(candidates)} : {ticker}")
                try:
                    ticker_history = extract_ticker_history(histories, ticker)
                    info = load_screener_info(ticker)
                    row = build_screener_row(ticker, ticker_history, info, mode)
                    if row is not None:
                        rows.append(row)
                except Exception:
                    pass
                progress.progress(index / len(candidates))
            progress.empty()
            status.empty()
            filtered = filter_screener_results(
                rows, minimum_score, maximum_pe, ignore_pe,
                minimum_growth_percent / 100, require_mm200, result_limit,
            )
            st.session_state["screener_run"] = {"examined": len(candidates), "rows": filtered}

    run = st.session_state.get("screener_run")
    if run is None:
        return
    rows = run["rows"]
    metrics = st.columns(3)
    metrics[0].metric("Actions examinées", run["examined"])
    metrics[1].metric("Actions retenues", len(rows))
    metrics[2].metric("Meilleur score", f"{rows[0]['global_score']:.0f}/100" if rows else "N/D")
    st.subheader("Résultats du screener")
    if not rows:
        st.info("Aucune action ne correspond actuellement à ces critères. Essayez d'assouplir les filtres.")
        return
    st.dataframe(_format_screener_table(rows), hide_index=True, use_container_width=True)
    st.caption("Ce classement est un outil de filtrage quantitatif basé sur les données disponibles. Il ne constitue pas une recommandation d'investissement.")
    st.subheader("📊 Ouvrir une action dans l'analyse complète")
    selected = st.selectbox("Action trouvée", [row["ticker"] for row in rows], key="screener_open_ticker")
    st.button(
        "Analyser cette action", key="screener_open_button",
        on_click=open_screener_ticker, args=(selected,),
    )
    if selected in st.session_state["watchlist"]:
        st.info("⭐ Cette action est déjà dans votre watchlist.")
    elif st.button("⭐ Ajouter cette action à ma watchlist", key="screener_add_watchlist"):
        if len(st.session_state["watchlist"]) >= WATCHLIST_LIMIT:
            st.warning("La watchlist est limitée à 25 actions.")
        else:
            add_to_watchlist(selected)
            st.rerun()


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
    company_name = str(info.get("longName") or info.get("shortName") or ticker)
    name = escape(company_name)
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
    # Une seule collecte événementielle par analyse ; le même objet alimente
    # l'interface, le moteur Timing et l'instantané Gemini.
    events = get_key_events(ticker, info)
    timing_history = history
    if len(history) < 220:
        with st.spinner("Chargement de l'historique nécessaire au Timing…"):
            timing_history, _ = load_stock_data(ticker, "1y")
    timing_data = calculate_indicators(timing_history)
    timing = calculate_entry_timing(timing_data, info, events, mode)

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

    if ticker in st.session_state["watchlist"]:
        watch_col, remove_col = st.columns([3, 1])
        watch_col.info("⭐ Dans ma watchlist")
        if remove_col.button("Retirer de la watchlist", key=f"analysis_watchlist_remove_{ticker}"):
            remove_from_watchlist(ticker)
            st.rerun()
    elif st.button("⭐ Ajouter à ma watchlist", key=f"analysis_watchlist_add_{ticker}"):
        if len(st.session_state["watchlist"]) >= WATCHLIST_LIMIT:
            st.warning("La watchlist est limitée à 25 actions.")
        else:
            add_to_watchlist(ticker)
            st.rerun()

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
    render_entry_timing(timing, global_score, verdict, scores.get("Valorisation"), currency, info)

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

    render_key_events(events, currency)
    ai_snapshot = build_ai_snapshot(
        ticker, company_name, mode, global_score, verdict, scores, info, current,
        rsi, macd, signal, volatility, distance_mm50, distance_mm200, position_52w,
        events, timing,
    )
    render_ai_opinion(ai_snapshot, ticker, mode, period)

    st.subheader("Synthèse d'aide à la décision")
    summary = build_summary(
        mode, verdict, info.get("trailingPE"), info.get("profitMargins"),
        info.get("revenueGrowth"), distance_mm50, distance_mm200, position_52w,
    )
    trade_rows = f"""
  <p>{escape(summary)}</p>
  <h3>Plan mécanique indicatif</h3>
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
initialize_watchlist()
if "navigation_mode" not in st.session_state:
    st.session_state["navigation_mode"] = "📊 Analyse"
if "analysis_ticker" not in st.session_state:
    st.session_state["analysis_ticker"] = "AAPL"
with st.sidebar:
    navigation = st.radio(
        "Mode", options=["📊 Analyse", "🥊 Comparateur", "🔎 Screener", "⭐ Watchlist", "🧪 Backtest", "🧬 Timing Lab"],
        key="navigation_mode",
    )
    st.header("Paramètres d'analyse")
    selected_period = st.selectbox("Période", options=list(PERIODS))
    selected_mode = st.radio("Profil", options=list(WEIGHTS))
    if navigation == "📊 Analyse":
        ticker_input = st.text_input(
            "Ticker", placeholder="AAPL, MSFT, MC.PA…", key="analysis_ticker",
        ).strip().upper()
        analyze = st.button("Analyser", type="primary", use_container_width=True)
        st.caption("Les tickers internationaux nécessitent leur suffixe de place (ex. MC.PA).")
    elif navigation == "🥊 Comparateur":
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
elif navigation == "🥊 Comparateur":
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
elif navigation == "🔎 Screener":
    render_screener(selected_mode)
elif navigation == "⭐ Watchlist":
    render_watchlist(selected_mode)
elif navigation == "🧪 Backtest":
    render_backtest()
else:
    render_timing_lab()
