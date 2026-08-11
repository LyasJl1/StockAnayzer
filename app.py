"""Tableau de bord Streamlit d'analyse fondamentale et technique d'actions."""

from __future__ import annotations

from html import escape
from math import isfinite
from datetime import datetime, timezone
from html import escape
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf


PERIODS = {"6 mois": "6m", "1 an": "1y", "2 ans": "2y", "5 ans": "5y"}
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
