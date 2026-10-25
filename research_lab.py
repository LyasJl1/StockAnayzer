"""Diagnostics historiques exploratoires, sans stratégie ni optimisation.

Les fonctions historiques de l'application sont injectées par les trois petits
adaptateurs d'app.py : aucun moteur existant n'est importé ou modifié ici.
"""
from __future__ import annotations

from time import perf_counter
from typing import Callable, Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

RESEARCH_HORIZONS = (5, 20, 60)
RESEARCH_DEFAULT_TICKERS = ("AAPL", "MSFT", "GOOGL", "NVDA", "JPM", "XOM", "KO", "SPY")
RESEARCH_BUCKET_EDGES = {
    "distance_mm50": (-float("inf"), -.08, -.05, -.02, .02, .05, .10, float("inf")),
    "distance_mm200": (-float("inf"), -.15, -.05, 0, .10, .25, float("inf")),
    "RSI": (-float("inf"), 30, 35, 40, 45, 50, 60, 70, float("inf")),
    "atr_percent": (-float("inf"), .015, .025, .04, float("inf")),
    "volatility": (-float("inf"), .15, .25, .35, .50, .70, float("inf")),
    "position_52w": (-float("inf"), .20, .40, .60, .80, float("inf")),
    "drawdown_60": (-float("inf"), -.20, -.10, -.05, float("inf")),
    "momentum_20": (-float("inf"), -.20, -.10, -.05, 0, .05, .10, .20, float("inf")),
    "momentum_60": (-float("inf"), -.20, -.10, -.05, 0, .05, .10, .20, float("inf")),
    "momentum_120": (-float("inf"), -.20, -.10, -.05, 0, .05, .10, .20, float("inf")),
    "relative_strength_60": (-float("inf"), -.10, -.05, 0, .05, .10, float("inf")),
}
RESEARCH_BUCKET_LABELS = {
    "distance_mm50": ("< −8 %", "−8 à −5 %", "−5 à −2 %", "−2 à +2 %", "+2 à +5 %", "+5 à +10 %", "≥ +10 %"),
    "distance_mm200": ("< −15 %", "−15 à −5 %", "−5 à 0 %", "0 à +10 %", "+10 à +25 %", "≥ +25 %"),
    "RSI": ("< 30", "30–35", "35–40", "40–45", "45–50", "50–60", "60–70", "≥ 70"),
    "atr_percent": ("< 1,5 %", "1,5–2,5 %", "2,5–4 %", "≥ 4 %"),
    "volatility": ("< 15 %", "15–25 %", "25–35 %", "35–50 %", "50–70 %", "≥ 70 %"),
    "position_52w": ("0–20 %", "20–40 %", "40–60 %", "60–80 %", "80–100 %"),
    "drawdown_60": ("< −20 %", "−20 à −10 %", "−10 à −5 %", "−5 à 0 %"),
    "relative_strength_60": ("< −10 pt", "−10 à −5 pt", "−5 à 0 pt", "0 à +5 pt", "+5 à +10 pt", "≥ +10 pt"),
}
for _momentum in ("momentum_20", "momentum_60", "momentum_120"):
    RESEARCH_BUCKET_LABELS[_momentum] = ("< −20 %", "−20 à −10 %", "−10 à −5 %", "−5 à 0 %", "0 à +5 %", "+5 à +10 %", "+10 à +20 %", "≥ +20 %")
RESEARCH_FACTOR_NAMES = {
    "distance_mm50": "Distance MM50", "distance_mm200": "Distance MM200", "RSI": "RSI",
    "atr_percent": "ATR / cours", "volatility": "Volatilité annualisée 20j",
    "position_52w": "Position 52 semaines", "drawdown_60": "Drawdown récent 60j",
    "momentum_20": "Momentum 20j", "momentum_60": "Momentum 60j", "momentum_120": "Momentum 120j",
    "relative_strength_60": "Force relative SPY 60j",
}
RESEARCH_CORRELATION_COLUMNS = ("RSI", "distance_mm50", "distance_mm200", "momentum_20", "momentum_60",
                                "momentum_120", "relative_strength_60", "volatility", "atr_percent", "drawdown_60")
RESEARCH_EXPORT_COLUMNS = (
    "Factor", "Bucket", "Horizon", "Regime", "Observations", "Tickers", "Mean Return", "Median Return",
    "Positive Rate", "Baseline Global", "Baseline Regime", "Alpha Global", "Alpha Regime",
    "Mean Drawdown", "Worst Drawdown", "Positive Tickers", "Valid Tickers", "Positive Windows",
    "Valid Windows", "Robustness", "Mean Return per Ticker", "Alpha Regime per Ticker", "Median Alpha",
    "Median Ticker Alpha", "Best Ticker", "Worst Ticker", "Sample Warning",
)


def clean_research_history(history: pd.DataFrame) -> pd.DataFrame:
    """Dates locales de séance, triées, uniques ; aucune imputation de prix."""
    frame = history.copy()
    if frame.empty or not {"Close", "High", "Low"}.issubset(frame.columns):
        return pd.DataFrame()
    frame.index = pd.DatetimeIndex(frame.index).tz_localize(None).normalize()
    frame = frame.loc[~frame.index.duplicated(keep="last")].sort_index()
    close = pd.to_numeric(frame["Close"], errors="coerce")
    return frame.loc[close.notna() & close.gt(0)].copy()


def prepare_research_features(history: pd.DataFrame, historical_preparer: Callable) -> pd.DataFrame:
    """Une seule préparation causale par ticker ; aucune cible dans les features."""
    history = clean_research_history(history)
    if history.empty:
        return history
    frame = historical_preparer(history)
    close = frame["Close"]
    frame["atr_percent"] = frame["ATR14"].div(close)
    frame["volatility"] = close.pct_change(fill_method=None).rolling(20, min_periods=20).std() * 252**.5
    high, low = frame["High"].rolling(252).max(), frame["Low"].rolling(252).min()
    frame["position_52w"] = close.sub(low).div(high.sub(low).where(high.gt(low)))
    frame["drawdown_60"] = close.div(close.rolling(60).max()).sub(1)
    for horizon in (20, 60, 120):
        frame[f"momentum_{horizon}"] = close.div(close.shift(horizon)).sub(1)
    frame["macd_gap"] = frame["MACD"] - frame["Signal"]
    frame["macd_gap_change_5"] = frame["macd_gap"] - frame["macd_gap"].shift(5)
    frame["history_count"] = range(1, len(frame) + 1)
    return frame.replace([float("inf"), -float("inf")], float("nan"))


def attach_research_market_context(features: pd.DataFrame, spy: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Alignement sur la même séance, jamais de backfill ni régime fictif."""
    frame = features.copy()
    market = spy.reindex(frame.index)
    # La pente est calculée dans les séances SPY avant l'alignement (jours fériés).
    slope = spy["MM200"].gt(spy["MM200"].shift(20)).reindex(frame.index)
    known = market["Close"].notna() & market["MM200"].notna() & spy["MM200"].shift(20).reindex(frame.index).notna()
    frame["spy_above_mm200"] = market["Close"].gt(market["MM200"]).astype("boolean").where(known)
    frame["spy_mm200_rising"] = slope.astype("boolean").where(known)
    bull = frame["spy_above_mm200"] & frame["spy_mm200_rising"]
    frame["Regime"] = pd.Series(pd.NA, index=frame.index, dtype="string")
    frame.loc[bull.fillna(False), "Regime"] = "Bull"
    frame.loc[(~bull).fillna(False), "Regime"] = "Non bull"
    frame["relative_strength_60"] = (frame["momentum_60"] - market["momentum_60"]
                                      if ticker.upper() != "SPY" else float("nan"))
    return frame


def research_bucket(values: pd.Series, feature: str) -> pd.Series:
    """Intervalles [borne basse, borne haute), y compris toutes les frontières."""
    values = values.where(values.abs().lt(float("inf")))
    if feature == "position_52w":
        values = values.where(values.between(0, 1))
    if feature == "drawdown_60":
        values = values.where(values.le(0))
    return pd.cut(values, RESEARCH_BUCKET_EDGES[feature], labels=RESEARCH_BUCKET_LABELS[feature],
                  right=False).astype("string")


def research_combinations(features: pd.DataFrame) -> dict[str, pd.Series]:
    """Cinq hypothèses écrites explicitement, aucune génération de règles."""
    bull = features["Regime"].eq("Bull").fillna(False)
    near = features["distance_mm50"].between(-.05, .02)
    strength = features["relative_strength_60"].gt(0)
    return {
        "Combo A": bull & near,
        "Combo B": bull & features["RSI"].between(35, 45),
        "Combo C": bull & strength,
        "Combo D": bull & near & strength,
        "Combo E": bull & features["drawdown_60"].between(-.15, -.05) & strength,
    }


def assign_research_buckets(features: pd.DataFrame) -> pd.DataFrame:
    result = pd.DataFrame(index=features.index)
    for feature, label in RESEARCH_FACTOR_NAMES.items():
        result[f"bucket::{label}"] = research_bucket(features[feature], feature)
    conditions = {
        "MM200 — cours": (features["Close"] - features["MM200"], "Au-dessus", "Au-dessous", False),
        "MM200 — pente 20j": (features["mm200_slope_20"], "Montante", "Descendante", False),
        "MACD — position": (features["macd_gap"], "MACD > Signal", "MACD < Signal", False),
        "MACD — évolution 5j": (features["macd_gap_change_5"], "Gap en hausse", "Gap stable / en baisse", True),
    }
    for name, (values, positive, negative, include_zero) in conditions.items():
        labels = pd.Series(pd.NA, index=features.index, dtype="string")
        labels.loc[values.gt(0)] = positive
        labels.loc[values.le(0) if include_zero else values.lt(0)] = negative
        result[f"bucket::{name}"] = labels
    for name, mask in research_combinations(features).items():
        result[f"bucket::{name}"] = pd.Series("Condition réunie", index=features.index, dtype="string").where(mask)
    return result


def prepare_forward_targets(features: pd.DataFrame, forward_returns: Callable, signal_drawdowns: Callable) -> pd.DataFrame:
    """Réutilise les conventions Close[T+h]/Close[T] et min Close[T:T+h]."""
    timeline = pd.DataFrame({"close": features["Close"]})
    observations = pd.DataFrame({"_position": range(len(timeline))}, index=timeline.index)
    targets = signal_drawdowns(forward_returns(observations, timeline), timeline)
    dates = pd.Series(timeline.index, index=timeline.index)
    for horizon in RESEARCH_HORIZONS:
        targets[f"target_date_{horizon}"] = dates.shift(-horizon)
    return targets.drop(columns="_position").rename(
        columns={f"drawdown_{h}": f"forward_drawdown_{h}" for h in RESEARCH_HORIZONS}
    ).replace([float("inf"), -float("inf")], float("nan"))


def build_research_panel(histories: dict[str, pd.DataFrame], start: Any, end: Any,
                         feature_preparer: Callable, target_preparer: Callable) -> tuple[pd.DataFrame, list[str]]:
    """Features et targets une fois par ticker, avant toute sélection de bucket."""
    start, end = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    prepared, warnings = {}, []
    for ticker, history in histories.items():
        clean = clean_research_history(history)
        if not clean.empty:
            # Borne d'étude AVANT le calcul des cibles : pas de sortie après end.
            prepared[ticker] = feature_preparer(clean.loc[clean.index <= end])
    if "SPY" not in prepared or prepared["SPY"].empty:
        raise ValueError("Historique SPY indisponible : régime et baseline de régime non déterminables.")
    frames = []
    for ticker, features in prepared.items():
        if features.empty:
            continue
        contextual = attach_research_market_context(features, prepared["SPY"], ticker)
        targets = target_preparer(features)
        buckets = assign_research_buckets(contextual)
        frame = pd.concat([contextual, targets, buckets], axis=1)
        frame["Date"], frame["Ticker"] = frame.index, ticker
        frame = frame.loc[(frame.index >= start) & (frame["history_count"] >= 252)]
        unknown = int(frame["Regime"].isna().sum())
        if unknown:
            warnings.append(f"{ticker} : {unknown} séances sans contexte SPY exclues des diagnostics.")
        if frame.empty:
            warnings.append(f"{ticker} : aucune séance admissible après le warm-up de 252 séances.")
        frames.append(frame)
    if not frames:
        raise ValueError("Aucun historique exploitable après préparation.")
    return pd.concat(frames, ignore_index=True), warnings


def research_sample_warning(observations: int) -> str:
    return "Échantillon très faible" if observations < 30 else "Échantillon limité" if observations <= 100 else ""


def research_robustness(row: dict[str, Any] | pd.Series) -> str:
    """Classification descriptive, aucune significativité statistique revendiquée."""
    count = row["Observations"]
    tickers, windows = row["Valid Tickers"], row["Valid Windows"]
    ticker_ratio = row["Positive Tickers"] / tickers if tickers else None
    window_ratio = row["Positive Windows"] / windows if windows else None
    if (row["Alpha Regime per Ticker"] <= 0 or
        (ticker_ratio is not None and ticker_ratio < .5) or
        (window_ratio is not None and window_ratio < .5)):
        return "Faible"
    if (count >= 100 and row["Alpha Regime"] > 0 and row["Median Alpha"] > 0 and
        tickers >= 2 and windows >= 2 and ticker_ratio > .5 and window_ratio > .5):
        return "Encourageant"
    return "Mitigé"


def _research_memberships(panel: pd.DataFrame, horizon: int, regime: str, ticker: str | None) -> pd.DataFrame:
    """Baselines par ticker sur le même échantillon temporel et horizon complet."""
    if horizon not in RESEARCH_HORIZONS:
        raise ValueError("Horizon Research non pris en charge")
    frame = panel.loc[panel[f"return_{horizon}"].notna() & panel[f"forward_drawdown_{horizon}"].notna()
                      & panel["Regime"].notna()].copy()
    if ticker:
        frame = frame.loc[frame["Ticker"].eq(ticker)].copy()
    frame["Return"] = frame[f"return_{horizon}"]
    frame["Drawdown"] = frame[f"forward_drawdown_{horizon}"]
    # Calculées avant de filtrer le régime demandé ; globale reste secondaire.
    frame["Baseline Global"] = frame.groupby("Ticker")["Return"].transform("mean")
    frame["Baseline Regime"] = frame.groupby(["Ticker", "Regime"])["Return"].transform("mean")
    if regime != "Tous":
        frame = frame.loc[frame["Regime"].eq(regime)]
    bucket_columns = [name for name in frame if name.startswith("bucket::")]
    ids = ["Ticker", "Date", "Return", "Drawdown", "Baseline Global", "Baseline Regime"]
    long = frame.melt(id_vars=ids, value_vars=bucket_columns, var_name="Factor", value_name="Bucket").dropna(subset=["Bucket"])
    long["Factor"] = long["Factor"].str.removeprefix("bucket::")
    long["Alpha Global"] = long["Return"] - long["Baseline Global"]
    long["Alpha Regime"] = long["Return"] - long["Baseline Regime"]
    long["Positive"] = long["Return"].gt(0).astype(float)
    return long


def _aggregate_research_memberships(long: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = ["Factor", "Bucket"]
    if long.empty:
        return pd.DataFrame(), pd.DataFrame()
    groups = long.groupby(keys, observed=True, sort=False)
    summary = groups.agg(**{
        "Observations": ("Return", "size"), "Mean Return": ("Return", "mean"),
        "Median Return": ("Return", "median"), "Positive Rate": ("Positive", "mean"),
        "Mean Drawdown": ("Drawdown", "mean"), "Worst Drawdown": ("Drawdown", "min"),
        "Baseline Global": ("Baseline Global", "mean"), "Baseline Regime": ("Baseline Regime", "mean"),
        "Alpha Global": ("Alpha Global", "mean"), "Alpha Regime": ("Alpha Regime", "mean"),
        "Median Alpha": ("Alpha Regime", "median"), "Tickers": ("Ticker", "nunique"),
    })
    per_ticker = long.groupby(keys + ["Ticker"], observed=True, sort=False).agg(**{
        "Observations": ("Return", "size"), "Mean Return": ("Return", "mean"),
        "Alpha Regime": ("Alpha Regime", "mean"), "Alpha Global": ("Alpha Global", "mean"),
    }).reset_index()
    per_ticker["Positive"] = per_ticker["Alpha Regime"].gt(0).astype(int)
    ticker_groups = per_ticker.groupby(keys, observed=True, sort=False)
    balanced = ticker_groups.agg(**{
        "Mean Return per Ticker": ("Mean Return", "mean"), "Alpha Regime per Ticker": ("Alpha Regime", "mean"),
        "Median Ticker Alpha": ("Alpha Regime", "median"), "Positive Tickers": ("Positive", "sum"),
        "Valid Tickers": ("Ticker", "size"),
    })
    best = per_ticker.loc[ticker_groups["Alpha Regime"].idxmax()].set_index(keys)["Ticker"].rename("Best Ticker")
    worst = per_ticker.loc[ticker_groups["Alpha Regime"].idxmin()].set_index(keys)["Ticker"].rename("Worst Ticker")
    return summary.join(balanced).join(best).join(worst).reset_index(), per_ticker


@st.cache_data(show_spinner=False, max_entries=24)
def analyze_research_panel(panel: pd.DataFrame, horizon: int, regime: str = "Tous", ticker: str | None = None,
                           windows: tuple = ()) -> dict[str, pd.DataFrame]:
    summary, per_ticker = _aggregate_research_memberships(_research_memberships(panel, horizon, regime, ticker))
    if summary.empty:
        return {"results": pd.DataFrame(columns=RESEARCH_EXPORT_COLUMNS), "tickers": per_ticker, "windows": pd.DataFrame()}
    window_results = []
    for label, start, end in windows:
        # Ni entrées ni sorties hors de la fenêtre de validation.
        bounded = panel.loc[panel["Date"].between(pd.Timestamp(start), pd.Timestamp(end)) &
                            panel[f"target_date_{horizon}"].le(pd.Timestamp(end))]
        result, _ = _aggregate_research_memberships(_research_memberships(bounded, horizon, regime, ticker))
        if not result.empty:
            result["Window"] = label
            window_results.append(result)
    window_details = pd.concat(window_results, ignore_index=True) if window_results else pd.DataFrame()
    keys = ["Factor", "Bucket"]
    if not window_details.empty:
        window_details["Positive"] = window_details["Alpha Regime per Ticker"].gt(0).astype(int)
        counts = window_details.groupby(keys, observed=True).agg(**{
            "Positive Windows": ("Positive", "sum"), "Valid Windows": ("Window", "size")})
        summary = summary.join(counts, on=keys)
    else:
        summary["Positive Windows"], summary["Valid Windows"] = 0, 0
    for name in ("Positive Windows", "Valid Windows"):
        summary[name] = summary[name].fillna(0).astype(int)
    summary["Horizon"], summary["Regime"] = horizon, regime
    summary["Sample Warning"] = summary["Observations"].map(research_sample_warning)
    summary["Robustness"] = summary.apply(research_robustness, axis=1)
    return {"results": summary.sort_values("Alpha Regime", ascending=False).reset_index(drop=True),
            "tickers": per_ticker, "windows": window_details}


def research_correlations(panel: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Moyenne des Pearson intra-ticker (au moins 30 paires), poids égaux."""
    cols = list(RESEARCH_CORRELATION_COLUMNS)
    matrices = [frame[cols].corr(min_periods=30) for _, frame in panel.groupby("Ticker")]
    if not matrices:
        return pd.DataFrame(), []
    matrix = pd.concat(matrices).groupby(level=0, sort=False).mean().reindex(index=cols, columns=cols)
    pairs = []
    for index, left in enumerate(cols):
        for right in cols[index+1:]:
            value = matrix.loc[left, right]
            if pd.notna(value) and abs(value) >= .8:
                pairs.append(f"{left} et {right} semblent partiellement redondants (r = {value:.2f}).")
    return matrix, pairs


@st.cache_data(ttl=900, show_spinner=False, max_entries=8)
def load_research_histories(tickers: tuple[str, ...], start: str, end: str) -> tuple[dict[str, pd.DataFrame], list[str]]:
    symbols = tuple(dict.fromkeys((*tickers, "SPY")))
    warmup = pd.Timestamp(start) - pd.Timedelta(days=550)
    downloaded = yf.download(list(symbols), start=warmup.date(), end=(pd.Timestamp(end) + pd.Timedelta(days=1)).date(),
                             auto_adjust=True, progress=False, threads=True, timeout=20, group_by="ticker")
    histories, warnings = {}, []
    for ticker in symbols:
        try:
            frame = downloaded[ticker] if isinstance(downloaded.columns, pd.MultiIndex) else downloaded
            clean = clean_research_history(frame)
            if clean.empty:
                warnings.append(f"{ticker} : historique indisponible.")
            else:
                histories[ticker] = clean
        except (KeyError, TypeError):
            warnings.append(f"{ticker} : historique indisponible.")
    return histories, warnings


def research_summary_table(results: pd.DataFrame) -> pd.DataFrame:
    table = pd.DataFrame({
        "Facteur": results["Factor"], "Bucket": results["Bucket"], "Obs": results["Observations"],
        "Tickers couverts": results["Tickers"], "Fenêtres couvertes": results["Valid Windows"],
        "Alpha régime": results["Alpha Regime"] * 100, "Alpha médian": results["Median Alpha"] * 100,
        "Alpha / ticker": results["Alpha Regime per Ticker"] * 100,
        "Tickers +": results["Positive Tickers"].astype(str) + " / " + results["Valid Tickers"].astype(str),
        "Fenêtres +": results["Positive Windows"].astype(str) + " / " + results["Valid Windows"].astype(str),
        "Drawdown moyen": results["Mean Drawdown"] * 100,
        "Robustesse": results["Robustness"], "Échantillon": results["Sample Warning"],
    })
    return table


def render_research_lab(feature_preparer: Callable, target_preparer: Callable, window_builder: Callable) -> None:
    st.title("🔬 Research Lab")
    st.caption("Outil expérimental d'analyse historique de facteurs prédéfinis. Aucun seuil optimisé, aucune stratégie nouvelle.")
    st.info("L'alpha est une différence de rendement historique par rapport à une baseline. Les rendements se chevauchent : "
            "aucune p-value ni garantie de performance. Aucun résultat n'alimente l'Analyse, le Timing ou le Scoring.")
    with st.expander("Configuration", expanded=True):
        tickers_text = st.text_area("Univers (20 tickers maximum)", "\n".join(RESEARCH_DEFAULT_TICKERS), key="research_tickers")
        years = st.selectbox("Période", (3, 5, 8, 10), index=3, format_func=lambda value: f"{value} ans", key="research_years")
        tickers = tuple(dict.fromkeys(tickers_text.upper().replace(",", " ").replace(";", " ").split()))
        st.caption("SPY est toujours chargé comme contexte de marché. Warm-up : 550 jours calendaires supplémentaires, "
                   "puis au moins 252 séances par action. Cours ajustés Yahoo Finance.")
        launch = st.button("🔬 Lancer la recherche", type="primary", key="research_run")
    if launch:
        if not tickers or len(tickers) > 20:
            st.error("Indiquez de 1 à 20 tickers distincts.")
            return
        start_time = perf_counter()
        end = pd.Timestamp(date_today()).normalize()
        start = end - pd.DateOffset(years=years)
        try:
            with st.spinner("Préparation unique des facteurs et des cibles…"):
                histories, warnings = load_research_histories(tickers, str(start.date()), str(end.date()))
                panel, preparation_warnings = build_research_panel(histories, start, end, feature_preparer, target_preparer)
                # SPY chargé pour le contexte ne devient pas une action de l'univers personnalisé.
                panel = panel.loc[panel["Ticker"].isin(tickers)].copy()
                if panel.empty:
                    raise ValueError("Aucune séance admissible pour l'univers demandé.")
                windows = tuple((window["window"], window["validation_start"], window["validation_end"])
                                for window in window_builder(start, end, 3, 1, 1))
                st.session_state["research_result"] = {
                    "panel": panel, "windows": windows, "warnings": warnings + preparation_warnings,
                    "seconds": perf_counter() - start_time, "signature": (tickers, years),
                }
        except Exception as error:
            st.session_state.pop("research_result", None)
            st.error(f"Recherche indisponible : {error}")
            return
    run = st.session_state.get("research_result")
    if run is None:
        return
    if run["signature"] != (tickers, years):
        st.warning("Configuration modifiée : relancez la recherche pour appliquer cet univers ou cette période.")
        return
    for warning in run["warnings"]:
        st.warning(warning)
    panel = run["panel"]
    filters = st.columns(4)
    horizon = filters[0].selectbox("Horizon (séances)", RESEARCH_HORIZONS, index=1, key="research_horizon")
    regime = filters[1].selectbox("Régime SPY", ("Tous", "Bull", "Non bull"), key="research_regime")
    selected_ticker = filters[2].selectbox("Ticker", ("Tous", *sorted(panel["Ticker"].unique())), key="research_ticker")
    minimum = filters[3].number_input("Minimum d'observations", min_value=0, value=30, step=10, key="research_minimum")
    timing = perf_counter()
    analysis = analyze_research_panel(panel, horizon, regime, None if selected_ticker == "Tous" else selected_ticker, run["windows"])
    results = analysis["results"]
    counts = st.columns(3)
    counts[0].metric("Observations admissibles", int((panel[f"return_{horizon}"].notna() & panel["Regime"].notna()
                                                  & (panel["Regime"].eq(regime) if regime != "Tous" else True)
                                                  & (panel["Ticker"].eq(selected_ticker) if selected_ticker != "Tous" else True)).sum()))
    counts[1].metric("Tickers couverts", int(results["Tickers"].max()) if not results.empty else 0)
    counts[2].metric("Fenêtres couvertes", int(results["Valid Windows"].max()) if not results.empty else 0)
    st.caption(f"Préparation : {run['seconds']:.2f} s ; diagnostic : {perf_counter()-timing:.2f} s. "
               "Observation 3 ans, validation 1 an, pas 1 an ; aucun apprentissage. "
               "Les sorties après chaque fin de validation sont exclues.")
    if not run["windows"]:
        st.warning("Aucune fenêtre complète 3 ans + 1 an : stabilité temporelle non déterminable sur cette période.")
    factor_filter = st.selectbox("Filtrer les facteurs", ("Tous", *sorted(results.loc[~results["Factor"].str.startswith("Combo"), "Factor"].unique())),
                                 key="research_factor_filter")
    visible = results.loc[results["Observations"].ge(minimum)]
    if factor_filter != "Tous":
        visible = visible.loc[visible["Factor"].eq(factor_filter)]
    individual = visible.loc[~visible["Factor"].str.startswith("Combo")]
    st.subheader("Facteurs les plus encourageants historiquement")
    st.caption("Tri : alpha moyen vs baseline de régime, par observation. Rendements et drawdowns en %, alphas en points. "
               "Chaque ligne indique son échantillon ; 'Encourageant' exige au moins 2 tickers et 2 fenêtres valides.")
    st.dataframe(research_summary_table(individual), hide_index=True, use_container_width=True)
    if visible.empty:
        st.info("Aucun bucket ne respecte le minimum d'observations.")
    st.download_button("⬇️ Exporter les diagnostics filtrés", visible.reindex(columns=RESEARCH_EXPORT_COLUMNS).to_csv(index=False).encode("utf-8-sig"),
                       file_name="research_lab_results.csv", mime="text/csv", key="research_export")
    if not individual.empty:
        st.subheader("Explorer un facteur")
        factor = st.selectbox("Facteur", sorted(individual["Factor"].unique()), key="research_factor")
        factor_rows = individual.loc[individual["Factor"].eq(factor)]
        feature = next((key for key, label in RESEARCH_FACTOR_NAMES.items() if label == factor), None)
        if feature:
            order = {label: index for index, label in enumerate(RESEARCH_BUCKET_LABELS[feature])}
            factor_rows = factor_rows.sort_values("Bucket", key=lambda column: column.map(order))
        figure = go.Figure(go.Bar(x=factor_rows["Bucket"], y=factor_rows["Alpha Regime"] * 100))
        figure.add_hline(y=0, line_color="#94a3b8")
        figure.update_layout(xaxis_title="Bucket fixe", yaxis_title=f"Alpha moyen {horizon}j vs régime (points)", showlegend=False)
        st.plotly_chart(figure, use_container_width=True)
        with st.expander("Statistiques détaillées et baselines"):
            st.dataframe(factor_rows.reindex(columns=RESEARCH_EXPORT_COLUMNS), hide_index=True)
            st.caption("Export et détail numérique : rendements en ratios (0,01 = 1 %). Baseline globale : toutes les séances "
                       "admissibles du même ticker ; baseline principale : même ticker et même régime. "
                       "En vue Tous, les baselines sont pondérées selon les régimes effectivement observés dans le bucket.")
        with st.expander("Stabilité entre actions et fenêtres"):
            for label, frame in (("Par ticker", analysis["tickers"]), ("Par fenêtre de validation", analysis["windows"])):
                st.markdown(f"**{label}**")
                if frame.empty:
                    st.info("Aucune fenêtre complète disponible.")
                else:
                    st.dataframe(frame.loc[frame["Factor"].eq(factor)], hide_index=True)
            st.caption("Les moyennes par ticker donnent le même poids à chaque action. Les alphas des fenêtres utilisent "
                       "leurs propres baselines, avec les mêmes contraintes de sortie. Pas d'optimisation entre fenêtres.")
        with st.expander("Effet du régime : Bull vs Non bull"):
            if st.checkbox("Calculer le détail par régime", key="research_regime_detail"):
                regimes = []
                for context in ("Bull", "Non bull"):
                    view = analyze_research_panel(panel, horizon, context, None if selected_ticker == "Tous" else selected_ticker, run["windows"])["results"]
                    regimes.append(view.loc[view["Factor"].eq(factor)])
                st.dataframe(pd.concat(regimes, ignore_index=True), hide_index=True)
    with st.expander("Redondance / corrélations"):
        if st.checkbox("Afficher les corrélations", key="research_correlations"):
            selected = panel.loc[(panel["Regime"].eq(regime) if regime != "Tous" else panel["Regime"].notna()) &
                                 (panel["Ticker"].eq(selected_ticker) if selected_ticker != "Tous" else True)]
            matrix, pairs = research_correlations(selected)
            st.caption("Moyenne à poids égaux des corrélations Pearson intra-ticker, au moins 30 paires par ticker. "
                       "|r| ≥ 0,80 suggère une redondance ; aucune feature n'est supprimée.")
            st.dataframe(matrix)
            for pair in pairs[:15]:
                st.write(pair)
    with st.expander("Combinaisons simples prédéfinies"):
        st.markdown("A : Bull + MM50 −5 à +2 %. B : Bull + RSI 35–45. C : Bull + force relative > 0. "
                    "D : Bull + MM50 −5 à +2 % + force relative > 0. E : Bull + drawdown60 −15 à −5 % + force relative > 0.")
        st.dataframe(research_summary_table(visible.loc[visible["Factor"].str.startswith("Combo")]), hide_index=True)
    st.caption("Buckets [borne basse, borne haute), dernier bucket ouvert ; position 52 semaines inclut 100 %. "
               "Combinaisons : bornes inclusives. Égalités de cours/MM200 ou MACD/Signal exclues de leurs tests stricts. "
               "Drawdown = excursion minimale du cours de clôture depuis T, pas une courbe de portefeuille. "
               "Les résultats restent exploratoires et sensibles à l'univers actuel (biais de survivance).")


def date_today():
    # Date UTC de l'étude, sans dépendance au fuseau du serveur.
    return pd.Timestamp.now(tz="UTC").date()
