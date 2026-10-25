"""Contrats causaux et statistiques du Research Lab, données synthétiques."""
from pathlib import Path
from time import perf_counter
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import research_lab as research
from scoring_helpers import load_functions


@pytest.fixture(scope="module")
def engine():
    namespace = load_functions()
    namespace["research"] = research
    return namespace


def prices(count=1600, seed=1, start="2018-01-01"):
    random = np.random.default_rng(seed)
    close = pd.Series(100 * np.exp(np.cumsum(random.normal(.0003, .012, count))),
                      index=pd.bdate_range(start, periods=count))
    return pd.DataFrame({"Open": close * .999, "High": close * 1.01, "Low": close * .99,
                         "Close": close, "Volume": 10000})


def prepare(engine, data):
    return research.prepare_research_features(data, engine["prepare_historical_features"])


def targets(engine, data):
    return research.prepare_forward_targets(data, engine["calculate_forward_returns"], engine["calculate_signal_drawdowns"])


@pytest.mark.parametrize("offset", [260, 500, 900])
def test_features_are_causal(engine, offset):
    original = prices()
    changed = original.copy()
    changed.iloc[offset+1:, :4] *= 4
    first, second = prepare(engine, original), prepare(engine, changed)
    pd.testing.assert_frame_equal(first.iloc[:offset+1], second.iloc[:offset+1])
    spy_first = prepare(engine, prices(seed=2))
    spy_changed = prices(seed=2)
    spy_changed.iloc[offset+1:, :4] *= .2
    contextual_first = research.attach_research_market_context(first, spy_first, "A")
    contextual_second = research.attach_research_market_context(second, prepare(engine, spy_changed), "A")
    pd.testing.assert_frame_equal(contextual_first.iloc[:offset+1], contextual_second.iloc[:offset+1])
    pd.testing.assert_frame_equal(research.assign_research_buckets(contextual_first).iloc[:offset+1],
                                  research.assign_research_buckets(contextual_second).iloc[:offset+1])


@pytest.mark.parametrize("value,expected", [(29.99,"< 30"), (30,"30–35"), (35,"35–40"),
    (40,"40–45"), (45,"45–50"), (50,"50–60"), (60,"60–70"), (70,"≥ 70")])
def test_rsi_bucket_boundaries(value, expected):
    assert research.research_bucket(pd.Series([value]), "RSI").iloc[0] == expected


@pytest.mark.parametrize("value,expected", [(-.08001,"< −8 %"), (-.08,"−8 à −5 %"),
    (-.05,"−5 à −2 %"), (-.02,"−2 à +2 %"), (.02,"+2 à +5 %"), (.05,"+5 à +10 %"), (.1,"≥ +10 %")])
def test_mm50_bucket_boundaries(value, expected):
    assert research.research_bucket(pd.Series([value]), "distance_mm50").iloc[0] == expected


@pytest.mark.parametrize("feature", research.RESEARCH_BUCKET_EDGES)
def test_buckets_preserve_missing_and_nonfinite_values(feature):
    assert research.research_bucket(pd.Series([np.nan, np.inf, -np.inf]), feature).isna().all()


def test_position_includes_upper_boundary_and_rejects_out_of_range():
    labels = research.research_bucket(pd.Series([0, .2, .8, 1, 1.1]), "position_52w")
    assert list(labels.iloc[:4]) == ["0–20 %", "20–40 %", "80–100 %", "80–100 %"]
    assert pd.isna(labels.iloc[4])


def test_relative_strength_uses_aligned_spy_and_no_self_comparison(engine):
    stock, spy = prepare(engine, prices(seed=3)), prepare(engine, prices(seed=4))
    contextual = research.attach_research_market_context(stock, spy, "AAPL")
    expected = stock["Close"] / stock["Close"].shift(60) - spy["Close"] / spy["Close"].shift(60)
    pd.testing.assert_series_equal(contextual["relative_strength_60"], expected, check_names=False)
    assert research.attach_research_market_context(spy, spy, "SPY")["relative_strength_60"].isna().all()


def test_unknown_spy_date_stays_unknown_and_does_not_backfill(engine):
    frame = prepare(engine, prices())
    date = frame.index[500]
    spy = prepare(engine, prices(seed=2)).drop(index=date)
    contextual = research.attach_research_market_context(frame, spy, "A")
    assert pd.isna(contextual.loc[date, "Regime"])
    assert pd.isna(contextual.loc[date, "relative_strength_60"])


def test_market_regime_requires_price_and_rising_mm200(engine):
    frame = prepare(engine, prices(300))
    spy = frame.copy()
    spy["MM200"] = np.arange(len(spy)) + 100.
    spy["Close"] = spy["MM200"] + 10
    contextual = research.attach_research_market_context(frame, spy, "A")
    assert contextual["Regime"].iloc[-1] == "Bull"
    spy.iloc[-1, spy.columns.get_loc("Close")] = spy["MM200"].iloc[-1] - 10
    assert research.attach_research_market_context(frame, spy, "A")["Regime"].iloc[-1] == "Non bull"
    spy["MM200"] = 1000 - np.arange(len(spy))
    spy["Close"] = spy["MM200"] + 10
    assert research.attach_research_market_context(frame, spy, "A")["Regime"].iloc[-1] == "Non bull"
    assert contextual["Regime"].iloc[:20].isna().all()


@pytest.mark.parametrize("horizon", research.RESEARCH_HORIZONS)
def test_forward_target_horizon_and_drawdown(engine, horizon):
    frame = pd.DataFrame({"Close": np.arange(100, 200, dtype=float)}, index=pd.bdate_range("2020-01-01", periods=100))
    result = targets(engine, frame)
    assert result[f"return_{horizon}"].iloc[0] == pytest.approx((100+horizon)/100 - 1)
    assert result[f"forward_drawdown_{horizon}"].iloc[0] == 0
    assert result[f"return_{horizon}"].iloc[-horizon:].isna().all()
    assert result[f"forward_drawdown_{horizon}"].iloc[-horizon:].isna().all()
    assert result[f"target_date_{horizon}"].iloc[-horizon:].isna().all()


def panel_fixture():
    # Rendements volontairement séparés entre régimes : une baseline globale
    # peut donner un faux alpha alors que la baseline Bull annule l'effet.
    return pd.DataFrame({"Ticker": ["A"]*4, "Date": pd.date_range("2020-01-01", periods=4),
        "Regime": ["Bull", "Bull", "Non bull", "Non bull"],
        "return_20": [.10, .10, -.10, -.10], "forward_drawdown_20": [-.02]*4,
        "target_date_20": pd.date_range("2020-01-21", periods=4),
        "bucket::RSI": ["35–40", "35–40", "60–70", "60–70"]})


def test_regime_baseline_does_not_use_other_regimes():
    result = research.analyze_research_panel(panel_fixture(), 20, "Bull")["results"].iloc[0]
    assert result["Baseline Global"] == pytest.approx(0)
    assert result["Baseline Regime"] == pytest.approx(.10)
    assert result["Alpha Global"] == pytest.approx(.10)
    assert result["Alpha Regime"] == pytest.approx(0)
    assert result["Observations"] == 2


def test_all_regime_alpha_uses_observed_ticker_regime_mix():
    panel = panel_fixture()
    panel["bucket::RSI"] = ["X", "Other", "X", "Other"]
    result = research.analyze_research_panel(panel, 20)["results"]
    assert result["Alpha Regime"].eq(0).all()


def test_equal_ticker_mean_differs_from_observation_mean():
    panel = panel_fixture().iloc[:1].copy()
    left = pd.concat([panel] * 9, ignore_index=True)
    left["return_20"] = .1
    right = panel.copy()
    right["Ticker"], right["return_20"] = "B", -.1
    result = research.analyze_research_panel(pd.concat([left, right]), 20)["results"].iloc[0]
    assert result["Mean Return"] == pytest.approx(.08)
    assert result["Mean Return per Ticker"] == pytest.approx(0)
    assert result["Tickers"] == 2


def test_temporal_windows_exclude_exits_after_validation_and_rebuild_baseline():
    panel = panel_fixture()
    panel["return_20"] = [.2, .1, -1, -1]
    # Seule la première observation et sa cible sont dans la fenêtre.
    panel["target_date_20"] = pd.to_datetime(["2020-01-02", "2021-01-01", "2021-01-01", "2021-01-01"])
    result = research.analyze_research_panel(panel, 20, windows=(("W", "2020-01-01", "2020-01-04"),))
    window = result["windows"].iloc[0]
    assert window["Observations"] == 1
    assert window["Baseline Global"] == .2
    assert window["Alpha Regime"] == 0


@pytest.mark.parametrize("changes,expected", [({}, "Encourageant"), ({"Observations":20}, "Mitigé"),
    ({"Valid Windows":0,"Positive Windows":0}, "Mitigé"), ({"Alpha Regime per Ticker":-.01}, "Faible"),
    ({"Positive Tickers":1}, "Faible"), ({"Positive Windows":1}, "Faible"), ({"Median Alpha":-.01}, "Mitigé")])
def test_robustness_describes_evidence(changes, expected):
    row = {"Observations":200,"Alpha Regime":.01,"Alpha Regime per Ticker":.01,"Median Alpha":.005,
           "Valid Tickers":8,"Positive Tickers":6,"Valid Windows":5,"Positive Windows":4}
    row.update(changes)
    assert research.research_robustness(row) == expected


@pytest.mark.parametrize("count,warning", [(0,"Échantillon très faible"),(29,"Échantillon très faible"),
    (30,"Échantillon limité"),(100,"Échantillon limité"),(101,"")])
def test_sample_warnings(count, warning):
    assert research.research_sample_warning(count) == warning


def test_explicit_combinations_exact_boundaries_and_no_unknown_market():
    frame = pd.DataFrame({"Regime":["Bull","Bull","Bull","Bull","Non bull",pd.NA],
        "distance_mm50":[-.05,.02,.02001,-.05001,0,0], "RSI":[35,45,45.01,34.99,40,40],
        "relative_strength_60":[.01,.01,0,-.01,.1,.1], "drawdown_60":[-.15,-.05,-.049,-.151,-.1,-.1]})
    result = research.research_combinations(frame)
    assert list(result) == ["Combo A","Combo B","Combo C","Combo D","Combo E"]
    for name in ("Combo A","Combo B","Combo D","Combo E"):
        assert result[name].tolist() == [True,True,False,False,False,False]
    assert result["Combo C"].tolist() == [True,True,False,False,False,False]


def test_missing_spy_is_explicit_blocker(engine):
    with pytest.raises(ValueError, match="SPY indisponible"):
        research.build_research_panel({"A":prices()}, "2020-01-01", "2023-01-01",
                                      lambda h: prepare(engine,h), lambda f: targets(engine,f))


def test_features_prepared_once_per_ticker_and_targets_bounded(engine):
    histories = {name: prices(1000,seed) for seed,name in enumerate(("A","B","SPY"))}
    counts = {"features":0,"targets":0}
    def features(history):
        counts["features"] += 1
        return prepare(engine, history)
    def forward(frame):
        counts["targets"] += 1
        return targets(engine, frame)
    panel, _ = research.build_research_panel(histories,"2019-01-01","2021-01-01",features,forward)
    assert counts == {"features":3,"targets":3}
    assert panel["Date"].max() <= pd.Timestamp("2021-01-01")
    for horizon in research.RESEARCH_HORIZONS:
        assert panel[f"target_date_{horizon}"].dropna().le(pd.Timestamp("2021-01-01")).all()
    assert panel["history_count"].ge(252).all()


def test_download_uses_calculated_8year_start_and_always_spy(monkeypatch):
    captured = {}
    def download(symbols, **kwargs):
        captured.update(symbols=symbols, **kwargs)
        frame = prices(10)
        return pd.concat({symbol:frame for symbol in symbols},axis=1)
    monkeypatch.setattr(research.yf,"download",download)
    research.load_research_histories.clear()
    result, warnings = research.load_research_histories(("A",),"2018-01-01","2026-01-01")
    assert set(result) == {"A","SPY"}
    assert captured["start"] == (pd.Timestamp("2018-01-01")-pd.Timedelta(days=550)).date()
    assert captured["end"] == pd.Timestamp("2026-01-02").date()
    assert "period" not in captured
    assert not warnings
    research.load_research_histories.clear()


def test_correlation_detects_redundancy_without_removing_features():
    count=100
    panel=pd.DataFrame({name:np.arange(count,dtype=float) for name in research.RESEARCH_CORRELATION_COLUMNS})
    panel["Ticker"]="A"
    matrix,pairs=research.research_correlations(panel)
    assert matrix.loc["RSI","momentum_20"] == pytest.approx(1)
    assert any("RSI et momentum_20" in pair for pair in pairs)
    assert len(matrix)==10


def test_empty_filter_keeps_export_schema():
    result=research.analyze_research_panel(panel_fixture(),20,"Bull","MISSING")
    assert result["results"].empty
    assert tuple(result["results"].columns)==research.RESEARCH_EXPORT_COLUMNS


def test_recent_and_forward_drawdown_are_distinct_at_all_horizons(engine):
    histories={name:prices(1300,seed) for seed,name in enumerate(("A","B","SPY"))}
    panel,_=research.build_research_panel(histories,"2019-01-01","2022-12-31",
                                         lambda h:prepare(engine,h),lambda f:targets(engine,f))
    assert panel.columns.is_unique
    assert panel.groupby("Ticker")["drawdown_60"].last().notna().all()
    assert panel.groupby("Ticker").tail(1)["forward_drawdown_60"].isna().all()
    for horizon in research.RESEARCH_HORIZONS:
        result=research.analyze_research_panel(panel,horizon)["results"]
        assert not result.empty
        assert result["Horizon"].eq(horizon).all()
    matrix,_=research.research_correlations(panel)
    assert matrix.shape==(10,10)
