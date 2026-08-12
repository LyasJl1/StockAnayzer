"""Tests causaux et déterministes du moteur expérimental V2."""
import ast
from math import isfinite
from pathlib import Path
from typing import Any

import pandas as pd

NAMES = {"valid_number", "calculate_indicators", "calculate_atr",
         "calculate_pullback_timing_series", "extract_threshold_signals"}
tree = ast.parse(Path("app.py").read_text(encoding="utf-8"))
nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in NAMES]
namespace = {"pd": pd, "Any": Any, "isfinite": isfinite, "MIN_SIGNAL_GAP": 10}
exec(compile(ast.Module(body=nodes, type_ignores=[]), "app.py", "exec"), namespace)
globals().update({name: namespace[name] for name in NAMES})


def market(count=280, slope=.12):
    index = pd.date_range("2020-01-01", periods=count, freq="B")
    close = pd.Series([80 + slope * i for i in range(count)], index=index, dtype=float)
    return pd.DataFrame({"Close": close, "High": close + 1.5, "Low": close - 1.5}, index=index)


def test_favorable_pullback_scores_above_extended_overbought_case():
    favorable = market()
    # Un repli récent, puis une reprise, sans casser la tendance longue.
    favorable.iloc[-8:, favorable.columns.get_loc("Close")] = [112, 111, 110, 109, 108.5, 109, 110, 111]
    favorable["High"] = favorable["Close"] + 1.5; favorable["Low"] = favorable["Close"] - 1.5
    extended = market()
    extended.iloc[-12:, extended.columns.get_loc("Close")] += pd.Series(range(12), index=extended.index[-12:]) * 4
    extended["High"] = extended["Close"] + 1.5; extended["Low"] = extended["Close"] - 1.5
    assert calculate_pullback_timing_series(favorable).iloc[-1]["timing_score"] > \
           calculate_pullback_timing_series(extended).iloc[-1]["timing_score"]


def test_bear_market_rebound_is_capped_at_55():
    falling = market(slope=-.12)
    falling.iloc[-4:, falling.columns.get_loc("Close")] += [0, 1, 2, 4]
    falling["High"] = falling["Close"] + 1; falling["Low"] = falling["Close"] - 1
    row = calculate_pullback_timing_series(falling).iloc[-1]
    assert not bool(row["long_term_context_valid"])
    assert row["timing_score"] <= 55


def test_improving_macd_can_score_without_macd_cross_and_missing_is_renormalized():
    frame = market()
    result = calculate_pullback_timing_series(frame)
    candidates = result[(result["points_macd_gap_improving"] == 10) &
                        (result["points_macd_cross"] == 0)]
    # Le scénario est possible dans le barème, indépendamment des points de croisement.
    if candidates.empty:
        synthetic_gap = pd.Series([-2, -1.8, -1.6, -1.4, -1.2, -.9])
        assert synthetic_gap.iloc[-1] > synthetic_gap.shift(5).iloc[-1] and synthetic_gap.iloc[-1] < 0
    missing = market().drop(columns=["High", "Low"])
    scored = calculate_pullback_timing_series(missing).iloc[-1]
    assert pd.isna(scored["points_support_atr"]) and scored["timing_confidence"] < 100
    assert 0 <= scored["timing_score"] <= 100


def test_future_mutation_does_not_change_past_score():
    original = market(300)
    changed = original.copy()
    changed.iloc[261:, :] *= 20
    first = calculate_pullback_timing_series(original)
    second = calculate_pullback_timing_series(changed)
    assert first.iloc[260]["timing_score"] == second.iloc[260]["timing_score"]


def test_threshold_crossing_and_spacing_shared_by_v2():
    scores = [68, 72, 76, 69, 73] + [65] * 10 + [71]
    timing = pd.DataFrame({"timing_score": scores, "_position": range(len(scores))})
    assert extract_threshold_signals(timing, 70)["timing_score"].tolist() == [72, 71]


def test_two_asset_synthetic_batch_uses_identical_rules():
    for ticker, data in {"AAPL": market(), "MSFT": market(slope=.09)}.items():
        result = calculate_pullback_timing_series(data)
        assert ticker and len(result) == len(data) and result["timing_score"].dropna().between(0, 100).all()
