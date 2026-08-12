"""Tests causaux et statistiques du backtest Timing."""
import ast
from pathlib import Path
from typing import Any

import pandas as pd

from test_entry_timing import (calculate_atr, calculate_entry_timing, valid_number)

NAMES = {"calculate_indicators", "calculate_historical_timing_series",
         "extract_backtest_signals", "calculate_forward_returns",
         "calculate_signal_drawdowns", "calculate_backtest_statistics",
         "calculate_baseline_statistics", "sample_size_warning",
         "build_backtest_interpretation", "format_percentage"}
tree = ast.parse(Path("app.py").read_text(encoding="utf-8"))
nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in NAMES]
namespace = {"pd": pd, "Any": Any, "calculate_atr": calculate_atr,
             "calculate_entry_timing": calculate_entry_timing, "valid_number": valid_number,
             "BACKTEST_HORIZONS": (5, 20, 60), "MIN_SIGNAL_GAP": 10}
exec(compile(ast.Module(body=nodes, type_ignores=[]), "app.py", "exec"), namespace)
globals().update({name: namespace[name] for name in NAMES})


def timeline(values):
    dates = pd.date_range("2020-01-01", periods=len(values), freq="B")
    return pd.DataFrame({"date": dates, "close": values, "timing_score": 60.0,
                         "_position": range(len(values))}, index=dates)


def prices(count=280):
    close = [80 + i * .15 for i in range(count)]
    close_series = pd.Series(close)
    return pd.DataFrame({"Close": close, "High": close_series.add(1).tolist(),
                         "Low": close_series.sub(1).tolist()},
                        index=pd.date_range("2020-01-01", periods=count, freq="B"))


def test_exact_forward_horizons_up_and_down_and_incomplete():
    data = timeline([100 + i for i in range(170)])
    signal = data.iloc[[100, 160]]
    result = calculate_forward_returns(signal, data)
    assert result.iloc[0]["return_5"] == 205 / 200 - 1
    assert result.iloc[0]["return_20"] == 220 / 200 - 1
    assert result.iloc[0]["return_60"] == 260 / 200 - 1
    assert pd.notna(result.iloc[1]["return_5"]) and pd.isna(result.iloc[1]["return_20"])
    falling = timeline([200 - i for i in range(100)])
    loss = calculate_forward_returns(falling.iloc[[20]], falling).iloc[0]["return_20"]
    assert loss < 0  # un résultat défavorable n'est jamais filtré


def test_threshold_crossings_and_minimum_gap():
    scores = [65, 68, 71, 74, 77, 69, 72]
    data = timeline([100] * len(scores)); data["timing_score"] = scores
    assert extract_backtest_signals(data, 70, False)["timing_score"].tolist() == [71, 72]
    assert extract_backtest_signals(data, 70, True)["timing_score"].tolist() == [71]


def test_future_mutation_cannot_change_past_timing_and_missing_mm200_is_excluded():
    original = prices()
    changed = original.copy()
    changed.iloc[241:, changed.columns.get_loc("Close")] *= 8
    changed.iloc[241:, changed.columns.get_loc("High")] *= 8
    changed.iloc[241:, changed.columns.get_loc("Low")] *= 8
    first = calculate_historical_timing_series(original, "Investisseur")
    second = calculate_historical_timing_series(changed, "Investisseur")
    assert first.iloc[240]["timing_score"] == second.iloc[240]["timing_score"]
    assert pd.isna(first.iloc[100]["MM200"])
    assert first.iloc[100]["timing_confidence"] < first.iloc[240]["timing_confidence"]


def test_drawdown_baseline_warning_and_honest_interpretation():
    data = timeline([100, 95, 90, 105, 110, 108] + [108] * 70)
    signal = calculate_forward_returns(data.iloc[[0]], data)
    signal = calculate_signal_drawdowns(signal, data)
    assert abs(signal.iloc[0]["drawdown_5"] - -.1) < 1e-12
    assert abs(signal.iloc[0]["mfe_5"] - .1) < 1e-12
    assert "très faible" in sample_size_warning(6)
    high = {20: {"observations": 6, "mean": -.02, "positive": .33}}
    baseline = {20: {"observations": 70, "mean": .03, "positive": .6}}
    low = {20: {"observations": 6, "mean": .01, "positive": .5}}
    text = build_backtest_interpretation(high, baseline, low, 70)
    assert "n’ont pas historiquement surperformé" in text
    baseline_result = calculate_baseline_statistics(data)
    assert baseline_result[5]["observations"] == len(data) - 5
