"""Tests déterministes de V3 — Régime / Setup / Trigger."""
import ast
from math import isfinite, sqrt
from pathlib import Path
from typing import Any

import pandas as pd
import pandas.testing as pdt


NAMES = {"valid_number", "calculate_atr", "_v3_prepared", "_v3_condition",
         "evaluate_v3_regime", "evaluate_v3_setup", "evaluate_v3_trigger",
         "build_v3_entry_status", "calculate_rigorous_entry_v3",
         "calculate_v3_timing_series_reference", "calculate_v3_timing_series",
         "extract_v3_signals", "analyze_v3_confirmations"}
tree = ast.parse(Path("app.py").read_text(encoding="utf-8"))
nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in NAMES]
namespace = {"pd": pd, "Any": Any, "isfinite": isfinite, "sqrt": sqrt, "MIN_SIGNAL_GAP": 10}
exec(compile(ast.Module(body=nodes, type_ignores=[]), "app.py", "exec"), namespace)
globals().update({name: namespace[name] for name in NAMES})


def scenario(close=119.0, rsi=43.0, bearish=False, extended=False):
    index = pd.date_range("2024-01-01", periods=260, freq="B")
    mm200 = pd.Series([90 + i * .1 for i in range(260)], index=index)
    mm50 = pd.Series(120.0, index=index)
    prices = pd.Series(118.0, index=index)
    prices.iloc[-6:] = [118, 117, 117.5, 118, 119.5, close]
    if bearish:
        mm200[:] = 120; mm50[:] = 105; prices[:] = 95; prices.iloc[-4:] = [94, 94.5, 95, 96]
    if extended:
        prices.iloc[-1] = 138; rsi = 74
    macd = pd.Series(-1.0, index=index); signal = pd.Series(-.5, index=index)
    macd.iloc[-6:] = [-1.2, -1.15, -1.1, -1.0, -.9, -.8]
    rsi_series = pd.Series(40.0, index=index); rsi_series.iloc[-4:] = [40, 41, 42, rsi]
    return pd.DataFrame({"Close": prices, "High": prices + 1, "Low": prices - 1,
                         "MM50": mm50, "MM200": mm200, "RSI": rsi_series,
                         "MACD": macd, "Signal": signal, "ATR14": 2.0}, index=index)


def test_favorable_setup_and_partial_then_strong_trigger():
    partial = calculate_rigorous_entry_v3(scenario())
    assert partial["regime"]["status"] == "Favorable"
    assert partial["setup"]["status"] == "Présent"
    assert partial["trigger"]["status"] == "Partiel"
    strong = calculate_rigorous_entry_v3(scenario(close=121, rsi=47))
    assert strong["trigger"]["status"] == "Fort"


def test_bad_regime_caps_strength_and_extended_case_has_no_setup():
    bearish = calculate_rigorous_entry_v3(scenario(bearish=True))
    assert bearish["regime"]["status"] == "Défavorable"
    assert bearish["metrics"]["v3_signal_strength"] <= 55
    assert "aucune entrée" in bearish["status"]
    assert calculate_rigorous_entry_v3(scenario(extended=True))["setup"]["status"] == "Absent"


def test_future_mutation_does_not_change_v3_at_t():
    original = scenario()
    changed = original.copy()
    changed.iloc[251:, changed.columns.get_loc("Close")] *= 20
    before = calculate_rigorous_entry_v3(original.iloc[:251])
    after = calculate_rigorous_entry_v3(changed.iloc[:251])
    assert (before["regime"]["status"], before["setup"]["status"],
            before["trigger"]["status"], before["metrics"]["v3_signal_strength"]) == \
           (after["regime"]["status"], after["setup"]["status"],
            after["trigger"]["status"], after["metrics"]["v3_signal_strength"])


def test_vectorized_series_has_strict_reference_parity_and_no_look_ahead():
    exact = ["regime_status", "setup_status", "trigger_status", "v3_early", "v3_confirmed"]
    datasets = []
    for offset in (0, 3, 7):
        data = scenario()
        # Trois historiques déterministes exercent des états et transitions différents.
        wave = pd.Series([(((i + offset) % 17) - 8) * .12 for i in range(len(data))], index=data.index)
        data["Close"] += wave
        data["High"], data["Low"] = data["Close"] + 1, data["Close"] - 1
        datasets.append(data)
        reference = calculate_v3_timing_series_reference(data)
        vectorized = calculate_v3_timing_series(data)
        pdt.assert_frame_equal(reference[exact], vectorized[exact], check_dtype=False)
        pdt.assert_series_equal(reference["v3_signal_strength"], vectorized["v3_signal_strength"],
                                check_exact=False, atol=1e-10, rtol=0)

    cutoff = 245
    data = datasets[0]
    changed = data.copy()
    changed.iloc[cutoff + 1:, changed.columns.get_loc("Close")] *= 10
    changed.iloc[cutoff + 1:, changed.columns.get_loc("High")] *= 10
    changed.iloc[cutoff + 1:, changed.columns.get_loc("Low")] *= 10
    before, after = calculate_v3_timing_series(data), calculate_v3_timing_series(changed)
    pdt.assert_series_equal(before.iloc[cutoff], after.iloc[cutoff])


def test_unconfirmed_early_is_marked_after_fifteen_sessions():
    timeline = pd.DataFrame({"close": [100.0] * 30, "_position": range(30)})
    early = timeline.iloc[[2]].copy()
    confirmed = timeline.iloc[0:0].copy()
    result = analyze_v3_confirmations(early, confirmed, timeline)
    assert result["confirmation_rate"] == 0
    assert result["records"][0]["outcome"] == "Setup non confirmé"
    assert len(result["unconfirmed"]) == 1
