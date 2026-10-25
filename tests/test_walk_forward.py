"""Tests déterministes de la validation walk-forward."""
import ast
from math import isfinite
from pathlib import Path
from typing import Any

import pandas as pd

NAMES = {"valid_number", "calculate_alpha", "build_walk_forward_windows",
         "_bounded_window_statistics", "aggregate_walk_forward",
         "walk_forward_robustness", "build_walk_forward_interpretation"}
tree = ast.parse(Path("app.py").read_text(encoding="utf-8"))
nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in NAMES]
namespace = {"pd": pd, "Any": Any, "isfinite": isfinite, "BACKTEST_HORIZONS": (5, 20, 60)}
exec(compile(ast.Module(body=nodes, type_ignores=[]), "app.py", "exec"), namespace)
globals().update({name: namespace[name] for name in NAMES})


def timeline(start="2024-12-01", count=300):
    index = pd.date_range(start, periods=count, freq="B")
    return pd.DataFrame({"close": range(100, 100 + count), "_position": range(count)}, index=index)


def test_calendar_windows_keep_observation_and_validation_separate():
    windows = build_walk_forward_windows("2021-01-01", "2025-12-31", 3, 1, 1)
    assert len(windows) == 2
    assert windows[0]["observation_end"] == pd.Timestamp("2023-12-31")
    assert windows[0]["validation_start"] == pd.Timestamp("2024-01-01")
    assert windows[1]["validation_start"] == pd.Timestamp("2025-01-01")


def test_signal_before_validation_excluded_and_signal_inside_included():
    data = timeline()
    signals = data.loc[[pd.Timestamp("2024-12-31"), pd.Timestamp("2025-01-02")]]
    result = _bounded_window_statistics(signals, data, "2025-01-01", "2025-12-31")
    assert result[5]["observations"] == 1


def test_horizon_cannot_cross_validation_boundary():
    data = timeline("2025-01-01", 261)
    signal_date = data.index[data.index.get_indexer([pd.Timestamp("2025-12-20")], method="bfill")[0]]
    result = _bounded_window_statistics(data.loc[[signal_date]], data, "2025-01-01", "2025-12-31")
    assert result[5]["observations"] == 1
    assert result[20]["observations"] == 0


def test_future_mutation_does_not_change_statistics_ending_at_t():
    original = timeline("2025-01-01", 300)
    changed = original.copy()
    changed.loc[changed.index > "2025-12-31", "close"] *= 100
    observations = original.iloc[[20, 100, 200]]
    first = _bounded_window_statistics(observations, original, "2025-01-01", "2025-12-31")
    second = _bounded_window_statistics(observations, changed, "2025-01-01", "2025-12-31")
    assert first == second


def test_aggregation_keeps_three_distinct_alpha_measures():
    rows = [
        {"Moteur": "V1", "Horizon": 20, "Alpha": -.01, "Signaux": 1},
        {"Moteur": "V1", "Horizon": 20, "Alpha": .05, "Signaux": 9},
    ]
    result = aggregate_walk_forward(rows).iloc[0]
    assert result["Alpha moyen"] == .02
    assert result["Alpha médian"] == .02
    assert abs(result["Alpha pondéré"] - .044) < 1e-12


def test_interpretation_and_robustness_are_deterministic():
    strong = {"Alpha moyen": .02, "Alpha médian": .01, "Fenêtres positives": 3,
              "Fenêtres valides": 4, "Observations": 40}
    assert walk_forward_robustness(strong, .75, True) == "Encourageante"
    assert "reste historique" in build_walk_forward_interpretation("V3 Early", strong)
    weak = {**strong, "Alpha moyen": -.01, "Fenêtres positives": 1}
    assert walk_forward_robustness(weak, .25, True) == "Faible"
    assert "ne conserve pas" in build_walk_forward_interpretation("V3 Early", weak)


def test_engines_are_precomputed_before_window_loop_and_rules_are_fixed():
    source = Path("app.py").read_text(encoding="utf-8")
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == "calculate_walk_forward")
    loops = [node for node in ast.walk(function) if isinstance(node, ast.For)]
    window_loop = next(node for node in loops if ast.unparse(node.iter) == "windows")
    loop_source = ast.unparse(window_loop)
    assert "calculate_historical_timing_series" not in loop_source
    assert 'extract_threshold_signals(v1, 70)' in source
    assert 'extract_threshold_signals(v2, 70)' in source
