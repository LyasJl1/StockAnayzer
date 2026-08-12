"""Tests déterministes de la validation hors échantillon."""
import ast
from math import isfinite
from pathlib import Path
from typing import Any

import pandas as pd


NAMES = {"valid_number", "calculate_alpha", "format_alpha", "split_validation_universes",
         "timing_oos_signature",
         "aggregate_out_of_sample", "out_of_sample_robustness",
         "build_out_of_sample_interpretation", "build_horizon_stability",
         "build_v3_oos_interpretation", "aggregate_v3_confirmations",
         "aggregate_unconfirmed_v3_alpha", "_render_out_of_sample_results"}
tree = ast.parse(Path("app.py").read_text(encoding="utf-8"))
nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in NAMES]
namespace = {"pd": pd, "Any": Any, "isfinite": isfinite, "BACKTEST_HORIZONS": (5, 20, 60)}
exec(compile(ast.Module(body=nodes, type_ignores=[]), "app.py", "exec"), namespace)
globals().update({name: namespace[name] for name in NAMES})


def stats(mean, observations=10, drawdown=-.03):
    return {"mean": mean, "median": mean, "observations": observations,
            "drawdown_mean": drawdown, "drawdown_median": drawdown,
            "drawdown_worst": drawdown * 2}


def asset(ticker, v1, v2, baseline, observations=10, missing_60=False):
    result = {"ticker": ticker, "V1": {}, "V2": {}, "baseline": {}}
    for horizon in (5, 20, 60):
        unavailable = missing_60 and horizon == 60
        result["V1"][horizon] = stats(None if unavailable else v1, 0 if unavailable else observations)
        result["V2"][horizon] = stats(None if unavailable else v2, 0 if unavailable else observations)
        result["baseline"][horizon] = stats(None if unavailable else baseline, 0 if unavailable else 100)
    return result


def test_alpha_positive_negative_and_point_format():
    assert abs(calculate_alpha(.048, .030) - .018) < 1e-12
    assert format_alpha(calculate_alpha(.048, .030)) == "+1.8 pt"
    assert abs(calculate_alpha(.020, .049) - -.029) < 1e-12
    assert format_alpha(calculate_alpha(.020, .049)) == "-2.9 pt"
    assert calculate_alpha(None, .03) is None and format_alpha(None) == "N/D"


def test_overlap_is_always_removed_and_limit_applied():
    effective, overlap = split_validation_universes(
        ["AAPL", "MSFT", "GOOGL"], ["META", "AAPL", "AMZN"])
    assert effective == ("META", "AMZN")
    assert overlap == ("AAPL",)


def test_equal_weight_mean_median_and_observation_weighting():
    # Alphas en points : -1, 0, 1, 10. Moyenne 2,5 ; médiane 0,5.
    raw = [asset(str(i), alpha / 100, alpha / 100, 0, observations=i + 1)
           for i, alpha in enumerate((-1, 0, 1, 10))]
    summary = next(row for row in aggregate_out_of_sample(raw)
                   if row["engine"] == "V1" and row["horizon"] == 20)
    assert abs(summary["alpha_mean"] * 100 - 2.5) < 1e-12
    assert abs(summary["alpha_median"] * 100 - .5) < 1e-12
    expected_weighted = (-1 * 1 + 0 * 2 + 1 * 3 + 10 * 4) / 10
    assert abs(summary["alpha_weighted"] * 100 - expected_weighted) < 1e-12
    assert summary["positive_assets"] == 2 and summary["nonpositive_assets"] == 2


def test_missing_60_days_does_not_remove_short_horizons():
    summaries = aggregate_out_of_sample([asset("META", .04, .03, .02, missing_60=True)])
    v1 = {row["horizon"]: row for row in summaries if row["engine"] == "V1"}
    assert v1[5]["assets"] == 1 and v1[20]["assets"] == 1
    assert v1[60]["assets"] == 0 and v1[60]["alpha_mean"] is None


def test_weak_and_strong_deterministic_conclusions():
    weak = {"alpha_mean": -.003, "alpha_median": -.005, "positive_ratio": 4 / 12,
            "positive_assets": 4, "assets": 12, "observations": 60, "drawdown_mean": -.03}
    neutral_60 = {"alpha_mean": 0}
    assert out_of_sample_robustness(weak, neutral_60, -.03) == "Faible"
    assert "V1 ne montre pas d'avantage robuste hors échantillon" in \
        build_out_of_sample_interpretation("V1", weak, "Faible")

    strong = {"alpha_mean": .012, "alpha_median": .009, "positive_ratio": 9 / 12,
              "positive_assets": 9, "assets": 12, "observations": 60, "drawdown_mean": -.03}
    assert out_of_sample_robustness(strong, neutral_60, -.03) == "Encourageante"
    text = build_out_of_sample_interpretation("V1", strong, "Encourageante")
    assert "V1 montre un avantage historique encourageant hors échantillon sur cet univers" in text
    assert "validation sur d'autres périodes reste nécessaire" in text


def test_v3_interpretation_covers_early_confirmed_and_weak_cases():
    early = {"alpha_mean": .02, "alpha_median": .01, "positive_ratio": .75}
    confirmed = {"alpha_mean": .005, "alpha_median": .002, "positive_ratio": .55}
    assert "Setup V3 semble apporter davantage" in build_v3_oos_interpretation(early, confirmed)
    assert "confirmation V3 semble" in build_v3_oos_interpretation(
        {"alpha_mean": .002, "alpha_median": .001, "positive_ratio": .45},
        {"alpha_mean": .015, "alpha_median": .01, "positive_ratio": .7})
    assert "ne montre pas d'avantage robuste" in build_v3_oos_interpretation(
        {"alpha_mean": -.01, "alpha_median": -.02, "positive_ratio": .25},
        {"alpha_mean": -.02, "alpha_median": -.01, "positive_ratio": .2})


def test_confirmation_aggregation_uses_observed_pairs_only():
    records = [{"confirmed_position": 8, "delay": 3, "performance_to_confirmation": .02},
               {"confirmed_position": None, "delay": None, "performance_to_confirmation": None},
               {"confirmed_position": 20, "delay": 7, "performance_to_confirmation": -.01}]
    raw = [{"v3_confirmation": {"records": records, "early_count": 3, "confirmed_count": 2}}]
    result = aggregate_v3_confirmations(raw)
    assert result["early"] == 3 and result["confirmed"] == 2 and result["unconfirmed"] == 1
    assert result["rate"] == 2 / 3 and result["delay_mean"] == 5
    assert result["delay_median"] == 5 and abs(result["cost_mean"] - .005) < 1e-12


class StreamlitRecorder:
    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        def record(*args, **kwargs):
            self.calls.append((name, args, kwargs))
        return record


def test_empty_validation_displays_explicit_diagnostics():
    recorder = StreamlitRecorder()
    old_st = _render_out_of_sample_results.__globals__.get("st")
    _render_out_of_sample_results.__globals__["st"] = recorder
    try:
        _render_out_of_sample_results({
            "validation": [], "conception": [], "ignored": ["META", "AMZN"],
            "downloaded_count": 2,
        })
    finally:
        _render_out_of_sample_results.__globals__["st"] = old_st
    messages = [args[0] for _, args, _ in recorder.calls if args]
    assert "❌ Aucun actif hors échantillon n'a pu être calculé." in messages
    assert any("META, AMZN" in message for message in messages)


def test_valid_result_aggregation_is_renderable_with_empty_horizons():
    raw = [asset("META", .04, .03, .02, missing_60=True)]
    summaries = aggregate_out_of_sample(raw)
    assert len(summaries) == 12
    assert all("alpha_mean" in row for row in summaries)


def test_validation_exception_is_reported_and_render_is_immediate():
    source = Path("app.py").read_text(encoding="utf-8")
    lab = next(node for node in ast.parse(source).body
               if isinstance(node, ast.FunctionDef) and node.name == "render_timing_lab")
    handlers = [handler for node in ast.walk(lab) if isinstance(node, ast.Try)
                for handler in node.handlers]
    handler_calls = {call.func.attr for handler in handlers for call in ast.walk(handler)
                     if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)}
    assert {"error", "exception"} <= handler_calls
    assert source.index('st.session_state["timing_oos_result"] = {') < \
        source.index("_render_out_of_sample_results(stored_after_calculation)")


def test_signature_is_immutable_and_stable():
    effective = ("META", "AMZN")
    before = timing_oos_signature(effective, "3y", 70, 70)
    after = timing_oos_signature(tuple(effective), str("3y"), int(70), int(70))
    assert before == after == (("META", "AMZN"), "3y", 70, 70)


def test_unconfirmed_v3_alpha_is_aggregated_ticker_by_ticker():
    raw = []
    for ticker, performance, baseline, observations in (
            ("META", .10, .02, 1), ("AMZN", .04, .03, 9)):
        item = {"ticker": ticker, "v3_unconfirmed": {}, "baseline": {}}
        for horizon in (5, 20, 60):
            item["v3_unconfirmed"][horizon] = stats(performance, observations)
            item["baseline"][horizon] = stats(baseline, 100)
        raw.append(item)
    result = aggregate_unconfirmed_v3_alpha(raw)[20]
    # Moyenne égale des alphas par ticker : ((10-2) + (4-3)) / 2 = 4,5 points.
    assert abs(result["alpha_mean"] - .045) < 1e-12
