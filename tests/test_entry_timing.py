"""Tests purs du moteur Timing sans lancer l'interface Streamlit."""
import ast
from math import isfinite, sqrt
from pathlib import Path
from typing import Any
import pandas as pd

NAMES = {"valid_number", "calculate_atr", "calculate_price_zones", "timing_verdict",
         "build_timing_conditions", "calculate_entry_timing", "build_entry_decision"}
tree = ast.parse(Path("app.py").read_text(encoding="utf-8"))
nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in NAMES]
namespace = {"pd": pd, "Any": Any, "isfinite": isfinite, "sqrt": sqrt}
exec(compile(ast.Module(body=nodes, type_ignores=[]), "app.py", "exec"), namespace)
globals().update({name: namespace[name] for name in NAMES})


def market(close=100.0, rsi=55.0, macd=2.0, signal=1.0, mm50=95.0, mm200=90.0):
    count = 252
    closes = pd.Series([close * (.82 + .18 * i/(count-1)) for i in range(count)])
    frame = pd.DataFrame({"Close": closes, "High": closes + 2, "Low": closes - 2})
    frame["MM50"] = mm50 - 2 + pd.Series(range(count)) * 2/(count-1)
    frame["MM200"] = mm200
    frame["RSI"] = rsi
    frame["MACD"] = macd - .5 + pd.Series(range(count)) * .5/(count-1)
    frame["Signal"] = signal
    return frame


def test_favorable_and_extension_penalty():
    favorable = calculate_entry_timing(market(), {}, {"days_until_earnings": 20}, "Investisseur")
    extended = calculate_entry_timing(market(close=118, rsi=77, mm50=100), {}, {"days_until_earnings": 20}, "Investisseur")
    assert favorable["score"] >= 70
    assert extended["score"] < favorable["score"]
    assert any("fortement étendu" in item for item in extended["warning_signals"])


def test_missing_data_are_excluded_and_event_is_optional():
    frame = market()
    frame["MM200"] = float("nan")
    result = calculate_entry_timing(frame, {}, {}, "Trader / Swing")
    criterion = next(x for x in result["criteria"] if x["key"] == "price_above_mm200")
    event = next(x for x in result["criteria"] if x["key"] == "earnings")
    assert not criterion["available"] and not event["available"]
    assert result["maximum_available_points"] == 72


def test_conditions_and_combined_decisions():
    conditions = build_timing_conditions(176, 181, 170, 41, -1.2, -.6, 5)
    assert abs(float(conditions[1]["detail"].split()[0].replace("+", "")) - 2.84) < .01
    assert not conditions[3]["passed"] and not conditions[4]["passed"]
    assert "Momentum intéressant" in build_entry_decision(45, {"score": 85}, 60)["verdict"]
    assert "attendre une confirmation" in build_entry_decision(82, {"score": 60}, 60)["verdict"]


def test_atr_and_pivot_zones():
    frame = market()
    atr = calculate_atr(frame).iloc[-1]
    assert atr > 0 and abs(atr / frame["Close"].iloc[-1] - atr/100) < 1e-12
    zones = calculate_price_zones(frame, atr)
    assert zones["support"] < frame["Close"].iloc[-1]
    assert zones["resistance"] > frame["Close"].iloc[-1]
