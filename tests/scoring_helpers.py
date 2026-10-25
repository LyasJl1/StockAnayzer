"""Charge les fonctions pures de l'application sans exécuter sa navigation."""
import ast
from html import escape
from math import isfinite, sqrt
from pathlib import Path
from typing import Any

import pandas as pd

APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


def load_functions():
    tree = ast.parse(APP_PATH.read_text(encoding="utf-8"))
    nodes = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)]
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            node.decorator_list = []
            nodes.append(node)
        elif isinstance(node, ast.Assign) and all(
            isinstance(target, ast.Name) and target.id.isupper() for target in node.targets
        ):
            nodes.append(node)
    namespace = {"pd": pd, "Any": Any, "sqrt": sqrt, "isfinite": isfinite, "escape": escape}
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(APP_PATH), "exec"), namespace)
    return namespace


def history(count=252):
    closes = pd.Series([80 + i * .08 + (i % 7) * .1 for i in range(count)],
                       index=pd.date_range("2024-01-01", periods=count))
    return pd.DataFrame({"Close": closes, "High": closes + 2, "Low": closes - 2, "Volume": 10000})


def complete_info():
    return {"trailingPE": 18, "priceToBook": 2.5, "profitMargins": .18,
            "returnOnEquity": .22, "freeCashflow": 1000, "revenueGrowth": .12,
            "earningsGrowth": .18, "debtToEquity": 40, "beta": .9, "currency": "USD"}
