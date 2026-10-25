"""Contrats du moteur multifactoriel, couverture et calibration (sans réseau)."""
import math

import pytest

from scoring_helpers import complete_info, history, load_functions


@pytest.fixture
def engine():
    return load_functions()


@pytest.mark.parametrize("values, expected", [([], None), ([None, None], None),
    ([None, float("nan"), float("inf")], None), ([0, None, 80], 40)])
def test_average_preserves_absence(engine, values, expected):
    assert engine["average_available"](*values) == expected


@pytest.mark.parametrize("mode", ["Investisseur", "Trader / Swing"])
def test_complete_data_preserve_legacy_raw_score(engine, mode):
    # Valeurs fixées des règles avant refactorisation, non recalculées par le test.
    data = engine["calculate_indicators"](history())
    result = engine["evaluate_scoring"](data, complete_info(), mode)
    expected = {"Fondamentaux": 82, "Valorisation": 68, "Croissance": 79,
                "Technique": 61.75, "Risque": 80}
    assert result["scores"] == pytest.approx(expected)
    weights = engine["WEIGHTS"][mode]
    raw = sum(expected[key] * weight for key, weight in weights.items()) / 100
    assert result["raw_score"] == pytest.approx(raw)
    assert result["coverage_ratio"] == 1
    assert result["data_coverage"] == {"available": 14, "total": 14, "ratio": 1}
    maximum = engine["calculate_theoretical_score_bounds"](mode)["maximum"]
    assert result["theoretical_max"] == pytest.approx(maximum)
    assert result["score"] == pytest.approx(raw / maximum * 100)


def test_absent_valuation_excludes_its_weight(engine):
    info = complete_info()
    del info["trailingPE"], info["priceToBook"]
    result = engine["evaluate_scoring"](engine["calculate_indicators"](history()), info, "Investisseur")
    assert result["scores"]["Valorisation"] is None
    assert result["available_weight"] == 80
    assert result["total_weight"] == 100
    assert result["coverage_ratio"] == .8
    assert result["missing_categories"] == ["Valorisation"]
    assert result["raw_score"] == pytest.approx((82*25 + 79*20 + 61.75*20 + 80*15)/80)


@pytest.mark.parametrize("category, keys", [
    ("Fondamentaux", ["profitMargins", "returnOnEquity", "freeCashflow"]),
    ("Croissance", ["revenueGrowth", "earningsGrowth"]),
    ("Valorisation", ["trailingPE", "priceToBook"]),
])
def test_all_metrics_of_category_missing(engine, category, keys):
    info = complete_info()
    for key in keys:
        info[key] = None
    scores, _, _ = engine["score_analysis"](engine["calculate_indicators"](history()), info, "Investisseur")
    assert scores[category] is None


def test_no_fundamental_data_never_becomes_neutral(engine):
    result = engine["evaluate_scoring"](engine["calculate_indicators"](history()), {}, "Investisseur")
    assert all(result["scores"][key] is None for key in ("Fondamentaux", "Valorisation", "Croissance"))
    assert result["available_categories"] == ["Technique", "Risque"]
    assert result["limited_coverage"]
    assert result["coverage_ratio"] == .35
    assert result["data_coverage"]["available"] == 5


@pytest.mark.parametrize("scores, count", [({}, 0), ({"Technique": 60}, 1)])
def test_fewer_than_two_categories_are_not_determinable(engine, scores, count):
    result = engine["calculate_global_score"](scores, "Investisseur")
    assert result["score"] is None
    assert result["status"] == "insufficient"
    assert len(result["available_categories"]) == count
    assert "non déterminable" in engine["score_verdict"](result["score"])


def test_weights_renormalized_for_two_categories(engine):
    result = engine["calculate_global_score"]({"Fondamentaux": 80, "Croissance": 60}, "Investisseur")
    assert result["raw_score"] == pytest.approx((80*25 + 60*20)/45)
    assert result["score"] == pytest.approx(result["raw_score"] / 92 * 100)
    assert result["available_weight"] == 45
    assert result["limited_coverage"]
    assert result["coverage_ratio"] == .45


@pytest.mark.parametrize("mode", ["Investisseur", "Trader / Swing"])
@pytest.mark.parametrize("value", [-50, 0, 30, 80, 100, 1000])
def test_normalized_score_is_bounded(engine, mode, value):
    result = engine["calculate_global_score"]({key: value for key in engine["WEIGHTS"][mode]}, mode)
    assert 0 <= result["score"] <= 100


def test_nonfinite_categories_are_missing(engine):
    result = engine["calculate_global_score"]({"Fondamentaux": math.nan, "Technique": math.inf}, "Investisseur")
    assert result["score"] is None
    assert result["raw_score"] is None
    assert result["coverage_ratio"] == 0


def test_available_metric_bounds_avoid_false_full_data_maximum(engine):
    # Sans FCF (plafond 78), Fondamentaux peut atteindre 92 et non 87,33.
    metrics = {"Fondamentaux": {"margin": 92, "roe": 92, "fcf": None},
               "Valorisation": {"pe": 90, "pb": None}}
    result = engine["calculate_global_score"]({"Fondamentaux": 92, "Valorisation": 90},
                                               "Investisseur", metric_scores=metrics)
    assert result["score"] == pytest.approx(100)
    assert result["theoretical_max"] == pytest.approx((92*25 + 90*20)/45)
    assert engine["calculate_data_coverage"](metrics)["available"] == 3


def test_metric_coverage_counts_usable_metrics_not_just_numbers(engine):
    info = complete_info()
    info.update(trailingPE=-1, priceToBook=0)
    data = engine["calculate_indicators"](history())
    data["MM50"] = 0
    data["Signal"] = float("nan")
    metrics = engine["calculate_scoring_metrics"](data, info)
    assert engine["calculate_data_coverage"](metrics) == {"available": 10, "total": 14, "ratio": 10/14}


@pytest.mark.parametrize("score, verdict", [(None, "Score global non déterminable — données insuffisantes"),
    (0, "Défavorable"), (39.99, "Défavorable"), (40, "Prudence"),
    (55, "Équilibré"), (70, "Favorable"), (85, "Très favorable"), (100, "Très favorable")])
def test_verdict_bands(engine, score, verdict):
    assert engine["score_verdict"](score) == verdict


@pytest.mark.parametrize("mode", ["Investisseur", "Trader / Swing"])
def test_shared_engine_in_comparison_screener_and_watchlist(engine, mode):
    sample, info = history(), complete_info()
    engine["load_stock_data"] = lambda *_: (sample, info)
    engine["load_screener_history"] = lambda *_: sample
    engine["load_screener_info"] = lambda *_: info
    expected = engine["evaluate_scoring"](engine["calculate_indicators"](sample), info, mode)
    comparison = engine["build_comparison_snapshot"]("XYZ", "1y", mode)
    row = engine["build_screener_row"]("XYZ", sample, info, mode)
    engine["extract_ticker_history"] = lambda *_: sample
    watchlist = engine["build_watchlist_rows"](["XYZ"], mode)[0]
    for snapshot in (comparison, row, watchlist):
        assert snapshot["global_score"] == expected["score"]
        assert snapshot["scoring"] == expected


@pytest.mark.parametrize("left,right,coverages", [(75,72,(.45,1)), (72,75,(1,.45)),
    (95,60,(.8,1)), (None,72,(.4,1))])
def test_comparison_reserves_winner_on_insufficient_coverage(engine, left, right, coverages):
    result = engine["comparison_winner"]("global_score", left, right, ("A", "B"), coverages)
    assert not result.startswith("🏆")


def test_comparable_coverage_keeps_score_advantage(engine):
    assert engine["comparison_winner"]("global_score", 78, 72, ("A", "B"), (1,1)) == "🏆 A"


@pytest.mark.parametrize("mode, maximum", [("Investisseur", 86.03333333333333),
                                          ("Trader / Swing", 84.1)])
def test_theoretical_bound_matches_extreme_existing_rules(engine, mode, maximum):
    # Toutes les branches maximales peuvent être activées sur ces entrées.
    data = history(3)
    data["Close"] = [98., 99., 100.]
    data["MM50"], data["MM200"] = 90., 90.
    data["RSI"], data["MACD"], data["Signal"] = 60., 2., 1.
    info = dict(profitMargins=.5, returnOnEquity=.5, freeCashflow=1,
                trailingPE=8, priceToBook=.8, revenueGrowth=.5,
                earningsGrowth=.5, debtToEquity=10, beta=.5)
    result = engine["evaluate_scoring"](data, info, mode)
    assert result["raw_score"] == pytest.approx(maximum)
    assert result["theoretical_max"] == pytest.approx(maximum)
    assert result["score"] == pytest.approx(100)
