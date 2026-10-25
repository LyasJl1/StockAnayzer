"""Filtrage, tri et présentation de résultats incomplets, sans Yahoo Finance."""
import pytest

from scoring_helpers import complete_info, history, load_functions


@pytest.fixture
def engine():
    return load_functions()


def row(ticker="A", score=75, coverage=.8, pe=20, growth=.1, mm200=.05):
    return {"ticker": ticker, "global_score": score, "data_coverage": {"ratio": coverage},
            "pe": pe, "revenue_growth": growth, "mm200": mm200}


def select(engine, rows, **overrides):
    options = dict(minimum_score=60, maximum_pe=30, ignore_pe=False,
                   minimum_growth=0, require_mm200=False, limit=15)
    options.update(overrides)
    return engine["filter_screener_results"](rows, **options)


@pytest.mark.parametrize("score", [None, float("nan"), float("inf")])
def test_absent_score_is_excluded(engine, score):
    assert select(engine, [row(score=score)]) == []


@pytest.mark.parametrize("coverage", [None, float("nan"), .45, .69999])
def test_missing_or_low_coverage_is_excluded_by_default(engine, coverage):
    assert select(engine, [row(coverage=coverage)]) == []


def test_unknown_coverage_is_not_assumed_complete(engine):
    sample = row()
    del sample["data_coverage"]
    assert select(engine, [sample]) == []


@pytest.mark.parametrize("threshold", [.5, .6, .7, .8, .9])
def test_coverage_threshold_is_inclusive(engine, threshold):
    sample = row(coverage=threshold)
    assert select(engine, [sample], minimum_coverage=threshold) == [sample]


def test_lowering_coverage_filter_restores_rows(engine):
    sample = row(coverage=.54)
    assert select(engine, [sample]) == []
    assert select(engine, [sample], minimum_coverage=.5) == [sample]


@pytest.mark.parametrize("sample", [row(score=59), row(pe=None), row(pe=-1),
    row(pe=31), row(growth=None), row(growth=-.01)])
def test_existing_score_pe_growth_filters_remain_effective(engine, sample):
    assert select(engine, [sample]) == []


def test_pe_filter_can_be_disabled(engine):
    sample = row(pe=None)
    assert select(engine, [sample], ignore_pe=True) == [sample]


@pytest.mark.parametrize("mm200", [None, -.05, 0])
def test_mm200_filter_remains_effective(engine, mm200):
    assert select(engine, [row(mm200=mm200)], require_mm200=True) == []


def test_filter_precedes_sort_and_limit(engine):
    rows = [row("LOW_COVERAGE", 100, .5), row("B", 70), row("C", 85), row("D", 80)]
    result = select(engine, rows, limit=2)
    assert [sample["ticker"] for sample in result] == ["C", "D"]


def test_watchlist_keeps_limited_rows_with_coverage_and_reserve(engine):
    sample = engine["build_screener_row"]("XYZ", history(), {}, "Investisseur")
    table = engine["_format_screener_table"]([sample], include_verdict=True)
    assert table.iloc[0]["Couverture"] == "36%"
    assert "couverture limitée" in table.iloc[0]["Réserve"]
    assert "Confiance" not in table.columns
    assert "Score brut" in table.columns


def test_watchlist_displays_undeterminable_score(engine):
    sample = engine["build_screener_row"]("XYZ", history(1), {}, "Investisseur")
    assert sample["global_score"] is None
    table = engine["_format_screener_table"]([sample], include_verdict=True)
    assert table.iloc[0]["Indice"] == "N/D"
    assert "non déterminable" in table.iloc[0]["Verdict"]


def test_screener_invalid_history_returns_no_row(engine):
    assert engine["build_screener_row"]("XYZ", history(0), complete_info(), "Investisseur") is None
