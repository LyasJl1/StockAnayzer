"""Navigation, exécution et filtres Research Lab via le vrai moteur Streamlit."""
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from test_research_lab import prices
from scoring_helpers import APP_PATH
import research_lab as research


def test_navigation_keeps_timing_lab_and_research_lab():
    app = AppTest.from_file(str(APP_PATH),default_timeout=20)
    app.session_state["navigation_mode"]="🔬 Research Lab"
    app.run()
    assert not app.exception
    assert "🧬 Timing Lab" in app.radio[0].options
    assert "🔬 Research Lab" in app.radio[0].options
    assert app.title[0].value=="🔬 Research Lab"
    assert app.selectbox(key="research_years").value==10


def local_history_fixture():
    # Sept années incluant le warm-up et toute la fenêtre 5 ans demandée.
    end = pd.Timestamp(research.date_today())
    count = 1850
    dates = pd.bdate_range(end=end, periods=count)
    histories={}
    for seed,ticker in enumerate(research.RESEARCH_DEFAULT_TICKERS):
        history=prices(count,seed)
        history.index=dates
        histories[ticker]=history
    return histories


def test_research_lab_run_default_universe_and_filters(monkeypatch):
    histories=local_history_fixture()
    monkeypatch.setattr(research,"load_research_histories",lambda *_:(histories,[]))
    app=AppTest.from_file(str(APP_PATH),default_timeout=30)
    app.session_state["navigation_mode"]="🔬 Research Lab"
    app.run()
    app.selectbox(key="research_years").set_value(5).run()
    app.button(key="research_run").click().run()
    assert not app.exception, [x.message for x in app.exception]
    assert not app.error, [x.value for x in app.error]
    assert any(x.label=="Tickers couverts" and x.value=="8" for x in app.metric)
    assert len(app.dataframe)>0
    assert any(x.label=="Horizon (séances)" and x.value==20 for x in app.selectbox)
    assert app.dataframe[0].value["Facteur"].str.startswith("Combo").eq(False).all()
    app.selectbox(key="research_factor_filter").set_value("RSI").run()
    assert app.dataframe[0].value["Facteur"].eq("RSI").all()
    app.selectbox(key="research_regime").set_value("Bull").run()
    assert not app.exception
    app.selectbox(key="research_ticker").set_value("AAPL").run()
    assert not app.exception
    assert app.dataframe[0].value["Tickers couverts"].eq(1).all()
    app.checkbox(key="research_correlations").check().run()
    app.checkbox(key="research_regime_detail").check().run()
    assert not app.exception
    app.selectbox(key="research_years").set_value(3).run()
    assert any("Configuration modifiée" in x.value for x in app.warning)


def test_research_lab_missing_spy_reports_error_without_stale_results(monkeypatch):
    monkeypatch.setattr(research,"load_research_histories",lambda *_:({"AAPL":prices()},[]))
    app=AppTest.from_file(str(APP_PATH),default_timeout=20)
    app.session_state["navigation_mode"]="🔬 Research Lab"
    app.run()
    app.button(key="research_run").click().run()
    assert not app.exception
    assert any("SPY indisponible" in x.value for x in app.error)


def test_research_lab_rejects_more_than_twenty_tickers():
    app=AppTest.from_file(str(APP_PATH),default_timeout=20)
    app.session_state["navigation_mode"]="🔬 Research Lab"
    app.run()
    app.text_area(key="research_tickers").set_value(" ".join(f"T{i}" for i in range(21))).run()
    app.button(key="research_run").click().run()
    assert not app.exception
    assert any("1 à 20" in x.value for x in app.error)
