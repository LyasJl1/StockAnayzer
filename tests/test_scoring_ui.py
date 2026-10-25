"""Rendu réel des vues Streamlit avec cours locaux et métadonnées absentes."""
import pytest
from streamlit.testing.v1 import AppTest

from scoring_helpers import APP_PATH


FIXTURES = '''
# Réponses locales : aucun accès Yahoo ou Gemini dans ces tests.
closes = pd.Series([100 + i*.05 + (i%7)*.1 for i in range(252)],
                   index=pd.date_range("2024-01-01", periods=252))
history_fixture = pd.DataFrame({"Open": closes-.1, "Close": closes, "High": closes+2, "Low": closes-2, "Volume": 1000})
def load_stock_data(*args):
    return history_fixture, {}
def get_key_events(*args):
    return {}
def load_screener_history(*args):
    return history_fixture
def extract_ticker_history(*args):
    return history_fixture
def load_screener_info(*args):
    return {}
st.session_state["watchlist"] = []
'''


def run_view(script):
    # Garde les fonctions réelles de l'app, remplace seulement les frontières IO
    # puis appelle explicitement la vue à tester sans la navigation par défaut.
    source = APP_PATH.read_text(encoding="utf-8").split("\napply_styles()\n", 1)[0]
    app = AppTest.from_string(source + FIXTURES + script, default_timeout=20).run()
    assert not app.exception, [entry.message for entry in app.exception]
    return app


@pytest.mark.parametrize("count", [1, 252])
def test_analysis_renders_absent_categories_and_insufficient_or_limited_score(count):
    app = run_view(f'\nhistory_fixture = history_fixture.iloc[:{count}]\nrender_dashboard("XYZ", "1y", "Investisseur")\n')
    html = "\n".join(element.value for element in app.markdown)
    assert "INDICE GLOBAL" in html
    assert 'width:0%;background:#94a3b8' in html
    assert "<b>N/D</b>" in html
    captions = "\n".join(element.value for element in app.caption)
    assert "Couverture des données" in captions
    assert "Normalisation : score brut" in captions
    if count == 1:
        assert any("non déterminable" in element.value for element in app.info)
        assert "Lecture combinée non déterminable" in html
    else:
        assert any("couverture limitée" in element.value for element in app.warning)


def test_comparison_renders_nullable_categories_and_coverage_reserve():
    app = run_view('''
left = build_comparison_snapshot("A", "1y", "Investisseur")
right = build_comparison_snapshot("B", "1y", "Investisseur")
left["global_score"], right["global_score"] = 75, 72
left["data_coverage"]["ratio"], right["data_coverage"]["ratio"] = .45, 1
render_comparison(left, right)
''')
    assert any("Comparaison réservée" in item.value for item in app.subheader)
    table = app.dataframe[0].value
    global_row = table.loc[table["Critère"] == "Indice global"].iloc[0]
    assert "réservée" in global_row["Avantage"]
    valuation_row = table.loc[table["Critère"] == "Score Valorisation"].iloc[0]
    assert valuation_row["Action A"] == "N/D"


def test_watchlist_exposes_coverage_next_to_nullable_score():
    app = run_view('''
st.session_state["watchlist"] = ["XYZ"]
history_fixture = history_fixture.iloc[:1]
render_watchlist("Investisseur")
''')
    table = app.dataframe[0].value
    assert table.iloc[0]["Indice"] == "N/D"
    assert "Couverture" in table.columns


def test_screener_coverage_filter_reacts_without_refetching():
    app = run_view('''
sample = build_screener_row("XYZ", history_fixture, {}, "Investisseur")
sample["data_coverage"]["ratio"] = .6
sample["revenue_growth"] = .1
sample["pe"] = 20
sample["global_score"] = 80
st.session_state["screener_run"] = {"examined": 1, "rows": [sample], "mode": "Investisseur"}
render_screener("Investisseur")
''')
    assert app.selectbox(key="screener_minimum_coverage").value == 70
    assert len(app.dataframe) == 0
    app.selectbox(key="screener_minimum_coverage").set_value(50).run()
    assert not app.exception
    assert app.dataframe[0].value.iloc[0]["Ticker"] == "XYZ"
