"""Smoke test della dashboard: l'app parte senza errori e mostra numeri reali.

Scarica dati veri (se non sono in cache), quindi gira solo con:
    pytest -m network
"""

from pathlib import Path

import pytest

APP = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"


@pytest.mark.network
def test_dashboard_runs_without_errors():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP), default_timeout=120).run()
    assert not at.exception, at.exception
    assert len(at.metric) == 4
    assert at.metric[0].label == "CAGR momentum"


@pytest.mark.network
def test_dashboard_warns_when_parameters_change():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP), default_timeout=120).run()
    assert not at.warning
    at.select_slider[0].set_value(126).run()
    assert not at.exception
    assert len(at.warning) == 1
