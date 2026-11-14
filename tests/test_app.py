"""Runs the Streamlit app headlessly with Streamlit's official test harness."""
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def test_app_runs_without_errors():
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception


def test_app_every_damage_type():
    at = AppTest.from_file(APP, default_timeout=120).run()
    for kind in at.selectbox[1].options:
        at.selectbox[1].select(kind).run()
        assert not at.exception, kind
