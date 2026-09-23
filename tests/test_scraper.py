import importlib.util
from pathlib import Path

import pandas as pd
import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "src" / "01_scrape_github_market.py"
spec = importlib.util.spec_from_file_location("scrape_module", MODULE_PATH)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_fetch_github_metrics_handles_timeout(monkeypatch):
    def fake_get(*args, **kwargs):
        raise module.requests.exceptions.ConnectTimeout("timed out")

    monkeypatch.setattr(module.requests, "get", fake_get)

    result = module.fetch_github_metrics("bitcoin", "bitcoin")

    assert result is None


def test_fetch_coingecko_data_handles_timeout(monkeypatch):
    def fake_get(*args, **kwargs):
        raise module.requests.exceptions.Timeout("timed out")

    monkeypatch.setattr(module.requests, "get", fake_get)

    result = module.fetch_coingecko_data(["bitcoin"])

    assert isinstance(result, pd.DataFrame)
    assert result.empty
