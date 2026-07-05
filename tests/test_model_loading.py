"""Smoke tests: verify models load and data pipeline runs."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import pytest

DATA_DIR = Path("data")
FEATURES_DIR = DATA_DIR / "features"
MODELS_DIR = DATA_DIR / "models"
METRICS_DIR = DATA_DIR / "metrics"


class TestModelLoading:
    """Each model file must load without pickle errors."""

    @pytest.mark.parametrize("name", ["hgb", "ridge"])
    def test_model_loads(self, name):
        path = MODELS_DIR / f"{name}_model.joblib"
        assert path.exists(), f"{path} not found"
        model = joblib.load(path)
        assert model is not None
        assert hasattr(model, "predict")

    @pytest.mark.parametrize("name", ["hgb", "ridge"])
    def test_model_meta_exists(self, name):
        path = MODELS_DIR / f"model_meta_{name}.json"
        assert path.exists(), f"{path} not found"
        with open(path) as f:
            meta = json.load(f)
        assert "sklearn_version" in meta
        assert meta["sklearn_version"] == sklearn.__version__, (
            f"Model trained with sklearn {meta['sklearn_version']}, "
            f"installed is {sklearn.__version__}"
        )

    def test_combined_meta(self):
        path = MODELS_DIR / "model_meta.json"
        assert path.exists()
        with open(path) as f:
            meta = json.load(f)
        assert "sklearn_version" in meta
        assert meta["sklearn_version"] == sklearn.__version__


class TestFeatureData:
    """Feature parquet files must be readable."""

    @pytest.mark.parametrize("season", [2024])
    def test_features_load(self, season):
        path = FEATURES_DIR / f"features_{season}.parquet"
        assert path.exists(), f"{path} not found"
        df = pd.read_parquet(path)
        assert len(df) > 0
        for col in ["LapTimeSeconds", "Compound", "Driver", "RoundNumber"]:
            assert col in df.columns, f"Missing column: {col}"

    def test_features_rounds(self):
        df = pd.read_parquet(FEATURES_DIR / "features_2024.parquet")
        rounds = sorted(df["RoundNumber"].dropna().unique().astype(int))
        assert len(rounds) >= 20, f"Expected 20+ rounds, got {len(rounds)}"


class TestMetricsData:
    """Metric files must exist and be parseable."""

    @pytest.mark.parametrize("name", ["hgb", "ridge"])
    def test_metrics_json(self, name):
        path = METRICS_DIR / "metrics.json"
        assert path.exists()
        with open(path) as f:
            metrics = json.load(f)
        assert name in metrics
        assert "mae" in metrics[name]
        assert "rmse" in metrics[name]

    @pytest.mark.parametrize("name", ["hgb", "ridge"])
    def test_rolling_metrics(self, name):
        path = METRICS_DIR / f"rolling_metrics_{name}.json"
        assert path.exists()
        with open(path) as f:
            data = json.load(f)
        assert isinstance(data, list)
        assert len(data) > 0
        for entry in data:
            assert "mae" in entry
            assert "rmse" in entry

    @pytest.mark.parametrize("name", ["hgb", "ridge"])
    def test_predictions_exist(self, name):
        path = METRICS_DIR / f"predictions_{name}.parquet"
        assert path.exists()
        df = pd.read_parquet(path)
        assert "residual" in df.columns
        assert len(df) > 0


class TestModelPredict:
    """Models must produce predictions on real feature data."""

    @pytest.mark.parametrize("name", ["hgb", "ridge"])
    def test_predict_shape(self, name):
        model = joblib.load(MODELS_DIR / f"{name}_model.joblib")
        df = pd.read_parquet(FEATURES_DIR / "features_2024.parquet")
        race = df[(df["RoundNumber"] == 24) & (df["Driver"] == "VER")].copy()
        race = race.sort_values("LapNumber").reset_index(drop=True)
        X = race.drop(columns=["LapTimeSeconds", "LapTimeDelta"], errors="ignore")
        preds = model.predict(X)
        assert len(preds) == len(race)
        assert np.all(np.isfinite(preds)), "Predictions contain NaN or inf"
