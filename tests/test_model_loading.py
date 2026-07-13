"""Smoke tests: verify models load and data pipeline runs."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import pytest

import app
from src.features.build_features import apply_pre_race_baseline
from src.sim.support import assess_strategy_support, build_support_profile

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

    def test_pre_race_baseline_uses_prior_seasons_only(self):
        target = pd.DataFrame({
            "Season": [2024], "EventName": ["Example GP"], "LapTimeSeconds": [91.0],
        })
        history = pd.DataFrame({
            "Season": [2022, 2023, 2024],
            "EventName": ["Example GP"] * 3,
            "LapTimeSeconds": [95.0, 93.0, 70.0],
            "IsPitLap": [False] * 3,
            "IsSafetyCar": [False] * 3,
        })
        result = apply_pre_race_baseline(target, history, 2024)
        assert result.loc[0, "PreRaceBaseline"] == 93.0
        assert result.loc[0, "BaselineSourceSeason"] == 2023
        assert result.loc[0, "LapTimeDelta"] == -2.0

    def test_app_only_exposes_races_with_pre_race_baselines(self):
        df = app.load_features(app.APP_MODEL_SEASON)
        assert df["PreRaceBaseline"].notna().all()
        assert (df["RoundNumber"] > 0).all()


class TestMetricsData:
    """Metric files must exist and be parseable."""

    def test_build_race_template_uses_round_lap_count_for_short_driver_data(self):
        df = pd.read_parquet(FEATURES_DIR / "features_2024.parquet")
        round_df = df[df["RoundNumber"] == 3].copy()
        template = app.build_race_template(round_df, selected_driver="VER", total_laps=58)

        assert len(template) == 58
        assert template["LapNumber"].tolist() == list(range(1, 59))
        assert template["Driver"].eq("VER").all()
        assert template["RoundNumber"].nunique() == 1

    @pytest.mark.parametrize("name", ["hgb", "ridge"])
    def test_metrics_json(self, name):
        path = METRICS_DIR / "metrics.json"
        assert path.exists()
        with open(path) as f:
            metrics = json.load(f)
        assert name in metrics
        assert "mae" in metrics[name]
        assert "rmse" in metrics[name]

        with open(METRICS_DIR / f"metrics_{name}.json") as f:
            model_metrics = json.load(f)["overall"]
        assert metrics[name] == model_metrics

        predictions = pd.read_parquet(METRICS_DIR / f"predictions_{name}.parquet")
        model = joblib.load(MODELS_DIR / f"{name}_model.joblib")
        target = "LapTimeDelta" if "LapTimeDelta" in predictions.columns else "LapTimeSeconds"
        X = predictions.drop(columns=[target, "LapTimeSeconds"], errors="ignore")
        predicted = model.predict(X)
        if target == "LapTimeDelta":
            predicted = predicted + predictions["PreRaceBaseline"].to_numpy()
        mae = np.mean(np.abs(predictions["LapTimeSeconds"].to_numpy() - predicted))
        rmse = np.sqrt(np.mean((predictions["LapTimeSeconds"].to_numpy() - predicted) ** 2))
        assert metrics[name]["mae"] == pytest.approx(mae)
        assert metrics[name]["rmse"] == pytest.approx(rmse)

    def test_season_metrics_match_default_metrics(self):
        with open(METRICS_DIR / "metrics.json") as f:
            default_metrics = json.load(f)
        with open(METRICS_DIR / "metrics_2024.json") as f:
            season_metrics = json.load(f)
        assert season_metrics == default_metrics

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
        assert np.allclose(df["residual"], df["LapTimeSeconds"] - df["pred"])

    def test_app_rejects_unsupported_model_seasons(self):
        assert app.load_model("hgb", app.APP_MODEL_SEASON) is not None
        assert app.load_model("hgb", app.APP_MODEL_SEASON - 1) is None
        assert app.load_model_metrics(app.APP_MODEL_SEASON - 1) is None
        assert app.load_residuals_cached("hgb", app.APP_MODEL_SEASON - 1) is None

    def test_support_profile_penalizes_unobserved_tyre_life(self):
        history = pd.DataFrame({
            "Season": [2023] * 20,
            "EventName": ["Example GP"] * 20,
            "Compound": ["SOFT"] * 20,
            "TyreLife": list(range(1, 21)),
            "IsPitLap": [False] * 20,
            "IsSafetyCar": [False] * 20,
        })
        profile = build_support_profile(history, 2024, quantile=1.0)
        supported = assess_strategy_support([("SOFT", 18)], "Example GP", profile)
        unsupported = assess_strategy_support([("SOFT", 25)], "Example GP", profile)
        assert supported["unsupported_laps"] == 0
        assert unsupported["unsupported_laps"] == 5


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
