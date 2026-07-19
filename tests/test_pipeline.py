"""
Two kinds of tests that make this a CI/CD-worthy pipeline instead of just
a training script:

1. Data validation -- catches schema drift/breakage in the data source
   before it ever reaches the model.
2. Model performance gate -- fails the build if the trained model's
   metrics fall below an agreed threshold, so a bad model can never get
   registered/deployed silently.
"""
import json
import os

import joblib
import pandas as pd
import pytest

from src.data import load_data, FEATURE_NAMES

METRICS_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "metrics.json")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "model.pkl")

# Agreed minimum bar for a model to be considered deployable. Set well
# below what the current model achieves (~0.99 ROC-AUC / ~0.80 F1) to
# leave room for normal variance across retrains, not to rubber-stamp
# a specific run.
ROC_AUC_THRESHOLD = 0.85
F1_THRESHOLD = 0.65

# With ~297 rows across 30 countries, the positive class ("YoY price
# decline") should land somewhere in this range. If a future data refresh
# pushes it outside this band, that's worth a human look before retraining.
EXPECTED_POSITIVE_RATE_RANGE = (0.05, 0.40)


class TestDataValidation:
    def test_expected_columns_present(self):
        df = load_data()
        for col in FEATURE_NAMES + ["target"]:
            assert col in df.columns, f"missing expected column: {col}"

    def test_no_nulls_in_model_columns(self):
        df = load_data()
        assert df[FEATURE_NAMES + ["target"]].isnull().sum().sum() == 0, \
            "unexpected nulls in feature/target columns"

    def test_target_is_binary(self):
        df = load_data()
        assert set(df["target"].unique()) <= {0, 1}

    def test_features_are_numeric(self):
        df = load_data()
        for col in FEATURE_NAMES:
            assert pd.api.types.is_numeric_dtype(df[col]), f"{col} is not numeric"

    def test_no_duplicate_country_quarter_rows(self):
        df = load_data()
        dupes = df.duplicated(subset=["country", "quarter"]).sum()
        assert dupes == 0, f"found {dupes} duplicate (country, quarter) rows"

    def test_reasonable_row_count(self):
        # Catches a broken CSV/parse silently dropping most of the data.
        df = load_data()
        assert len(df) >= 250, f"expected >=250 rows after lag features, got {len(df)}"

    def test_class_balance_in_expected_range(self):
        df = load_data()
        positive_rate = df["target"].mean()
        lo, hi = EXPECTED_POSITIVE_RATE_RANGE
        assert lo <= positive_rate <= hi, (
            f"positive rate {positive_rate:.2%} is outside the expected "
            f"{lo:.0%}-{hi:.0%} range — check for a data or pipeline change"
        )


class TestModelPerformanceGate:
    @classmethod
    @pytest.fixture(scope="class")
    def metrics(cls):
        if not os.path.exists(METRICS_PATH):
            pytest.skip("model/metrics.json not found — run `python src/train.py` first")
        with open(METRICS_PATH) as f:
            return json.load(f)

    def test_roc_auc_above_threshold(self, metrics):
        roc_auc = metrics["champion"]["roc_auc"]
        assert roc_auc >= ROC_AUC_THRESHOLD, (
            f"roc_auc {roc_auc:.4f} is below the deployment threshold of {ROC_AUC_THRESHOLD}"
        )

    def test_f1_above_threshold(self, metrics):
        f1 = metrics["champion"]["f1"]
        assert f1 >= F1_THRESHOLD, (
            f"f1 {f1:.4f} is below the deployment threshold of {F1_THRESHOLD}"
        )

    def test_baseline_comparison_is_recorded(self, metrics):
        # This does NOT assert the model beats the baseline -- on this
        # small, highly autocorrelated dataset it sometimes won't, and
        # that's a real finding, not a bug. It only asserts the honest
        # comparison was actually computed and saved, so it can never be
        # quietly dropped from the pipeline.
        assert "baseline_persistence" in metrics
        assert "f1" in metrics["baseline_persistence"]


class TestModelArtifact:
    def test_model_file_exists(self):
        if not os.path.exists(MODEL_PATH):
            pytest.skip("model/model.pkl not found — run `python src/train.py` first")
        model = joblib.load(MODEL_PATH)
        assert hasattr(model, "predict"), "loaded artifact is not a fitted model"

    def test_model_predicts_on_sample_rows(self):
        if not os.path.exists(MODEL_PATH):
            pytest.skip("model/model.pkl not found — run `python src/train.py` first")
        model = joblib.load(MODEL_PATH)
        df = load_data().head(5)
        preds = model.predict(df[FEATURE_NAMES])
        assert len(preds) == 5
