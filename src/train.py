"""
Trains a housing price-correction risk classifier on real European house
price index data (Eurostat-style, 30 countries, 2022-2025). Uses the same
techniques as the original prototype (SMOTE + GridSearchCV), but adds the
piece that was missing: every run's params/metrics are logged to MLflow,
and the best model is registered so the API and CI/CD pipeline always
serve a known, versioned artifact instead of a loose .pkl file.

Usage:
    python src/train.py
"""
import json
import os
import sys

import mlflow
import mlflow.sklearn
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data import load_data, train_test_split_df  # noqa: E402

MODEL_NAME = "housing_price_correction_risk_classifier"
METRICS_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "metrics.json")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "model.pkl")

SEARCH_SPACE = {
    "random_forest": {
        "estimator": RandomForestClassifier(random_state=42),
        "params": {
            "n_estimators": [100, 200],
            "max_depth": [4, 8, None],
        },
    },
    "logistic_regression": {
        "estimator": LogisticRegression(max_iter=1000),
        "params": {
            "C": [0.1, 1.0, 10.0],
        },
    },
}


def evaluate(model, X_test, y_test) -> dict:
    proba = model.predict_proba(X_test)[:, 1]
    preds = model.predict(X_test)
    return {
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "f1": float(f1_score(y_test, preds)),
        "precision": float(precision_score(y_test, preds)),
        "recall": float(recall_score(y_test, preds)),
    }


def evaluate_baseline(X_test, y_test) -> dict:
    """
    Naive persistence baseline: 'this quarter will look like last quarter'
    (predict decline if lag1_yearly_change_pct was already negative).

    yearly_change_pct is a rolling 4-quarter measure, so it's strongly
    autocorrelated with itself one quarter later almost by construction.
    Any model on this task should be checked against this baseline before
    its ROC-AUC/F1 is treated as evidence the model learned something —
    a high score that merely matches the baseline isn't a modeling win.
    """
    baseline_pred = (X_test["lag1_yearly_change_pct"] < 0).astype(int)
    return {
        "f1": float(f1_score(y_test, baseline_pred)),
        "accuracy": float((baseline_pred.values == y_test.values).mean()),
    }


def main():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("housing-price-correction-risk")

    df = load_data()
    X_train, X_test, y_train, y_test = train_test_split_df(df)

    baseline_metrics = evaluate_baseline(X_test, y_test)
    print(f"[baseline: naive persistence] metrics={baseline_metrics}")

    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    best_run = {"roc_auc": -1, "run_id": None, "model_key": None}

    with mlflow.start_run(run_name="baseline_persistence"):
        mlflow.log_param("model_type", "naive_persistence_baseline")
        mlflow.log_metrics(baseline_metrics)

    for model_key, cfg in SEARCH_SPACE.items():
        with mlflow.start_run(run_name=model_key) as run:
            grid = GridSearchCV(cfg["estimator"], cfg["params"], scoring="roc_auc", cv=5, n_jobs=-1)
            grid.fit(X_train_res, y_train_res)

            metrics = evaluate(grid.best_estimator_, X_test, y_test)

            mlflow.log_param("model_type", model_key)
            mlflow.log_params(grid.best_params_)
            mlflow.log_metrics(metrics)
            mlflow.log_metric("f1_vs_baseline", metrics["f1"] - baseline_metrics["f1"])
            mlflow.sklearn.log_model(grid.best_estimator_, "model")

            print(f"[{model_key}] best_params={grid.best_params_} metrics={metrics}")

            if metrics["roc_auc"] > best_run["roc_auc"]:
                best_run = {"roc_auc": metrics["roc_auc"], "run_id": run.info.run_id,
                            "model_key": model_key, "metrics": metrics}

    # Register the best run's model as a new version in the MLflow Model Registry
    model_uri = f"runs:/{best_run['run_id']}/model"
    registered = mlflow.register_model(model_uri, MODEL_NAME)
    print(f"Registered {MODEL_NAME} version {registered.version} "
          f"(model_type={best_run['model_key']}, roc_auc={best_run['roc_auc']:.4f})")

    # Also drop a plain artifact + metrics file so the API/tests don't need
    # an MLflow server running just to load the champion model.
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    best_model = mlflow.sklearn.load_model(model_uri)
    import joblib
    joblib.dump(best_model, MODEL_PATH)
    output_metrics = {
        "champion": best_run["metrics"],
        "champion_model_type": best_run["model_key"],
        "baseline_persistence": baseline_metrics,
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(output_metrics, f, indent=2)

    print(f"Saved champion model -> {MODEL_PATH}")
    print(f"Saved metrics -> {METRICS_PATH}")

    if best_run["metrics"]["f1"] <= baseline_metrics["f1"]:
        print(
            "\nNOTE: the trained model does not clearly beat the naive "
            "persistence baseline on F1. With ~300 rows and a strongly "
            "autocorrelated target, this is a real and common outcome — "
            "it means the ML model isn't yet adding value over 'assume "
            "this quarter looks like last quarter', and more history "
            "(or a harder target, e.g. predicting regime CHANGES) is "
            "needed before trusting the model over the simple heuristic."
        )


if __name__ == "__main__":
    main()
