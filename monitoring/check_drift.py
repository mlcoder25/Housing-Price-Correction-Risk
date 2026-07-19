"""
Compares a 'reference' dataset (what the model was trained on) against a
'current' dataset (simulated incoming production traffic) and generates an
HTML drift report.

In production, `current` would be the most recent quarter(s) of real
Eurostat data once released. Here it's simulated by shifting the reference
distribution to mimic a macro shock (e.g. a rate-hike cycle cooling price
momentum across the board), so the report has something real to flag.

Usage:
    python monitoring/check_drift.py
Output:
    monitoring/drift_report.html
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data import load_data, FEATURE_NAMES  # noqa: E402

from evidently import Report
from evidently.presets import DataDriftPreset

REPORT_PATH = os.path.join(os.path.dirname(__file__), "drift_report.html")


def simulate_incoming_traffic(reference_df, shift: float = -1.5, random_state: int = 7):
    """Fakes a real-world shift: e.g. a rate-hike cycle cooling price
    momentum across most countries, shifting the two momentum features
    negative relative to what the model was trained on."""
    rng = np.random.RandomState(random_state)
    current = reference_df.copy()
    current["lag1_quarterly_change_pct"] = current["lag1_quarterly_change_pct"] + rng.normal(shift, 0.5, size=len(current))
    current["momentum_change"] = current["momentum_change"] + rng.normal(shift, 0.5, size=len(current))
    return current


def main():
    reference = load_data()[FEATURE_NAMES]
    current = simulate_incoming_traffic(reference)

    report = Report([DataDriftPreset()])
    result = report.run(reference_data=reference, current_data=current)
    result.save_html(REPORT_PATH)

    print(f"Drift report written to {REPORT_PATH}")
    # Exit non-zero if drift is detected on a majority of columns, so this
    # can gate a CI job or trigger a retraining alert.
    result_dict = result.dict()
    try:
        drift_share = result_dict["metrics"][0]["value"]["share"]
        n_drifted = result_dict["metrics"][0]["value"]["count"]
        print(f"Drifted columns: {n_drifted:.0f} ({drift_share:.0%} of all features)")
        if drift_share > 0.3:
            print("WARNING: significant drift detected — consider retraining.")
            sys.exit(1)
    except (KeyError, IndexError):
        print("Could not parse drift share from report; check drift_report.html manually.")


if __name__ == "__main__":
    main()
