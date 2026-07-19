"""
Data loading for the housing price-correction risk classifier.

Real dataset: Eurostat-style European House Price Index, quarterly,
2022-Q4 to 2025-Q3/Q4, across 30 individual countries (5 EU/Euro-area
aggregate rows are excluded -- they're weighted combinations of the
individual countries, not independent observations).

Task: predict whether a country will report a year-over-year house
price DECLINE (yearly_change_pct < 0) in a given quarter -- a "price
correction risk" signal.

Leakage note: the features are deliberately built only from what was
knowable at the END of the PRIOR quarter (lag 1 / lag 2 values) plus
static country attributes. None of the current quarter's own price
figures are used as inputs -- otherwise the model would trivially see
the answer inside its own features.
"""
import os

import pandas as pd

RAW_CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "european_housing_prices_clean.csv")

FEATURE_NAMES = [
    "lag1_quarterly_change_pct",   # last quarter's QoQ price change
    "lag1_yearly_change_pct",      # last quarter's YoY price change
    "lag2_quarterly_change_pct",   # QoQ change two quarters ago
    "momentum_change",             # lag1_qoq - lag2_qoq: is momentum accelerating or fading?
    "eu_member",                   # 1/0
    "eurozone_member",              # 1/0
    "quarter_num",                 # 1-4, captures seasonality
]


def load_data(csv_path: str = RAW_CSV_PATH) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Aggregate rows (Euro area / EU as a whole) are combinations of the
    # individual countries already in the data -- keeping both would mean
    # double-counting the same underlying signal.
    df = df[df["country_type"] == "Individual"].copy()
    df = df.sort_values(["country", "year", "quarter_num"]).reset_index(drop=True)

    g = df.groupby("country")
    df["lag1_quarterly_change_pct"] = g["quarterly_change_pct"].shift(1)
    df["lag1_yearly_change_pct"] = g["yearly_change_pct"].shift(1)
    df["lag2_quarterly_change_pct"] = g["quarterly_change_pct"].shift(2)
    df["momentum_change"] = df["lag1_quarterly_change_pct"] - df["lag2_quarterly_change_pct"]

    df["eu_member"] = (df["eu_member"] == "Yes").astype(int)
    df["eurozone_member"] = (df["eurozone_member"] == "Yes").astype(int)

    df["target"] = (df["yearly_change_pct"] < 0).astype(int)

    # First two quarters of each country's history have no lag2 available.
    df = df.dropna(subset=FEATURE_NAMES + ["target"]).reset_index(drop=True)

    return df[FEATURE_NAMES + ["target", "country", "year", "quarter"]]


def train_test_split_df(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    from sklearn.model_selection import train_test_split
    X = df[FEATURE_NAMES]
    y = df["target"]
    # NOTE: this is a stratified RANDOM split, not a walk-forward time split.
    # With only ~300 rows across 30 countries and 12 quarters, a strict
    # time-based split would leave too little data in each fold to train
    # or evaluate reliably. A production version of this model should
    # move to walk-forward validation (train on quarters 1..k, test on k+1)
    # once more history is available -- flagged in the README as a known
    # limitation, not hidden.
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)
