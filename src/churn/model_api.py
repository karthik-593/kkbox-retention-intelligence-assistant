"""Churn tool: get_churn_risk(msno) -> calibrated risk + top SHAP drivers (restricted to the 12
training features) + segment context (tenure_days, n_prior_cycles -- descriptive only, NEVER a
driver, per the model/monitoring feature split in the build plan).

Reuses the already-deployed churn-retention model artifacts (models/churn_model/, copied from the
companion project's MLflow registry export) and its SHAP setup (TreeExplainer on the raw xgb model,
same as that project's notebooks/03_explainability.ipynb) rather than retraining or re-deriving
anything here.

The demo "subscriber base" is data/raw/sample_subscribers.csv -- 420 real KKBox test-split, paid
subscribers with fully computed point-in-time features, sampled across the risk spectrum.
"""
import json

import joblib
import pandas as pd
import shap
from dataclasses import dataclass

from src import config

_xgb = None
_calibrator = None
_features = None
_explainer = None
_subscribers = None


def _load():
    global _xgb, _calibrator, _features, _explainer, _subscribers
    if _xgb is None:
        _xgb = joblib.load(config.MODEL_DIR / "xgb.joblib")
        _calibrator = joblib.load(config.MODEL_DIR / "calibrator.joblib")
        _features = json.load(open(config.MODEL_DIR / "features.json", encoding="utf-8"))
        _explainer = shap.TreeExplainer(_xgb)
        _subscribers = pd.read_csv(config.SUBSCRIBERS_PATH).set_index("msno")
    return _xgb, _calibrator, _features, _explainer, _subscribers


@dataclass
class Driver:
    feature: str
    shap_value: float
    feature_value: float


@dataclass
class ChurnRisk:
    msno: str
    risk: float               # calibrated P(churn)
    top_drivers: list[Driver]  # sorted by |shap value| desc, restricted to the 12 training features
    segment: dict              # tenure_days / n_prior_cycles -- context only, never a driver


def get_churn_risk(msno: str, top_n: int = 3) -> ChurnRisk:
    xgb, calibrator, features, explainer, subscribers = _load()
    if msno not in subscribers.index:
        raise KeyError(f"unknown subscriber msno={msno!r}")
    row = subscribers.loc[[msno]]

    raw_p = xgb.predict_proba(row[features])[:, 1]
    risk = float(calibrator.predict(raw_p)[0])

    shap_values = explainer.shap_values(row[features])[0]
    ranked = sorted(zip(features, shap_values, row[features].iloc[0]), key=lambda t: -abs(t[1]))
    top_drivers = [
        Driver(feature=f, shap_value=float(v), feature_value=float(x)) for f, v, x in ranked[:top_n]
    ]

    segment = {
        "tenure_days": int(row["tenure_days"].iloc[0]),
        "n_prior_cycles": int(row["n_prior_cycles"].iloc[0]),
    }
    return ChurnRisk(msno=msno, risk=risk, top_drivers=top_drivers, segment=segment)
