"""Inference for the locally stored, frozen Model A XGBoost bundle."""
from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


DEFAULT_BUNDLE = Path("D:/0mimic/TBI/04_Models/TBI_A_MI/XGBoost/model_bundle.joblib")
FEATURES = ("age", "gender", "gcs_motor_first", "sbp_mean", "spo2_mean", "hr_mean",
            "hemoglobin_g_dl", "creatinine_mg_dl")
BOUNDS = {
    "gcs_motor_first": (1, 6, "GCS motor score"),
    "sbp_mean": (30, 300, "Mean systolic blood pressure (mmHg)"),
    "spo2_mean": (50, 100, "Mean oxygen saturation (%)"),
    "hr_mean": (10, 280, "Mean heart rate (beats/min)"),
    "hemoglobin_g_dl": (0.1, 30, "Hemoglobin (g/dL)"),
    "creatinine_mg_dl": (0.01, 150, "Creatinine (mg/dL)"),
}


def numeric(value, label):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{label} must be a valid number.")
    try:
        number = float(value)
    except OverflowError as error:
        raise ValueError(f"{label} must be a valid number.") from error
    if not math.isfinite(number):
        raise ValueError(f"{label} must be a finite number.")
    return number


def validate_complete(values):
    if not isinstance(values, Mapping) or set(values) != set(FEATURES):
        raise ValueError("Enter all eight predictors.")
    age = numeric(values["age"], "Age")
    if age < 18 or not age.is_integer():
        raise ValueError("Enter age in whole years (18 or older).")
    sex = values["gender"]
    if not isinstance(sex, str) or sex not in {"F", "M"}:
        raise ValueError("Select a sex value (F or M).")
    checked = {"age": min(age, 90.0), "gender": sex}
    for key, (low, high, label) in BOUNDS.items():
        number = numeric(values[key], label)
        if not low <= number <= high:
            raise ValueError(f"{label} must be between {low:g} and {high:g}. Check the value and unit.")
        if key == "gcs_motor_first" and not number.is_integer():
            raise ValueError("GCS motor score must be an integer from 1 to 6.")
        checked[key] = number
    return checked


def prepare_features(frame):
    """The same deterministic column mapping as TBI_A_Model/tbi_a_models.py."""
    result = frame.loc[:, list(FEATURES)].copy()
    for name in FEATURES:
        if name == "gender":
            value = result[name].astype("string").str.strip().str.upper().replace({"FEMALE": "F", "MALE": "M"})
            if not value.isin(["F", "M"]).all():
                raise ValueError("Sex must be complete and coded as F or M.")
            result[name] = value.astype(object)
        else:
            value = pd.to_numeric(result[name], errors="raise").astype(float)
            if not np.isfinite(value).all():
                raise ValueError(f"{name} has missing or non-finite values.")
            result[name] = value
    return result


@dataclass(frozen=True)
class ModelA:
    pipelines: tuple
    imputation_ids: tuple
    model_id: str

    @classmethod
    def load(cls, bundle_path):
        path = Path(bundle_path).resolve()
        # Joblib is a deserializer for the operator's own frozen artifact. There is no model upload route.
        digest = sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
            handle.seek(0)
            bundle = joblib.load(handle)

        expected = {"format_version": "TBI_A_1", "algorithm": "XGBoost", "outcome": "death_in_hosp"}
        if not isinstance(bundle, dict) or any(bundle.get(k) != v for k, v in expected.items()):
            raise ValueError("The bundle must use the XGBoost Model A format for in-hospital mortality.")
        if bundle.get("synthetic_demo") is True:
            raise ValueError("Load the frozen study Model A bundle.")
        if tuple(bundle.get("features", ())) != FEATURES:
            raise ValueError("The bundle predictor names or order differ from those expected by the app.")
        ids = tuple(bundle.get("imputation_ids", ()))
        pipelines = tuple(bundle.get("models", ()))
        if (not ids or len(ids) != len(pipelines) or any(type(i) is not int for i in ids)
                or ids != tuple(range(1, len(ids) + 1))):
            raise ValueError("The bundle imputation IDs and model count are inconsistent.")

        marker_path = path.with_name("COMPLETE.json")
        if marker_path.is_file():
            marker = json.loads(marker_path.read_text(encoding="utf-8-sig"))
            keys = ("algorithm", "format_version", "features", "input_fingerprint")
            if not isinstance(marker, dict) or any(k not in marker or marker[k] != bundle.get(k) for k in keys):
                raise ValueError("The bundle metadata differs from the adjacent COMPLETE.json file.")

        for model in pipelines:
            steps = getattr(model, "named_steps", {})
            estimator = steps.get("model")
            if ("preprocess" not in steps or type(estimator).__name__ != "XGBClassifier"
                    or not type(estimator).__module__.startswith("xgboost.")
                    or not callable(getattr(model, "predict_proba", None))
                    or tuple(getattr(model, "classes_", ())) != (0, 1)):
                raise ValueError("A member lacks the fitted preprocessor or binary XGBoost classifier.")
        return cls(pipelines, ids, f"Model A · {digest.hexdigest()[:12]}")

    def member_probability(self, model_index, frame):
        return self._member_prepared(model_index, prepare_features(frame))

    def _member_prepared(self, model_index, frame):
        result = np.asarray(self.pipelines[model_index].predict_proba(frame))
        if result.shape != (len(frame), 2) or not np.isfinite(result).all() or not ((0 <= result) & (result <= 1)).all():
            raise ValueError("A model member returned invalid probabilities.")
        return result[:, 1]

    def predict_complete(self, values):
        frame = prepare_features(pd.DataFrame([validate_complete(values)], columns=list(FEATURES)))
        probabilities = [self._member_prepared(j, frame) for j in range(len(self.pipelines))]
        return float(np.vstack(probabilities).mean(axis=0)[0])
