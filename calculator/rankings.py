"""Read the same cohort-level result manifest used by the project's GitHub Pages site."""

import json
import math
from pathlib import Path

from risk_model import FEATURES


DEFAULT_RANKINGS = Path(__file__).resolve().with_name("published_rankings.json")
COHORT_LABELS = {
    "training": "MIMIC-IV training",
    "internal_test": "MIMIC-IV internal test",
    "external_eicu": "eICU external validation",
}
FEATURE_LABELS = {
    "age": "Age",
    "gender": "Sex",
    "gcs_motor_first": "First GCS motor score",
    "sbp_mean": "Mean systolic blood pressure",
    "spo2_mean": "Mean SpO₂",
    "hr_mean": "Mean heart rate",
    "hemoglobin_g_dl": "Hemoglobin",
    "creatinine_mg_dl": "Creatinine",
}


def _positive_integer(number):
    return type(number) is int and number > 0


def load_rankings(path):
    data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict) or data.get("published_from") != "model_a_shap_integrated":
        raise ValueError("Use the results.json file from the Model A SHAP results site.")
    rows = data.get("cohorts")
    if not isinstance(rows, list) or len(rows) != len(COHORT_LABELS) or any(not isinstance(row, dict) for row in rows):
        raise ValueError("The rankings file must contain all three cohorts.")
    known = [row.get("id") for row in rows]
    if any(not isinstance(key, str) for key in known) or set(known) != set(COHORT_LABELS):
        raise ValueError("The rankings file must identify the training, internal test and external validation cohorts.")

    by_id = dict(zip(known, rows))
    cohorts = []
    for key, label in COHORT_LABELS.items():
        row = by_id[key]
        n, count = row.get("n"), row.get("n_imputations")
        total = row.get("n_cohort", n)
        if not all(map(_positive_integer, (n, count, total))) or total < n:
            raise ValueError(f"{label}: invalid study or imputation count.")
        raw = row.get("importance")
        if not isinstance(raw, list) or len(raw) != len(FEATURES):
            raise ValueError(f"{label}: the ranking must include all eight predictors.")
        items = []
        for item in raw:
            if not isinstance(item, dict):
                raise ValueError(f"{label}: malformed ranking entry.")
            feature, rank, value = (item.get(field) for field in ("feature", "rank", "mean_abs_shap_pp"))
            if not isinstance(feature, str) or feature not in FEATURES or not _positive_integer(rank):
                raise ValueError(f"{label}: invalid predictor name or rank.")
            if (type(value) not in {int, float} or not 0 <= value <= 100 or not math.isfinite(value)):
                raise ValueError(f"{label}: SHAP importance must be finite and between 0 and 100 percentage points.")
            items.append({"feature": feature, "label": FEATURE_LABELS[feature], "rank": rank,
                          "mean_abs_shap_pp": value})
        items.sort(key=lambda item: item["rank"])
        if ([item["rank"] for item in items] != list(range(1, len(FEATURES) + 1))
                or {item["feature"] for item in items} != set(FEATURES)
                or any(a["mean_abs_shap_pp"] < b["mean_abs_shap_pp"] for a, b in zip(items, items[1:]))):
            raise ValueError(f"{label}: duplicate or misordered SHAP ranks.")
        cohorts.append({"id": key, "label": label, "n": n, "n_cohort": total, "n_imputations": count,
                        "importance": items})
    return {"cohorts": cohorts, "published_from": "model_a_shap_integrated"}
