"""Compare local app inference with the original frozen validation predictions."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from risk_model import FEATURES, ModelA, validate_complete


ROOT = Path("D:/0mimic/TBI")
BUNDLE = Path("04_Models/TBI_A_MI/XGBoost/model_bundle.joblib")
MI = Path("04_Models/TBI_MI_Step1")
COHORTS = {"internal_test": "mimiciv_internal_test.csv", "external_eicu": "eicu_external_v2.csv"}
KEYS = ("stay_id", "subject_id", "hadm_id")
OUTCOME = "death_in_hosp"
TOLERANCE = 1e-6
MISSING = {"", "na", "n/a", "null", "nan", "none", "unknown", "<na>"}


def read_rows(path, identity, required):
    fields = (*identity, OUTCOME, *required)
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}")
    frame = pd.read_csv(path, encoding="utf-8-sig", dtype={key: "string" for key in identity},
                        float_precision="round_trip")
    missing = set(fields) - set(frame.columns)
    if frame.empty or missing:
        raise ValueError(f"{path.name}: no rows or missing fields {sorted(missing)}")
    frame = frame.loc[:, list(fields)].copy()
    for key in identity:
        key_values = frame[key].str.strip()
        if key_values.isna().any() or key_values.eq("").any():
            raise ValueError(f"{path.name}: missing {key}")
        if key in KEYS and key_values.duplicated().any():
            raise ValueError(f"{path.name}: duplicate {key}")
        frame[key] = key_values
    outcome = pd.to_numeric(frame[OUTCOME], errors="coerce")
    if not outcome.isin((0, 1)).all():
        raise ValueError(f"{path.name}: {OUTCOME} must be complete and coded as 0 or 1")
    frame[OUTCOME] = outcome.astype("int8")
    return frame.set_index("stay_id")


def align_rows(frame, reference, identity, label, *, allow_extra=False):
    missing = len(reference.index.difference(frame.index))
    extra = len(frame.index.difference(reference.index))
    if missing or (extra and not allow_extra):
        raise ValueError(f"{label}: validation roster mismatch ({missing} absent, {extra} additional)")
    aligned = frame.loc[reference.index]
    for key in (*identity[1:], OUTCOME):
        n_mismatch = int(aligned[key].ne(reference[key]).sum())
        if n_mismatch:
            raise ValueError(f"{label}: {n_mismatch} mismatched {key} values after alignment")
    return aligned


def probability(values, n, label):
    values = np.asarray(values)
    if values.shape != (n,) or not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError(f"{label}: expected {n} finite probabilities in [0, 1]")
    return values


def member_probabilities(model, model_index, frame, label):
    return probability(model.member_probability(model_index, frame.loc[:, list(FEATURES)]), len(frame), label)


def check_complete_endpoint(model):
    # Synthetic values exercise the same complete-input method used by the app.
    values = dict(age=60, gender="F", gcs_motor_first=5, sbp_mean=120, spo2_mean=96,
                  hr_mean=82, hemoglobin_g_dl=12, creatinine_mg_dl=1)
    row = pd.DataFrame([values])
    members = [member_probabilities(model, index, row, "complete-input member")
               for index in range(len(model.imputation_ids))]
    expected = float(np.vstack(members).mean(axis=0)[0])
    observed = float(model.predict_complete(values))
    err = abs(observed - expected)
    if not np.isfinite(observed) or not 0 <= observed <= 1 or err > TOLERANCE:
        raise ValueError("Complete-input app endpoint did not reproduce the mean member probability")
    return err


def summarize_raw(model, path, frozen, archived, identity, cohort):
    raw = align_rows(read_rows(path, identity, FEATURES), frozen, identity, f"{cohort} raw",
                     allow_extra=cohort == "internal_test")
    features = raw.loc[:, list(FEATURES)].copy()
    features["gender"] = (features.gender.astype("string").str.strip().str.upper()
                          .replace({"FEMALE": "F", "MALE": "M"}))
    missing = np.zeros(len(features), dtype=bool)
    blocked = np.zeros(len(features), dtype=bool)
    allowed_rows = []
    for index, row in enumerate(features.itertuples(index=False, name=None)):
        if any(pd.isna(value) or (isinstance(value, str) and value.strip().casefold() in MISSING) for value in row):
            missing[index] = True
            continue
        try:
            # CSV columns with an invalid field may still contain valid numeric text in other rows.
            values = {key: (value if key == "gender" else float(value)) for key, value in zip(FEATURES, row)}
            allowed_rows.append(validate_complete(values))
        except ValueError:
            blocked[index] = True
    complete = ~(missing | blocked)
    n_complete = len(allowed_rows)
    y = raw[OUTCOME].to_numpy()
    result = {"n_raw_complete": n_complete, "deaths_raw_complete": int(y[complete].sum()),
              "n_raw_missing": int(missing.sum()), "deaths_raw_missing": int(y[missing].sum()),
              "n_raw_blocked": int(blocked.sum()), "deaths_raw_blocked": int(y[blocked].sum()),
              "raw_complete_max_abs_error": None, "raw_complete_within_tolerance": None}
    if n_complete:
        observed = pd.DataFrame(allowed_rows, columns=FEATURES)
        probabilities = [member_probabilities(model, index, observed, f"{cohort} observed complete")
                         for index in range(len(model.imputation_ids))]
        raw_pooled = np.vstack(probabilities).mean(axis=0)
        error = np.abs(raw_pooled - archived[complete])
        result.update(raw_complete_max_abs_error=float(error.max()),
                      raw_complete_within_tolerance=int(np.count_nonzero(error <= TOLERANCE)))
    return result


def audit(root=ROOT, *, bundle=None, mi_dir=None, prediction_dir=None, raw_internal=None, raw_external=None):
    root = Path(root)
    bundle = Path(bundle) if bundle is not None else root / BUNDLE
    mi_dir = Path(mi_dir) if mi_dir is not None else root / MI
    prediction_dir = Path(prediction_dir) if prediction_dir is not None else bundle.parent / "evaluation"
    model = ModelA.load(bundle)
    raw_paths = {"internal_test": raw_internal, "external_eicu": raw_external}
    reports = []
    for cohort, filename in COHORTS.items():
        identity = (*KEYS, "hospital_id") if cohort == "external_eicu" else KEYS
        frozen = read_rows(prediction_dir / f"{cohort}_predictions.csv", identity, ("p_A",))
        p_frozen = probability(pd.to_numeric(frozen.p_A, errors="coerce"), len(frozen), f"{cohort} frozen")
        member_predictions = []
        for index, imp in enumerate(model.imputation_ids):
            path = mi_dir / f"imp_{imp:02d}" / filename
            label = f"{cohort} imputation {imp}"
            completed = align_rows(read_rows(path, identity, FEATURES), frozen, identity, label)
            member_predictions.append(member_probabilities(model, index, completed, label))
        pooled = np.vstack(member_predictions).mean(axis=0)
        errors = np.abs(p_frozen - pooled)
        n_mismatch = int(np.count_nonzero(errors > TOLERANCE))
        if n_mismatch:
            raise ValueError(f"{cohort}: {n_mismatch}/{len(frozen)} predictions exceed {TOLERANCE:g}; "
                             f"max_abs_error={errors.max():.8g}")
        result = {"cohort": cohort, "n": len(frozen), "deaths": int(frozen[OUTCOME].sum()),
                  "members": len(member_predictions), "max_abs_error": float(errors.max()),
                  "mean_abs_error": float(errors.mean())}
        if raw_paths[cohort] is not None:
            result.update(summarize_raw(model, Path(raw_paths[cohort]), frozen, p_frozen, identity, cohort))
        reports.append(result)
    return reports, check_complete_endpoint(model)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--mi-dir", type=Path)
    parser.add_argument("--prediction-dir", type=Path)
    parser.add_argument("--raw-internal", type=Path, help="Original pre-imputation MIMIC-IV cohort CSV")
    parser.add_argument("--raw-external", type=Path, help="Original pre-imputation eICU cohort CSV")
    args = parser.parse_args()
    reports, endpoint_error = audit(args.root, bundle=args.bundle, mi_dir=args.mi_dir,
                                    prediction_dir=args.prediction_dir, raw_internal=args.raw_internal,
                                    raw_external=args.raw_external)
    for row in reports:
        print(f"PASS {row['cohort']}: n={row['n']}, deaths={row['deaths']}, members={row['members']}, "
              f"max_abs_error={row['max_abs_error']:.8g}, mean_abs_error={row['mean_abs_error']:.8g}")
        if "n_raw_complete" in row:
            print(f"  Pre-imputation cases accepted by app: {row['n_raw_complete']} "
                  f"({row['deaths_raw_complete']} deaths). Missing inputs: {row['n_raw_missing']} "
                  f"({row['deaths_raw_missing']} deaths). Present but invalid: {row['n_raw_blocked']} "
                  f"({row['deaths_raw_blocked']} deaths).")
            if row["n_raw_complete"]:
                print(f"  Complete cases within {TOLERANCE:g} of frozen p_A: "
                      f"{row['raw_complete_within_tolerance']}/{row['n_raw_complete']}; "
                      f"max_abs_error={row['raw_complete_max_abs_error']:.8g}")
    print(f"PASS complete-input inference vs. mean member probabilities: abs_error={endpoint_error:.8g}")


if __name__ == "__main__":
    main()
