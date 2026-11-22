"""Exact single-case probability SHAP for the eight inputs of frozen Model A."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import comb
from pathlib import Path
from threading import Lock

import numpy as np
import pandas as pd

from risk_model import FEATURES, ModelA, prepare_features, validate_complete


DEFAULT_MI_DIR = Path("D:/0mimic/TBI/04_Models/TBI_MI_Step1")
DEFAULT_BACKGROUND_SIZE = 100
DEFAULT_SEED = 20261122
MAX_BACKGROUND_SIZE = 200
METHOD = "Exact interventional SHAP on the probability scale"
OUTCOME = "death_in_hosp"
IDS = ("stay_id", "subject_id", "hadm_id")
TRAINING_FILE = "mimiciv_train.csv"
SPLIT_FILE = "mimiciv_split_manifest.csv"
TOLERANCE = 1e-6

N_FEATURES = len(FEATURES)
FEATURE_BITS = 1 << np.arange(N_FEATURES)
SUBSETS = np.arange(1 << N_FEATURES)
MASKS = (SUBSETS[:, None] & FEATURE_BITS) != 0
WEIGHTS = np.array([1 / (N_FEATURES * comb(N_FEATURES - 1, k)) for k in range(N_FEATURES)])
CONTRASTS = tuple(
    (subsets, subsets | bit, WEIGHTS[MASKS[subsets].sum(axis=1)])
    for index, bit in enumerate(FEATURE_BITS)
    for subsets in (SUBSETS[~MASKS[:, index]],)
)


def _read(path, fields, identity):
    if not path.is_file():
        raise FileNotFoundError(f"Missing training background file: {path}")
    # Match the original CSV parser and preserve identifiers as text for linkage.
    try:
        frame = pd.read_csv(path, encoding="utf-8-sig", usecols=list(fields),
                            dtype={name: "string" for name in identity})
    except (ValueError, UnicodeError) as error:
        raise ValueError(f"{path.name}: required background columns could not be read.") from error
    if frame.empty:
        raise ValueError(f"{path.name}: no background rows were found.")
    for name in identity:
        values = frame[name].str.strip().str.replace(r"^([0-9]+)\.0+$", r"\1", regex=True)
        if values.isna().any() or values.eq("").any() or values.duplicated().any():
            raise ValueError(f"{path.name}: missing or duplicate {name} values.")
        frame[name] = values
    outcome = pd.to_numeric(frame[OUTCOME], errors="coerce")
    if not outcome.isin((0, 1)).all():
        raise ValueError(f"{path.name}: the recorded outcome must be coded as 0 or 1.")
    frame[OUTCOME] = outcome.astype(np.int8)
    return frame


def _training_manifest(mi_dir):
    path = mi_dir / SPLIT_FILE
    split = _read(path, ("subject_id", "partition", OUTCOME), ("subject_id",))
    partition = split.partition.astype("string").str.strip()
    if partition.isna().any() or set(partition) != {"training", "internal_test"}:
        raise ValueError(f"{path.name}: expected training and internal-test partitions.")
    return split.loc[partition.eq("training")].set_index("subject_id")[OUTCOME]


def _member_background(mi_dir, imputation, expected):
    path = mi_dir / f"imp_{imputation:02d}" / TRAINING_FILE
    frame = _read(path, (*IDS, OUTCOME, *FEATURES), IDS).set_index("stay_id")
    subjects = pd.Index(frame.subject_id)
    if set(subjects) != set(expected.index) or not np.array_equal(
            frame[OUTCOME].to_numpy(), expected.loc[subjects].to_numpy()):
        raise ValueError(f"Imputation {imputation}: the background does not match the original training split.")
    return frame


def _transformed(preprocessor, frame):
    result = np.asarray(preprocessor.transform(frame))
    if result.shape != (len(frame), N_FEATURES) or not np.isfinite(result).all():
        raise ValueError("The frozen preprocessor did not produce eight finite input columns.")
    return result


def _native_probabilities(estimator, values):
    probabilities = np.asarray(estimator.predict_proba(values))
    if probabilities.shape != (len(values), 2):
        raise ValueError("The model returned an invalid probability array.")
    positive = probabilities[:, 1]
    if not np.isfinite(positive).all() or not ((0 <= positive) & (positive <= 1)).all():
        raise ValueError("The model returned invalid probabilities.")
    return positive


@dataclass(frozen=True)
class _Member:
    background: np.ndarray
    baseline: float
    mask_by_column: np.ndarray

    @classmethod
    def prepare(cls, pipeline, training_features):
        preprocessor = pipeline.named_steps["preprocess"]
        estimator = pipeline.named_steps["model"]
        names = list(preprocessor.get_feature_names_out())
        # In the original frozen model each continuous input is transformed
        # independently and sex contributes one dropped-first indicator.
        by_feature = {feature: f"numeric__{feature}" for feature in FEATURES if feature != "gender"}
        by_feature["gender"] = "gender__gender_M"
        if len(names) != N_FEATURES or set(names) != set(by_feature.values()):
            raise ValueError("Local SHAP requires the frozen Model A eight-variable preprocessing map.")
        feature_indices = {by_feature[feature]: i for i, feature in enumerate(FEATURES)}
        mask_by_column = MASKS[1:-1, [feature_indices[name] for name in names]]
        mask_by_column.setflags(write=False)
        background = _transformed(preprocessor, training_features)
        background.setflags(write=False)
        baseline = float(_native_probabilities(estimator, background).mean(dtype=np.float64))
        return cls(background, baseline, mask_by_column)

    def value_function(self, pipeline, patient, prediction):
        """Average predictions over the training reference for every input subset."""
        point = _transformed(pipeline.named_steps["preprocess"], patient)[0]
        hybrid = np.where(self.mask_by_column[:, None, :], point, self.background)
        probabilities = _native_probabilities(pipeline.named_steps["model"], hybrid.reshape(-1, N_FEATURES))
        intermediate = probabilities.reshape(len(SUBSETS) - 2, len(self.background)).mean(axis=1, dtype=np.float64)
        values = np.empty(len(SUBSETS))
        values[0] = self.baseline
        values[-1] = prediction
        values[1:-1] = intermediate
        return values


@dataclass(frozen=True)
class LocalShap:
    model: ModelA
    members: tuple[_Member, ...]
    background_size: int
    seed: int
    _lock: Lock = field(default_factory=Lock, init=False, repr=False, compare=False)

    @classmethod
    def load(cls, model, mi_dir=DEFAULT_MI_DIR, *, background_size=DEFAULT_BACKGROUND_SIZE, seed=DEFAULT_SEED):
        if type(background_size) is not int or not 1 <= background_size <= MAX_BACKGROUND_SIZE:
            raise ValueError(f"The training background size must be an integer from 1 to {MAX_BACKGROUND_SIZE}.")
        if type(seed) is not int or seed < 0:
            raise ValueError("The training background seed must be a non-negative integer.")
        mi_dir = Path(mi_dir)
        expected = _training_manifest(mi_dir)
        reference = None
        selected = None
        members = []
        for imputation, pipeline in zip(model.imputation_ids, model.pipelines):
            frame = _member_background(mi_dir, imputation, expected)
            identity = frame.loc[:, [*IDS[1:], OUTCOME]]
            if reference is None:
                reference = identity
                size = min(background_size, len(reference))
                # Match the row sampling of the original cohort SHAP analysis.
                positions = np.sort(np.random.default_rng(seed).choice(len(reference), size, replace=False))
                selected = reference.index[positions]
            elif not reference.index.difference(identity.index).empty or not identity.loc[reference.index].equals(reference):
                raise ValueError(f"Imputation {imputation}: training stays are inconsistent across imputations.")
            try:
                features = prepare_features(frame.loc[selected, list(FEATURES)])
            except ValueError as error:
                raise ValueError(f"Imputation {imputation}: the training background contains invalid predictors.") from error
            members.append(_Member.prepare(pipeline, features))
        return cls(model, tuple(members), len(selected), seed)

    def explain(self, values):
        checked = validate_complete(values)
        patient = prepare_features(pd.DataFrame([checked], columns=FEATURES))
        probabilities = []
        baselines = []
        contributions = []
        max_error = 0.0
        # A single CPU calculation at a time keeps requests from oversubscribing
        # the frozen tree models and from allocating simultaneous coalition arrays.
        with self._lock:
            for index, member in enumerate(self.members):
                pipeline = self.model.pipelines[index]
                original = self.model._member_prepared(index, patient)[0]
                subset_values = member.value_function(pipeline, patient, original)
                phi = np.array([weight @ (subset_values[with_feature] - subset_values[without_feature])
                                for without_feature, with_feature, weight in CONTRASTS])
                error = abs(float(member.baseline + phi.sum() - original))
                if not np.isfinite(error) or error > TOLERANCE:
                    raise ValueError("A member's SHAP contributions did not reconstruct its prediction.")
                max_error = max(error, max_error)
                probabilities.append(original)
                baselines.append(member.baseline)
                contributions.append(phi)
        # Preserve the existing app's float32 pooled prediction while computing
        # baseline and attribution means in double precision.
        probability = float(np.asarray(probabilities).mean())
        baseline = float(np.mean(baselines))
        phi = np.mean(contributions, axis=0)
        error = abs(float(baseline + phi.sum() - probability))
        if not np.isfinite(error) or error > TOLERANCE:
            raise ValueError("The pooled SHAP contributions did not reconstruct the prediction.")
        return {
            "method": METHOD,
            "baseline_probability": baseline,
            "predicted_probability": probability,
            "background_size": self.background_size,
            "additivity_error": error,
            "max_member_additivity_error": max_error,
            "contributions": [
                {"feature": feature, "value": checked[feature], "contribution": float(value)}
                for feature, value in zip(FEATURES, phi)
            ],
        }
