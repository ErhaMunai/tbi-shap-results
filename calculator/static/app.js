"use strict";

const form = document.querySelector("#prediction-form");
const submit = document.querySelector("#run-prediction");
const fillExample = document.querySelector("#fill-example");
const clearInputs = document.querySelector("#clear-inputs");
const formError = document.querySelector("#form-error");
const exampleNote = document.querySelector("#example-note");
const serviceStatus = document.querySelector("#service-status");
const serviceNotice = document.querySelector("#service-notice");
const serviceMessage = document.querySelector("#service-message");
const retryStatus = document.querySelector("#retry-status");
const resultCard = document.querySelector("#result-card");
const resultTitle = document.querySelector("#result-title");
const resultValue = document.querySelector("#result-value");
const resultCaption = document.querySelector("#result-caption");
const modelId = document.querySelector("#model-id");
const predictorCount = document.querySelector("#predictor-count");
const imputationCount = document.querySelector("#imputation-count");
const cohortSelect = document.querySelector("#ranking-cohort");
const rankStatus = document.querySelector("#rank-status");
const fieldGrid = document.querySelector("#ranked-fields");
const comparison = document.querySelector("#comparison");
const clearComparison = document.querySelector("#clear-comparison");
const previousLabel = document.querySelector("#previous-label");
const previousValue = document.querySelector("#previous-value");
const deltaRow = document.querySelector("#delta-row");
const deltaValue = document.querySelector("#delta-value");

const waterfall = document.querySelector("#waterfall");
const waterfallRows = document.querySelector("#waterfall-rows");
const waterfallTicks = document.querySelector("#waterfall-ticks");
const explanationPlaceholder = document.querySelector("#explanation-placeholder");
const explanationText = document.querySelector("#explanation-placeholder-text");
const blankExplanation = "Submit an eligible case to see its SHAP waterfall.";

const fieldNames = [
  "eligible", "age", "gender", "gcs_motor_first", "sbp_mean", "spo2_mean",
  "hr_mean", "hemoglobin_g_dl", "creatinine_mg_dl",
];
const predictorNames = fieldNames.slice(1);
const fieldCards = new Map(Array.from(fieldGrid.children, (card) => [card.dataset.feature, card]));
const cohortLabels = new Map([
  ["training", "MIMIC-IV training"],
  ["internal_test", "MIMIC-IV internal test"],
  ["external_eicu", "eICU external validation"],
]);
const cohortOrder = [...cohortLabels.keys()];
const featureDisplay = {
  age: ["Age", "years"],
  gender: ["Sex", ""],
  gcs_motor_first: ["GCS motor score", ""],
  sbp_mean: ["Mean systolic BP", "mmHg"],
  spo2_mean: ["Mean SpO₂", "%"],
  hr_mean: ["Mean heart rate", "bpm"],
  hemoglobin_g_dl: ["Hemoglobin", "g/dL"],
  creatinine_mg_dl: ["Creatinine", "mg/dL"],
};
const example = {
  age: 56,
  gender: "M",
  gcs_motor_first: 5,
  sbp_mean: 124.6,
  spo2_mean: 97.2,
  hr_mean: 84.1,
  hemoglobin_g_dl: 12.8,
  creatinine_mg_dl: 0.92,
};
const percentage = new Intl.NumberFormat("en-US", {
  style: "percent", minimumFractionDigits: 2, maximumFractionDigits: 2,
});
const signedDifference = new Intl.NumberFormat("en-US", {
  minimumFractionDigits: 2, maximumFractionDigits: 2, signDisplay: "exceptZero",
});
const importanceNumber = new Intl.NumberFormat("en-US", {
  minimumFractionDigits: 2, maximumFractionDigits: 3,
});

let ready = false;
let activePrediction = null;
let checkingStatus = false;
let rankings = new Map();
let rankingScale = 1;
let lastScenario = null;

function field(name) {
  return form.elements.namedItem(name);
}

function updateSelectLabel(select) {
  select.nextElementSibling.textContent = select.options[select.selectedIndex]?.text || "";
}

function syncSubmit() {
  submit.disabled = !ready || activePrediction !== null;
  submit.textContent = activePrediction === null ? "Predict mortality risk" : "Calculating…";
}

function setFormError(message = "") {
  formError.textContent = message;
  formError.hidden = !message;
}

function updateModelDetails(info) {
  if (typeof info.model_id === "string" && info.model_id.trim()) {
    modelId.textContent = info.model_id;
  } else {
    modelId.textContent = "XGBoost Model A";
  }
  const predictors = Number.isInteger(info.predictor_count) ? info.predictor_count : 8;
  predictorCount.textContent = `${predictors}`;
  imputationCount.textContent = Number.isInteger(info.imputation_count) && info.imputation_count > 0
    ? `${info.imputation_count}` : "—";
}

function setServiceStatus(state, label, notice = "") {
  ready = state === "ready";
  serviceStatus.dataset.state = state;
  serviceStatus.textContent = label;
  serviceMessage.textContent = notice;
  serviceNotice.hidden = !notice;
  syncSubmit();
}

async function fetchJson(url) {
  const response = await fetch(url, { cache: "no-store" });
  if (!response.ok) throw new Error("resource unavailable");
  return response.json();
}

function readRankings(data) {
  if (!Array.isArray(data?.cohorts)) throw new Error("invalid cohorts");
  const result = new Map();
  for (const cohort of data.cohorts) {
    if (!cohortOrder.includes(cohort?.id) || result.has(cohort.id)
        || typeof cohort.label !== "string" || !cohort.label.trim()
        || !Array.isArray(cohort.importance) || cohort.importance.length !== predictorNames.length) {
      throw new Error("invalid cohort ranking");
    }
    const used = new Set();
    const numbers = new Set();
    for (const row of cohort.importance) {
      if (!predictorNames.includes(row?.feature) || used.has(row.feature)
          || !Number.isInteger(row.rank) || row.rank < 1 || row.rank > predictorNames.length || numbers.has(row.rank)
          || !Number.isFinite(row.mean_abs_shap_pp) || row.mean_abs_shap_pp < 0) {
        throw new Error("invalid ranking row");
      }
      used.add(row.feature);
      numbers.add(row.rank);
    }
    result.set(cohort.id, { label: cohortLabels.get(cohort.id), rows: [...cohort.importance].sort((a, b) => a.rank - b.rank) });
  }
  if (!result.has("training")) throw new Error("missing training ranking");
  return result;
}

function showRanking(id) {
  const cohort = rankings.get(id);
  if (!cohort) return;
  rankStatus.textContent = `Inputs ordered by importance in ${cohort.label}.`;
  delete rankStatus.dataset.state;
  fieldGrid.dataset.cohort = id;
  for (const row of cohort.rows) {
    const card = fieldCards.get(row.feature);
    const position = card.querySelector(".rank-position");
    const importance = card.querySelector(".rank-importance");
    const meter = card.querySelector(".rank-meter");
    position.textContent = String(row.rank);
    importance.textContent = `${importanceNumber.format(row.mean_abs_shap_pp)} pp`;
    meter.max = rankingScale;
    meter.optimum = rankingScale;
    meter.value = row.mean_abs_shap_pp;
    position.hidden = importance.hidden = meter.hidden = false;
    fieldGrid.append(card);
  }
}

function configureRankings(next) {
  const selected = next.has(cohortSelect.value) ? cohortSelect.value : "training";
  rankings = next;
  rankingScale = Math.max(...Array.from(rankings.values()).flatMap((cohort) =>
    cohort.rows.map((row) => row.mean_abs_shap_pp))) || 1;
  const options = cohortOrder.filter((id) => rankings.has(id)).map((id) => new Option(rankings.get(id).label, id));
  cohortSelect.replaceChildren(...options);
  cohortSelect.value = selected;
  cohortSelect.disabled = false;
  updateSelectLabel(cohortSelect);
  showRanking(selected);
}

function disableRankings() {
  rankings = new Map();
  cohortSelect.replaceChildren(new Option("Rankings unavailable", ""));
  cohortSelect.disabled = true;
  updateSelectLabel(cohortSelect);
  rankStatus.dataset.state = "error";
  rankStatus.textContent = "A valid cohort ranking could not be loaded.";
  delete fieldGrid.dataset.cohort;
  for (const name of predictorNames) {
    const card = fieldCards.get(name);
    for (const element of card.querySelectorAll(".rank-position, .rank-importance, .rank-meter")) {
      element.hidden = true;
    }
    fieldGrid.append(card);
  }
}

async function checkStatus() {
  if (checkingStatus) return;
  checkingStatus = true;
  retryStatus.disabled = true;
  setServiceStatus("checking", "Checking model");
  try {
    const [status, ranking] = await Promise.allSettled([
      fetchJson("/api/status"), fetchJson("/api/rankings"),
    ]);
    if (status.status !== "fulfilled") throw new Error("status unavailable");
    const data = status.value;
    updateModelDetails(data);
    let next;
    if (data.ranking_ready === true && ranking.status === "fulfilled") {
      try {
        next = readRankings(ranking.value);
      } catch {
        next = null;
      }
    }
    if (next) configureRankings(next);
    else disableRankings();
    if (!next) {
      const notice = data.ready === true
        ? "Load a valid SHAP ranking file, then retry."
        : "Load the frozen model and a valid SHAP ranking file, then retry.";
      setServiceStatus("unavailable", "Rankings unavailable", notice);
    } else if (data.ready === true) {
      setServiceStatus("ready", "Model ready");
    } else {
      setServiceStatus("unavailable", "Model unavailable", "Load the frozen model to enable prediction.");
    }
  } catch {
    updateModelDetails({});
    disableRankings();
    setServiceStatus("unavailable", "Service unavailable", "Open this page through the running prediction service, then retry.");
  } finally {
    checkingStatus = false;
    retryStatus.disabled = false;
  }
}

function clearExplanation(message = blankExplanation) {
  waterfall.hidden = true;
  waterfallRows.replaceChildren();
  waterfallTicks.replaceChildren();
  explanationText.textContent = message;
  explanationPlaceholder.hidden = false;
}

function isProbability(value) {
  return Number.isFinite(value) && 0 <= value && value <= 1;
}

function waterfallData(explanation, probability) {
  if (!isProbability(explanation?.baseline_probability)
      || !isProbability(explanation.predicted_probability)
      || Math.abs(explanation.predicted_probability - probability) > 1e-5
      || !Array.isArray(explanation.contributions)
      || explanation.contributions.length !== predictorNames.length) return null;

  const seen = new Set();
  for (const row of explanation.contributions) {
    if (!predictorNames.includes(row?.feature) || seen.has(row.feature)
        || !Number.isFinite(row.contribution) || Math.abs(row.contribution) > 1
        || (row.feature === "gender" ? !["F", "M"].includes(row.value) : !Number.isFinite(row.value))) return null;
    seen.add(row.feature);
  }

  let cumulative = explanation.baseline_probability;
  const steps = [...explanation.contributions]
    .sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution))
    .map((item) => {
      const start = cumulative;
      cumulative += item.contribution;
      return { ...item, start, end: cumulative };
    });
  if (Math.abs(cumulative - probability) > 1e-5) return null;
  return { baseline: explanation.baseline_probability, steps };
}

function chartScale(values) {
  const min = Math.min(...values);
  const max = Math.max(...values);
  const span = Math.max(max - min, .01);
  const niceSteps = [.0025, .005, .01, .02, .025, .05, .1, .2, .25, .5, 1, 2, 5];
  const step = niceSteps.find((candidate) => candidate >= span / 4) || 5;
  const minTick = Math.floor((min - span * .08) / step);
  const maxTick = Math.ceil((max + span * .08) / step);
  const ticks = Array.from({ length: maxTick - minTick + 1 }, (_, index) => (minTick + index) * step);
  return { ticks, position: (value) => (value / step - minTick) / (maxTick - minTick) * 100 };
}

const shortNumber = new Intl.NumberFormat("en-US", { maximumFractionDigits: 4 });
const tickNumber = new Intl.NumberFormat("en-US", { maximumFractionDigits: 2 });

function featureValue(feature, value) {
  if (feature === "gender") return value === "F" ? "Female" : "Male";
  const units = featureDisplay[feature][1];
  return `${shortNumber.format(value)}${units && units !== "%" ? " " : ""}${units}`;
}

function createElement(tag, className, text = "") {
  const element = document.createElement(tag);
  element.className = className;
  element.textContent = text;
  return element;
}

function plotStep(step, position, ticks) {
  const terminal = !step.feature;
  const direction = terminal ? "terminal" : step.contribution > 0 ? "positive" : step.contribution < 0 ? "negative" : "zero";
  const row = createElement("li", `waterfall-row ${direction}${step.final ? " final" : ""}`);
  if (step.feature) row.dataset.feature = step.feature;
  const name = step.label || featureDisplay[step.feature][0];
  const label = createElement("span", "waterfall-label", name);
  if (!terminal) label.append(createElement("small", "waterfall-input", featureValue(step.feature, step.value)));

  const track = createElement("span", "waterfall-track");
  track.setAttribute("aria-hidden", "true");
  for (const tick of ticks) {
    const gridLine = createElement("i", "waterfall-grid");
    gridLine.style.left = `${position(tick)}%`;
    track.append(gridLine);
  }
  const mark = createElement("i", "waterfall-mark");
  mark.style.left = `${position(Math.min(step.start, step.end))}%`;
  mark.style.width = `${Math.abs(position(step.end) - position(step.start))}%`;
  track.append(mark);
  if (!step.final) {
    const connector = createElement("i", "waterfall-connector");
    connector.style.left = `${position(step.end)}%`;
    track.append(connector);
  }

  const magnitude = terminal ? percentage.format(step.end) : `${signedDifference.format(100 * step.contribution)} pp`;
  const numeric = createElement("span", "waterfall-numeric", magnitude);
  if (!terminal) {
    numeric.append(createElement("span", "visually-hidden", `; cumulative model output ${percentage.format(step.end)}`));
  }
  row.append(label, track, numeric);
  return row;
}

function showExplanation(data) {
  const output = waterfallData(data.explanation, data.probability);
  if (!output) {
    const detail = typeof data.explanation_message === "string" && data.explanation_message.trim();
    clearExplanation(detail || "An individual SHAP explanation is unavailable for this prediction.");
    return;
  }
  const { baseline, steps } = output;
  const { ticks, position } = chartScale([baseline, ...steps.map((step) => step.end), data.probability]);
  const rows = [
    { label: "Model baseline", start: baseline, end: baseline },
    ...steps,
    { label: "Final prediction", start: data.probability, end: data.probability, final: true },
  ];
  waterfallRows.replaceChildren(...rows.map((row) => plotStep(row, position, ticks)));
  waterfallTicks.replaceChildren(...ticks.map((value) => {
    const tick = createElement("span", "waterfall-tick", tickNumber.format(100 * value));
    tick.style.left = `${position(value)}%`;
    return tick;
  }));
  explanationPlaceholder.hidden = true;
  waterfall.hidden = false;
}

function clearResult(caption = "Submit the predictors to calculate a probability.") {
  resultValue.textContent = "—";
  resultCaption.textContent = caption;
}

function showComparison(previous, current = null) {
  comparison.hidden = !previous;
  deltaRow.hidden = !previous || !current;
  if (!previous) {
    previousValue.textContent = "—";
    deltaValue.textContent = "—";
    return;
  }
  previousLabel.textContent = previous.simulated ? "Previous prediction (example)" : "Previous prediction";
  previousValue.textContent = percentage.format(previous.probability);
  if (current) {
    const difference = 100 * (current.probability - previous.probability);
    deltaValue.textContent = `${signedDifference.format(difference)} pp`;
  } else {
    deltaValue.textContent = "—";
  }
}

function resetCurrent(caption, explanationMessage) {
  clearResult(caption);
  clearExplanation(explanationMessage);
  showComparison(lastScenario);
}

function cancelPrediction() {
  if (activePrediction) {
    activePrediction.abort();
    activePrediction = null;
  }
  resultCard.removeAttribute("aria-busy");
  syncSubmit();
}

function clearValidation() {
  for (const name of fieldNames) field(name).removeAttribute("aria-invalid");
  setFormError();
}

function validationMessage(input) {
  if (input.name === "eligible") return "Confirm that the study eligibility criteria are met.";
  const label = featureDisplay[input.name][0];
  if (input.validity.badInput) return `Enter a valid number for ${label}.`;
  if (input.validity.valueMissing) return `Complete the ${label} field.`;
  if (input.name === "gcs_motor_first") return "Enter a whole number from 1 to 6 for the GCS motor score.";
  if (input.validity.rangeUnderflow || input.validity.rangeOverflow) {
    return input.max
      ? `Enter a value from ${input.min} to ${input.max} for ${label}.`
      : `Age must be at least ${input.min} years.`;
  }
  if (input.name === "age" && input.validity.stepMismatch) return "Enter age in whole years.";
  return `Check the value entered for ${label}.`;
}

function validate() {
  const invalid = [];
  for (const name of fieldNames) {
    const input = field(name);
    if (input.validity.valid) {
      input.removeAttribute("aria-invalid");
    } else {
      input.setAttribute("aria-invalid", "true");
      invalid.push(input);
    }
  }
  if (!invalid.length) return true;

  setFormError(validationMessage(invalid[0]));
  invalid[0].focus();
  return false;
}

function predictionPayload() {
  const payload = { eligible: field("eligible").checked };
  for (const name of predictorNames) {
    const input = field(name);
    payload[name] = input.type === "number" ? input.valueAsNumber : input.value;
  }
  return payload;
}

form.addEventListener("input", (event) => {
  if (!fieldNames.includes(event.target.name)) return;
  if (event.target.name === "gender") updateSelectLabel(event.target);
  event.target.removeAttribute("aria-invalid");
  cancelPrediction();
  setFormError();
  resetCurrent();
});

clearInputs.addEventListener("click", () => {
  cancelPrediction();
  clearValidation();
  for (const name of predictorNames) field(name).value = "";
  updateSelectLabel(field("gender"));
  field("eligible").checked = false;
  exampleNote.hidden = true;
  resetCurrent();
});

fillExample.addEventListener("click", () => {
  cancelPrediction();
  clearValidation();
  resetCurrent();
  for (const [name, value] of Object.entries(example)) field(name).value = String(value);
  updateSelectLabel(field("gender"));
  field("eligible").checked = true;
  exampleNote.hidden = false;
});

cohortSelect.addEventListener("change", () => {
  updateSelectLabel(cohortSelect);
  showRanking(cohortSelect.value);
});
clearComparison.addEventListener("click", () => {
  lastScenario = null;
  showComparison(null);
  resultTitle.focus({ preventScroll: true });
});
retryStatus.addEventListener("click", checkStatus);

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!ready || activePrediction !== null) return;
  if (!validate()) return;

  setFormError();
  resetCurrent("Calculating the prediction.", "Calculating individual SHAP contributions…");
  resultCard.setAttribute("aria-busy", "true");
  const request = new AbortController();
  activePrediction = request;
  syncSubmit();

  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(predictionPayload()),
      signal: request.signal,
    });
    const data = await response.json();
    if (activePrediction !== request) return;
    if (!response.ok) {
      if (response.status === 503) void checkStatus();
      throw new Error(typeof data.error === "string" ? data.error : "Prediction failed. Try again.");
    }
    if (!isProbability(data.probability)) {
      throw new Error("The service returned an invalid probability. Try again.");
    }

    updateModelDetails(data);
    const current = { probability: data.probability, simulated: !exampleNote.hidden };
    resultValue.textContent = percentage.format(current.probability);
    resultCaption.textContent = current.simulated ? "Predicted probability for the simulated example" : "Predicted probability";
    showExplanation(data);
    showComparison(lastScenario, current);
    lastScenario = current;
    if (window.matchMedia("(max-width: 860px)").matches) resultCard.scrollIntoView({ block: "start" });
    resultTitle.focus({ preventScroll: true });
  } catch (error) {
    if (activePrediction !== request) return;
    resetCurrent("The prediction could not be calculated.");
    setFormError(error instanceof TypeError || error instanceof SyntaxError
      ? "The prediction service could not be reached. Try again." : error.message);
  } finally {
    if (activePrediction === request) {
      activePrediction = null;
      resultCard.removeAttribute("aria-busy");
      syncSubmit();
    }
  }
});

void checkStatus();
