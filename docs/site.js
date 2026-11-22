"use strict";

const cohortOrder = ["training", "internal_test", "external_eicu"];
const figureSection = document.querySelector("#shap-figure");
const figureNav = document.querySelector("#figure-nav");
const figureMessage = document.querySelector("#figure-message");
const figureLink = document.querySelector("#figure-link");
const image = document.querySelector("#integrated-image");
const openOriginal = document.querySelector("#open-original");
const downloadPng = document.querySelector("#download-png");
const downloadPdf = document.querySelector("#download-pdf");
const tableMessage = document.querySelector("#table-message");
const tabList = document.querySelector("#cohort-tabs");
const panelList = document.querySelector("#cohort-panels");
const integers = new Intl.NumberFormat("zh-CN");
const contributions = new Intl.NumberFormat("zh-CN", {
  minimumFractionDigits: 2,
  maximumFractionDigits: 3,
});

function node(tag, options = {}) {
  const element = document.createElement(tag);
  if (options.className) element.className = options.className;
  if (options.text !== undefined) element.textContent = options.text;
  return element;
}

function posInteger(value) {
  return Number.isInteger(value) && value > 0;
}

function nonEmpty(value) {
  return typeof value === "string" && Boolean(value.trim());
}

function readManifest(data) {
  if (data?.published_from !== "model_a_shap_integrated" || !Array.isArray(data.cohorts)) {
    throw new Error("invalid manifest");
  }
  const assets = data.assets;
  if (!assets || ![null, "assets/shap-integrated.png"].includes(assets.png)
      || ![null, "assets/shap-integrated.pdf"].includes(assets.pdf)) {
    throw new Error("invalid assets");
  }
  const byId = new Map(data.cohorts.map((cohort) => [cohort?.id, cohort]));
  if (byId.size !== cohortOrder.length || data.cohorts.length !== cohortOrder.length
      || !cohortOrder.every((id) => byId.has(id))) {
    throw new Error("invalid cohorts");
  }
  const cohorts = cohortOrder.map((id) => byId.get(id));
  for (const cohort of cohorts) {
    if (!nonEmpty(cohort.label) || !posInteger(cohort.n) || !posInteger(cohort.n_imputations)
        || (cohort.n_cohort !== undefined && (!posInteger(cohort.n_cohort) || cohort.n_cohort < cohort.n))
        || !Array.isArray(cohort.importance) || !cohort.importance.length) {
      throw new Error("invalid cohort metadata");
    }
    const features = new Set();
    const ranks = new Set();
    for (const row of cohort.importance) {
      if (!nonEmpty(row?.feature) || !nonEmpty(row.label) || !posInteger(row.rank)
          || !Number.isFinite(row.mean_abs_shap_pp) || row.mean_abs_shap_pp < 0
          || features.has(row.feature) || ranks.has(row.rank)) {
        throw new Error("invalid importance row");
      }
      features.add(row.feature);
      ranks.add(row.rank);
    }
  }
  return { cohorts, assets };
}

function setLink(link, asset) {
  link.href = `./${asset}`;
  link.hidden = false;
}

function showFigure(assets) {
  if (assets.png === null) return;

  figureSection.hidden = false;
  figureNav.hidden = false;
  setLink(downloadPng, assets.png);
  if (assets.pdf) setLink(downloadPdf, assets.pdf);

  image.addEventListener("load", () => {
    setLink(figureLink, assets.png);
    setLink(openOriginal, assets.png);
    figureMessage.hidden = true;
  }, { once: true });
  image.addEventListener("error", () => {
    downloadPng.hidden = true;
    figureMessage.dataset.state = "error";
    figureMessage.textContent = "未能加载发布图版。";
  }, { once: true });
  image.src = `./${assets.png}`;
}

function getCohortDetails(cohort) {
  const details = node("p", { className: "cohort-details" });
  const sample = node("span", { text: "解释人数 " });
  const count = integers.format(cohort.n);
  const denominator = cohort.n_cohort > cohort.n ? ` / ${integers.format(cohort.n_cohort)}` : "";
  sample.append(node("strong", { text: count + denominator }));
  const imputation = node("span", { text: "插补模型 " });
  imputation.append(node("strong", { text: integers.format(cohort.n_imputations) }));
  details.append(sample, imputation);
  return details;
}

function getMetricCell(row, scale) {
  const cell = node("td");
  const metric = node("div", { className: "metric" });
  const bar = node("meter", { className: "metric-bar" });
  bar.min = 0;
  bar.max = scale;
  bar.optimum = scale;
  bar.value = row.mean_abs_shap_pp;
  bar.setAttribute("aria-hidden", "true");
  const number = node("span", { className: "metric-number", text: contributions.format(row.mean_abs_shap_pp) });
  metric.append(bar, number);
  cell.append(metric);
  return cell;
}

function getImportanceTable(cohort, scale) {
  const table = node("table", { className: "importance-table" });
  table.setAttribute("aria-label", `${cohort.label}特征重要性`);
  const columns = node("colgroup");
  columns.append(node("col", { className: "col-rank" }), node("col", { className: "col-feature" }), node("col"));
  const head = node("thead");
  const headers = node("tr");
  const rank = node("th", { text: "排名" });
  const feature = node("th", { text: "特征" });
  const mean = node("th");
  for (const cell of [rank, feature, mean]) cell.scope = "col";
  const sort = node("button", { className: "sort-header", text: "平均绝对 SHAP（百分点）" });
  sort.type = "button";
  const direction = node("span", { className: "sort-icon" });
  direction.setAttribute("aria-hidden", "true");
  sort.append(direction);
  mean.append(sort);
  headers.append(rank, feature, mean);
  head.append(headers);
  const body = node("tbody");
  table.append(columns, head, body);

  let descending = true;
  function renderRows() {
    mean.setAttribute("aria-sort", descending ? "descending" : "ascending");
    sort.setAttribute("aria-label", `平均绝对 SHAP，点按切换为${descending ? "升序" : "降序"}`);
    direction.textContent = descending ? "↓" : "↑";

    const rows = [...cohort.importance].sort((a, b) => {
      const difference = a.mean_abs_shap_pp - b.mean_abs_shap_pp;
      return (descending ? -difference : difference) || a.rank - b.rank;
    });
    const next = document.createDocumentFragment();
    for (const row of rows) {
      const record = node("tr");
      const rankCell = node("td", { className: "rank-value", text: integers.format(row.rank) });
      const featureCell = node("th", { text: row.label });
      featureCell.scope = "row";
      record.append(rankCell, featureCell, getMetricCell(row, scale));
      next.append(record);
    }
    body.replaceChildren(next);
  }
  sort.addEventListener("click", () => {
    descending = !descending;
    renderRows();
  });
  renderRows();
  return table;
}

function showTables(cohorts) {
  const scale = Math.max(...cohorts.flatMap((cohort) => cohort.importance.map((row) => row.mean_abs_shap_pp))) || 1;
  const tabs = [];
  const panels = [];
  for (const cohort of cohorts) {
    const tab = node("button", { className: "cohort-tab", text: cohort.label });
    const panel = node("section", { className: "cohort-panel" });
    tab.type = "button";
    tab.id = `tab-${cohort.id}`;
    panel.id = `cohort-${cohort.id}`;
    tab.setAttribute("role", "tab");
    tab.setAttribute("aria-controls", panel.id);
    panel.setAttribute("role", "tabpanel");
    panel.setAttribute("aria-labelledby", tab.id);
    panel.tabIndex = 0;
    panel.dataset.cohort = cohort.id;
    panel.append(getCohortDetails(cohort), getImportanceTable(cohort, scale));
    tabs.push(tab);
    panels.push(panel);
  }

  function select(index, focus = false) {
    tabs.forEach((tab, position) => {
      const active = position === index;
      tab.setAttribute("aria-selected", String(active));
      tab.tabIndex = active ? 0 : -1;
      panels[position].hidden = !active;
    });
    if (focus) tabs[index].focus();
  }

  tabs.forEach((tab, index) => tab.addEventListener("click", () => select(index)));
  tabList.addEventListener("keydown", (event) => {
    const current = tabs.indexOf(event.target);
    if (current === -1) return;
    const next = {
      ArrowLeft: (current + tabs.length - 1) % tabs.length,
      ArrowRight: (current + 1) % tabs.length,
      Home: 0,
      End: tabs.length - 1,
    }[event.key];
    if (!Number.isInteger(next)) return;
    event.preventDefault();
    select(next, true);
  });
  select(0);
  tabList.replaceChildren(...tabs);
  panelList.replaceChildren(...panels);
  tableMessage.hidden = true;
  tabList.hidden = false;
}

async function loadResults() {
  try {
    const response = await fetch("./assets/results.json", { cache: "no-store" });
    if (!response.ok) throw new Error("results unavailable");
    const { cohorts, assets } = readManifest(await response.json());
    showTables(cohorts);
    showFigure(assets);
  } catch {
    tableMessage.dataset.state = "error";
    tableMessage.textContent = "未能读取已发布的结果。请确认结果文件已随网站发布，再刷新页面。";
  }
}

void loadResults();
