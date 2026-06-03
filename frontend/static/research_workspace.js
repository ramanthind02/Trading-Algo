/**
 * Shared research workspace dashboard (feature + portfolio research UIs).
 */
(function (global) {
  "use strict";

  function escapeHtml(value) {
    return String(value).replace(/[&<>]/g, (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[char]));
  }

  function trafficLightClass(light) {
    const normalized = String(light || "").toUpperCase();
    if (normalized === "GREEN") {
      return "traffic-green";
    }
    if (normalized === "YELLOW") {
      return "traffic-yellow";
    }
    if (normalized === "RED") {
      return "traffic-red";
    }
    return "";
  }

  function trafficLightBadge(light) {
    const css = trafficLightClass(light);
    const label = String(light || "—").toUpperCase();
    return `<span class="traffic-badge ${css}">${escapeHtml(label)}</span>`;
  }

  function lineChartSvg(chart) {
    const values = chart.y || [];
    if (!values.length) {
      return "";
    }
    const width = 800;
    const height = 220;
    const padding = 16;
    const min = Math.min(...values);
    const max = Math.max(...values);
    const span = Math.max(max - min, 1e-9);
    const points = values
      .map((value, index) => {
        const x = padding + (index / Math.max(values.length - 1, 1)) * (width - padding * 2);
        const y = height - padding - ((value - min) / span) * (height - padding * 2);
        return `${x.toFixed(1)},${y.toFixed(1)}`;
      })
      .join(" ");
    return `<svg viewBox="0 0 ${width} ${height}" preserveAspectRatio="none" role="img" aria-label="${escapeHtml(chart.label)}"><polyline fill="none" stroke="#38bdf8" stroke-width="2.5" points="${points}"></polyline></svg>`;
  }

  class ResearchWorkspace {
    constructor(options) {
      this.apiPrefix = options.apiPrefix;
      this.state = options.state;
      this.elements = options.elements;
      this.features = options.features || {};
      this.hooks = options.hooks || {};
      this.panelCache = this.state.panelCache || (this.state.panelCache = {});
      this.activeSections = this.state.activeSections || (this.state.activeSections = {});
      this.activeEmbedPanels = this.state.activeEmbedPanels || (this.state.activeEmbedPanels = {});
    }

    artifactRawUrl(relativePath) {
      return `${this.apiPrefix}/artifact-raw?path=${encodeURIComponent(relativePath)}`;
    }

    async loadArtifactPreview(relativePath) {
      if (this.panelCache[relativePath]) {
        return this.panelCache[relativePath];
      }
      const response = await fetch(
        `${this.apiPrefix}/artifact-preview?path=${encodeURIComponent(relativePath)}`
      );
      const payload = await response.json();
      if (response.ok) {
        this.panelCache[relativePath] = payload;
      }
      return payload;
    }

    renderCards(target, cards, options = {}) {
      target.replaceChildren(
        ...cards.map((card) => {
          const wrapper = document.createElement("div");
          const trafficClass =
            options.trafficLights && card.label
              ? trafficLightClass(card.label)
              : "";
          wrapper.className = `metric-card${trafficClass ? ` ${trafficClass}` : ""}`;
          const label = document.createElement("div");
          label.className = "metric-label";
          label.textContent = card.label;
          const value = document.createElement("div");
          value.className = "metric-value";
          value.textContent = card.value;
          wrapper.append(label, value);
          return wrapper;
        })
      );
    }

    renderTablePreview(preview) {
      const chartHtml = preview.chart
        ? `<div class="inline-chart">${lineChartSvg(preview.chart)}</div>`
        : "";
      const rows = preview.rows
        .map(
          (row) =>
            `<tr>${preview.columns.map((column) => `<td>${escapeHtml(row[column] ?? "")}</td>`).join("")}</tr>`
        )
        .join("");
      return `
        <div class="metric-card">
          <div class="metric-label">Rows</div>
          <div class="metric-value">${preview.row_count}</div>
        </div>
        ${chartHtml}
        <div style="overflow:auto;">
          <table>
            <thead><tr>${preview.columns.map((column) => `<th>${escapeHtml(column)}</th>`).join("")}</tr></thead>
            <tbody>${rows}</tbody>
          </table>
        </div>
      `;
    }

    buildEmbedElement(panel) {
      const rawUrl = this.artifactRawUrl(panel.relative_path);
      if (panel.kind === "image") {
        const image = document.createElement("img");
        image.className = "report-image";
        image.src = rawUrl;
        image.alt = panel.title;
        return image;
      }
      const iframe = document.createElement("iframe");
      iframe.className = "embed-viewer";
      iframe.src = rawUrl;
      iframe.title = panel.title;
      return iframe;
    }

    activeEmbedPanelId(phase, sectionId, embedPanels) {
      const stored = this.activeEmbedPanels[`${phase}:${sectionId}`];
      if (stored && embedPanels.some((panel) => panel.panel_id === stored)) {
        return stored;
      }
      return embedPanels[0].panel_id;
    }

    renderEmbedSwitcher(phase, sectionId, embedPanels, host, openLink) {
      const switcher = document.createElement("div");
      switcher.className = "embed-switcher";
      const stage = document.createElement("div");
      stage.className = "embed-stage";
      const renderActive = () => {
        const currentId = this.activeEmbedPanelId(phase, sectionId, embedPanels);
        const panel = embedPanels.find((entry) => entry.panel_id === currentId) || embedPanels[0];
        stage.replaceChildren();
        stage.append(this.buildEmbedElement(panel));
        if (openLink) {
          openLink.href = this.artifactRawUrl(panel.relative_path);
        }
        switcher.querySelectorAll("button").forEach((button) => {
          button.classList.toggle("active", button.dataset.panelId === panel.panel_id);
        });
      };
      embedPanels.forEach((panel) => {
        const button = document.createElement("button");
        button.type = "button";
        button.dataset.panelId = panel.panel_id;
        button.textContent = panel.title;
        button.addEventListener("click", () => {
          this.activeEmbedPanels[`${phase}:${sectionId}`] = panel.panel_id;
          renderActive();
        });
        switcher.append(button);
      });
      renderActive();
      host.append(switcher, stage);
    }

    createReportPanel(panel) {
      const shell = document.createElement("article");
      shell.className = "report-panel";
      const header = document.createElement("div");
      header.className = "report-panel-header";
      header.innerHTML = `
        <h3 class="report-panel-title">${escapeHtml(panel.title)}</h3>
        <a class="preview-link" href="${this.artifactRawUrl(panel.relative_path)}" target="_blank" rel="noreferrer">Open in new tab</a>
      `;
      const body = document.createElement("div");
      body.className = "report-panel-body";
      shell.append(header, body);
      return { shell, body };
    }

    async renderDataPanel(panel, body) {
      body.innerHTML = `<div class="results-empty">Loading ${escapeHtml(panel.title)}…</div>`;
      const preview = await this.loadArtifactPreview(panel.relative_path);
      if (preview.error) {
        body.innerHTML = `<div class="results-empty">${escapeHtml(preview.error)}</div>`;
        return;
      }
      if (preview.kind === "table") {
        body.innerHTML = this.renderTablePreview(preview);
        return;
      }
      if (preview.kind === "json") {
        body.innerHTML = `<pre>${escapeHtml(JSON.stringify(preview.data, null, 2))}</pre>`;
        return;
      }
      body.innerHTML = `<pre>${escapeHtml(preview.text || "")}</pre>`;
    }

    monitoringStatusCards(monitoring) {
      if (!monitoring || typeof monitoring !== "object") {
        return "";
      }
      const calibration = monitoring.reference_calibration || {};
      const evalWindow = monitoring.evaluation_window || {};
      const refWindow = monitoring.reference_window || {};
      const volWindow = monitoring.reference_volatility_window || {};
      const override =
        monitoring.override_weight_fraction === null ||
        monitoring.override_weight_fraction === undefined ||
        monitoring.override_weight_fraction === ""
          ? "—"
          : Number(monitoring.override_weight_fraction).toFixed(2);
      const cards = [
        {
          labelHtml: "Traffic light",
          valueHtml: trafficLightBadge(monitoring.traffic_light),
        },
        {
          label: "Tests failed",
          value: `${monitoring.tests_failed ?? "—"}/4`,
        },
        {
          label: "Advisory weight",
          value: Number(monitoring.advisory_weight_fraction ?? 0).toFixed(2),
        },
        { label: "Override weight", value: override },
        {
          label: "Effective weight",
          value: Number(monitoring.effective_weight_fraction ?? 0).toFixed(2),
        },
        {
          label: "Evaluation window",
          value: `${evalWindow.start ?? "—"} → ${evalWindow.end ?? "—"}`,
        },
        {
          label: "Reference μ window",
          value: `${refWindow.start ?? "—"} → ${refWindow.end ?? "—"}`,
        },
        {
          label: "Reference σ window",
          value: `${volWindow.start ?? "—"} → ${volWindow.end ?? "—"}`,
        },
      ];
      if (calibration.sigma_method) {
        cards.push({
          label: "σ calibration",
          value: `${calibration.sigma_method} (shift ${(Number(calibration.sigma_relative_shift ?? 0) * 100).toFixed(0)}%)`,
        });
      }
      return cards
        .map((card) => {
          const label = card.labelHtml ? card.labelHtml : escapeHtml(card.label);
          const value = card.valueHtml ? card.valueHtml : escapeHtml(card.value);
          return `
        <div class="metric-card">
          <div class="metric-label">${label}</div>
          <div class="metric-value">${value}</div>
        </div>`;
        })
        .join("");
    }

    renderMonitoringRollupTable(rows) {
      if (!rows.length) {
        return `<div class="results-empty">No monitoring rollup rows yet.</div>`;
      }
      const columns = [
        "strategy",
        "traffic_light",
        "tests_failed",
        "advisory_weight_fraction",
        "effective_weight_fraction",
        "traffic_light_prev_month",
        "traffic_light_2mo_ago",
        "eval_start",
        "eval_end",
        "reference_sigma_method",
      ];
      const header = columns.map((column) => `<th>${escapeHtml(column.replace(/_/g, " "))}</th>`).join("");
      const body = rows
        .map((row) => {
          const cells = columns
            .map((column) => {
              const raw = row[column] ?? "";
              if (column === "traffic_light") {
                return `<td>${trafficLightBadge(raw)}</td>`;
              }
              if (column.startsWith("traffic_light_") && raw) {
                return `<td>${trafficLightBadge(raw)}</td>`;
              }
              return `<td>${escapeHtml(raw)}</td>`;
            })
            .join("");
          return `<tr>${cells}</tr>`;
        })
        .join("");
      return `
        <table class="monitoring-rollup-table">
          <thead><tr>${header}</tr></thead>
          <tbody>${body}</tbody>
        </table>
      `;
    }

    renderMonitoringHistoryGrid(histories) {
      if (!histories.length) {
        return "";
      }
      const cards = histories
        .map((entry) => {
          const rows = entry.rows || [];
          const lights = rows.map((row) => String(row.traffic_light || "").toUpperCase());
          const chart =
            lights.length > 0
              ? lineChartSvg({
                  label: `${entry.strategy} traffic light trend`,
                  y: lights.map((light) => (light === "GREEN" ? 0 : light === "YELLOW" ? 1 : 2)),
                })
              : "";
          return `
            <article class="monitoring-history-card">
              <h4>${escapeHtml(entry.strategy)}</h4>
              <p class="section-copy">${rows.length} month-end snapshots</p>
              ${chart ? `<div class="inline-chart">${chart}</div>` : ""}
              <a class="preview-link" href="${this.artifactRawUrl(entry.relative_path)}" target="_blank" rel="noreferrer">Open history CSV</a>
            </article>
          `;
        })
        .join("");
      return `<div class="monitoring-history-grid">${cards}</div>`;
    }

    renderMonitoringDashboard(section, container) {
      const dashboard = section.monitoring_dashboard || {};
      const rows = dashboard.rollup_rows || [];
      const counts = dashboard.traffic_light_counts || {};
      const summaryCards = [
        { label: "Strategies", value: String(dashboard.strategy_count ?? rows.length) },
        { label: "GREEN", value: String(counts.GREEN ?? 0) },
        { label: "YELLOW", value: String(counts.YELLOW ?? 0) },
        { label: "RED", value: String(counts.RED ?? 0) },
      ];
      const summary = document.createElement("div");
      summary.className = "card-grid";
      this.renderCards(summary, summaryCards, { trafficLights: true });

      const tableWrap = document.createElement("div");
      tableWrap.className = "table-panel-body";
      tableWrap.style.marginTop = "16px";
      tableWrap.innerHTML = this.renderMonitoringRollupTable(rows);

      const histories = document.createElement("div");
      histories.innerHTML = `
        <h3 style="margin: 24px 0 8px; font-size: 1rem;">Monthly history</h3>
        <p class="section-copy">Traffic-light trend by strategy (0=Green, 1=Yellow, 2=Red). Full series in each strategy folder.</p>
        ${this.renderMonitoringHistoryGrid(dashboard.histories || [])}
      `;

      const portfolio = dashboard.portfolio_monitoring;
      const portfolioBlock = document.createElement("div");
      if (portfolio && portfolio.status) {
        portfolioBlock.innerHTML = `
          <h3 style="margin: 24px 0 8px; font-size: 1rem;">Portfolio monitoring</h3>
          <div class="card-grid">${this.monitoringStatusCards(portfolio.status)}</div>
        `;
      }

      container.replaceChildren(summary, tableWrap, histories, portfolioBlock);
    }

    async renderMonitoringRollupPanel(panel, body) {
      body.innerHTML = `<div class="results-empty">Loading ${escapeHtml(panel.title)}…</div>`;
      const preview = await this.loadArtifactPreview(panel.relative_path);
      if (preview.error || preview.kind !== "table") {
        body.innerHTML = `<div class="results-empty">${escapeHtml(preview.error || "Rollup unavailable.")}</div>`;
        return;
      }
      body.innerHTML = `<div class="table-panel-body">${this.renderMonitoringRollupTable(preview.rows || [])}</div>`;
    }

    async renderMonitoringHistoryPanel(panel, body) {
      body.innerHTML = `<div class="results-empty">Loading ${escapeHtml(panel.title)}…</div>`;
      const preview = await this.loadArtifactPreview(panel.relative_path);
      if (preview.error || preview.kind !== "table") {
        body.innerHTML = `<div class="results-empty">${escapeHtml(preview.error || "History unavailable.")}</div>`;
        return;
      }
      const strategy = panel.title.split("—")[0]?.trim() || panel.title;
      body.innerHTML = `
        <div class="table-panel-body">
          ${this.renderMonitoringHistoryGrid([
            {
              strategy,
              relative_path: panel.relative_path,
              rows: preview.rows || [],
            },
          ])}
          ${this.renderTablePreview(preview)}
        </div>
      `;
    }

    async renderMonitoringStatusPanel(panel, body) {
      body.innerHTML = `<div class="results-empty">Loading ${escapeHtml(panel.title)}…</div>`;
      const preview = await this.loadArtifactPreview(panel.relative_path);
      if (preview.error || preview.kind !== "json") {
        body.innerHTML = `<div class="results-empty">${escapeHtml(preview.error || "Status unavailable.")}</div>`;
        return;
      }
      body.innerHTML = `
        <div class="table-panel-body">
          <div class="card-grid">${this.monitoringStatusCards(preview.data || {})}</div>
          <details style="margin-top: 16px;">
            <summary class="preview-link" style="cursor: pointer;">Full monitoring status JSON</summary>
            <pre>${escapeHtml(JSON.stringify(preview.data, null, 2))}</pre>
          </details>
        </div>
      `;
    }

    async renderRobustnessSummaryPanel(panel, body) {
      body.innerHTML = `<div class="results-empty">Loading ${escapeHtml(panel.title)}…</div>`;
      const preview = await this.loadArtifactPreview(panel.relative_path);
      if (preview.error || preview.kind !== "json") {
        body.innerHTML = `<div class="results-empty">${escapeHtml(preview.error || "Summary unavailable.")}</div>`;
        return;
      }
      const payload = preview.data || {};
      const monitoring = payload.monitoring || {};
      const sharpe = payload.sharpe_comparison || {};
      const cusum = payload.cusum || {};
      const bands = payload.equity_curve_bands || {};
      const rolling = payload.rolling_sharpe_zscore || {};
      const rank = payload.rank_correlation || {};
      const ciIs = sharpe.ci_is || {};
      const ciVal = sharpe.ci_val || {};
      const holdoutLabel = panel.relative_path.includes("holdout") ? "Holdout" : "Val";
      const monitoringBlock = monitoring.traffic_light
        ? `
          <h3 style="margin: 0 0 8px; font-size: 1rem;">Monitoring status</h3>
          <div class="card-grid">${this.monitoringStatusCards(monitoring)}</div>
        `
        : "";
      const cards = [
        {
          label: "IS Sharpe",
          value: `${Number(sharpe.sr_is ?? 0).toFixed(2)} [${Number(ciIs.lower ?? 0).toFixed(2)}, ${Number(ciIs.upper ?? 0).toFixed(2)}]`,
        },
        {
          label: `${holdoutLabel} Sharpe`,
          value: `${Number(sharpe.sr_val ?? 0).toFixed(2)} [${Number(ciVal.lower ?? 0).toFixed(2)}, ${Number(ciVal.upper ?? 0).toFixed(2)}]`,
        },
        { label: "Degradation ratio", value: Number(sharpe.degradation_ratio ?? 0).toFixed(2) },
        { label: "CI overlap", value: sharpe.ci_overlap ? "Yes" : "No" },
        { label: "CUSUM", value: cusum.passed ? "PASS" : "FAIL" },
        {
          label: "Equity bands",
          value: `${bands.passed ? "PASS" : "FAIL"} (${(Number(bands.fraction_below_lower ?? 0) * 100).toFixed(0)}% below lower)`,
        },
        {
          label: "Rolling SR z-score",
          value: `${rolling.passed ? "PASS" : "FAIL"} (${(Number(rolling.fraction_below_threshold ?? 0) * 100).toFixed(0)}% below z)`,
        },
        { label: "Rank correlation ρ", value: Number(rank.spearman_rho ?? 0).toFixed(2) },
        { label: "Legacy overall", value: payload.all_passed ? "PASS" : "FAIL" },
      ];
      const cardHtml = cards
        .map(
          (card) => `
        <div class="metric-card">
          <div class="metric-label">${escapeHtml(card.label)}</div>
          <div class="metric-value">${escapeHtml(card.value)}</div>
        </div>`
        )
        .join("");
      body.innerHTML = `
        <div class="table-panel-body">
          ${monitoringBlock}
          <h3 style="margin: 16px 0 8px; font-size: 1rem;">Test detail</h3>
          <div class="card-grid">${cardHtml}</div>
          <p style="margin-top: 16px; color: #94a3b8;">${escapeHtml(payload.interpretation || "")}</p>
          <details style="margin-top: 16px;">
            <summary class="preview-link" style="cursor: pointer;">Full robustness JSON</summary>
            <pre>${escapeHtml(JSON.stringify(payload, null, 2))}</pre>
          </details>
        </div>
      `;
    }

    async renderPortfolioHoldoutSummaryPanel(panel, body) {
      body.innerHTML = `<div class="results-empty">Loading ${escapeHtml(panel.title)}…</div>`;
      const preview = await this.loadArtifactPreview(panel.relative_path);
      if (preview.error || preview.kind !== "json") {
        body.innerHTML = `<div class="results-empty">${escapeHtml(preview.error || "Summary unavailable.")}</div>`;
        return;
      }
      const payload = preview.data || {};
      const corr = payload.correlation_realisation || {};
      const idm = payload.idm_improvement || {};
      const cards = [
        {
          label: "Overall",
          value: payload.passed ? "PASS" : "FAIL",
        },
        {
          label: "Mean pairwise corr",
          value: Number(corr.mean_pairwise_corr ?? 0).toFixed(2),
        },
        {
          label: "Δ IDM",
          value: `${Number(idm.delta_idm ?? 0) >= 0 ? "+" : ""}${Number(idm.delta_idm ?? 0).toFixed(2)}`,
        },
      ];
      const cardHtml = cards
        .map(
          (card) => `
        <div class="metric-card">
          <div class="metric-label">${escapeHtml(card.label)}</div>
          <div class="metric-value">${escapeHtml(card.value)}</div>
        </div>`
        )
        .join("");
      body.innerHTML = `
        <div class="table-panel-body">
          <div class="card-grid">${cardHtml}</div>
          <details style="margin-top: 16px;">
            <summary class="preview-link" style="cursor: pointer;">Full portfolio holdout JSON</summary>
            <pre>${escapeHtml(JSON.stringify(payload, null, 2))}</pre>
          </details>
        </div>
      `;
    }

    formatWeightPct(value) {
      return `${(Number(value || 0) * 100).toFixed(1)}%`;
    }

    formatWeightDelta(value) {
      const n = Number(value || 0);
      const sign = n >= 0 ? "+" : "";
      return `${sign}${(n * 100).toFixed(1)}%`;
    }

    weightLayerMethodLabel(context, meta) {
      const detail = context.weight_layer_detail || {};
      if (detail.weight_layer_policy) {
        return String(detail.weight_layer_policy);
      }
      if (context.weight_layer_policy) {
        return String(context.weight_layer_policy);
      }
      const method = String(context.weight_layer_method || meta.weight_layer_method || "unknown");
      if (context.weight_layer_hierarchy_mode === "asset_first" || method === "hierarchy_equal") {
        return "hierarchy_equal (asset-first)";
      }
      return method;
    }

    buildWeightBudgetBarsHtml(detail, snapshotMode = false) {
      const after = detail.asset_budgets_with || detail.asset_budgets_without || {};
      const before = snapshotMode ? after : detail.asset_budgets_without || {};
      const keys = [...new Set([...Object.keys(before), ...Object.keys(after)])].sort();
      if (!keys.length) {
        return "";
      }
      const rows = keys
        .map((key) => {
          const b = Number(before[key] || 0);
          const a = Number(after[key] || 0);
          const max = Math.max(b, a, 0.001);
          const label = snapshotMode
            ? `${this.formatWeightPct(a)}`
            : `${this.formatWeightPct(b)} → ${this.formatWeightPct(a)} (${this.formatWeightDelta(a - b)})`;
          return `
          <div class="wl-budget-row">
            <div class="wl-budget-label">
              <span>${escapeHtml(key)}</span>
              <span>${label}</span>
            </div>
            <div class="wl-budget-track">
              ${snapshotMode ? "" : `<div class="wl-bar-before" title="Before"><span style="width:${(b / max) * 100}%"></span></div>`}
              <div class="wl-bar-after" title="${snapshotMode ? "Weight" : "After"}"><span style="width:${(a / max) * 100}%"></span></div>
            </div>
          </div>`;
        })
        .join("");
      return `
        <div class="wl-budget-bars">
          <div class="wl-budget-label" style="margin-bottom: 4px;">
            <span>Asset-class budget (global stream weights)</span>
            <span style="color:#64748b;">${snapshotMode ? "blue = fitted weight" : "gray = before · blue = after"}</span>
          </div>
          ${rows}
        </div>`;
    }

    buildWeightTreeNodeHtml(node, snapshotMode = false) {
      if (!node || typeof node !== "object") {
        return "";
      }
      const delta = Number(node.weight_delta || 0);
      const weight = Number(node.weight_with || node.weight_without || 0);
      const weightLine = snapshotMode
        ? `<span class="wl-tree-weights">${this.formatWeightPct(weight)}</span>`
        : `<span class="wl-tree-weights">${this.formatWeightPct(node.weight_without)} → ${this.formatWeightPct(node.weight_with)} (${this.formatWeightDelta(delta)})</span>`;
      const type = String(node.type || "");
      if (type === "stream") {
        const label = String(node.model_name || node.stream_id || "stream");
        const ticker = node.ticker ? `${node.ticker} · ` : "";
        return `
          <li class="wl-tree-node">
            <div class="wl-tree-head">
              <span class="wl-tree-id">${escapeHtml(ticker + label)}</span>
              ${weightLine}
            </div>
          </li>`;
      }
      const id = String(node.id || type);
      const count = node.stream_count != null ? ` · ${node.stream_count} stream(s)` : "";
      const children = Array.isArray(node.children) ? node.children : [];
      const childHtml = children.length
        ? `<ul>${children.map((child) => this.buildWeightTreeNodeHtml(child, snapshotMode)).join("")}</ul>`
        : "";
      return `
        <li class="wl-tree-node">
          <div class="wl-tree-head">
            <span class="wl-tree-id">${escapeHtml(id)}${escapeHtml(count)}</span>
            ${weightLine}
          </div>
          ${childHtml}
        </li>`;
    }

    buildWeightLayerStructureHtml(context, meta) {
      const detail = context.weight_layer_detail;
      if (!detail || typeof detail !== "object") {
        return "";
      }
      const snapshotMode = Boolean(detail.snapshot_mode || meta.snapshot_mode);
      const method = this.weightLayerMethodLabel(context, meta);
      const streamCount = detail.stream_count_with ?? detail.stream_count_without ?? "—";
      const metaItems = snapshotMode
        ? [
            ["Method", method],
            ["FDM", `${detail.fdm_with ?? "—"} (cap ${detail.fdm_max ?? "—"})`],
            ["Global streams", String(streamCount)],
            ["Ensembles", String(detail.ensemble_count_with ?? meta.ensemble_count ?? "—")],
          ]
        : [
            ["Method", method],
            ["FDM (before → after)", `${detail.fdm_without ?? "—"} → ${detail.fdm_with ?? "—"} (cap ${detail.fdm_max ?? "—"})`],
            ["Global streams", `${detail.stream_count_without ?? "—"} → ${detail.stream_count_with ?? "—"}`],
            ["Ensembles", `${detail.ensemble_count_without ?? "—"} → ${detail.ensemble_count_with ?? "—"}`],
          ];
      const metaGrid = metaItems
        .map(
          ([label, value]) => `
        <div class="wl-meta-item"><strong>${escapeHtml(label)}</strong>${escapeHtml(String(value))}</div>`
        )
        .join("");
      const tree = Array.isArray(detail.tree) ? detail.tree : [];
      const treeHtml = tree.length
        ? `<ul class="wl-tree">${tree.map((node) => this.buildWeightTreeNodeHtml(node, snapshotMode)).join("")}</ul>`
        : `<p style="color: var(--muted); margin: 0;">No hierarchy tree in report.</p>`;
      const outline = Array.isArray(detail.hierarchy_outline_with) ? detail.hierarchy_outline_with : [];
      const outlineHtml =
        outline.length && detail.hierarchy_mode === "asset_first"
          ? `<details style="margin-top: 12px;">
            <summary class="preview-link" style="cursor: pointer;">Hierarchy outline</summary>
            <table class="artifact-table" style="margin-top: 8px; width: 100%; font-size: 0.85rem;">
              <thead><tr><th>Path</th><th>Kind</th><th>Leaves</th></tr></thead>
              <tbody>${outline
                .filter((row) => row.kind === "group")
                .map(
                  (row) => `
                <tr>
                  <td><code>${escapeHtml(String(row.path || ""))}</code></td>
                  <td>group</td>
                  <td>${Number(row.child_leaves || 0)}</td>
                </tr>`
                )
                .join("")}</tbody>
            </table>
          </details>`
          : "";
      const intro = snapshotMode
        ? "Fitted portfolio weights from the global WeightLayer (train fit, scored on the phase window). SR tilt and within-group policy come from portfolio_research/config.py."
        : "Three-level asset-first tree: root → asset class → vault style group → instrument stream. Sibling groups split weight equally at each level before SR / inv-corr adjustments.";
      return `
        <div class="wl-section">
          <h4>Weight layer structure</h4>
          <p style="color: var(--muted); font-size: 0.9rem; margin: 0 0 12px;">${escapeHtml(intro)}</p>
          <div class="wl-meta-grid">${metaGrid}</div>
          ${this.buildWeightBudgetBarsHtml(detail, snapshotMode)}
          ${treeHtml}
          ${outlineHtml}
        </div>`;
    }

    async renderWeightLayerReportPanel(panel, body) {
      body.innerHTML = `<div class="results-empty">Loading ${escapeHtml(panel.title)}…</div>`;
      const preview = await this.loadArtifactPreview(panel.relative_path);
      if (preview.error || preview.kind !== "json") {
        body.innerHTML = `<div class="results-empty">${escapeHtml(preview.error || "Weight layer report unavailable.")}</div>`;
        return;
      }
      const payload = preview.data || {};
      const meta = payload.meta || {};
      const context = payload.context || {};
      const phase = meta.phase ? `${meta.phase}` : panel.title;
      const windowLabel =
        meta.fit_start && meta.predict_end
          ? `Fit ${meta.fit_start} → ${meta.fit_end} · Score ${meta.predict_start} → ${meta.predict_end}`
          : "";
      body.innerHTML = `
        <div class="table-panel-body">
          ${windowLabel ? `<p class="raw-data-banner">${escapeHtml(windowLabel)}</p>` : ""}
          ${this.buildWeightLayerStructureHtml(context, meta)}
          <details style="margin-top: 16px;">
            <summary class="preview-link" style="cursor: pointer;">Full weight layer JSON (${escapeHtml(phase)})</summary>
            <pre>${escapeHtml(JSON.stringify(payload, null, 2))}</pre>
          </details>
        </div>
      `;
    }

    resolvePanelRenderer(panel) {
      if (this.hooks.resolvePanelRenderer) {
        const custom = this.hooks.resolvePanelRenderer(panel, this);
        if (custom) {
          return custom;
        }
      }
      const path = panel.relative_path || "";
      if (panel.view_type === "monitoring_rollup" || path.endsWith("monitoring_rollup.csv")) {
        return (p, b) => this.renderMonitoringRollupPanel(p, b);
      }
      if (path.endsWith("monitoring_history.csv")) {
        return (p, b) => this.renderMonitoringHistoryPanel(p, b);
      }
      if (path.endsWith("monitoring_status.json")) {
        return (p, b) => this.renderMonitoringStatusPanel(p, b);
      }
      if (path.endsWith("holdout_robustness_report.json") || path.endsWith("validation_robustness_report.json")) {
        return (p, b) => this.renderRobustnessSummaryPanel(p, b);
      }
      if (path.endsWith("portfolio_holdout_report.json")) {
        return (p, b) => this.renderPortfolioHoldoutSummaryPanel(p, b);
      }
      if (path.endsWith("weight_layer_report.json")) {
        return (p, b) => this.renderWeightLayerReportPanel(p, b);
      }
      if (path.endsWith("robustness_report.json")) {
        return (p, b) => this.renderRobustnessSummaryPanel(p, b);
      }
      return (p, b) => this.renderDataPanel(p, b);
    }

    activePhaseView() {
      const workspace = this.state.workspace;
      if (!workspace || !workspace.phases) {
        return null;
      }
      return (
        workspace.phases.find((phaseView) => phaseView.phase === this.state.selectedPhase) ||
        workspace.phases[0]
      );
    }

    ensureActiveSection(phaseView) {
      if (!phaseView.report_sections.length) {
        return null;
      }
      const current = this.activeSections[phaseView.phase];
      const valid = phaseView.report_sections.some((section) => section.id === current);
      if (!valid) {
        this.activeSections[phaseView.phase] =
          phaseView.default_section_id || phaseView.report_sections[0].id;
      }
      return this.activeSections[phaseView.phase];
    }

    async renderSectionPanels(phase, section, container) {
      container.replaceChildren();
      const phaseView = this.activePhaseView();
      const isRawSection = Boolean(section.is_raw_data);
      if (section.id === "monitoring_status" && section.monitoring_dashboard) {
        this.renderMonitoringDashboard(section, container);
        return;
      }
      if (this.hooks.renderSectionExtras) {
        const handled = await this.hooks.renderSectionExtras(phase, section, container, phaseView, this);
        if (handled) {
          return;
        }
      }
      if (!section.panels.length) {
        container.innerHTML = `<div class="results-empty">No outputs in this section yet.</div>`;
        return;
      }
      const embedPanels = isRawSection ? [] : section.panels.filter((panel) => panel.view_type === "embed");
      const summaryPanels = isRawSection ? section.panels : section.panels.filter((panel) => panel.view_type !== "embed");
      const stack = document.createElement("div");
      stack.className = "panel-stack";
      if (isRawSection) {
        const banner = document.createElement("p");
        banner.className = "raw-data-banner";
        banner.textContent =
          "Tabular exports and JSON dumps live here so plots and summaries stay readable above.";
        stack.append(banner);
      }
      if (embedPanels.length === 1) {
        const { shell, body } = this.createReportPanel(embedPanels[0]);
        body.append(this.buildEmbedElement(embedPanels[0]));
        stack.append(shell);
      } else if (embedPanels.length > 1) {
        const { shell, body } = this.createReportPanel({
          title: section.label,
          relative_path: embedPanels[0].relative_path,
        });
        const openLink = shell.querySelector(".preview-link");
        this.renderEmbedSwitcher(phase, section.id, embedPanels, body, openLink);
        stack.append(shell);
      }
      summaryPanels.forEach((panel) => {
        const { shell, body } = this.createReportPanel(panel);
        body.classList.add(panel.view_type === "table" ? "table-panel-body" : "text-panel-body");
        stack.append(shell);
        const renderPanel = this.resolvePanelRenderer(panel);
        renderPanel(panel, body).catch(() => {
          body.innerHTML = `<div class="results-empty">Unable to load ${escapeHtml(panel.title)}.</div>`;
        });
      });
      if (!stack.children.length) {
        stack.innerHTML = `<div class="results-empty">No primary outputs in this section yet. Check Raw data for CSV/JSON exports.</div>`;
      }
      container.append(stack);
    }

    renderPhaseSummary(phaseView) {
      if (phaseView.phase !== this.state.selectedPhase) {
        this.elements.phaseSummaryCards.replaceChildren();
        return;
      }
      const cards = phaseView.summary_cards || [];
      const trafficLights = cards.some((card) =>
        ["GREEN", "YELLOW", "RED"].includes(String(card.label || "").toUpperCase())
      );
      this.renderCards(this.elements.phaseSummaryCards, cards, { trafficLights });
    }

    renderSectionContent(phaseView) {
      const dashboard = this.elements.phaseDashboards.querySelector(`[data-phase="${phaseView.phase}"]`);
      if (!dashboard) {
        return;
      }
      const sectionId = this.ensureActiveSection(phaseView);
      const section = phaseView.report_sections.find((entry) => entry.id === sectionId);
      const intro = dashboard.querySelector(".section-intro");
      const content = dashboard.querySelector(".section-content");
      const nav = dashboard.querySelector(".section-nav");
      nav.querySelectorAll("button").forEach((button) => {
        button.classList.toggle("active", button.dataset.sectionId === sectionId);
      });
      if (!section) {
        intro.textContent = "Run this phase to populate the dashboard.";
        content.innerHTML = `<div class="results-empty">No ${escapeHtml(phaseView.label)} results yet. Run the phase, then refresh.</div>`;
        return;
      }
      intro.textContent = section.description;
      this.renderSectionPanels(phaseView.phase, section, content).catch(() => {
        content.innerHTML = `<div class="results-empty">Unable to render ${escapeHtml(section.label)}.</div>`;
      });
    }

    selectPhase(phase) {
      this.state.selectedPhase = phase;
      this.elements.phaseTabs.querySelectorAll(".tab-button").forEach((button) => {
        button.classList.toggle("active", button.dataset.phase === phase);
      });
      this.elements.phaseDashboards.querySelectorAll(".phase-dashboard").forEach((dashboard) => {
        dashboard.classList.toggle("active", dashboard.dataset.phase === phase);
      });
      const phaseView = this.activePhaseView();
      if (phaseView) {
        this.renderPhaseSummary(phaseView);
        this.renderSectionContent(phaseView);
      }
    }

    focusResultsDashboard(phase) {
      this.selectPhase(phase);
      this.elements.resultsDashboard.scrollIntoView({ behavior: "smooth", block: "start" });
    }

    renderWorkspace() {
      const workspace = this.state.workspace;
      const { phaseTabs, phaseDashboards, phaseSummaryCards } = this.elements;
      phaseTabs.replaceChildren();
      phaseDashboards.replaceChildren();
      if (!workspace || !workspace.phases) {
        phaseSummaryCards.replaceChildren();
        if (this.hooks.afterRenderWorkspace) {
          this.hooks.afterRenderWorkspace(workspace);
        }
        return;
      }
      workspace.phases.forEach((phaseView, index) => {
        if (phaseView.phase === this.state.selectedPhase || (!this.state.selectedPhase && index === 0)) {
          if (!this.state.selectedPhase) {
            this.state.selectedPhase = phaseView.phase;
          }
        }
        const tab = document.createElement("button");
        tab.className = `tab-button${phaseView.phase === this.state.selectedPhase ? " active" : ""}`;
        tab.textContent = phaseView.label;
        tab.type = "button";
        tab.dataset.phase = phaseView.phase;
        tab.addEventListener("click", () => this.selectPhase(phaseView.phase));
        phaseTabs.append(tab);
        const dashboard = document.createElement("section");
        dashboard.className = `phase-dashboard${phaseView.phase === this.state.selectedPhase ? " active" : ""}`;
        dashboard.dataset.phase = phaseView.phase;
        const nav = document.createElement("div");
        nav.className = "section-nav";
        if (!phaseView.report_sections.length) {
          nav.innerHTML = `<span class="section-intro">No sections yet — run ${escapeHtml(phaseView.label)} first.</span>`;
        } else {
          this.ensureActiveSection(phaseView);
          phaseView.report_sections.forEach((section) => {
            const button = document.createElement("button");
            button.type = "button";
            const rawCount = section.raw_panel_count ?? section.panels?.length ?? 0;
            button.textContent =
              section.is_raw_data && rawCount ? `Raw data (${rawCount})` : section.label;
            button.dataset.sectionId = section.id;
            if (section.is_raw_data) {
              button.classList.add("raw-data-tab");
            }
            button.classList.toggle("active", section.id === this.activeSections[phaseView.phase]);
            button.addEventListener("click", () => {
              this.activeSections[phaseView.phase] = section.id;
              this.renderSectionContent(phaseView);
            });
            nav.append(button);
          });
        }
        const intro = document.createElement("p");
        intro.className = "section-intro";
        const content = document.createElement("div");
        content.className = "section-content";
        dashboard.append(nav, intro, content);
        phaseDashboards.append(dashboard);
      });
      const phaseView = this.activePhaseView();
      if (phaseView) {
        this.renderPhaseSummary(phaseView);
        this.renderSectionContent(phaseView);
      }
      if (this.hooks.afterRenderWorkspace) {
        this.hooks.afterRenderWorkspace(workspace);
      }
    }

    applyCompletedPhase(completedPhase) {
      if (!completedPhase) {
        return;
      }
      this.panelCache = {};
      this.state.panelCache = this.panelCache;
      const phaseView = this.state.workspace?.phases?.find((entry) => entry.phase === completedPhase);
      if (phaseView?.default_section_id) {
        this.activeSections[completedPhase] = phaseView.default_section_id;
      }
      this.state.selectedPhase = completedPhase;
      this.renderWorkspace();
      this.focusResultsDashboard(completedPhase);
    }
  }

  global.ResearchWorkspace = ResearchWorkspace;
  global.researchWorkspaceEscapeHtml = escapeHtml;
})(window);
