import { Group, SegmentedControl, Stack, Text } from "@mantine/core";
import { useState } from "react";

import type { PlateauPoint } from "../api/types";
import { Plot, baseLayout, plotConfig } from "./Plot";

// ── palette ─────────────────────────────────────────────────────────────────
// Teal accent for best; muted indigo for the rest; matches Mantine dark theme.
const COLOR_BEST = "#20c997";   // teal accent
const COLOR_REST = "#4c5e9e";   // muted indigo brand

// Perceptually-uniform, blue-green diverging scale that reads well on dark bg.
// "RdYlGn" is a standard Plotly named scale; works out of the box.
const HEATMAP_COLORSCALE = "RdYlGn";

// ── label map ────────────────────────────────────────────────────────────────
const METRIC_LABEL: Record<string, string> = {
  t_stat: "t-stat",
  sharpe: "Sharpe",
  sortino: "Sortino",
  nw_sharpe: "NW Sharpe",
};

// ── helpers ──────────────────────────────────────────────────────────────────
function asNumber(v: number | string): number {
  const n = typeof v === "number" ? v : Number(v);
  return Number.isNaN(n) ? 0 : n;
}

function uniqueSorted(values: (number | string)[]): (number | string)[] {
  const seen = new Map<string, number | string>();
  for (const v of values) seen.set(String(v), v);
  return [...seen.values()].sort((a, b) => asNumber(a) - asNumber(b));
}

/** Human-readable param string for hover annotations. */
function paramSummary(params: Record<string, number | string>): string {
  return Object.entries(params)
    .map(([k, v]) => `${k}=${v}`)
    .join(", ");
}

// ── caption ──────────────────────────────────────────────────────────────────
function Caption({ swept, points }: { swept: string[]; points: PlateauPoint[] }) {
  if (points.length === 0) {
    return (
      <Text size="xs" c="dimmed" fs="italic">
        No data — run a parameter sweep first.
      </Text>
    );
  }
  if (swept.length === 1) {
    return (
      <Text size="xs" c="dimmed">
        Each bar is one value of <b>{swept[0]}</b>. A <em>broad plateau</em> means the strategy
        is robust; a lone spike suggests the best result is a fluke of curve-fitting.
      </Text>
    );
  }
  if (swept.length === 2) {
    return (
      <Text size="xs" c="dimmed">
        Heat map of <b>{swept[0]}</b> × <b>{swept[1]}</b>. A smooth high-value region is
        robust; an isolated hot cell is fragile — avoid strategies that depend on a single
        parameter combination.
      </Text>
    );
  }
  // 0 or >2 params
  return (
    <Text size="xs" c="dimmed">
      Ranked bar of all parameter combinations (no 2-D grid to map). The teal bar is the
      selected best combo; look for a cluster of good results rather than a single outlier.
    </Text>
  );
}

// ── component ────────────────────────────────────────────────────────────────
export function PlateauChart({
  swept,
  points,
  metricOptions,
}: {
  swept: string[];
  points: PlateauPoint[];
  metricOptions: string[];
}) {
  const [metric, setMetric] = useState(metricOptions[0] ?? "t_stat");
  const metricLabel = METRIC_LABEL[metric] ?? metric;
  const value = (p: PlateauPoint): number | null => p.metrics[metric] ?? null;

  // ── empty state ─────────────────────────────────────────────────────────
  if (points.length === 0) {
    return (
      <Stack gap="xs">
        <Caption swept={swept} points={points} />
      </Stack>
    );
  }

  // ── 1-D bar ─────────────────────────────────────────────────────────────
  let chart: React.ReactNode;

  if (swept.length === 1) {
    const key = swept[0];
    const sorted = [...points].sort(
      (a, b) => asNumber(a.params[key]) - asNumber(b.params[key]),
    );
    const xs = sorted.map((p) => String(p.params[key]));
    const ys = sorted.map(value);
    const colors = sorted.map((p) => (p.is_best ? COLOR_BEST : COLOR_REST));
    const hovertext = sorted.map(
      (p) =>
        `<b>${metricLabel}:</b> ${(value(p) ?? 0).toFixed(3)}<br>${paramSummary(p.params)}${p.is_best ? "<br><i>★ best combo</i>" : ""}`,
    );

    chart = (
      <Plot
        data={[
          {
            type: "bar",
            x: xs,
            y: ys,
            marker: { color: colors, line: { color: "rgba(0,0,0,0)", width: 0 } },
            hovertext,
            hoverinfo: "text",
            hoverlabel: { bgcolor: "#1a1b1e", font: { color: "#c1c2c5" } },
          },
        ]}
        layout={{
          ...baseLayout,
          height: 320,
          bargap: 0.25,
          xaxis: {
            ...baseLayout.xaxis,
            title: { text: key },
            tickfont: { size: 11 },
          },
          yaxis: {
            ...baseLayout.yaxis,
            title: { text: metricLabel },
            tickfont: { size: 11 },
          },
        }}
        config={plotConfig}
        style={{ width: "100%" }}
      />
    );

  // ── 2-D heatmap ─────────────────────────────────────────────────────────
  } else if (swept.length === 2) {
    const [kx, ky] = swept;
    const xs = uniqueSorted(points.map((p) => p.params[kx]));
    const ys = uniqueSorted(points.map((p) => p.params[ky]));

    const z: (number | null)[][] = ys.map((yv) =>
      xs.map((xv) => {
        const p = points.find(
          (pp) =>
            String(pp.params[kx]) === String(xv) &&
            String(pp.params[ky]) === String(yv),
        );
        return p ? value(p) : null;
      }),
    );

    // Flat list for annotation text (show value in each cell when grid is small enough).
    const totalCells = xs.length * ys.length;
    const showAnnotations = totalCells <= 100;

    const annotations: Partial<Plotly.Annotations>[] = [];
    if (showAnnotations) {
      ys.forEach((yv, yi) => {
        xs.forEach((xv, xi) => {
          const v = z[yi][xi];
          if (v !== null) {
            annotations.push({
              x: String(xv),
              y: String(yv),
              text: v.toFixed(2),
              showarrow: false,
              font: { size: 9, color: "#ffffff" },
              xref: "x",
              yref: "y",
            });
          }
        });
      });
    }

    // Build hover text matching z dimensions.
    const hovertext: string[][] = ys.map((yv) =>
      xs.map((xv) => {
        const p = points.find(
          (pp) =>
            String(pp.params[kx]) === String(xv) &&
            String(pp.params[ky]) === String(yv),
        );
        if (!p) return "";
        return `<b>${kx}:</b> ${xv}<br><b>${ky}:</b> ${yv}<br><b>${metricLabel}:</b> ${(value(p) ?? 0).toFixed(3)}${p.is_best ? "<br><i>★ best combo</i>" : ""}`;
      }),
    );

    chart = (
      <Plot
        data={[
          {
            type: "heatmap",
            x: xs.map(String),
            y: ys.map(String),
            z,
            colorscale: HEATMAP_COLORSCALE,
            colorbar: {
              title: { text: metricLabel, side: "right" },
              thickness: 14,
              len: 0.9,
              tickfont: { size: 10 },
            },
            hoverongaps: false,
            // heatmap hovertext is 2-D at runtime; the plotly type is too narrow.
            hovertext: hovertext as unknown as string[],
            hoverinfo: "text",
            hoverlabel: { bgcolor: "#1a1b1e", font: { color: "#c1c2c5" } },
          },
        ]}
        layout={{
          ...baseLayout,
          height: 380,
          margin: { ...baseLayout.margin, r: 80 }, // room for colorbar
          xaxis: {
            ...baseLayout.xaxis,
            title: { text: kx },
            tickfont: { size: 11 },
          },
          yaxis: {
            ...baseLayout.yaxis,
            title: { text: ky },
            tickfont: { size: 11 },
          },
          annotations: showAnnotations ? annotations : [],
        }}
        config={plotConfig}
        style={{ width: "100%" }}
      />
    );

  // ── fallback: ranked bar across all combos ───────────────────────────────
  } else {
    const sorted = [...points].sort((a, b) => (value(b) ?? 0) - (value(a) ?? 0));
    const colors = sorted.map((p) => (p.is_best ? COLOR_BEST : COLOR_REST));
    const hovertext = sorted.map(
      (p) =>
        `<b>${p.label}</b><br><b>${metricLabel}:</b> ${(value(p) ?? 0).toFixed(3)}<br>${paramSummary(p.params)}${p.is_best ? "<br><i>★ best combo</i>" : ""}`,
    );

    chart = (
      <Plot
        data={[
          {
            type: "bar",
            x: sorted.map((p) => p.label),
            y: sorted.map(value),
            marker: { color: colors, line: { color: "rgba(0,0,0,0)", width: 0 } },
            hovertext,
            hoverinfo: "text",
            hoverlabel: { bgcolor: "#1a1b1e", font: { color: "#c1c2c5" } },
          },
        ]}
        layout={{
          ...baseLayout,
          height: 340,
          bargap: 0.2,
          xaxis: {
            ...baseLayout.xaxis,
            tickangle: -35,
            tickfont: { size: 10 },
          },
          yaxis: {
            ...baseLayout.yaxis,
            title: { text: metricLabel },
            tickfont: { size: 11 },
          },
        }}
        config={plotConfig}
        style={{ width: "100%" }}
      />
    );
  }

  // ── layout ───────────────────────────────────────────────────────────────
  return (
    <Stack gap={4}>
      <Group justify="space-between" align="flex-start" wrap="nowrap">
        <Caption swept={swept} points={points} />
        <SegmentedControl
          size="xs"
          value={metric}
          onChange={setMetric}
          data={metricOptions.map((m) => ({ value: m, label: METRIC_LABEL[m] ?? m }))}
          style={{ flexShrink: 0 }}
        />
      </Group>
      {chart}
    </Stack>
  );
}
