/**
 * WeightLayerViz — visualizes the portfolio weight layer hierarchy as a Plotly treemap.
 *
 * Props:
 *   data: WeightLayerData  (comes from the API, no fetching here)
 *
 * Sections:
 *   1. Header  — policy badge + SR-knob chips
 *   2. Phase selector — SegmentedControl over available phases
 *   3. Treemap — Plotly treemap of weight distribution (or hierarchy skeleton pre-run)
 *   4. Empty state — shown when weights are unavailable, still renders the hierarchy structure
 */

import { Badge, Card, Chip, Group, SegmentedControl, Stack, Text, Center, ThemeIcon } from "@mantine/core";
import { IconChartTreemap } from "@tabler/icons-react";
import { useMemo, useState } from "react";

import type { WeightLayerData, HierarchyNode, WeightRow } from "../api/types";
import { Plot, baseLayout, plotConfig } from "./Plot";
import { fmtNum, fmtPct } from "../lib/format";

// ── helpers ───────────────────────────────────────────────────────────────────

/** Split a stream_id like "rsi_signal_D_lookback_14__ES__my_model" into a readable label. */
function shortLabel(row: WeightRow): string {
  // Prefer ticker + timeframe + a 2-token model hint from stream_id
  const ticker = row.ticker ?? "";
  const tf = row.timeframe ?? "";

  // Extract first two underscore-separated tokens from stream_id as model hint
  const parts = row.stream_id.split("__");
  // stream_ids often look like "<node_id>__<ticker>__<label>" or just "<node_id>"
  const hint = parts[0]
    .split("_")
    .slice(0, 3) // e.g. rsi_signal_D
    .join("_");

  const segments = [ticker, tf, hint].filter(Boolean);
  return segments.length > 0 ? segments.join(" · ") : row.stream_id.slice(0, 30);
}

/** Derive a short label for a path segment (the last component). */
function segmentLabel(segment: string): string {
  return segment
    .replace(/_/g, " ")
    .replace(/\b\w/g, (c) => c.toUpperCase());
}

// ── treemap data builders ─────────────────────────────────────────────────────

interface TreemapNode {
  id: string;
  parent: string;
  label: string;
  value: number;
  hoverText: string;
}

/**
 * Build flat treemap node list from WeightRow[], grouping by path segments.
 * Path format: "root/group/subgroup/.../stream_id"
 * Leaves get weight value; internal nodes get 0 (branchvalues:"total" sums them).
 */
function buildTreemapFromRows(rows: WeightRow[]): TreemapNode[] {
  const nodeMap = new Map<string, TreemapNode>();
  const childSums = new Map<string, number>();

  // Ensure root exists
  nodeMap.set("root", { id: "root", parent: "", label: "Portfolio", value: 0, hoverText: "Portfolio" });

  for (const row of rows) {
    if (row.weight === null) continue;

    const segments = row.path.split("/").filter(Boolean);
    // segments[0] is "root"; build intermediate group nodes
    for (let depth = 1; depth < segments.length - 1; depth++) {
      const nodeId = segments.slice(0, depth + 1).join("/");
      const parentId = segments.slice(0, depth).join("/");
      if (!nodeMap.has(nodeId)) {
        nodeMap.set(nodeId, {
          id: nodeId,
          parent: parentId,
          label: segmentLabel(segments[depth]),
          value: 0,
          hoverText: segmentLabel(segments[depth]),
        });
      }
    }

    // Leaf node — use the stream_id as unique id to avoid collisions
    const leafId = row.path + "__" + row.stream_id;
    const parentId = segments.slice(0, -1).join("/");
    const label = shortLabel(row);
    const pct = fmtPct(row.weight);
    const corrStr = row.mean_signal_corr !== null ? fmtNum(row.mean_signal_corr, 3) : "—";
    const fdmStr = row.fdm !== null ? fmtNum(row.fdm, 2) : "—";

    nodeMap.set(leafId, {
      id: leafId,
      parent: parentId,
      label,
      value: row.weight,
      hoverText: [
        `<b>${label}</b>`,
        `Weight: ${pct}`,
        `FDM: ${fdmStr}`,
        `Mean corr: ${corrStr}`,
        `Stream: ${row.stream_id}`,
      ].join("<br>"),
    });

    // Accumulate parent sums for display
    childSums.set(parentId, (childSums.get(parentId) ?? 0) + row.weight);
  }

  // Annotate group nodes with their aggregated weight for hover
  for (const [nodeId, node] of nodeMap.entries()) {
    if (node.value === 0 && nodeId !== "root") {
      const sum = childSums.get(nodeId) ?? 0;
      node.hoverText = `<b>${node.label}</b><br>Group weight: ${fmtPct(sum)}`;
    }
  }

  // Root hover
  const rootNode = nodeMap.get("root");
  if (rootNode) {
    rootNode.hoverText = "<b>Portfolio</b><br>All streams";
  }

  return Array.from(nodeMap.values());
}

/**
 * Build treemap from a HierarchyNode tree (pre-run: equal sizing per leaf).
 */
function buildTreemapFromHierarchy(root: HierarchyNode): TreemapNode[] {
  const nodes: TreemapNode[] = [];

  // Count leaves for equal sizing
  function countLeaves(node: HierarchyNode): number {
    if (node.type === "leaf") return 1;
    return (node.children ?? []).reduce((sum, c) => sum + countLeaves(c), 0);
  }
  const totalLeaves = Math.max(countLeaves(root), 1);

  function walk(node: HierarchyNode, parentId: string, depth: number): void {
    const rawId = node.id ?? node.stream_id ?? `node_${Math.random().toString(36).slice(2)}`;
    const nodeId = parentId ? `${parentId}/${rawId}` : rawId;
    const label = segmentLabel(rawId);

    if (node.type === "leaf") {
      nodes.push({
        id: nodeId,
        parent: parentId,
        label,
        value: 1 / totalLeaves,
        hoverText: `<b>${label}</b><br>(structure preview — run pipeline for weights)`,
      });
    } else {
      nodes.push({
        id: nodeId,
        parent: parentId,
        label,
        value: 0,
        hoverText: `<b>${label}</b><br>(group)`,
      });
      for (const child of node.children ?? []) {
        walk(child, nodeId, depth + 1);
      }
    }
  }

  // Synthetic root if the top-level node is itself a group
  nodes.push({
    id: "__root__",
    parent: "",
    label: "Portfolio Structure",
    value: 0,
    hoverText: "<b>Portfolio Structure</b><br>(pre-run preview)",
  });
  walk(root, "__root__", 0);
  return nodes;
}

// ── colour palette ────────────────────────────────────────────────────────────

const TREEMAP_COLORSCALE: [number, string][] = [
  [0, "#1a1b2e"],
  [0.25, "#2c2f6b"],
  [0.5, "#364fc7"],
  [0.75, "#4dabf7"],
  [1, "#20c997"],
];

// ── sub-components ────────────────────────────────────────────────────────────

function SrKnobs({ sr, policy }: { sr: WeightLayerData["sr"]; policy: string }) {
  const chips: { label: string; value: string; active?: boolean }[] = [
    {
      label: "SR tilt",
      value: sr.sr_adjustment ? "ON" : "OFF",
      active: sr.sr_adjustment,
    },
    {
      label: "SR avg",
      value: sr.sr_avg !== null ? fmtNum(sr.sr_avg, 2) : "—",
    },
    {
      label: "Max depth",
      value: sr.sr_tilt_max_depth !== null ? String(sr.sr_tilt_max_depth) : "—",
    },
    {
      label: "Within-group",
      value: sr.within_group_method,
    },
    {
      label: "FDM max",
      value: sr.fdm_max !== null ? fmtNum(sr.fdm_max, 1) : "—",
    },
  ];

  return (
    <Group gap="xs" wrap="wrap" align="center">
      <Badge
        size="lg"
        variant="gradient"
        gradient={{ from: "indigo", to: "teal", deg: 135 }}
        radius="sm"
        style={{ fontWeight: 600, fontSize: 13, maxWidth: "100%", whiteSpace: "normal", height: "auto", padding: "6px 12px" }}
      >
        {policy}
      </Badge>

      {chips.map((chip) => (
        <Chip
          key={chip.label}
          checked={false}
          readOnly
          size="xs"
          variant="outline"
          color={chip.active ? "teal" : "indigo"}
          styles={{
            label: {
              cursor: "default",
              color: chip.active ? "var(--mantine-color-teal-4)" : "var(--mantine-color-indigo-3)",
              borderColor: chip.active
                ? "var(--mantine-color-teal-6)"
                : "var(--mantine-color-dark-4)",
              background: "transparent",
              fontSize: 11,
              paddingInline: 8,
            },
          }}
        >
          <Text component="span" size="xs" c="dimmed" mr={4}>
            {chip.label}:
          </Text>
          <Text component="span" size="xs" fw={500}>
            {chip.value}
          </Text>
        </Chip>
      ))}
    </Group>
  );
}

function EmptyState() {
  return (
    <Center py="xl">
      <Stack align="center" gap="sm">
        <ThemeIcon size={48} variant="light" color="indigo" radius="xl">
          <IconChartTreemap size={28} />
        </ThemeIcon>
        <Text size="sm" c="dimmed" ta="center">
          Run a portfolio pipeline to compute weights.
        </Text>
        <Text size="xs" c="dimmed" ta="center" maw={320}>
          The structure preview above shows the hierarchy with equal sizing. Actual weights will
          appear after the pipeline runs.
        </Text>
      </Stack>
    </Center>
  );
}

// ── treemap chart ─────────────────────────────────────────────────────────────

function WeightTreemap({ nodes, title }: { nodes: TreemapNode[]; title: string }) {
  if (nodes.length === 0) {
    return (
      <Center py="xl">
        <Text size="sm" c="dimmed">No data to display.</Text>
      </Center>
    );
  }

  const ids = nodes.map((n) => n.id);
  const labels = nodes.map((n) => n.label);
  const parents = nodes.map((n) => n.parent);
  const values = nodes.map((n) => n.value);
  const customdata = nodes.map((n) => n.hoverText);

  const layout = {
    ...baseLayout,
    title: {
      text: title,
      font: { color: "#c1c2c5", size: 13 },
      x: 0.01,
      xanchor: "left" as const,
    },
    margin: { l: 0, r: 0, t: 36, b: 0 },
  };

  return (
    <Plot
      data={[
        {
          type: "treemap" as const,
          ids,
          labels,
          parents,
          values,
          branchvalues: "total" as const,
          customdata,
          hovertemplate: "%{customdata}<extra></extra>",
          texttemplate: "%{label}",
          textposition: "middle center",
          marker: {
            colorscale: TREEMAP_COLORSCALE,
            colorbar: { tickfont: { color: "#c1c2c5" } },
            line: { width: 1.5, color: "rgba(255,255,255,0.07)" },
          },
        },
      ]}
      layout={layout}
      config={plotConfig}
      style={{ width: "100%", height: 420 }}
      useResizeHandler
    />
  );
}

// ── main component ────────────────────────────────────────────────────────────

export default function WeightLayerViz({ data }: { data: WeightLayerData }) {
  const { weights, sr, policy, hierarchy } = data;

  // ── phase selector ────────────────────────────────────────────────────────
  const defaultPhase = useMemo(() => {
    if (!weights.phases || weights.phases.length === 0) return null;
    const testPhase = weights.phases.find((p) => p.toLowerCase() === "test");
    return testPhase ?? weights.phases[weights.phases.length - 1];
  }, [weights.phases]);

  const [selectedPhase, setSelectedPhase] = useState<string | null>(defaultPhase);

  // Keep selectedPhase in sync if phases change (e.g., after a run)
  const activePhase = selectedPhase ?? defaultPhase;

  // ── filter rows for active phase ──────────────────────────────────────────
  const phaseRows = useMemo<WeightRow[]>(() => {
    if (!weights.available || !activePhase) return [];
    return weights.rows.filter((r) => r.phase === activePhase);
  }, [weights, activePhase]);

  // ── treemap nodes ─────────────────────────────────────────────────────────
  const treemapNodes = useMemo<TreemapNode[]>(() => {
    if (weights.available && phaseRows.length > 0) {
      return buildTreemapFromRows(phaseRows);
    }
    if (hierarchy) {
      return buildTreemapFromHierarchy(hierarchy);
    }
    return [];
  }, [weights.available, phaseRows, hierarchy]);

  const treemapTitle = weights.available && activePhase
    ? `Weight Distribution — ${activePhase}`
    : "Hierarchy Structure (preview)";

  const segmentedData = weights.phases.map((p) => ({ label: p, value: p }));

  return (
    <Card withBorder padding="md" radius="md" style={{ background: "var(--mantine-color-dark-7)" }}>
      <Stack gap="md">
        {/* ── 1. Header: policy + SR knobs ── */}
        <SrKnobs sr={sr} policy={policy} />

        {/* ── 2. Phase selector (only when weights available and multiple phases) ── */}
        {weights.available && segmentedData.length > 1 && activePhase && (
          <Group gap="xs" align="center">
            <Text size="xs" c="dimmed" fw={500}>
              Phase:
            </Text>
            <SegmentedControl
              size="xs"
              value={activePhase}
              onChange={setSelectedPhase}
              data={segmentedData}
              color="indigo"
              styles={{
                root: { background: "var(--mantine-color-dark-6)" },
                label: { fontSize: 11 },
              }}
            />
          </Group>
        )}

        {/* ── 3. Treemap ── */}
        <WeightTreemap nodes={treemapNodes} title={treemapTitle} />

        {/* ── 4. Empty state overlay ── */}
        {!weights.available && <EmptyState />}

        {/* ── Footer: method hint ── */}
        <Group gap="xs" justify="flex-end">
          <Text size="xs" c="dimmed">
            Method:
          </Text>
          <Badge size="xs" variant="dot" color="indigo">
            {data.method}
          </Badge>
        </Group>
      </Stack>
    </Card>
  );
}
