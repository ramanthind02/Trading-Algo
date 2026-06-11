import { Box, Group, MultiSelect, Stack, Switch, Text } from "@mantine/core";
import { useMemo, useState } from "react";

import type { EquitySeries } from "../api/types";
import { Plot, baseLayout, plotConfig } from "./Plot";

// Teal-to-indigo palette — visually distinct in dark theme, up to 10 combos.
const COMBO_COLORS = [
  "#20c997", // teal
  "#4dabf7", // blue
  "#748ffc", // indigo
  "#da77f2", // violet
  "#f783ac", // pink
  "#ffa94d", // orange
  "#a9e34b", // lime
  "#63e6be", // cyan
  "#ffd43b", // yellow
  "#ff8787", // red
];

export function EquityChart({
  series,
  combos,
  tickers,
}: {
  series: EquitySeries[];
  combos: string[];
  tickers: string[];
}) {
  const [selectedCombos, setSelectedCombos] = useState<string[]>(combos.slice(0, 3));
  const [perTicker, setPerTicker] = useState(false);

  // Stable color assignment keyed by combo index in the master list.
  const comboColor = useMemo(
    () => new Map(combos.map((c, i) => [c, COMBO_COLORS[i % COMBO_COLORS.length]])),
    [combos],
  );

  const traces = useMemo(() => {
    const wanted = new Set(selectedCombos);
    const filtered = series.filter((s) => wanted.has(s.combo));

    // Zero baseline — drawn first so it sits under all data lines.
    const allDates = filtered.flatMap((s) => s.datetime);
    const minDate = allDates.length ? allDates.reduce((a, b) => (a < b ? a : b)) : null;
    const maxDate = allDates.length ? allDates.reduce((a, b) => (a > b ? a : b)) : null;
    const baseline =
      minDate && maxDate
        ? [
            {
              type: "scatter" as const,
              mode: "lines" as const,
              name: "zero",
              x: [minDate, maxDate],
              y: [0, 0],
              line: { color: "rgba(255,255,255,0.18)", width: 1, dash: "dot" as const },
              hoverinfo: "skip" as const,
              showlegend: false,
            },
          ]
        : [];

    if (perTicker) {
      const dataTraces = filtered.map((s) => {
        const color = comboColor.get(s.combo) ?? "#aaa";
        return {
          type: "scatter" as const,
          mode: "lines" as const,
          name: `${shortCombo(s.combo)} · ${s.ticker}`,
          x: s.datetime,
          y: s.cumulative,
          line: { color, width: 1.5 },
          hovertemplate: `<b>%{fullData.name}</b><br>%{x|%Y-%m-%d}<br>Cum. return: %{y:.3f}<extra></extra>`,
        };
      });
      return [...baseline, ...dataTraces];
    }

    // Equal-weight mean across tickers, one line per combo.
    const byCombo = new Map<string, EquitySeries[]>();
    for (const s of filtered) {
      const list = byCombo.get(s.combo) ?? [];
      list.push(s);
      byCombo.set(s.combo, list);
    }
    const dataTraces = [...byCombo.entries()].map(([combo, group]) => {
      const base = group[0];
      const y = base.cumulative.map((_, i) => {
        const vals = group.map((g) => g.cumulative[i]).filter((v) => v != null);
        return vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : null;
      });
      const color = comboColor.get(combo) ?? "#aaa";
      return {
        type: "scatter" as const,
        mode: "lines" as const,
        name: shortCombo(combo),
        x: base.datetime,
        y,
        line: { color, width: 2 },
        hovertemplate: `<b>%{fullData.name}</b><br>%{x|%Y-%m-%d}<br>Cum. return: %{y:.3f}<extra></extra>`,
      };
    });
    return [...baseline, ...dataTraces];
  }, [series, selectedCombos, perTicker, comboColor]);

  const isEmpty = selectedCombos.length === 0 || traces.filter((t) => !("hoverinfo" in t)).length === 0;

  return (
    <Stack gap="sm">
      <Group justify="space-between" align="flex-end" wrap="wrap" gap="xs">
        <MultiSelect
          size="xs"
          label="Parameter combos"
          placeholder={combos.length ? "Select combos…" : "No combos available"}
          data={combos.map((c) => ({ value: c, label: shortCombo(c) }))}
          value={selectedCombos}
          onChange={setSelectedCombos}
          style={{ flex: 1, minWidth: 240, maxWidth: 520 }}
          searchable
          clearable
          maxDropdownHeight={240}
        />
        <Box>
          <Text size="xs" c="dimmed" mb={4}>
            Breakdown
          </Text>
          <Switch
            size="xs"
            label="Per ticker"
            description={tickers.length ? tickers.join(", ") : undefined}
            checked={perTicker}
            onChange={(e) => setPerTicker(e.currentTarget.checked)}
          />
        </Box>
      </Group>

      {isEmpty ? (
        <Box
          h={380}
          style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            border: "1px dashed rgba(255,255,255,0.12)",
            borderRadius: 8,
          }}
        >
          <Text size="sm" c="dimmed">
            Select at least one combo to display equity curves.
          </Text>
        </Box>
      ) : (
        <Plot
          data={traces}
          layout={{
            ...baseLayout,
            height: 380,
            yaxis: {
              ...baseLayout.yaxis,
              title: { text: "Cumulative return" },
              tickformat: ".2f",
              zeroline: false, // we draw our own dotted baseline
            },
            xaxis: {
              ...baseLayout.xaxis,
              type: "date",
            },
            legend: {
              orientation: "h",
              y: -0.22,
              font: { size: 11 },
              itemclick: "toggleothers",
            },
            hovermode: "x unified",
          }}
          config={plotConfig}
          style={{ width: "100%" }}
        />
      )}

      <Text size="xs" c="dimmed" fs="italic">
        {perTicker
          ? `Each line is one combo × ticker pair (${tickers.join(", ")}). Useful for spotting per-instrument divergence.`
          : `Each line is the equal-weight mean cumulative return across all tickers (${tickers.join(", ")}). Toggle "Per ticker" to see individual instruments.`}
      </Text>
    </Stack>
  );
}

function shortCombo(combo: string): string {
  return combo.replace(/^.*?__/, "");
}
