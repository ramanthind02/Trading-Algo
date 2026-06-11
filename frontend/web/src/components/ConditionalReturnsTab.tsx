// Conditional returns: bucket the run's best combo's returns by an indicator's value, with an
// optional binary regime cross-cut. Default-exported so RunResults can React.lazy() it (keeps
// Plotly out of the initial bundle).
import {
  Alert,
  Badge,
  Box,
  Button,
  Card,
  Divider,
  Grid,
  Group,
  NumberInput,
  SegmentedControl,
  Select,
  Stack,
  Switch,
  Table,
  Text,
  TextInput,
  ThemeIcon,
  Title,
  Tooltip,
} from "@mantine/core";
import { IconChartHistogram, IconFilter, IconInfoCircle, IconPlayerPlay } from "@tabler/icons-react";
import { useEffect, useMemo, useRef, useState } from "react";

import { useConditionalReturns, useModuleDetail, useModules } from "../api/hooks";
import type {
  ConditionalBin,
  ConditionalReturnsData,
  IndicatorChoice,
  ModuleDetail,
  ParamValue,
} from "../api/types";
import { fmtInt, fmtNum, fmtPct, metricColor } from "../lib/format";
import { Plot, baseLayout, plotConfig } from "./Plot";

const PLUMBING = new Set(["ticker", "tf", "timeframe", "self"]);
const PCT_METRICS = new Set(["mean_return", "cumulative", "hit_rate", "instrument_mean_return"]);
const SERIES_COLORS = ["#4dabf7", "#20c997", "#da77f2", "#ffa94d", "#f783ac", "#a9e34b"];

const METRIC_LABEL: Record<string, string> = {
  mean_return: "Mean return / bar",
  sharpe: "Sharpe (ann.)",
  t_stat: "t-stat",
  hit_rate: "Hit rate",
  sortino: "Sortino (ann.)",
  cumulative: "Cumulative",
};

// Per-bar means are tiny; give them more decimals than window-level pct metrics.
function pctFormat(metric: string): string {
  return metric === "mean_return" || metric === "instrument_mean_return" ? ".2%" : ".1%";
}

export default function ConditionalReturnsTab({ runId }: { runId: string }) {
  const modules = useModules();
  const moduleData = useMemo(() => groupModules(modules.data ?? []), [modules.data]);

  const [indicator, setIndicator] = useState<IndicatorChoice>({ module: "", params: {} });
  const [binMode, setBinMode] = useState<"quantile" | "fixed">("quantile");
  const [nBins, setNBins] = useState(5);
  const [edgesText, setEdgesText] = useState("30, 70");
  const [perTicker, setPerTicker] = useState(false);

  const [useRegime, setUseRegime] = useState(false);
  const [regime, setRegime] = useState<IndicatorChoice>({ module: "", params: {} });
  const [threshold, setThreshold] = useState(25);
  const [above, setAbove] = useState(true);

  const analysis = useConditionalReturns(runId);

  const run = () => {
    const edges = edgesText
      .split(/[,\s]+/)
      .map((t) => Number(t.trim()))
      .filter((n) => Number.isFinite(n));
    analysis.mutate({
      indicator,
      bin_mode: binMode,
      n_bins: nBins,
      edges,
      per_ticker: perTicker,
      regime:
        useRegime && regime.module
          ? { ...regime, threshold, above }
          : null,
    });
  };

  const canRun = !!indicator.module && (binMode === "quantile" || edgesText.trim().length > 0);

  return (
    <Stack gap="lg">
      <Card withBorder padding="lg" radius="md">
        <SectionHeader
          icon={<IconChartHistogram size={15} />}
          title="Conditional returns"
          hint="Bucket the best combo's returns by an indicator's value to see in which regimes the strategy makes money."
        />
        <Divider mt="sm" mb="md" />

        <Grid gutter="md">
          {/* Conditioning indicator */}
          <Grid.Col span={{ base: 12, md: 6 }}>
            <Text size="xs" fw={600} c="dimmed" tt="uppercase" mb={6} style={{ letterSpacing: "0.05em" }}>
              Bucket by indicator
            </Text>
            <IndicatorPicker
              value={indicator}
              onChange={setIndicator}
              moduleData={moduleData}
              loading={modules.isLoading}
              placeholder="Pick an indicator (RSI, ADX, ATR…)"
            />
          </Grid.Col>

          {/* Bin controls */}
          <Grid.Col span={{ base: 12, md: 6 }}>
            <Text size="xs" fw={600} c="dimmed" tt="uppercase" mb={6} style={{ letterSpacing: "0.05em" }}>
              Buckets
            </Text>
            <Stack gap="xs">
              <SegmentedControl
                size="xs"
                value={binMode}
                onChange={(v) => setBinMode(v as "quantile" | "fixed")}
                data={[
                  { label: "Quantile (equal count)", value: "quantile" },
                  { label: "Fixed thresholds", value: "fixed" },
                ]}
              />
              {binMode === "quantile" ? (
                <NumberInput
                  size="xs"
                  label="Number of buckets"
                  min={2}
                  max={10}
                  value={nBins}
                  onChange={(v) => setNBins(typeof v === "number" ? v : 5)}
                  style={{ maxWidth: 180 }}
                />
              ) : (
                <TextInput
                  size="xs"
                  label="Thresholds (comma-separated)"
                  placeholder="e.g. 30, 70"
                  value={edgesText}
                  onChange={(e) => setEdgesText(e.currentTarget.value)}
                  description="Splits the indicator into ranges, e.g. <30, 30–70, ≥70."
                />
              )}
              <Switch
                size="xs"
                label="Break down per ticker"
                checked={perTicker}
                onChange={(e) => setPerTicker(e.currentTarget.checked)}
              />
            </Stack>
          </Grid.Col>
        </Grid>

        {/* Optional binary regime filter */}
        <Box mt="md">
          <Switch
            size="sm"
            label={
              <Group gap={6}>
                <IconFilter size={14} />
                <Text size="sm">Add a binary regime filter (cross-cut each bucket into on / off)</Text>
              </Group>
            }
            checked={useRegime}
            onChange={(e) => setUseRegime(e.currentTarget.checked)}
          />
          {useRegime && (
            <Card withBorder mt="sm" padding="md" radius="sm" bg="dark.6">
              <Grid gutter="md" align="flex-end">
                <Grid.Col span={{ base: 12, md: 6 }}>
                  <IndicatorPicker
                    value={regime}
                    onChange={setRegime}
                    moduleData={moduleData}
                    loading={modules.isLoading}
                    placeholder="Regime indicator (e.g. ADX)"
                  />
                </Grid.Col>
                <Grid.Col span={{ base: 6, md: 3 }}>
                  <NumberInput
                    size="xs"
                    label="Threshold"
                    value={threshold}
                    onChange={(v) => setThreshold(typeof v === "number" ? v : 0)}
                  />
                </Grid.Col>
                <Grid.Col span={{ base: 6, md: 3 }}>
                  <SegmentedControl
                    size="xs"
                    fullWidth
                    value={above ? "above" : "below"}
                    onChange={(v) => setAbove(v === "above")}
                    data={[
                      { label: "On if ≥", value: "above" },
                      { label: "On if ≤", value: "below" },
                    ]}
                  />
                </Grid.Col>
              </Grid>
            </Card>
          )}
        </Box>

        <Group mt="lg" justify="flex-end">
          <Button
            leftSection={<IconPlayerPlay size={15} />}
            onClick={run}
            loading={analysis.isPending}
            disabled={!canRun}
          >
            Analyze
          </Button>
        </Group>
      </Card>

      {analysis.isError && (
        <Alert color="red" icon={<IconInfoCircle size={16} />} title="Analysis failed">
          {String(analysis.error)}
        </Alert>
      )}

      {analysis.data && <ResultPanel data={analysis.data} />}

      {!analysis.data && !analysis.isPending && (
        <Text size="sm" c="dimmed" ta="center" py="md">
          Pick an indicator and click <b>Analyze</b> to slice the best combo's returns by regime.
        </Text>
      )}
    </Stack>
  );
}

// ── Indicator picker (module Select + tunable param inputs) ───────────────────

function IndicatorPicker({
  value,
  onChange,
  moduleData,
  loading,
  placeholder,
}: {
  value: IndicatorChoice;
  onChange: (next: IndicatorChoice) => void;
  moduleData: { group: string; items: { value: string; label: string }[] }[];
  loading: boolean;
  placeholder: string;
}) {
  const detail = useModuleDetail(value.module || null);
  const params = useMemo(() => (detail.data ? tunableParams(detail.data) : []), [detail.data]);

  // Seed each module's defaults once when its detail loads (cleared params → defaults).
  const seeded = useRef<string | null>(null);
  useEffect(() => {
    if (!value.module || !detail.data) return;
    if (seeded.current === value.module) return;
    seeded.current = value.module;
    const seededParams = Object.fromEntries(
      tunableParams(detail.data).map((p) => [p.name, p.default ?? p.choices?.[0] ?? 0]),
    );
    onChange({ module: value.module, params: seededParams });
  }, [value.module, detail.data, onChange]);

  const setParam = (name: string, raw: ParamValue) =>
    onChange({ module: value.module, params: { ...value.params, [name]: raw } });

  return (
    <Stack gap="xs">
      <Select
        size="xs"
        placeholder={placeholder}
        data={moduleData}
        value={value.module || null}
        onChange={(m) => {
          seeded.current = null;
          onChange({ module: m ?? "", params: {} });
        }}
        searchable
        clearable
        nothingFoundMessage={loading ? "Loading…" : "No match"}
        maxDropdownHeight={280}
      />
      {params.length > 0 && (
        <Group gap="xs" wrap="wrap">
          {params.map((p) =>
            p.choices && p.choices.length > 0 ? (
              <Select
                key={p.name}
                size="xs"
                label={p.name}
                data={p.choices}
                value={String(value.params[p.name] ?? p.default ?? p.choices[0])}
                onChange={(v) => v != null && setParam(p.name, v)}
                style={{ maxWidth: 150 }}
              />
            ) : (
              <NumberInput
                key={p.name}
                size="xs"
                label={p.name}
                value={asNum(value.params[p.name] ?? p.default)}
                onChange={(v) => setParam(p.name, typeof v === "number" ? v : Number(v) || 0)}
                style={{ maxWidth: 120 }}
              />
            ),
          )}
        </Group>
      )}
    </Stack>
  );
}

// ── Results: bar chart + stats table ──────────────────────────────────────────

function ResultPanel({ data }: { data: ConditionalReturnsData }) {
  const [metric, setMetric] = useState<string>("sharpe");

  if (!data.available)
    return (
      <Alert color="yellow" icon={<IconInfoCircle size={16} />} title="No conditional returns">
        {data.reason ?? "The analysis produced no populated buckets."}
      </Alert>
    );

  return (
    <Card withBorder padding="lg" radius="md">
      <Group justify="space-between" align="flex-start" wrap="wrap" gap="xs">
        <Stack gap={2}>
          <Title order={5}>{data.strategy.label}</Title>
          <Group gap="xs">
            <Badge size="sm" variant="light" color="indigo" tt="none">
              by {data.condition.label}
            </Badge>
            <Badge size="sm" variant="outline" color="gray" tt="capitalize">
              {data.strategy.direction}
            </Badge>
            {data.regime && (
              <Badge size="sm" variant="light" color="grape" tt="none">
                regime: {data.regime.label}
              </Badge>
            )}
            <Badge size="sm" variant="dot" color="gray">
              {data.bin_mode === "quantile" ? `${data.n_bins} quantile buckets` : "fixed thresholds"}
            </Badge>
          </Group>
        </Stack>
        <SegmentedControl
          size="xs"
          value={metric}
          onChange={setMetric}
          data={data.metric_options.map((m) => ({ value: m, label: METRIC_LABEL[m] ?? m }))}
        />
      </Group>

      <Divider my="md" />
      <BinChart bins={data.bins} metric={metric} hasRegime={!!data.regime} perTicker={data.per_ticker} />
      <Divider my="md" />
      <BinTable bins={data.bins} hasRegime={!!data.regime} perTicker={data.per_ticker} />

      <Text size="xs" c="dimmed" fs="italic" mt="sm">
        Returns are the best combo's signal-weighted returns (signal × forward return) on the bars it
        holds a position, bucketed by {data.condition.label} at decision time. {data.tickers.join(", ")}.
      </Text>
    </Card>
  );
}

function BinChart({
  bins,
  metric,
  hasRegime,
  perTicker,
}: {
  bins: ConditionalBin[];
  metric: string;
  hasRegime: boolean;
  perTicker: boolean;
}) {
  const isPct = PCT_METRICS.has(metric);
  const yFmt = isPct ? pctFormat(metric) : ".2f";

  const { categoryarray, traces } = useMemo(() => {
    const ordered = [...new Map(bins.map((b) => [b.bin_index, b.label])).entries()].sort(
      (a, b) => a[0] - b[0],
    );
    const categoryarray = ordered.map(([, label]) => label);

    const keyOf = (b: ConditionalBin): string =>
      hasRegime && perTicker
        ? `${b.ticker} · ${b.regime}`
        : hasRegime
          ? (b.regime ?? "all")
          : perTicker
            ? (b.ticker ?? "all")
            : "all";

    const groups = new Map<string, ConditionalBin[]>();
    for (const b of bins) {
      const list = groups.get(keyOf(b)) ?? [];
      list.push(b);
      groups.set(keyOf(b), list);
    }
    const single = groups.size === 1;

    const traces = [...groups.entries()].map(([key, group], gi) => {
      const ys = group.map((b) => (b[metric as keyof ConditionalBin] as number | null) ?? null);
      const marker = single
        ? { color: ys.map((y) => (y != null && y >= 0 ? "#20c997" : "#ff8787")) }
        : { color: SERIES_COLORS[gi % SERIES_COLORS.length] };
      return {
        type: "bar" as const,
        name: single ? METRIC_LABEL[metric] ?? metric : key,
        x: group.map((b) => b.label),
        y: ys,
        marker,
        hovertemplate: `%{x}<br>%{y:${yFmt}} · n=%{customdata}<extra>${single ? "" : key}</extra>`,
        customdata: group.map((b) => b.n_obs),
      };
    });
    return { categoryarray, traces };
  }, [bins, metric, hasRegime, perTicker, yFmt]);

  return (
    <Plot
      data={traces}
      layout={{
        ...baseLayout,
        height: 340,
        barmode: "group",
        xaxis: { ...baseLayout.xaxis, type: "category", categoryorder: "array", categoryarray },
        yaxis: {
          ...baseLayout.yaxis,
          title: { text: METRIC_LABEL[metric] ?? metric },
          tickformat: yFmt,
          zeroline: true,
        },
        legend: { orientation: "h", y: -0.22, font: { size: 11 } },
        showlegend: traces.length > 1,
      }}
      config={plotConfig}
      style={{ width: "100%" }}
    />
  );
}

function BinTable({
  bins,
  hasRegime,
  perTicker,
}: {
  bins: ConditionalBin[];
  hasRegime: boolean;
  perTicker: boolean;
}) {
  return (
    <Table.ScrollContainer minWidth={640}>
      <Table striped highlightOnHover withTableBorder verticalSpacing={6} fz="xs">
        <Table.Thead>
          <Table.Tr>
            <Table.Th>Bucket</Table.Th>
            {perTicker && <Table.Th>Ticker</Table.Th>}
            {hasRegime && <Table.Th>Regime</Table.Th>}
            <Table.Th ta="right">n</Table.Th>
            <Table.Th ta="right">Mean ret./bar</Table.Th>
            <Table.Th ta="right">Sharpe</Table.Th>
            <Table.Th ta="right">t-stat</Table.Th>
            <Table.Th ta="right">Hit rate</Table.Th>
            <Table.Th ta="right">Cumulative</Table.Th>
          </Table.Tr>
        </Table.Thead>
        <Table.Tbody>
          {bins.map((b, i) => (
            <Table.Tr key={`${b.ticker ?? ""}-${b.regime ?? ""}-${b.bin_index}-${i}`}>
              <Table.Td ff="monospace">{b.label}</Table.Td>
              {perTicker && <Table.Td>{b.ticker ?? "—"}</Table.Td>}
              {hasRegime && (
                <Table.Td>
                  <Badge size="xs" variant="light" color={b.regime === "on" ? "teal" : "gray"}>
                    {b.regime}
                  </Badge>
                </Table.Td>
              )}
              <Table.Td ta="right" c="dimmed">
                {fmtInt(b.n_obs)}
              </Table.Td>
              <Table.Td ta="right" c={metricColor(b.mean_return, 0, 0)}>
                {fmtPct(b.mean_return, 2)}
              </Table.Td>
              <Table.Td ta="right" c={metricColor(b.sharpe, 0.5, 0)}>
                {fmtNum(b.sharpe)}
              </Table.Td>
              <Table.Td ta="right" c={metricColor(b.t_stat, 1.5, 0)}>
                {fmtNum(b.t_stat)}
              </Table.Td>
              <Table.Td ta="right">{fmtPct(b.hit_rate)}</Table.Td>
              <Table.Td ta="right" c={metricColor(b.cumulative, 0, 0)}>
                {fmtPct(b.cumulative)}
              </Table.Td>
            </Table.Tr>
          ))}
        </Table.Tbody>
      </Table>
    </Table.ScrollContainer>
  );
}

// ── helpers ───────────────────────────────────────────────────────────────────

function SectionHeader({ icon, title, hint }: { icon: React.ReactNode; title: string; hint: string }) {
  return (
    <Group gap="xs" align="center">
      <ThemeIcon size={22} variant="light" color="indigo" radius="sm">
        {icon}
      </ThemeIcon>
      <Title order={5} style={{ lineHeight: 1 }}>
        {title}
      </Title>
      <Tooltip label={hint} multiline w={300} withArrow>
        <ThemeIcon size={16} variant="transparent" color="dimmed">
          <IconInfoCircle size={14} />
        </ThemeIcon>
      </Tooltip>
    </Group>
  );
}

function groupModules(
  modules: { name: string; category: string }[],
): { group: string; items: { value: string; label: string }[] }[] {
  const byCat = new Map<string, { value: string; label: string }[]>();
  for (const m of modules) {
    const list = byCat.get(m.category) ?? [];
    list.push({ value: m.name, label: m.name });
    byCat.set(m.category, list);
  }
  return [...byCat.entries()]
    .sort((a, b) => a[0].localeCompare(b[0]))
    .map(([group, items]) => ({ group, items }));
}

function tunableParams(detail: ModuleDetail) {
  return detail.params.filter(
    (p) =>
      !PLUMBING.has(p.name) &&
      (typeof p.default === "number" || (p.choices != null && p.choices.length > 0)),
  );
}

function asNum(value: ParamValue | null | undefined): number | "" {
  if (value === null || value === undefined || value === "") return "";
  const n = Number(value);
  return Number.isFinite(n) ? n : "";
}
