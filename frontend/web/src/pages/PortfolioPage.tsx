import {
  Alert,
  Badge,
  Button,
  Card,
  Code,
  Grid,
  Group,
  Loader,
  NavLink,
  NumberInput,
  Paper,
  ScrollArea,
  Select,
  MultiSelect,
  Stack,
  Text,
  Tooltip,
} from "@mantine/core";
import { notifications } from "@mantine/notifications";
import {
  IconAlertTriangle,
  IconAdjustments,
  IconFileText,
  IconInfoCircle,
  IconStack2,
  IconPlayerPlay,
  IconRefresh,
} from "@tabler/icons-react";
import { Suspense, lazy, useEffect, useRef, useState } from "react";

import {
  usePortfolioArtifacts,
  usePortfolioDefaults,
  usePortfolioJob,
  useStartPortfolioRun,
  useWeightLayer,
} from "../api/hooks";
import type { ArtifactEntry, PortfolioDefaults } from "../api/types";
import { ArtifactPreview } from "../components/ArtifactPreview";
import { RunStatusBadge } from "../components/RunStatusBadge";
import WeightLayerControls, { type WeightLayerOverrides } from "../components/WeightLayerControls";
import { PageHeader } from "../components/ui/PageHeader";

// Lazy so Plotly (the treemap) stays out of the initial bundle.
const WeightLayerViz = lazy(() => import("../components/WeightLayerViz"));
import { fmtNum, fmtPct } from "../lib/format";

interface ConfigForm {
  phase: string;
  tickers: string[];
  fit_mode: string;
  weight_layer_method: string;
  target_volatility: number;
  max_position_pct: number;
}

function formFromDefaults(d: PortfolioDefaults): ConfigForm {
  return {
    phase: d.default_phase,
    tickers: d.tickers ?? [],
    fit_mode: d.fit_mode,
    weight_layer_method: d.weight_layer_method,
    target_volatility: d.target_volatility,
    max_position_pct: d.max_position_pct,
  };
}

// Summary strip — compact read-only facts derived from defaults + current form
function SummaryStrip({
  d,
  form,
}: {
  d: PortfolioDefaults;
  form: ConfigForm;
}) {
  const items: { label: string; value: string; tip?: string }[] = [
    {
      label: "Ensembles",
      value: String(d.ensemble_count),
      tip: "Number of ensembles currently in the vault portfolio",
    },
    {
      label: "Tickers",
      value: form.tickers.length ? String(form.tickers.length) : "all",
      tip: form.tickers.length ? form.tickers.join(", ") : "All available tickers",
    },
    {
      label: "Fit mode",
      value: form.fit_mode,
      tip: "How ensembles are fit: walk-forward, in-sample, or out-of-sample",
    },
    {
      label: "Weight-layer",
      value: form.weight_layer_method,
      tip: "Cross-timeframe weighting method applied by WeightLayer",
    },
    {
      label: "Target vol",
      value: fmtPct(form.target_volatility),
      tip: "Annualised portfolio volatility target",
    },
    {
      label: "Max position",
      value: fmtNum(form.max_position_pct, 1) + "%",
      tip: "Maximum allocation to a single instrument",
    },
  ];

  return (
    <Paper withBorder p="xs" radius="md">
      <Group gap="xl" wrap="wrap">
        {items.map(({ label, value, tip }) => (
          <Tooltip key={label} label={tip} disabled={!tip} withArrow position="top">
            <Stack gap={0} style={{ cursor: tip ? "help" : "default" }}>
              <Text size="xs" c="dimmed" style={{ whiteSpace: "nowrap" }}>
                {label}
              </Text>
              <Text size="sm" fw={600} style={{ whiteSpace: "nowrap" }}>
                {value}
              </Text>
            </Stack>
          </Tooltip>
        ))}
      </Group>
    </Paper>
  );
}

export function PortfolioPage() {
  const defaults = usePortfolioDefaults();
  const startRun = useStartPortfolioRun();
  const [form, setForm] = useState<ConfigForm | null>(null);
  const [jobId, setJobId] = useState<string | null>(null);

  useEffect(() => {
    if (defaults.data && form === null) setForm(formFromDefaults(defaults.data.defaults));
    if (defaults.data?.job && jobId === null) setJobId(defaults.data.job.job_id);
  }, [defaults.data, form, jobId]);

  const job = usePortfolioJob(jobId);
  const terminal = job.data?.status === "completed" || job.data?.status === "failed";
  const artifacts = usePortfolioArtifacts(job.data?.status === "completed");
  const [selected, setSelected] = useState<string | null>(null);
  const weightLayer = useWeightLayer();
  const [wlOverrides, setWlOverrides] = useState<WeightLayerOverrides>({});

  if (defaults.isLoading || !form) return <Loader />;
  if (defaults.isError || !defaults.data) return <Text c="red">Failed to load portfolio settings.</Text>;
  const d = defaults.data.defaults;
  const tickerOptions = d.ticker_options && d.ticker_options.length ? d.ticker_options : d.tickers;
  const set = <K extends keyof ConfigForm>(key: K, value: ConfigForm[K]) =>
    setForm((f) => (f ? { ...f, [key]: value } : f));
  const dirty = JSON.stringify(form) !== JSON.stringify(formFromDefaults(d));

  const launch = () => {
    startRun.mutate(
      { ...form, ...wlOverrides },
      {
      onSuccess: (newJob) => {
        setJobId(newJob.job_id);
        notifications.show({ message: `Portfolio ${newJob.phase} started`, color: "blue" });
      },
      onError: (err) => notifications.show({ message: String(err), color: "red" }),
    });
  };

  const configSourceLabel = d.config_source_path
    ? d.config_source_path.split(/[\\/]/).pop()
    : "the canonical config";

  return (
    <Stack gap="lg">
      {/* Page header */}
      <PageHeader
        title="Portfolio research"
        subtitle={`Runs the portfolio pipeline on the vault portfolio (${d.ensemble_count} ensembles). These settings are also the baseline the feature portfolio-addition gate scores against.`}
        actions={
          dirty ? (
            <Button
              variant="subtle"
              color="gray"
              size="sm"
              leftSection={<IconRefresh size={14} />}
              onClick={() => setForm(formFromDefaults(d))}
            >
              Reset to config
            </Button>
          ) : undefined
        }
      />

      {/* Summary strip */}
      <SummaryStrip d={d} form={form} />

      {/* Configuration card */}
      <Card withBorder padding="lg">
        <Group justify="space-between" align="center" mb="md">
          <Group gap="xs">
            <IconAdjustments size={16} color="var(--mantine-color-dimmed)" />
            <Text fw={600}>Configuration</Text>
            {dirty && (
              <Badge color="orange" variant="dot" size="sm">
                modified
              </Badge>
            )}
          </Group>
          <Group gap={6}>
            <IconInfoCircle size={13} color="var(--mantine-color-dimmed)" />
            <Text size="xs" c="dimmed">
              Overrides apply per-run only —{" "}
              <Text component="span" size="xs" c="dimmed" fs="italic">
                {configSourceLabel}
              </Text>{" "}
              is never modified.
            </Text>
          </Group>
        </Group>

        {/* Group 1: Scope */}
        <Text size="xs" c="dimmed" fw={500} tt="uppercase" mb={6} ml={2}>
          Scope
        </Text>
        <Grid gutter="md" mb="md">
          <Grid.Col span={{ base: 12, md: 6 }}>
            <Select
              label="Phase"
              data={d.phase_options.map((p) => ({ value: p.value, label: p.label }))}
              value={form.phase}
              onChange={(v) => v && set("phase", v)}
              description={d.phase_options.find((p) => p.value === form.phase)?.description}
            />
          </Grid.Col>
          <Grid.Col span={{ base: 12, md: 6 }}>
            <MultiSelect
              label="Tickers"
              description="Leave empty to include all available tickers."
              data={tickerOptions}
              value={form.tickers}
              onChange={(v) => set("tickers", v)}
              searchable
              clearable
            />
          </Grid.Col>
        </Grid>

        {/* Group 2: Model */}
        <Text size="xs" c="dimmed" fw={500} tt="uppercase" mb={6} ml={2}>
          Model
        </Text>
        <Grid gutter="md" mb="md">
          <Grid.Col span={{ base: 12, md: 6 }}>
            <Select
              label="Fit mode"
              description="Controls whether ensembles are re-fitted or reuse cached weights."
              data={d.fit_modes}
              value={form.fit_mode}
              onChange={(v) => v && set("fit_mode", v)}
            />
          </Grid.Col>
          <Grid.Col span={{ base: 12, md: 6 }}>
            <Select
              label="Weight-layer method"
              description="Cross-timeframe combination method used by WeightLayer."
              data={d.weight_layer_methods}
              value={form.weight_layer_method}
              onChange={(v) => v && set("weight_layer_method", v)}
              searchable
            />
          </Grid.Col>
        </Grid>

        {/* Group 3: Risk */}
        <Text size="xs" c="dimmed" fw={500} tt="uppercase" mb={6} ml={2}>
          Risk
        </Text>
        <Grid gutter="md" mb="lg">
          <Grid.Col span={{ base: 6, md: 3 }}>
            <NumberInput
              label="Target vol"
              description="Annualised portfolio vol target (e.g. 0.2 = 20%)."
              step={0.01}
              decimalScale={3}
              value={form.target_volatility}
              onChange={(v) => set("target_volatility", Number(v))}
            />
          </Grid.Col>
          <Grid.Col span={{ base: 6, md: 3 }}>
            <NumberInput
              label="Max position %"
              description="Maximum capital allocated to any single instrument."
              step={0.5}
              value={form.max_position_pct}
              onChange={(v) => set("max_position_pct", Number(v))}
            />
          </Grid.Col>
        </Grid>

        {/* Run controls */}
        <Group align="center" gap="sm">
          <Button
            leftSection={<IconPlayerPlay size={16} />}
            loading={startRun.isPending}
            disabled={!terminal && job.data?.status === "running"}
            onClick={launch}
          >
            Run portfolio research
          </Button>
          {job.data && <RunStatusBadge status={job.data.status} />}
          <Text size="xs" c="dimmed">
            Portfolio runs can take several minutes (walk-forward across train / validation / test).
          </Text>
        </Group>
      </Card>

      {/* Weight layer — Sharpe-tilt policy, editable knobs + hierarchy, weights treemap */}
      {weightLayer.data && (
        <Card withBorder padding="lg">
          <Group gap="xs" mb="md">
            <IconAdjustments size={16} color="var(--mantine-color-dimmed)" />
            <Text fw={600}>Weight layer</Text>
            <Badge variant="light" color="teal" size="sm">
              {weightLayer.data.policy}
            </Badge>
          </Group>
          <Grid gutter="lg">
            <Grid.Col span={{ base: 12, lg: 5 }}>
              <WeightLayerControls
                data={weightLayer.data}
                value={wlOverrides}
                onChange={(patch) => setWlOverrides((o) => ({ ...o, ...patch }))}
              />
            </Grid.Col>
            <Grid.Col span={{ base: 12, lg: 7 }}>
              <Suspense fallback={<Loader />}>
                <WeightLayerViz data={weightLayer.data} />
              </Suspense>
            </Grid.Col>
          </Grid>
        </Card>
      )}

      {/* Error alert */}
      {job.data?.status === "failed" && (
        <Alert color="red" icon={<IconAlertTriangle size={18} />} title="Run failed" variant="light">
          {job.data.error_text}
        </Alert>
      )}

      {/* Log + artifacts */}
      {job.data && (
        <Grid gutter="lg">
          <Grid.Col span={{ base: 12, md: job.data.status === "completed" ? 7 : 12 }}>
            <Card withBorder padding="md" h="100%">
              <Group justify="space-between" mb="xs">
                <Group gap="xs">
                  <IconStack2 size={15} color="var(--mantine-color-dimmed)" />
                  <Text fw={600} size="sm">
                    Log
                  </Text>
                  <Text size="xs" c="dimmed">
                    — {job.data.phase}
                  </Text>
                </Group>
                {!terminal ? (
                  <Loader size="xs" />
                ) : (
                  <Badge
                    color={job.data.status === "completed" ? "teal" : "red"}
                    variant="light"
                    size="xs"
                  >
                    {job.data.status}
                  </Badge>
                )}
              </Group>
              <LogView text={job.data.log_text} live={!terminal} />
            </Card>
          </Grid.Col>

          {job.data.status === "completed" && (
            <Grid.Col span={{ base: 12, md: 5 }}>
              <Card withBorder padding="sm" h="100%">
                <Group gap="xs" mb="sm">
                  <IconFileText size={15} color="var(--mantine-color-dimmed)" />
                  <Text fw={600} size="sm">
                    Artifacts
                  </Text>
                  {artifacts.data?.[0]?.files.length ? (
                    <Badge color="indigo" variant="light" size="xs">
                      {artifacts.data[0].files.length}
                    </Badge>
                  ) : null}
                  {artifacts.isLoading && <Loader size={12} />}
                </Group>

                <ScrollArea h={220} mb="sm" offsetScrollbars>
                  {artifacts.data?.[0]?.files.map((file) => (
                    <ArtifactLink
                      key={file.path}
                      file={file}
                      active={selected === file.path}
                      onClick={() => setSelected((prev) => (prev === file.path ? null : file.path))}
                    />
                  ))}
                  {artifacts.data && artifacts.data[0]?.files.length === 0 && (
                    <Text size="xs" c="dimmed" p="xs">
                      No artifacts yet.
                    </Text>
                  )}
                </ScrollArea>

                {selected && (
                  <Stack gap="xs">
                    <Text size="xs" c="dimmed" style={{ wordBreak: "break-all" }}>
                      {selected}
                    </Text>
                    <ArtifactPreview path={selected} />
                  </Stack>
                )}
              </Card>
            </Grid.Col>
          )}
        </Grid>
      )}
    </Stack>
  );
}

function ArtifactLink({ file, active, onClick }: { file: ArtifactEntry; active: boolean; onClick: () => void }) {
  return (
    <NavLink
      label={file.name}
      description={file.kind}
      active={active}
      onClick={onClick}
      leftSection={<IconFileText size={14} />}
      styles={{ label: { fontSize: 12 }, description: { fontSize: 10 } }}
    />
  );
}

function LogView({ text, live }: { text: string; live: boolean }) {
  const viewport = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (live && viewport.current) viewport.current.scrollTop = viewport.current.scrollHeight;
  }, [text, live]);
  return (
    <ScrollArea h={460} viewportRef={viewport}>
      <Code block style={{ whiteSpace: "pre-wrap", fontSize: 11 }}>
        {text || (live ? "Waiting for output…" : "(no output)")}
      </Code>
    </ScrollArea>
  );
}
