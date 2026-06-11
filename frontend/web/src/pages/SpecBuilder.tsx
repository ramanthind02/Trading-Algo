import {
  Alert,
  Badge,
  Box,
  Button,
  Card,
  Code,
  Collapse,
  Divider,
  Grid,
  Group,
  JsonInput,
  Loader,
  LoadingOverlay,
  MultiSelect,
  NumberInput,
  RingProgress,
  SegmentedControl,
  Select,
  Stack,
  Switch,
  Text,
  Textarea,
  TextInput,
  ThemeIcon,
  Tooltip,
} from "@mantine/core";
import { useDebouncedValue } from "@mantine/hooks";
import { notifications } from "@mantine/notifications";
import { useQuery } from "@tanstack/react-query";
import {
  IconAlertTriangle,
  IconBolt,
  IconCalendarStats,
  IconChartBar,
  IconChevronLeft,
  IconCircleCheck,
  IconCode,
  IconDeviceFloppy,
  IconGauge,
  IconInfoCircle,
  IconLayersLinked,
  IconPlayerPlay,
  IconTag,
  IconTimeline,
  IconWand,
  IconWorld,
} from "@tabler/icons-react";
import { useEffect, useMemo, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";

import { api } from "../api/client";
import {
  useModuleDetail,
  useModules,
  useSaveSpec,
  useSchema,
  useSpec,
  useStartRun,
} from "../api/hooks";
import type { EnumOption, SpecValidation, StrategySpec } from "../api/types";
import { FormSection } from "../components/FormSection";
import { ParamGridEditor, gridToRows, rowsToGrid, type ParamRow } from "../components/ParamGridEditor";
import { PageHeader } from "../components/ui/PageHeader";
import { fmtInt } from "../lib/format";
import { blankSpec } from "../specDefaults";

function selectData(options: EnumOption[]) {
  return options.map((o) => ({ value: o.value, label: o.label }));
}

export function SpecBuilder() {
  const { id } = useParams();
  const navigate = useNavigate();
  const schema = useSchema();
  const modules = useModules();
  const existing = useSpec(id ?? null);

  const [spec, setSpec] = useState<StrategySpec>(blankSpec);
  const [paramRows, setParamRows] = useState<ParamRow[]>([{ key: "", values: "" }]);
  const [customWindows, setCustomWindows] = useState(false);
  const [showJson, setShowJson] = useState(false);
  const [loaded, setLoaded] = useState(false);

  // Load an existing spec into local state once it arrives.
  useEffect(() => {
    if (id && existing.data && !loaded) {
      setSpec(existing.data.spec);
      setParamRows(gridToRows(existing.data.spec.signal.param_grid));
      setCustomWindows(existing.data.spec.windows !== null);
      setLoaded(true);
    }
    if (!id && !loaded) setLoaded(true);
  }, [id, existing.data, loaded]);

  const moduleDetail = useModuleDetail(spec.signal.module_name || null);

  const paramChoices = useMemo(() => {
    const result: Record<string, string[]> = {};
    for (const p of moduleDetail.data?.params ?? []) {
      if (p.choices) result[p.name] = p.choices;
    }
    return result;
  }, [moduleDetail.data]);

  const effectiveSpec = useMemo<StrategySpec>(
    () => ({
      ...spec,
      signal: { module_name: spec.signal.module_name, param_grid: rowsToGrid(paramRows) },
      windows: customWindows ? spec.windows : null,
    }),
    [spec, paramRows, customWindows],
  );

  const [debouncedJson] = useDebouncedValue(JSON.stringify(effectiveSpec), 350);
  const validation = useQuery({
    queryKey: ["validate", debouncedJson],
    queryFn: () => api.post<SpecValidation>("/api/specs/validate", JSON.parse(debouncedJson)),
    enabled: loaded,
  });

  const save = useSaveSpec();
  const startRun = useStartRun();

  const update = <K extends keyof StrategySpec>(key: K, value: StrategySpec[K]) =>
    setSpec((s) => ({ ...s, [key]: value }));

  const moduleOptions = useMemo(() => {
    const byCat = new Map<string, { value: string; label: string }[]>();
    for (const m of modules.data ?? []) {
      const list = byCat.get(m.category) ?? [];
      list.push({ value: m.name, label: m.name });
      byCat.set(m.category, list);
    }
    return [...byCat.entries()].map(([group, items]) => ({ group, items }));
  }, [modules.data]);

  const timeframeOptions =
    spec.mode === "intraday" ? schema.data?.timeframes.intraday : schema.data?.timeframes.daily;

  // Keep timeframe consistent with the selected mode.
  useEffect(() => {
    if (!timeframeOptions) return;
    if (!timeframeOptions.includes(spec.timeframe)) update("timeframe", timeframeOptions[0]);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [spec.mode]);

  const loadModuleParams = () => {
    if (!moduleDetail.data) return;
    const skip = new Set(["ticker", "tf", "timeframe", "direction", "strategy_mode"]);
    const rows: ParamRow[] = moduleDetail.data.params
      .filter((p) => !skip.has(p.name))
      .map((p) => ({ key: p.name, values: p.default == null ? "" : String(p.default) }));
    setParamRows(rows.length ? rows : [{ key: "", values: "" }]);
  };

  const persist = (then?: (savedId: string) => void) => {
    save.mutate(
      { id: id ?? null, spec: effectiveSpec },
      {
        onSuccess: (detail) => {
          notifications.show({ message: `Saved ${detail.spec.name}`, color: "teal" });
          if (!id) navigate(`/builder/${detail.id}`, { replace: true });
          then?.(detail.id);
        },
        onError: (err) => notifications.show({ message: String(err), color: "red" }),
      },
    );
  };

  const saveAndRun = () =>
    persist((savedId) =>
      startRun.mutate(
        { spec_id: savedId },
        {
          onSuccess: (run) => navigate(`/runs/${run.run_id}`),
          onError: (err) => notifications.show({ message: String(err), color: "red" }),
        },
      ),
    );

  if (schema.isLoading || (id && existing.isLoading)) return <Loader />;
  if (schema.isError) return <Text c="red">Failed to load schema: {String(schema.error)}</Text>;
  const s = schema.data!;
  const v = validation.data;

  return (
    <Stack gap="lg">
      <PageHeader
        title={id ? "Edit strategy" : "New strategy"}
        subtitle={id ? spec.name || undefined : "Define a new StrategySpec for research"}
        actions={
          <Button
            variant="subtle"
            size="sm"
            leftSection={<IconChevronLeft size={15} />}
            onClick={() => navigate("/")}
          >
            Library
          </Button>
        }
      />

      <Grid gutter="lg">
        <Grid.Col span={{ base: 12, md: 8 }}>
          <Stack gap="lg" pos="relative">
            <LoadingOverlay visible={save.isPending} />

            {/* ── Identity ── */}
            <FormSection
              title="Identity"
              description="Name and hypothesis — the 'why' behind this edge."
              icon={<IconTag size={12} />}
            >
              <TextInput
                label="Name"
                description="Short slug used in filenames and vault paths."
                placeholder="es_double7s_mr"
                required
                value={spec.name}
                onChange={(e) => update("name", e.currentTarget.value)}
              />
              <TextInput
                label="Author"
                placeholder="your name"
                value={spec.author}
                onChange={(e) => update("author", e.currentTarget.value)}
              />
              <Textarea
                label="Hypothesis"
                description="The economic rationale — why the edge exists and should persist."
                autosize
                minRows={2}
                value={spec.hypothesis}
                onChange={(e) => update("hypothesis", e.currentTarget.value)}
              />
            </FormSection>

            {/* ── Universe & timeframe ── */}
            <FormSection
              title="Universe & timeframe"
              description="Instruments and when the strategy runs."
              icon={<IconWorld size={12} />}
            >
              <MultiSelect
                label="Tickers"
                description="The instruments this spec will be tested against."
                placeholder="ES, NQ, …"
                searchable
                data={s.tickers}
                value={spec.tickers}
                onChange={(value) => update("tickers", value)}
              />
              <div>
                <Text size="sm" fw={500} mb={6}>
                  Mode
                </Text>
                <SegmentedControl
                  data={s.modes.map((m) => ({ value: m.value, label: m.label }))}
                  value={spec.mode}
                  onChange={(value) => update("mode", value)}
                />
              </div>
              <Select
                label="Timeframe"
                description="Bar resolution — changes available options based on mode."
                data={timeframeOptions ?? []}
                value={spec.timeframe}
                onChange={(value) => value && update("timeframe", value)}
              />
            </FormSection>

            {/* ── Signal ── */}
            <FormSection
              title="Signal"
              description="Bias node module and its parameter sweep grid."
              icon={<IconTimeline size={12} />}
            >
              <Group align="flex-end" gap="xs">
                <Select
                  label="Module"
                  description="The bias node (50+ technical indicators available)."
                  placeholder="search nodes…"
                  searchable
                  data={moduleOptions}
                  value={spec.signal.module_name || null}
                  onChange={(value) =>
                    setSpec((sp) => ({ ...sp, signal: { ...sp.signal, module_name: value ?? "" } }))
                  }
                  style={{ flex: 1 }}
                />
                <Tooltip
                  label={
                    !spec.signal.module_name
                      ? "Select a module first"
                      : "Prefill param grid with module defaults"
                  }
                  withArrow
                >
                  <Button
                    variant="light"
                    leftSection={<IconWand size={15} />}
                    disabled={!spec.signal.module_name || moduleDetail.isLoading}
                    onClick={loadModuleParams}
                  >
                    Prefill params
                  </Button>
                </Tooltip>
              </Group>
              {moduleDetail.data?.doc && (
                <Alert variant="light" color="gray" icon={<IconInfoCircle size={14} />} py={8} px={12}>
                  <Text size="xs" c="dimmed" lineClamp={3}>
                    {moduleDetail.data.doc}
                  </Text>
                </Alert>
              )}
              <ParamGridEditor rows={paramRows} onChange={setParamRows} maxCombos={s.max_grid_combos} paramChoices={paramChoices} />
              <Select
                label="Direction"
                description="Long, short, or both — filters signal polarity."
                data={selectData(s.directions)}
                value={spec.direction}
                onChange={(value) => value && update("direction", value)}
              />
            </FormSection>

            {/* ── Volatility scaling ── */}
            <FormSection
              title="Volatility scaling"
              description="F = signal · τ / σ. Controls EWSD σ blend used for forecast sizing."
              icon={<IconGauge size={12} />}
            >
              <Select
                label="Vol scaling"
                description="The volatility estimator applied to raw signals."
                data={selectData(s.vol_scalings)}
                value={spec.vol_scaling}
                onChange={(value) => value && update("vol_scaling", value)}
              />
              <Select
                label="Vol scaling model"
                description="Intraday σ source — only relevant for intraday mode."
                data={selectData(s.vol_scaling_models)}
                value={spec.vol_scaling_model}
                onChange={(value) => value && update("vol_scaling_model", value)}
              />
            </FormSection>

            {/* ── Risk & sizing ── */}
            <FormSection
              title="Risk & sizing"
              description="Target volatility and position constraints passed to the PositionSizer."
              icon={<IconChartBar size={12} />}
            >
              <Group grow>
                <NumberInput
                  label="Target vol (τ)"
                  description="Annualised vol target (e.g. 0.25 = 25%)."
                  step={0.01}
                  decimalScale={3}
                  value={spec.risk.target_vol}
                  onChange={(val) => setSpec((sp) => ({ ...sp, risk: { ...sp.risk, target_vol: Number(val) } }))}
                />
                <NumberInput
                  label="Forecast cap"
                  description="Hard cap on scaled forecast (default 2.0)."
                  step={0.5}
                  value={spec.risk.forecast_cap}
                  onChange={(val) => setSpec((sp) => ({ ...sp, risk: { ...sp.risk, forecast_cap: Number(val) } }))}
                />
              </Group>
              <Group grow>
                <NumberInput
                  label="Max position %"
                  description="Upper bound on position as % of capital."
                  step={0.5}
                  value={spec.risk.max_position_pct}
                  onChange={(val) =>
                    setSpec((sp) => ({ ...sp, risk: { ...sp.risk, max_position_pct: Number(val) } }))
                  }
                />
                <NumberInput
                  label="Buffer fraction"
                  description="Dead-zone buffer to reduce unnecessary rebalancing."
                  step={0.05}
                  decimalScale={2}
                  value={spec.risk.buffer_fraction}
                  onChange={(val) =>
                    setSpec((sp) => ({ ...sp, risk: { ...sp.risk, buffer_fraction: Number(val) } }))
                  }
                />
              </Group>
              <NumberInput
                label="Account capital"
                description="Notional capital used for contract sizing."
                thousandSeparator
                value={spec.account.capital}
                onChange={(val) =>
                  setSpec((sp) => ({ ...sp, account: { ...sp.account, capital: Number(val) } }))
                }
              />
            </FormSection>

            {/* ── Execution (validation only) ── */}
            <FormSection
              title="Validation execution (realistic sim)"
              description="Market or passive-limit fills for the realistic validation sim — ignored during exploration (which is frictionless intraday). No per-trade stops."
              icon={<IconBolt size={12} />}
            >
              <Alert variant="light" color="gray" icon={<IconInfoCircle size={14} />} py={8} px={12}>
                <Text size="xs" c="dimmed">
                  These controls only affect the realistic validation sim. Exploration runs both
                  result lanes (Futures + CFD) on the same frictionless-intraday signal.
                </Text>
              </Alert>
              <Group grow>
                <Select
                  label="Entry policy"
                  data={selectData(s.order_policies)}
                  value={spec.execution.entry_policy}
                  onChange={(value) =>
                    value && setSpec((sp) => ({ ...sp, execution: { ...sp.execution, entry_policy: value } }))
                  }
                />
                <Select
                  label="Exit policy"
                  data={selectData(s.order_policies)}
                  value={spec.execution.exit_policy}
                  onChange={(value) =>
                    value && setSpec((sp) => ({ ...sp, execution: { ...sp.execution, exit_policy: value } }))
                  }
                />
              </Group>
              <Group grow>
                <Select
                  label="Unfilled limit"
                  description="What to do when a limit order expires unfilled."
                  data={selectData(s.unfilled_limit_policies)}
                  value={spec.execution.unfilled_limit}
                  onChange={(value) =>
                    value && setSpec((sp) => ({ ...sp, execution: { ...sp.execution, unfilled_limit: value } }))
                  }
                />
                <Select
                  label="Holding"
                  description="Validation-only realism switch: intraday vs overnight holding constraint (exploration is always frictionless intraday)."
                  data={selectData(s.holdings)}
                  value={spec.execution.holding}
                  onChange={(value) =>
                    value && setSpec((sp) => ({ ...sp, execution: { ...sp.execution, holding: value } }))
                  }
                />
              </Group>
              <Select
                label="Fill feed"
                description={
                  v?.derived
                    ? `Resolves to: ${v.derived.resolved_fill_feed}${v.derived.uses_limit ? " (uses limit)" : ""}`
                    : "Derived from entry / exit leg policies."
                }
                data={selectData(s.fill_feeds)}
                value={spec.execution.fill_feed}
                onChange={(value) =>
                  value && setSpec((sp) => ({ ...sp, execution: { ...sp.execution, fill_feed: value } }))
                }
              />
            </FormSection>

            {/* ── Research windows ── */}
            <FormSection
              title="Research windows"
              description="Override default train / validation / test splits. Off = per-mode defaults."
              icon={<IconCalendarStats size={12} />}
            >
              <Switch
                label="Use custom train / validation / test windows"
                checked={customWindows}
                onChange={(e) => {
                  const on = e.currentTarget.checked;
                  setCustomWindows(on);
                  if (on && !spec.windows) {
                    update("windows", {
                      train: { start: "2000-01-01", end: "2018-12-31" },
                      validation: { start: "2019-01-01", end: "2022-12-31" },
                      test: { start: "2023-01-01", end: "2026-01-01" },
                    });
                  }
                }}
              />
              <Collapse in={customWindows}>
                {spec.windows && (
                  <Stack gap="sm" pt={4}>
                    {(["train", "validation", "test"] as const).map((w) => (
                      <Group key={w} grow>
                        <TextInput
                          label={`${w.charAt(0).toUpperCase() + w.slice(1)} start`}
                          placeholder="YYYY-MM-DD"
                          value={spec.windows![w].start}
                          onChange={(e) =>
                            update("windows", {
                              ...spec.windows!,
                              [w]: { ...spec.windows![w], start: e.currentTarget.value },
                            })
                          }
                        />
                        <TextInput
                          label={`${w.charAt(0).toUpperCase() + w.slice(1)} end`}
                          placeholder="YYYY-MM-DD"
                          value={spec.windows![w].end}
                          onChange={(e) =>
                            update("windows", {
                              ...spec.windows!,
                              [w]: { ...spec.windows![w], end: e.currentTarget.value },
                            })
                          }
                        />
                      </Group>
                    ))}
                  </Stack>
                )}
              </Collapse>
            </FormSection>

            {/* ── Vault target ── */}
            <FormSection
              title="Vault target"
              description="Where this strategy lands on approval. Promotion is a separate gated step."
              icon={<IconLayersLinked size={12} />}
            >
              <Select
                label="Weight hierarchy group (sleeve)"
                description="The named portfolio sleeve in the global weight hierarchy."
                searchable
                data={s.vault_sleeves}
                value={spec.vault.weight_hierarchy_group}
                onChange={(value) =>
                  value && setSpec((sp) => ({ ...sp, vault: { ...sp.vault, weight_hierarchy_group: value } }))
                }
              />
              <TextInput
                label="Ensemble name"
                description="Leaf directory name inside the sleeve — usually matches the spec name."
                placeholder="es_double7s_mr"
                value={spec.vault.ensemble_name}
                onChange={(e) =>
                  setSpec((sp) => ({ ...sp, vault: { ...sp.vault, ensemble_name: e.currentTarget.value } }))
                }
              />
            </FormSection>
          </Stack>
        </Grid.Col>

        {/* ── Sticky right rail ── */}
        <Grid.Col span={{ base: 12, md: 4 }}>
          <Stack gap="md" style={{ position: "sticky", top: 76 }}>
            <ValidationPanel
              validation={v}
              loading={validation.isFetching}
              maxCombos={s.max_grid_combos}
            />

            <Card withBorder radius="md" padding="md">
              <Stack gap="xs">
                <Button
                  fullWidth
                  leftSection={<IconDeviceFloppy size={15} />}
                  disabled={!v?.valid}
                  loading={save.isPending}
                  onClick={() => persist()}
                >
                  Save spec
                </Button>
                <Button
                  fullWidth
                  variant="light"
                  leftSection={<IconPlayerPlay size={15} />}
                  disabled={!v?.valid}
                  loading={save.isPending || startRun.isPending}
                  onClick={saveAndRun}
                >
                  Save &amp; run exploration
                </Button>
                <Divider />
                <Button
                  fullWidth
                  variant="subtle"
                  color="gray"
                  leftSection={<IconCode size={15} />}
                  onClick={() => setShowJson((x) => !x)}
                >
                  {showJson ? "Hide" : "View"} JSON
                </Button>
              </Stack>
            </Card>

            <Collapse in={showJson}>
              <JsonInput value={JSON.stringify(effectiveSpec, null, 2)} autosize minRows={10} readOnly />
            </Collapse>
          </Stack>
        </Grid.Col>
      </Grid>
    </Stack>
  );
}

// ---------------------------------------------------------------------------
// Validation panel
// ---------------------------------------------------------------------------

function ValidationPanel({
  validation,
  loading,
  maxCombos,
}: {
  validation?: SpecValidation;
  loading: boolean;
  maxCombos: number;
}) {
  const isValid = validation?.valid;
  const hasData = validation !== undefined;
  const color = isValid ? "teal" : hasData ? "red" : "gray";

  const combos = validation?.derived?.num_combos ?? 0;
  const pct = maxCombos > 0 ? Math.min((combos / maxCombos) * 100, 100) : 0;
  const overCap = combos > maxCombos;
  const ringColor = overCap ? "red" : pct > 75 ? "orange" : "teal";

  return (
    <Card withBorder radius="md" padding="md">
      {/* Header */}
      <Group justify="space-between" mb="sm" align="center">
        <Group gap="xs" align="center">
          <ThemeIcon
            size="sm"
            variant="light"
            color={color}
            radius="xl"
          >
            {isValid ? (
              <IconCircleCheck size={13} />
            ) : (
              <IconAlertTriangle size={13} />
            )}
          </ThemeIcon>
          <Text fw={600} size="sm">
            {isValid ? "Valid spec" : hasData ? "Invalid spec" : "Validating…"}
          </Text>
        </Group>
        {loading && <Loader size="xs" />}
      </Group>

      {/* Error */}
      {validation?.error && (
        <Code block style={{ whiteSpace: "pre-wrap", fontSize: 11 }} mb="sm">
          {validation.error}
        </Code>
      )}

      {/* Derived info */}
      {validation?.derived && (
        <Stack gap="sm">
          <Divider />

          {/* Combo count with ring progress */}
          <Group gap="sm" align="center">
            <Tooltip
              label={
                overCap
                  ? `${fmtInt(combos)} combos exceeds cap of ${fmtInt(maxCombos)}`
                  : `${fmtInt(combos)} of ${fmtInt(maxCombos)} max combos`
              }
              withArrow
            >
              <Box style={{ cursor: "default" }}>
                <RingProgress
                  size={52}
                  thickness={5}
                  roundCaps
                  sections={[{ value: combos === 0 ? 0 : pct, color: ringColor }]}
                  label={
                    <Text size="xs" fw={700} ta="center" c={ringColor}>
                      {combos === 0 ? "—" : fmtInt(combos)}
                    </Text>
                  }
                />
              </Box>
            </Tooltip>
            <Stack gap={2}>
              <Text size="xs" fw={600}>
                {fmtInt(combos)} combo{combos === 1 ? "" : "s"}
              </Text>
              <Text size="xs" c="dimmed">
                cap: {fmtInt(maxCombos)}
              </Text>
              {overCap && (
                <Badge color="red" variant="filled" size="xs">
                  Over cap
                </Badge>
              )}
            </Stack>
          </Group>

          {/* Derived chips */}
          <Stack gap={6}>
            <Group gap="xs" wrap="wrap">
              <Tooltip label="Resolved fill feed (derived from entry/exit policies)" withArrow>
                <Badge variant="light" color="gray" size="sm" style={{ cursor: "default" }}>
                  fill: {validation.derived.resolved_fill_feed}
                </Badge>
              </Tooltip>
              {validation.derived.uses_limit && (
                <Badge variant="light" color="orange" size="sm" style={{ cursor: "default" }}>
                  uses limit
                </Badge>
              )}
            </Group>

            <Group gap="xs" wrap="wrap">
              <Tooltip
                label={
                  validation.derived.windows_explicit
                    ? "Custom windows set"
                    : "Using per-mode default windows"
                }
                withArrow
              >
                <Badge variant="dot" color={validation.derived.windows_explicit ? "brand" : "gray"} size="sm" style={{ cursor: "default" }}>
                  {validation.derived.windows_explicit ? "custom windows" : "default windows"}
                </Badge>
              </Tooltip>
            </Group>

            {/* Window spans if available */}
            {validation.derived.windows && (
              <Stack gap={2} mt={2}>
                {(["train", "validation", "test"] as const).map((w) => {
                  const [s, e] = validation.derived!.windows![w];
                  return (
                    <Group key={w} gap={4} justify="space-between">
                      <Text size="xs" c="dimmed" tt="capitalize" w={70}>
                        {w}
                      </Text>
                      <Text size="xs" ff="monospace" c="dimmed">
                        {s.slice(0, 10)} → {e.slice(0, 10)}
                      </Text>
                    </Group>
                  );
                })}
              </Stack>
            )}
          </Stack>
        </Stack>
      )}
    </Card>
  );
}
