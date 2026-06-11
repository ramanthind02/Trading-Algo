import {
  Badge,
  Box,
  Card,
  Divider,
  Grid,
  Group,
  Loader,
  SegmentedControl,
  Stack,
  Text,
  ThemeIcon,
  Title,
} from "@mantine/core";
import { IconChartLine, IconLayoutGrid, IconMountain } from "@tabler/icons-react";
import { useMemo, useState } from "react";

import { useRunResults } from "../api/hooks";
import type { LaneResults } from "../api/types";
import { EquityChart } from "./EquityChart";
import { GridTable } from "./GridTable";
import { HeadlinePanel } from "./HeadlinePanel";
import { PlateauChart } from "./PlateauChart";
import { ValidationResultsView } from "./ValidationResultsView";

// Human labels for the lane keys the backend emits.
const LANE_LABELS: Record<string, string> = {
  futures: "Futures",
  cfd: "CFD",
  default: "Results",
};

function laneLabel(name: string): string {
  return LANE_LABELS[name] ?? name.charAt(0).toUpperCase() + name.slice(1);
}

/** True if a lane is a validation lane (has validation-specific output). */
function isValidationLane(lane: LaneResults | undefined): boolean {
  return Boolean(lane?.validation_images !== undefined);
}

/** True if a lane has anything worth rendering. */
function laneHasData(lane: LaneResults | undefined): boolean {
  if (!lane) return false;
  if (isValidationLane(lane)) {
    return Boolean(lane.headline) || (lane.validation_images?.length ?? 0) > 0;
  }
  return Boolean(lane.headline) || lane.grid.available || lane.plateau.available || lane.equity.available;
}

// Default export so it can be React.lazy()-loaded (keeps Plotly out of the initial bundle).
export default function ResultsTab({ runId }: { runId: string }) {
  const results = useRunResults(runId, true);

  // Lane order with a stable fallback if the backend omits lane_order.
  const laneOrder = useMemo(() => {
    const data = results.data;
    if (!data) return [];
    if (data.lane_order?.length) return data.lane_order.filter((l) => l in data.lanes);
    return Object.keys(data.lanes ?? {});
  }, [results.data]);

  const primary = results.data?.primary_lane && laneOrder.includes(results.data.primary_lane)
    ? results.data.primary_lane
    : laneOrder[0] ?? null;

  const [active, setActive] = useState<string | null>(null);
  const selected = active && laneOrder.includes(active) ? active : primary;

  if (results.isLoading)
    return (
      <Group justify="center" mt="xl" gap="xs">
        <Loader size="sm" />
        <Text c="dimmed" size="sm">
          Loading results…
        </Text>
      </Group>
    );

  if (results.isError || !results.data)
    return (
      <Card withBorder padding="lg" radius="md">
        <Group gap="xs">
          <Text c="red" size="sm">
            Failed to load results. The run output may be incomplete or corrupt.
          </Text>
        </Group>
      </Card>
    );

  if (laneOrder.length === 0 || !selected)
    return (
      <Card withBorder padding="lg" radius="md">
        <Text c="dimmed" size="sm">
          No result lanes were produced for this run.
        </Text>
      </Card>
    );

  const lane = results.data.lanes[selected];
  const multiLane = laneOrder.length > 1;

  return (
    <Stack gap="xl">
      {multiLane && (
        <Group justify="space-between" align="center">
          <SegmentedControl
            value={selected}
            onChange={setActive}
            data={laneOrder.map((name) => ({
              value: name,
              label: laneHasData(results.data!.lanes[name]) ? laneLabel(name) : `${laneLabel(name)} (empty)`,
            }))}
          />
          <Group gap="xs">
            <Text size="xs" c="dimmed">
              Same signal (futures-additive) · two return feeds
            </Text>
            {selected === primary && (
              <Badge size="xs" variant="light" color="indigo">
                primary
              </Badge>
            )}
          </Group>
        </Group>
      )}

      {laneHasData(lane) ? (
        isValidationLane(lane) ? (
          <ValidationResultsView lane={lane} />
        ) : (
          <LaneResultsView lane={lane} />
        )
      ) : (
        <Card withBorder padding="lg" radius="md">
          <Text c="dimmed" size="sm">
            The {laneLabel(selected)} lane produced no results
            {selected === "cfd" ? " — no real CFD return feed for these tickers." : "."}
          </Text>
        </Card>
      )}
    </Stack>
  );
}

/** Renders a single result lane (headline + plateau + equity + grid). */
function LaneResultsView({ lane }: { lane: LaneResults }) {
  const { headline, plateau, equity, grid } = lane;

  return (
    <Stack gap="xl">
      {/* ── Section 1: Headline metrics ─────────────────────────── */}
      {headline && (
        <Card withBorder padding="lg" radius="md">
          <SectionHeader icon={<IconChartLine size={15} />} title="Best combo summary" />
          <Box mt="md">
            <HeadlinePanel headline={headline} />
          </Box>
        </Card>
      )}

      {/* ── Section 2: Charts ───────────────────────────────────── */}
      <Grid gutter="lg">
        <Grid.Col span={{ base: 12, lg: 6 }}>
          <Card withBorder padding="lg" radius="md" h="100%">
            <SectionHeader icon={<IconMountain size={15} />} title="Parameter plateau" />
            <Divider mt="sm" mb="md" />
            {plateau.available && plateau.points ? (
              <PlateauChart
                swept={plateau.swept_params ?? []}
                points={plateau.points}
                metricOptions={plateau.metric_options ?? ["t_stat"]}
              />
            ) : (
              <EmptyState message="No parameter-sensitivity data for this lane." />
            )}
          </Card>
        </Grid.Col>

        <Grid.Col span={{ base: 12, lg: 6 }}>
          <Card withBorder padding="lg" radius="md" h="100%">
            <SectionHeader icon={<IconChartLine size={15} />} title="Equity curves" />
            <Divider mt="sm" mb="md" />
            {equity.available ? (
              <EquityChart series={equity.series} combos={equity.combos} tickers={equity.tickers} />
            ) : (
              <EmptyState message="No equity curve data for this lane." />
            )}
          </Card>
        </Grid.Col>
      </Grid>

      {/* ── Section 3: Per-combo grid table ─────────────────────── */}
      <Card withBorder padding="lg" radius="md">
        <SectionHeader icon={<IconLayoutGrid size={15} />} title="Per-combo results" />
        <Text size="xs" c="dimmed" mt={2}>
          Click column headers to sort. Best combo highlighted in green.
        </Text>
        <Divider mt="sm" mb="md" />
        {grid.available ? (
          <GridTable rows={grid.rows} paramKeys={grid.param_keys} />
        ) : (
          <EmptyState message="No grid data available." />
        )}
      </Card>
    </Stack>
  );
}

/** Small labelled section header with icon. */
function SectionHeader({ icon, title }: { icon: React.ReactNode; title: string }) {
  return (
    <Group gap="xs" align="center">
      <ThemeIcon size={22} variant="light" color="indigo" radius="sm">
        {icon}
      </ThemeIcon>
      <Title order={5} style={{ lineHeight: 1 }}>
        {title}
      </Title>
    </Group>
  );
}

/** Graceful placeholder for sections with no data. */
function EmptyState({ message }: { message: string }) {
  return (
    <Group justify="center" align="center" py="xl">
      <Text size="sm" c="dimmed">
        {message}
      </Text>
    </Group>
  );
}
