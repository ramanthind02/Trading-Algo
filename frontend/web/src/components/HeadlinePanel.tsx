import {
  Badge,
  Card,
  Divider,
  Group,
  SimpleGrid,
  Stack,
  Text,
  ThemeIcon,
} from "@mantine/core";
import { IconAlertTriangle, IconInfoCircle } from "@tabler/icons-react";

import { fmtNum, fmtPct, metricColor } from "../lib/format";

type Headline = Record<string, string | number | boolean | null>;

interface StatDef {
  key: string;
  label: string;
  sublabel?: string;
  /** Format the raw number to a display string. */
  render: (v: number) => string;
  /** Threshold at or above which the value is considered "good" (teal). */
  goodAt: number;
  /** Threshold below which the value is considered "bad" (red). Optional. */
  badBelow?: number;
}

const STATS: StatDef[] = [
  {
    key: "nw_adjusted_sharpe_annualized",
    label: "NW Sharpe",
    sublabel: "annualised",
    render: (v) => fmtNum(v),
    goodAt: 0.5,
    badBelow: 0.2,
  },
  {
    key: "nw_t_adjusted",
    label: "NW t-stat",
    sublabel: "adjusted",
    render: (v) => fmtNum(v),
    goodAt: 2.0,
    badBelow: 1.0,
  },
  {
    key: "dsr_probability",
    label: "DSR",
    sublabel: "prob. of skill",
    render: (v) => fmtPct(v, 1),
    goodAt: 0.9,
    badBelow: 0.7,
  },
  {
    key: "n_effective",
    label: "N_eff",
    sublabel: "independent obs.",
    render: (v) => fmtNum(v, 0),
    goodAt: 50,
    badBelow: 20,
  },
  {
    key: "mean_pairwise_corr",
    label: "Mean pairwise ρ",
    sublabel: "signal correlation",
    render: (v) => fmtNum(v, 3),
    // Low correlation is good (more diversification)
    goodAt: -Infinity,
    badBelow: 0.7,
  },
  {
    key: "rolling_positive_fraction",
    label: "Rolling +frac",
    sublabel: "% positive windows",
    render: (v) => fmtPct(v, 1),
    goodAt: 0.6,
    badBelow: 0.45,
  },
];

function StatCard({ def, value }: { def: StatDef; value: number | null | undefined }) {
  const displayVal =
    value === null || value === undefined || typeof value !== "number"
      ? "—"
      : def.render(value);

  // Special case: for mean_pairwise_corr, low is good; derive color differently.
  let color: string;
  if (def.key === "mean_pairwise_corr") {
    if (value === null || value === undefined) {
      color = "gray";
    } else if ((value as number) < 0.3) {
      color = "teal";
    } else if ((value as number) >= 0.7) {
      color = "red";
    } else {
      color = "gray";
    }
  } else {
    color = metricColor(value as number, def.goodAt, def.badBelow);
  }

  return (
    <Card withBorder padding="sm" radius="md">
      <Stack gap={2}>
        <Text size="xs" c="dimmed" lh={1.2}>
          {def.label}
        </Text>
        {def.sublabel && (
          <Text size="xs" c="dimmed" fs="italic" lh={1.1} style={{ opacity: 0.6 }}>
            {def.sublabel}
          </Text>
        )}
        <Text size="lg" fw={700} c={color} mt={4}>
          {displayVal}
        </Text>
      </Stack>
    </Card>
  );
}

export function HeadlinePanel({ headline }: { headline: Headline }) {
  const bestLabel = headline.best_param_combo_label;
  const selMetric = headline.selection_metric;
  const nCombos = headline.n_combinations;
  const rawSharpe = headline.raw_sharpe_annualized;
  const cusum = headline.cusum_break_detected;
  const interpretation = headline.interpretation;

  return (
    <Stack gap="md">
      {/* ── Summary line ── */}
      <Group gap="sm" wrap="wrap" align="center">
        <Text size="sm" c="dimmed" fw={500}>
          Best combo
        </Text>
        <Badge variant="light" color="indigo" radius="sm" size="md">
          {bestLabel != null ? String(bestLabel) : "—"}
        </Badge>
        {selMetric != null && (
          <Text size="xs" c="dimmed">
            selected on{" "}
            <Text span fw={600} c="dimmed">
              {String(selMetric)}
            </Text>
          </Text>
        )}
        {nCombos != null && (
          <Text size="xs" c="dimmed">
            · {String(nCombos)} combos
          </Text>
        )}
        {rawSharpe != null && typeof rawSharpe === "number" && (
          <>
            <Divider orientation="vertical" />
            <Text size="xs" c="dimmed">
              Raw Sharpe{" "}
              <Text span fw={600} c="dimmed">
                {fmtNum(rawSharpe)}
              </Text>
            </Text>
          </>
        )}
        {cusum === true && (
          <Badge
            variant="light"
            color="orange"
            leftSection={<IconAlertTriangle size={11} />}
            size="sm"
            radius="sm"
          >
            CUSUM break
          </Badge>
        )}
      </Group>

      {/* ── Stat cards ── */}
      <SimpleGrid cols={{ base: 2, sm: 3, md: 6 }} spacing="xs">
        {STATS.map((s) => {
          const raw = headline[s.key];
          const numVal = typeof raw === "number" ? raw : null;
          return <StatCard key={s.key} def={s} value={numVal} />;
        })}
      </SimpleGrid>

      {/* ── Interpretation callout ── */}
      {interpretation && (
        <Card
          withBorder
          padding="sm"
          radius="md"
          style={({ colors }) => ({
            borderColor: colors.indigo[8],
            backgroundColor: "rgba(92, 108, 196, 0.08)",
          })}
        >
          <Group gap="xs" wrap="nowrap" align="flex-start">
            <ThemeIcon color="indigo" variant="light" size="sm" radius="xl" mt={2}>
              <IconInfoCircle size={13} />
            </ThemeIcon>
            <Text size="sm" c="dimmed" lh={1.55}>
              {String(interpretation)}
            </Text>
          </Group>
        </Card>
      )}
    </Stack>
  );
}
