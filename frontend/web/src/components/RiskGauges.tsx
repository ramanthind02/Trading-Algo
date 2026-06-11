import { Alert, Card, Group, Progress, SimpleGrid, Stack, Text } from "@mantine/core";
import { IconInfoCircle } from "@tabler/icons-react";

import type { LiveRisk } from "../api/types";
import { fmtNum } from "../lib/format";

/** Utilisation → color: green under 60%, amber 60–90%, red ≥90% of a risk limit. */
function riskColor(usedPct: number | null): string {
  if (usedPct == null) return "gray";
  if (usedPct >= 90) return "red";
  if (usedPct >= 60) return "yellow";
  return "teal";
}

function Gauge({
  label,
  valueText,
  fillPct,
  color,
  sub,
}: {
  label: string;
  valueText: string;
  fillPct: number | null;
  color: string;
  sub?: string;
}) {
  return (
    <Card withBorder radius="md" padding="md">
      <Stack gap={6}>
        <Group justify="space-between" wrap="nowrap">
          <Text size="xs" c="dimmed" fw={600} tt="uppercase" style={{ letterSpacing: 0.4 }}>
            {label}
          </Text>
          <Text size="sm" fw={700} ff="monospace">
            {valueText}
          </Text>
        </Group>
        <Progress
          value={fillPct == null ? 0 : Math.min(100, Math.max(0, fillPct))}
          color={color}
          size="md"
          radius="sm"
        />
        {sub && (
          <Text size="xs" c="dimmed">
            {sub}
          </Text>
        )}
      </Stack>
    </Card>
  );
}

export function RiskGauges({ risk }: { risk: LiveRisk }) {
  const g = risk.gauges;

  if (!risk.has_limits) {
    return (
      <Alert color="gray" variant="light" icon={<IconInfoCircle size={16} />} title="No prop-firm limits">
        This broker is live-retail — no evaluation limits are configured, so daily-loss / drawdown /
        profit-target gauges don't apply. Equity and leverage are still shown above.
      </Alert>
    );
  }

  return (
    <Stack gap="sm">
      <SimpleGrid cols={{ base: 1, sm: 2, lg: 4 }} spacing="md">
        <Gauge
          label="Daily loss"
          valueText={`${fmtNum(g.daily_drawdown_pct, 2)}% / ${fmtNum(g.daily_limit_pct, 1)}%`}
          fillPct={g.daily_used_pct}
          color={riskColor(g.daily_used_pct)}
          sub={g.daily_headroom_pct != null ? `${fmtNum(g.daily_headroom_pct, 2)}% headroom` : "awaiting equity baseline"}
        />
        <Gauge
          label="Total drawdown"
          valueText={`${fmtNum(g.total_drawdown_pct, 2)}% / ${fmtNum(g.total_limit_pct, 1)}%`}
          fillPct={g.total_used_pct}
          color={riskColor(g.total_used_pct)}
          sub={g.total_headroom_pct != null ? `${fmtNum(g.total_headroom_pct, 2)}% headroom` : "awaiting equity baseline"}
        />
        <Gauge
          label="Profit target"
          valueText={`${fmtNum(g.total_pnl_pct, 2)}% / ${fmtNum(g.profit_target_pct, 1)}%`}
          fillPct={g.profit_progress_pct}
          color={(g.profit_progress_pct ?? 0) >= 100 ? "teal" : "indigo"}
          sub="progress to target"
        />
        <Gauge
          label="Leverage"
          valueText={`${fmtNum(g.leverage_in_use, 2)}x${g.max_leverage != null ? ` / ${g.max_leverage}x` : ""}`}
          fillPct={g.max_leverage ? ((g.leverage_in_use ?? 0) / g.max_leverage) * 100 : null}
          color={riskColor(g.max_leverage ? ((g.leverage_in_use ?? 0) / g.max_leverage) * 100 : null)}
        />
      </SimpleGrid>
      {risk.day_start_estimated && (
        <Alert color="yellow" variant="light" icon={<IconInfoCircle size={16} />}>
          Daily baseline is <b>estimated</b> — the node restarted across the daily reset, so the
          day-start equity could not be observed. Treat the daily-loss headroom as approximate until
          the next clean reset.
        </Alert>
      )}
      <Text size="xs" c="dimmed" fs="italic">
        Limits are dollar amounts = the firm's % of the initial balance; the daily anchor is the
        day-start balance/equity. Confirm the drawdown basis (static-from-initial vs trailing
        high-water) and the daily reset time against your firm's program rules before trusting the
        headroom.
      </Text>
    </Stack>
  );
}
