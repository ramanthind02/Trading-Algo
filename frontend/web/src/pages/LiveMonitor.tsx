import {
  Alert,
  Badge,
  Button,
  Card,
  Group,
  Loader,
  Modal,
  SegmentedControl,
  SimpleGrid,
  Stack,
  Table,
  Text,
  TextInput,
  Title,
} from "@mantine/core";
import { notifications } from "@mantine/notifications";
import { IconAlertTriangle, IconBolt, IconCircleFilled, IconPlugConnectedX } from "@tabler/icons-react";
import { useEffect, useMemo, useState } from "react";

import { useDataAccounts, useFlatten, useLiveBrokers, useLiveEquity, useLiveRisk, useLiveSnapshot } from "../api/hooks";
import type { LiveSnapshot } from "../api/types";
import { Plot, baseLayout, plotConfig } from "../components/Plot";
import { PositionsTable } from "../components/PositionsTable";
import { RiskGauges } from "../components/RiskGauges";
import { PageHeader } from "../components/ui/PageHeader";
import { fmtNum } from "../lib/format";

const TIER_COLOR: Record<string, string> = { sandbox: "gray", demo: "blue", live: "red" };

function ageLabel(seconds: number | null): string {
  if (seconds == null) return "no data";
  if (seconds < 60) return `${Math.round(seconds)}s ago`;
  if (seconds < 3600) return `${Math.round(seconds / 60)}m ago`;
  return `${Math.round(seconds / 3600)}h ago`;
}

/** One labelled metric card (equity / balance / floating P&L / leverage). */
function Stat({ label, value, color }: { label: string; value: string; color?: string }) {
  return (
    <Card withBorder radius="md" padding="md">
      <Text size="xs" c="dimmed" fw={600} tt="uppercase" style={{ letterSpacing: 0.4 }}>
        {label}
      </Text>
      <Text size="xl" fw={700} ff="monospace" c={color} mt={4}>
        {value}
      </Text>
    </Card>
  );
}

export function LiveMonitor() {
  const brokers = useLiveBrokers();
  const [broker, setBroker] = useState<string | null>(null);

  // Default to the first online broker, else the first known broker. Self-heal if
  // the selected broker disappears from config while selected.
  useEffect(() => {
    if (!brokers.data) return;
    if (broker && !brokers.data.some((b) => b.broker === broker)) {
      setBroker(null);
      return;
    }
    if (broker || brokers.data.length === 0) return;
    const online = brokers.data.find((b) => b.online);
    setBroker((online ?? brokers.data[0]).broker);
  }, [brokers.data, broker]);

  const snapshot = useLiveSnapshot(broker);
  const risk = useLiveRisk(broker);
  const equity = useLiveEquity(broker);

  const selector = (
    <SegmentedControl
      value={broker ?? ""}
      onChange={setBroker}
      data={(brokers.data ?? []).map((b) => ({
        value: b.broker,
        label: (
          <Group gap={6} wrap="nowrap">
            <IconCircleFilled size={9} color={b.online ? "var(--mantine-color-teal-5)" : "var(--mantine-color-gray-6)"} />
            <span>{b.broker}</span>
          </Group>
        ),
      }))}
    />
  );

  return (
    <Stack gap="xl">
      <PageHeader
        title="Live monitor"
        subtitle="Read-only feed published by the running vault node (no MT5 contention). Positions, portfolio health vs prop-firm limits, and a node-mediated flatten-all kill switch."
        actions={brokers.data && brokers.data.length > 0 ? selector : undefined}
      />

      {brokers.isLoading && (
        <Group justify="center" py="xl">
          <Loader size="sm" />
        </Group>
      )}

      {broker && (snapshot.isError || (snapshot.data && !snapshot.data.online)) && (
        <Alert color="orange" variant="light" icon={<IconPlugConnectedX size={18} />} title={`No live feed for ${broker}`}>
          The vault node for <b>{broker}</b> isn't publishing a fresh snapshot
          {snapshot.data ? ` (last seen ${ageLabel(snapshot.data.age_seconds)})` : ""}. Start it with{" "}
          <Text span ff="monospace" size="sm">
            python -m deployment.live.run_vault_sandbox --broker {broker} --exec-tier demo --arm --live
          </Text>
          . While the node is offline the dashboard cannot flatten — use{" "}
          <Text span ff="monospace" size="sm">
            python -m deployment.live.manual_trade flatten --broker {broker}
          </Text>{" "}
          on a free terminal.
        </Alert>
      )}

      {broker && snapshot.data && snapshot.data.online && (
        <LiveBody
          broker={broker}
          snapshot={snapshot.data}
          risk={risk.data}
          equitySeries={equity.data?.series ?? []}
        />
      )}
    </Stack>
  );
}

function LiveBody({
  broker,
  snapshot,
  risk,
  equitySeries,
}: {
  broker: string;
  snapshot: LiveSnapshot;
  risk: ReturnType<typeof useLiveRisk>["data"];
  equitySeries: { ts: string; equity: number }[];
}) {
  const acct = snapshot.account;
  const tier = snapshot.exec_tier;

  const registryAccounts = useDataAccounts();
  const registryAccount = registryAccounts.data?.find(
    (a) => a.broker === broker && String(a.login) === String(acct.login),
  );

  const equityTraces = useMemo(() => {
    if (equitySeries.length === 0) return [];
    return [
      {
        type: "scatter" as const,
        mode: "lines" as const,
        name: "equity",
        x: equitySeries.map((p) => p.ts),
        y: equitySeries.map((p) => p.equity),
        line: { color: "#20c997", width: 2 },
        hovertemplate: "%{x|%Y-%m-%d %H:%M}<br>Equity: %{y:,.2f}<extra></extra>",
      },
    ];
  }, [equitySeries]);

  return (
    <Stack gap="xl">
      {/* ── status pills ── */}
      <Group gap="sm">
        <Badge color={TIER_COLOR[tier] ?? "gray"} variant="filled" size="lg" radius="sm">
          {tier.toUpperCase()}
        </Badge>
        <Badge color="teal" variant="light" leftSection={<IconCircleFilled size={9} />}>
          live · {ageLabel(snapshot.age_seconds)}
        </Badge>
        <Badge color={snapshot.halted ? "red" : "indigo"} variant={snapshot.halted ? "filled" : "light"}>
          {snapshot.halted ? "HALTED" : snapshot.strategy_state}
        </Badge>
        {!snapshot.ready_to_trade && !snapshot.halted && (
          <Badge color="yellow" variant="light">
            not ready (warmup/startup)
          </Badge>
        )}
        {!snapshot.marks_fresh && snapshot.positions.length > 0 && (
          <Badge color="orange" variant="light">
            marks stale (between windows)
          </Badge>
        )}
        <Text size="xs" c="dimmed">
          {acct.server || "—"} · login {acct.login || "—"}
        </Text>
        {registryAccount && (
          <>
            <Badge variant="outline" size="sm" color="indigo" radius="sm">
              {registryAccount.program_phase}
            </Badge>
            {registryAccount.status !== "active" && (
              <Badge variant="light" size="sm" color="red" radius="sm">
                {registryAccount.status}
              </Badge>
            )}
          </>
        )}
      </Group>

      {snapshot.halted && (
        <Alert color="red" variant="light" icon={<IconAlertTriangle size={18} />} title="Node halted">
          The node flattened and stopped trading (kill switch fired). It will NOT re-open positions.
          Restart it to resume:{" "}
          <Text span ff="monospace" size="sm">
            run_vault_sandbox --broker {broker} ...
          </Text>
        </Alert>
      )}

      {/* ── account stats ── */}
      <SimpleGrid cols={{ base: 2, sm: 4 }} spacing="md">
        <Stat label="Equity" value={fmtNum(acct.equity, 2)} />
        <Stat label="Balance" value={fmtNum(acct.balance, 2)} />
        <Stat
          label={`Floating P&L (${acct.currency})`}
          value={fmtNum(acct.floating_pnl, 2)}
          color={acct.floating_pnl == null ? undefined : acct.floating_pnl >= 0 ? "teal" : "red"}
        />
        <Stat label="Leverage" value={acct.leverage_in_use == null ? "—" : `${fmtNum(acct.leverage_in_use, 2)}x`} />
      </SimpleGrid>

      {/* ── prop-firm health ── */}
      <Stack gap="sm">
        <Title order={4}>Portfolio health</Title>
        {risk ? <RiskGauges risk={risk} /> : <Loader size="sm" />}
      </Stack>

      {/* ── equity curve ── */}
      <Stack gap="sm">
        <Title order={4}>Equity</Title>
        {equityTraces.length > 0 ? (
          <Plot
            data={equityTraces}
            layout={{ ...baseLayout, height: 280, xaxis: { ...baseLayout.xaxis, type: "date" }, yaxis: { ...baseLayout.yaxis, tickformat: ",.0f" } }}
            config={plotConfig}
            style={{ width: "100%" }}
          />
        ) : (
          <Text size="sm" c="dimmed">
            No equity history yet — samples accumulate once the node has been running for a minute.
          </Text>
        )}
      </Stack>

      {/* ── positions ── */}
      <Stack gap="sm">
        <Group gap="xs">
          <Title order={4}>Open positions</Title>
          <Badge variant="light" color="gray" radius="sm">
            {snapshot.positions.length}
          </Badge>
          <Text size="xs" c="dimmed">
            gross notional {fmtNum(snapshot.gross_notional, 0)} {acct.currency}
          </Text>
        </Group>
        <PositionsTable positions={snapshot.positions} currency={acct.currency} />
      </Stack>

      {/* ── target vs actual ── */}
      <TargetVsActual snapshot={snapshot} />

      {/* ── warmup ── */}
      {snapshot.warmup && (
        <Stack gap="sm">
          <Group gap="xs">
            <Title order={4}>Signal readiness</Title>
            <Badge color={snapshot.warmup.ready ? "teal" : "yellow"} variant="light">
              {snapshot.warmup.ready ? "ready" : "warming up"}
            </Badge>
            {snapshot.warmup.as_of && (
              <Text size="xs" c="dimmed">
                as of {new Date(snapshot.warmup.as_of).toLocaleDateString()}
              </Text>
            )}
          </Group>
          <Group gap={6} wrap="wrap">
            {snapshot.warmup.per_ticker.map((t) => (
              <Badge
                key={t.ticker}
                color={t.ready ? "teal" : "gray"}
                variant={t.ready ? "light" : "outline"}
                radius="sm"
              >
                {t.ticker} · {t.bars}/{snapshot.warmup!.min_bars}
              </Badge>
            ))}
          </Group>
        </Stack>
      )}

      {/* ── controls ── */}
      <FlattenControl broker={broker} snapshot={snapshot} />
    </Stack>
  );
}

function TargetVsActual({ snapshot }: { snapshot: LiveSnapshot }) {
  if (snapshot.targets.length === 0) {
    return (
      <Stack gap="sm">
        <Title order={4}>Target vs actual</Title>
        <Text size="sm" c="dimmed">
          No target book yet (forecast warming up or not refreshed).
        </Text>
      </Stack>
    );
  }
  return (
    <Stack gap="sm">
      <Title order={4}>Target vs actual</Title>
      <Table.ScrollContainer minWidth={560}>
        <Table striped highlightOnHover withTableBorder fz="sm">
          <Table.Thead>
            <Table.Tr>
              <Table.Th>Ticker</Table.Th>
              <Table.Th ta="right">Target frac</Table.Th>
              <Table.Th ta="right">Forecast</Table.Th>
              <Table.Th ta="right">Target lots</Table.Th>
              <Table.Th ta="right">Current lots</Table.Th>
              <Table.Th ta="right">Drift</Table.Th>
            </Table.Tr>
          </Table.Thead>
          <Table.Tbody>
            {snapshot.targets.map((t) => {
              const driftColor =
                t.drift == null ? undefined : Math.abs(t.drift) < 0.01 ? "dimmed" : "yellow";
              return (
                <Table.Tr key={t.canonical}>
                  <Table.Td fw={600}>{t.canonical}</Table.Td>
                  <Table.Td ta="right" ff="monospace">
                    {fmtNum(t.target_fraction, 3)}
                  </Table.Td>
                  <Table.Td ta="right" ff="monospace">
                    {fmtNum(t.forecast_score, 2)}
                  </Table.Td>
                  <Table.Td ta="right" ff="monospace">
                    {fmtNum(t.target_qty, 2)}
                  </Table.Td>
                  <Table.Td ta="right" ff="monospace">
                    {fmtNum(t.current_qty, 2)}
                  </Table.Td>
                  <Table.Td ta="right">
                    <Text component="span" c={driftColor} ff="monospace" size="sm">
                      {fmtNum(t.drift, 2)}
                    </Text>
                  </Table.Td>
                </Table.Tr>
              );
            })}
          </Table.Tbody>
        </Table>
      </Table.ScrollContainer>
    </Stack>
  );
}

function FlattenControl({ broker, snapshot }: { broker: string; snapshot: LiveSnapshot }) {
  const [open, setOpen] = useState(false);
  const [confirm, setConfirm] = useState("");
  const flatten = useFlatten();
  const isLive = snapshot.exec_tier.toLowerCase() === "live";
  const result = snapshot.last_command_result;

  const submit = () => {
    flatten.mutate(
      { broker, confirm },
      {
        onSuccess: (r) => {
          notifications.show({ message: r.note, color: "orange" });
          setOpen(false);
          setConfirm("");
        },
        onError: (err) => notifications.show({ message: String(err), color: "red" }),
      },
    );
  };

  return (
    <Card withBorder radius="md" padding="lg">
      <Stack gap="sm">
        <Group justify="space-between" wrap="nowrap">
          <Stack gap={2}>
            <Title order={4}>Controls</Title>
            <Text size="sm" c="dimmed">
              Flatten closes every position through the node and HALTS it (no re-open). Demo accounts only.
            </Text>
          </Stack>
          <Button
            color="red"
            leftSection={<IconBolt size={16} />}
            disabled={isLive || (snapshot.halted && snapshot.positions.length === 0)}
            onClick={() => setOpen(true)}
          >
            {snapshot.halted && snapshot.positions.length > 0 ? "Re-flatten (cleanup)" : "Flatten all"}
          </Button>
        </Group>

        {isLive && (
          <Alert color="red" variant="light" icon={<IconAlertTriangle size={16} />}>
            This is a live/funded account — flatten is disabled (demo-only). Manage it from the launching terminal.
          </Alert>
        )}

        {result && (
          <Alert
            color={
              result.status === "executed"
                ? "teal"
                : result.status === "rejected" || result.status === "partial"
                  ? "yellow"
                  : "red"
            }
            variant="light"
            title={`Last command: ${result.action} → ${result.status}`}
          >
            {result.detail}
            <Text size="xs" c="dimmed" mt={4}>
              {new Date(result.at).toLocaleString()}
            </Text>
          </Alert>
        )}
      </Stack>

      <Modal opened={open} onClose={() => setOpen(false)} title="Confirm flatten-all" centered>
        <Stack gap="md">
          <Alert color="red" variant="light" icon={<IconAlertTriangle size={16} />}>
            This will close <b>all {snapshot.positions.length} open position(s)</b> on{" "}
            <b>{broker}</b> at market and halt the node until restart. This cannot be undone.
          </Alert>
          <TextInput
            label={`Type "${broker}" to confirm`}
            placeholder={broker}
            value={confirm}
            onChange={(e) => setConfirm(e.currentTarget.value)}
            autoFocus
          />
          <Group justify="flex-end">
            <Button variant="default" onClick={() => setOpen(false)}>
              Cancel
            </Button>
            <Button
              color="red"
              leftSection={<IconBolt size={16} />}
              disabled={confirm !== broker}
              loading={flatten.isPending}
              onClick={submit}
            >
              Flatten all + halt
            </Button>
          </Group>
        </Stack>
      </Modal>
    </Card>
  );
}
