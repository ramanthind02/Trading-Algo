import {
  Alert,
  Badge,
  Card,
  Group,
  Loader,
  SimpleGrid,
  Stack,
  Table,
  Text,
  Title,
} from "@mantine/core";
import { IconDatabase } from "@tabler/icons-react";

import { useDataAccounts, useDataCoverage, useDataFreshness, useDataJobs } from "../api/hooks";
import type { DataAccount } from "../api/types";
import { PageHeader } from "../components/ui/PageHeader";
import { fmtInt } from "../lib/format";

const TIER_COLOR: Record<string, string> = { sandbox: "gray", demo: "blue", live: "red" };
const STATUS_COLOR: Record<string, string> = {
  active: "teal",
  retired: "gray",
  breached: "red",
  passed: "green",
};

function freshnessVerdict(coverageEnd: string | null): { color: string; label: string } {
  if (!coverageEnd) return { color: "gray", label: "unknown" };
  const diffDays = (Date.now() - new Date(coverageEnd).getTime()) / 86_400_000;
  if (diffDays <= 3) return { color: "teal", label: "fresh" };
  if (diffDays <= 7) return { color: "yellow", label: "stale" };
  return { color: "red", label: "old" };
}

function fmtDuration(started: string, finished: string | null): string {
  if (!finished) return "—";
  const secs = Math.round(
    (new Date(finished).getTime() - new Date(started).getTime()) / 1000,
  );
  if (secs < 0) return "—";
  if (secs < 60) return `${secs}s`;
  return `${Math.floor(secs / 60)}m ${secs % 60}s`;
}

export function DataCoveragePage() {
  const coverage = useDataCoverage();
  const freshness = useDataFreshness();
  const jobs = useDataJobs();
  const accounts = useDataAccounts();

  const freshnessMap = new Map(
    (freshness.data ?? []).map((s) => [s.store, s.coverage_end]),
  );

  return (
    <Stack gap="xl">
      <PageHeader
        title="Data coverage"
        subtitle="Per-store bar coverage, ingestion job history, and registered broker accounts."
      />

      {/* ── A: stores ── */}
      <Stack gap="sm">
        <Title order={4}>Stores</Title>
        {coverage.isLoading && (
          <Group justify="center" py="md">
            <Loader size="sm" />
          </Group>
        )}
        {coverage.isError && (
          <Alert color="red" icon={<IconDatabase size={16} />}>
            Could not load coverage data.
          </Alert>
        )}
        {!coverage.isLoading && coverage.data?.length === 0 && (
          <Text size="sm" c="dimmed">
            No stores found — registry DB may not exist yet.
          </Text>
        )}
        {coverage.data && coverage.data.length > 0 && (
          <>
            <Table.ScrollContainer minWidth={620}>
              <Table striped highlightOnHover withTableBorder fz="sm">
                <Table.Thead>
                  <Table.Tr>
                    <Table.Th>Store</Table.Th>
                    <Table.Th ta="right">Keys</Table.Th>
                    <Table.Th>Earliest</Table.Th>
                    <Table.Th>Latest</Table.Th>
                    <Table.Th ta="right">Total rows</Table.Th>
                    <Table.Th>Freshness</Table.Th>
                  </Table.Tr>
                </Table.Thead>
                <Table.Tbody>
                  {coverage.data.map((row) => {
                    const verdict = freshnessVerdict(
                      freshnessMap.get(row.store) ?? null,
                    );
                    return (
                      <Table.Tr key={row.store}>
                        <Table.Td ff="monospace">{row.store}</Table.Td>
                        <Table.Td ta="right" ff="monospace">
                          {fmtInt(row.key_count)}
                        </Table.Td>
                        <Table.Td ff="monospace">{row.earliest ?? "—"}</Table.Td>
                        <Table.Td ff="monospace">{row.latest ?? "—"}</Table.Td>
                        <Table.Td ta="right" ff="monospace">
                          {row.total_rows != null ? fmtInt(row.total_rows) : "—"}
                        </Table.Td>
                        <Table.Td>
                          <Badge color={verdict.color} variant="light" size="sm" radius="sm">
                            {verdict.label}
                          </Badge>
                        </Table.Td>
                      </Table.Tr>
                    );
                  })}
                </Table.Tbody>
              </Table>
            </Table.ScrollContainer>
            <Text size="xs" c="dimmed">
              MT5 timestamps are broker wall-clock (EET/EEST, not UTC). Freshness: teal ≤ 3 days,
              yellow ≤ 7 days, red older.
            </Text>
          </>
        )}
      </Stack>

      {/* ── B: job history ── */}
      <Stack gap="sm">
        <Title order={4}>Job history</Title>
        {jobs.isLoading && (
          <Group justify="center" py="md">
            <Loader size="sm" />
          </Group>
        )}
        {!jobs.isLoading && jobs.data?.length === 0 && (
          <Text size="sm" c="dimmed">
            No job records — registry DB may not exist yet.
          </Text>
        )}
        {jobs.data && jobs.data.length > 0 && (
          <Table.ScrollContainer minWidth={580}>
            <Table striped highlightOnHover withTableBorder fz="sm">
              <Table.Thead>
                <Table.Tr>
                  <Table.Th>Job</Table.Th>
                  <Table.Th>Started</Table.Th>
                  <Table.Th>Duration</Table.Th>
                  <Table.Th>Exit</Table.Th>
                  <Table.Th ta="right">Rows written</Table.Th>
                </Table.Tr>
              </Table.Thead>
              <Table.Tbody>
                {jobs.data.map((job) => (
                  <Table.Tr key={job.job_run_id}>
                    <Table.Td ff="monospace">{job.job_name}</Table.Td>
                    <Table.Td ff="monospace">
                      {new Date(job.started_at).toLocaleString()}
                    </Table.Td>
                    <Table.Td ff="monospace">
                      {fmtDuration(job.started_at, job.finished_at)}
                    </Table.Td>
                    <Table.Td>
                      {job.exit_code == null ? (
                        <Badge color="gray" variant="outline" size="sm" radius="sm">
                          running
                        </Badge>
                      ) : job.exit_code === 0 ? (
                        <Badge color="teal" variant="light" size="sm" radius="sm">
                          ok
                        </Badge>
                      ) : (
                        <Badge color="red" variant="light" size="sm" radius="sm">
                          {`fail (${job.exit_code})`}
                        </Badge>
                      )}
                    </Table.Td>
                    <Table.Td ta="right" ff="monospace">
                      {job.rows_written != null ? fmtInt(job.rows_written) : "—"}
                    </Table.Td>
                  </Table.Tr>
                ))}
              </Table.Tbody>
            </Table>
          </Table.ScrollContainer>
        )}
      </Stack>

      {/* ── C: accounts ── */}
      <Stack gap="sm">
        <Title order={4}>Accounts</Title>
        {accounts.isLoading && (
          <Group justify="center" py="md">
            <Loader size="sm" />
          </Group>
        )}
        {!accounts.isLoading && accounts.data?.length === 0 && (
          <Text size="sm" c="dimmed">
            No accounts registered.
          </Text>
        )}
        {accounts.data && accounts.data.length > 0 && (
          <SimpleGrid cols={{ base: 1, sm: 2, lg: 3 }} spacing="sm">
            {accounts.data.map((a) => (
              <AccountCard key={a.account_id} account={a} />
            ))}
          </SimpleGrid>
        )}
      </Stack>
    </Stack>
  );
}

function AccountCard({ account: a }: { account: DataAccount }) {
  const masked =
    String(a.login).length > 4 ? `****${String(a.login).slice(-4)}` : a.login;
  return (
    <Card withBorder radius="md" padding="md">
      <Group justify="space-between" wrap="nowrap" mb={6}>
        <Text fw={600} size="sm">
          {a.broker}
        </Text>
        <Badge
          color={STATUS_COLOR[a.status] ?? "gray"}
          variant={a.status === "active" ? "filled" : "light"}
          size="sm"
          radius="sm"
        >
          {a.status}
        </Badge>
      </Group>
      <Group gap="xs" wrap="wrap">
        <Text size="xs" c="dimmed" ff="monospace">
          login {masked}
        </Text>
        <Badge color={TIER_COLOR[a.exec_tier] ?? "gray"} variant="light" size="xs" radius="sm">
          {a.exec_tier}
        </Badge>
        <Badge color="indigo" variant="light" size="xs" radius="sm">
          {a.program_phase}
        </Badge>
        {a.initial_balance != null && (
          <Text size="xs" c="dimmed">
            {a.currency} {fmtInt(a.initial_balance)}
          </Text>
        )}
      </Group>
    </Card>
  );
}
