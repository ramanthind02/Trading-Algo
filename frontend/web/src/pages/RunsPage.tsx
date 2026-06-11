import {
  ActionIcon,
  Badge,
  Card,
  Group,
  Loader,
  Stack,
  Table,
  Text,
  Tooltip,
} from "@mantine/core";
import { IconList, IconTrash } from "@tabler/icons-react";
import { useNavigate } from "react-router-dom";

import { useDeleteRun, useRuns } from "../api/hooks";
import type { RunStatus } from "../api/types";
import { RunStatusBadge } from "../components/RunStatusBadge";
import { PageHeader } from "../components/ui/PageHeader";

/** Returns a human-readable relative time string, e.g. "3m ago", "2h ago". */
function relativeTime(iso: string): string {
  const diffMs = Date.now() - new Date(iso).getTime();
  const diffSec = Math.floor(diffMs / 1000);
  if (diffSec < 60) return `${diffSec}s ago`;
  const diffMin = Math.floor(diffSec / 60);
  if (diffMin < 60) return `${diffMin}m ago`;
  const diffHr = Math.floor(diffMin / 60);
  if (diffHr < 24) return `${diffHr}h ago`;
  const diffDay = Math.floor(diffHr / 24);
  return `${diffDay}d ago`;
}

export function RunsPage() {
  const runs = useRuns();
  const deleteRun = useDeleteRun();
  const navigate = useNavigate();

  return (
    <Stack gap="lg">
      <PageHeader
        title="Runs & Results"
        subtitle="Exploration runs launched from a spec. Click a row to watch its log and review results."
        actions={
          runs.isFetching ? (
            <Group gap={6} align="center">
              <Loader size={14} color="dimmed" />
              <Text size="xs" c="dimmed">
                refreshing…
              </Text>
            </Group>
          ) : undefined
        }
      />

      {runs.isLoading && (
        <Group justify="center" py="xl">
          <Loader />
        </Group>
      )}

      {runs.data && runs.data.length === 0 && (
        <Card withBorder padding="xl">
          <Stack align="center" gap="xs" py="md">
            <IconList size={36} style={{ opacity: 0.3 }} />
            <Text fw={500} c="dimmed">
              No runs yet
            </Text>
            <Text size="sm" c="dimmed">
              Launch one from a spec in the Library or Builder.
            </Text>
          </Stack>
        </Card>
      )}

      {runs.data && runs.data.length > 0 && (
        <Card withBorder padding={0}>
          <Table highlightOnHover>
            <Table.Thead>
              <Table.Tr>
                <Table.Th>Spec</Table.Th>
                <Table.Th>Phase</Table.Th>
                <Table.Th>Status</Table.Th>
                <Table.Th>Combos</Table.Th>
                <Table.Th>Started</Table.Th>
                <Table.Th w={56} />
              </Table.Tr>
            </Table.Thead>
            <Table.Tbody>
              {runs.data.map((run) => (
                <Table.Tr
                  key={run.run_id}
                  style={{ cursor: "pointer" }}
                  onClick={() => navigate(`/runs/${run.run_id}`)}
                >
                  <Table.Td>
                    <Text size="sm" fw={500}>
                      {run.spec_name}
                    </Text>
                  </Table.Td>
                  <Table.Td>
                    <Badge variant="outline" size="sm" color="indigo">
                      {run.phase}
                    </Badge>
                  </Table.Td>
                  <Table.Td>
                    <RunStatusBadge status={run.status as RunStatus} />
                  </Table.Td>
                  <Table.Td>
                    <Text size="sm" c={run.num_combos ? undefined : "dimmed"}>
                      {run.num_combos ?? "—"}
                    </Text>
                  </Table.Td>
                  <Table.Td>
                    {run.started_at ? (
                      <Tooltip
                        label={new Date(run.started_at).toLocaleString()}
                        withArrow
                        position="left"
                      >
                        <Text size="xs" c="dimmed" style={{ cursor: "default" }}>
                          {relativeTime(run.started_at)}
                        </Text>
                      </Tooltip>
                    ) : (
                      <Text size="xs" c="dimmed">
                        —
                      </Text>
                    )}
                  </Table.Td>
                  <Table.Td>
                    <Tooltip label="Delete run" withArrow position="left">
                      <ActionIcon
                        variant="subtle"
                        color="red"
                        size="sm"
                        disabled={run.status === "queued" || run.status === "running"}
                        loading={deleteRun.isPending && deleteRun.variables === run.run_id}
                        onClick={(e) => {
                          e.stopPropagation();
                          if (!window.confirm(`Delete run for "${run.spec_name}"?`)) return;
                          deleteRun.mutate(run.run_id);
                        }}
                      >
                        <IconTrash size={15} />
                      </ActionIcon>
                    </Tooltip>
                  </Table.Td>
                </Table.Tr>
              ))}
            </Table.Tbody>
          </Table>
        </Card>
      )}
    </Stack>
  );
}
