import {
  Alert,
  Badge,
  Box,
  Button,
  Card,
  Code,
  Divider,
  Grid,
  Group,
  Loader,
  NavLink,
  ScrollArea,
  Stack,
  Tabs,
  Text,
  ThemeIcon,
  Title,
} from "@mantine/core";
import {
  IconAlertTriangle,
  IconChartBar,
  IconChartHistogram,
  IconChevronLeft,
  IconClock,
  IconFileText,
  IconFiles,
  IconShieldCheck,
  IconTerminal2,
} from "@tabler/icons-react";
import { Suspense, lazy, useEffect, useRef, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";

import { notifications } from "@mantine/notifications";
import { IconPlayerPlay } from "@tabler/icons-react";

import { useRun, useRunArtifacts, useStartRun } from "../api/hooks";
import type { ArtifactEntry } from "../api/types";
import { ArtifactPreview } from "../components/ArtifactPreview";
import { RunStatusBadge } from "../components/RunStatusBadge";
import { VaultPanel } from "../components/VaultPanel";

// Lazy so Plotly + chart code load only when the Results tab opens.
const ResultsTab = lazy(() => import("../components/ResultsTab"));
const ConditionalReturnsTab = lazy(() => import("../components/ConditionalReturnsTab"));

/** Format an ISO timestamp to a compact local string, e.g. "Jun 7, 14:32" */
function fmtTs(iso: string | null | undefined): string {
  if (!iso) return "—";
  const d = new Date(iso);
  return d.toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

export function RunResults() {
  const { id } = useParams();
  const navigate = useNavigate();
  const run = useRun(id ?? null);
  const startRun = useStartRun();
  const completed = run.data?.status === "completed";
  const terminal = completed || run.data?.status === "failed";

  if (run.isLoading)
    return (
      <Group justify="center" mt="xl">
        <Loader size="sm" />
        <Text c="dimmed" size="sm">
          Loading run…
        </Text>
      </Group>
    );

  if (run.isError || !run.data)
    return (
      <Alert color="red" icon={<IconAlertTriangle size={18} />} title="Run not found">
        Could not load this run. It may have been deleted or the ID is invalid.
      </Alert>
    );

  const r = run.data;

  const runValidation = () =>
    startRun.mutate(
      { spec_id: r.spec_id, phase: "validation" },
      {
        onSuccess: (newRun) => navigate(`/runs/${newRun.run_id}`),
        onError: (err) => notifications.show({ message: String(err), color: "red" }),
      },
    );

  return (
    <Stack gap="lg">
      {/* ── Header ─────────────────────────────────────────────────── */}
      <Card withBorder padding="md" radius="md">
        <Group justify="space-between" wrap="nowrap" align="flex-start">
          <Stack gap={6}>
            <Group gap="xs" align="center">
              <Button
                variant="subtle"
                size="xs"
                leftSection={<IconChevronLeft size={14} />}
                onClick={() => navigate("/runs")}
                px={6}
              >
                Runs
              </Button>
              <Divider orientation="vertical" />
              <Title order={3} style={{ lineHeight: 1.2 }}>
                {r.spec_name}
              </Title>
            </Group>

            <Group gap="xs" wrap="wrap">
              <RunStatusBadge status={r.status} />
              <Badge variant="outline" color="indigo" size="sm" tt="capitalize">
                {r.phase}
              </Badge>
              <Badge variant="dot" color="gray" size="sm">
                {r.num_combos} combos
              </Badge>

              {r.started_at && (
                <Group gap={4}>
                  <ThemeIcon size={14} variant="transparent" color="dimmed">
                    <IconClock size={12} />
                  </ThemeIcon>
                  <Text size="xs" c="dimmed">
                    Started {fmtTs(r.started_at)}
                  </Text>
                </Group>
              )}
              {r.finished_at && (
                <Text size="xs" c="dimmed">
                  · Finished {fmtTs(r.finished_at)}
                </Text>
              )}
            </Group>
          </Stack>

          {completed && r.phase === "exploration" && (
            <Button
              variant="filled"
              color="indigo"
              leftSection={<IconPlayerPlay size={15} />}
              loading={startRun.isPending}
              onClick={runValidation}
              size="sm"
              style={{ flexShrink: 0 }}
            >
              Run validation
            </Button>
          )}
        </Group>
      </Card>

      {/* ── Failed alert ────────────────────────────────────────────── */}
      {r.status === "failed" && (
        <Alert
          color="red"
          variant="light"
          icon={<IconAlertTriangle size={18} />}
          title="Run failed"
          styles={{ title: { fontWeight: 600 } }}
        >
          <Text size="sm" style={{ whiteSpace: "pre-wrap", fontFamily: "monospace" }}>
            {r.error_text ?? "No error details available."}
          </Text>
        </Alert>
      )}

      {/* ── Tabs ────────────────────────────────────────────────────── */}
      <Tabs defaultValue={completed ? "results" : "log"} keepMounted={false}>
        <Tabs.List>
          <Tabs.Tab value="results" leftSection={<IconChartBar size={15} />} disabled={!completed}>
            Results
          </Tabs.Tab>
          <Tabs.Tab
            value="conditional"
            leftSection={<IconChartHistogram size={15} />}
            disabled={!completed}
          >
            Conditional returns
          </Tabs.Tab>
          <Tabs.Tab value="log" leftSection={<IconTerminal2 size={15} />}>
            <Group gap={6} wrap="nowrap">
              Log
              {!terminal && <Loader size={10} />}
            </Group>
          </Tabs.Tab>
          <Tabs.Tab value="artifacts" leftSection={<IconFiles size={15} />} disabled={!completed}>
            Artifacts
          </Tabs.Tab>
          <Tabs.Tab value="promote" leftSection={<IconShieldCheck size={15} />} disabled={!completed}>
            Promote
          </Tabs.Tab>
        </Tabs.List>

        <Tabs.Panel value="results" pt="lg">
          {completed && (
            <Suspense
              fallback={
                <Group justify="center" mt="xl">
                  <Loader size="sm" />
                  <Text c="dimmed" size="sm">
                    Loading results…
                  </Text>
                </Group>
              }
            >
              <ResultsTab runId={r.run_id} />
            </Suspense>
          )}
        </Tabs.Panel>

        <Tabs.Panel value="conditional" pt="lg">
          {completed && (
            <Suspense
              fallback={
                <Group justify="center" mt="xl">
                  <Loader size="sm" />
                  <Text c="dimmed" size="sm">
                    Loading…
                  </Text>
                </Group>
              }
            >
              <ConditionalReturnsTab runId={r.run_id} />
            </Suspense>
          )}
        </Tabs.Panel>

        <Tabs.Panel value="log" pt="lg">
          <Card withBorder padding={0} radius="md">
            <Box px="md" py="xs" style={{ borderBottom: "1px solid var(--mantine-color-dark-4)" }}>
              <Group gap="xs">
                <ThemeIcon size={16} variant="transparent" color="dimmed">
                  <IconTerminal2 size={14} />
                </ThemeIcon>
                <Text size="xs" c="dimmed" ff="monospace">
                  run/{r.run_id.slice(0, 8)}…
                </Text>
                {!terminal && (
                  <Badge size="xs" color="blue" variant="dot">
                    live
                  </Badge>
                )}
              </Group>
            </Box>
            <Box p="md">
              <LogView text={r.log_text} live={!terminal} />
            </Box>
          </Card>
        </Tabs.Panel>

        <Tabs.Panel value="artifacts" pt="lg">
          {completed && <ArtifactsTab runId={r.run_id} />}
        </Tabs.Panel>

        <Tabs.Panel value="promote" pt="lg">
          <Card withBorder padding="lg" radius="md">
            {completed && <VaultPanel specId={r.spec_id} />}
          </Card>
        </Tabs.Panel>
      </Tabs>
    </Stack>
  );
}

function ArtifactsTab({ runId }: { runId: string }) {
  const artifacts = useRunArtifacts(runId, true);
  const [selected, setSelected] = useState<string | null>(null);

  return (
    <Grid gutter="md">
      <Grid.Col span={{ base: 12, sm: 4 }}>
        <Card withBorder padding="sm" radius="md" h="100%">
          <Text size="xs" c="dimmed" mb="xs" fw={500} tt="uppercase" style={{ letterSpacing: "0.05em" }}>
            Files
          </Text>
          {artifacts.isLoading && (
            <Group gap="xs">
              <Loader size="xs" />
              <Text size="xs" c="dimmed">
                Loading…
              </Text>
            </Group>
          )}
          <ScrollArea h={500}>
            {artifacts.data?.map((group) => (
              <div key={group.label}>
                <Text size="xs" c="dimmed" mt="sm" mb={4} fw={500}>
                  {group.label}
                </Text>
                {group.files.length === 0 && (
                  <Text size="xs" c="dimmed" pl="xs">
                    (none)
                  </Text>
                )}
                {group.files.map((file) => (
                  <ArtifactLink
                    key={file.path}
                    file={file}
                    active={selected === file.path}
                    onClick={() => setSelected(file.path)}
                  />
                ))}
              </div>
            ))}
          </ScrollArea>
        </Card>
      </Grid.Col>
      <Grid.Col span={{ base: 12, sm: 8 }}>
        <Card withBorder padding="md" radius="md" mih={540}>
          {selected ? (
            <>
              <Text size="xs" c="dimmed" mb="xs" ff="monospace">
                {selected}
              </Text>
              <Divider mb="md" />
              <ArtifactPreview path={selected} />
            </>
          ) : (
            <Group justify="center" align="center" h={480}>
              <Stack align="center" gap="xs">
                <ThemeIcon size={40} variant="light" color="gray" radius="xl">
                  <IconFiles size={20} />
                </ThemeIcon>
                <Text c="dimmed" size="sm">
                  Select a file to preview
                </Text>
              </Stack>
            </Group>
          )}
        </Card>
      </Grid.Col>
    </Grid>
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
    <ScrollArea h={480} viewportRef={viewport}>
      <Code
        block
        style={{
          whiteSpace: "pre-wrap",
          fontSize: 11,
          lineHeight: 1.6,
          background: "transparent",
          border: "none",
          padding: 0,
        }}
      >
        {text || (live ? "Waiting for output…" : "(no output)")}
      </Code>
    </ScrollArea>
  );
}
