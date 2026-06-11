import {
  Alert,
  Badge,
  Box,
  Button,
  Card,
  Code,
  Divider,
  Group,
  List,
  Loader,
  Stack,
  Text,
  ThemeIcon,
} from "@mantine/core";
import { notifications } from "@mantine/notifications";
import {
  IconAlertTriangle,
  IconCircleCheck,
  IconCircleX,
  IconClipboardCheck,
  IconInfoCircle,
  IconPlayerPlay,
  IconShieldCheck,
  IconShieldLock,
  IconShieldX,
} from "@tabler/icons-react";
import { useNavigate } from "react-router-dom";

import { useStartRun, useVaultCommit, useVaultPreview } from "../api/hooks";

// ── gate status helpers ────────────────────────────────────────────────────────

const GATE_COLOR: Record<string, string> = {
  passed: "teal",
  failed: "red",
  missing: "yellow",
  skipped: "gray",
};

const GATE_LABEL: Record<string, string> = {
  passed: "Gate passed",
  failed: "Gate failed",
  missing: "Gate missing",
  skipped: "Gate skipped",
};

function GateIcon({ status, size = 20 }: { status: string; size?: number }) {
  if (status === "passed") return <IconShieldCheck size={size} />;
  if (status === "failed") return <IconShieldX size={size} />;
  return <IconShieldLock size={size} />;
}

// ── component ──────────────────────────────────────────────────────────────────

export function VaultPanel({ specId }: { specId: string }) {
  const preview = useVaultPreview(specId);
  const commit = useVaultCommit();
  const startRun = useStartRun();
  const navigate = useNavigate();

  const runValidation = () =>
    startRun.mutate(
      { spec_id: specId, phase: "validation" },
      {
        onSuccess: (run) => {
          notifications.show({ message: "Validation started — produces the portfolio-addition gate", color: "blue" });
          navigate(`/runs/${run.run_id}`);
        },
        onError: (err) => notifications.show({ message: String(err), color: "red" }),
      },
    );

  if (preview.isLoading)
    return (
      <Group gap="sm" p="md">
        <Loader size="sm" color="teal" />
        <Text size="sm" c="dimmed">Loading vault preview…</Text>
      </Group>
    );

  if (preview.isError || !preview.data)
    return (
      <Alert icon={<IconAlertTriangle size={16} />} color="red" title="Preview unavailable">
        Failed to load vault preview. Refresh or check server logs.
      </Alert>
    );

  const { eligibility, preview: dryRun, preview_error } = preview.data;
  const gateColor = GATE_COLOR[eligibility.gate_status] ?? "gray";
  const gateLabel = GATE_LABEL[eligibility.gate_status] ?? eligibility.gate_status;

  const doCommit = () => {
    if (!window.confirm(`Write ${specId} to the vault at ${dryRun?.ensemble_dir_repo_relative}?`)) return;
    commit.mutate(specId, {
      onSuccess: (res) => {
        notifications.show({ message: `Promoted to ${res.ensemble_dir_repo_relative}`, color: "teal" });
        preview.refetch();
      },
      onError: (err) => notifications.show({ message: String(err), color: "red" }),
    });
  };

  return (
    <Stack gap="md">
      {/* ── Header ── */}
      <Group gap="sm" align="center">
        <ThemeIcon color={gateColor} variant="light" size="lg" radius="md">
          <GateIcon status={eligibility.gate_status} size={18} />
        </ThemeIcon>
        <Box>
          <Text fw={700} size="sm" lh={1.3}>
            Promote to vault
          </Text>
          <Text size="xs" c="dimmed" lh={1.3}>
            {specId}
          </Text>
        </Box>
        <Badge
          color={gateColor}
          variant="filled"
          size="md"
          radius="sm"
          ml="auto"
          leftSection={
            eligibility.gate_status === "passed"
              ? <IconCircleCheck size={12} />
              : <IconCircleX size={12} />
          }
        >
          {gateLabel}
        </Badge>
      </Group>

      <Divider />

      {/* ── Blockers ── */}
      {eligibility.blockers.length > 0 && (
        <Alert
          color="yellow"
          variant="light"
          icon={<IconAlertTriangle size={16} />}
          title="Commit blocked"
          styles={{ title: { fontWeight: 700 } }}
        >
          <List size="sm" spacing={4} mt={4}>
            {eligibility.blockers.map((b) => (
              <List.Item key={b} icon={<IconCircleX size={14} color="var(--mantine-color-yellow-6)" />}>
                {b}
              </List.Item>
            ))}
          </List>
          <Group gap="xs" align="center" mt="sm">
            <IconInfoCircle size={13} color="var(--mantine-color-dimmed)" />
            <Text size="xs" c="dimmed">
              The portfolio-addition gate is produced by a validation run against your portfolio settings.
            </Text>
          </Group>
          <Button
            mt="sm"
            size="xs"
            color="yellow"
            variant="light"
            leftSection={<IconPlayerPlay size={14} />}
            loading={startRun.isPending}
            onClick={runValidation}
          >
            Run validation
          </Button>
        </Alert>
      )}

      {/* ── Preview error ── */}
      {preview_error && (
        <Alert color="red" variant="light" icon={<IconAlertTriangle size={16} />} title="Preview error">
          {preview_error}
        </Alert>
      )}

      {/* ── Dry-run preview ── */}
      {dryRun && (
        <Card withBorder padding="md" radius="md">
          <Group gap="xs" mb="sm">
            <IconClipboardCheck size={15} color="var(--mantine-color-teal-5)" />
            <Text size="xs" fw={700} tt="uppercase" c="dimmed" style={{ letterSpacing: "0.05em" }}>
              What will be written
            </Text>
          </Group>
          <Stack gap={6}>
            <Field label="Vault root" value={dryRun.vault_root} />
            <Field label="Ensemble dir" value={dryRun.ensemble_dir_repo_relative} mono />
            <Field label="Feature column" value={dryRun.feature_column} mono />
            <Field label="Direction" value={dryRun.direction} />
            <Field label="Sleeve" value={dryRun.weight_hierarchy_group ?? "—"} />
            <Field label="Tickers" value={dryRun.tickers.join(", ")} />
            <Field label="Model id" value={dryRun.model_id ?? "—"} mono />
          </Stack>
        </Card>
      )}

      {/* ── Commit action ── */}
      <Card withBorder padding="md" radius="md" bg={eligibility.ready ? "teal.9" : undefined}
        style={{ borderColor: eligibility.ready ? "var(--mantine-color-teal-7)" : undefined }}>
        <Group justify="space-between" align="center">
          <Box>
            <Text size="sm" fw={600} c={eligibility.ready ? "teal.1" : undefined}>
              Commit to vault
            </Text>
            <Text size="xs" c={eligibility.ready ? "teal.3" : "dimmed"}>
              {eligibility.ready
                ? "Gate passed — ready to promote this ensemble."
                : "Commit unlocks when the portfolio-addition gate passes."}
            </Text>
          </Box>
          <Button
            color="teal"
            variant={eligibility.ready ? "filled" : "light"}
            leftSection={<IconShieldCheck size={16} />}
            disabled={!eligibility.ready}
            loading={commit.isPending}
            onClick={doCommit}
            size="sm"
          >
            Commit to vault
          </Button>
        </Group>
      </Card>
    </Stack>
  );
}

// ── Field row ─────────────────────────────────────────────────────────────────

function Field({ label, value, mono }: { label: string; value: string; mono?: boolean }) {
  return (
    <Group gap="xs" wrap="nowrap" align="flex-start">
      <Text size="xs" c="dimmed" w={120} style={{ flexShrink: 0, paddingTop: 1 }}>
        {label}
      </Text>
      {mono ? (
        <Code fz="xs" style={{ wordBreak: "break-all" }}>
          {value}
        </Code>
      ) : (
        <Text size="sm">{value}</Text>
      )}
    </Group>
  );
}
