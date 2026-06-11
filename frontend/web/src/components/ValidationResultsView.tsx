import {
  Badge,
  Box,
  Card,
  Divider,
  Grid,
  Group,
  Image,
  SimpleGrid,
  Stack,
  Text,
  ThemeIcon,
  Title,
} from "@mantine/core";
import {
  IconChartBar,
  IconChartLine,
  IconCheck,
  IconShieldCheck,
  IconX,
} from "@tabler/icons-react";

import type { LaneResults, ValidationImage, ValidationSummary } from "../api/types";
import { fmtNum } from "../lib/format";
import { HeadlinePanel } from "./HeadlinePanel";

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

function MetricRow({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <Group justify="space-between" gap="xs">
      <Text size="sm" c="dimmed">
        {label}
      </Text>
      <Text size="sm" fw={500}>
        {value}
      </Text>
    </Group>
  );
}

function RobustnessCard({ summary }: { summary: ValidationSummary }) {
  const passed = summary.robustness_passed;
  const color = passed === true ? "teal" : passed === false ? "red" : "gray";

  return (
    <Card withBorder padding="lg" radius="md" h="100%">
      <SectionHeader icon={<IconChartLine size={15} />} title="Walkforward Robustness" />
      <Divider mt="sm" mb="md" />
      <Stack gap="xs">
        <Group gap="xs">
          <Badge
            color={color}
            variant="light"
            leftSection={passed ? <IconCheck size={11} /> : <IconX size={11} />}
          >
            {passed === true ? "PASSED" : passed === false ? "FAILED" : "—"}
          </Badge>
          {summary.cusum_break && (
            <Badge color="orange" variant="light" size="sm">
              CUSUM break
            </Badge>
          )}
        </Group>
        <MetricRow
          label="IS Sharpe"
          value={summary.sr_is != null ? fmtNum(summary.sr_is) : "—"}
        />
        <MetricRow
          label="Val Sharpe"
          value={summary.sr_val != null ? fmtNum(summary.sr_val) : "—"}
        />
        <MetricRow
          label="Degradation ratio"
          value={summary.degradation_ratio != null ? fmtNum(summary.degradation_ratio) : "—"}
        />
        <MetricRow
          label="Degradation z"
          value={summary.degradation_z != null ? fmtNum(summary.degradation_z) : "—"}
        />
        <MetricRow
          label="CI overlap"
          value={
            summary.ci_overlap === true
              ? "Yes"
              : summary.ci_overlap === false
              ? "No"
              : "—"
          }
        />
        {summary.robustness_interpretation && (
          <Text size="xs" c="dimmed" mt="xs" lh={1.5}>
            {summary.robustness_interpretation}
          </Text>
        )}
      </Stack>
    </Card>
  );
}

function GateCard({ summary }: { summary: ValidationSummary }) {
  const passed = summary.gate_passed;
  const color = passed === true ? "teal" : passed === false ? "red" : "gray";

  return (
    <Card withBorder padding="lg" radius="md" h="100%">
      <SectionHeader icon={<IconShieldCheck size={15} />} title="Portfolio Addition Gate" />
      <Divider mt="sm" mb="md" />
      <Stack gap="xs">
        <Badge
          color={color}
          variant="light"
          size="md"
          leftSection={passed ? <IconCheck size={11} /> : <IconX size={11} />}
        >
          {passed === true ? "PASSED" : passed === false ? "FAILED" : "—"}
        </Badge>
        {summary.gate_n_existing != null && (
          <MetricRow label="Existing strategies" value={String(summary.gate_n_existing)} />
        )}
        {summary.gate_weight_method && (
          <MetricRow label="Weight method" value={summary.gate_weight_method} />
        )}
        {summary.gate_mean_peer_corr != null && (
          <MetricRow
            label="Mean peer ρ"
            value={fmtNum(summary.gate_mean_peer_corr, 3)}
          />
        )}
        {summary.gate_interpretation && (
          <Text size="xs" c="dimmed" mt="xs" lh={1.5}>
            {summary.gate_interpretation}
          </Text>
        )}
      </Stack>
    </Card>
  );
}

function ImageCard({ img }: { img: ValidationImage }) {
  const url = `/api/artifacts/raw?path=${encodeURIComponent(img.path)}`;
  return (
    <Card withBorder padding="sm" radius="md">
      <Text size="xs" c="dimmed" fw={500} mb="xs" lineClamp={1}>
        {img.label}
      </Text>
      <Image
        src={url}
        alt={img.label}
        radius="sm"
        fit="contain"
        style={{ maxHeight: 240 }}
        fallbackSrc="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg'/%3E"
      />
    </Card>
  );
}

/** Images split into two groups: the first 4 (validation-specific) and the rest (gate charts). */
function ImageGrid({ images }: { images: ValidationImage[] }) {
  if (images.length === 0) return null;

  const validationImgs = images.slice(0, 4);
  const gateImgs = images.slice(4);

  return (
    <Stack gap="md">
      <Card withBorder padding="lg" radius="md">
        <SectionHeader icon={<IconChartBar size={15} />} title="Validation Charts" />
        <Divider mt="sm" mb="md" />
        <SimpleGrid cols={{ base: 1, sm: 2 }} spacing="md">
          {validationImgs.map((img) => (
            <ImageCard key={img.path} img={img} />
          ))}
        </SimpleGrid>
      </Card>

      {gateImgs.length > 0 && (
        <Card withBorder padding="lg" radius="md">
          <SectionHeader icon={<IconShieldCheck size={15} />} title="Portfolio Gate Charts" />
          <Divider mt="sm" mb="md" />
          <SimpleGrid cols={{ base: 1, sm: 2 }} spacing="md">
            {gateImgs.map((img) => (
              <ImageCard key={img.path} img={img} />
            ))}
          </SimpleGrid>
        </Card>
      )}
    </Stack>
  );
}

export function ValidationResultsView({ lane }: { lane: LaneResults }) {
  const { headline, validation_summary: summary, validation_images: images } = lane;

  return (
    <Stack gap="xl">
      {/* ── Headline ─────────────────────────────────────── */}
      {headline && (
        <Card withBorder padding="lg" radius="md">
          <SectionHeader icon={<IconChartLine size={15} />} title="Validation Summary" />
          <Box mt="md">
            <HeadlinePanel headline={headline} />
          </Box>
        </Card>
      )}

      {/* ── Robustness + Gate side-by-side ───────────────── */}
      {summary && Object.keys(summary).length > 0 && (
        <Grid gutter="lg">
          <Grid.Col span={{ base: 12, md: 6 }}>
            <RobustnessCard summary={summary} />
          </Grid.Col>
          <Grid.Col span={{ base: 12, md: 6 }}>
            <GateCard summary={summary} />
          </Grid.Col>
        </Grid>
      )}

      {/* ── Charts ───────────────────────────────────────── */}
      {images && images.length > 0 && <ImageGrid images={images} />}

      {/* Empty state when no validation data at all */}
      {!headline && (!images || images.length === 0) && (
        <Card withBorder padding="lg" radius="md">
          <Text c="dimmed" size="sm">
            No validation results available yet.
          </Text>
        </Card>
      )}
    </Stack>
  );
}
