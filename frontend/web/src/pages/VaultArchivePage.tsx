import {
  Badge,
  Box,
  Button,
  Card,
  Divider,
  Group,
  Loader,
  SegmentedControl,
  SimpleGrid,
  Stack,
  Text,
  Tooltip,
} from "@mantine/core";
import { notifications } from "@mantine/notifications";
import { IconArchive, IconFlask2, IconInbox } from "@tabler/icons-react";
import { useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";

import { useVaultFeatureToSpec, useVaultFeatures } from "../api/hooks";
import type { VaultFeature } from "../api/types";
import { PageHeader } from "../components/ui/PageHeader";

/** Direction → color mapping for badge accent. */
const DIRECTION_COLOR: Record<string, string> = {
  long: "teal",
  short: "red",
  both: "indigo",
};

export function VaultArchivePage() {
  const navigate = useNavigate();
  const [profile, setProfile] = useState("prop");
  const features = useVaultFeatures(profile);
  const toSpec = useVaultFeatureToSpec();

  const bySleeve = useMemo(() => {
    const map = new Map<string, VaultFeature[]>();
    for (const f of features.data ?? []) {
      const key = `${f.timeframe ?? "?"} · ${f.sleeve ?? "?"}`;
      const list = map.get(key) ?? [];
      list.push(f);
      map.set(key, list);
    }
    return [...map.entries()].sort(([a], [b]) => a.localeCompare(b));
  }, [features.data]);

  const reResearch = (f: VaultFeature) => {
    toSpec.mutate(
      { profile: f.profile, path: f.path },
      {
        onSuccess: (detail) => {
          notifications.show({
            message: `Created spec ${detail.id} from ${f.ensemble_name}`,
            color: "teal",
          });
          navigate(`/builder/${detail.id}`);
        },
        onError: (err) =>
          notifications.show({ message: String(err), color: "red" }),
      },
    );
  };

  const profileToggle = (
    <SegmentedControl
      value={profile}
      onChange={setProfile}
      data={[
        { value: "prop", label: "Prop vault" },
        { value: "personal", label: "Personal vault" },
      ]}
    />
  );

  return (
    <Stack gap="xl">
      <PageHeader
        title="Vault features archive"
        subtitle="Every feature promoted to the vault. Re-research one to regenerate its results in the workbench — past research artifacts aren't stored in the vault itself."
        actions={profileToggle}
      />

      {features.isLoading && (
        <Group justify="center" py="xl">
          <Loader size="sm" />
        </Group>
      )}

      {!features.isLoading && features.data && features.data.length === 0 && (
        <Card withBorder padding="xl" radius="md">
          <Stack align="center" gap="sm" py="md">
            <IconInbox size={36} stroke={1.4} color="var(--mantine-color-dimmed)" />
            <Text c="dimmed" size="sm" ta="center">
              No features found in the{" "}
              <Text span fw={600} c="dimmed">
                {profile}
              </Text>{" "}
              vault.
            </Text>
          </Stack>
        </Card>
      )}

      {bySleeve.map(([group, items]) => (
        <Stack key={group} gap="sm">
          {/* Section header */}
          <Group gap="sm" align="center">
            <IconArchive size={15} stroke={1.6} color="var(--mantine-color-indigo-4)" />
            <Text fw={700} size="sm" c="bright" style={{ letterSpacing: "0.01em" }}>
              {group}
            </Text>
            <Badge
              size="sm"
              variant="light"
              color="indigo"
              radius="sm"
            >
              {items.length}
            </Badge>
            <Box style={{ flex: 1 }}>
              <Divider />
            </Box>
          </Group>

          {/* Feature cards */}
          <SimpleGrid cols={{ base: 1, sm: 2, lg: 3 }} spacing="md">
            {items.map((f) => (
              <FeatureCard
                key={f.path}
                feature={f}
                isPending={toSpec.isPending}
                onReResearch={reResearch}
              />
            ))}
          </SimpleGrid>
        </Stack>
      ))}
    </Stack>
  );
}

/** Individual vault feature card — name, direction/ticker badges, meta, re-research CTA. */
function FeatureCard({
  feature: f,
  isPending,
  onReResearch,
}: {
  feature: VaultFeature;
  isPending: boolean;
  onReResearch: (f: VaultFeature) => void;
}) {
  const dirColor = DIRECTION_COLOR[f.direction?.toLowerCase() ?? ""] ?? "gray";

  return (
    <Card
      withBorder
      padding="md"
      radius="md"
      style={{
        display: "flex",
        flexDirection: "column",
        gap: 0,
        transition: "border-color 120ms ease, box-shadow 120ms ease",
      }}
      styles={{
        root: {
          "&:hover": {
            borderColor: "var(--mantine-color-indigo-6)",
            boxShadow: "0 2px 12px rgba(0,0,0,0.18)",
          },
        },
      }}
    >
      {/* Name */}
      <Text fw={700} size="sm" truncate mb={8} style={{ lineHeight: 1.3 }}>
        {f.ensemble_name}
      </Text>

      {/* Direction + ticker badges */}
      <Group gap={6} wrap="wrap" mb={10}>
        {f.direction && (
          <Badge variant="filled" color={dirColor} size="sm" radius="sm">
            {f.direction}
          </Badge>
        )}
        {f.tickers.map((t) => (
          <Badge key={t} variant="light" color="gray" size="sm" radius="sm">
            {t}
          </Badge>
        ))}
      </Group>

      {/* Meta line */}
      <Text size="xs" c="dimmed" mb="md">
        {f.feature_count} feature{f.feature_count !== 1 ? "s" : ""}
        {f.updated_at
          ? ` · ${new Date(f.updated_at).toLocaleDateString(undefined, {
              year: "numeric",
              month: "short",
              day: "numeric",
            })}`
          : ""}
      </Text>

      {/* Re-research CTA */}
      <Tooltip
        label="Converts this vault entry into a runnable StrategySpec in the workbench"
        withArrow
        position="bottom"
        multiline
        w={230}
      >
        <Button
          size="xs"
          variant="light"
          color="indigo"
          leftSection={<IconFlask2 size={13} />}
          loading={isPending}
          onClick={() => onReResearch(f)}
          style={{ alignSelf: "flex-start", marginTop: "auto" }}
        >
          Re-research
        </Button>
      </Tooltip>
    </Card>
  );
}
