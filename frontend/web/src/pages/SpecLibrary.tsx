import {
  ActionIcon,
  Badge,
  Box,
  Button,
  Card,
  Center,
  Group,
  Loader,
  Menu,
  SimpleGrid,
  Stack,
  Text,
  TextInput,
  ThemeIcon,
  Tooltip,
} from "@mantine/core";
import { notifications } from "@mantine/notifications";
import {
  IconAlertTriangle,
  IconCheck,
  IconChevronRight,
  IconDotsVertical,
  IconLayersLinked,
  IconPencil,
  IconPlayerPlay,
  IconPlus,
  IconSearch,
  IconTrash,
} from "@tabler/icons-react";
import { useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";

import { useDeleteSpec, useSpecs, useStartRun } from "../api/hooks";
import type { SpecSummary } from "../api/types";
import { PageHeader } from "../components/ui/PageHeader";
import { fmtInt } from "../lib/format";

// Color map for known direction values so the badge feels intentional.
const DIRECTION_COLOR: Record<string, string> = {
  long: "teal",
  short: "red",
  both: "blue",
};

// Color map for timeframe badges.
const TIMEFRAME_COLOR: Record<string, string> = {
  D: "violet",
  W: "indigo",
  M: "cyan",
};

export function SpecLibrary() {
  const navigate = useNavigate();
  const specs = useSpecs();
  const startRun = useStartRun();
  const deleteSpec = useDeleteSpec();
  const [query, setQuery] = useState("");

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q || !specs.data) return specs.data ?? [];
    return specs.data.filter((s) =>
      [s.name, s.module ?? "", s.tickers.join(" "), s.vault_sleeve ?? "", s.direction ?? ""]
        .join(" ")
        .toLowerCase()
        .includes(q),
    );
  }, [specs.data, query]);

  const launch = (spec: SpecSummary) => {
    startRun.mutate(
      { spec_id: spec.id },
      {
        onSuccess: (run) => {
          notifications.show({ message: `Exploration started for ${spec.name}`, color: "blue" });
          navigate(`/runs/${run.run_id}`);
        },
        onError: (err) => notifications.show({ message: String(err), color: "red" }),
      },
    );
  };

  const remove = (spec: SpecSummary) => {
    if (!window.confirm(`Delete spec "${spec.name}"? This removes research/specs/${spec.id}.json.`)) return;
    deleteSpec.mutate(spec.id, {
      onSuccess: () => notifications.show({ message: `Deleted ${spec.name}`, color: "gray" }),
    });
  };

  const total = specs.data?.length ?? 0;
  const matchCount = filtered.length;
  const showMatchCount = query.trim().length > 0 && total > 0;

  return (
    <Stack gap="lg">
      <PageHeader
        title="Spec Library"
        subtitle="Strategy specs in research/specs/. Build one here or have an agent write the JSON."
        actions={
          <Button leftSection={<IconPlus size={16} />} onClick={() => navigate("/builder/new")}>
            New strategy
          </Button>
        }
      />

      {specs.isLoading && (
        <Center py="xl">
          <Loader size="sm" />
        </Center>
      )}

      {specs.isError && (
        <Card withBorder radius="md" p="md">
          <Group gap="xs">
            <ThemeIcon color="red" variant="light" size="sm">
              <IconAlertTriangle size={14} />
            </ThemeIcon>
            <Text size="sm" c="red">
              Failed to load specs: {String(specs.error)}
            </Text>
          </Group>
        </Card>
      )}

      {specs.data && specs.data.length === 0 && (
        <Card withBorder radius="md" p="xl">
          <Stack align="center" gap="md" py="md">
            <ThemeIcon size={52} radius="xl" variant="light" color="gray">
              <IconLayersLinked size={26} />
            </ThemeIcon>
            <Stack gap={4} align="center">
              <Text fw={600} size="lg">
                No specs yet
              </Text>
              <Text c="dimmed" size="sm" ta="center" maw={320}>
                Create your first strategy spec, or drop a JSON into{" "}
                <Text component="span" ff="monospace" size="sm">
                  research/specs/
                </Text>
                .
              </Text>
            </Stack>
            <Button
              mt="xs"
              leftSection={<IconPlus size={16} />}
              rightSection={<IconChevronRight size={14} />}
              onClick={() => navigate("/builder/new")}
            >
              New strategy
            </Button>
          </Stack>
        </Card>
      )}

      {total > 0 && (
        <Group justify="space-between" align="center">
          <TextInput
            placeholder="Filter by name, module, ticker, sleeve…"
            leftSection={<IconSearch size={16} />}
            value={query}
            onChange={(e) => setQuery(e.currentTarget.value)}
            maw={400}
            style={{ flex: "0 0 auto" }}
          />
          {showMatchCount && (
            <Text size="sm" c="dimmed">
              {fmtInt(matchCount)} of {fmtInt(total)} spec{total !== 1 ? "s" : ""}
            </Text>
          )}
        </Group>
      )}

      {filtered.length === 0 && query.trim().length > 0 && (
        <Center py="xl">
          <Stack align="center" gap="xs">
            <ThemeIcon size={40} radius="xl" variant="light" color="gray">
              <IconSearch size={20} />
            </ThemeIcon>
            <Text c="dimmed" size="sm">
              No specs match <strong>&ldquo;{query}&rdquo;</strong>
            </Text>
          </Stack>
        </Center>
      )}

      <SimpleGrid cols={{ base: 1, sm: 2, lg: 3 }} spacing="md">
        {filtered.map((spec) => (
          <SpecCard
            key={spec.id}
            spec={spec}
            onEdit={() => navigate(`/builder/${spec.id}`)}
            onRun={() => launch(spec)}
            onDelete={() => remove(spec)}
            running={startRun.isPending}
          />
        ))}
      </SimpleGrid>
    </Stack>
  );
}

function SpecCard({
  spec,
  onEdit,
  onRun,
  onDelete,
  running,
}: {
  spec: SpecSummary;
  onEdit: () => void;
  onRun: () => void;
  onDelete: () => void;
  running: boolean;
}) {
  const isInvalid = !spec.valid;

  return (
    <Card
      withBorder
      padding="md"
      radius="md"
      style={{
        cursor: "default",
        transition: "box-shadow 120ms ease, transform 120ms ease",
        borderColor: isInvalid ? "var(--mantine-color-red-8)" : undefined,
      }}
      styles={{
        root: {
          "&:hover": {
            boxShadow: "var(--mantine-shadow-md)",
            transform: "translateY(-2px)",
          },
        },
      }}
    >
      {/* Header row: name + kebab menu */}
      <Group justify="space-between" wrap="nowrap" align="flex-start" mb={6}>
        <Stack gap={2} style={{ minWidth: 0, flex: 1 }}>
          <Text fw={700} size="md" truncate title={spec.name}>
            {spec.name}
          </Text>
          <Text size="xs" c="dimmed" lineClamp={2} style={{ lineHeight: 1.5 }}>
            {spec.hypothesis || (
              <Text component="span" fs="italic" c="dimmed" size="xs">
                No hypothesis
              </Text>
            )}
          </Text>
        </Stack>
        <Menu position="bottom-end" withinPortal>
          <Menu.Target>
            <ActionIcon variant="subtle" color="gray" size="sm" style={{ flexShrink: 0 }}>
              <IconDotsVertical size={15} />
            </ActionIcon>
          </Menu.Target>
          <Menu.Dropdown>
            <Menu.Item leftSection={<IconPencil size={14} />} onClick={onEdit}>
              Edit
            </Menu.Item>
            <Menu.Item color="red" leftSection={<IconTrash size={14} />} onClick={onDelete}>
              Delete
            </Menu.Item>
          </Menu.Dropdown>
        </Menu>
      </Group>

      {/* Badge row */}
      <Group gap={6} mt="xs" wrap="wrap">
        {spec.valid ? (
          <Badge
            color="teal"
            variant="light"
            size="sm"
            leftSection={<IconCheck size={11} />}
          >
            {fmtInt(spec.num_combos)} combo{spec.num_combos !== 1 ? "s" : ""}
          </Badge>
        ) : (
          <Tooltip label={spec.error ?? "Invalid spec"} multiline w={280} withArrow>
            <Badge
              color="red"
              variant="filled"
              size="sm"
              leftSection={<IconAlertTriangle size={11} />}
              style={{ cursor: "help" }}
            >
              invalid
            </Badge>
          </Tooltip>
        )}
        {spec.module && (
          <Badge variant="default" size="sm" color="gray">
            {spec.module}
          </Badge>
        )}
        {spec.timeframe && (
          <Badge variant="light" size="sm" color={TIMEFRAME_COLOR[spec.timeframe] ?? "gray"}>
            {spec.timeframe}
          </Badge>
        )}
        {spec.direction && (
          <Badge variant="light" size="sm" color={DIRECTION_COLOR[spec.direction.toLowerCase()] ?? "gray"}>
            {spec.direction}
          </Badge>
        )}
      </Group>

      {/* Meta line */}
      <Box mt="xs">
        <Text size="xs" c="dimmed" truncate>
          {spec.tickers.length > 0 ? spec.tickers.join(", ") : "no tickers"}
          {spec.vault_sleeve ? ` · ${spec.vault_sleeve}` : ""}
        </Text>
      </Box>

      {/* Action buttons */}
      <Group mt="md" gap="xs">
        <Button
          size="xs"
          leftSection={<IconPlayerPlay size={13} />}
          disabled={!spec.valid}
          loading={running}
          onClick={onRun}
        >
          Run exploration
        </Button>
        <Button size="xs" variant="default" leftSection={<IconPencil size={13} />} onClick={onEdit}>
          Edit
        </Button>
      </Group>
    </Card>
  );
}
