/**
 * WeightLayerControls — controlled edit panel for the portfolio weight-layer.
 *
 * Renders inside a Card; parent (PortfolioPage) owns the `value` state and
 * includes it in the run payload alongside the other ConfigForm fields.
 *
 * All controls are per-run overrides — the canonical config on disk is never
 * touched.
 */

import {
  ActionIcon,
  Button,
  Card,
  Collapse,
  Divider,
  Group,
  JsonInput,
  NumberInput,
  Select,
  Stack,
  Switch,
  Text,
  TextInput,
  Tooltip,
} from "@mantine/core";
import { useDisclosure } from "@mantine/hooks";
import { notifications } from "@mantine/notifications";
import {
  IconCheck,
  IconChevronDown,
  IconChevronRight,
  IconCirclePlus,
  IconFile,
  IconFilePlus,
  IconFolder,
  IconFolderPlus,
  IconInfoCircle,
  IconPencil,
  IconRotate2,
  IconSitemap,
  IconSlash,
  IconTrash,
  IconX,
} from "@tabler/icons-react";
import { useState } from "react";

import { useAddSleeve } from "../api/hooks";
import type { HierarchyNode, WeightLayerData } from "../api/types";

// ---------------------------------------------------------------------------
// Public interface — exported so PortfolioPage can include it in the run body
// ---------------------------------------------------------------------------
export interface WeightLayerOverrides {
  sr_adjustment?: boolean;
  sr_tilt_max_depth?: number;
  within_group_method?: string;
  sr_avg?: number;
  fdm_max?: number;
  hierarchy_spec?: HierarchyNode;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function SectionLabel({ label, tip }: { label: string; tip?: string }) {
  return (
    <Group gap={4} align="center">
      <Text size="xs" c="dimmed" fw={500} tt="uppercase">
        {label}
      </Text>
      {tip && (
        <Tooltip label={tip} withArrow position="top" maw={300} multiline>
          <IconInfoCircle
            size={13}
            color="var(--mantine-color-dimmed)"
            style={{ cursor: "help", flexShrink: 0 }}
          />
        </Tooltip>
      )}
    </Group>
  );
}

// ---------------------------------------------------------------------------
// Immutable tree helpers
// ---------------------------------------------------------------------------

function setNodeAtPath(
  root: HierarchyNode,
  path: number[],
  updater: (node: HierarchyNode) => HierarchyNode
): HierarchyNode {
  const [head, ...rest] = path;
  if (head === undefined) return updater(root);
  return {
    ...root,
    children: (root.children ?? []).map((child, i) =>
      i === head ? setNodeAtPath(child, rest, updater) : child
    ),
  };
}

function deleteNodeAtPath(root: HierarchyNode, path: number[]): HierarchyNode {
  const lastIdx = path.at(-1);
  if (lastIdx === undefined) return root;
  return setNodeAtPath(root, path.slice(0, -1), (parent) => ({
    ...parent,
    children: (parent.children ?? []).filter((_, i) => i !== lastIdx),
  }));
}

function addChildAtPath(
  root: HierarchyNode,
  path: number[],
  child: HierarchyNode
): HierarchyNode {
  return setNodeAtPath(root, path, (parent) => ({
    ...parent,
    children: [...(parent.children ?? []), child],
  }));
}

function getNodeAtPath(
  root: HierarchyNode,
  path: number[]
): HierarchyNode | null {
  let current: HierarchyNode = root;
  for (const idx of path) {
    const next = current.children?.[idx];
    if (!next) return null;
    current = next;
  }
  return current;
}

const pathKey = (path: number[]): string => path.join(",");

// ---------------------------------------------------------------------------
// Add-a-sleeve row
// ---------------------------------------------------------------------------

function AddSleeveRow() {
  const addSleeve = useAddSleeve();
  const [name, setName] = useState("");

  const handleAdd = () => {
    const trimmed = name.trim();
    if (!trimmed) return;
    addSleeve.mutate(trimmed, {
      onSuccess: () => {
        notifications.show({
          message: `Sleeve "${trimmed}" added — it will appear as a vault target in the Spec Builder.`,
          color: "teal",
        });
        setName("");
      },
      onError: (e) =>
        notifications.show({ message: String(e), color: "red" }),
    });
  };

  return (
    <Group gap="sm" align="flex-end">
      <TextInput
        label="New sleeve name"
        description="New sleeves become selectable as vault targets in the Spec Builder."
        placeholder="e.g. momentum_gc"
        value={name}
        onChange={(e) => setName(e.currentTarget.value)}
        onKeyDown={(e) => e.key === "Enter" && handleAdd()}
        style={{ flex: 1 }}
        size="sm"
      />
      <Button
        leftSection={<IconCirclePlus size={15} />}
        loading={addSleeve.isPending}
        disabled={!name.trim()}
        onClick={handleAdd}
        size="sm"
        variant="light"
        color="teal"
      >
        Add sleeve
      </Button>
    </Group>
  );
}

// ---------------------------------------------------------------------------
// Hierarchy visual tree editor
// ---------------------------------------------------------------------------

interface HierarchyEditorProps {
  current: HierarchyNode | null;
  overrideValue: HierarchyNode | undefined;
  onChange: (patch: WeightLayerOverrides) => void;
}

function HierarchyEditor({
  current,
  overrideValue,
  onChange,
}: HierarchyEditorProps) {
  const [opened, { toggle }] = useDisclosure(false);
  const [showJson, setShowJson] = useState(false);
  const [jsonError, setJsonError] = useState<string | null>(null);

  const treeValue: HierarchyNode | null = overrideValue ?? current;
  const [localText, setLocalText] = useState<string>(
    () => JSON.stringify(treeValue, null, 2)
  );

  const [expandedPaths, setExpandedPaths] = useState<Set<string>>(
    () => new Set([""])
  );

  const [editPath, setEditPath] = useState<string | null>(null);
  const [editValue, setEditValue] = useState("");

  const [addPath, setAddPath] = useState<string | null>(null);
  const [addKind, setAddKind] = useState<"group" | "leaf">("leaf");
  const [addValue, setAddValue] = useState("");

  const mutateTree = (newTree: HierarchyNode) => {
    setLocalText(JSON.stringify(newTree, null, 2));
    setJsonError(null);
    onChange({ hierarchy_spec: newTree });
  };

  const handleJsonChange = (raw: string) => {
    setLocalText(raw);
    try {
      const parsed = JSON.parse(raw) as HierarchyNode;
      setJsonError(null);
      onChange({ hierarchy_spec: parsed });
    } catch {
      setJsonError("Invalid JSON — fix before the override takes effect.");
    }
  };

  const handleReset = () => {
    setLocalText(JSON.stringify(current, null, 2));
    setJsonError(null);
    onChange({ hierarchy_spec: undefined });
    setEditPath(null);
    setAddPath(null);
  };

  const toggleExpand = (pKey: string) => {
    setExpandedPaths((prev) => {
      const next = new Set(prev);
      if (next.has(pKey)) next.delete(pKey);
      else next.add(pKey);
      return next;
    });
  };

  const startEdit = (path: number[], currentVal: string) => {
    setAddPath(null);
    setEditPath(pathKey(path));
    setEditValue(currentVal);
  };

  const confirmEdit = (path: number[], node: HierarchyNode) => {
    const trimmed = editValue.trim();
    if (!trimmed || !treeValue) return;
    const updated =
      node.type === "group"
        ? setNodeAtPath(treeValue, path, (n) => ({ ...n, id: trimmed }))
        : setNodeAtPath(treeValue, path, (n) => ({ ...n, stream_id: trimmed }));
    mutateTree(updated);
    setEditPath(null);
  };

  const startAdd = (path: number[], kind: "group" | "leaf") => {
    setEditPath(null);
    setAddPath(pathKey(path));
    setAddKind(kind);
    setAddValue("");
    setExpandedPaths((prev) => new Set([...prev, pathKey(path)]));
  };

  const confirmAdd = (path: number[]) => {
    const trimmed = addValue.trim();
    if (!trimmed || !treeValue) return;
    const child: HierarchyNode =
      addKind === "group"
        ? { type: "group", id: trimmed, children: [] }
        : { type: "leaf", stream_id: trimmed };
    mutateTree(addChildAtPath(treeValue, path, child));
    setAddPath(null);
    setAddValue("");
  };

  const canDeleteNode = (path: number[]): boolean => {
    if (path.length === 0 || !treeValue) return false;
    const parent = getNodeAtPath(treeValue, path.slice(0, -1));
    return (parent?.children ?? []).length > 1;
  };

  const deleteTreeNode = (path: number[]) => {
    if (!treeValue || path.length === 0) return;
    const pKey = pathKey(path);
    if (editPath !== null && (editPath === pKey || editPath.startsWith(pKey + ","))) {
      setEditPath(null);
    }
    if (addPath !== null && (addPath === pKey || addPath.startsWith(pKey + ","))) {
      setAddPath(null);
    }
    mutateTree(deleteNodeAtPath(treeValue, path));
  };

  const renderNode = (node: HierarchyNode, path: number[]): JSX.Element => {
    const pKey = pathKey(path);
    const isGroup = node.type === "group";
    const isExpanded = isGroup && expandedPaths.has(pKey);
    const isEditing = editPath === pKey;
    const isAddingHere = isGroup && addPath === pKey;
    const indentPx = path.length * 16;
    const canDel = canDeleteNode(path);

    const streamId = node.stream_id ?? "";
    const streamDisplay =
      streamId.length > 60 ? streamId.slice(0, 57) + "…" : streamId;

    return (
      <Stack key={pKey} gap={0}>
        <Group
          gap={4}
          align="center"
          style={{ paddingLeft: indentPx, minHeight: 30, flexWrap: "nowrap" }}
        >
          {isGroup ? (
            <ActionIcon
              variant="subtle"
              color="gray"
              size="xs"
              onClick={() => toggleExpand(pKey)}
              aria-label={isExpanded ? "Collapse group" : "Expand group"}
              style={{ flexShrink: 0 }}
            >
              {isExpanded ? (
                <IconChevronDown size={11} />
              ) : (
                <IconChevronRight size={11} />
              )}
            </ActionIcon>
          ) : (
            <div style={{ width: 22, flexShrink: 0 }} />
          )}

          {isGroup ? (
            <IconFolder
              size={13}
              color="var(--mantine-color-yellow-5)"
              style={{ flexShrink: 0 }}
            />
          ) : (
            <IconFile
              size={13}
              color="var(--mantine-color-blue-4)"
              style={{ flexShrink: 0 }}
            />
          )}

          {isEditing ? (
            <Group gap={4} style={{ flex: 1, minWidth: 0 }}>
              <TextInput
                value={editValue}
                onChange={(e) => setEditValue(e.currentTarget.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter") confirmEdit(path, node);
                  if (e.key === "Escape") setEditPath(null);
                }}
                size="xs"
                style={{ flex: 1 }}
                autoFocus
              />
              <ActionIcon
                variant="light"
                color="teal"
                size="xs"
                onClick={() => confirmEdit(path, node)}
                disabled={!editValue.trim()}
                aria-label="Confirm rename"
              >
                <IconCheck size={11} />
              </ActionIcon>
              <ActionIcon
                variant="subtle"
                color="gray"
                size="xs"
                onClick={() => setEditPath(null)}
                aria-label="Cancel rename"
              >
                <IconX size={11} />
              </ActionIcon>
            </Group>
          ) : (
            <>
              {isGroup ? (
                <Text
                  size="xs"
                  fw={500}
                  style={{
                    flex: 1,
                    minWidth: 0,
                    overflow: "hidden",
                    whiteSpace: "nowrap",
                    textOverflow: "ellipsis",
                  }}
                >
                  {node.id ?? "(unnamed group)"}
                </Text>
              ) : (
                <Tooltip
                  label={streamId || "(no stream_id)"}
                  withArrow
                  position="top"
                  disabled={streamId.length <= 60}
                  maw={500}
                  multiline
                >
                  <Text
                    size="xs"
                    c="dimmed"
                    ff="monospace"
                    style={{
                      flex: 1,
                      minWidth: 0,
                      overflow: "hidden",
                      whiteSpace: "nowrap",
                      textOverflow: "ellipsis",
                    }}
                  >
                    {streamDisplay || "(no stream_id)"}
                  </Text>
                </Tooltip>
              )}

              <Group gap={2} style={{ flexShrink: 0 }}>
                <Tooltip
                  label={isGroup ? "Rename group" : "Edit stream ID"}
                  withArrow
                  position="top"
                >
                  <ActionIcon
                    variant="subtle"
                    color="gray"
                    size="xs"
                    onClick={() =>
                      startEdit(path, isGroup ? (node.id ?? "") : streamId)
                    }
                    aria-label={isGroup ? "Rename group" : "Edit stream ID"}
                  >
                    <IconPencil size={11} />
                  </ActionIcon>
                </Tooltip>

                {isGroup && (
                  <Tooltip label="Add child group" withArrow position="top">
                    <ActionIcon
                      variant="subtle"
                      color="indigo"
                      size="xs"
                      onClick={() => startAdd(path, "group")}
                      aria-label="Add child group"
                    >
                      <IconFolderPlus size={11} />
                    </ActionIcon>
                  </Tooltip>
                )}

                {isGroup && (
                  <Tooltip label="Add leaf" withArrow position="top">
                    <ActionIcon
                      variant="subtle"
                      color="teal"
                      size="xs"
                      onClick={() => startAdd(path, "leaf")}
                      aria-label="Add leaf"
                    >
                      <IconFilePlus size={11} />
                    </ActionIcon>
                  </Tooltip>
                )}

                {path.length > 0 && (
                  <Tooltip
                    label={
                      canDel
                        ? "Delete node"
                        : "Parent group must keep ≥ 1 child"
                    }
                    withArrow
                    position="top"
                  >
                    <span style={{ display: "inline-flex" }}>
                      <ActionIcon
                        variant="subtle"
                        color="red"
                        size="xs"
                        onClick={() => deleteTreeNode(path)}
                        disabled={!canDel}
                        aria-label="Delete node"
                      >
                        <IconTrash size={11} />
                      </ActionIcon>
                    </span>
                  </Tooltip>
                )}
              </Group>
            </>
          )}
        </Group>

        {isGroup && isExpanded && (
          <>
            {(node.children ?? []).map((child, i) =>
              renderNode(child, [...path, i])
            )}

            {isAddingHere && (
              <Group
                gap={4}
                align="center"
                style={{
                  paddingLeft: indentPx + 38,
                  marginTop: 4,
                  marginBottom: 4,
                }}
              >
                <TextInput
                  placeholder={
                    addKind === "group"
                      ? "New group ID"
                      : "stream_id (e.g. ES::D::rsi_signal_D_lookback_14::…)"
                  }
                  value={addValue}
                  onChange={(e) => setAddValue(e.currentTarget.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter") confirmAdd(path);
                    if (e.key === "Escape") setAddPath(null);
                  }}
                  size="xs"
                  style={{ flex: 1 }}
                  autoFocus
                />
                <ActionIcon
                  variant="light"
                  color="teal"
                  size="xs"
                  onClick={() => confirmAdd(path)}
                  disabled={!addValue.trim()}
                  aria-label="Confirm add"
                >
                  <IconCheck size={11} />
                </ActionIcon>
                <ActionIcon
                  variant="subtle"
                  color="gray"
                  size="xs"
                  onClick={() => setAddPath(null)}
                  aria-label="Cancel add"
                >
                  <IconX size={11} />
                </ActionIcon>
              </Group>
            )}
          </>
        )}
      </Stack>
    );
  };

  return (
    <Stack gap="xs">
      <Group
        gap="xs"
        style={{ cursor: "pointer", userSelect: "none" }}
        onClick={toggle}
      >
        {opened ? (
          <IconChevronDown size={14} color="var(--mantine-color-dimmed)" />
        ) : (
          <IconChevronRight size={14} color="var(--mantine-color-dimmed)" />
        )}
        <IconSitemap size={14} color="var(--mantine-color-dimmed)" />
        <Text size="sm" fw={500}>
          Hierarchy structure
        </Text>
        {overrideValue !== undefined && (
          <Text size="xs" c="orange" fw={500}>
            (overridden)
          </Text>
        )}
      </Group>

      <Collapse in={opened}>
        <Stack gap="sm" pl="md">
          <Text size="xs" c="dimmed">
            The weight-layer group tree: groups (with children) are combined by
            the within-group method; leaves reference individual forecast
            streams. Stream IDs must match the vault exactly.
          </Text>

          <Card withBorder padding="xs" style={{ overflowX: "auto" }}>
            {treeValue ? (
              renderNode(treeValue, [])
            ) : (
              <Text size="xs" c="dimmed" fs="italic">
                No hierarchy configured.
              </Text>
            )}
          </Card>

          <Group gap="sm">
            <Button
              size="xs"
              variant="subtle"
              color="gray"
              onClick={() => setShowJson((v) => !v)}
            >
              {showJson ? "Hide JSON" : "View JSON"}
            </Button>
            <Button
              leftSection={<IconRotate2 size={13} />}
              size="xs"
              variant="subtle"
              color="gray"
              onClick={handleReset}
              disabled={overrideValue === undefined}
            >
              Reset to current
            </Button>
            {overrideValue === undefined && (
              <Text size="xs" c="dimmed" fs="italic">
                No override — run will use the canonical hierarchy from config.
              </Text>
            )}
          </Group>

          {showJson && (
            <JsonInput
              label="Hierarchy spec (JSON)"
              description={
                overrideValue !== undefined
                  ? "Override active — differs from the researched default."
                  : "Showing the current config default. Edit to create a per-run override."
              }
              placeholder="{}"
              value={localText}
              onChange={handleJsonChange}
              formatOnBlur
              validationError={jsonError ?? undefined}
              autosize
              minRows={6}
              maxRows={24}
              styles={{ input: { fontFamily: "monospace", fontSize: 12 } }}
            />
          )}
        </Stack>
      </Collapse>
    </Stack>
  );
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

interface WeightLayerControlsProps {
  data: WeightLayerData;
  value: WeightLayerOverrides;
  onChange: (patch: WeightLayerOverrides) => void;
}

export default function WeightLayerControls({
  data,
  value,
  onChange,
}: WeightLayerControlsProps) {
  const sr = data.sr;

  const srAdjustment = value.sr_adjustment ?? sr.sr_adjustment;
  const srAvg = value.sr_avg ?? sr.sr_avg ?? undefined;
  const fdmMax = value.fdm_max ?? sr.fdm_max ?? undefined;
  const srTiltMaxDepth = value.sr_tilt_max_depth ?? sr.sr_tilt_max_depth ?? undefined;
  const withinGroupMethod =
    value.within_group_method ?? sr.within_group_method ?? null;

  const withinGroupOptions = data.within_group_methods.map((m) => ({
    value: m,
    label: m,
  }));

  return (
    <Card withBorder padding="lg">
      <Stack gap="md">
        <Group gap="xs" align="center">
          <IconSlash size={16} color="var(--mantine-color-dimmed)" />
          <Text fw={600}>Weight-layer overrides</Text>
          <Text size="xs" c="dimmed">
            — per-run only; the researched default in config is never modified
          </Text>
        </Group>

        <Divider />

        <SectionLabel
          label="Sharpe-tilt"
          tip="SR-tilt adjusts per-group weights by estimated Sharpe ratios. Enable sr_adjustment to activate the tilt; the other fields control its aggressiveness and depth."
        />

        <Group gap="md" align="flex-start" wrap="wrap">
          <Stack gap={4} style={{ minWidth: 140 }}>
            <Text size="sm" fw={500}>
              SR adjustment
            </Text>
            <Text size="xs" c="dimmed">
              Enable Sharpe-ratio tilt across groups.
            </Text>
            <Switch
              checked={srAdjustment}
              onChange={(e) =>
                onChange({ sr_adjustment: e.currentTarget.checked })
              }
              color="indigo"
              mt={4}
            />
          </Stack>

          <NumberInput
            label="SR avg"
            description="Prior estimate of group-level Sharpe ratio (overrides config default)."
            placeholder={sr.sr_avg != null ? String(sr.sr_avg) : "from config"}
            value={srAvg}
            onChange={(v) =>
              onChange({ sr_avg: v === "" ? undefined : Number(v) })
            }
            step={0.05}
            decimalScale={3}
            min={0}
            disabled={!srAdjustment}
            style={{ width: 160 }}
            size="sm"
          />

          <NumberInput
            label="FDM max"
            description="Cap on the Forecast Diversification Multiplier (default 2.0)."
            placeholder={sr.fdm_max != null ? String(sr.fdm_max) : "from config"}
            value={fdmMax}
            onChange={(v) =>
              onChange({ fdm_max: v === "" ? undefined : Number(v) })
            }
            step={0.1}
            decimalScale={2}
            min={1}
            max={5}
            style={{ width: 160 }}
            size="sm"
          />

          <NumberInput
            label="Tilt max depth"
            description="How many hierarchy levels deep the SR tilt is applied (integer ≥ 1)."
            placeholder={
              sr.sr_tilt_max_depth != null
                ? String(sr.sr_tilt_max_depth)
                : "from config"
            }
            value={srTiltMaxDepth}
            onChange={(v) =>
              onChange({
                sr_tilt_max_depth:
                  v === "" ? undefined : Math.max(1, Math.round(Number(v))),
              })
            }
            step={1}
            decimalScale={0}
            min={1}
            disabled={!srAdjustment}
            style={{ width: 160 }}
            size="sm"
          />

          <Select
            label="Within-group method"
            description="How streams inside each hierarchy group are combined."
            data={withinGroupOptions}
            value={withinGroupMethod}
            onChange={(v) =>
              onChange({ within_group_method: v ?? undefined })
            }
            searchable
            clearable
            placeholder="from config"
            style={{ width: 240 }}
            size="sm"
          />
        </Group>

        <Divider />

        <HierarchyEditor
          current={data.hierarchy}
          overrideValue={value.hierarchy_spec}
          onChange={onChange}
        />

        <Divider />

        <SectionLabel
          label="Sleeves"
          tip="Sleeves are named vault groups. Adding one here registers it with the backend so it becomes available as a vault target in the Spec Builder."
        />
        <AddSleeveRow />
      </Stack>
    </Card>
  );
}
