import { Badge, Group, Table, Text } from "@mantine/core";
import { IconArrowDown, IconArrowUp, IconArrowsUpDown } from "@tabler/icons-react";
import { useMemo, useState } from "react";

import type { GridRow } from "../api/types";
import { fmtInt, fmtNum, metricColor } from "../lib/format";

// ── column descriptors ────────────────────────────────────────────────────────

type ColColor = (v: number) => string;

type MetricCol = {
  key: keyof GridRow;
  label: string;
  fmt: (v: number) => string;
  color?: ColColor;
  align?: "right" | "left";
};

const METRIC_COLS: MetricCol[] = [
  {
    key: "t_stat",
    label: "t-stat",
    fmt: (v) => fmtNum(v, 2),
    color: (v) => metricColor(v, 2, 1),
  },
  {
    key: "sharpe",
    label: "Sharpe",
    fmt: (v) => fmtNum(v, 3),
    color: (v) => metricColor(v, 0.5, 0.2),
  },
  {
    key: "sortino",
    label: "Sortino",
    fmt: (v) => fmtNum(v, 3),
    color: (v) => metricColor(v, 0.7, 0.3),
  },
  {
    key: "nw_sharpe",
    label: "NW Sharpe",
    fmt: (v) => fmtNum(v, 3),
    color: (v) => metricColor(v, 0.5, 0.2),
  },
  {
    key: "n_observations",
    label: "n obs",
    fmt: (v) => fmtInt(v),
    align: "right",
  },
  {
    key: "n_nonzero_signal",
    label: "n signal",
    fmt: (v) => fmtInt(v),
    align: "right",
  },
];

// ── styles (defined once, not per-render) ─────────────────────────────────────

const STYLES = {
  /** sticky thead — requires the scroll container to have a fixed height */
  thead: {
    position: "sticky" as const,
    top: 0,
    zIndex: 1,
    background: "var(--mantine-color-dark-7)",
  },
  th: {
    cursor: "pointer",
    whiteSpace: "nowrap" as const,
    userSelect: "none" as const,
    paddingTop: 8,
    paddingBottom: 8,
  },
  thActive: {
    color: "var(--mantine-color-indigo-4)",
  },
  numTd: {
    textAlign: "right" as const,
    fontFamily: "var(--mantine-font-family-monospace, monospace)",
    fontSize: "var(--mantine-font-size-sm)",
  },
  bestRow: {
    background: "rgba(32,201,151,0.08)",
    outline: "1px solid rgba(32,201,151,0.20)",
    outlineOffset: "-1px",
  },
};

// ── sort icon helper ──────────────────────────────────────────────────────────

function SortIcon({ active, desc }: { active: boolean; desc: boolean }) {
  if (!active)
    return <IconArrowsUpDown size={11} style={{ opacity: 0.3 }} />;
  return desc ? (
    <IconArrowDown size={12} style={{ color: "var(--mantine-color-indigo-4)" }} />
  ) : (
    <IconArrowUp size={12} style={{ color: "var(--mantine-color-indigo-4)" }} />
  );
}

// ── component ─────────────────────────────────────────────────────────────────

export function GridTable({ rows, paramKeys }: { rows: GridRow[]; paramKeys: string[] }) {
  const [sortKey, setSortKey] = useState<string>("t_stat");
  const [desc, setDesc] = useState(true);

  // ── sorting ──────────────────────────────────────────────────────────────
  const sorted = useMemo(() => {
    const get = (r: GridRow): number | string => {
      if (paramKeys.includes(sortKey)) return r.params[sortKey] ?? "";
      return (r[sortKey as keyof GridRow] as number) ?? -Infinity;
    };
    return [...rows].sort((a, b) => {
      const av = get(a);
      const bv = get(b);
      const cmp =
        typeof av === "number" && typeof bv === "number"
          ? av - bv
          : String(av).localeCompare(String(bv));
      return desc ? -cmp : cmp;
    });
  }, [rows, sortKey, desc, paramKeys]);

  const toggle = (key: string) => {
    if (sortKey === key) setDesc((d) => !d);
    else {
      setSortKey(key);
      setDesc(true);
    }
  };

  // ── header cell factory ───────────────────────────────────────────────────
  const Th = ({ colKey, label, rightAlign }: { colKey: string; label: string; rightAlign?: boolean }) => {
    const active = sortKey === colKey;
    return (
      <Table.Th
        key={colKey}
        style={{
          ...STYLES.th,
          ...(active ? STYLES.thActive : {}),
          textAlign: rightAlign ? "right" : undefined,
        }}
        onClick={() => toggle(colKey)}
      >
        <Group gap={4} wrap="nowrap" justify={rightAlign ? "flex-end" : "flex-start"}>
          <span>{label}</span>
          <SortIcon active={active} desc={desc} />
        </Group>
      </Table.Th>
    );
  };

  // ── render ────────────────────────────────────────────────────────────────
  return (
    <Table.ScrollContainer minWidth={560} style={{ maxHeight: "60vh", overflow: "auto" }}>
      <Table striped highlightOnHover withTableBorder fz="sm" style={{ borderCollapse: "separate", borderSpacing: 0 }}>
        <Table.Thead style={STYLES.thead}>
          <Table.Tr>
            <Table.Th style={STYLES.th}>Combo</Table.Th>
            {paramKeys.map((k) => (
              <Th key={k} colKey={k} label={k} />
            ))}
            {METRIC_COLS.map((c) => (
              <Th key={c.key as string} colKey={c.key as string} label={c.label} rightAlign />
            ))}
          </Table.Tr>
        </Table.Thead>

        <Table.Tbody>
          {sorted.map((row) => (
            <Table.Tr key={row.label} style={row.is_best ? STYLES.bestRow : undefined}>
              {/* ── label ── */}
              <Table.Td>
                <Group gap="xs" wrap="nowrap">
                  <Text size="xs" ff="monospace">
                    {row.label.replace(/^.*?__/, "")}
                  </Text>
                  {row.is_best && (
                    <Badge size="xs" color="teal" variant="light" radius="sm">
                      best
                    </Badge>
                  )}
                </Group>
              </Table.Td>

              {/* ── param columns ── */}
              {paramKeys.map((k) => (
                <Table.Td key={k} style={STYLES.numTd}>
                  {String(row.params[k] ?? "")}
                </Table.Td>
              ))}

              {/* ── metric columns ── */}
              {METRIC_COLS.map((c) => {
                const v = row[c.key] as number | null;
                const formatted = v == null ? "—" : c.fmt(v);
                const col = v != null && c.color ? c.color(v) : undefined;
                return (
                  <Table.Td key={c.key as string} style={STYLES.numTd}>
                    {col ? (
                      <Text component="span" c={col} ff="monospace" size="sm" fw={col === "teal" ? 500 : undefined}>
                        {formatted}
                      </Text>
                    ) : (
                      formatted
                    )}
                  </Table.Td>
                );
              })}
            </Table.Tr>
          ))}
        </Table.Tbody>
      </Table>
    </Table.ScrollContainer>
  );
}
