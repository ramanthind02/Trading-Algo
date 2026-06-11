import {
  ActionIcon,
  Badge,
  Box,
  Button,
  Group,
  MultiSelect,
  Progress,
  Stack,
  Text,
  TextInput,
  Tooltip,
} from "@mantine/core";
import { IconPlus, IconTrash, IconWand } from "@tabler/icons-react";

import { fmtInt } from "../lib/format";
import type { ParamValue } from "../api/types";

export interface ParamRow {
  key: string;
  values: string; // comma-separated; parsed to a list on build
}

/** Parse one token: bool, number, else string (mirrors how the agent writes param lists). */
function parseToken(token: string): ParamValue {
  const s = token.trim();
  if (s === "true") return true;
  if (s === "false") return false;
  const n = Number(s);
  if (s !== "" && !Number.isNaN(n)) return n;
  return s;
}

export function rowsToGrid(rows: ParamRow[]): Record<string, ParamValue[]> {
  const grid: Record<string, ParamValue[]> = {};
  for (const row of rows) {
    const key = row.key.trim();
    if (!key) continue;
    grid[key] = row.values
      .split(",")
      .map((t) => t.trim())
      .filter((t) => t !== "")
      .map(parseToken);
  }
  return grid;
}

export function gridToRows(grid: Record<string, ParamValue[]>): ParamRow[] {
  const rows = Object.entries(grid).map(([key, values]) => ({
    key,
    values: values.join(", "),
  }));
  return rows.length ? rows : [{ key: "", values: "" }];
}

function comboCount(rows: ParamRow[]): number {
  const grid = rowsToGrid(rows);
  const lengths = Object.values(grid).map((v) => v.length);
  if (lengths.length === 0 || lengths.some((l) => l === 0)) return 0;
  return lengths.reduce((a, b) => a * b, 1);
}

/**
 * Parse the single value from a row that has exactly one token.
 * Returns null if the row doesn't have exactly one token or it isn't numeric.
 */
function parseSingleNumeric(valuesStr: string): number | null {
  const tokens = valuesStr
    .split(",")
    .map((t) => t.trim())
    .filter((t) => t !== "");
  if (tokens.length !== 1) return null;
  const n = Number(tokens[0]);
  if (Number.isNaN(n)) return null;
  return n;
}

/**
 * Generate a symmetric sweep of values centered on `center`.
 * - For integers: use integer steps; min step is 1 for small values.
 * - For floats: use fractional steps rounded to a sensible precision.
 * Always includes the center. Drops non-positive values when center > 0.
 * Caps the result at `budget` entries (min 1) and MAX_EXPAND.
 */
const MAX_EXPAND = 12;

function generateSweep(center: number, budget: number): number[] {
  const cappedBudget = Math.max(1, Math.min(budget, MAX_EXPAND));

  const isInt = Number.isInteger(center);

  // Pick a step size (~20% of magnitude, min 1 for integers)
  const rawStep = Math.abs(center) * 0.2;
  let step: number;
  if (isInt) {
    step = Math.max(1, Math.round(rawStep));
    // For small integers like 2, use step=1 so we get ±1, ±2
    if (Math.abs(center) <= 5) step = 1;
  } else {
    // Round step to 1 significant figure at an appropriate decimal place
    const mag = Math.floor(Math.log10(Math.abs(rawStep)));
    const factor = Math.pow(10, mag);
    step = Math.round(rawStep / factor) * factor;
    if (step === 0) step = 0.01;
  }

  // How many steps on each side we can afford (symmetric, center counts as 1 slot)
  const halfCount = Math.floor((cappedBudget - 1) / 2);
  const stepsEachSide = Math.max(0, halfCount);

  const values: number[] = [];
  for (let k = -stepsEachSide; k <= stepsEachSide; k++) {
    const v = center + k * step;
    // Drop non-positive if center is positive
    if (center > 0 && v <= 0) continue;
    values.push(v);
  }

  // Ensure center is present (it always should be, but guard)
  if (!values.includes(center)) values.push(center);

  // Round floats to avoid floating-point noise
  const roundedValues = isInt
    ? values.map(Math.round)
    : (() => {
        // Determine decimal places from step
        const stepStr = step.toString();
        const dotIdx = stepStr.indexOf(".");
        const decimals = dotIdx === -1 ? 0 : stepStr.length - dotIdx - 1;
        const precision = Math.max(decimals, 2);
        return values.map((v) => parseFloat(v.toFixed(precision)));
      })();

  // Deduplicate and sort
  const unique = [...new Set(roundedValues)].sort((a, b) => a - b);

  // Re-cap after dedup (edge case: positive filter reduced count)
  return unique.slice(0, cappedBudget);
}

/**
 * Compute the product of all OTHER rows' value counts (excluding row at `excludeIdx`).
 * Returns 1 if there are no other parsed rows (so budget = maxCombos).
 */
function otherRowsProduct(rows: ParamRow[], excludeIdx: number): number {
  const grid = rowsToGrid(rows);
  const lengths = Object.entries(grid)
    .filter((_, idx) => idx !== excludeIdx)
    .map(([, vals]) => vals.length)
    .filter((l) => l > 0);
  if (lengths.length === 0) return 1;
  return lengths.reduce((a, b) => a * b, 1);
}

export function ParamGridEditor({
  rows,
  onChange,
  maxCombos,
  paramChoices = {},
}: {
  rows: ParamRow[];
  onChange: (rows: ParamRow[]) => void;
  maxCombos: number;
  paramChoices?: Record<string, string[]>;
}) {
  const update = (i: number, patch: Partial<ParamRow>) =>
    onChange(rows.map((r, idx) => (idx === i ? { ...r, ...patch } : r)));
  const add = () => onChange([...rows, { key: "", values: "" }]);
  const remove = (i: number) => onChange(rows.filter((_, idx) => idx !== i));

  const combos = comboCount(rows);
  const over = combos > maxCombos;
  const pct = maxCombos > 0 ? Math.min((combos / maxCombos) * 100, 100) : 0;
  const progressColor = over ? "red" : pct > 75 ? "orange" : "teal";

  /** Handle auto-expand for row i */
  const handleExpand = (i: number) => {
    const center = parseSingleNumeric(rows[i].values);
    if (center === null) return;

    const otherProduct = otherRowsProduct(rows, i);
    const budget = maxCombos > 0 ? Math.floor(maxCombos / Math.max(otherProduct, 1)) : MAX_EXPAND;

    const swept = generateSweep(center, budget);
    const isInt = Number.isInteger(center);
    const newValues = swept.map((v) => (isInt ? v.toString() : v.toString())).join(", ");
    update(i, { values: newValues });
  };

  return (
    <Stack gap="sm">
      {/* Column header labels */}
      <Group gap="xs" align="center" wrap="nowrap">
        <Text size="xs" fw={600} c="dimmed" style={{ flex: "0 0 200px" }}>
          PARAMETER NAME
        </Text>
        <Text size="xs" fw={600} c="dimmed" style={{ flex: 1 }}>
          VALUES (comma-separated)
        </Text>
        {/* spacer for action-icon column */}
        <Box w={60} />
      </Group>

      {rows.map((row, i) => {
        const choices = paramChoices[row.key.trim()];
        const hasChoices = choices != null && choices.length > 0;

        const center = parseSingleNumeric(row.values);
        const isNumeric = center !== null;
        const otherProduct = otherRowsProduct(rows, i);
        const budget = maxCombos > 0 ? Math.floor(maxCombos / Math.max(otherProduct, 1)) : MAX_EXPAND;
        const budgetTooTight = budget < 2;

        // Determine wand tooltip
        let wandTip: string;
        if (!isNumeric) {
          const tokens = row.values.split(",").map((t) => t.trim()).filter(Boolean);
          if (tokens.length === 0) wandTip = "Enter a single numeric value to sweep";
          else if (tokens.length > 1) wandTip = "Auto-expand works only when there is exactly one value";
          else wandTip = "Value must be numeric for auto-expand";
        } else if (budgetTooTight) {
          wandTip = `Combo cap prevents expansion — other rows use ${fmtInt(otherProduct)} of ${fmtInt(maxCombos)} slots`;
        } else {
          wandTip = `Auto-expand: generate a sweep of ~${Math.min(budget, MAX_EXPAND)} values around ${center}`;
        }

        const wandDisabled = !isNumeric || budgetTooTight;

        // Parse current comma-separated values string → selected string array for MultiSelect.
        const selectedValues = row.values
          .split(",")
          .map((t) => t.trim())
          .filter(Boolean);

        return (
          <Group key={i} gap="xs" align="center" wrap="nowrap">
            <TextInput
              placeholder="e.g. short_period"
              value={row.key}
              onChange={(e) => update(i, { key: e.currentTarget.value })}
              style={{ flex: "0 0 200px" }}
              size="sm"
            />
            {hasChoices ? (
              <MultiSelect
                data={choices}
                value={selectedValues}
                onChange={(selected) => update(i, { values: selected.join(", ") })}
                placeholder="select values…"
                style={{ flex: 1 }}
                size="sm"
                hidePickedOptions={false}
              />
            ) : (
              <TextInput
                placeholder="5, 7, 9"
                value={row.values}
                onChange={(e) => update(i, { values: e.currentTarget.value })}
                style={{ flex: 1 }}
                size="sm"
              />
            )}
            {hasChoices ? (
              /* Wand irrelevant for discrete-choice params — keep column width consistent */
              <Box w={22} />
            ) : (
              <Tooltip label={wandTip} withArrow>
                <ActionIcon
                  variant="subtle"
                  color={wandDisabled ? "gray" : "indigo"}
                  onClick={() => !wandDisabled && handleExpand(i)}
                  disabled={wandDisabled}
                  size="sm"
                  aria-label="Auto-expand to parameter sweep"
                >
                  <IconWand size={14} />
                </ActionIcon>
              </Tooltip>
            )}
            <Tooltip label="Remove parameter" withArrow>
              <ActionIcon
                variant="subtle"
                color="red"
                onClick={() => remove(i)}
                disabled={rows.length === 1}
                size="sm"
              >
                <IconTrash size={14} />
              </ActionIcon>
            </Tooltip>
          </Group>
        );
      })}

      {/* Auto-expand hint */}
      <Text size="xs" c="dimmed">
        Tip: enter a single value and click{" "}
        <IconWand size={11} style={{ display: "inline", verticalAlign: "middle" }} />{" "}
        to auto-generate a swept range around it.
      </Text>

      {/* Footer: add button + combo counter */}
      <Group justify="space-between" align="center" mt={4}>
        <Button size="xs" variant="light" leftSection={<IconPlus size={13} />} onClick={add}>
          Add parameter
        </Button>

        <Group gap="xs" align="center">
          {combos > 0 && (
            <Box w={80}>
              <Progress
                value={pct}
                color={progressColor}
                size="xs"
                radius="xl"
              />
            </Box>
          )}
          <Tooltip
            label={
              combos === 0
                ? "Enter param values to see combo count"
                : over
                  ? `Exceeds cap of ${fmtInt(maxCombos)} — reduce values`
                  : `${fmtInt(combos)} of ${fmtInt(maxCombos)} max combos`
            }
            withArrow
          >
            <Badge
              variant={over ? "filled" : "light"}
              color={over ? "red" : combos === 0 ? "gray" : "teal"}
              size="sm"
              style={{ cursor: "default" }}
            >
              {combos === 0 ? "no combos" : `${fmtInt(combos)} combo${combos === 1 ? "" : "s"}`}
              {over ? " — over cap" : ""}
            </Badge>
          </Tooltip>
        </Group>
      </Group>
    </Stack>
  );
}
