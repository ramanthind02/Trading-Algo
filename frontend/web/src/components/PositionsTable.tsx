import { Badge, Table, Text } from "@mantine/core";

import type { LivePosition } from "../api/types";
import { fmtNum } from "../lib/format";

const SIDE_COLOR: Record<LivePosition["side"], string> = {
  LONG: "teal",
  SHORT: "red",
  FLAT: "gray",
};

export function PositionsTable({ positions, currency }: { positions: LivePosition[]; currency: string }) {
  if (positions.length === 0) {
    return (
      <Text c="dimmed" size="sm" py="sm">
        Flat — no open positions.
      </Text>
    );
  }

  return (
    <Table.ScrollContainer minWidth={620}>
      <Table striped highlightOnHover withTableBorder fz="sm">
        <Table.Thead>
          <Table.Tr>
            <Table.Th>Instrument</Table.Th>
            <Table.Th>Side</Table.Th>
            <Table.Th ta="right">Qty (lots)</Table.Th>
            <Table.Th ta="right">Open</Table.Th>
            <Table.Th ta="right">Last</Table.Th>
            <Table.Th ta="right">P&amp;L ({currency})</Table.Th>
            <Table.Th ta="right">Notional</Table.Th>
          </Table.Tr>
        </Table.Thead>
        <Table.Tbody>
          {positions.map((p) => {
            const pnlColor =
              p.unrealized_pnl == null ? undefined : p.unrealized_pnl >= 0 ? "teal" : "red";
            return (
              <Table.Tr key={p.instrument_id}>
                <Table.Td>
                  <Text size="sm" fw={600}>
                    {p.canonical ?? p.symbol}
                  </Text>
                  {p.canonical && (
                    <Text size="xs" c="dimmed">
                      {p.symbol}
                    </Text>
                  )}
                </Table.Td>
                <Table.Td>
                  <Badge color={SIDE_COLOR[p.side]} variant="light" radius="sm">
                    {p.side}
                  </Badge>
                </Table.Td>
                <Table.Td ta="right" ff="monospace">
                  {fmtNum(Math.abs(p.net_qty), 2)}
                </Table.Td>
                <Table.Td ta="right" ff="monospace">
                  {fmtNum(p.avg_px_open, 2)}
                </Table.Td>
                <Table.Td ta="right" ff="monospace">
                  <Text
                    component="span"
                    size="sm"
                    ff="monospace"
                    c={p.mark_stale ? "dimmed" : undefined}
                    title={p.mark_stale ? "stale mark (no live quote between rollover windows)" : undefined}
                  >
                    {fmtNum(p.last_px, 2)}
                    {p.mark_stale ? " *" : ""}
                  </Text>
                </Table.Td>
                <Table.Td ta="right">
                  <Text component="span" c={pnlColor} ff="monospace" size="sm" fw={600}>
                    {fmtNum(p.unrealized_pnl, 2)}
                  </Text>
                </Table.Td>
                <Table.Td ta="right" ff="monospace">
                  {fmtNum(p.notional, 0)}
                </Table.Td>
              </Table.Tr>
            );
          })}
        </Table.Tbody>
      </Table>
    </Table.ScrollContainer>
  );
}
