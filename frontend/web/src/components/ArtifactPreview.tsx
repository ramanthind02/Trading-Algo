import {
  Anchor,
  Badge,
  Box,
  Center,
  Group,
  Image,
  Loader,
  Paper,
  ScrollArea,
  Table,
  Text,
} from "@mantine/core";
import {
  IconAlertCircle,
  IconBinary,
  IconCsv,
  IconDownload,
  IconFileCode,
  IconFileText,
  IconHtml,
  IconMarkdown,
  IconPhoto,
} from "@tabler/icons-react";

import { rawArtifactUrl, useArtifactPreview } from "../api/hooks";
import { fmtInt } from "../lib/format";

// ---------------------------------------------------------------------------
// Public component
// ---------------------------------------------------------------------------

export function ArtifactPreview({ path }: { path: string }) {
  const preview = useArtifactPreview(path);

  if (preview.isLoading) {
    return (
      <Center py="xl">
        <Group gap="xs" c="dimmed">
          <Loader size="xs" />
          <Text size="sm">Loading preview…</Text>
        </Group>
      </Center>
    );
  }

  if (preview.isError) {
    return (
      <Group gap="xs" c="red" py="sm">
        <IconAlertCircle size={16} />
        <Text size="sm">Failed to load: {String(preview.error)}</Text>
      </Group>
    );
  }

  const data = preview.data;
  if (!data) return null;

  return (
    <Box>
      {/* ── top bar: kind badge + download link ── */}
      <Group justify="space-between" mb="sm" wrap="nowrap">
        <KindBadge kind={data.kind} />
        <Anchor
          href={rawArtifactUrl(path)}
          target="_blank"
          rel="noreferrer"
          size="xs"
          c="dimmed"
          style={{ display: "flex", alignItems: "center", gap: 4, whiteSpace: "nowrap" }}
        >
          <IconDownload size={12} style={{ flexShrink: 0 }} />
          Download / open raw
        </Anchor>
      </Group>

      {/* ── content ── */}
      {renderBody(data)}
    </Box>
  );
}

// ---------------------------------------------------------------------------
// Kind badge
// ---------------------------------------------------------------------------

const KIND_META: Record<
  string,
  { label: string; color: string; icon: React.ReactNode }
> = {
  csv:      { label: "CSV",      color: "teal",   icon: <IconCsv      size={11} /> },
  json:     { label: "JSON",     color: "indigo", icon: <IconFileCode size={11} /> },
  markdown: { label: "Markdown", color: "grape",  icon: <IconMarkdown size={11} /> },
  text:     { label: "Text",     color: "gray",   icon: <IconFileText size={11} /> },
  image:    { label: "Image",    color: "cyan",   icon: <IconPhoto    size={11} /> },
  html:     { label: "HTML",     color: "orange", icon: <IconHtml     size={11} /> },
  binary:   { label: "Binary",   color: "gray",   icon: <IconBinary   size={11} /> },
};

function KindBadge({ kind }: { kind: string }) {
  const meta = KIND_META[kind] ?? { label: kind.toUpperCase(), color: "gray", icon: null };
  return (
    <Badge
      variant="light"
      color={meta.color}
      size="sm"
      leftSection={meta.icon}
      style={{ textTransform: "none", fontWeight: 500 }}
    >
      {meta.label}
    </Badge>
  );
}

// ---------------------------------------------------------------------------
// Per-kind renderers
// ---------------------------------------------------------------------------

function renderBody(data: NonNullable<ReturnType<typeof useArtifactPreview>["data"]>) {
  // ── CSV ──
  if (data.kind === "csv") {
    return (
      <>
        <ScrollArea h={400} type="auto" offsetScrollbars>
          <Table
            striped
            highlightOnHover
            withTableBorder
            withColumnBorders
            stickyHeader
            fz="xs"
            style={{ fontVariantNumeric: "tabular-nums" }}
          >
            <Table.Thead>
              <Table.Tr>
                {data.columns.map((col) => (
                  <Table.Th
                    key={col}
                    style={{
                      whiteSpace: "nowrap",
                      fontWeight: 600,
                      letterSpacing: "0.02em",
                    }}
                  >
                    {col}
                  </Table.Th>
                ))}
              </Table.Tr>
            </Table.Thead>
            <Table.Tbody>
              {data.rows.map((row, i) => (
                <Table.Tr key={i}>
                  {data.columns.map((col) => {
                    const raw = row[col];
                    const isNum = typeof raw === "number";
                    return (
                      <Table.Td
                        key={col}
                        style={isNum ? { fontFamily: "monospace", textAlign: "right" } : undefined}
                      >
                        {formatCell(raw)}
                      </Table.Td>
                    );
                  })}
                </Table.Tr>
              ))}
            </Table.Tbody>
          </Table>
        </ScrollArea>
        {data.truncated && (
          <Text size="xs" c="dimmed" mt={6}>
            Showing first {fmtInt(data.rows.length)} of {fmtInt(data.total_rows)} rows
          </Text>
        )}
      </>
    );
  }

  // ── JSON ──
  if (data.kind === "json") {
    return (
      <ScrollArea h={400} type="auto" offsetScrollbars>
        <Paper
          withBorder
          radius="sm"
          p="sm"
          style={{
            background: "var(--mantine-color-dark-8, #1a1b1e)",
            fontFamily: "monospace",
            fontSize: "var(--mantine-font-size-xs)",
            lineHeight: 1.6,
            whiteSpace: "pre-wrap",
            wordBreak: "break-word",
            color: "var(--mantine-color-gray-3)",
          }}
        >
          {JSON.stringify(data.data, null, 2)}
        </Paper>
      </ScrollArea>
    );
  }

  // ── Markdown / Text ──
  if (data.kind === "markdown" || data.kind === "text") {
    return (
      <ScrollArea h={400} type="auto" offsetScrollbars>
        <Paper
          withBorder
          radius="sm"
          p="sm"
          style={{
            background: "var(--mantine-color-dark-8, #1a1b1e)",
            fontFamily: data.kind === "text" ? "monospace" : "inherit",
            fontSize: "var(--mantine-font-size-xs)",
            lineHeight: 1.7,
            whiteSpace: "pre-wrap",
            wordBreak: "break-word",
            color: "var(--mantine-color-gray-3)",
          }}
        >
          {data.text}
        </Paper>
      </ScrollArea>
    );
  }

  // ── Image ──
  if (data.kind === "image") {
    return (
      <Paper withBorder radius="sm" p="xs" style={{ background: "var(--mantine-color-dark-7, #25262b)" }}>
        <Image
          src={rawArtifactUrl(data.path)}
          fit="contain"
          mah={480}
          radius="xs"
          style={{ display: "block" }}
        />
      </Paper>
    );
  }

  // ── HTML ──
  if (data.kind === "html") {
    return (
      <Paper withBorder radius="sm" p="md" style={{ background: "var(--mantine-color-dark-7, #25262b)" }}>
        <Group gap="xs" c="dimmed">
          <IconHtml size={16} />
          <Text size="sm">
            HTML file —{" "}
            <Anchor href={rawArtifactUrl(data.path)} target="_blank" rel="noreferrer" size="sm">
              open {data.path.split("/").pop()}
            </Anchor>
          </Text>
        </Group>
      </Paper>
    );
  }

  // ── Binary / fallback ──
  return (
    <Paper withBorder radius="sm" p="md" style={{ background: "var(--mantine-color-dark-7, #25262b)" }}>
      <Group gap="xs" c="dimmed">
        <IconBinary size={16} />
        <Text size="sm">No inline preview available for this file type.</Text>
      </Group>
    </Paper>
  );
}

// ---------------------------------------------------------------------------
// Cell formatter
// ---------------------------------------------------------------------------

function formatCell(value: unknown): string {
  if (value === null || value === undefined) return "";
  if (typeof value === "number") {
    return Number.isInteger(value) ? String(value) : value.toFixed(4);
  }
  return String(value);
}
