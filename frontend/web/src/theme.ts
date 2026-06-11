import { createTheme, type MantineColorsTuple } from "@mantine/core";

// Indigo/periwinkle brand ramp.
const brand: MantineColorsTuple = [
  "#eef2ff",
  "#e0e7ff",
  "#c7d2fe",
  "#a5b4fc",
  "#818cf8",
  "#6366f1",
  "#4f46e5",
  "#4338ca",
  "#3730a3",
  "#312e81",
];

// Teal accent for "good"/success metrics.
const accent: MantineColorsTuple = [
  "#ecfdf5",
  "#d1fae5",
  "#a7f3d0",
  "#6ee7b7",
  "#34d399",
  "#10b981",
  "#059669",
  "#047857",
  "#065f46",
  "#064e3b",
];

export const theme = createTheme({
  primaryColor: "brand",
  colors: { brand, accent },
  primaryShade: { light: 6, dark: 5 },
  autoContrast: true,
  defaultRadius: "md",
  fontFamily:
    "Inter, ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif",
  fontFamilyMonospace: "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace",
  headings: { fontWeight: "680" },
  shadows: {
    xs: "0 1px 2px rgba(0,0,0,0.18)",
    sm: "0 2px 8px rgba(0,0,0,0.22)",
    md: "0 6px 20px rgba(0,0,0,0.28)",
  },
  // App-wide component defaults so every page gets consistent surfaces without per-page work.
  components: {
    Card: { defaultProps: { withBorder: true, radius: "md" } },
    Paper: { defaultProps: { radius: "md" } },
    Button: { defaultProps: { radius: "md" } },
    Badge: { defaultProps: { radius: "sm" } },
    Tooltip: { defaultProps: { withArrow: true, openDelay: 200 } },
    Table: { defaultProps: { verticalSpacing: "xs", horizontalSpacing: "md" } },
    Select: { defaultProps: { checkIconPosition: "right", comboboxProps: { shadow: "md" } } },
    MultiSelect: { defaultProps: { comboboxProps: { shadow: "md" } } },
  },
});
