import { AppShell, Badge, Group, NavLink, ScrollArea, Text, ThemeIcon, Title } from "@mantine/core";
import {
  IconActivity,
  IconArchive,
  IconBriefcase,
  IconChartHistogram,
  IconDatabase,
  IconFlask2,
  IconLibrary,
  IconPlus,
} from "@tabler/icons-react";
import { NavLink as RouterNavLink, Route, Routes, useLocation } from "react-router-dom";

import { DataCoveragePage } from "./pages/DataCoveragePage";
import { LiveMonitor } from "./pages/LiveMonitor";
import { PortfolioPage } from "./pages/PortfolioPage";
import { RunResults } from "./pages/RunResults";
import { RunsPage } from "./pages/RunsPage";
import { SpecBuilder } from "./pages/SpecBuilder";
import { SpecLibrary } from "./pages/SpecLibrary";
import { VaultArchivePage } from "./pages/VaultArchivePage";

interface NavItem {
  to: string;
  label: string;
  icon: React.ReactNode;
  match: (path: string) => boolean;
}

const NAV: NavItem[] = [
  {
    to: "/",
    label: "Spec Library",
    icon: <IconLibrary size={18} />,
    match: (p) => p === "/",
  },
  {
    to: "/builder/new",
    label: "New strategy",
    icon: <IconPlus size={18} />,
    match: (p) => p.startsWith("/builder"),
  },
  {
    to: "/runs",
    label: "Runs & results",
    icon: <IconChartHistogram size={18} />,
    match: (p) => p.startsWith("/runs"),
  },
  {
    to: "/portfolio",
    label: "Portfolio research",
    icon: <IconBriefcase size={18} />,
    match: (p) => p.startsWith("/portfolio"),
  },
  {
    to: "/vault",
    label: "Vault archive",
    icon: <IconArchive size={18} />,
    match: (p) => p.startsWith("/vault"),
  },
  {
    to: "/data",
    label: "Data coverage",
    icon: <IconDatabase size={18} />,
    match: (p) => p.startsWith("/data"),
  },
  {
    to: "/live",
    label: "Live monitor",
    icon: <IconActivity size={18} />,
    match: (p) => p.startsWith("/live"),
  },
];

export function App() {
  const location = useLocation();
  return (
    <AppShell header={{ height: 56 }} navbar={{ width: 240, breakpoint: "sm" }} padding="lg">
      <AppShell.Header withBorder>
        <Group h="100%" px="md" gap="sm">
          <ThemeIcon
            size="lg"
            radius="md"
            variant="gradient"
            gradient={{ from: "brand.6", to: "accent.5", deg: 135 }}
          >
            <IconFlask2 size={20} />
          </ThemeIcon>
          <Title order={4}>Trading-Algo Research</Title>
          <Badge variant="light" color="gray" size="sm">
            StrategySpec workbench
          </Badge>
        </Group>
      </AppShell.Header>

      <AppShell.Navbar p="sm">
        <AppShell.Section grow component={ScrollArea}>
          <Text size="xs" fw={700} c="dimmed" tt="uppercase" px="xs" mb={6} style={{ letterSpacing: 0.5 }}>
            Research
          </Text>
          {NAV.map((item) => (
            <NavLink
              key={item.to}
              component={RouterNavLink}
              to={item.to}
              label={item.label}
              leftSection={item.icon}
              active={item.match(location.pathname)}
              style={{ borderRadius: 8 }}
            />
          ))}
        </AppShell.Section>
        <AppShell.Section>
          <Text size="xs" c="dimmed" px="xs">
            Specs live in research/specs/. The agent and this UI share the same JSON.
          </Text>
        </AppShell.Section>
      </AppShell.Navbar>

      <AppShell.Main>
        <Routes>
          <Route path="/" element={<SpecLibrary />} />
          <Route path="/builder/new" element={<SpecBuilder key="new" />} />
          <Route path="/builder/:id" element={<SpecBuilder />} />
          <Route path="/runs" element={<RunsPage />} />
          <Route path="/runs/:id" element={<RunResults />} />
          <Route path="/portfolio" element={<PortfolioPage />} />
          <Route path="/vault" element={<VaultArchivePage />} />
          <Route path="/data" element={<DataCoveragePage />} />
          <Route path="/live" element={<LiveMonitor />} />
        </Routes>
      </AppShell.Main>
    </AppShell>
  );
}
