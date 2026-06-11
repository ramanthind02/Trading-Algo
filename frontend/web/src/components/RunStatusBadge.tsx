import { Badge, Loader } from "@mantine/core";
import { IconCircleCheck, IconCircleX } from "@tabler/icons-react";

import type { RunStatus } from "../api/types";

const COLOR: Record<RunStatus, string> = {
  queued: "gray",
  running: "blue",
  completed: "teal",
  failed: "red",
};

const LABEL: Record<RunStatus, string> = {
  queued: "Queued",
  running: "Running",
  completed: "Completed",
  failed: "Failed",
};

function StatusIcon({ status }: { status: RunStatus }) {
  const size = 12;
  if (status === "running") {
    return <Loader size={size} color={COLOR[status]} />;
  }
  if (status === "queued") {
    return <Loader size={size} color={COLOR[status]} type="dots" />;
  }
  if (status === "completed") {
    return <IconCircleCheck size={size} />;
  }
  if (status === "failed") {
    return <IconCircleX size={size} />;
  }
  return null;
}

export function RunStatusBadge({ status }: { status: RunStatus }) {
  return (
    <Badge
      color={COLOR[status]}
      variant="light"
      leftSection={<StatusIcon status={status} />}
    >
      {LABEL[status]}
    </Badge>
  );
}
