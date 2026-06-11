import { Card, Divider, Group, Stack, Text, ThemeIcon, Title } from "@mantine/core";

export function FormSection({
  title,
  description,
  icon,
  children,
}: {
  title: string;
  description?: string;
  icon?: React.ReactNode;
  children: React.ReactNode;
}) {
  return (
    <Card withBorder padding="lg" radius="md">
      <Stack gap={4} mb="md">
        <Group gap="xs" align="center">
          {icon && (
            <ThemeIcon size="sm" variant="light" color="brand" radius="sm">
              {icon}
            </ThemeIcon>
          )}
          <Title order={5} style={{ lineHeight: 1.3 }}>
            {title}
          </Title>
        </Group>
        {description && (
          <Text size="xs" c="dimmed" lh={1.5}>
            {description}
          </Text>
        )}
      </Stack>
      <Divider mb="md" opacity={0.4} />
      <Stack gap="md">{children}</Stack>
    </Card>
  );
}
