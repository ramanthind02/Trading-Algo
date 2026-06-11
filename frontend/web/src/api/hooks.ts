// TanStack Query hooks over the research API.
import {
  useMutation,
  useQuery,
  useQueryClient,
  type UseQueryResult,
} from "@tanstack/react-query";

import { api } from "./client";
import type {
  ArtifactGroup,
  ArtifactPreview,
  ConditionalReturnsData,
  ConditionalReturnsRequest,
  DataAccount,
  DataCoverageStore,
  DataFreshnessStore,
  DataJobRun,
  FlattenResult,
  FormSchema,
  LiveBrokerSummary,
  LiveEquitySeries,
  LiveRisk,
  LiveSnapshot,
  ModuleCatalogItem,
  ModuleDetail,
  PortfolioDefaultsResponse,
  PortfolioJob,
  PortfolioRunOverrides,
  RunResultsResponse,
  SpecDetail,
  SpecRun,
  SpecSummary,
  SpecValidation,
  StrategySpec,
  VaultFeature,
  VaultPreview,
  VaultSaveResult,
  WeightLayerData,
} from "./types";

const keys = {
  schema: ["schema"] as const,
  modules: ["modules"] as const,
  module: (name: string) => ["module", name] as const,
  specs: ["specs"] as const,
  spec: (id: string) => ["spec", id] as const,
  runs: ["runs"] as const,
  run: (id: string) => ["run", id] as const,
  runArtifacts: (id: string) => ["run", id, "artifacts"] as const,
  runResults: (id: string) => ["run", id, "results"] as const,
  preview: (path: string) => ["artifact", path] as const,
};

export function useSchema(): UseQueryResult<FormSchema> {
  return useQuery({ queryKey: keys.schema, queryFn: () => api.get<FormSchema>("/api/schema"), staleTime: Infinity });
}

export function useModules(): UseQueryResult<ModuleCatalogItem[]> {
  return useQuery({
    queryKey: keys.modules,
    queryFn: async () => (await api.get<{ modules: ModuleCatalogItem[] }>("/api/modules")).modules,
    staleTime: Infinity,
  });
}

export function useModuleDetail(name: string | null): UseQueryResult<ModuleDetail> {
  return useQuery({
    queryKey: keys.module(name ?? ""),
    queryFn: () => api.get<ModuleDetail>(`/api/modules/${name}`),
    enabled: !!name,
    staleTime: Infinity,
  });
}

export function useSpecs(): UseQueryResult<SpecSummary[]> {
  return useQuery({
    queryKey: keys.specs,
    queryFn: async () => (await api.get<{ specs: SpecSummary[] }>("/api/specs")).specs,
  });
}

export function useSpec(id: string | null): UseQueryResult<SpecDetail> {
  return useQuery({
    queryKey: keys.spec(id ?? ""),
    queryFn: () => api.get<SpecDetail>(`/api/specs/${id}`),
    enabled: !!id,
  });
}

export function useValidateSpec() {
  return useMutation({
    mutationFn: (spec: StrategySpec) => api.post<SpecValidation>("/api/specs/validate", spec),
  });
}

export function useSaveSpec() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ id, spec }: { id: string | null; spec: StrategySpec }) =>
      id
        ? api.put<SpecDetail>(`/api/specs/${id}`, spec)
        : api.post<SpecDetail>("/api/specs", spec),
    onSuccess: (detail) => {
      qc.invalidateQueries({ queryKey: keys.specs });
      qc.invalidateQueries({ queryKey: keys.spec(detail.id) });
    },
  });
}

export function useDeleteSpec() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (id: string) => api.del<{ status: string }>(`/api/specs/${id}`),
    onSuccess: () => qc.invalidateQueries({ queryKey: keys.specs }),
  });
}

export function useDeleteRun() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (id: string) => api.del<{ status: string }>(`/api/runs/${id}`),
    onSuccess: (_data, id) => {
      qc.invalidateQueries({ queryKey: keys.runs });
      qc.removeQueries({ queryKey: keys.run(id) });
      qc.removeQueries({ queryKey: keys.runResults(id) });
    },
  });
}

export function useStartRun() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: { spec_id?: string; spec?: StrategySpec; phase?: "exploration" | "validation" }) =>
      api.post<SpecRun>("/api/runs", body),
    onSuccess: () => qc.invalidateQueries({ queryKey: keys.runs }),
  });
}

export function useRuns(): UseQueryResult<SpecRun[]> {
  return useQuery({
    queryKey: keys.runs,
    queryFn: async () => (await api.get<{ runs: SpecRun[] }>("/api/runs")).runs,
  });
}

/** Poll a run while it is queued/running; stop once terminal. */
export function useRun(id: string | null): UseQueryResult<SpecRun> {
  return useQuery({
    queryKey: keys.run(id ?? ""),
    queryFn: () => api.get<SpecRun>(`/api/runs/${id}`),
    enabled: !!id,
    refetchInterval: (query) => {
      const status = query.state.data?.status;
      return status === "completed" || status === "failed" ? false : 1500;
    },
  });
}

export function useRunArtifacts(id: string | null, enabled: boolean): UseQueryResult<ArtifactGroup[]> {
  return useQuery({
    queryKey: keys.runArtifacts(id ?? ""),
    queryFn: async () => (await api.get<{ groups: ArtifactGroup[] }>(`/api/runs/${id}/artifacts`)).groups,
    enabled: !!id && enabled,
  });
}

export function useRunResults(id: string | null, enabled: boolean): UseQueryResult<RunResultsResponse> {
  return useQuery({
    queryKey: keys.runResults(id ?? ""),
    queryFn: () => api.get<RunResultsResponse>(`/api/runs/${id}/results`),
    enabled: !!id && enabled,
  });
}

/** On-demand: bucket a run's best combo's returns by a chosen indicator (+ optional regime). */
export function useConditionalReturns(runId: string) {
  return useMutation({
    mutationFn: (body: ConditionalReturnsRequest) =>
      api.post<ConditionalReturnsData>(`/api/runs/${runId}/conditional-returns`, body),
  });
}

export function useArtifactPreview(path: string | null): UseQueryResult<ArtifactPreview> {
  return useQuery({
    queryKey: keys.preview(path ?? ""),
    queryFn: () => api.get<ArtifactPreview>(`/api/artifacts/preview?path=${encodeURIComponent(path!)}`),
    enabled: !!path,
  });
}

export function rawArtifactUrl(path: string): string {
  return `/api/artifacts/raw?path=${encodeURIComponent(path)}`;
}

export function useVaultPreview(specId: string | null): UseQueryResult<VaultPreview> {
  return useQuery({
    queryKey: ["vault-preview", specId ?? ""],
    queryFn: () => api.post<VaultPreview>("/api/vault/preview", { spec_id: specId }),
    enabled: !!specId,
  });
}

export function useVaultCommit() {
  return useMutation({
    mutationFn: (specId: string) => api.post<VaultSaveResult>("/api/vault/commit", { spec_id: specId }),
  });
}

// --- portfolio research ---

export function usePortfolioDefaults(): UseQueryResult<PortfolioDefaultsResponse> {
  return useQuery({
    queryKey: ["portfolio-defaults"],
    queryFn: () => api.get<PortfolioDefaultsResponse>("/api/portfolio/defaults"),
  });
}

export function useStartPortfolioRun() {
  return useMutation({
    mutationFn: (body: PortfolioRunOverrides) => api.post<PortfolioJob>("/api/portfolio/runs", body),
  });
}

export function usePortfolioJob(jobId: string | null): UseQueryResult<PortfolioJob> {
  return useQuery({
    queryKey: ["portfolio-job", jobId ?? ""],
    queryFn: () => api.get<PortfolioJob>(`/api/portfolio/runs/${jobId}`),
    enabled: !!jobId,
    refetchInterval: (query) => {
      const status = query.state.data?.status;
      return status === "completed" || status === "failed" ? false : 2000;
    },
  });
}

export function usePortfolioArtifacts(enabled: boolean): UseQueryResult<ArtifactGroup[]> {
  return useQuery({
    queryKey: ["portfolio-artifacts"],
    queryFn: async () => (await api.get<{ groups: ArtifactGroup[] }>("/api/portfolio/artifacts")).groups,
    enabled,
  });
}

export function useWeightLayer(): UseQueryResult<WeightLayerData> {
  return useQuery({
    queryKey: ["weight-layer"],
    queryFn: () => api.get<WeightLayerData>("/api/portfolio/weight-layer"),
  });
}

export function useAddSleeve() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (name: string) => api.post<{ sleeves: string[] }>("/api/sleeves", { name }),
    onSuccess: () => qc.invalidateQueries({ queryKey: keys.schema }),
  });
}

// --- vault features archive ---

export function useVaultFeatures(profile: string): UseQueryResult<VaultFeature[]> {
  return useQuery({
    queryKey: ["vault-features", profile],
    queryFn: async () =>
      (await api.get<{ features: VaultFeature[] }>(`/api/vault/features?profile=${profile}`)).features,
  });
}

export function useVaultFeatureToSpec() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: { profile: string; path: string }) =>
      api.post<SpecDetail>("/api/vault/features/to-spec", body),
    onSuccess: () => qc.invalidateQueries({ queryKey: keys.specs }),
  });
}

// --- live monitoring (node-mediated dashboard feed) ---

/** Which brokers have a live node publishing (polled slowly). */
export function useLiveBrokers(): UseQueryResult<LiveBrokerSummary[]> {
  return useQuery({
    queryKey: ["live-brokers"],
    queryFn: async () => (await api.get<{ brokers: LiveBrokerSummary[] }>("/api/live/brokers")).brokers,
    refetchInterval: 10000,
  });
}

/** The live snapshot for a broker (account / positions / targets / warmup). */
export function useLiveSnapshot(broker: string | null): UseQueryResult<LiveSnapshot> {
  return useQuery({
    queryKey: ["live-snapshot", broker ?? ""],
    queryFn: () => api.get<LiveSnapshot>(`/api/live/snapshot?broker=${encodeURIComponent(broker!)}`),
    enabled: !!broker,
    refetchInterval: 4000,
    retry: false, // a 404 means the node is offline — don't hammer
  });
}

/** Prop-firm limits + derived utilisation gauges. */
export function useLiveRisk(broker: string | null): UseQueryResult<LiveRisk> {
  return useQuery({
    queryKey: ["live-risk", broker ?? ""],
    queryFn: () => api.get<LiveRisk>(`/api/live/risk?broker=${encodeURIComponent(broker!)}`),
    enabled: !!broker,
    refetchInterval: 4000,
  });
}

/** The persisted live equity series for the curve. */
export function useLiveEquity(broker: string | null): UseQueryResult<LiveEquitySeries> {
  return useQuery({
    queryKey: ["live-equity", broker ?? ""],
    queryFn: () => api.get<LiveEquitySeries>(`/api/live/equity?broker=${encodeURIComponent(broker!)}`),
    enabled: !!broker,
    refetchInterval: 15000,
  });
}

/** Queue the node-mediated flatten-all kill switch (demo-only, confirm-gated). */
export function useFlatten() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ broker, confirm }: { broker: string; confirm: string }) =>
      api.post<FlattenResult>("/api/live/flatten", { broker, confirm }),
    onSuccess: (_data, vars) => {
      qc.invalidateQueries({ queryKey: ["live-snapshot", vars.broker] });
      qc.invalidateQueries({ queryKey: ["live-brokers"] });
    },
  });
}

// --- data coverage (data registry) ---

export function useDataCoverage(): UseQueryResult<DataCoverageStore[]> {
  return useQuery({
    queryKey: ["data-coverage"],
    queryFn: async () =>
      (await api.get<{ stores: DataCoverageStore[] }>("/api/data/coverage")).stores,
  });
}

export function useDataFreshness(): UseQueryResult<DataFreshnessStore[]> {
  return useQuery({
    queryKey: ["data-freshness"],
    queryFn: async () =>
      (await api.get<{ stores: DataFreshnessStore[] }>("/api/data/freshness")).stores,
  });
}

const DATA_JOB_NAMES = [
  "mt5_scrape",
  "signal_refresh",
  "registry_ingest_live",
  "registry_backup",
  "catalog_materialize",
  "registry_rebuild",
] as const;

export function useDataJobs(): UseQueryResult<DataJobRun[]> {
  return useQuery({
    queryKey: ["data-jobs"],
    queryFn: async () => {
      const results = await Promise.all(
        DATA_JOB_NAMES.map((name) =>
          api
            .get<{ jobs: DataJobRun[] }>(`/api/data/jobs?name=${name}&limit=5`)
            .then((r) => r.jobs),
        ),
      );
      return results.flat().sort((a, b) => b.started_at.localeCompare(a.started_at));
    },
  });
}

export function useDataAccounts(): UseQueryResult<DataAccount[]> {
  return useQuery({
    queryKey: ["data-accounts"],
    queryFn: async () =>
      (await api.get<{ accounts: DataAccount[] }>("/api/data/accounts")).accounts,
  });
}
