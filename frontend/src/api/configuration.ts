import axios from '@/api/client.js'

export type RestoreDecision = 'keep' | 'add' | 'replace' | 'skip' | 'unresolved'

export interface BackupItem {
  kind: 'preference' | 'model' | 'template' | 'routing'
  id: string
  action: RestoreDecision
  local_id?: string
  reason?: string
}

export interface BackupPreview {
  schema_version: number
  plan_id: string | null
  applicable: boolean
  notice: string
  revisions: Record<string, string>
  items: BackupItem[]
  limits: { includes: string[]; excludes: string[] }
}

export interface RestoreSelection {
  backup: Record<string, unknown>
  decisions: Record<string, Record<string, RestoreDecision>>
  mapping: Record<string, string>
}

export interface HistoryEntry {
  id: string
  created_at: string
  document: string
  reason: string
}

export interface HistoryChange {
  path: string
  before: unknown
  after: unknown
}

export interface HistoryDiff extends HistoryEntry {
  current_revision: string
  changes: HistoryChange[]
}

export interface HistoryScope {
  kind: 'preference' | 'model' | 'template' | 'profile' | 'selector'
  item_id: string
}

export interface BenchmarkResult {
  id: string
  created_at: number
  model_id: string
  proxy_name: string
  config_revision: string
  config_fingerprint: string
  prompt: string
  max_tokens: number
  time_to_first_token_ms: number
  total_seconds: number
  completion_tokens: number | null
  tokens_per_second: number | null
  peak_observed_gpu_memory_bytes: number | null
  output_preview: string
}

export async function previewConfiguration(selection: RestoreSelection): Promise<BackupPreview> {
  const { data } = await axios.post<BackupPreview>('/api/config-backup/preview', selection)
  return data
}

export async function applyConfiguration(selection: RestoreSelection, planId: string) {
  const { data } = await axios.post('/api/config-backup/apply', {
    ...selection,
    plan_id: planId,
  })
  return data as { plan_id: string; outcome: 'completed'; notice: string }
}

export async function reconcileConfiguration() {
  const { data } = await axios.post('/api/config-backup/reconcile')
  return data as { outcome: string }
}

export async function listConfigurationHistory(): Promise<HistoryEntry[]> {
  const { data } = await axios.get<HistoryEntry[]>('/api/config-history')
  return data
}

export async function configurationHistoryDiff(entryId: string): Promise<HistoryDiff> {
  const { data } = await axios.get<HistoryDiff>(
    `/api/config-history/${encodeURIComponent(entryId)}`,
  )
  return data
}

export async function restoreConfigurationHistory(
  entryId: string,
  scope: HistoryScope,
  expectedRevision: string,
) {
  const { data } = await axios.post(`/api/config-history/${encodeURIComponent(entryId)}/restore`, {
    ...scope,
    expected_revision: expectedRevision,
  })
  return data as { outcome: 'completed'; notice: string; revision: string }
}

export async function listModelBenchmarks(modelId: string): Promise<BenchmarkResult[]> {
  const { data } = await axios.get<BenchmarkResult[]>(
    `/api/benchmarks/${encodeURIComponent(modelId)}`,
  )
  return data
}

export async function runModelBenchmark(modelId: string, prompt: string, maxTokens: number) {
  const { data } = await axios.post<BenchmarkResult>('/api/benchmarks/run', {
    model_id: modelId,
    prompt,
    max_tokens: maxTokens,
  })
  return data
}
