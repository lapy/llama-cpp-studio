import axios from '@/api/client.js'

export type StorageCategoryId =
  | 'models'
  | 'hf_cache'
  | 'engine_installs'
  | 'cuda_toolkits'
  | 'jit_caches'
  | 'history_backups'
  | 'accounting_db'

export type StorageUnitKind =
  | 'model_payload'
  | 'hf_repository'
  | 'engine_install'
  | 'engine_cache'
  | 'build_workspace'
  | 'ccache'
  | 'cuda_installer_cache'
  | 'cuda_toolkit'
  | 'history_revision'
  | 'accounting_db'

export interface StorageBlocked {
  code: string
  message: string
}

export interface StorageUnit {
  id: string
  label: string
  kind: StorageUnitKind
  bytes: number
  cleanable: boolean
  blocked: StorageBlocked | null
  manage_path: string | null
  outside_data_root: boolean
  detail: Record<string, unknown>
}

export interface StorageCategory {
  id: StorageCategoryId
  label: string
  description: string
  bytes: number
  unmanaged_bytes: number
  unit_count: number
  cleanable: boolean
  cleanable_bytes: number
  blocked: StorageBlocked | null
  manage_path: string | null
  outside_data_root: boolean
  available: boolean
  truncated: boolean
  notes: string[]
  units: StorageUnit[]
}

export interface StorageFilesystem {
  data_root: string
  total_bytes: number | null
  used_bytes: number | null
  free_bytes: number | null
}

export interface StorageScan {
  quality: 'complete' | 'partial'
  cached: boolean
  entries_scanned: number
  max_scan_entries: number
  max_units_per_category: number
  truncated: boolean
  category_count: number
  notes: string[]
}

export interface StorageInventory {
  schema_version: number
  generated_at: string
  cached: boolean
  filesystem: StorageFilesystem
  scan: StorageScan
  categories: StorageCategory[]
}

export interface StorageCleanupUnit {
  id: string
  label: string
  bytes: number
}

export interface StorageCleanupRefusal extends StorageCleanupUnit {
  code: string
  message: string
}

export interface StorageCleanupPlan {
  plan_id: string
  category: StorageCategoryId
  category_label: string
  keep_history: number | null
  units: StorageCleanupUnit[]
  refused: StorageCleanupRefusal[]
  reclaimable_bytes: number
  confirmation_phrase: string
  scope_cap_units: number
  expires_at: string
  notice: string
}

export interface StorageCleanupResult {
  plan_id: string
  category: StorageCategoryId
  category_label: string
  outcome: 'completed' | 'partial' | 'refused'
  freed_bytes: number
  removed: StorageCleanupUnit[]
  failed: StorageCleanupRefusal[]
  notice: string
}

export interface StorageCleanupPreviewRequest {
  category: StorageCategoryId
  unit_ids?: string[]
  keep_history?: number
}

export async function fetchStorageInventory(options: { refresh?: boolean } = {}) {
  const { data } = await axios.get<StorageInventory>('/api/storage/inventory', {
    params: options.refresh ? { refresh: true } : undefined,
  })
  return data
}

export async function previewStorageCleanup(
  request: StorageCleanupPreviewRequest,
): Promise<StorageCleanupPlan> {
  const { data } = await axios.post<StorageCleanupPlan>('/api/storage/cleanup/preview', request)
  return data
}

export async function applyStorageCleanup(planId: string): Promise<StorageCleanupResult> {
  const { data } = await axios.post<StorageCleanupResult>('/api/storage/cleanup/apply', {
    plan_id: planId,
    confirm: true,
  })
  return data
}
