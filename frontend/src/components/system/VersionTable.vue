<template>
  <div class="table-shell version-table">
    <div v-if="!versions.length" class="table-shell__state version-table__empty">
      <i class="pi pi-inbox" aria-hidden="true" />
      <span>{{ emptyMessage }}</span>
    </div>
    <div v-else class="table-shell__scroll">
      <table>
        <caption class="sr-only">
          Installed engine versions
        </caption>
        <thead>
          <tr>
            <th scope="col">Version</th>
            <th scope="col">Status</th>
            <th scope="col">Source</th>
            <th scope="col" class="version-table__actions-heading">Actions</th>
          </tr>
        </thead>
        <tbody>
          <tr
            v-for="v in versions"
            :key="v.id ?? v.version"
            class="version-row"
            :class="{ active: v.is_active, broken: isUnusable(v) }"
          >
            <td class="version-name-cell">
              <code class="version-name">{{ v.version }}</code>
            </td>
            <td class="version-status-cell">
              <Tag v-if="v.is_active" value="Active" severity="success" />
              <Tag
                v-if="buildStatus(v)"
                :value="buildStatusLabel(v)"
                :severity="buildStatusSeverity(v)"
                v-tooltip.top="buildStatusTooltip(v)"
              />
            </td>
            <td class="version-source-cell">
              <div class="version-source">
                <Tag
                  :value="versionTypeLabel(v)"
                  :severity="versionTypeSeverity(v)"
                  v-tooltip.top="forkTooltip(v)"
                />
                <small v-if="v.repository_source" class="repo-label">{{
                  v.repository_source
                }}</small>
                <small v-if="sourceBranch(v)" class="branch-label">
                  <i class="pi pi-code" aria-hidden="true" />
                  {{ sourceBranch(v) }}
                </small>
                <small
                  v-if="
                    v.build_config?.cuda ||
                    v.build_config?.enable_cuda ||
                    v.build_config?.backend === 'cuda'
                  "
                  class="cuda-badge"
                  >CUDA</small
                >
              </div>
            </td>
            <td class="version-actions">
              <Button
                v-if="canRetryVersion(v)"
                icon="pi pi-replay"
                text
                size="small"
                severity="warning"
                :aria-label="`Retry ${v.version}`"
                :loading="retrying === versionId(v)"
                :disabled="retrying === versionId(v)"
                v-tooltip.top="'Retry this failed build'"
                @click="$emit('retry', versionId(v))"
              />
              <Button
                v-if="canSyncSourceVersion(v)"
                icon="pi pi-refresh"
                text
                size="small"
                severity="info"
                :aria-label="`Sync ${sourceBranch(v)} and rebuild ${v.version}`"
                :loading="syncing === versionId(v)"
                :disabled="syncing === versionId(v)"
                v-tooltip.top="`Sync ${sourceBranch(v)} and rebuild incrementally`"
                @click="$emit('sync', versionId(v))"
              />
              <Button
                v-if="canEditBuildConfig(v)"
                icon="pi pi-sliders-h"
                text
                size="small"
                severity="secondary"
                :aria-label="`Edit build settings for ${v.version}`"
                v-tooltip.top="'Edit CMake settings for this build'"
                @click="$emit('edit-config', versionId(v))"
              />
              <Button
                v-if="canVerify && buildStatus(v) !== 'building'"
                icon="pi pi-check-circle"
                text
                size="small"
                :aria-label="`Verify ${v.version}`"
                v-tooltip.top="'Check llama-server, llama-cli, and llama-quantize'"
                @click="$emit('verify', v.id ?? v.version)"
              />
              <Button
                v-if="!v.is_active && canActivateVersion(v)"
                :label="
                  activating === (v.id ?? v.version) && activatingMessage
                    ? activatingMessage
                    : 'Activate'
                "
                icon="pi pi-play"
                size="small"
                severity="success"
                outlined
                :loading="activating === (v.id ?? v.version)"
                @click="$emit('activate', v.id ?? v.version)"
              />
              <Button
                icon="pi pi-trash"
                text
                severity="danger"
                size="small"
                :aria-label="`Delete ${v.version}`"
                :loading="deleting === versionId(v)"
                :disabled="deleting === versionId(v)"
                v-tooltip.top="'Delete version'"
                @click="$emit('delete', v.id ?? v.version)"
              />
            </td>
          </tr>
        </tbody>
      </table>
    </div>
  </div>
</template>

<script setup>
import Button from 'primevue/button'
import Tag from 'primevue/tag'
import { CMAKE_ENGINE_IDS, ENGINE_DESCRIPTORS } from '../../engines/registry.js'

defineProps({
  versions: {
    type: Array,
    default: () => [],
  },
  activating: {
    type: [String, Number],
    default: null,
  },
  activatingMessage: {
    type: String,
    default: '',
  },
  syncing: {
    type: [String, Number],
    default: null,
  },
  retrying: {
    type: [String, Number],
    default: null,
  },
  deleting: {
    type: [String, Number],
    default: null,
  },
  emptyMessage: {
    type: String,
    default: 'No versions installed yet.',
  },
  canVerify: {
    type: Boolean,
    default: false,
  },
})

defineEmits(['activate', 'delete', 'sync', 'retry', 'edit-config', 'verify'])

function versionId(version) {
  return version?.id ?? version?.version
}

function looksLikeCommitRef(value) {
  return /^[0-9a-f]{7,40}$/i.test(String(value || '').trim())
}

function looksLikeReleaseTag(value) {
  return /^(?:v?\d+(?:\.\d+){1,}(?:[-+][0-9A-Za-z._-]+)?|b\d+)$/i.test(String(value || '').trim())
}

function sourceBranch(version) {
  const branch = String(version?.source_branch || '').trim()
  if (branch) return branch
  const ref = String(version?.source_ref || '').trim()
  const refType = String(version?.source_ref_type || '')
    .trim()
    .toLowerCase()
  if (refType === 'branch' && ref) return ref
  if (refType === 'commit' || refType === 'release') return ''
  if (ref && !looksLikeCommitRef(ref) && !looksLikeReleaseTag(ref)) return ref
  return ''
}

function versionTypeLabel(version) {
  return version?.type || version?.install_type || 'source'
}

function versionTypeSeverity(version) {
  const kind = String(versionTypeLabel(version)).toLowerCase()
  if (kind === 'fork' || version?.is_fork) return 'warning'
  if (kind === 'patched') return 'info'
  return 'secondary'
}

function forkTooltip(version) {
  if (!(version?.is_fork || versionTypeLabel(version) === 'fork')) return undefined
  const repo = String(version?.source_repo || '').trim()
  return repo ? `Built from fork: ${repo}` : 'Built from a non-default GitHub owner (fork)'
}

function canSyncSourceVersion(version) {
  if (!canActivateVersion(version)) return false
  const installType = String(version?.install_type || version?.type || '').toLowerCase()
  return (
    ['source', 'fork', 'patched', 'local'].includes(installType) && Boolean(sourceBranch(version))
  )
}

function buildStatus(version) {
  const status = String(version?.build_status || '')
    .trim()
    .toLowerCase()
  if (['building', 'failed', 'cancelled', 'broken'].includes(status)) return status
  return ''
}

function buildStatusLabel(version) {
  const status = buildStatus(version)
  if (status === 'building') return 'Building'
  if (status === 'failed') return 'Failed'
  if (status === 'cancelled') return 'Cancelled'
  if (status === 'broken') return 'Broken'
  return status
}

function buildStatusSeverity(version) {
  const status = buildStatus(version)
  if (status === 'building') return 'info'
  if (status === 'cancelled') return 'warn'
  return 'danger'
}

function buildStatusTooltip(version) {
  const error = String(version?.build_error || '').trim()
  return error || undefined
}

function isUnusable(version) {
  return ['building', 'failed', 'cancelled', 'broken'].includes(buildStatus(version))
}

function canActivateVersion(version) {
  return !isUnusable(version)
}

function canRetryVersion(version) {
  if (version?.retryable === true) return true
  if (version?.retryable === false) return false
  return ['failed', 'cancelled', 'broken'].includes(buildStatus(version))
}

const CMAKE_ENGINES = new Set(CMAKE_ENGINE_IDS)

function versionEngine(version) {
  const id = String(version?.id || '')
  const prefix = id.includes(':') ? id.split(':')[0] : ''
  if (CMAKE_ENGINES.has(prefix)) return prefix
  const repo = String(version?.repository_source || '')
  return (
    Object.entries(ENGINE_DESCRIPTORS).find(
      ([, descriptor]) => descriptor.repoSource === repo,
    )?.[0] || ''
  )
}

function canEditBuildConfig(version) {
  if (version?.orphan) return false
  if (version?.cmake_editable === false) return false
  if (version?.cmake_editable === true) return true
  return CMAKE_ENGINES.has(versionEngine(version))
}
</script>

<style scoped>
.version-table {
  margin: 0;
}

.version-table__empty {
  justify-content: flex-start;
  gap: 0.5rem;
  padding: 0.75rem;
  font-size: var(--text-sm);
  line-height: 1.45;
  text-align: left;
}

.version-table__empty :deep(.pi) {
  flex-shrink: 0;
  margin-top: 0.1rem;
  color: var(--text-secondary);
}

.version-table .table-shell__scroll > table {
  min-width: 34rem;
  font-size: var(--text-sm);
}

.version-row.active {
  background: color-mix(in srgb, var(--status-success) 8%, transparent);
}

.version-row.broken {
  background: color-mix(in srgb, var(--status-error) 8%, transparent);
}

.version-row.active > td:first-child {
  box-shadow: inset 2px 0 0 var(--status-success);
}

.version-row.broken > td:first-child {
  box-shadow: inset 2px 0 0 var(--status-error);
}

.version-name-cell {
  white-space: nowrap;
}

.version-name {
  font-family: monospace;
  font-size: var(--text-xs);
  font-weight: 600;
}

.version-status-cell {
  white-space: nowrap;
}

.version-status-cell :deep(.p-tag) {
  margin-right: 0.25rem;
}

.version-source {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 0.4rem;
  min-width: 0;
}

.repo-label {
  color: var(--text-secondary);
  font-size: var(--text-xs);
}

.branch-label {
  display: inline-flex;
  align-items: center;
  gap: 0.25rem;
  max-width: 14rem;
  overflow: hidden;
  color: var(--text-secondary);
  font-size: var(--text-xs);
  text-overflow: ellipsis;
  white-space: nowrap;
}

.branch-label .pi {
  flex-shrink: 0;
  font-size: var(--text-xs);
}

.cuda-badge {
  padding: 0.1em 0.4em;
  border: 1px solid color-mix(in srgb, var(--accent-primary) 45%, transparent);
  border-radius: var(--radius-sm);
  background: var(--accent-cyan-soft);
  color: var(--accent-primary);
  font-size: var(--text-xs);
  font-weight: 600;
}

.version-table__actions-heading {
  text-align: right;
}

.version-actions {
  display: flex;
  flex-wrap: wrap;
  justify-content: flex-end;
  gap: 0.25rem;
  white-space: nowrap;
}

.version-actions :deep(.p-button) {
  min-width: 2rem;
}
</style>
