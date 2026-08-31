<template>
  <div>
    <div v-if="!versions.length" class="empty-state-mini">
      <i class="pi pi-inbox" aria-hidden="true" />
      <span>{{ emptyMessage }}</span>
    </div>
    <div v-else class="version-table">
      <div
        v-for="v in versions"
        :key="v.id ?? v.version"
        class="version-row"
        :class="{ active: v.is_active, broken: isUnusable(v) }"
      >
        <div class="version-info">
          <code class="version-name">{{ v.version }}</code>
          <Tag v-if="v.is_active" value="Active" severity="success" />
          <Tag
            v-if="buildStatus(v)"
            :value="buildStatusLabel(v)"
            :severity="buildStatusSeverity(v)"
            v-tooltip.top="buildStatusTooltip(v)"
          />
          <Tag
            :value="versionTypeLabel(v)"
            :severity="versionTypeSeverity(v)"
            v-tooltip.top="forkTooltip(v)"
          />
          <small v-if="v.repository_source" class="repo-label">{{ v.repository_source }}</small>
          <small v-if="sourceBranch(v)" class="branch-label">
            <i class="pi pi-code" aria-hidden="true" />
            {{ sourceBranch(v) }}
          </small>
          <small v-if="v.build_config?.cuda || v.build_config?.enable_cuda || v.build_config?.backend === 'cuda'" class="cuda-badge">CUDA</small>
          <small v-else-if="v.build_config?.backend" class="cuda-badge">{{ String(v.build_config.backend).toUpperCase() }}</small>
        </div>
        <div class="version-actions">
          <Button
            v-if="canRetryVersion(v)"
            icon="pi pi-replay"
            text
            size="small"
            severity="warning"
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
            v-tooltip.top="'Edit CMake settings for this build'"
            @click="$emit('edit-config', versionId(v))"
          />
          <Button
            v-if="!v.is_active && canActivateVersion(v)"
            label="Activate"
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
            :disabled="v.is_active"
            v-tooltip.top="v.is_active ? 'Active versions cannot be deleted' : 'Delete version'"
            @click="$emit('delete', v.id ?? v.version)"
          />
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import Button from 'primevue/button'
import Tag from 'primevue/tag'

defineProps({
  versions: {
    type: Array,
    default: () => [],
  },
  activating: {
    type: [String, Number],
    default: null,
  },
  syncing: {
    type: [String, Number],
    default: null,
  },
  retrying: {
    type: [String, Number],
    default: null,
  },
  emptyMessage: {
    type: String,
    default: 'No versions installed yet.',
  },
})

defineEmits(['activate', 'delete', 'sync', 'retry', 'edit-config'])

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
  const refType = String(version?.source_ref_type || '').trim().toLowerCase()
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
  return repo
    ? `Built from fork: ${repo}`
    : 'Built from a non-default GitHub owner (fork)'
}

function canSyncSourceVersion(version) {
  if (!canActivateVersion(version)) return false
  const installType = String(version?.install_type || version?.type || '').toLowerCase()
  return ['source', 'fork', 'patched', 'local'].includes(installType) && Boolean(sourceBranch(version))
}

function buildStatus(version) {
  const status = String(version?.build_status || '').trim().toLowerCase()
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

const CMAKE_ENGINES = new Set(['llama_cpp', 'ik_llama', 'audio_cpp'])

function versionEngine(version) {
  const id = String(version?.id || '')
  const prefix = id.includes(':') ? id.split(':')[0] : ''
  if (CMAKE_ENGINES.has(prefix)) return prefix
  const repo = String(version?.repository_source || '')
  if (repo === 'llama.cpp') return 'llama_cpp'
  if (repo === 'ik_llama.cpp') return 'ik_llama'
  if (repo === 'audio.cpp') return 'audio_cpp'
  return ''
}

function canEditBuildConfig(version) {
  if (version?.orphan) return false
  if (version?.cmake_editable === false) return false
  if (version?.cmake_editable === true) return true
  return CMAKE_ENGINES.has(versionEngine(version))
}
</script>

<style scoped>
.empty-state-mini {
  display: flex;
  align-items: flex-start;
  gap: 0.5rem;
  color: var(--text-secondary);
  font-size: 0.875rem;
  line-height: 1.45;
  padding: 0.5rem 0 0;
}

.empty-state-mini > .pi {
  margin-top: 0.1rem;
  flex-shrink: 0;
  opacity: 0.85;
}

.version-table {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}

.version-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0.5rem 0.75rem;
  background: var(--bg-surface);
  border: 1px solid var(--border-primary);
  border-radius: var(--radius-md);
  gap: 0.5rem;
  transition: border-color 0.15s;
}

.version-row.active {
  border-color: var(--accent-green);
}

.version-row.broken {
  border-color: var(--accent-red, #e24c4c);
}

.version-info {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  flex: 1;
  min-width: 0;
  flex-wrap: wrap;
}

.version-name {
  font-weight: 600;
  font-size: 0.875rem;
  font-family: monospace;
}

.repo-label {
  color: var(--text-secondary);
  font-size: 0.75rem;
}

.branch-label {
  display: inline-flex;
  align-items: center;
  gap: 0.25rem;
  color: var(--text-secondary);
  font-size: 0.75rem;
  min-width: 0;
  max-width: 14rem;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.branch-label .pi {
  font-size: 0.7rem;
  flex-shrink: 0;
}

.cuda-badge {
  background: rgba(34, 211, 238, 0.1);
  color: var(--accent-cyan);
  border: 1px solid rgba(34, 211, 238, 0.3);
  border-radius: 0.25rem;
  padding: 0.1em 0.4em;
  font-size: 0.7rem;
  font-weight: 600;
}

.version-actions {
  display: flex;
  gap: 0.25rem;
  flex-shrink: 0;
}
</style>
