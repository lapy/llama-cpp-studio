<template>
  <div class="model-library page-shell">

    <PageHeader title="Models">
      <template #meta>
        <Tag
          v-if="totalModels"
          :value="`${totalModels} model${totalModels !== 1 ? 's' : ''}`"
          severity="info"
        />
      </template>
      <template #actions>
        <Button
          icon="pi pi-refresh"
          text
          severity="secondary"
          :loading="modelStore.loading"
          v-tooltip.top="'Refresh'"
          @click="modelStore.fetchModels()"
        />
        <Button
          label="Search &amp; Download"
          icon="pi pi-search"
          severity="success"
          outlined
          @click="$router.push('/search')"
        />
      </template>
    </PageHeader>

    <!-- Download progress (GGUF + Safetensors) -->
    <ProgressTracker type="download" :show-completed="true" />

    <!-- Token Warning -->
    <div v-if="!modelStore.hasHuggingfaceToken" class="token-warning">
      <i class="pi pi-key" />
      <span>No HuggingFace token set. Gated models won't be accessible.</span>
      <Button label="Set Token" icon="pi pi-pencil" size="small" text @click="showTokenDialog = true" />
    </div>

    <LoadingState
      v-if="(modelStore.loading || modelStore.safetensorsLoading) && !modelStore.models.length && !modelStore.safetensorsModels.length"
      message="Loading models…"
    />

    <EmptyState
      v-else-if="!modelStore.loading && !modelStore.safetensorsLoading && !modelStore.models.length && !modelStore.safetensorsModels.length"
      icon="pi pi-inbox"
      title="No models downloaded yet"
      description="Search HuggingFace to find and download models."
    >
      <Button label="Search Models" icon="pi pi-search" @click="$router.push('/search')" />
    </EmptyState>

    <!-- Model groups (GGUF + Safetensors) -->
    <div v-else class="model-groups">
      <div
        v-for="group in displayGroups"
        :key="group.huggingface_id"
        class="model-group"
      >
        <!-- Group header: GGUF (expandable) -->
        <div
          v-if="!isSafetensorsGroup(group)"
          class="group-header interactive-row"
          tabindex="0"
          role="button"
          :aria-expanded="expandedGroups.has(group.huggingface_id)"
          :aria-label="`Toggle ${group.huggingface_id}`"
          @click="toggleGroup(group.huggingface_id)"
          @keydown.enter.prevent="toggleGroup(group.huggingface_id)"
          @keydown.space.prevent="toggleGroup(group.huggingface_id)"
        >
          <div class="group-title">
            <i
              :class="['pi', 'group-chevron', expandedGroups.has(group.huggingface_id) ? 'pi-chevron-down' : 'pi-chevron-right']"
            />
            <span class="group-name">{{ group.huggingface_id }}</span>
            <span
              v-if="groupTotalFileSize(group) > 0"
              class="file-size file-size--total"
              title="Total size (all quantizations)"
            >
              {{ formatBytes(groupTotalFileSize(group)) }}
            </span>
            <Tag
              v-if="group.quantizations?.some(q => q.run_state === 'loading')"
              value="Loading"
              severity="warning"
              class="running-badge"
            />
            <Tag
              v-else-if="group.quantizations?.some(q => q.is_active)"
              value="Running"
              severity="success"
              class="running-badge"
            />
            <Tag
              v-if="primaryQuant(group)"
              :value="(primaryQuant(group).config && primaryQuant(group).config.engine) || (primaryQuant(group).format === 'safetensors' ? 'lmdeploy' : 'llama_cpp')"
              severity="secondary"
              class="engine-tag"
            />
            <Tag
              v-if="primaryQuant(group) && primaryQuant(group).format"
              :value="primaryQuant(group).format"
              severity="info"
            />
          </div>
          <div class="group-meta">
            <span
              v-if="group.quantizations?.length"
              class="group-delete-preview"
              :title="`Remove ${group.quantizations.length} quantization${group.quantizations.length !== 1 ? 's' : ''}`"
              @click.stop
            >
              {{ group.quantizations.length }} {{ group.quantizations.length === 1 ? 'item' : 'items' }}
            </span>
            <Button
              icon="pi pi-trash"
              text
              severity="danger"
              size="small"
              v-tooltip.top="'Delete all quantizations in this group'"
              @click.stop="confirmDeleteGroup(group.huggingface_id)"
            />
          </div>
        </div>

        <!-- Group header: Safetensors (single-row, non-expandable) -->
        <div
          v-else
          class="group-header safetensors-header"
        >
          <div class="group-title">
            <span class="group-name">{{ group.huggingface_id }}</span>
            <span
              v-if="primaryQuant(group) && primaryQuant(group).file_size"
              class="file-size"
            >
              {{ formatBytes(primaryQuant(group).file_size) }}
            </span>
            <Tag
              v-if="primaryQuant(group) && primaryQuant(group).run_state === 'loading'"
              value="Loading"
              severity="warning"
              class="running-badge"
            />
            <Tag
              v-else-if="primaryQuant(group) && primaryQuant(group).is_active"
              value="Running"
              severity="success"
              class="running-badge"
            />
            <Tag
              v-if="primaryQuant(group)"
              :value="(primaryQuant(group).config && primaryQuant(group).config.engine) || (primaryQuant(group).format === 'safetensors' ? 'lmdeploy' : 'llama_cpp')"
              severity="secondary"
              class="engine-tag"
            />
            <Tag
              v-if="primaryQuant(group) && primaryQuant(group).format"
              :value="primaryQuant(group).format"
              severity="info"
            />
          </div>
          <div class="group-meta">
            <ModelStartStopButton
              v-if="primaryQuant(group)"
              :is-active="primaryQuant(group).is_active"
              :is-proxy-loading="primaryQuant(group).run_state === 'loading'"
              :is-starting="primaryQuant(group) && isQuantStarting(primaryQuant(group))"
              :is-stopping="primaryQuant(group) && stoppingModels.has(primaryQuant(group).id)"
              stop-propagation
              @start="primaryQuant(group) && startModel(primaryQuant(group).id)"
              @stop="primaryQuant(group) && stopModel(primaryQuant(group).id)"
            />
            <Button
              v-if="primaryQuant(group)"
              icon="pi pi-cog"
              text
              severity="secondary"
              size="small"
              v-tooltip.top="'Configure'"
              @click.stop="configureModel(primaryQuant(group).id)"
            />
            <Button
              icon="pi pi-trash"
              text
              severity="danger"
              size="small"
              v-tooltip.top="'Delete model'"
              @click.stop="primaryQuant(group) ? confirmDeleteModel(primaryQuant(group).id) : confirmDeleteGroup(group.huggingface_id)"
            />
          </div>
          <div
            v-if="primaryQuant(group) && primaryQuant(group).downloaded_at"
            class="safetensors-header__footer"
          >
            <span class="downloaded-at">
              Downloaded {{ formatDate(primaryQuant(group).downloaded_at) }}
            </span>
          </div>
        </div>

        <!-- Quantizations list (GGUF only) -->
        <Transition v-if="!isSafetensorsGroup(group)" name="group-collapse">
          <div v-if="expandedGroups.has(group.huggingface_id)" class="quantizations">
            <ModelRow
              v-for="quant in group.quantizations"
              :key="quant.id"
              :quant="quant"
              :is-starting="isQuantStarting(quant)"
              :is-stopping="stoppingModels.has(quant.id)"
              :format-bytes="formatBytes"
              :format-date="formatDate"
              @start="startModel"
              @stop="stopModel"
              @configure="configureModel"
              @delete="confirmDeleteModel"
            />
          </div>
        </Transition>
      </div>
    </div>

    <!-- HuggingFace Token Dialog -->
    <Dialog v-model:visible="showTokenDialog" header="HuggingFace Token" modal class="dialog-width-sm">
      <div class="token-form">
        <p class="token-desc">Required to access gated models (e.g. Llama, Gemma).</p>
        <div class="form-field">
          <label>Token</label>
          <Password v-model="tokenInput" placeholder="hf_…" :feedback="false" toggleMask class="w-full" />
        </div>
        <div v-if="modelStore.hasHuggingfaceToken" class="token-current">
          <i class="pi pi-check-circle token-current__icon" aria-hidden="true" />
          <span>Token set: {{ modelStore.huggingfaceToken || '••••••••' }}</span>
          <Button label="Clear" severity="danger" text size="small" @click="clearToken" />
        </div>
      </div>
      <template #footer>
        <Button label="Cancel" severity="secondary" outlined @click="showTokenDialog = false" />
        <Button label="Save Token" icon="pi pi-save" severity="success"
          :disabled="!tokenInput" :loading="savingToken" @click="saveToken" />
      </template>
    </Dialog>

  </div>
</template>

<script setup>
import { ref, computed, onMounted, onUnmounted } from 'vue'
import { useRouter } from 'vue-router'
import { useConfirm } from 'primevue/useconfirm'
import { useToast } from 'primevue/usetoast'
import Button from 'primevue/button'
import Tag from 'primevue/tag'
import Dialog from 'primevue/dialog'
import Password from 'primevue/password'
import ConfirmDialog from 'primevue/confirmdialog'
import ModelRow from '@/components/ModelRow.vue'
import ModelStartStopButton from '@/components/ModelStartStopButton.vue'
import { useModelStore } from '@/stores/models'
import { useProgressStore } from '@/stores/progress'
import ProgressTracker from '@/components/common/ProgressTracker.vue'
import PageHeader from '@/components/common/PageHeader.vue'
import LoadingState from '@/components/common/LoadingState.vue'
import EmptyState from '@/components/common/EmptyState.vue'

const router = useRouter()
const confirm = useConfirm()
const toast = useToast()
const modelStore = useModelStore()
const progressStore = useProgressStore()

// ── State ──────────────────────────────────────────────────
const expandedGroups = ref(new Set())
const startingModels = ref(new Set())
const stoppingModels = ref(new Set())
const showTokenDialog = ref(false)
const tokenInput = ref('')
const savingToken = ref(false)
let pollTimer = null
let unsubscribeDownloadComplete = null
let unsubscribeModelStatus = null
let unsubscribeModelEvent = null

async function refreshCatalogs() {
  await Promise.allSettled([modelStore.fetchModels(), modelStore.fetchSafetensorsModels()])
}

// ── Computed ───────────────────────────────────────────────
// Backend /api/models already returns both GGUF and safetensors models
// grouped appropriately, so we can display models directly from there.
const displayGroups = computed(() => modelStore.models || [])

const totalModels = computed(() =>
  displayGroups.value.reduce((acc, g) => acc + (g.quantizations?.length ?? 0), 0)
)

// ── Group expand/collapse ──────────────────────────────────
function isSafetensorsGroup(group) {
  if (!group || !Array.isArray(group.quantizations) || !group.quantizations.length) return false
  return group.quantizations.every(q => q.format === 'safetensors')
}

function primaryQuant(group) {
  if (!group || !Array.isArray(group.quantizations) || !group.quantizations.length) return null
  return group.quantizations[0]
}

/** Sum of file_size across all quantizations in a GGUF group (bytes). */
function groupTotalFileSize(group) {
  if (!group?.quantizations?.length) return 0
  return group.quantizations.reduce((sum, q) => sum + (Number(q.file_size) || 0), 0)
}

function toggleGroup(hfId) {
  if (expandedGroups.value.has(hfId)) {
    expandedGroups.value.delete(hfId)
  } else {
    expandedGroups.value.add(hfId)
  }
}

function expandAllGroups() {
  displayGroups.value.forEach(g => expandedGroups.value.add(g.huggingface_id))
}

/** Play-button loading until the catalog reflects an active or loading proxy slot. */
function isQuantStarting(quant) {
  if (!quant?.id) return false
  if (!startingModels.value.has(quant.id)) return false
  if (quant.is_active) return false
  return true
}

// ── Model actions ──────────────────────────────────────────
async function startModel(modelId) {
  startingModels.value.add(modelId)
  startingModels.value = new Set(startingModels.value)
  try {
    await modelStore.startModel(modelId)
    toast.add({ severity: 'success', summary: 'Model started', life: 3000 })
  } catch (e) {
    toast.add({ severity: 'error', summary: 'Failed to start', detail: e.message, life: 4000 })
  } finally {
    startingModels.value.delete(modelId)
    startingModels.value = new Set(startingModels.value) // trigger reactivity
  }
}

async function stopModel(modelId) {
  stoppingModels.value.add(modelId)
  stoppingModels.value = new Set(stoppingModels.value)
  try {
    await modelStore.stopModel(modelId)
    toast.add({ severity: 'info', summary: 'Model stopped', life: 3000 })
  } catch (e) {
    toast.add({ severity: 'error', summary: 'Failed to stop', detail: e.message, life: 4000 })
  } finally {
    stoppingModels.value.delete(modelId)
    stoppingModels.value = new Set(stoppingModels.value)
  }
}

function configureModel(modelId) {
  router.push(`/models/${encodeURIComponent(modelId)}/config`)
}

function confirmDeleteModel(modelId) {
  confirm.require({
    message:
      'Remove this model from the library? Downloaded files and manifests will be deleted from disk.',
    header: 'Confirm Remove',
    icon: 'pi pi-exclamation-triangle',
    acceptClass: 'p-button-danger',
    accept: async () => {
      try {
        await modelStore.deleteModel(modelId)
        toast.add({ severity: 'info', summary: 'Model removed', life: 3000 })
      } catch (e) {
        toast.add({ severity: 'error', summary: 'Failed', detail: e.message, life: 4000 })
      }
    },
  })
}

function confirmDeleteGroup(huggingfaceId) {
  confirm.require({
    message: `Remove all quantizations for "${huggingfaceId}"?`,
    header: 'Confirm Remove Group',
    icon: 'pi pi-exclamation-triangle',
    acceptClass: 'p-button-danger',
    accept: async () => {
      try {
        await modelStore.deleteModelGroup(huggingfaceId)
        toast.add({ severity: 'info', summary: 'Group removed', life: 3000 })
      } catch (e) {
        toast.add({ severity: 'error', summary: 'Failed', detail: e.message, life: 4000 })
      }
    },
  })
}

// ── Token management ───────────────────────────────────────
async function saveToken() {
  savingToken.value = true
  try {
    await modelStore.setHuggingfaceToken(tokenInput.value)
    tokenInput.value = ''
    showTokenDialog.value = false
    toast.add({ severity: 'success', summary: 'Token saved', life: 3000 })
  } catch (e) {
    toast.add({ severity: 'error', summary: 'Failed', detail: e.message, life: 4000 })
  } finally {
    savingToken.value = false
  }
}

async function clearToken() {
  try {
    await modelStore.clearHuggingfaceToken()
    toast.add({ severity: 'info', summary: 'Token cleared', life: 3000 })
  } catch (e) {
    toast.add({ severity: 'error', summary: 'Failed', detail: e.message, life: 4000 })
  }
}

// ── Formatters ─────────────────────────────────────────────
// Decimal (1000) so MB/GB match Hugging Face
function formatBytes(bytes) {
  if (!bytes) return ''
  const units = ['B', 'KB', 'MB', 'GB', 'TB']
  let i = 0; let val = bytes
  while (val >= 1000 && i < units.length - 1) { val /= 1000; i++ }
  return `${val.toFixed(1)} ${units[i]}`
}

function formatDate(iso) {
  if (!iso) return ''
  try {
    return new Intl.RelativeTimeFormat('en', { numeric: 'auto' }).format(
      Math.round((new Date(iso) - Date.now()) / 86400000), 'day'
    )
  } catch {
    return iso.slice(0, 10)
  }
}

// ── Lifecycle ──────────────────────────────────────────────
onMounted(async () => {
  await Promise.all([
    modelStore.fetchModels(),
    modelStore.fetchSafetensorsModels(),
    modelStore.fetchHuggingfaceTokenStatus(),
  ])
  expandAllGroups()
  unsubscribeDownloadComplete = progressStore.subscribeToDownloadComplete(() => {
    refreshCatalogs()
  })
  unsubscribeModelStatus = progressStore.subscribe('model_status', () => {
    refreshCatalogs()
  })
  unsubscribeModelEvent = progressStore.subscribe('model_event', () => {
    refreshCatalogs()
  })
  // Poll as a fallback when SSE misses updates (tighter while something is starting/stopping)
  pollTimer = setInterval(() => {
    refreshCatalogs()
  }, 5000)
})

onUnmounted(() => {
  if (pollTimer) clearInterval(pollTimer)
  if (unsubscribeDownloadComplete) unsubscribeDownloadComplete()
  if (unsubscribeModelStatus) unsubscribeModelStatus()
  if (unsubscribeModelEvent) unsubscribeModelEvent()
})
</script>

<style scoped>
/* layout: .page-shell */

/* ── Token warning ────────────────────────────────────── */
.token-warning {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.5rem 0.875rem;
  background: var(--status-warning-soft);
  border: 1px solid color-mix(in srgb, var(--status-warning) 35%, transparent);
  border-radius: var(--radius-md, 0.5rem);
  font-size: 0.875rem;
  color: var(--status-warning);
}

/* ── Groups ───────────────────────────────────────────── */
.model-groups {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}

.model-group {
  background: var(--bg-card, #161b2e);
  border: 1px solid var(--border-primary, #2a2f45);
  border-radius: var(--radius-lg, 0.75rem);
  overflow: hidden;
}

.group-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0.75rem 1rem;
  cursor: pointer;
  user-select: none;
  background: var(--bg-surface, #1e2235);
  transition: background 0.15s;
  gap: 0.5rem;
}

.group-header:hover { background: var(--bg-card-hover, #232a42); }

.group-header.safetensors-header {
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto;
  grid-template-rows: auto auto;
  gap: 0.1rem 0.75rem;
  align-items: start;
  justify-items: stretch;
  padding-block: 0.75rem;
  padding-inline: 1rem;
  box-sizing: border-box;
  cursor: default;
}

.safetensors-header:hover { background: var(--bg-surface, #1e2235); }

.safetensors-header > .group-title {
  grid-column: 1;
  grid-row: 1;
  min-width: 0;
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 0.4rem;
  row-gap: 0.25rem;
}

.safetensors-header > .group-meta {
  grid-column: 2;
  grid-row: 1 / -1;
  align-self: center;
  justify-self: end;
  display: flex;
  align-items: center;
  gap: 0.25rem;
  flex-shrink: 0;
}

.safetensors-header > .safetensors-header__footer {
  grid-column: 1 / -1;
  grid-row: 2;
  min-width: 0;
}

.safetensors-header .downloaded-at {
  display: block;
  font-size: 0.65rem;
  line-height: 1.15;
  color: var(--text-muted, var(--text-secondary, #9ca3af));
  opacity: 0.9;
}

.group-title {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  flex: 1;
  min-width: 0;
}

.group-chevron { font-size: 0.75rem; color: var(--text-secondary, #9ca3af); }

.group-name {
  font-weight: 600;
  font-size: 0.9rem;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.running-badge { flex-shrink: 0; }

.group-meta {
  display: flex;
  align-items: center;
  gap: 0.25rem;
  flex-shrink: 0;
}

.group-delete-preview {
  font-size: 0.75rem;
  line-height: 1.25;
  color: var(--text-secondary, #9ca3af);
  white-space: nowrap;
  user-select: none;
}

.group-meta small {
  font-size: 0.75rem;
  color: var(--text-secondary, #9ca3af);
  font-family: monospace;
  max-width: 200px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

/* ── Quantizations ────────────────────────────────────── */
.group-collapse-enter-active,
.group-collapse-leave-active { transition: all 0.2s ease; overflow: hidden; }
.group-collapse-enter-from,
.group-collapse-leave-to    { max-height: 0; opacity: 0; }
.group-collapse-enter-to,
.group-collapse-leave-from  { max-height: 1000px; opacity: 1; }

:deep(.quantizations) {
  padding: 0.5rem;
  display: flex;
  flex-direction: column;
  gap: 0.375rem;
}

:deep(.quant-row) {
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto;
  grid-template-rows: auto auto;
  gap: 0.1rem 0.75rem;
  align-items: start;
  justify-items: stretch;
  padding: 0.5rem 0.75rem;
  background: var(--bg-surface, #1e2235);
  border: 1px solid var(--border-primary, #2a2f45);
  border-radius: var(--radius-md, 0.5rem);
  transition: border-color 0.15s;
  box-sizing: border-box;
}

:deep(.quant-row > .quant-info) {
  grid-column: 1;
  grid-row: 1;
  min-width: 0;
}

:deep(.quant-row > .quant-actions) {
  display: flex;
  grid-column: 2;
  grid-row: 1 / -1;
  align-self: center;
  justify-self: end;
  align-items: center;
  gap: 0.25rem;
  flex-shrink: 0;
}

:deep(.quant-row > .quant-row__footer) {
  grid-column: 1 / -1;
  grid-row: 2;
  min-width: 0;
}

:deep(.quant-row__footer .downloaded-at) {
  display: block;
  font-size: 0.65rem;
  line-height: 1.15;
  color: var(--text-muted, var(--text-secondary, #9ca3af));
  opacity: 0.9;
}

:deep(.quant-row.is-active) {
  border-color: rgba(34, 197, 94, 0.4);
  background: rgba(34, 197, 94, 0.04);
}

:deep(.quant-info) {
  display: flex;
  align-items: center;
  min-width: 0;
}

:deep(.quant-main) {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 0.4rem;
  row-gap: 0.25rem;
}

:deep(.quant-name) {
  font-weight: 600;
  font-size: 0.875rem;
  font-family: monospace;
}

/* Group title + quant row: compact size label (GGUF total, safetensors single, per-quant) */
.group-header .group-title .file-size,
:deep(.quant-main .file-size) {
  font-size: 0.75rem;
  line-height: 1.25;
  color: var(--text-secondary, #9ca3af);
  font-variant-numeric: tabular-nums;
}

/* Emphasize engine tag with a distinct background */
.engine-tag {
  background-color: rgba(59, 130, 246, 0.15); /* soft blue */
  border-color: rgba(59, 130, 246, 0.65);
  color: #bfdbfe;
}

/* ── Token dialog ─────────────────────────────────────── */
.token-form { display: flex; flex-direction: column; gap: 0.75rem; }
.token-desc { font-size: 0.875rem; color: var(--text-secondary, #9ca3af); margin: 0; }
.form-field { display: flex; flex-direction: column; gap: 0.25rem; }
.form-field label { font-size: 0.875rem; font-weight: 500; }

.token-current {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  font-size: 0.875rem;
  background: var(--status-success-soft);
  border: 1px solid color-mix(in srgb, var(--status-success) 35%, transparent);
  border-radius: var(--radius-md, 0.5rem);
  padding: 0.5rem 0.75rem;
}

.token-current__icon {
  color: var(--status-success);
}
</style>
