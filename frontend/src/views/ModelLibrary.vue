<template>
  <div class="model-library page-shell page-shell--wide">

    <PageHeader title="Model library">
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
          aria-label="Refresh models"
          :loading="modelStore.loading"
          v-tooltip.top="'Refresh'"
          @click="retryCatalogs"
        />
        <Button
          label="Discover"
          icon="pi pi-search"
          severity="success"
          class="page-header__cta"
          @click="$router.push('/search')"
        />
      </template>
    </PageHeader>

    <!-- Token Warning -->
    <div v-if="!modelStore.hasHuggingfaceToken" class="token-warning">
      <i class="pi pi-key" aria-hidden="true" />
      <span class="token-warning__text">No HuggingFace token set. Gated models won't be accessible.</span>
      <Button
        label="Set Token"
        icon="pi pi-pencil"
        size="small"
        text
        class="token-warning__action"
        @click="showTokenDialog = true"
      />
    </div>

    <SetupChecklist />

    <div v-if="modelStore.modelsStale" class="state-banner" role="status">
      <span>Showing the last loaded library. The latest refresh failed.</span>
      <Button label="Retry" size="small" @click="retryCatalogs" />
    </div>

    <LoadingState
      v-if="catalogLoading && !storeHasModels"
      message="Loading models…"
    />

    <EmptyState
      v-else-if="catalogFailed && !storeHasModels"
      icon="pi pi-exclamation-circle"
      title="Could not load models"
      :description="modelStore.modelsError || 'The model library request failed.'"
    >
      <Button label="Retry" icon="pi pi-refresh" @click="retryCatalogs" />
    </EmptyState>

    <EmptyState
      v-else-if="!catalogLoading && !storeHasModels"
      icon="pi pi-inbox"
      title="No models downloaded yet"
      description="Install a compatible engine, then search for a model."
    >
      <Button label="Discover models" icon="pi pi-search" @click="$router.push('/search')" />
      <Button label="Engines" icon="pi pi-cog" severity="secondary" outlined @click="$router.push('/engines')" />
    </EmptyState>

    <div v-else class="library-workspace">
    <div class="library-toolbar">
      <label class="sr-only" for="library-search">Search library</label>
      <input id="library-search" v-model="libraryQuery" type="search" class="library-search" placeholder="Search name or repository" />
      <label for="library-status">Status</label>
      <select id="library-status" v-model="statusFilter">
        <option value="all">All</option>
        <option value="running">Running</option>
        <option value="stopped">Stopped</option>
        <option value="attention">Needs attention</option>
      </select>
      <label for="library-engine">Engine</label>
      <select id="library-engine" v-model="engineFilter">
        <option value="all">Any engine</option>
        <option v-for="engine in engineChoices" :key="engine" :value="engine">{{ engine }}</option>
      </select>
      <label for="library-task">Task</label>
      <select id="library-task" v-model="taskFilter">
        <option value="all">Any task</option>
        <option v-for="task in taskChoices" :key="task" :value="task">{{ task }}</option>
      </select>
      <label for="library-sort">Sort</label>
      <select id="library-sort" v-model="sortBy">
        <option value="name">Name</option>
        <option value="size">Size</option>
        <option value="recent">Recent use</option>
      </select>
      <div class="library-view" role="group" aria-label="Library layout">
        <button type="button" :aria-pressed="viewMode === 'cards'" @click="setViewMode('cards')">Cards</button>
        <button type="button" :aria-pressed="viewMode === 'list'" @click="setViewMode('list')">List</button>
      </div>
    </div>

    <EmptyState
      v-if="!displayGroups.length"
      icon="pi pi-filter"
      title="No models match"
      description="Try a different name, status, engine, or task."
    />

    <!-- Model groups (GGUF + Safetensors) -->
    <div v-else-if="viewMode === 'list'" class="library-table" role="table" aria-label="Model library">
      <div class="library-table__row library-table__row--head" role="row">
        <span role="columnheader">Model</span>
        <span role="columnheader">Variant</span>
        <span role="columnheader">Engine</span>
        <span role="columnheader">Size</span>
        <span role="columnheader">Status</span>
        <span role="columnheader">Actions</span>
      </div>
      <div v-for="row in libraryRows" :key="row.id" class="library-table__row" role="row">
        <span role="cell">{{ row.model }}</span>
        <span role="cell"><code>{{ row.variant }}</code></span>
        <span role="cell">{{ row.engine }}</span>
        <span role="cell">{{ row.size }}</span>
        <span role="cell">{{ row.status }}</span>
        <span role="cell" class="library-table__actions">
          <ModelStartStopButton
            :name="row.model"
            show-label
            :is-active="row.quant.is_active"
            :is-proxy-loading="quantStatus(row.quant) === 'loading'"
            :is-starting="isQuantStarting(row.quant)"
            :is-stopping="isQuantStopping(row.quant)"
            @start="startModel(row.quant.id)"
            @stop="stopModel(row.quant.id)"
          />
          <Button
            v-if="row.quant.is_active && !isAudioQuant(row.quant)"
            label="Connect"
            icon="pi pi-link"
            size="small"
            :aria-label="`Connect ${row.model}`"
            @click="openConnect(row.quant)"
          />
          <Button
            v-if="isAudioQuant(row.quant)"
            label="Audio"
            icon="pi pi-volume-up"
            size="small"
            text
            :aria-label="`Open audio for ${row.model}`"
            @click="openAudio(row.quant.id)"
          />
          <Button
            label="Configure"
            icon="pi pi-cog"
            size="small"
            text
            :aria-label="`Configure ${row.model}`"
            @click="configureModel(row.quant.id)"
          />
        </span>
      </div>
    </div>
    <div v-else class="model-groups">
      <div
        v-for="group in displayGroups"
        :key="group.huggingface_id"
        class="model-group"
        :class="{ 'is-running': groupIsRunning(group) }"
      >
        <!-- Group header: multi-variant model (expandable) -->
        <div
          v-if="!isStandaloneGroup(group)"
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
            <div class="group-heading">
              <i
                :class="['pi', 'group-chevron', expandedGroups.has(group.huggingface_id) ? 'pi-chevron-down' : 'pi-chevron-right']"
              />
              <span class="group-name">{{ groupTitle(group) }}</span>
              <span v-if="group.huggingface_id && groupTitle(group) !== group.huggingface_id" class="group-repo">{{ group.huggingface_id }}</span>
              <span
                v-if="groupTotalFileSize(group) > 0"
                class="file-size file-size--total"
                title="Total size (all quantizations)"
              >
                {{ formatBytes(groupTotalFileSize(group)) }}
              </span>
              <Tag
                v-if="group.quantizations?.some(q => quantStatus(q) === 'loading')"
                value="Loading"
                severity="warning"
                class="running-badge"
              />
              <Tag
                v-else-if="group.quantizations?.some(q => quantStatus(q) === 'ready')"
                value="Ready"
                severity="success"
                class="running-badge"
              />
              <Tag
                v-else-if="group.quantizations?.some(q => q.is_active)"
                value="Running"
                severity="success"
                class="running-badge"
              />
            </div>
            <div class="group-tags">
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
              <Tag v-if="group.family" :value="group.family" severity="secondary" />
              <Tag
                v-for="task in group.tasks || []"
                :key="`group-task-${task}`"
                :value="task"
                severity="success"
              />
            </div>
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
            <details class="row-menu" @click.stop>
              <summary :aria-label="`More actions for ${groupTitle(group)}`">More</summary>
              <button type="button" @click="confirmDeleteGroup(group.huggingface_id)">Delete group</button>
            </details>
          </div>
        </div>

        <!-- Standalone snapshot or prepared bundle (single-row, non-expandable) -->
        <div
          v-else
          class="group-header safetensors-header"
        >
          <div class="group-title">
            <div class="group-heading">
              <span class="group-name">{{ groupTitle(group) }}</span>
              <span v-if="group.huggingface_id && groupTitle(group) !== group.huggingface_id" class="group-repo">{{ group.huggingface_id }}</span>
              <span
                v-if="primaryQuant(group) && primaryQuant(group).file_size"
                class="file-size"
              >
                {{ formatBytes(primaryQuant(group).file_size) }}
              </span>
              <Tag
                v-if="primaryQuant(group) && quantStatus(primaryQuant(group)) === 'loading'"
                value="Loading"
                severity="warning"
                class="running-badge"
              />
              <Tag
                v-else-if="primaryQuant(group) && quantStatus(primaryQuant(group)) === 'ready'"
                value="Ready"
                severity="success"
                class="running-badge"
              />
              <Tag
                v-else-if="primaryQuant(group) && primaryQuant(group).is_active"
                value="Running"
                severity="success"
                class="running-badge"
              />
            </div>
            <div class="group-tags">
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
              <Tag v-if="group.family" :value="group.family" severity="secondary" />
              <Tag
                v-for="task in group.tasks || []"
                :key="`standalone-task-${task}`"
                :value="task"
                severity="success"
              />
              <Tag
                v-for="modality in group.output_modalities || []"
                :key="`standalone-output-${modality}`"
                :value="`${modality} out`"
                severity="info"
              />
            </div>
          </div>
          <div class="group-meta">
            <ModelStartStopButton
              v-if="primaryQuant(group)"
              :name="groupTitle(group)"
              show-label
              :is-active="primaryQuant(group).is_active"
              :is-proxy-loading="quantStatus(primaryQuant(group)) === 'loading'"
              :is-starting="primaryQuant(group) && isQuantStarting(primaryQuant(group))"
              :is-stopping="primaryQuant(group) && isQuantStopping(primaryQuant(group))"
              stop-propagation
              @start="primaryQuant(group) && startModel(primaryQuant(group).id)"
              @stop="primaryQuant(group) && stopModel(primaryQuant(group).id)"
            />
            <Button
              v-if="primaryQuant(group) && isAudioQuant(primaryQuant(group))"
              icon="pi pi-volume-up"
              text
              severity="secondary"
              size="small"
              :aria-label="`Open audio for ${groupTitle(group)}`"
              v-tooltip.top="'Audio'"
              @click.stop="openAudio(primaryQuant(group).id)"
            />
            <Button
              v-if="primaryQuant(group) && primaryQuant(group).is_active && !isAudioQuant(primaryQuant(group))"
              label="Connect"
              icon="pi pi-link"
              size="small"
              :aria-label="`Connect ${groupTitle(group)}`"
              @click.stop="openConnect(primaryQuant(group))"
            />
            <Button
              v-if="primaryQuant(group)"
              label="Configure"
              icon="pi pi-cog"
              text
              severity="secondary"
              size="small"
              :aria-label="`Configure ${groupTitle(group)}`"
              @click.stop="configureModel(primaryQuant(group).id)"
            />
            <details class="row-menu" @click.stop>
              <summary :aria-label="`More actions for ${groupTitle(group)}`">More</summary>
              <button type="button" @click="primaryQuant(group) ? confirmDeleteModel(primaryQuant(group).id) : confirmDeleteGroup(group.huggingface_id)">Delete</button>
            </details>
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

        <!-- Variant rows for grouped models -->
        <Transition v-if="!isStandaloneGroup(group)" name="group-collapse">
          <div v-if="expandedGroups.has(group.huggingface_id)" class="quantizations">
            <ModelRow
              v-for="quant in group.quantizations"
              :key="quant.id"
              :quant="quant"
              :is-starting="isQuantStarting(quant)"
              :is-stopping="isQuantStopping(quant)"
              :format-bytes="formatBytes"
              :format-date="formatDate"
              @start="startModel"
              @stop="stopModel"
              @configure="configureModel"
              @audio="openAudio"
              @connect="openConnect"
              @delete="confirmDeleteModel"
            />
          </div>
        </Transition>
      </div>
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

    <ConnectDialog v-model:visible="connectVisible" :model="connectModel" :proxy-port="proxyPort" />

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
import ModelRow from '@/components/ModelRow.vue'
import ModelStartStopButton from '@/components/ModelStartStopButton.vue'
import { useModelStore } from '@/stores/models'
import { useProgressStore } from '@/stores/progress'
import { audioTabFromConfig } from '@/composables/useAudioInferenceClient'
import { requireSingleConfirmation } from '@/composables/singleConfirm'
import PageHeader from '@/components/common/PageHeader.vue'
import LoadingState from '@/components/common/LoadingState.vue'
import EmptyState from '@/components/common/EmptyState.vue'
import SetupChecklist from '@/components/common/SetupChecklist.vue'
import ConnectDialog from '@/components/common/ConnectDialog.vue'
import { useEnginesStore } from '@/stores/engines'

const router = useRouter()
const confirm = useConfirm()
const toast = useToast()
const modelStore = useModelStore()
const progressStore = useProgressStore()
const enginesStore = useEnginesStore()
const connectVisible = ref(false)
const connectModel = ref(null)
const proxyPort = computed(() => enginesStore.systemStatus?.proxy_status?.port || 2000)

// ── State ──────────────────────────────────────────────────
const expandedGroups = ref(new Set())
const libraryQuery = ref('')
const statusFilter = ref('all')
const engineFilter = ref('all')
const taskFilter = ref('all')
const sortBy = ref('name')
const viewMode = ref(typeof localStorage !== 'undefined' && localStorage.getItem('llama-studio.library.view') === 'list' ? 'list' : 'cards')
const EXPANDED_KEY = 'llama-studio.library.expanded'
const startingModels = ref(new Set())
const stoppingModels = ref(new Set())
const showTokenDialog = ref(false)
const tokenInput = ref('')
const savingToken = ref(false)
const catalogRefreshInFlight = ref(false)
const FAST_POLL_MS = 1000
const IDLE_POLL_MS = 5000
let pollTimer = null
let unsubscribeDownloadComplete = null
let unsubscribeModelStatus = null
let unsubscribeModelEvent = null

async function retryCatalogs() {
  try {
    await refreshCatalogs()
  } catch {
    /* The store keeps the error for the empty and stale states. */
  }
}

async function refreshCatalogs() {
  if (catalogRefreshInFlight.value) return
  catalogRefreshInFlight.value = true
  try {
    await Promise.allSettled([modelStore.fetchModels(), modelStore.fetchSafetensorsModels()])
  } finally {
    catalogRefreshInFlight.value = false
  }
}

// ── Computed ───────────────────────────────────────────────
// Backend /api/models already returns both GGUF and safetensors models
// grouped appropriately, so we can display models directly from there.
const sourceGroups = computed(() => modelStore.models || [])

const storeHasModels = computed(() => sourceGroups.value.some((group) => (group.quantizations || []).length > 0))

const catalogLoading = computed(() =>
  (modelStore.loading || modelStore.safetensorsLoading) && !storeHasModels.value,
)

const catalogFailed = computed(() => Boolean(modelStore.modelsError) && !modelStore.loading)

function groupEngine(group) {
  const quant = primaryQuant(group)
  return (quant?.config && quant.config.engine) || (quant?.format === 'safetensors' ? 'lmdeploy' : quant ? 'llama_cpp' : '')
}

function groupNeedsAttention(group) {
  return (group?.quantizations || []).some((quant) => {
    const status = quantStatus(quant)
    return status === 'failed' || status === 'error' || quant?.last_error
  })
}

function groupRecentStamp(group) {
  const stamps = (group?.quantizations || []).map((quant) => Date.parse(quant?.last_used_at || quant?.downloaded_at || '') || 0)
  return Math.max(0, ...stamps)
}

const engineChoices = computed(() => {
  const values = new Set(sourceGroups.value.map(groupEngine).filter(Boolean))
  return [...values].sort()
})

const taskChoices = computed(() => {
  const values = new Set()
  sourceGroups.value.forEach((group) => (group.tasks || []).forEach((task) => values.add(task)))
  return [...values].sort()
})

const displayGroups = computed(() => {
  const query = libraryQuery.value.trim().toLowerCase()
  let groups = sourceGroups.value.filter((group) => {
    if (query) {
      const hay = [
        group.huggingface_id,
        group.base_model_name,
        ...(group.quantizations || []).map((quant) => quant.quantization || quant.name || ''),
      ].join(' ').toLowerCase()
      if (!hay.includes(query)) return false
    }
    if (statusFilter.value === 'running' && !groupIsRunning(group)) return false
    if (statusFilter.value === 'stopped' && groupIsRunning(group)) return false
    if (statusFilter.value === 'attention' && !groupNeedsAttention(group)) return false
    if (engineFilter.value !== 'all' && groupEngine(group) !== engineFilter.value) return false
    if (taskFilter.value !== 'all' && !(group.tasks || []).includes(taskFilter.value)) return false
    return true
  })
  groups = [...groups]
  if (sortBy.value === 'size') {
    groups.sort((a, b) => groupTotalFileSize(b) - groupTotalFileSize(a))
  } else if (sortBy.value === 'recent') {
    groups.sort((a, b) => groupRecentStamp(b) - groupRecentStamp(a))
  } else {
    groups.sort((a, b) => groupTitle(a).localeCompare(groupTitle(b)))
  }
  return groups
})

const libraryRows = computed(() => {
  const rows = []
  for (const group of displayGroups.value) {
    for (const quant of group.quantizations || []) {
      rows.push({
        id: quant.id,
        model: groupTitle(group),
        variant: quant.quantization || quant.name || quant.format || '—',
        engine: quant.config?.engine || quant.engine || '—',
        size: quant.file_size ? formatBytes(quant.file_size) : '—',
        status: libraryStatus(quant),
        quant,
      })
    }
  }
  return rows
})

function groupTitle(group) {
  return group?.base_model_name || group?.huggingface_id || 'Model'
}

function setViewMode(mode) {
  viewMode.value = mode
  try {
    localStorage.setItem('llama-studio.library.view', mode)
  } catch {
    /* ignore */
  }
}

const totalModels = computed(() =>
  sourceGroups.value.reduce((acc, g) => acc + (g.quantizations?.length ?? 0), 0)
)

// ── Group expand/collapse ──────────────────────────────────
function isStandaloneGroup(group) {
  if (!group || !Array.isArray(group.quantizations) || !group.quantizations.length) return false
  return group.quantizations.every((q) => (
    q.format === 'safetensors'
    || ['prepared_bundle', 'builtin'].includes(q.artifact?.package_kind)
  ))
}

function primaryQuant(group) {
  if (!group || !Array.isArray(group.quantizations) || !group.quantizations.length) return null
  return group.quantizations[0]
}

function groupIsRunning(group) {
  return (group?.quantizations || []).some((quant) => {
    if (quant?.is_active) return true
    const status = quantStatus(quant)
    return status === 'ready' || status === 'loading'
  })
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
  expandedGroups.value = new Set(expandedGroups.value)
  try {
    sessionStorage.setItem(EXPANDED_KEY, JSON.stringify([...expandedGroups.value]))
  } catch {
    /* ignore */
  }
}

function modelIdKey(id) {
  return String(id)
}

function quantStatus(quant) {
  return String(quant?.status || quant?.run_state || '').toLowerCase()
}

function libraryStatus(quant) {
  const status = quantStatus(quant)
  if (status === 'loading') return 'Loading'
  if (status === 'ready') return 'Ready'
  if (quant?.is_active) return 'Running'
  return 'Stopped'
}

function openConnect(quant) {
  connectModel.value = quant
  connectVisible.value = true
}

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms))
}

function formatAxiosDetail(e) {
  const payload = e?.response?.data
  const d = payload?.detail ?? payload
  if (typeof d === 'string') return d
  if (Array.isArray(d)) {
    return d
      .map((x) =>
        typeof x === 'object' && x?.msg
          ? `${Array.isArray(x.loc) ? x.loc.join('.') : ''}: ${x.msg}`.replace(/^\.\s*/, '')
          : String(x)
      )
      .filter(Boolean)
      .join('; ')
  }
  if (d && typeof d === 'object') {
    if (typeof d.msg === 'string') return d.msg
    if (typeof d.message === 'string') return d.message
    if (typeof d.error === 'string') return d.error
  }
  return e?.message || 'Request failed'
}

function findQuantById(modelId) {
  const k = modelIdKey(modelId)
  for (const g of modelStore.models || []) {
    for (const q of g.quantizations || []) {
      if (modelIdKey(q.id) === k) return q
    }
  }
  return null
}

/** Play busy while start is in flight or we are waiting for the catalog to show running/loading. */
function isQuantStarting(quant) {
  if (!quant?.id) return false
  return startingModels.value.has(modelIdKey(quant.id))
}

function isQuantStopping(quant) {
  if (!quant?.id) return false
  return stoppingModels.value.has(modelIdKey(quant.id))
}

const hasLiveModelTransitions = computed(() => {
  if (startingModels.value.size || stoppingModels.value.size) return true
  return displayGroups.value.some(group =>
    (group.quantizations || []).some(quant => quantStatus(quant) === 'loading')
  )
})

function queueCatalogPoll(delay = hasLiveModelTransitions.value ? FAST_POLL_MS : IDLE_POLL_MS) {
  if (pollTimer) clearTimeout(pollTimer)
  pollTimer = setTimeout(async () => {
    await refreshCatalogs()
    queueCatalogPoll()
  }, delay)
}

// ── Model actions ──────────────────────────────────────────
async function startModel(modelId) {
  const k = modelIdKey(modelId)
  startingModels.value.add(k)
  startingModels.value = new Set(startingModels.value)
  queueCatalogPoll(FAST_POLL_MS)
  let ok = false
  try {
    await modelStore.startModel(modelId)
    toast.add({ severity: 'success', summary: 'Model started', life: 3000 })
    ok = true
  } catch (e) {
    toast.add({ severity: 'error', summary: 'Failed to start', detail: formatAxiosDetail(e), life: 4000 })
  } finally {
    if (ok) {
      const deadline = Date.now() + 30000
      while (Date.now() < deadline) {
        const q = findQuantById(modelId)
        if (q?.is_active || quantStatus(q) === 'loading') break
        await sleep(300)
        await refreshCatalogs()
      }
    }
    startingModels.value.delete(k)
    startingModels.value = new Set(startingModels.value)
    queueCatalogPoll()
  }
}

async function stopModel(modelId) {
  const k = modelIdKey(modelId)
  stoppingModels.value.add(k)
  stoppingModels.value = new Set(stoppingModels.value)
  queueCatalogPoll(FAST_POLL_MS)
  let ok = false
  try {
    await modelStore.stopModel(modelId)
    toast.add({ severity: 'info', summary: 'Model stopped', life: 3000 })
    ok = true
  } catch (e) {
    toast.add({ severity: 'error', summary: 'Failed to stop', detail: formatAxiosDetail(e), life: 4000 })
  } finally {
    if (ok) {
      const deadline = Date.now() + 30000
      while (Date.now() < deadline) {
        const q = findQuantById(modelId)
        if (!q || (!q.is_active && quantStatus(q) !== 'loading')) break
        await sleep(300)
        await refreshCatalogs()
      }
    }
    stoppingModels.value.delete(k)
    stoppingModels.value = new Set(stoppingModels.value)
    queueCatalogPoll()
  }
}

function configureModel(modelId) {
  router.push(`/models/${encodeURIComponent(modelId)}/config`)
}

function isAudioQuant(quant) {
  if (!quant) return false
  const engine = quant.config?.engine || quant.engine
  return engine === 'audio_cpp' || quant.format === 'audio_cpp'
}

function openAudio(modelId) {
  const quant = modelStore.allQuantizations.find((m) => m.id === modelId)
  const tab = audioTabFromConfig({
    task: quant?.config?.task || quant?.task || quant?.tasks?.[0],
    family: quant?.config?.family || quant?.family,
  })
  router.push({ name: 'audio', query: { model: modelId, tab } })
}

function confirmDeleteModel(modelId) {
  requireSingleConfirmation(confirm, {
    message:
      'Remove this model from the library? Downloaded files will be deleted from disk.',
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
  requireSingleConfirmation(confirm, {
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
onMounted(() => {
  try {
    const saved = JSON.parse(sessionStorage.getItem(EXPANDED_KEY) || '[]')
    if (Array.isArray(saved)) expandedGroups.value = new Set(saved)
  } catch {
    /* ignore */
  }
  unsubscribeDownloadComplete = progressStore.subscribeToDownloadComplete(() => {
    refreshCatalogs()
  })
  unsubscribeModelStatus = progressStore.subscribe('model_status', () => {
    refreshCatalogs()
  })
  unsubscribeModelEvent = progressStore.subscribe('model_event', () => {
    refreshCatalogs()
  })
  queueCatalogPoll()
  Promise.allSettled([
    modelStore.fetchModels(),
    modelStore.fetchSafetensorsModels(),
    modelStore.fetchHuggingfaceTokenStatus?.(),
  ])
})

onUnmounted(() => {
  if (pollTimer) clearTimeout(pollTimer)
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
  flex-wrap: wrap;
  align-items: center;
  gap: 0.5rem;
  padding: 0.5rem 0.875rem;
  background: var(--status-warning-soft);
  border: 1px solid color-mix(in srgb, var(--status-warning) 35%, transparent);
  border-radius: var(--radius-md, 0.5rem);
  font-size: 0.875rem;
  color: var(--status-warning);
}

.token-warning__text {
  flex: 1 1 12rem;
  min-width: 0;
}

.token-warning__action {
  margin-left: auto;
}

/* ── Groups ───────────────────────────────────────────── */
.model-groups {
  display: grid;
  grid-template-columns: 1fr;
  gap: 0.5rem;
  align-items: start;
}

@media (min-width: 769px) {
  .model-groups {
    grid-template-columns: repeat(auto-fill, minmax(min(100%, 20rem), 1fr));
    gap: 0.75rem;
  }
}

@supports (grid-template-rows: masonry) {
  @media (min-width: 769px) {
    .model-groups {
      grid-template-rows: masonry;
    }
  }
}

@supports not (grid-template-rows: masonry) {
  @media (min-width: 769px) {
    .model-groups {
      display: block;
      column-width: 20rem;
      column-gap: 0.75rem;
    }

    .model-group {
      break-inside: avoid;
      margin-bottom: 0.75rem;
      width: 100%;
    }
  }
}

.model-group {
  background: var(--bg-card, #161b2e);
  border: 1px solid var(--border-primary, #2a2f45);
  border-radius: var(--radius-lg, 0.75rem);
  overflow: hidden;
  container-type: inline-size;
  container-name: model-card;
  transition: border-color 0.2s ease, box-shadow 0.2s ease;
}

.model-group.is-running {
  border-color: color-mix(in srgb, var(--accent-green) 55%, var(--border-primary));
  box-shadow:
    var(--glow-success),
    0 0 28px color-mix(in srgb, var(--accent-green) 22%, transparent);
}

.group-header {
  display: flex;
  flex-wrap: wrap;
  justify-content: space-between;
  align-items: flex-start;
  padding: 0.75rem 1rem;
  cursor: pointer;
  user-select: none;
  background: var(--bg-surface, #1e2235);
  transition: background 0.15s;
  gap: 0.35rem 0.5rem;
}

.group-header:hover { background: var(--bg-card-hover, #232a42); }

.group-header .group-title {
  flex: 1 1 100%;
}

.group-header .group-meta {
  width: 100%;
  justify-content: flex-end;
}

.group-header.safetensors-header {
  display: flex;
  flex-direction: column;
  align-items: stretch;
  gap: 0.35rem;
  padding-block: 0.75rem;
  padding-inline: 1rem;
  box-sizing: border-box;
  cursor: default;
}

.safetensors-header:hover { background: var(--bg-surface, #1e2235); }

.safetensors-header > .group-title {
  min-width: 0;
}

.safetensors-header > .group-meta {
  width: 100%;
  justify-content: flex-end;
}

.safetensors-header > .safetensors-header__footer {
  min-width: 0;
}

@container model-card (min-width: 28rem) {
  .group-header:not(.safetensors-header) .group-title {
    flex: 1 1 0;
    min-width: 0;
  }

  .group-header:not(.safetensors-header) .group-meta {
    width: auto;
    flex-shrink: 0;
  }

  .group-header.safetensors-header {
    display: grid;
    grid-template-columns: minmax(0, 1fr) auto;
    grid-template-rows: auto auto;
    gap: 0.1rem 0.75rem;
    align-items: start;
  }

  .safetensors-header > .group-title {
    grid-column: 1;
    grid-row: 1;
  }

  .safetensors-header > .group-meta {
    grid-column: 2;
    grid-row: 1 / -1;
    width: auto;
    align-self: center;
    justify-self: end;
  }

  .safetensors-header > .safetensors-header__footer {
    grid-column: 1 / -1;
    grid-row: 2;
  }

  :deep(.quant-row) {
    grid-template-columns: minmax(0, 1fr) auto;
    grid-template-rows: auto auto;
    gap: 0.1rem 0.75rem;
  }

  :deep(.quant-row > .quant-info) {
    grid-column: 1;
    grid-row: 1;
  }

  :deep(.quant-row > .quant-actions) {
    grid-column: 2;
    grid-row: 1 / -1;
    align-self: center;
    justify-self: end;
  }

  :deep(.quant-row > .quant-row__footer) {
    grid-column: 1 / -1;
    grid-row: 2;
  }
}

.group-title {
  display: flex;
  flex-direction: column;
  align-items: flex-start;
  gap: 0.3rem;
  flex: 1;
  min-width: 0;
}

.group-heading {
  display: flex;
  align-items: center;
  gap: 0.35rem;
  min-width: 0;
  width: 100%;
}

.group-tags {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 0.3rem;
  width: 100%;
}

.group-tags:empty {
  display: none;
}

.safetensors-header .downloaded-at {
  display: block;
  font-size: 0.65rem;
  line-height: 1.15;
  color: var(--text-muted, var(--text-secondary, #9ca3af));
  opacity: 0.9;
}

.group-chevron { font-size: 0.75rem; color: var(--text-secondary, #9ca3af); }

.group-name {
  font-weight: 600;
  font-size: 0.9rem;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  min-width: 0;
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
  grid-template-columns: minmax(0, 1fr);
  gap: 0.25rem 0.75rem;
  align-items: start;
  justify-items: stretch;
  padding: 0.5rem 0.75rem;
  background: var(--bg-surface, #1e2235);
  border: 1px solid var(--border-primary, #2a2f45);
  border-radius: var(--radius-md, 0.5rem);
  transition: border-color 0.15s ease, box-shadow 0.2s ease, background 0.2s ease;
  box-sizing: border-box;
}

:deep(.quant-row > .quant-info) {
  min-width: 0;
}

:deep(.quant-row > .quant-actions) {
  display: flex;
  align-items: center;
  justify-content: flex-end;
  gap: 0.25rem;
}

:deep(.quant-row > .quant-row__footer) {
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
  border-color: color-mix(in srgb, var(--accent-green) 70%, var(--border-primary));
  background: color-mix(in srgb, var(--accent-green) 14%, var(--bg-surface));
  box-shadow:
    var(--glow-success),
    0 0 22px color-mix(in srgb, var(--accent-green) 30%, transparent);
}

:deep(.quant-info) {
  display: flex;
  align-items: center;
  min-width: 0;
}

:deep(.quant-main) {
  display: flex;
  flex-direction: column;
  align-items: flex-start;
  gap: 0.25rem;
  min-width: 0;
  width: 100%;
}

:deep(.quant-heading),
:deep(.quant-tags) {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 0.3rem;
  width: 100%;
}

:deep(.quant-tags:empty) {
  display: none;
}

:deep(.quant-name) {
  font-weight: 600;
  font-size: 0.875rem;
  font-family: monospace;
}

/* Group title + quant row: compact size label (GGUF total, safetensors single, per-quant) */
.group-header .group-heading .file-size,
:deep(.quant-heading .file-size) {
  font-size: 0.75rem;
  line-height: 1.25;
  color: var(--text-secondary, #9ca3af);
  font-variant-numeric: tabular-nums;
}

/* Emphasize engine tag with a distinct background */
.engine-tag {
  background-color: var(--tag-engine-bg);
  border-color: var(--tag-engine-border);
  color: var(--tag-engine-fg);
}

.library-toolbar {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 0.45rem 0.65rem;
  margin-bottom: 0.85rem;
}

.library-toolbar label {
  font-size: 0.8rem;
  color: var(--text-secondary);
}

.library-search,
.library-toolbar select {
  min-height: 2.25rem;
  border: 1px solid var(--border-primary);
  border-radius: var(--radius-md);
  background: var(--bg-secondary);
  color: var(--text-primary);
  padding: 0.3rem 0.55rem;
}

.library-search {
  flex: 1 1 12rem;
}

.library-view {
  display: inline-flex;
  gap: 0.25rem;
}

.library-view button,
.row-menu summary,
.row-menu button {
  min-height: 2rem;
  border: 1px solid var(--border-primary);
  border-radius: var(--radius-md);
  background: var(--bg-secondary);
  color: var(--text-primary);
  cursor: pointer;
  font: inherit;
  font-size: 0.8rem;
  padding: 0.2rem 0.55rem;
}

.library-view button[aria-pressed="true"] {
  border-color: var(--accent-cyan);
  font-weight: 700;
}

.row-menu {
  position: relative;
}

.row-menu summary {
  list-style: none;
}

.row-menu summary::-webkit-details-marker {
  display: none;
}

.row-menu[open] {
  position: relative;
}

.row-menu button {
  display: block;
  width: 100%;
  margin-top: 0.25rem;
  color: var(--status-error);
}

.group-repo {
  color: var(--text-secondary);
  font-size: 0.75rem;
}

.library-table {
  border: 1px solid var(--border-primary);
  border-radius: var(--radius-md);
  overflow: auto;
}

.library-table__row {
  display: grid;
  grid-template-columns: minmax(8rem, 1.4fr) minmax(6rem, 1fr) minmax(5rem, 0.7fr) minmax(4rem, 0.5fr) minmax(4.5rem, 0.5fr) auto;
  gap: 0.5rem;
  align-items: center;
  padding: 0.55rem 0.75rem;
  border-bottom: 1px solid var(--border-primary);
}

.library-table__row:last-child {
  border-bottom: 0;
}

.library-table__row--head {
  font-size: 0.75rem;
  font-weight: 600;
  color: var(--text-secondary);
  background: var(--bg-tertiary);
}

.library-table__actions {
  display: flex;
  flex-wrap: wrap;
  gap: 0.35rem;
  justify-content: flex-end;
}

@media (max-width: 720px) {
  .library-table__row {
    grid-template-columns: 1fr 1fr;
  }

  .library-table__row--head {
    display: none;
  }

  .library-table__actions {
    grid-column: 1 / -1;
    justify-content: flex-start;
  }
}

.model-groups--list {
  grid-template-columns: 1fr;
}

.model-groups--list .group-header {
  display: grid;
  grid-template-columns: minmax(0, 1.6fr) auto;
  align-items: center;
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

@media (max-width: 768px) {
  .token-warning__action {
    flex-basis: 100%;
    margin-left: 0;
  }

  .group-name {
    white-space: normal;
    overflow: visible;
    text-overflow: unset;
    word-break: break-word;
    overflow-wrap: anywhere;
  }

  .group-delete-preview {
    white-space: normal;
  }

  .group-meta small {
    max-width: none;
    white-space: normal;
  }

  .group-meta .p-button,
  :deep(.quant-actions .p-button) {
    min-width: 2.5rem;
    min-height: 2.5rem;
  }
}

@container model-card (max-width: 28rem) {
  .group-name {
    white-space: normal;
    overflow: visible;
    text-overflow: unset;
    word-break: break-word;
    overflow-wrap: anywhere;
  }
}
</style>
