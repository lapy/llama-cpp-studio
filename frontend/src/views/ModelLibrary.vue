<template>
  <div class="model-library page-shell page-shell--wide">
    <PageHeader
      title="Model library"
      description="Manage local models, configure their engines, and connect your apps."
    >
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
          label="Add model"
          icon="pi pi-search"
          class="page-header__cta"
          @click="$router.push('/search')"
        />
      </template>
    </PageHeader>

    <!-- Token Warning -->
    <HuggingFaceTokenNotice v-if="showTokenWarning" @open="showTokenDialog = true" />

    <SetupChecklist />

    <StateBanner
      v-if="modelStore.modelsStale"
      severity="warning"
      message="Showing the last loaded library. The latest refresh failed."
    >
      <template #actions>
        <Button label="Retry" size="small" @click="retryCatalogs" />
      </template>
    </StateBanner>

    <StateBanner v-else-if="runtimeOutage" severity="warning" :message="runtimeOutage">
      <template #actions>
        <Button label="Retry" size="small" @click="retryCatalogs" />
      </template>
    </StateBanner>

    <LoadingState v-if="catalogLoading && !storeHasModels" message="Loading models…" />

    <EmptyState
      v-else-if="catalogFailed && !storeHasModels"
      icon="pi pi-exclamation-circle"
      role="alert"
      title="Could not load models"
      :description="modelStore.modelsError || 'The model library request failed.'"
    >
      <Button label="Retry" icon="pi pi-refresh" @click="retryCatalogs" />
      <RouterLink class="page-action-link" to="/search">Search models</RouterLink>
    </EmptyState>

    <FirstModelPanel v-else-if="!catalogLoading && !storeHasModels" />

    <PageSection v-else variant="plain" class="library-workspace">
      <LibraryToolbar
        v-model:query="libraryQuery"
        v-model:status-filter="statusFilter"
        v-model:engine-filter="engineFilter"
        v-model:task-filter="taskFilter"
        v-model:sort-by="sortBy"
        v-model:filters-open="filtersOpen"
        :view-mode="viewMode"
        :secondary-filter-count="secondaryFilterCount"
        :engine-options="engineOptions"
        :task-options="taskOptions"
        :preset-filters="libraryPresetFilters"
        @apply-preset="applyLibraryPreset"
        @change-view="setViewMode"
      />

      <PageToolbar plain label="Library selection" class="library-results-summary">
        <template #summary>
          <span role="status">{{ visibleModelCount }} of {{ totalModels }} models</span>
          <label class="library-select-all">
            <input
              type="checkbox"
              :checked="selectionSummary.all"
              :indeterminate="selectionSummary.some"
              :disabled="!selectableQuantIds.length"
              aria-label="Select all visible models"
              @change="toggleSelectAll"
            />
            Select all visible
          </label>
          <Button
            v-if="hasLibraryFilters"
            label="Clear filters"
            size="small"
            text
            severity="secondary"
            @click="clearLibraryFilters"
          />
        </template>
      </PageToolbar>
      <LibraryBulkActions
        :models="selectedModels"
        :busy-ids="transitionIds"
        @complete="onBulkComplete"
        @clear="clearSelection"
      />
      <LibraryTransitionStatus :entries="transitionEntries" />
      <div v-if="hasLibraryFilters" class="search-chips" aria-label="Active library filters">
        <button v-if="libraryQuery" type="button" class="search-chip" @click="libraryQuery = ''">
          Name: {{ libraryQuery }} ×
        </button>
        <button
          v-if="statusFilter !== 'all'"
          type="button"
          class="search-chip"
          @click="statusFilter = 'all'"
        >
          Status: {{ statusFilter }} ×
        </button>
        <button
          v-if="engineFilter !== 'all'"
          type="button"
          class="search-chip"
          @click="engineFilter = 'all'"
        >
          Engine: {{ prettyEngine(engineFilter) }} ×
        </button>
        <button
          v-if="taskFilter !== 'all'"
          type="button"
          class="search-chip"
          @click="taskFilter = 'all'"
        >
          Task: {{ taskLabel(taskFilter) }} ×
        </button>
      </div>
      <EmptyState
        v-if="!displayGroups.length"
        icon="pi pi-filter"
        title="No models match"
        description="Try a different name, status, engine, or task."
      >
        <Button label="Clear filters" severity="secondary" outlined @click="clearLibraryFilters" />
      </EmptyState>

      <!-- Model groups (GGUF + Safetensors) -->
      <LibraryListView
        v-else-if="viewMode === 'list'"
        :lines="libraryLines"
        :is-selected="isSelected"
        :is-starting="isQuantStarting"
        :is-stopping="isQuantStopping"
        @toggle-selection="toggleSelection"
        @start="startModel"
        @stop="stopModel"
        @configure="configureModel"
        @audio="openAudio"
        @connect="openConnect"
        @refresh="refreshModelStatus"
        @delete="confirmDeleteModel"
        @delete-group="confirmDeleteGroup"
      />
      <div v-else class="page-card-grid model-groups">
        <div
          v-for="group in displayGroups"
          :key="group.huggingface_id"
          class="model-group"
          :class="{ 'is-running': groupIsRunning(group) }"
        >
          <!-- Group header: multi-variant model (expandable) -->
          <div v-if="!isStandaloneGroup(group)" class="group-header">
            <button
              type="button"
              class="group-toggle"
              :aria-expanded="expandedGroups.has(group.huggingface_id)"
              @click="toggleGroup(group.huggingface_id)"
            >
              <span class="group-title">
                <span class="group-heading">
                  <i
                    :class="[
                      'pi',
                      'group-chevron',
                      expandedGroups.has(group.huggingface_id)
                        ? 'pi-chevron-down'
                        : 'pi-chevron-right',
                    ]"
                  />
                  <span class="group-name">{{ groupTitle(group) }}</span>
                  <span
                    v-if="group.huggingface_id && groupTitle(group) !== group.huggingface_id"
                    class="group-repo"
                    >{{ group.huggingface_id }}</span
                  >
                  <span
                    v-if="groupTotalFileSize(group) > 0"
                    class="file-size file-size--total"
                    title="Total size (all quantizations)"
                  >
                    {{ formatLibraryBytes(groupTotalFileSize(group)) }}
                  </span>
                  <Tag
                    :value="groupDisplayState(group).label"
                    :severity="groupDisplayState(group).severity"
                    :title="groupDisplayState(group).detail"
                    class="running-badge"
                    role="status"
                  />
                </span>
                <span class="group-tags">
                  <Tag
                    v-if="primaryQuant(group)"
                    :value="prettyEngine(groupEngine(group))"
                    severity="secondary"
                    class="engine-tag"
                  />
                  <Tag
                    v-if="primaryQuant(group) && primaryQuant(group).format"
                    :value="formatLabel(primaryQuant(group).format)"
                    severity="info"
                  />
                  <Tag v-if="group.family" :value="group.family" severity="secondary" />
                  <Tag
                    v-for="task in group.tasks || []"
                    :key="`group-task-${task}`"
                    :value="taskLabel(task)"
                    severity="success"
                  />
                </span>
              </span>
            </button>
            <div class="group-meta">
              <span
                v-if="group.quantizations?.length"
                class="group-delete-preview"
                :title="`Remove ${group.quantizations.length} quantization${group.quantizations.length !== 1 ? 's' : ''}`"
                @click.stop
              >
                {{ group.quantizations.length }}
                {{ group.quantizations.length === 1 ? 'item' : 'items' }}
              </span>
              <details class="row-menu" @click.stop>
                <summary :aria-label="`More actions for ${groupTitle(group)}`">More</summary>
                <button type="button" @click="confirmDeleteGroup(group.huggingface_id)">
                  Delete group
                </button>
              </details>
            </div>
          </div>

          <!-- Standalone snapshot or prepared bundle (single-row, non-expandable) -->
          <div v-else class="group-header safetensors-header">
            <div class="group-title">
              <div class="group-heading">
                <label v-if="primaryQuant(group)" class="library-check">
                  <input
                    type="checkbox"
                    :checked="isSelected(primaryQuant(group).id)"
                    :aria-label="`Select ${groupTitle(group)}`"
                    @change="toggleSelection(primaryQuant(group).id)"
                  />
                </label>
                <span class="group-name">{{ groupTitle(group) }}</span>
                <span
                  v-if="group.huggingface_id && groupTitle(group) !== group.huggingface_id"
                  class="group-repo"
                  >{{ group.huggingface_id }}</span
                >
                <span v-if="primaryQuant(group) && primaryQuant(group).file_size" class="file-size">
                  {{ formatLibraryBytes(primaryQuant(group).file_size) }}
                </span>
                <Tag
                  :value="groupDisplayState(group).label"
                  :severity="groupDisplayState(group).severity"
                  :title="groupDisplayState(group).detail"
                  class="running-badge"
                  role="status"
                />
              </div>
              <div class="group-tags">
                <Tag
                  v-if="primaryQuant(group)"
                  :value="prettyEngine(groupEngine(group))"
                  severity="secondary"
                  class="engine-tag"
                />
                <Tag
                  v-if="primaryQuant(group) && primaryQuant(group).format"
                  :value="formatLabel(primaryQuant(group).format)"
                  severity="info"
                />
                <Tag v-if="group.family" :value="group.family" severity="secondary" />
                <Tag
                  v-for="task in group.tasks || []"
                  :key="`standalone-task-${task}`"
                  :value="taskLabel(task)"
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
              <ModelCardActions
                v-if="primaryQuant(group)"
                :model="primaryQuant(group)"
                :name="groupTitle(group)"
                :is-starting="isQuantStarting(primaryQuant(group))"
                :is-stopping="isQuantStopping(primaryQuant(group))"
                stop-propagation
                @start="startModel(primaryQuant(group).id)"
                @stop="stopModel(primaryQuant(group).id)"
                @configure="configureModel(primaryQuant(group).id)"
                @audio="openAudio(primaryQuant(group).id)"
                @connect="openConnect(primaryQuant(group))"
                @refresh="refreshModelStatus"
                @delete="confirmDeleteModel(primaryQuant(group).id)"
              />
            </div>
            <div
              v-if="primaryQuant(group) && primaryQuant(group).downloaded_at"
              class="safetensors-header__footer"
            >
              <span class="downloaded-at">
                Downloaded {{ formatLibraryDate(primaryQuant(group).downloaded_at) }}
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
                :format-bytes="formatLibraryBytes"
                :format-date="formatLibraryDate"
                selectable
                :selected="isSelected(quant.id)"
                @select="toggleSelection"
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
    </PageSection>

    <LibraryToast @configure="configureModel" />

    <HuggingFaceTokenDialog v-model:visible="showTokenDialog" />

    <ConnectDialog
      v-if="connectVisible"
      v-model:visible="connectVisible"
      :model="connectModel"
      :proxy-port="proxyPort"
      :public-inference-url="publicInferenceUrl"
      @update:public-inference-url="onPublicInferenceUrl"
    />
  </div>
</template>

<script setup>
import {
  modelRuntimeState,
  groupRuntimeState,
  taskLabel,
  formatLabel,
} from '@/utils/modelPresentation'
import {
  buildLibraryLines,
  formatLibraryBytes,
  formatLibraryDate,
  groupEngine,
  groupIsRunning,
  groupNeedsAttention,
  groupRecentStamp,
  groupRuntimeQuality,
  groupTitle,
  isStandaloneGroup,
  primaryQuant,
  prettyEngine,
  quantStatus,
  groupTotalFileSize,
} from '@/utils/libraryPresentation'
import { formatAxiosDetail } from '@/utils/apiErrors'
import { useLibrarySelection } from '@/composables/useLibrarySelection'
import { createDownloadSetupAnnouncer } from '@/composables/useDownloadSetupToast'
import { useQueryState } from '@/composables/useQueryState'
import { ref, computed, onMounted, onUnmounted } from 'vue'
import { watchDialogFocus } from '@/composables/useFocusReturn'
import { RouterLink, useRouter } from 'vue-router'
import { useConfirm } from 'primevue/useconfirm'
import { useToast } from 'primevue/usetoast'
import Button from 'primevue/button'
import Tag from 'primevue/tag'
import ModelRow from '@/components/ModelRow.vue'
import ModelCardActions from '@/components/models/ModelCardActions.vue'
import LibraryTransitionStatus from '@/components/models/LibraryTransitionStatus.vue'
import LibraryToast from '@/components/models/LibraryToast.vue'
import LibraryBulkActions from '@/components/models/LibraryBulkActions.vue'
import { useModelStore } from '@/stores/models'
import { useProgressStore } from '@/stores/progress'
import { audioTabFromConfig } from '@/composables/useAudioInferenceClient'
import { requireSingleConfirmation } from '@/composables/singleConfirm'
import {
  versionDeleteErrorText,
  versionDeleteRetry,
  withheldConfirmation,
} from '@/composables/actionConfirmation'
import PageHeader from '@/components/common/PageHeader.vue'
import PageSection from '@/components/common/PageSection.vue'
import PageToolbar from '@/components/common/PageToolbar.vue'
import StateBanner from '@/components/common/StateBanner.vue'
import LoadingState from '@/components/common/LoadingState.vue'
import EmptyState from '@/components/common/EmptyState.vue'
import {
  ConnectDialog,
  FirstModelPanel,
  HuggingFaceTokenDialog,
  HuggingFaceTokenNotice,
  LibraryListView,
  LibraryToolbar,
  SetupChecklist,
} from '@/components/models/lazyLibraryUi'
import { useEnginesStore } from '@/stores/engines'

const router = useRouter()
const confirm = useConfirm()
const toast = useToast()
const modelStore = useModelStore()
const progressStore = useProgressStore()
const enginesStore = useEnginesStore()
const announceDownloadedSetup = createDownloadSetupAnnouncer({ modelStore, toast })
const connectVisible = ref(false)
const connectModel = ref(null)
const proxyPort = computed(() => enginesStore.systemStatus?.proxy_status?.port || 2000)
const publicInferenceUrl = computed(
  () => enginesStore.systemStatus?.proxy_status?.public_inference_url || '',
)
const showTokenWarning = computed(
  () =>
    !modelStore.hasHuggingfaceToken &&
    sourceGroups.value.some(
      (group) => group?.gated || (group.quantizations || []).some((quant) => quant?.gated),
    ),
)

function onPublicInferenceUrl(url) {
  const status = enginesStore.systemStatus
  if (!status || typeof status !== 'object') return
  status.proxy_status = { ...(status.proxy_status || {}), public_inference_url: url }
}

// ── State ──────────────────────────────────────────────────
const expandedGroups = ref(new Set())
const filtersOpen = ref(false)
const libraryQuery = ref('')
const statusFilter = ref('all')
const engineFilter = ref('all')
const taskFilter = ref('all')
const sortBy = ref('name')
const viewMode = ref(
  typeof localStorage !== 'undefined' &&
    localStorage.getItem('llama-studio.library.view') === 'list'
    ? 'list'
    : 'cards',
)
const secondaryFilterCount = computed(
  () =>
    Number(engineFilter.value !== 'all') +
    Number(taskFilter.value !== 'all') +
    Number(sortBy.value !== 'name'),
)
const hasLibraryFilters = computed(() =>
  Boolean(
    libraryQuery.value ||
    statusFilter.value !== 'all' ||
    engineFilter.value !== 'all' ||
    taskFilter.value !== 'all',
  ),
)
const visibleModelCount = computed(() =>
  displayGroups.value.reduce((sum, group) => sum + (group.quantizations?.length || 0), 0),
)
function clearLibraryFilters() {
  libraryQuery.value = ''
  statusFilter.value = engineFilter.value = taskFilter.value = 'all'
}
const EXPANDED_KEY = 'llama-studio.library.expanded'
useQueryState(
  Object.fromEntries(
    [
      ['q', libraryQuery, ''],
      ['status', statusFilter, 'all'],
      ['engine', engineFilter, 'all'],
      ['task', taskFilter, 'all'],
      ['sort', sortBy, 'name'],
      ['layout', viewMode, viewMode.value],
    ].map(([key, target, fallback]) => [
      key,
      {
        get: () => target.value,
        set: (value) => {
          target.value = value
        },
        default: fallback,
        parse: (value) => {
          const choices = {
            status: ['all', 'running', 'stopped', 'attention'],
            sort: ['name', 'size', 'recent'],
            layout: ['cards', 'list'],
          }
          return choices[key]
            ? choices[key].includes(value)
              ? value
              : fallback
            : value.slice(0, 200) || fallback
        },
      },
    ]),
  ),
)
const libraryPresetFilters = computed(() => ({
  q: libraryQuery.value,
  status: statusFilter.value,
  engine: engineFilter.value,
  task: taskFilter.value,
  sort: sortBy.value,
}))
function applyLibraryPreset(filters) {
  libraryQuery.value = filters?.q ?? ''
  statusFilter.value = filters?.status ?? 'all'
  engineFilter.value = filters?.engine ?? 'all'
  taskFilter.value = filters?.task ?? 'all'
  sortBy.value = filters?.sort ?? 'name'
}
const startingModels = ref(new Set())
const stoppingModels = ref(new Set())
const showTokenDialog = ref(false)
watchDialogFocus(connectVisible)
watchDialogFocus(showTokenDialog)
const catalogRefreshInFlight = ref(false)
const FAST_POLL_MS = 1000
const IDLE_POLL_MS = 5000
let pollTimer = null
let pollGeneration = 0
let catalogDisposed = false
let catalogAbort = null
let unsubscribeDownloadComplete = null

async function retryCatalogs() {
  try {
    await refreshCatalogs()
  } catch {
    /* The store keeps the error for the empty and stale states. */
  }
}

async function refreshCatalogs() {
  if (catalogDisposed || catalogRefreshInFlight.value) return
  const controller = new AbortController()
  catalogAbort = controller
  catalogRefreshInFlight.value = true
  try {
    await modelStore.fetchModels({ signal: controller.signal })
  } catch {
    /* The store keeps the error for the empty and stale states. */
  } finally {
    if (catalogAbort === controller) catalogAbort = null
    catalogRefreshInFlight.value = false
  }
}

function routinePollPaused() {
  return (
    typeof document !== 'undefined' &&
    document.visibilityState === 'hidden' &&
    !hasLiveModelTransitions.value
  )
}

function stopCatalogPoll() {
  catalogDisposed = true
  pollGeneration += 1
  if (pollTimer) clearTimeout(pollTimer)
  pollTimer = null
  catalogAbort?.abort()
  catalogAbort = null
}

// ── Computed ───────────────────────────────────────────────
// Backend /api/models already returns both GGUF and safetensors models
// grouped appropriately, so we can display models directly from there.
const sourceGroups = computed(() => modelStore.models || [])

const storeHasModels = computed(() =>
  sourceGroups.value.some((group) => (group.quantizations || []).length > 0),
)

const catalogLoading = computed(() => modelStore.loading && !storeHasModels.value)

const runtimeOutage = computed(() => {
  const quants = sourceGroups.value.flatMap((group) => group.quantizations || [])
  if (quants.some((quant) => quant?.runtime_quality === 'unreachable')) {
    return 'The inference proxy could not be reached. Running models are not shown as stopped.'
  }
  if (quants.some((quant) => quant?.runtime_quality === 'stale')) {
    return 'The inference proxy is unreachable. Showing the last known running state.'
  }
  return ''
})

const catalogFailed = computed(() => Boolean(modelStore.modelsError) && !modelStore.loading)

/** Runtime state for a single quantization, including in-flight transitions. */
function quantRuntimeState(quant) {
  return modelRuntimeState(quant, {
    isStarting: isQuantStarting(quant),
    isStopping: isQuantStopping(quant),
  })
}

/** Group status follows an in-flight transition before its settled state. */
function groupDisplayState(group) {
  const transition = (group?.quantizations || [])
    .map((quant) => quantRuntimeState(quant))
    .find((state) => state.key === 'starting' || state.key === 'stopping')
  return transition || groupRuntimeState(group)
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

const engineOptions = computed(() => [
  { label: 'Any engine', value: 'all' },
  ...engineChoices.value.map((engine) => ({ label: prettyEngine(engine), value: engine })),
])

const taskOptions = computed(() => [
  { label: 'Any task', value: 'all' },
  ...taskChoices.value.map((task) => ({ label: taskLabel(task), value: taskLabel(task) })),
])

const displayGroups = computed(() => {
  const query = libraryQuery.value.trim().toLowerCase()
  let groups = sourceGroups.value.filter((group) => {
    if (query) {
      const hay = [
        group.huggingface_id,
        group.base_model_name,
        ...(group.quantizations || []).map((quant) => quant.quantization || quant.name || ''),
      ]
        .join(' ')
        .toLowerCase()
      if (!hay.includes(query)) return false
    }
    if (statusFilter.value === 'running' && !groupIsRunning(group)) return false
    if (
      statusFilter.value === 'stopped' &&
      (groupIsRunning(group) || groupRuntimeQuality(group) !== 'verified')
    )
      return false
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

const libraryLines = computed(() =>
  buildLibraryLines(displayGroups.value, { stateFor: quantRuntimeState }),
)

/** Quantizations represented by a visible checkbox in the current layout. */
const selectableQuants = computed(() => {
  if (viewMode.value === 'list') {
    return libraryLines.value.flatMap((line) => (line.quant ? [line.quant] : []))
  }
  return displayGroups.value.flatMap((group) => {
    if (isStandaloneGroup(group)) {
      const quant = primaryQuant(group)
      return quant ? [quant] : []
    }
    if (!expandedGroups.value.has(group.huggingface_id)) return []
    return group.quantizations || []
  })
})
const selectableQuantIds = computed(() =>
  selectableQuants.value.map((quant) => quant.id).filter(Boolean),
)
const libraryFilterKey = computed(() =>
  [libraryQuery.value.trim(), statusFilter.value, engineFilter.value, taskFilter.value].join('|'),
)
const {
  selectedIds,
  summary: selectionSummary,
  isSelected,
  toggle: toggleSelection,
  toggleSelectAll,
  clearSelection,
} = useLibrarySelection({ visibleIds: selectableQuantIds, filterKey: libraryFilterKey })
const transitionIds = computed(() => [...startingModels.value, ...stoppingModels.value])
const transitionEntries = computed(() => {
  const entries = []
  for (const id of startingModels.value) {
    const quant = findQuantById(id)
    entries.push({ id, label: quant?.quantization || quant?.name || id, verb: 'Starting' })
  }
  for (const id of stoppingModels.value) {
    const quant = findQuantById(id)
    entries.push({ id, label: quant?.quantization || quant?.name || id, verb: 'Stopping' })
  }
  return entries
})
function refreshModelStatus() {
  void retryCatalogs()
}
const selectedModels = computed(() => {
  const selected = new Set(selectedIds.value)
  return selectableQuants.value.filter((quant) => selected.has(String(quant.id)))
})
function onBulkComplete(result) {
  if (result?.done) clearSelection()
  queueCatalogPoll()
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
  sourceGroups.value.reduce((acc, g) => acc + (g.quantizations?.length ?? 0), 0),
)

// ── Group expand/collapse ──────────────────────────────────
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

function openConnect(quant) {
  connectModel.value = quant
  connectVisible.value = true
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms))
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

/** Play busy while start is in flight or we are waiting for a verified running model. */
function isQuantStarting(quant) {
  if (!quant?.id) return false
  return startingModels.value.has(modelIdKey(quant.id))
}

const START_CONFIRM_MS = 30_000

function modelRunState(quant) {
  return String(quant?.run_state || quant?.status || '').toLowerCase()
}

function modelHasStarted(quant) {
  if (!quant) return false
  return modelRuntimeState(quant).key === 'running'
}

function modelHasFailed(quant) {
  if (!quant) return false
  return modelRuntimeState(quant).key === 'error'
}

function modelDisplayName(modelId) {
  const quant = findQuantById(modelId)
  return quant?.quantization || quant?.name || String(modelId)
}

/** Failures land in the library toast so Configure stays one click away. */
function notifyStartFailure(modelId, detail) {
  toast.add({
    severity: 'error',
    summary: 'Failed to start',
    detail,
    life: 7000,
    group: 'model-library',
    configureModelId: modelId,
  })
}

function isQuantStopping(quant) {
  if (!quant?.id) return false
  return stoppingModels.value.has(modelIdKey(quant.id))
}

const hasLiveModelTransitions = computed(() => {
  if (startingModels.value.size || stoppingModels.value.size) return true
  return displayGroups.value.some((group) =>
    (group.quantizations || []).some((quant) => quantStatus(quant) === 'loading'),
  )
})

function queueCatalogPoll(delay = hasLiveModelTransitions.value ? FAST_POLL_MS : IDLE_POLL_MS) {
  if (catalogDisposed || routinePollPaused()) return
  const generation = pollGeneration
  if (pollTimer) clearTimeout(pollTimer)
  pollTimer = setTimeout(async () => {
    pollTimer = null
    if (catalogDisposed || generation !== pollGeneration || routinePollPaused()) return
    try {
      await refreshCatalogs()
    } catch {
      /* The store keeps the error for the empty and stale states. */
    }
    if (catalogDisposed || generation !== pollGeneration || routinePollPaused()) return
    queueCatalogPoll()
  }, delay)
}

function onLibraryVisibility() {
  if (catalogDisposed) return
  if (routinePollPaused()) {
    pollGeneration += 1
    if (pollTimer) clearTimeout(pollTimer)
    pollTimer = null
    return
  }
  pollGeneration += 1
  refreshCatalogs().finally(() => {
    if (!catalogDisposed) queueCatalogPoll()
  })
}

// ── Model actions ──────────────────────────────────────────
function confirmWithheldAction(error, verb, retry) {
  const withheld = withheldConfirmation(error)
  if (!withheld) return false
  // An earlier request may still take effect; the backend withholds a repeat
  // until this exact operation and state are confirmed. Refresh visible
  // state first, then let the user decide.
  refreshCatalogs().catch(() => {})
  requireSingleConfirmation(confirm, {
    header: `Confirm model ${verb}`,
    message: `${withheld.message} ${verb[0].toUpperCase()}${verb.slice(1)} this model again?`,
    icon: 'pi pi-exclamation-triangle',
    accept: () =>
      retry({
        confirm_operation_id: withheld.confirm_operation_id,
        confirm_state: withheld.confirm_state,
      }),
  })
  return true
}

async function startModel(modelId, confirmation = null) {
  const k = modelIdKey(modelId)
  startingModels.value.add(k)
  startingModels.value = new Set(startingModels.value)
  queueCatalogPoll(FAST_POLL_MS)
  let startingNoted = false
  try {
    await modelStore.startModel(modelId, confirmation)
    const deadline = Date.now() + START_CONFIRM_MS
    while (Date.now() < deadline && !catalogDisposed) {
      const quant = findQuantById(modelId)
      if (modelHasStarted(quant) || modelHasFailed(quant) || stoppingModels.value.has(k)) break
      if (!startingNoted && modelRunState(quant) === 'loading') {
        startingNoted = true
        toast.add({ severity: 'info', summary: 'Model is starting', life: 3000 })
      }
      await sleep(300)
      await refreshCatalogs()
    }
    try {
      await refreshCatalogs()
    } catch {
      /* The following toast reports that startup was not confirmed. */
    }
    const quant = findQuantById(modelId)
    if (stoppingModels.value.has(k)) {
      /* The user cancelled this start with Stop; the stop toast reports the outcome. */
    } else if (modelHasStarted(quant)) {
      toast.add({
        severity: 'success',
        summary: 'Model started',
        detail: `${modelDisplayName(modelId)} is running.`,
        life: 3000,
      })
    } else if (modelHasFailed(quant)) {
      notifyStartFailure(
        modelId,
        quant?.last_error || 'The model reported a failure while starting.',
      )
    } else {
      toast.add({
        severity: 'warn',
        summary: 'Startup not confirmed',
        detail: 'The model was not observed running. Refresh finished; you can start it again.',
        life: 7000,
        group: 'model-library',
        configureModelId: modelId,
      })
    }
  } catch (e) {
    if (!confirmation && confirmWithheldAction(e, 'start', (pair) => startModel(modelId, pair))) {
      return
    }
    await refreshCatalogs().catch(() => {})
    notifyStartFailure(modelId, formatAxiosDetail(e))
  } finally {
    startingModels.value.delete(k)
    startingModels.value = new Set(startingModels.value)
    queueCatalogPoll()
  }
}

async function stopModel(modelId, confirmation = null) {
  const k = modelIdKey(modelId)
  startingModels.value.delete(k)
  startingModels.value = new Set(startingModels.value)
  stoppingModels.value.add(k)
  stoppingModels.value = new Set(stoppingModels.value)
  queueCatalogPoll(FAST_POLL_MS)
  let ok = false
  try {
    await modelStore.stopModel(modelId, confirmation)
    toast.add({ severity: 'info', summary: 'Model stopped', life: 3000 })
    ok = true
  } catch (e) {
    if (!confirmation && confirmWithheldAction(e, 'stop', (pair) => stopModel(modelId, pair))) {
      return
    }
    toast.add({
      severity: 'error',
      summary: 'Failed to stop',
      detail: formatAxiosDetail(e),
      life: 4000,
    })
  } finally {
    if (ok) {
      const deadline = Date.now() + 30000
      while (Date.now() < deadline && !catalogDisposed) {
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

function openAudio(modelId) {
  const quant = modelStore.allQuantizations.find((m) => m.id === modelId)
  const tab = audioTabFromConfig({
    task: quant?.config?.task || quant?.task || quant?.tasks?.[0],
    family: quant?.config?.family || quant?.family,
  })
  router.push({ name: 'audio', query: { model: modelId, tab } })
}

async function performModelDelete(run, params, summary = 'Model removed') {
  try {
    await run(params)
    toast.add({ severity: 'info', summary, life: 3000 })
  } catch (e) {
    const retry = versionDeleteRetry(e)
    if (retry) {
      requireSingleConfirmation(confirm, {
        message: retry.message,
        header: 'Confirm Remove',
        icon: 'pi pi-exclamation-triangle',
        acceptClass: 'p-button-danger',
        accept: () => performModelDelete(run, retry.params, summary),
      })
      return
    }
    toast.add({
      severity: 'error',
      summary: 'Failed',
      detail: versionDeleteErrorText(e),
      life: 4000,
    })
  }
}

function confirmDeleteModel(modelId) {
  requireSingleConfirmation(confirm, {
    message: 'Remove this model from the library? Downloaded files will be deleted from disk.',
    header: 'Confirm Remove',
    icon: 'pi pi-exclamation-triangle',
    acceptClass: 'p-button-danger',
    accept: () => performModelDelete((params) => modelStore.deleteModel(modelId, params)),
  })
}

function confirmDeleteGroup(huggingfaceId) {
  requireSingleConfirmation(confirm, {
    message: `Remove all quantizations for "${huggingfaceId}"?`,
    header: 'Confirm Remove Group',
    icon: 'pi pi-exclamation-triangle',
    acceptClass: 'p-button-danger',
    accept: () =>
      performModelDelete(
        (params) => modelStore.deleteModelGroup(huggingfaceId, params),
        undefined,
        'Group removed',
      ),
  })
}

// ── Lifecycle ──────────────────────────────────────────────
onMounted(() => {
  try {
    const saved = JSON.parse(sessionStorage.getItem(EXPANDED_KEY) || '[]')
    if (Array.isArray(saved)) expandedGroups.value = new Set(saved)
  } catch {
    /* ignore */
  }
  unsubscribeDownloadComplete = progressStore.subscribeToDownloadComplete(async (payload) => {
    await refreshCatalogs()
    announceDownloadedSetup(payload)
  })
  catalogDisposed = false
  document.addEventListener('visibilitychange', onLibraryVisibility)
  queueCatalogPoll()
  Promise.allSettled([modelStore.fetchModels(), modelStore.fetchHuggingfaceTokenStatus?.()])
})

onUnmounted(() => {
  document.removeEventListener('visibilitychange', onLibraryVisibility)
  stopCatalogPoll()
  if (unsubscribeDownloadComplete) unsubscribeDownloadComplete()
})
</script>

<style scoped>
.group-toggle {
  display: flex;
  flex: 1 1 100%;
  min-width: 0;
  padding: 0;
  border: 0;
  background: transparent;
  color: inherit;
  text-align: left;
  cursor: pointer;
}

@container model-card (min-width: 28rem) {
  .group-toggle {
    flex: 1 1 0;
  }
}

/* layout: .page-shell */

/* ── Groups ───────────────────────────────────────────── */
.model-group {
  background: var(--bg-card);
  border: 1px solid var(--border-primary);
  border-radius: var(--radius-lg);
  overflow: hidden;
  container-type: inline-size;
  container-name: model-card;
  transition:
    border-color 0.2s ease,
    box-shadow 0.2s ease;
}

.model-group.is-running {
  box-shadow: inset 0 2px 0 var(--status-success);
}

.group-header {
  display: flex;
  flex-wrap: wrap;
  justify-content: space-between;
  align-items: flex-start;
  padding: 0.75rem 1rem;
  cursor: pointer;
  user-select: none;
  background: var(--bg-surface);
  transition: background 0.15s;
  gap: 0.35rem 0.5rem;
}

.group-header:hover {
  background: var(--bg-card-hover);
}

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
  gap: 1rem;
  padding: 1.25rem;
  box-sizing: border-box;
  cursor: default;
}

.safetensors-header:hover {
  background: var(--bg-surface);
}

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
  font-size: var(--text-xs);
  line-height: 1.15;
  color: var(--text-secondary);
  opacity: 0.9;
}

.group-chevron {
  font-size: var(--text-xs);
  color: var(--text-secondary);
}

.group-name {
  font-weight: 600;
  font-size: 0.9rem;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  min-width: 0;
}

.running-badge {
  flex-shrink: 0;
}

.group-meta {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 0.25rem;
  flex-shrink: 0;
}

.group-delete-preview {
  font-size: var(--text-xs);
  line-height: 1.25;
  color: var(--text-secondary);
  white-space: nowrap;
  user-select: none;
}

.group-meta small {
  font-size: var(--text-xs);
  color: var(--text-secondary);
  font-family: monospace;
  max-width: 200px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

/* ── Quantizations ────────────────────────────────────── */
.group-collapse-enter-active,
.group-collapse-leave-active {
  transition: all 0.2s ease;
  overflow: hidden;
}
.group-collapse-enter-from,
.group-collapse-leave-to {
  max-height: 0;
  opacity: 0;
}
.group-collapse-enter-to,
.group-collapse-leave-from {
  max-height: 1000px;
  opacity: 1;
}

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
  background: var(--bg-surface);
  border: 1px solid var(--border-primary);
  border-radius: var(--radius-md);
  transition:
    border-color 0.15s ease,
    box-shadow 0.2s ease,
    background 0.2s ease;
  box-sizing: border-box;
}

:deep(.quant-row > .quant-info) {
  min-width: 0;
}

:deep(.quant-row > .quant-actions) {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  justify-content: flex-end;
  gap: 0.4rem;
}

@media (max-width: 600px) {
  :deep(.quant-row > .quant-actions) {
    flex-wrap: nowrap;
    justify-content: flex-start;
    gap: 0.25rem;
  }

  :deep(.quant-actions .p-button) {
    min-width: 0;
    padding: 0.3rem 0.4rem;
    font-size: 0.72rem;
  }
}

:deep(.quant-row > .quant-row__footer) {
  min-width: 0;
}

:deep(.quant-row__footer .downloaded-at) {
  display: block;
  font-size: var(--text-xs);
  line-height: 1.15;
  color: var(--text-secondary);
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
  font-size: var(--text-xs);
  line-height: 1.25;
  color: var(--text-secondary);
  font-variant-numeric: tabular-nums;
}

/* Emphasize engine tag with a distinct background */
.engine-tag {
  background-color: var(--bg-surface);
  border-color: var(--border-secondary);
  color: var(--text-secondary);
}

.row-menu {
  position: relative;
}

.row-menu summary {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  min-height: 2rem;
  padding: 0.2rem 0.55rem;
  border: 1px solid var(--border-primary);
  border-radius: var(--radius-sm);
  background: var(--bg-secondary);
  color: var(--text-primary);
  font: inherit;
  font-size: var(--text-xs);
  cursor: pointer;
  list-style: none;
}

.row-menu summary::-webkit-details-marker {
  display: none;
}

.row-menu button {
  display: block;
  width: 100%;
  margin-top: 0.25rem;
  padding: 0.35rem 0.6rem;
  border: 1px solid var(--border-primary);
  border-radius: var(--radius-sm);
  background: var(--bg-secondary);
  color: var(--status-error);
  font: inherit;
  font-size: var(--text-xs);
  cursor: pointer;
}

.group-repo {
  color: var(--text-secondary);
  font-size: var(--text-xs);
}

@media (max-width: 768px) {
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
  .group-heading {
    flex-wrap: wrap;
  }
  .group-heading .group-name {
    flex: 1 1 100%;
  }
  .group-repo {
    flex: 1 1 100%;
    overflow-wrap: anywhere;
  }
  .group-meta > :deep(.p-button) {
    flex-shrink: 0;
  }

  .group-name {
    white-space: normal;
    overflow: visible;
    text-overflow: unset;
    word-break: break-word;
    overflow-wrap: anywhere;
  }
}

.library-results-summary :deep(.page-toolbar__summary) {
  gap: 1rem;
}

.library-select-all {
  display: inline-flex;
  align-items: center;
  gap: 0.4rem;
  font-size: var(--text-sm);
  color: var(--text-secondary);
  cursor: pointer;
}

.library-select-all input,
.library-check input {
  width: 1rem;
  height: 1rem;
  margin: 0;
  accent-color: var(--accent-primary);
}

.library-check {
  display: inline-flex;
  align-items: center;
  flex: 0 0 auto;
}

.safetensors-header {
  flex-direction: column;
  align-items: stretch;
  gap: 1rem;
  padding: 1.25rem;
}
.safetensors-header .group-title {
  width: 100%;
  min-width: 0;
}
.safetensors-header .group-heading {
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
}
.safetensors-header .group-name {
  flex: 1 1 100%;
  white-space: normal;
  overflow: visible;
  text-overflow: clip;
  font-size: 1rem;
}
.safetensors-header .group-meta {
  width: 100%;
  justify-content: flex-start;
  padding-top: 0.75rem;
  border-top: 1px solid var(--border-primary);
}
</style>
