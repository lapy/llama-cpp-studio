<template>
  <div class="proxy-page page-shell page-shell--wide">
    <PageHeader title="Proxy" description="Live traffic, engine health, and gateway controls.">
      <template #meta>
        <Tag :value="statusLabel" :severity="statusSeverity" />
        <Tag v-if="liveRunning.length" :value="`${liveRunning.length} running`" severity="info" />
      </template>
      <template #actions>
        <Button
          icon="pi pi-refresh"
          text
          severity="secondary"
          :loading="loading"
          aria-label="Refresh proxy status"
          v-tooltip.top="'Refresh'"
          @click="refresh()"
        />
      </template>
    </PageHeader>

    <UnsavedChangesDialog
      :visible="leavePromptVisible"
      :saving="saving"
      @stay="finishLeave(false)"
      @discard="finishLeave(true)"
      @save="saveBeforeLeaving"
    />
    <FreshnessStatus :updated="lastUpdated" :failed="Boolean(error)" :offline="offline" />
    <StateBanner v-if="error" severity="error" role="status">
      {{ error }}
      <template #actions>
        <Button
          label="Retry"
          size="small"
          severity="secondary"
          outlined
          :loading="loading"
          @click="refresh()"
        />
      </template>
    </StateBanner>
    <StateBanner v-if="!activeBackend && configuredBackend" severity="warning">
      The proxy is stopped. Start it to accept inference requests.
      <template #actions>
        <Button
          label="Start proxy"
          size="small"
          severity="warn"
          :loading="lifecycleBusy"
          @click="lifecycle('start')"
        />
      </template>
    </StateBanner>
    <StateBanner v-else-if="restartRequired" severity="warning">
      Saved settings changed since the proxy last (re)started. Restart to activate them.
      <template #actions>
        <Button
          label="Restart &amp; apply"
          size="small"
          severity="warn"
          outlined
          :loading="lifecycleBusy"
          @click="confirmRestart"
        />
      </template>
    </StateBanner>

    <LoadingState v-if="loading && !loaded" message="Loading proxy state…" />

    <template v-else>
      <SectionTabs
        v-model="activeTab"
        :items="tabItems"
        id-prefix="proxy"
        aria-label="Proxy sections"
      />
      <div
        :id="`proxy-panel-${activeTab}`"
        class="proxy-panels"
        role="tabpanel"
        :aria-labelledby="`proxy-tab-${activeTab}`"
      >
        <PageSection
          v-show="activeTab === 'settings'"
          title="Serving limits"
          description="Saved limits and timeouts apply when the proxy restarts."
        >
          <template #actions>
            <Button
              label="Save settings"
              :disabled="!dirty"
              :loading="saving"
              @click="saveSettings"
            />
            <Button
              v-if="dirty"
              label="Discard changes"
              size="small"
              severity="secondary"
              text
              @click="resetDraft"
            />
          </template>
          <PageToolbar plain label="Proxy lifecycle">
            <template #actions>
              <Button
                label="Start"
                icon="pi pi-play"
                size="small"
                severity="success"
                outlined
                :disabled="Boolean(activeBackend)"
                @click="lifecycle('start')"
              />
              <Button
                label="Restart &amp; apply"
                icon="pi pi-replay"
                size="small"
                severity="secondary"
                outlined
                :loading="lifecycleBusy"
                @click="confirmRestart"
              />
              <Button
                label="Stop"
                icon="pi pi-stop-circle"
                size="small"
                severity="danger"
                outlined
                :disabled="!activeBackend"
                @click="confirmStop"
              />
            </template>
          </PageToolbar>
          <div class="page-form">
            <label class="page-form__label">
              Inference port
              <input
                v-model.number="draft.proxy_port"
                class="field"
                type="number"
                min="1"
                max="65535"
              />
              <small class="page-form__hint">Applies on the next proxy restart.</small>
            </label>
            <label v-for="field in performanceFields" :key="field.name" class="page-form__label">
              {{ field.label }}
              <input
                v-model.number="draft[field.name]"
                class="field"
                type="number"
                :min="field.min || 1"
                :step="field.step || 1"
                :placeholder="field.placeholder"
              />
              <small class="page-form__hint">{{ field.hint }}</small>
            </label>
            <label class="page-form__label">
              Idle unload (seconds)
              <input
                v-model.number="draft.idle_ttl_seconds"
                class="field"
                type="number"
                min="0"
                placeholder="Disabled"
              />
            </label>
            <ProxyCaptureToggle v-model="draft.capture_request_payloads" />
            <label class="page-form__label">
              Maximum request size (bytes)
              <input
                v-model.number="draft.max_body_bytes"
                class="field"
                type="number"
                min="1"
                placeholder="268435456"
              />
            </label>
            <label class="page-form__label">
              Maximum loaded models
              <input
                v-model.number="draft.max_loaded_models"
                class="field"
                type="number"
                min="1"
                placeholder="Unlimited"
              />
            </label>
            <label class="page-form__label">
              Global concurrent requests
              <input
                v-model.number="draft.global_concurrency"
                class="field"
                type="number"
                min="1"
                placeholder="Unlimited"
              />
            </label>
            <label class="page-form__label">
              Concurrent requests per model
              <input
                v-model.number="draft.per_model_concurrency"
                class="field"
                type="number"
                min="1"
                placeholder="Unlimited"
              />
            </label>
            <label class="page-form__label">
              Queue timeout (seconds)
              <input
                v-model.number="draft.queue_timeout_seconds"
                class="field"
                type="number"
                min="0.1"
                step="0.1"
              />
            </label>
            <label class="page-form__label">
              Total request deadline (seconds)
              <input
                v-model.number="draft.request_timeout_seconds"
                class="field"
                type="number"
                min="0.1"
                placeholder="Unlimited"
              />
            </label>
            <label class="page-form__label">
              Engine readiness timeout (seconds)
              <input
                v-model.number="draft.readiness_timeout_seconds"
                class="field"
                type="number"
                min="1"
              />
            </label>
            <label class="page-form__label">
              Graceful drain timeout (seconds)
              <input
                v-model.number="draft.drain_timeout_seconds"
                class="field"
                type="number"
                min="1"
              />
            </label>
          </div>
          <div class="inline-control">
            <ToggleSwitch v-model="draft.allow_anonymous" input-id="allow-anonymous" />
            <label for="allow-anonymous">
              Allow anonymous inference while no access key is configured
              <small class="page-form__hint">
                Once any key exists, anonymous traffic is rejected regardless of this setting.
              </small>
            </label>
          </div>
        </PageSection>

        <PageSection
          v-show="activeTab === 'settings'"
          title="Routing"
          description="Routing changes apply immediately without restarting engines or interrupting streams. Choose a default model or route an alias to one or more targets. Multiple targets are selected round-robin."
        >
          <div class="page-form">
            <label class="page-form__label">
              Default model
              <select v-model="draft.default_model" class="field">
                <option :value="null">Automatic — single running model</option>
                <option v-for="model in catalog" :key="model.id" :value="model.id">
                  {{ model.id }}
                </option>
              </select>
            </label>
          </div>
          <div class="alias-list">
            <div v-for="(row, index) in aliasRows" :key="index" class="alias-row">
              <input
                v-model="row.alias"
                class="field"
                placeholder="Alias, for example fast"
                aria-label="Alias name"
              />
              <input
                v-model="row.targets"
                class="field"
                placeholder="Target model IDs, comma-separated"
                aria-label="Alias targets"
              />
              <Button
                icon="pi pi-times"
                text
                rounded
                severity="danger"
                size="small"
                aria-label="Remove alias"
                v-tooltip.top="'Remove alias'"
                @click="aliasRows.splice(index, 1)"
              />
            </div>
            <div>
              <Button
                label="Add alias"
                icon="pi pi-plus"
                size="small"
                severity="secondary"
                outlined
                @click="aliasRows.push({ alias: '', targets: '' })"
              />
            </div>
          </div>
        </PageSection>

        <GpuAdmissionPanel
          v-show="activeTab === 'settings'"
          v-model:mode="draft.vram_admission"
          v-model:margin="draft.vram_safety_margin_mib"
          :rows="vramRows"
          :catalog="catalog"
          :snapshot="vramSnapshot"
          @update:rows="vramRows = $event"
        />

        <NetworkAccessPanel v-show="activeTab === 'settings'" :network="network" />

        <ProxyOverviewPanel
          v-show="activeTab === 'overview'"
          :summary="summary"
          :running="running"
          :live-running="liveRunning"
          :requests="overviewRequests"
          :now="observedAt"
          :settings="observedSettings"
          :catalog="catalog"
          :vram-snapshot="vramSnapshot"
          :semantics="metricSemantics"
          :active-backend="activeBackend"
          :open-logs="openLogs"
          :logs="logs"
          @reset-stats="resetStats"
          @inspect-outcome="inspectOutcome"
          @inspect-failures="inspectFailures"
          @inspect-requests="inspectRequests"
          @open-settings="activeTab = 'settings'"
          @toggle-logs="toggleLogs"
          @download-logs="downloadLogs"
          @clear-logs="confirmClearLogs"
          @remove-failed="removeFailedEngine"
        />

        <ProxyActivityPanel v-show="activeTab === 'events'" :activity="activity" />

        <ProxyAccessKeys
          v-show="activeTab === 'keys'"
          :keys="keys"
          @refresh="refresh({ quiet: true })"
        />

        <ProxyRequestsPanel
          v-show="activeTab === 'requests'"
          :requests="requests"
          :total="requestTotal"
          :offset="requestOffset"
          :loading="requestsLoading"
          :error="requestsError"
          :filters="requestFilters"
          :keys="keys"
          :states="requestStates"
          :selected-request="selectedRequest"
          :semantics="metricSemantics"
          @update:filter="(key, value) => (requestFilters[key] = value)"
          @apply="loadRequests(true)"
          @reset="resetRequestFilters"
          @page="changeRequestPage"
          @select-request="setInspectedRequest"
        />
      </div>
    </template>
  </div>
</template>

<script setup>
import { computed, onMounted, onUnmounted, reactive, ref, watch } from 'vue'
import axios from 'axios'
import { useConfirm } from 'primevue/useconfirm'
import { useToast } from 'primevue/usetoast'
import Button from 'primevue/button'
import Tag from 'primevue/tag'
import ToggleSwitch from 'primevue/toggleswitch'
import { useDraftGuard } from '@/composables/useDraftGuard'
import { useQueryState } from '@/composables/useQueryState'
import { requireSingleConfirmation } from '@/composables/singleConfirm'
import FreshnessStatus from '@/components/common/FreshnessStatus.vue'
import LoadingState from '@/components/common/LoadingState.vue'
import PageHeader from '@/components/common/PageHeader.vue'
import PageSection from '@/components/common/PageSection.vue'
import PageToolbar from '@/components/common/PageToolbar.vue'
import SectionTabs from '@/components/common/SectionTabs.vue'
import StateBanner from '@/components/common/StateBanner.vue'
import UnsavedChangesDialog from '@/components/common/UnsavedChangesDialog.vue'
import GpuAdmissionPanel from '@/components/proxy/GpuAdmissionPanel.vue'
import NetworkAccessPanel from '@/components/proxy/NetworkAccessPanel.vue'
import ProxyAccessKeys from '@/components/proxy/ProxyAccessKeys.vue'
import ProxyActivityPanel from '@/components/proxy/ProxyActivityPanel.vue'
import ProxyCaptureToggle from '@/components/proxy/ProxyCaptureToggle.vue'
import ProxyOverviewPanel from '@/components/proxy/ProxyOverviewPanel.vue'
import ProxyRequestsPanel from '@/components/proxy/ProxyRequestsPanel.vue'
import { MAX_ENGINE_LOG_LINES } from '@/utils/proxyLogLevels'
import { vramEstimatePairs } from '@/utils/vramAdmission'

const tabs = [
  { id: 'overview', label: 'Overview', icon: 'pi-chart-bar' },
  { id: 'requests', label: 'Requests', icon: 'pi-list' },
  { id: 'events', label: 'Events', icon: 'pi-bolt' },
  { id: 'keys', label: 'Access keys', icon: 'pi-key' },
  { id: 'settings', label: 'Settings & routing', icon: 'pi-sliders-h' },
]

const activeTab = ref('overview')
const requests = ref([])
const selectedRequest = ref(null)
const overviewRequests = ref([])
const observedAt = ref(Date.now() / 1000)
const observedSettings = ref({})
const summary = ref({})
const metricSemantics = ref({})
const requestTotal = ref(0)
const requestOffset = ref(0)
const requestsLoading = ref(false)
const requestsError = ref('')
const requestFilters = reactive({
  search: '',
  model: '',
  key_id: '',
  state: '',
  status: '',
  outcome: '',
})
const requestStates = [
  'received',
  'queued',
  'loading',
  'forwarding',
  'counting',
  'streaming',
  'completed',
  'rejected',
  'failed',
  'cancelled',
]
const performanceFields = [
  {
    name: 'max_queue_size',
    label: 'Maximum queued requests',
    hint: 'Excess requests receive HTTP 429.',
  },
  {
    name: 'stream_idle_timeout_seconds',
    label: 'Stream inactivity timeout (seconds)',
    step: 0.1,
    min: 0.1,
    placeholder: 'Unlimited',
    hint: 'Maximum wait between upstream chunks.',
  },
  {
    name: 'pool_timeout_seconds',
    label: 'Connection pool wait (seconds)',
    step: 0.1,
    min: 0.1,
    hint: 'Bounds the wait for an upstream connection.',
  },
  {
    name: 'max_connections',
    label: 'Maximum upstream connections',
    hint: 'Shared across running engines.',
  },
  {
    name: 'max_keepalive_connections',
    label: 'Idle keepalive connections',
    hint: 'Must not exceed total connections.',
  },
  {
    name: 'history_retention_days',
    label: 'Request history retention (days)',
    hint: 'Metadata only; older entries are removed automatically.',
  },
  {
    name: 'history_max_rows',
    label: 'Maximum retained request records',
    min: 100,
    hint: 'A second bound on history disk usage.',
  },
  {
    name: 'default_output_tokens',
    label: 'Reserved output tokens when unspecified',
    hint: 'Applied to text requests made with token-budgeted keys.',
  },
]

const toast = useToast()
const confirm = useConfirm()
const loading = ref(false)
const saving = ref(false)
const lifecycleBusy = ref(false)
const loaded = ref(false)
const error = ref('')
const restartRequired = ref(false)
const keys = ref([])
const running = ref([])
const activity = ref([])
const catalog = ref([])
const vramSnapshot = ref({})
const network = ref({})
const configuredBackend = ref('')
const activeBackend = ref('')
const openLogs = ref('')
const logs = reactive({})
const logSeq = reactive({})
const aliasRows = ref([])
const vramRows = ref([])
let logSource = null
let logOpenToken = 0
let timer
let requestSequence = 0
let requestController = null
const baseline = ref('')

const tabItems = computed(() =>
  tabs.map((tab) =>
    tab.id === 'requests' && summary.value.active
      ? { ...tab, count: `${summary.value.active} active` }
      : tab,
  ),
)
const liveRunning = computed(() =>
  running.value.filter((item) => !['failed', 'stopped'].includes(item.state)),
)
const draft = reactive({
  max_queue_size: 100,
  stream_idle_timeout_seconds: null,
  pool_timeout_seconds: 10,
  max_connections: 200,
  max_keepalive_connections: 40,
  history_retention_days: 7,
  history_max_rows: 100000,
  default_output_tokens: 4096,
  capture_request_payloads: false,
  proxy_port: 2000,
  idle_ttl_seconds: null,
  max_body_bytes: null,
  allow_anonymous: true,
  max_loaded_models: null,
  global_concurrency: null,
  per_model_concurrency: null,
  queue_timeout_seconds: 30,
  request_timeout_seconds: null,
  readiness_timeout_seconds: 2700,
  drain_timeout_seconds: 30,
  default_model: null,
  vram_admission: 'off',
  vram_safety_margin_mib: 1024,
})

const statusLabel = computed(() =>
  activeBackend.value ? `Proxy active · ${activeBackend.value}` : 'Proxy not running',
)
const statusSeverity = computed(() => (activeBackend.value ? 'success' : 'danger'))

function setInspectedRequest(request) {
  if (request) {
    selectedRequest.value = request
    return
  }
  selectedRequest.value = null
  restoredRequestId.value = ''
}

async function resetStats() {
  try {
    await axios.post('/api/proxy/stats/reset')
    await refresh({ quiet: true })
  } catch (err) {
    reportFailure('Resetting stats failed', err)
  }
}

async function loadRequests(reset = false) {
  if (reset) requestOffset.value = 0
  requestController?.abort()
  requestController = new AbortController()
  const signal = requestController.signal
  const sequence = ++requestSequence
  requestsLoading.value = true
  try {
    const params = {
      limit: 100,
      offset: requestOffset.value,
      ...Object.fromEntries(Object.entries(requestFilters).filter(([, value]) => value !== '')),
    }
    const { data } = await axios.get('/api/proxy/requests', { params, signal })
    if (sequence !== requestSequence || signal.aborted) return
    requests.value = data.requests || []
    requestTotal.value = data.total || 0
    requestsError.value = ''
    if (selectedRequest.value)
      selectedRequest.value =
        requests.value.find((request) => request.id === selectedRequest.value.id) ||
        selectedRequest.value
  } catch (err) {
    if (sequence === requestSequence && !signal.aborted)
      requestsError.value = err?.response?.data?.detail || err.message
  } finally {
    if (sequence === requestSequence) requestsLoading.value = false
  }
}
function changeRequestPage(delta) {
  requestOffset.value = Math.max(0, requestOffset.value + delta)
  loadRequests()
}
watch(
  () => activeTab.value,
  (tab) => {
    if (tab === 'requests' && loaded.value) loadRequests()
  },
)

function inspectRequests(model = '', failed = false) {
  Object.assign(requestFilters, {
    search: '',
    model,
    key_id: '',
    state: '',
    outcome: failed ? 'errors' : '',
    status: '',
  })
  activeTab.value = 'requests'
  loadRequests(true)
}
function inspectOutcome(outcome) {
  Object.assign(requestFilters, {
    search: '',
    model: '',
    key_id: '',
    state: '',
    outcome,
    status: '',
  })
  activeTab.value = 'requests'
  loadRequests(true)
}
function resetRequestFilters() {
  Object.assign(requestFilters, {
    search: '',
    model: '',
    key_id: '',
    state: '',
    outcome: '',
    status: '',
  })
  loadRequests(true)
}
function inspectFailures(model = '') {
  inspectRequests(model, true)
}

function aliasPairs() {
  const aliases = {}
  for (const row of aliasRows.value) {
    const targets = row.targets
      .split(',')
      .map((item) => item.trim())
      .filter(Boolean)
    if (row.alias.trim() && targets.length) aliases[row.alias.trim()] = targets
  }
  return aliases
}
function vramPairs() {
  return vramEstimatePairs(vramRows.value)
}
function snapshot() {
  return JSON.stringify({ settings: { ...draft }, aliases: aliasPairs(), vram: vramPairs() })
}
const dirty = computed(() => loaded.value && snapshot() !== baseline.value)

const { leavePromptVisible, finishLeave, confirmDiscard } = useDraftGuard(() => dirty.value)
async function saveBeforeLeaving() {
  await saveSettings()
  if (!dirty.value && !error.value) finishLeave(true)
}
const lastUpdated = ref(0)
const offline = ref(navigator.onLine === false)
let pendingRefresh = null
let refreshController = null
let disposed = false
let failures = 0
async function refresh(options = {}) {
  if (disposed) return
  if (pendingRefresh) {
    if (!options.discardDraft) return pendingRefresh
    await pendingRefresh
  }
  if (disposed) return
  refreshController = new AbortController()
  pendingRefresh = performRefresh(options, refreshController.signal)
  try {
    await pendingRefresh
  } finally {
    pendingRefresh = null
  }
}
async function performRefresh({ quiet = false, discardDraft = false } = {}, signal) {
  if (!quiet) loading.value = true
  try {
    const [{ data }, keyResponse] = await Promise.all([
      axios.get('/api/proxy', { signal }),
      axios.get('/api/inference/keys', { signal }),
    ])
    if (signal.aborted || disposed) return
    const preserveDraft = dirty.value && !discardDraft
    if (!preserveDraft) {
      Object.assign(
        draft,
        Object.fromEntries(
          Object.keys(draft).map((key) => [key, data.settings?.[key] ?? draft[key]]),
        ),
      )
      aliasRows.value = Object.entries(data.settings?.routing_aliases || {}).map(
        ([alias, targets]) => ({
          alias,
          targets: (Array.isArray(targets) ? targets : [targets]).join(', '),
        }),
      )
      vramRows.value = Object.entries(data.settings?.vram_estimates || {}).map(([model, mib]) => ({
        model,
        mib,
      }))
      baseline.value = snapshot()
    }
    if (
      activeTab.value === 'requests' &&
      (requestOffset.value || Object.values(requestFilters).some(Boolean))
    )
      await loadRequests()
    else {
      requests.value = data.requests || []
      requestTotal.value = data.request_total ?? requests.value.length
    }
    summary.value = data.summary || {}
    metricSemantics.value = data.metric_semantics || {}
    overviewRequests.value = data.requests || []
    observedAt.value = Date.now() / 1000
    observedSettings.value = data.settings || {}
    if (selectedRequest.value)
      selectedRequest.value =
        requests.value.find((request) => request.id === selectedRequest.value.id) ||
        selectedRequest.value
    running.value = data.running || []
    activity.value = data.activity || []
    catalog.value = data.catalog || []
    vramSnapshot.value = data.vram || {}
    network.value = data.network || {}
    configuredBackend.value = data.configured_backend || data.settings?.backend || ''
    activeBackend.value = data.active_backend || ''
    if (data.restart_required !== undefined) restartRequired.value = Boolean(data.restart_required)
    keys.value = keyResponse.data?.keys || []
    error.value = ''
    lastUpdated.value = Date.now()
    failures = 0
    loaded.value = true
    if (restoredRequestId.value && !selectedRequest.value) await restoreRequest()
    if (openLogs.value && !running.value.some((row) => row.model === openLogs.value)) {
      openLogs.value = ''
      logOpenToken += 1
      closeLogStream()
    }
  } catch (err) {
    if (signal.aborted || disposed) return
    failures++
    error.value = err?.response?.data?.detail || err.message || 'Could not load proxy status.'
  } finally {
    loading.value = false
  }
}

async function resetDraft() {
  if (await confirmDiscard()) await refresh({ quiet: true, discardDraft: true })
}

async function saveSettings() {
  const rows = aliasRows.value.filter((row) => row.alias.trim())
  if (new Set(rows.map((row) => row.alias.trim().toLowerCase())).size !== rows.length) {
    error.value = 'Alias names must be unique.'
    return
  }
  saving.value = true
  try {
    const { data } = await axios.put(
      '/api/proxy/settings',
      Object.fromEntries(
        Object.entries({
          ...draft,
          vram_safety_margin_mib: Number.isFinite(Number(draft.vram_safety_margin_mib))
            ? Number(draft.vram_safety_margin_mib)
            : 1024,
          routing_aliases: aliasPairs(),
          vram_estimates: vramPairs(),
        }).map(([key, value]) => [key, value === '' ? null : value]),
      ),
    )
    restartRequired.value = Boolean(data.restart_required)
    toast.add({
      severity: 'success',
      summary: 'Proxy settings saved',
      detail: data.applied_live_fields?.length
        ? 'Routing and access changes applied live. Existing streams are preserved.'
        : undefined,
      life: 3500,
    })
    await refresh({ quiet: true, discardDraft: true })
  } catch (err) {
    error.value = err?.response?.data?.detail || err.message
  } finally {
    saving.value = false
  }
}

function reportFailure(summary, err) {
  toast.add({
    severity: 'error',
    summary,
    detail: err?.response?.data?.detail || err?.message || 'Request failed',
    life: 5000,
  })
}

async function loadLogs(model, token = 0) {
  const { data } = await axios.get(`/api/proxy/models/${encodeURIComponent(model)}/logs`)
  if (token && token !== logOpenToken) return
  logs[model] = Array.isArray(data?.lines) ? data.lines : []
  logSeq[model] = Number(data?.sequence) || 0
}

function closeLogStream() {
  if (!logSource) return
  logSource.close()
  logSource = null
}

function applyEngineLog(model, line, seq) {
  if (typeof line !== 'string') return
  const buffer = logs[model] || (logs[model] = [])
  const current = Number(logSeq[model]) || 0
  if (!Number.isFinite(seq) || seq < current) return
  if (seq === current && buffer.length) {
    // A terminal carriage-return redraw reuses the ring sequence so the
    // loading/progress line updates in place.
    buffer[buffer.length - 1] = line
  } else {
    buffer.push(line)
    if (buffer.length > MAX_ENGINE_LOG_LINES) {
      buffer.splice(0, buffer.length - MAX_ENGINE_LOG_LINES)
    }
  }
  logSeq[model] = seq
}

function openLogStream(model) {
  closeLogStream()
  if (typeof EventSource === 'undefined') return
  const cursor = Number(logSeq[model]) || 0
  const source = new EventSource(
    `/api/proxy/models/${encodeURIComponent(model)}/logs/stream?last_event_id=${cursor}`,
  )
  source.addEventListener('log', (event) => {
    if (openLogs.value !== model) return
    try {
      const payload = JSON.parse(event.data)
      applyEngineLog(model, payload.line, Number(payload.seq))
    } catch (_) {
      // Ignore malformed frames; EventSource keeps the stream alive.
    }
  })
  logSource = source
}

async function toggleLogs(model) {
  if (openLogs.value === model) {
    openLogs.value = ''
    logOpenToken += 1
    closeLogStream()
    return
  }
  logOpenToken += 1
  const token = logOpenToken
  closeLogStream()
  openLogs.value = model
  logs[model] = []
  logSeq[model] = 0
  try {
    await loadLogs(model, token)
  } catch (err) {
    reportFailure('Loading engine logs failed', err)
    return
  }
  if (token !== logOpenToken || openLogs.value !== model) return
  openLogStream(model)
}

function confirmClearLogs(model) {
  requireSingleConfirmation(confirm, {
    header: 'Clear captured logs?',
    message: 'The in-memory log ring is emptied. New engine output keeps streaming.',
    icon: 'pi pi-trash',
    acceptClass: 'p-button-danger',
    accept: async () => {
      try {
        await axios.delete(`/api/proxy/models/${encodeURIComponent(model)}/logs`)
        logs[model] = []
      } catch (err) {
        reportFailure('Clearing logs failed', err)
      }
    },
  })
}

function removeFailedEngine(item) {
  const model = item?.model
  if (!model) return
  requireSingleConfirmation(confirm, {
    header: 'Remove failed engine instance?',
    message: `Clear “${model}” and its captured failure status? The model configuration and files are not changed.`,
    icon: 'pi pi-trash',
    acceptClass: 'p-button-danger',
    accept: async () => {
      try {
        await axios.delete(`/api/proxy/models/${encodeURIComponent(model)}/instance`)
        if (openLogs.value === model) {
          openLogs.value = ''
          logOpenToken += 1
          closeLogStream()
        }
        delete logs[model]
        delete logSeq[model]
        await refresh({ quiet: true })
        toast.add({
          severity: 'success',
          summary: 'Failed engine removed',
          detail: `${model} status cleared.`,
          life: 3000,
        })
      } catch (err) {
        reportFailure(`Removing “${model}” failed`, err)
      }
    },
  })
}

function downloadLogs(model, lines = null) {
  const body = Array.isArray(lines) ? lines.join('\n') : (logs[model] || []).join('\n')
  const blob = new Blob([body], { type: 'text/plain' })
  const link = document.createElement('a')
  link.href = URL.createObjectURL(blob)
  link.download = `${model.replace(/[^a-z0-9._-]+/gi, '_')}.log`
  link.click()
  URL.revokeObjectURL(link.href)
}

async function lifecycle(action) {
  lifecycleBusy.value = true
  try {
    await axios.post(`/api/proxy/lifecycle/${action}`)
    restartRequired.value = false
    toast.add({
      severity: 'success',
      summary: `Proxy ${action === 'apply' ? 'restarted with new settings' : action}`,
      life: 2500,
    })
    await refresh({ quiet: true })
  } catch (err) {
    toast.add({
      severity: 'error',
      summary: `Proxy ${action} failed`,
      detail: err?.response?.data?.detail || err.message,
      life: 5000,
    })
  } finally {
    lifecycleBusy.value = false
  }
}
function confirmRestart() {
  requireSingleConfirmation(confirm, {
    header: 'Restart proxy and apply settings?',
    message:
      'Running engines are drained and relaunched so new limits and timeouts take effect. In-flight requests finish first.',
    icon: 'pi pi-refresh',
    acceptClass: 'p-button-warning',
    accept: () => lifecycle('apply'),
  })
}
function confirmStop() {
  requireSingleConfirmation(confirm, {
    header: 'Stop the proxy?',
    message:
      'Inference clients cannot connect until it starts again. Running engines are drained and terminated.',
    icon: 'pi pi-exclamation-triangle',
    acceptClass: 'p-button-danger',
    accept: () => lifecycle('stop'),
  })
}

const restoredRequestId = ref('')
async function restoreRequest() {
  const id = restoredRequestId.value
  if (!id) {
    selectedRequest.value = null
    return
  }
  const found = requests.value.find((request) => request.id === id)
  if (found) {
    selectedRequest.value = found
    return
  }
  try {
    const { data } = await axios.get('/api/proxy/requests', {
      params: { search: id, limit: 100, offset: 0 },
    })
    if (disposed || restoredRequestId.value !== id) return
    selectedRequest.value = (data.requests || []).find((request) => request.id === id) || null
    if (!selectedRequest.value)
      requestsError.value = 'This request is no longer in retained history.'
  } catch {
    if (!disposed)
      requestsError.value = 'Could not restore request details. Retry refreshing this view.'
  }
}
const queryBindings = {
  request: {
    get: () => selectedRequest.value?.id || restoredRequestId.value,
    set: (v) => {
      restoredRequestId.value = v
      selectedRequest.value = null
    },
    parse: (v) => (/^[a-zA-Z0-9_-]{1,128}$/.test(v) ? v : ''),
    default: '',
  },
  tab: {
    history: 'push',
    get: () => activeTab.value,
    set: (v) => {
      activeTab.value = v
    },
    parse: (v) => (tabs.some((t) => t.id === v) ? v : 'overview'),
    default: 'overview',
  },
  offset: {
    get: () => requestOffset.value,
    set: (v) => {
      requestOffset.value = v
    },
    parse: (v) => (/^\d+$/.test(v) ? Math.min(1000000, Math.floor(Number(v) / 100) * 100) : 0),
    default: 0,
  },
}
for (const key of Object.keys(requestFilters))
  queryBindings[key] = {
    get: () => requestFilters[key],
    set: (v) => {
      requestFilters[key] = v
    },
    parse: (v) => v.slice(0, 200),
    default: '',
  }
useQueryState(queryBindings, () => {
  if (loaded.value) {
    if (activeTab.value === 'requests') void loadRequests().then(restoreRequest)
    else void restoreRequest()
  }
})
async function tick() {
  clearTimeout(timer)
  offline.value = navigator.onLine === false
  if (!document.hidden && !offline.value) await refresh({ quiet: loaded.value })
  if (!disposed) {
    clearTimeout(timer)
    timer = setTimeout(tick, Math.min(60000, 5000 * 2 ** Math.min(failures, 4)))
  }
}
function resumeReads() {
  if (!document.hidden) void tick()
}
onMounted(() => {
  void tick()
  window.addEventListener('online', resumeReads)
  window.addEventListener('offline', resumeReads)
  document.addEventListener('visibilitychange', resumeReads)
})
onUnmounted(() => {
  disposed = true
  refreshController?.abort()
  requestController?.abort()
  requestSequence++
  clearTimeout(timer)
  window.removeEventListener('online', resumeReads)
  window.removeEventListener('offline', resumeReads)
  document.removeEventListener('visibilitychange', resumeReads)
  logOpenToken += 1
  closeLogStream()
})
</script>

<style scoped>
/* The proxy workspace owns an internal scroll region so the tab strip and the
   footer stay fixed while a long request table scrolls locally. This is the
   documented page-height exception; keep it on this page only. */
:global(.layout-wrapper:has(.proxy-page)) {
  height: 100dvh;
  min-height: 0;
  grid-template-rows: auto minmax(0, 1fr) auto;
}

:global(.layout-main:has(.proxy-page)) {
  max-width: none;
  min-height: 0;
  overflow: hidden;
}

.proxy-page {
  height: 100%;
  min-height: 0;
  gap: 0.65rem;
  overflow: hidden;
}

.proxy-panels {
  display: grid;
  flex: 1;
  align-content: start;
  gap: 0.75rem;
  min-height: 0;
  padding-bottom: 0.5rem;
  overflow: auto;
  scroll-padding-bottom: 1rem;
}

.proxy-panels > * {
  min-width: 0;
}

@media (max-width: 900px) {
  .proxy-panels {
    padding-bottom: calc(var(--bottom-nav-height) + 0.5rem);
    scroll-padding-bottom: var(--bottom-nav-height);
  }
}

/* Labels wrap their control for the accessible name; the shared form grid
   still owns spacing and the label/hint placement. */
.page-form > label {
  display: grid;
  gap: 0.35rem;
  align-content: start;
}

.page-form > label .field {
  font-weight: 400;
}

.inline-control {
  display: flex;
  align-items: flex-start;
  gap: 0.75rem;
}

.inline-control label {
  display: grid;
  gap: 0.2rem;
}

.alias-list {
  display: grid;
  gap: 0.6rem;
}

.alias-row {
  display: grid;
  grid-template-columns: 1fr 2fr auto;
  gap: 0.6rem;
  align-items: center;
}

@media (max-width: 800px) {
  .alias-row {
    grid-template-columns: 1fr;
  }

  .proxy-page :deep(.page-header) {
    position: relative;
    padding-right: 2.25rem;
  }

  .proxy-page :deep(.page-header__actions) {
    position: absolute;
    top: 0;
    right: 0;
    width: auto;
  }

  .proxy-page :deep(.page-header__meta) {
    gap: 0.3rem;
  }
}
</style>
