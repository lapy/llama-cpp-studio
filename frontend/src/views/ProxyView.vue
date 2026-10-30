<template>
  <div class="proxy-page page-shell page-shell--relaxed">
    <PageHeader title="Proxy" description="Inference gateway: serving backend, access keys, and native engine supervision.">
      <template #meta>
        <Tag :value="statusLabel" :severity="statusSeverity" />
        <Tag v-if="running.length" :value="`${running.length} running`" severity="info" />
        <Tag v-if="activeBackend && activeBackend !== configuredBackend" value="backend mismatch" severity="warn" />
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

    <Message v-if="error" severity="error" :closable="true" @close="error = ''">{{ error }}</Message>
    <Message v-if="!activeBackend && configuredBackend" severity="warn" :closable="false">
      The configured proxy ({{ configuredBackend }}) is not running — inference clients cannot connect.
      <Button label="Start proxy" size="small" severity="warn" :loading="lifecycleBusy" @click="lifecycle('start')" />
    </Message>
    <Message v-else-if="activeBackend && activeBackend !== configuredBackend" severity="warn" :closable="false">
      The active proxy is “{{ activeBackend }}”, but configuration names “{{ configuredBackend }}”.
      Switching stops the running proxy, then starts the configured one.
      <Button label="Switch backend" size="small" severity="warn" :loading="switching" @click="confirmSwitch" />
    </Message>
    <Message v-else-if="restartRequired" severity="warn" :closable="false">
      Saved settings changed since the proxy last (re)started. Use “Restart &amp; apply” to activate them.
    </Message>

    <LoadingState v-if="loading && !loaded" message="Loading proxy state…" />

    <template v-else>
      <section class="proxy-card" aria-labelledby="proxy-backend-heading">
        <div class="section-head">
          <div>
            <h2 id="proxy-backend-heading">Backend &amp; serving limits</h2>
            <p>The native Studio proxy enforces these limits live; a restart rebinds the port and timeouts.</p>
          </div>
          <dl class="backend-state">
            <div><dt>Configured</dt><dd>{{ configuredBackend || '—' }}</dd></div>
            <div><dt>Active</dt><dd :class="{ 'state-alert': !activeBackend }">{{ activeBackend || 'not running' }}</dd></div>
          </dl>
        </div>
        <div class="settings-grid">
          <label>Backend
            <select v-model="draft.backend" class="field">
              <option value="studio">Studio native proxy</option>
              <option value="llama_swap">llama-swap (legacy)</option>
            </select>
          </label>
          <label>Inference port
            <input v-model.number="draft.proxy_port" class="field" type="number" min="1" max="65535" />
            <small class="hint">Applies on the next proxy restart.</small>
          </label>
          <label>Idle unload (seconds)
            <input v-model.number="draft.idle_ttl_seconds" class="field" type="number" min="0" placeholder="Disabled" />
          </label>
          <label>Maximum request size (bytes)
            <input v-model.number="draft.max_body_bytes" class="field" type="number" min="1" placeholder="268435456" />
          </label>
          <label>Maximum loaded models
            <input v-model.number="draft.max_loaded_models" class="field" type="number" min="1" placeholder="Unlimited" />
          </label>
          <label>Global concurrent requests
            <input v-model.number="draft.global_concurrency" class="field" type="number" min="1" placeholder="Unlimited" />
          </label>
          <label>Concurrent requests per model
            <input v-model.number="draft.per_model_concurrency" class="field" type="number" min="1" placeholder="Unlimited" />
          </label>
          <label>Queue timeout (seconds)
            <input v-model.number="draft.queue_timeout_seconds" class="field" type="number" min="0.1" step="0.1" />
          </label>
          <label>Request timeout (seconds)
            <input v-model.number="draft.request_timeout_seconds" class="field" type="number" min="0.1" placeholder="Unlimited" />
          </label>
          <label>Engine readiness timeout (seconds)
            <input v-model.number="draft.readiness_timeout_seconds" class="field" type="number" min="1" />
          </label>
          <label>Graceful drain timeout (seconds)
            <input v-model.number="draft.drain_timeout_seconds" class="field" type="number" min="1" />
          </label>
        </div>
        <div class="inline-control">
          <ToggleSwitch v-model="draft.allow_anonymous" input-id="allow-anonymous" />
          <label for="allow-anonymous">Allow anonymous inference while no access key is configured
            <small class="hint">Once any key exists, anonymous traffic is rejected regardless of this setting.</small>
          </label>
        </div>
        <div class="runtime-actions">
          <span class="save-group">
            <Button label="Save settings" :disabled="!dirty" :loading="saving" @click="saveSettings" />
            <Button v-if="dirty" label="Discard changes" size="small" severity="secondary" text @click="resetDraft" />
          </span>
          <span class="lifecycle-group">
            <Button label="Start" icon="pi pi-play" size="small" severity="success" outlined :disabled="Boolean(activeBackend)" @click="lifecycle('start')" />
            <Button label="Restart &amp; apply" icon="pi pi-replay" size="small" severity="secondary" outlined :loading="lifecycleBusy" @click="confirmRestart" />
            <Button label="Stop" icon="pi pi-stop-circle" size="small" severity="danger" outlined :disabled="!activeBackend" @click="confirmStop" />
          </span>
        </div>
      </section>

      <section class="proxy-card" aria-labelledby="proxy-routing-heading">
        <div class="section-head">
          <div>
            <h2 id="proxy-routing-heading">Routing</h2>
            <p>Default model for requests without a model field, plus aliases. An alias with several targets is round-robined.</p>
          </div>
        </div>
        <div class="settings-grid">
          <label>Default model
            <select v-model="draft.default_model" class="field">
              <option :value="null">Automatic — single running model</option>
              <option v-for="model in catalog" :key="model.id" :value="model.id">{{ model.id }}</option>
            </select>
          </label>
        </div>
        <div class="alias-list">
          <div v-for="(row, index) in aliasRows" :key="index" class="alias-row">
            <input v-model="row.alias" class="field" placeholder="Alias, for example fast" aria-label="Alias name" />
            <input v-model="row.targets" class="field" placeholder="Target model IDs, comma-separated" aria-label="Alias targets" />
            <Button icon="pi pi-times" text rounded severity="danger" size="small" aria-label="Remove alias" v-tooltip.top="'Remove alias'" @click="aliasRows.splice(index, 1)" />
          </div>
          <div>
            <Button label="Add alias" icon="pi pi-plus" size="small" severity="secondary" outlined @click="aliasRows.push({ alias: '', targets: '' })" />
          </div>
        </div>
      </section>

      <section class="proxy-card" aria-labelledby="proxy-keys-heading">
        <div class="section-head">
          <div>
            <h2 id="proxy-keys-heading">Client access keys</h2>
            <p>Keys can carry an expiry, model restrictions, and a per-minute rate limit. The secret is shown once.</p>
          </div>
          <span class="count">{{ keys.length }} active</span>
        </div>
        <div class="settings-grid key-options">
          <label>Key name
            <input ref="keyNameInput" v-model="keyName" class="field" maxlength="120" placeholder="for example Production" @keyup.enter="createKey" />
          </label>
          <label>Expires at
            <input v-model="keyExpires" class="field" type="datetime-local" />
          </label>
          <label>Requests per minute
            <input v-model.number="keyRateLimit" class="field" type="number" min="1" placeholder="Unlimited" />
          </label>
          <label>Allowed model IDs
            <input v-model="keyModels" class="field" placeholder="All models, or comma-separated IDs" />
          </label>
        </div>
        <div>
          <Button label="Create key" icon="pi pi-key" :disabled="!keyName.trim()" :loading="creating" @click="createKey" />
        </div>
        <div v-if="createdKey" class="secret" role="status">
          <strong>{{ createdKeyLabel }}</strong>
          <code>{{ createdKey }}</code>
          <Button label="Copy" icon="pi pi-copy" size="small" severity="secondary" outlined @click="copyKey" />
        </div>
        <div v-if="keys.length" class="key-list">
          <div v-for="key in keys" :key="key.id" class="key-row">
            <div class="key-summary">
              <span class="key-title"><strong>{{ key.name }}</strong><Tag :value="keyStatus(key)" :severity="keySeverity(key)" /></span>
              <code>{{ key.prefix }}…</code>
              <small>{{ key.models?.length ? key.models.join(', ') : 'All models' }} · {{ key.rate_limit_per_minute ? `${key.rate_limit_per_minute} req/min` : 'no rate limit' }} · {{ key.expires_at ? `expires ${formatDate(key.expires_at)}` : 'no expiry' }} · {{ key.last_used_at ? `last used ${formatAgo(key.last_used_at)}` : 'never used' }}</small>
            </div>
            <span class="key-actions">
              <Button label="Rotate" size="small" severity="warn" text @click="confirmRotate(key)" />
              <Button label="Revoke" size="small" severity="danger" text @click="confirmRevoke(key)" />
            </span>
          </div>
        </div>
        <EmptyState v-else icon="pi pi-key" title="No access keys yet" description="Without a key the inference port follows the anonymous-access policy above. Create a key to lock it down.">
          <Button label="Create the first key" icon="pi pi-key" size="small" @click="focusKeyForm" />
        </EmptyState>
      </section>

      <section class="proxy-card" aria-labelledby="proxy-catalog-heading">
        <div class="section-head">
          <div>
            <h2 id="proxy-catalog-heading">Catalog</h2>
            <p>Preload starts a model now; Warmup adds a tiny completion so caches and samplers are hot before real traffic.</p>
          </div>
        </div>
        <div v-if="catalog.length" class="catalog-list">
          <div v-for="model in catalog" :key="model.id" class="catalog-row">
            <span class="catalog-id"><strong>{{ model.id }}</strong><small>{{ model.engine }}<template v-if="model.aliases?.length"> · aliases: {{ model.aliases.join(', ') }}</template></small></span>
            <Tag :value="stateFor(model.id)" :severity="stateSeverity(model.id)" />
            <span class="catalog-actions">
              <Button label="Preload" icon="pi pi-download" size="small" severity="secondary" outlined :loading="preloading === model.id" @click="preload(model.id, false)" />
              <Button label="Warmup" icon="pi pi-bolt" size="small" severity="secondary" outlined :loading="preloading === model.id" @click="preload(model.id, true)" />
            </span>
          </div>
        </div>
        <EmptyState v-else icon="pi pi-box" title="Nothing to preload yet" description="Models appear here once they compile to a proxy entry with an active engine." />
      </section>

      <section class="proxy-card" aria-labelledby="proxy-running-heading">
        <div class="section-head">
          <div>
            <h2 id="proxy-running-heading">Running engines</h2>
            <p>Process generations supervised by the native proxy. Logs stream live while a panel is open.</p>
          </div>
          <span class="count">{{ running.length }} running</span>
        </div>
        <div v-if="running.length" class="runtime-list">
          <article v-for="item in running" :key="`${item.model}-${item.pid}`" class="runtime">
            <div class="runtime-head">
              <div><strong>{{ item.model }}</strong><span>{{ item.engine || 'engine' }} · PID {{ item.pid }} · port {{ item.port }}</span></div>
              <Tag :value="item.state" :severity="item.ready ? 'success' : 'warn'" />
            </div>
            <ProgressBar v-if="!item.ready && item.loading_progress != null" :value="Math.round(item.loading_progress)" :show-value="false" aria-label="Model loading progress" />
            <dl class="metrics">
              <div><dt>In-flight</dt><dd>{{ item.inflight }}</dd></div>
              <div><dt>Requests</dt><dd>{{ item.request_count || 0 }}</dd></div>
              <div><dt>Errors</dt><dd :class="{ 'state-error': item.error_count }">{{ item.error_count || 0 }}</dd></div>
              <div><dt>Latency p50 / p95</dt><dd>{{ metric(item.p50_latency_ms) }} · {{ metric(item.p95_latency_ms) }}</dd></div>
              <div><dt>Tokens out</dt><dd>{{ item.output_tokens || 0 }} ({{ rate(item.output_tokens_per_second) }} tok/s)</dd></div>
              <div><dt>Traffic</dt><dd>{{ formatBytes(item.input_bytes) }} in · {{ formatBytes(item.output_bytes) }} out</dd></div>
              <div><dt>Revision</dt><dd><code>{{ shortRevision(item.revision) }}</code></dd></div>
              <div><dt>Started</dt><dd>{{ formatTime(item.created_at) }}</dd></div>
            </dl>
            <div class="runtime-actions">
              <Button :label="openLogs === item.model ? 'Hide logs' : 'View logs'" :icon="openLogs === item.model ? 'pi pi-eye-slash' : 'pi pi-list'" size="small" severity="secondary" outlined :aria-expanded="openLogs === item.model" @click="toggleLogs(item.model)" />
              <Button label="Reload" icon="pi pi-replay" size="small" severity="secondary" outlined @click="reload(item.model)" />
              <Button label="Unload" icon="pi pi-eject" size="small" severity="danger" outlined @click="confirmUnload(item)" />
            </div>
            <template v-if="openLogs === item.model">
              <div class="log-toolbar">
                <span class="log-search">
                  <i class="pi pi-search" aria-hidden="true" />
                  <input v-model="logSearch" class="field" placeholder="Search log lines…" aria-label="Search logs" @input="debouncedLoadLogs(item.model)" />
                </span>
                <select v-model="logLevel" class="field" aria-label="Log severity" @change="loadLogs(item.model)">
                  <option value="">All severities</option>
                  <option value="warn">Warnings &amp; errors</option>
                  <option value="error">Errors only</option>
                </select>
                <span class="log-meta">{{ (logs[item.model] || []).length }} shown · live</span>
                <Button label="Download" icon="pi pi-download" size="small" severity="secondary" text @click="downloadLogs(item.model)" />
                <Button label="Clear" icon="pi pi-trash" size="small" severity="danger" text @click="confirmClearLogs(item.model)" />
              </div>
              <pre class="logs" tabindex="0" aria-label="Engine log output">{{ logs[item.model]?.join('\n') || 'No engine output matches the current filter.' }}</pre>
            </template>
          </article>
        </div>
        <EmptyState v-else icon="pi pi-play" title="No engine processes running" description="A process appears after the first request or preload for a model." />
      </section>

      <section class="proxy-card" aria-labelledby="proxy-network-heading">
        <div class="section-head">
          <div>
            <h2 id="proxy-network-heading">Network</h2>
            <p>Read-only view of how the inference port is bound. Listen host comes from STUDIO_PROXY_LISTEN; terminate TLS at a reverse proxy.</p>
          </div>
        </div>
        <dl class="network">
          <div><dt>Listen host</dt><dd><code>{{ network.listen_host || '—' }}</code> <small>{{ network.listen_source }}</small></dd></div>
          <div><dt>Port</dt><dd>{{ network.port ?? '—' }}</dd></div>
          <div><dt>Public URL</dt><dd>{{ network.public_url || 'Same origin (default)' }}</dd></div>
          <div><dt>TLS</dt><dd>{{ network.tls_note }}</dd></div>
        </dl>
      </section>

      <section class="proxy-card" aria-labelledby="proxy-activity-heading">
        <div class="section-head">
          <div>
            <h2 id="proxy-activity-heading">Recent activity</h2>
            <p>Lifecycle events from the native proxy, newest first. Refreshes every few seconds.</p>
          </div>
        </div>
        <ul v-if="activity.length" class="activity-list">
          <li v-for="(item, index) in activityDesc" :key="`${item.type}-${item.created_at}-${index}`">
            <span class="activity-time">{{ formatClock(item.created_at) }}</span>
            <Tag :value="item.type" :severity="activitySeverity(item.type)" />
            <span class="activity-text">{{ activityText(item) }}</span>
          </li>
        </ul>
        <p v-else class="empty">No proxy activity recorded in this session.</p>
      </section>
    </template>
  </div>
</template>

<script setup>
import { computed, onMounted, onUnmounted, reactive, ref } from 'vue'
import axios from 'axios'
import { useConfirm } from 'primevue/useconfirm'
import { useToast } from 'primevue/usetoast'
import Button from 'primevue/button'
import Tag from 'primevue/tag'
import Message from 'primevue/message'
import ProgressBar from 'primevue/progressbar'
import ToggleSwitch from 'primevue/toggleswitch'
import PageHeader from '@/components/common/PageHeader.vue'
import EmptyState from '@/components/common/EmptyState.vue'
import LoadingState from '@/components/common/LoadingState.vue'
import { requireSingleConfirmation } from '@/composables/singleConfirm'

const toast = useToast()
const confirm = useConfirm()
const loading = ref(false), saving = ref(false), creating = ref(false), switching = ref(false), lifecycleBusy = ref(false)
const loaded = ref(false)
const error = ref(''), restartRequired = ref(false), keys = ref([]), running = ref([]), activity = ref([])
const catalog = ref([]), network = ref({})
const configuredBackend = ref(''), activeBackend = ref('')
const keyName = ref(''), createdKey = ref(''), createdKeyLabel = ref('Copy this key now — it will not be shown again.'), openLogs = ref('')
const logs = reactive({})
const logSearch = ref(''), logLevel = ref(''), preloading = ref(''), keyNameInput = ref(null)
const keyExpires = ref(''), keyRateLimit = ref(null), keyModels = ref('')
const aliasRows = ref([])
const draft = reactive({
  backend: 'llama_swap', proxy_port: 2000, idle_ttl_seconds: null, max_body_bytes: null,
  allow_anonymous: true, max_loaded_models: null, global_concurrency: null, per_model_concurrency: null,
  queue_timeout_seconds: 30, request_timeout_seconds: null, readiness_timeout_seconds: 2700,
  drain_timeout_seconds: 30, default_model: null,
})
let timer, logTimer
const baseline = ref('')

const statusLabel = computed(() => (activeBackend.value ? `Proxy active · ${activeBackend.value}` : 'Proxy not running'))
const statusSeverity = computed(() => (activeBackend.value ? 'success' : 'danger'))

const activityDesc = computed(() => [...activity.value].reverse())
function activitySeverity(type) {
  return { ready: 'success', starting: 'info', loading_progress: 'info', drain_timeout: 'warn', stopped: 'secondary' }[type] || 'secondary'
}
function activityText(item) {
  const bits = []
  if (item.model) bits.push(item.model)
  if (item.type === 'loading_progress' && item.loading_progress != null) bits.push(`${Math.round(item.loading_progress)}% loaded`)
  if (item.type === 'stopped' && item.draining) bits.push('while draining')
  if (item.pid) bits.push(`pid ${item.pid}`)
  return bits.join(' · ') || item.type
}

function stateFor(modelId) {
  const row = running.value.find((item) => item.model === modelId)
  return row ? row.state : 'stopped'
}
function stateSeverity(modelId) {
  const state = stateFor(modelId)
  return state === 'ready' ? 'success' : state === 'loading' ? 'warn' : 'secondary'
}
function keyStatus(key) {
  if (key.expires_at && new Date(key.expires_at) <= new Date()) return 'Expired'
  return 'Active'
}
function keySeverity(key) {
  return keyStatus(key) === 'Expired' ? 'danger' : 'success'
}

function aliasPairs() {
  const aliases = {}
  for (const row of aliasRows.value) {
    const targets = row.targets.split(',').map((item) => item.trim()).filter(Boolean)
    if (row.alias.trim() && targets.length) aliases[row.alias.trim()] = targets
  }
  return aliases
}
function snapshot() {
  return JSON.stringify({ settings: { ...draft }, aliases: aliasPairs() })
}
const dirty = computed(() => loaded.value && snapshot() !== baseline.value)

async function refresh({ quiet = false } = {}) {
  if (!quiet) loading.value = true
  try {
    const [{ data }, keyResponse] = await Promise.all([axios.get('/api/proxy'), axios.get('/api/inference/keys')])
    Object.assign(draft, data.settings || {})
    aliasRows.value = Object.entries(data.settings?.routing_aliases || {}).map(([alias, targets]) => ({
      alias,
      targets: (Array.isArray(targets) ? targets : [targets]).join(', '),
    }))
    running.value = data.running || []
    activity.value = data.activity || []
    catalog.value = data.catalog || []
    network.value = data.network || {}
    configuredBackend.value = data.configured_backend || data.settings?.backend || ''
    activeBackend.value = data.active_backend || ''
    keys.value = keyResponse.data?.keys || []
    error.value = ''
    loaded.value = true
    baseline.value = snapshot()
    if (openLogs.value && running.value.some((row) => row.model === openLogs.value)) await loadLogs(openLogs.value)
  } catch (err) {
    error.value = err?.response?.data?.detail || err.message || 'Could not load proxy status.'
  } finally {
    loading.value = false
  }
}

function resetDraft() {
  refresh({ quiet: true })
}

async function saveSettings() {
  const rows = aliasRows.value.filter((row) => row.alias.trim())
  if (new Set(rows.map((row) => row.alias.trim().toLowerCase())).size !== rows.length) {
    error.value = 'Alias names must be unique.'
    return
  }
  saving.value = true
  try {
    const { data } = await axios.put('/api/proxy/settings', Object.fromEntries(
      Object.entries({ ...draft, routing_aliases: aliasPairs() }).map(([key, value]) => [key, value === '' ? null : value]),
    ))
    restartRequired.value = Boolean(data.restart_required)
    toast.add({ severity: 'success', summary: 'Proxy settings saved', life: 2500 })
    await refresh({ quiet: true })
  } catch (err) {
    error.value = err?.response?.data?.detail || err.message
  } finally {
    saving.value = false
  }
}

function confirmSwitch() {
  requireSingleConfirmation(confirm, {
    header: `Switch proxy to ${draft.backend}`,
    message: 'The running proxy is stopped (engines drained and terminated), then the configured backend starts.',
    acceptClass: 'p-button-warning',
    accept: async () => {
      switching.value = true
      try {
        // Switch to the persisted configuration the banner describes, never
        // to an unsaved selection still sitting in the form.
        await axios.post('/api/proxy/switch', { backend: configuredBackend.value })
        restartRequired.value = false
        toast.add({ severity: 'success', summary: 'Backend switched', detail: configuredBackend.value, life: 2500 })
        await refresh({ quiet: true })
      } catch (err) {
        error.value = err?.response?.data?.detail || err.message
      } finally {
        switching.value = false
      }
    },
  })
}

async function createKey() {
  if (!keyName.value.trim()) return
  creating.value = true
  try {
    const { data } = await axios.post('/api/inference/keys', {
      name: keyName.value.trim(),
      expires_at: keyExpires.value ? new Date(keyExpires.value).toISOString() : null,
      models: keyModels.value.split(',').map((item) => item.trim()).filter(Boolean),
      scopes: ['inference', 'models:read'],
      rate_limit_per_minute: keyRateLimit.value || null,
    })
    createdKey.value = data.key
    createdKeyLabel.value = 'Key created. Copy it now — it will not be shown again.'
    keyName.value = ''
    keyExpires.value = ''; keyModels.value = ''; keyRateLimit.value = null
    await refresh({ quiet: true })
  } catch (err) {
    error.value = err?.response?.data?.detail || err.message
  } finally {
    creating.value = false
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

function confirmRevoke(key) {
  requireSingleConfirmation(confirm, {
    header: `Revoke “${key.name}”?`,
    message: 'Clients using this key stop working immediately.',
    icon: 'pi pi-exclamation-triangle',
    acceptClass: 'p-button-danger',
    accept: async () => {
      try {
        await axios.delete(`/api/inference/keys/${encodeURIComponent(key.id)}`)
        toast.add({ severity: 'success', summary: 'Key revoked', life: 2500 })
        await refresh({ quiet: true })
      } catch (err) {
        reportFailure(`Revoking “${key.name}” failed`, err)
      }
    },
  })
}

function confirmRotate(key) {
  requireSingleConfirmation(confirm, {
    header: `Rotate “${key.name}”?`,
    message: 'A new secret is issued and the old one stops working immediately. Update clients before rotating.',
    icon: 'pi pi-refresh',
    acceptClass: 'p-button-warning',
    accept: async () => {
      try {
        const { data } = await axios.post(`/api/inference/keys/${encodeURIComponent(key.id)}/rotate`)
        createdKey.value = data.key
        createdKeyLabel.value = 'Key rotated. Copy the new secret — the old one no longer works.'
        await refresh({ quiet: true })
      } catch (err) {
        reportFailure(`Rotating “${key.name}” failed`, err)
      }
    },
  })
}

async function copyKey() {
  try {
    await navigator.clipboard.writeText(createdKey.value)
    toast.add({ severity: 'success', summary: 'Key copied to clipboard', life: 2000 })
  } catch {
    toast.add({ severity: 'warn', summary: 'Clipboard unavailable — copy the key manually', life: 4000 })
  }
}

async function loadLogs(model) {
  const { data } = await axios.get(`/api/proxy/models/${encodeURIComponent(model)}/logs`, {
    params: { search: logSearch.value || undefined, level: logLevel.value || undefined },
  })
  logs[model] = data.lines || []
}
function debouncedLoadLogs(model) {
  clearTimeout(logTimer)
  logTimer = setTimeout(() => loadLogs(model), 250)
}
async function toggleLogs(model) {
  openLogs.value = openLogs.value === model ? '' : model
  logSearch.value = ''
  logLevel.value = ''
  if (openLogs.value) await loadLogs(model)
}

function confirmUnload(item) {
  const run = async () => {
    try {
      await axios.post(`/api/proxy/models/${encodeURIComponent(item.model)}/unload`)
      if (openLogs.value === item.model) openLogs.value = ''
      toast.add({ severity: 'success', summary: `Unloading ${item.model}`, life: 2500 })
      await refresh({ quiet: true })
    } catch (err) {
      reportFailure(`Unloading ${item.model} failed`, err)
    }
  }
  if (item.inflight) {
    requireSingleConfirmation(confirm, {
      header: `Unload “${item.model}”?`,
      message: `${item.inflight} in-flight request(s) are drained first, then the engine stops. Clients see brief latency during drain.`,
      icon: 'pi pi-exclamation-triangle',
      acceptClass: 'p-button-danger',
      accept: run,
    })
    return
  }
  run()
}

async function reload(model) {
  try {
    await axios.post(`/api/proxy/models/${encodeURIComponent(model)}/reload`)
    toast.add({ severity: 'success', summary: `Reloading ${model}`, life: 2500 })
    await refresh({ quiet: true })
  } catch (err) {
    reportFailure(`Reloading ${model} failed`, err)
  }
}

async function preload(model, warm) {
  preloading.value = model
  try {
    const { data } = await axios.post(`/api/proxy/models/${encodeURIComponent(model)}/preload`, null, { params: { warm } })
    toast.add({
      severity: warm && data.warmed === false ? 'warn' : 'success',
      summary: `Preload ${model}`,
      detail: warm ? (data.warmed ? 'Ready and warmed' : 'Ready — warmup request failed') : 'Ready',
      life: 3000,
    })
  } catch (err) {
    toast.add({ severity: 'error', summary: `Preload ${model} failed`, detail: err?.response?.data?.detail || err.message, life: 5000 })
  } finally {
    preloading.value = ''
    await refresh({ quiet: true })
  }
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

function downloadLogs(model) {
  const blob = new Blob([(logs[model] || []).join('\n')], { type: 'text/plain' })
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
    toast.add({ severity: 'success', summary: `Proxy ${action === 'apply' ? 'restarted with new settings' : action}`, life: 2500 })
    await refresh({ quiet: true })
  } catch (err) {
    toast.add({ severity: 'error', summary: `Proxy ${action} failed`, detail: err?.response?.data?.detail || err.message, life: 5000 })
  } finally {
    lifecycleBusy.value = false
  }
}
function confirmRestart() {
  requireSingleConfirmation(confirm, {
    header: 'Restart proxy and apply settings?',
    message: 'Running engines are drained and relaunched so new limits and timeouts take effect. In-flight requests finish first.',
    icon: 'pi pi-refresh',
    acceptClass: 'p-button-warning',
    accept: () => lifecycle('apply'),
  })
}
function confirmStop() {
  requireSingleConfirmation(confirm, {
    header: 'Stop the proxy?',
    message: 'Inference clients cannot connect until it starts again. Running engines are drained and terminated.',
    icon: 'pi pi-exclamation-triangle',
    acceptClass: 'p-button-danger',
    accept: () => lifecycle('stop'),
  })
}

function shortRevision(value) { return String(value || 'unknown').slice(0, 12) }
function formatTime(value) { return value ? new Date(value * 1000).toLocaleString() : 'Unknown' }
function formatClock(value) { return value ? new Date(value * 1000).toLocaleTimeString() : '' }
function formatDate(value) { return new Date(value).toLocaleString() }
function formatAgo(iso) {
  const seconds = Math.max(0, (Date.now() - new Date(iso).getTime()) / 1000)
  if (seconds < 60) return 'just now'
  if (seconds < 3600) return `${Math.floor(seconds / 60)} min ago`
  if (seconds < 86400) return `${Math.floor(seconds / 3600)} h ago`
  return `${Math.floor(seconds / 86400)} d ago`
}
function metric(value) { return value == null ? '—' : `${Number(value).toFixed(1)} ms` }
function rate(value) { return value == null ? '—' : Number(value).toFixed(1) }
function formatBytes(value) { const bytes = Number(value || 0); if (bytes < 1024) return `${bytes} B`; if (bytes < 1048576) return `${(bytes / 1024).toFixed(1)} KiB`; return `${(bytes / 1048576).toFixed(1)} MiB` }

function tick() {
  if (document.hidden) return
  refresh({ quiet: true })
}
function focusKeyForm() {
  keyNameInput.value?.focus()
}
onMounted(() => {
  refresh()
  timer = setInterval(tick, 5000)
})
onUnmounted(() => {
  clearInterval(timer)
  clearTimeout(logTimer)
})
</script>

<style scoped>
.proxy-card {
  padding: 1.25rem;
  border: 1px solid var(--border-primary);
  border-radius: var(--radius-lg);
  background: var(--bg-card);
  display: grid;
  gap: 1rem;
}
.section-head { display: flex; align-items: flex-start; justify-content: space-between; gap: 1rem; flex-wrap: wrap; }
.section-head h2 { margin: 0; font-size: 1.05rem; }
.section-head p, .empty, .count { color: var(--text-secondary); margin: .3rem 0 0; }
.count { margin-top: 0; white-space: nowrap; }
.backend-state { display: flex; gap: 1.4rem; margin: 0; font-size: .82rem; }
.backend-state div { display: grid; gap: .2rem; }
.backend-state dt { color: var(--text-secondary); }
.backend-state dd { margin: 0; font-variant-numeric: tabular-nums; }
.settings-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(14rem, 1fr)); gap: 1rem; }
.settings-grid label { display: grid; gap: .45rem; font-size: .82rem; color: var(--text-secondary); }
.field {
  width: 100%;
  padding: .65rem .7rem;
  border: 1px solid var(--border-primary);
  border-radius: var(--radius-md);
  background: var(--bg-surface);
  color: var(--text-primary);
  box-sizing: border-box;
}
.field:focus-visible { outline: 2px solid var(--accent-primary, var(--accent-blue)); outline-offset: 1px; }
.settings-grid small, .inline-control small { color: var(--text-secondary); font-weight: 400; }
.inline-control { display: flex; align-items: flex-start; gap: .75rem; }
.inline-control label { display: grid; gap: .2rem; font-size: .85rem; cursor: pointer; }
.runtime-actions { display: flex; align-items: center; justify-content: space-between; gap: 1rem; flex-wrap: wrap; }
.save-group, .lifecycle-group, .catalog-actions, .key-actions { display: flex; align-items: center; gap: .5rem; flex-wrap: wrap; }
.alias-list { display: grid; gap: .6rem; }
.alias-row { display: grid; grid-template-columns: 1fr 2fr auto; gap: .6rem; align-items: center; }
.secret { display: grid; gap: .7rem; padding: 1rem; border: 1px solid var(--status-warning); border-radius: var(--radius-md); justify-items: start; }
.secret code { overflow-wrap: anywhere; font-size: .85rem; }
.key-list, .runtime-list, .catalog-list { display: grid; gap: .7rem; }
.key-row { display: flex; align-items: center; justify-content: space-between; gap: 1rem; padding: .7rem 0; border-top: 1px solid var(--border-primary); flex-wrap: wrap; }
.key-summary { display: grid; gap: .25rem; min-width: 0; }
.key-summary code { font-size: .8rem; }
.key-summary small { color: var(--text-secondary); }
.key-title { display: flex; align-items: center; gap: .6rem; }
.catalog-row { display: flex; align-items: center; gap: 1rem; padding: .55rem 0; border-top: 1px solid var(--border-primary); flex-wrap: wrap; }
.catalog-id { display: grid; gap: .15rem; flex: 1; min-width: 0; }
.catalog-id small { color: var(--text-secondary); }
.runtime { padding: 1rem; border: 1px solid var(--border-primary); border-radius: var(--radius-md); display: grid; gap: .9rem; }
.runtime-head { display: flex; align-items: flex-start; justify-content: space-between; gap: 1rem; }
.runtime-head div { display: grid; gap: .25rem; min-width: 0; }
.runtime-head span:not(.p-tag) { color: var(--text-secondary); font-size: .82rem; }
.metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(9.5rem, 1fr)); gap: .8rem 2rem; margin: 0; font-size: .82rem; }
.metrics div { display: grid; gap: .2rem; }
.metrics dt { color: var(--text-secondary); }
.metrics dd { margin: 0; color: var(--text-primary); font-variant-numeric: tabular-nums; }
.state-error { color: var(--p-red-500, #f87171); }
.state-alert { color: var(--status-warning); }
.log-toolbar { display: flex; align-items: center; gap: .6rem; flex-wrap: wrap; }
.log-search { position: relative; flex: 1; min-width: 12rem; display: flex; align-items: center; }
.log-search i { position: absolute; left: .65rem; color: var(--text-secondary); font-size: .85rem; pointer-events: none; }
.log-search .field { padding-left: 2rem; }
.log-meta { color: var(--text-secondary); font-size: .8rem; margin-left: auto; white-space: nowrap; }
.logs { max-height: 22rem; overflow: auto; margin: 0; padding: 1rem; background: var(--bg-surface); border-radius: var(--radius-md); white-space: pre-wrap; font-size: .78rem; }
.activity-list { list-style: none; margin: 0; padding: 0; display: grid; gap: .35rem; max-height: 20rem; overflow: auto; }
.activity-list li { display: flex; align-items: center; gap: .7rem; font-size: .82rem; padding: .3rem 0; border-bottom: 1px dashed color-mix(in srgb, var(--border-primary) 55%, transparent); }
.activity-time { color: var(--text-secondary); font-variant-numeric: tabular-nums; white-space: nowrap; }
.activity-text { min-width: 0; overflow-wrap: anywhere; }
.network { display: grid; grid-template-columns: repeat(auto-fit, minmax(16rem, 1fr)); gap: 1rem; margin: 0; font-size: .85rem; }
.network div { display: grid; gap: .2rem; }
.network dt { color: var(--text-secondary); }
.network dd { margin: 0; }

@media (max-width: 800px) {
  .alias-row { grid-template-columns: 1fr; }
  .runtime-actions { align-items: stretch; }
}
</style>
