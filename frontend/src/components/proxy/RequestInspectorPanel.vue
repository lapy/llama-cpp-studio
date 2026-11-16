<template>
  <Dialog
    v-model:visible="dialogVisible"
    header="Request details"
    modal
    :style="{ width: '62rem', maxWidth: '95vw' }"
  >
    <div v-if="request" class="request-inspector-summary">
      <Tag :value="request.state || 'completed'" :severity="requestSeverity(request)" /><code
        >{{ request.method }} {{ request.path }}</code
      >
      <label class="inspector-field-toggle"
        ><input v-model="allRequestFields" type="checkbox" />Show unavailable fields</label
      >
      <div ref="exportRoot" class="request-export" @focusout="onExportFocusOut">
        <button
          ref="exportToggle"
          type="button"
          class="request-export-toggle"
          aria-haspopup="menu"
          :aria-expanded="exportOpen ? 'true' : 'false'"
          aria-controls="request-export-menu"
          @click="toggleExportMenu"
        >
          <i class="pi pi-download" aria-hidden="true" />Export
        </button>
        <div v-if="exportOpen" class="request-export-popover">
          <div
            id="request-export-menu"
            ref="exportMenu"
            class="request-export-menu"
            role="menu"
            aria-label="Export captured request"
            @keydown="onExportMenuKeydown"
          >
            <button
              type="button"
              role="menuitem"
              class="request-export-item"
              data-export-action="curl"
              :title="
                capturedPayload
                  ? 'Copy a cURL command including the captured headers and body'
                  : 'Copy a cURL command with the captured method and URL; headers and body were not captured'
              "
              @click="copyAsCurl"
            >
              <i class="pi pi-copy" aria-hidden="true" />Copy as cURL
            </button>
            <button
              type="button"
              role="menuitem"
              class="request-export-item"
              data-export-action="json"
              title="Download the raw captured request as pretty-printed JSON"
              @click="downloadJson"
            >
              <i class="pi pi-file" aria-hidden="true" />Download JSON
            </button>
            <button
              type="button"
              role="menuitem"
              class="request-export-item"
              data-export-action="har"
              title="Download a single-entry HAR 1.2 archive"
              @click="downloadHar"
            >
              <i class="pi pi-download" aria-hidden="true" />Download HAR
            </button>
          </div>
          <p v-if="!capturedPayload" class="request-export-note" role="note">
            Headers and bodies are not retained, so the cURL command contains the method and URL
            only.
          </p>
        </div>
      </div>
      <button
        v-if="request.replayable"
        type="button"
        class="request-replay-button"
        data-request-replay
        :disabled="replaying"
        @click="replayRequest"
      >
        <i :class="[replaying ? 'pi pi-spin pi-spinner' : 'pi pi-replay', '']" aria-hidden="true" />
        {{ replaying ? 'Replaying…' : 'Replay request' }}
      </button>
    </div>
    <Message v-if="replayError" severity="error" :closable="false">{{ replayError }}</Message>
    <div v-if="replayResult" class="request-replay-result" data-replay-result>
      <h3>
        Replay result
        <Tag
          :value="String(replayResult.status)"
          :severity="replayResult.status < 400 ? 'success' : 'danger'"
        />
      </h3>
      <pre class="request-captured-body">{{ replayResult.body || '(empty body)' }}</pre>
      <p v-if="replayResult.body_truncated" class="hint">Response body truncated for display.</p>
    </div>
    <section v-if="request.captured_body" class="request-detail-section" data-captured-body>
      <h3>Captured request body</h3>
      <pre class="request-captured-body">{{ request.captured_body }}</pre>
      <p v-if="request.captured_body_truncated" class="hint">Captured body truncated at 256 KiB.</p>
      <p v-if="(request.captured_redacted_headers || []).length" class="hint">
        Redacted headers: {{ (request.captured_redacted_headers || []).join(', ') }}
      </p>
    </section>
    <div class="request-inspector">
      <section
        v-for="section in visibleRequestDetailSections"
        :key="section.title"
        class="request-detail-section"
      >
        <h3>{{ section.title }}</h3>
        <dl class="request-details">
          <div v-for="field in section.fields" :key="field.name">
            <dt>{{ field.label }}</dt>
            <dd>{{ requestDetail(field.name) }}</dd>
          </div>
        </dl>
      </section>
    </div>
    <p class="hint">
      Usage marked “reservation” is a conservative charge retained because the engine did not report
      final usage. Unknown token counts remain blank. Request headers and bodies are retained only
      in memory when payload capture is enabled; otherwise only metadata is kept.
    </p>
  </Dialog>
</template>

<script setup>
import { computed, nextTick, ref, watch } from 'vue'
import Dialog from 'primevue/dialog'
import Tag from 'primevue/tag'
import Message from 'primevue/message'
import { useToast } from 'primevue/usetoast'
import axios from 'axios'
import { copyText } from '@/utils/clipboard'
import {
  buildCurlCommand,
  buildHarExport,
  buildJsonExport,
  downloadTextFile,
  exportFilename,
  hasCapturedPayload,
} from '@/utils/proxyRequestExport'
import {
  formatBuildIdentity,
  formatTime,
  metric,
  requestSeverity,
} from '@/utils/proxyRequestPresentation'

const props = defineProps({
  request: {
    type: Object,
    default: null,
  },
  visible: {
    type: Boolean,
    default: false,
  },
})
const emit = defineEmits(['update:visible'])

const dialogVisible = computed({
  get: () => props.visible,
  set: (value) => emit('update:visible', value),
})

const toast = useToast()
const appVersion = typeof __APP_VERSION__ !== 'undefined' ? __APP_VERSION__ : '1.0.0'
const capturedPayload = computed(() => hasCapturedPayload(props.request))
const replaying = ref(false)
const replayError = ref('')
const replayResult = ref(null)
const exportOpen = ref(false)
const exportRoot = ref(null)
const exportToggle = ref(null)
const exportMenu = ref(null)

function currentOrigin() {
  return typeof window === 'undefined' ? '' : window.location?.origin || ''
}

async function toggleExportMenu() {
  if (exportOpen.value) {
    closeExportMenu()
    return
  }
  exportOpen.value = true
  await nextTick()
  exportMenu.value?.querySelector('[role="menuitem"]')?.focus()
}

function closeExportMenu({ restoreFocus = false } = {}) {
  if (!exportOpen.value) return
  exportOpen.value = false
  if (restoreFocus) nextTick(() => exportToggle.value?.focus())
}

function onExportFocusOut(event) {
  const next = event.relatedTarget
  if (next && exportRoot.value?.contains(next)) return
  closeExportMenu()
}

function exportMenuItems() {
  return Array.from(exportMenu.value?.querySelectorAll('[role="menuitem"]') || [])
}

function onExportMenuKeydown(event) {
  const items = exportMenuItems()
  if (!items.length) return
  if (event.key === 'Escape') {
    event.preventDefault()
    closeExportMenu({ restoreFocus: true })
    return
  }
  const current = items.indexOf(document.activeElement)
  let next = null
  if (event.key === 'ArrowDown') next = current < 0 ? 0 : (current + 1) % items.length
  else if (event.key === 'ArrowUp') next = current <= 0 ? items.length - 1 : current - 1
  else if (event.key === 'Home') next = 0
  else if (event.key === 'End') next = items.length - 1
  if (next === null) return
  event.preventDefault()
  items[next].focus()
}

async function replayRequest() {
  if (replaying.value || !props.request?.id) return
  replaying.value = true
  replayError.value = ''
  replayResult.value = null
  try {
    const { data } = await axios.post(
      `/api/proxy/requests/${encodeURIComponent(props.request.id)}/replay`,
    )
    replayResult.value = data
  } catch (error) {
    replayError.value =
      error?.response?.data?.detail?.message ||
      error?.response?.data?.detail ||
      error.message ||
      'Replay failed'
  } finally {
    replaying.value = false
  }
}

async function copyAsCurl() {
  closeExportMenu({ restoreFocus: true })
  try {
    await copyText(buildCurlCommand(props.request, { origin: currentOrigin() }))
    toast.add({
      severity: 'success',
      summary: 'cURL command copied',
      detail: capturedPayload.value
        ? 'Includes the captured headers and body.'
        : 'Headers and body were not captured; the command contains the method and URL.',
      life: 3000,
    })
  } catch (error) {
    toast.add({
      severity: 'error',
      summary: 'Could not copy the cURL command',
      detail: error?.message || 'Clipboard access is unavailable.',
      life: 4000,
    })
  }
}

function downloadJson() {
  closeExportMenu({ restoreFocus: true })
  const filename = exportFilename(props.request?.id, 'json')
  try {
    downloadTextFile(buildJsonExport(props.request), filename, 'application/json')
    toast.add({
      severity: 'success',
      summary: 'JSON downloaded',
      detail: filename,
      life: 3000,
    })
  } catch (error) {
    toast.add({
      severity: 'error',
      summary: 'Could not download the JSON',
      detail: error?.message || 'The browser could not save the file.',
      life: 4000,
    })
  }
}

function downloadHar() {
  closeExportMenu({ restoreFocus: true })
  const filename = exportFilename(props.request?.id, 'har')
  try {
    const content = JSON.stringify(
      buildHarExport(props.request, { origin: currentOrigin(), appVersion }),
      null,
      2,
    )
    downloadTextFile(content, filename, 'application/json')
    toast.add({
      severity: 'success',
      summary: 'HAR downloaded',
      detail: filename,
      life: 3000,
    })
  } catch (error) {
    toast.add({
      severity: 'error',
      summary: 'Could not download the HAR',
      detail: error?.message || 'The browser could not save the file.',
      life: 4000,
    })
  }
}

const requestDetailFields = [
  ['id', 'Request ID'],
  ['state', 'State'],
  ['status', 'Outcome status'],
  ['http_status', 'Delivered HTTP status'],
  ['key_name', 'Client'],
  ['key_id', 'Key ID'],
  ['requested_model', 'Requested model / alias'],
  ['model', 'Resolved model'],
  ['routing_revision', 'Routing revision'],
  ['engine', 'Engine'],
  ['engine_revision', 'Launch revision'],
  ['engine_build', 'Installed engine build identity'],
  ['protocol', 'Protocol adapter'],
  ['requested_stream', 'Client requested streaming'],
  ['response_transport', 'Response transport'],
  ['nonstream_response', 'Non-streaming response'],
  ['incremental_output', 'Incremental output events'],
  ['observation_schema_version', 'Observation schema'],
  ['pid', 'Process ID'],
  ['method', 'Method'],
  ['path', 'Path'],
  ['started_at', 'Started'],
  ['completed_at', 'Completed'],
  ['queue_position', 'Queue position'],
  ['token_count_ms', 'Input token counting'],
  ['input_reserved_source', 'Input accounting source'],
  ['token_count_endpoint', 'Engine counting endpoint'],
  ['admission_ms', 'Key admission'],
  ['accounting_mode', 'Accounting mode'],
  ['queue_ms', 'Queue wait'],
  ['loading_ms', 'Model acquisition'],
  ['first_byte_ms', 'Time to first byte'],
  ['ttft_ms', 'Proxy TTFT'],
  ['first_visible_text_ms', 'Time to first visible text'],
  ['time_to_first_visible_text_ms', 'Time to first visible text (stable name)'],
  ['first_output_ms', 'First output event'],
  ['last_output_ms', 'Last output event'],
  ['output_interval_ms', 'Proxy output interval'],
  ['terminal_ms', 'Terminal event'],
  ['generation_ms', 'Proxy output interval (legacy generation_ms)'],
  ['response_streaming_ms', 'Response streaming time'],
  ['proxy_decode_tokens_per_second', 'Proxy decode rate (estimate)'],
  ['decode_rate_quality', 'Decode rate quality'],
  ['request_effective_tokens_per_second', 'Effective request throughput'],
  ['output_tokens_per_second', 'Output tokens per second (legacy alias)'],
  ['latency_ms', 'Total duration'],
  ['input_tokens', 'Input tokens (normalized alias)'],
  ['input_tokens_total', 'Logical input tokens'],
  ['input_tokens_cached', 'Cached input tokens'],
  ['input_tokens_cache_write', 'Input cache write tokens'],
  ['input_tokens_uncached', 'Logical uncached input tokens'],
  ['input_tokens_processed', 'Engine processed prompt tokens'],
  ['input_tokens_preflight', 'Preflight engine count'],
  ['input_tokens_provenance', 'Input token provenance'],
  ['output_tokens', 'Output tokens (normalized alias)'],
  ['output_tokens_total', 'Output tokens total'],
  ['output_tokens_reasoning', 'Reasoning output tokens'],
  ['native_timing', 'Engine timing fields'],
  ['engine_metrics', 'Engine metric fields'],
  ['observed_input_tokens', 'Partial input tokens observed'],
  ['observed_output_tokens', 'Partial output tokens observed'],
  ['usage_status', 'Usage authority'],
  ['usage_complete', 'Complete protocol usage'],
  ['usage_source', 'Usage source'],
  ['observation_reasons', 'Observation reasons'],
  ['retained_charge_reason', 'Retained charge reason'],
  ['terminal_kind', 'Protocol terminal state'],
  ['input_reserved', 'Input token reservation'],
  ['output_reserved', 'Output token reservation'],
  ['input_bytes', 'Request bytes'],
  ['output_bytes', 'Response bytes'],
  ['error', 'Failure reason'],
  ['accounting_note', 'Accounting note'],
].map(([name, label]) => ({ name, label }))

const allRequestFields = ref(false)
watch(
  () => props.request?.id,
  () => {
    allRequestFields.value = false
    closeExportMenu()
  },
)
watch(
  () => props.visible,
  (visible) => {
    if (!visible) closeExportMenu()
  },
)

watch(
  () => props.request?.id,
  () => {
    replayResult.value = null
    replayError.value = ''
  },
)

const requestDetailSections = [
  {
    title: 'Request & outcome',
    names: [
      'id',
      'state',
      'status',
      'http_status',
      'method',
      'path',
      'error',
      'requested_stream',
      'response_transport',
      'nonstream_response',
      'incremental_output',
      'observation_schema_version',
    ],
  },
  {
    title: 'Client & routing',
    names: [
      'key_name',
      'key_id',
      'requested_model',
      'model',
      'routing_revision',
      'engine',
      'engine_revision',
      'engine_build',
      'protocol',
      'pid',
    ],
  },
  {
    title: 'Timing',
    names: [
      'started_at',
      'completed_at',
      'queue_position',
      'token_count_ms',
      'admission_ms',
      'queue_ms',
      'loading_ms',
      'first_byte_ms',
      'ttft_ms',
      'first_visible_text_ms',
      'time_to_first_visible_text_ms',
      'first_output_ms',
      'last_output_ms',
      'output_interval_ms',
      'terminal_ms',
      'generation_ms',
      'response_streaming_ms',
      'latency_ms',
      'proxy_decode_tokens_per_second',
      'decode_rate_quality',
      'request_effective_tokens_per_second',
      'output_tokens_per_second',
    ],
  },
  {
    title: 'Usage & accounting',
    names: [
      'input_reserved_source',
      'token_count_endpoint',
      'accounting_mode',
      'input_tokens',
      'input_tokens_total',
      'input_tokens_cached',
      'input_tokens_cache_write',
      'input_tokens_uncached',
      'input_tokens_processed',
      'input_tokens_preflight',
      'input_tokens_provenance',
      'output_tokens',
      'output_tokens_total',
      'output_tokens_reasoning',
      'native_timing',
      'engine_metrics',
      'observed_input_tokens',
      'observed_output_tokens',
      'usage_status',
      'usage_complete',
      'usage_source',
      'observation_reasons',
      'retained_charge_reason',
      'terminal_kind',
      'input_reserved',
      'output_reserved',
      'input_bytes',
      'output_bytes',
      'accounting_note',
    ],
  },
].map((section) => ({
  title: section.title,
  fields: section.names.map((name) => requestDetailFields.find((field) => field.name === name)),
}))

const visibleRequestDetailSections = computed(() => {
  const required = new Set([
    'id',
    'state',
    'status',
    'method',
    'path',
    'key_name',
    'model',
    'input_tokens',
    'output_tokens',
    'usage_source',
    'latency_ms',
    'ttft_ms',
  ])
  return requestDetailSections.map((section) => ({
    ...section,
    fields: section.fields.filter((field) => {
      const value = props.request?.[field.name]
      return allRequestFields.value || required.has(field.name) || (value != null && value !== '')
    }),
  }))
})

function requestDetail(name) {
  const value = props.request?.[name]
  if (value == null || value === '') return '—'
  if (name.endsWith('_at')) return formatTime(value)
  if (name.endsWith('_ms')) return metric(value)
  if (name.endsWith('_tokens_per_second')) return `${Number(value).toFixed(2)} tok/s`
  if (name === 'engine_build') return formatBuildIdentity(value)
  if (name === 'observation_reasons') return Array.isArray(value) ? value.join(', ') : value
  if (typeof value === 'object') return JSON.stringify(value)
  if (typeof value === 'boolean') return value ? 'Yes' : 'No'
  return value
}
</script>

<style scoped>
.request-inspector-summary {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 0.75rem;
  padding-bottom: 1rem;
  overflow-wrap: anywhere;
}
.request-inspector {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 1rem;
}
.request-detail-section {
  min-width: 0;
  padding: 1rem;
  border: 1px solid var(--border-primary);
  border-radius: var(--radius-md);
}
.request-detail-section h3 {
  margin: 0 0 0.75rem;
  font-size: 0.85rem;
}
.request-details {
  font-size: var(--text-xs);
}
.request-details dt {
  color: var(--text-secondary);
}
.request-details div {
  display: grid;
  grid-template-columns: 1fr 2fr;
  gap: 1rem;
  padding: 0.4rem;
  overflow-wrap: anywhere;
}
.request-details dd {
  margin: 0;
}
.inspector-field-toggle {
  display: inline-flex;
  align-items: center;
  gap: 0.4rem;
  margin-left: auto;
  color: var(--text-secondary);
  font-size: var(--text-xs);
  white-space: nowrap;
}
.inspector-field-toggle input {
  accent-color: var(--accent-primary);
}
.request-export {
  position: relative;
}
.request-export-toggle {
  display: inline-flex;
  align-items: center;
  gap: 0.4rem;
  padding: 0.4rem 0.7rem;
  border: 1px solid var(--border-primary);
  border-radius: var(--radius-sm);
  background: var(--bg-card);
  color: var(--text-primary);
  font-size: var(--text-xs);
  cursor: pointer;
}
.request-export-toggle:hover {
  background: var(--hover-bg);
  border-color: var(--border-secondary);
}
.request-export-toggle:focus-visible,
.request-export-item:focus-visible {
  outline: 2px solid var(--accent-primary);
  outline-offset: 2px;
}
.request-export-popover {
  position: absolute;
  top: calc(100% + 0.35rem);
  right: 0;
  z-index: 20;
  min-width: 14rem;
  padding: 0.3rem;
  border: 1px solid var(--border-primary);
  border-radius: var(--radius-md);
  background: var(--bg-card);
  box-shadow: var(--shadow-md);
}
.request-export-menu {
  display: grid;
}
.request-export-item {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.5rem 0.6rem;
  border: 0;
  border-radius: var(--radius-sm);
  background: transparent;
  color: var(--text-primary);
  font-size: var(--text-xs);
  text-align: left;
  cursor: pointer;
}
.request-export-item:hover {
  background: var(--hover-bg);
}
.request-export-note {
  max-width: 17rem;
  margin: 0.35rem 0.6rem 0.25rem;
  color: var(--text-secondary);
  font-size: var(--text-xs);
}

@media (max-width: 600px) {
  .request-inspector {
    grid-template-columns: 1fr;
  }
  .request-details div {
    grid-template-columns: 1fr;
    gap: 0.15rem;
  }
}
.request-replay-button {
  display: inline-flex;
  align-items: center;
  gap: 0.35rem;
  padding: 0.3rem 0.6rem;
  border: 1px solid var(--border-primary);
  border-radius: var(--radius-md);
  background: var(--bg-surface);
  color: var(--text-primary);
  font-size: 0.78rem;
  cursor: pointer;
}

.request-replay-button:hover:not(:disabled) {
  border-color: var(--accent-cyan);
}

.request-replay-button:disabled {
  opacity: 0.65;
  cursor: progress;
}

.request-replay-result {
  margin: 0.75rem 0;
  padding: 0.65rem;
  border: 1px solid var(--border-primary);
  border-radius: var(--radius-md);
  background: var(--bg-surface);
}

.request-replay-result h3 {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  margin: 0 0 0.4rem;
  font-size: 0.85rem;
}

.request-captured-body {
  max-height: 18rem;
  overflow: auto;
  margin: 0.35rem 0;
  padding: 0.5rem;
  border-radius: var(--radius-sm);
  background: var(--bg-elevated);
  color: var(--text-primary);
  font-size: 0.75rem;
  white-space: pre-wrap;
  overflow-wrap: anywhere;
}
</style>
