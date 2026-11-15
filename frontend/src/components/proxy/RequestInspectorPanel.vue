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
    </div>
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
      final usage. Unknown token counts remain blank. Request and response bodies are not retained.
    </p>
  </Dialog>
</template>

<script setup>
import { computed, ref, watch } from 'vue'
import Dialog from 'primevue/dialog'
import Tag from 'primevue/tag'
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

@media (max-width: 600px) {
  .request-inspector {
    grid-template-columns: 1fr;
  }
  .request-details div {
    grid-template-columns: 1fr;
    gap: 0.15rem;
  }
}
</style>
