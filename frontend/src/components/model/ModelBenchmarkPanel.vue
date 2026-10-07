<template>
  <section class="benchmark-panel" aria-labelledby="benchmark-heading">
    <div class="benchmark-heading">
      <div>
        <div id="benchmark-heading" class="section-label">Local benchmark</div>
        <p>
          Compare this model using the saved configuration revision. Results stay on this machine.
        </p>
      </div>
      <button type="button" :disabled="busy" @click="refresh">Refresh results</button>
    </div>

    <label>
      Prompt
      <textarea v-model="prompt" rows="2" maxlength="1000" :disabled="busy" />
    </label>
    <label>
      Maximum generated tokens
      <input v-model.number="maxTokens" type="number" min="1" max="512" :disabled="busy" />
    </label>
    <button type="button" :disabled="busy || !running" @click="run">
      {{
        running
          ? runningBenchmark
            ? 'Benchmark running…'
            : 'Run benchmark'
          : 'Start the model to benchmark'
      }}
    </button>

    <p v-if="error" class="benchmark-error" role="alert">{{ error }}</p>
    <p v-if="status" role="status">{{ status }}</p>

    <div v-if="results.length" class="benchmark-results">
      <article v-for="result in results" :key="result.id">
        <header>
          <strong>{{ formatDate(result.created_at) }}</strong>
          <code>{{ result.config_fingerprint }}</code>
        </header>
        <dl>
          <div>
            <dt>First token</dt>
            <dd>{{ result.time_to_first_token_ms.toFixed(0) }} ms</dd>
          </div>
          <div>
            <dt>Generation</dt>
            <dd>{{ metric(result.tokens_per_second, ' tok/s') }}</dd>
          </div>
          <div>
            <dt>Generated</dt>
            <dd>{{ metric(result.completion_tokens, ' tokens') }}</dd>
          </div>
          <div>
            <dt>Observed GPU memory</dt>
            <dd>{{ memory(result.peak_observed_gpu_memory_bytes) }}</dd>
          </div>
        </dl>
        <p v-if="result.output_preview" class="output-preview">{{ result.output_preview }}</p>
      </article>
    </div>
  </section>
</template>

<script setup>
import { computed, onMounted, ref, watch } from 'vue'
import { listModelBenchmarks, runModelBenchmark } from '@/api/configuration'

const props = defineProps({
  modelId: { type: String, required: true },
  running: { type: Boolean, default: false },
})

const prompt = ref('Reply with a short description of the sky.')
const maxTokens = ref(64)
const results = ref([])
const loading = ref(false)
const runningBenchmark = ref(false)
const error = ref('')
const status = ref('')
const busy = computed(() => loading.value || runningBenchmark.value)

function formatDate(seconds) {
  return new Date(Number(seconds) * 1000).toLocaleString()
}

function metric(value, suffix) {
  return value == null ? 'Unavailable' : `${Number(value).toFixed(2)}${suffix}`
}

function memory(bytes) {
  if (bytes == null) return 'Unavailable'
  return `${(Number(bytes) / 1024 ** 3).toFixed(2)} GiB total GPU use`
}

async function refresh() {
  if (busy.value || !props.modelId) return
  loading.value = true
  error.value = ''
  try {
    results.value = await listModelBenchmarks(props.modelId)
  } catch (requestError) {
    error.value = requestError?.response?.data?.detail || 'Benchmark results could not be loaded.'
  } finally {
    loading.value = false
  }
}

async function run() {
  if (busy.value || !props.running) return
  runningBenchmark.value = true
  error.value = ''
  status.value = ''
  try {
    const result = await runModelBenchmark(props.modelId, prompt.value.trim(), maxTokens.value)
    results.value = [result, ...results.value.filter((item) => item.id !== result.id)]
    status.value = 'Benchmark completed and saved locally.'
  } catch (requestError) {
    error.value = requestError?.response?.data?.detail || 'The benchmark did not complete.'
  } finally {
    runningBenchmark.value = false
  }
}

watch(() => props.modelId, refresh)
onMounted(refresh)
</script>

<style scoped>
.benchmark-panel,
.benchmark-results,
.benchmark-panel label {
  display: flex;
  flex-direction: column;
  gap: 0.6rem;
}

.benchmark-panel {
  padding: 1rem;
  border: 1px solid var(--border-primary);
  border-radius: var(--radius-lg);
  background: var(--bg-surface);
}

.benchmark-heading {
  display: flex;
  justify-content: space-between;
  gap: 1rem;
  flex-wrap: wrap;
}

.benchmark-heading p,
.benchmark-results p {
  margin: 0;
}

.benchmark-panel button,
.benchmark-panel textarea,
.benchmark-panel input {
  font: inherit;
  color: var(--text-primary);
  background: var(--bg-tertiary);
  border: 1px solid var(--border-secondary);
  border-radius: var(--radius-md);
  padding: 0.5rem 0.75rem;
}

.benchmark-results article {
  padding: 0.75rem;
  border: 1px solid var(--border-secondary);
  border-radius: var(--radius-md);
}

.benchmark-results header,
.benchmark-results dl {
  display: flex;
  gap: 0.75rem;
  flex-wrap: wrap;
  justify-content: space-between;
}

.benchmark-results dl div {
  min-width: 8rem;
}

.benchmark-results dt {
  color: var(--text-secondary);
}

.benchmark-results dd {
  margin: 0;
}

.output-preview {
  margin-top: 0.75rem !important;
  overflow-wrap: anywhere;
}

.benchmark-error {
  color: var(--status-warning);
}
</style>
