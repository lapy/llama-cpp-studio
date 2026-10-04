<template>
  <Dialog
    :visible="visible"
    header="Connect"
    modal
    class="dialog-width-md"
    @update:visible="emit('update:visible', $event)"
  >
    <p class="connect-lead">{{ lead }}</p>
    <div class="connect-field">
      <span class="connect-label">API model ID</span>
      <code class="connect-value">{{ modelId || 'Unavailable until this model is registered with the proxy.' }}</code>
      <Button label="Copy" size="small" severity="secondary" outlined :disabled="!modelId" @click="copy(modelId)" />
    </div>
    <div class="connect-field">
      <label class="connect-label" for="connect-public-url">Public URL</label>
      <input
        id="connect-public-url"
        v-model="publicUrlDraft"
        class="connect-input"
        type="url"
        placeholder="http://127.0.0.1:2000"
        @change="savePublicUrl"
      />
    </div>
    <div class="connect-field">
      <span class="connect-label">Endpoint</span>
      <code class="connect-value">{{ endpoint }}</code>
      <Button label="Copy" size="small" severity="secondary" outlined @click="copy(endpoint)" />
    </div>
    <div class="connect-field">
      <span class="connect-label">Request</span>
      <pre class="connect-example">{{ requestExample }}</pre>
      <Button label="Copy request" size="small" severity="secondary" outlined @click="copy(requestExample)" />
    </div>
    <div class="connect-actions">
      <Button
        label="Send test"
        icon="pi pi-send"
        :loading="testing"
        :disabled="!canTest"
        @click="sendTest"
      />
      <span v-if="testNote" class="connect-note">{{ testNote }}</span>
    </div>
    <pre v-if="testResult" class="connect-result" role="status">{{ testResult }}</pre>
  </Dialog>
</template>

<script setup>
import { computed, ref, watch } from 'vue'
import Dialog from 'primevue/dialog'
import Button from 'primevue/button'
import { useToast } from 'primevue/usetoast'
import axios from 'axios'
import { audioInferenceModelId } from '@/composables/useAudioInferenceClient'
import { connectEndpoint, connectRequest, normalizePublicInferenceUrl } from '@/composables/connectTarget'

const props = defineProps({
  visible: { type: Boolean, default: false },
  model: { type: Object, default: null },
  proxyPort: { type: [Number, String], default: 2000 },
  publicInferenceUrl: { type: String, default: '' },
})

const emit = defineEmits(['update:visible', 'update:publicInferenceUrl'])
const toast = useToast()
const testing = ref(false)
const testResult = ref('')
const publicUrlDraft = ref('')

const modelId = computed(() => audioInferenceModelId(props.model, props.model?.config))
const example = computed(() => connectRequest(props.model))
const endpoint = computed(() => connectEndpoint(props.model, {
  publicUrl: publicUrlDraft.value,
  proxyPort: props.proxyPort,
}))
const requestExample = computed(() => JSON.stringify(example.value.body, null, 2))
const lead = computed(() => {
  if (example.value.kind === 'embeddings') return 'This model serves embeddings. The example calls /v1/embeddings.'
  if (example.value.kind === 'audio') return 'This model is served by audio.cpp. Run it from the Audio page.'
  return 'This model serves chat completions. The example is a short text request.'
})
const canTest = computed(() =>
  Boolean(modelId.value && props.model?.is_active && example.value.kind !== 'audio'),
)
const testNote = computed(() => {
  if (example.value.kind === 'audio') return 'Use the Audio page to run this model.'
  if (!props.model?.is_active) return 'Start the model before sending a test.'
  return 'The test is sent through Studio, not directly to the proxy port.'
})

watch(() => props.visible, (open) => {
  if (!open) return
  testResult.value = ''
  publicUrlDraft.value = props.publicInferenceUrl || ''
})

watch(() => props.publicInferenceUrl, (value) => {
  if (!props.visible) publicUrlDraft.value = value || ''
})

async function copy(text) {
  try {
    await navigator.clipboard.writeText(String(text || ''))
    toast.add({ severity: 'success', summary: 'Copied', life: 2000 })
  } catch {
    toast.add({ severity: 'warn', summary: 'Copy failed', detail: 'Select the text and copy it manually.', life: 3000 })
  }
}

async function savePublicUrl() {
  const normalized = normalizePublicInferenceUrl(publicUrlDraft.value)
  if (publicUrlDraft.value.trim() && !normalized) {
    toast.add({
      severity: 'warn',
      summary: 'URL not saved',
      detail: 'Use an http or https URL without a username or password.',
      life: 4000,
    })
    return
  }
  try {
    const { data } = await axios.put('/api/settings/inference', {
      public_inference_url: normalized,
    })
    publicUrlDraft.value = data?.public_inference_url || ''
    emit('update:publicInferenceUrl', publicUrlDraft.value)
  } catch (error) {
    toast.add({
      severity: 'error',
      summary: 'URL not saved',
      detail: error?.response?.data?.detail || error?.message || 'Could not save the public URL.',
      life: 4000,
    })
  }
}

async function sendTest() {
  testing.value = true
  testResult.value = ''
  try {
    const modelKey = props.model?.id
    const { data } = await axios.post(`/api/models/${encodeURIComponent(modelKey)}/connect-test`)
    const status = data?.status_code
    const body = data?.body || ''
    testResult.value = status >= 200 && status < 300 ? body : `HTTP ${status}\n${body}`
  } catch (error) {
    testResult.value = error?.response?.data?.detail || error?.message || 'The test request did not reach the endpoint.'
  } finally {
    testing.value = false
  }
}
</script>

<style scoped>
.connect-lead {
  margin: 0 0 0.75rem;
  color: var(--text-secondary);
}

.connect-field {
  display: flex;
  flex-wrap: wrap;
  align-items: flex-start;
  gap: 0.5rem;
  margin-bottom: 0.75rem;
}

.connect-label {
  flex: 0 0 7.5rem;
  font-size: 0.8rem;
  font-weight: 600;
  color: var(--text-secondary);
}

.connect-input,
.connect-value,
.connect-example,
.connect-result {
  flex: 1 1 12rem;
  margin: 0;
  padding: 0.45rem 0.6rem;
  border-radius: var(--radius-md);
  background: var(--bg-tertiary);
  color: var(--text-primary);
  font-size: 0.8rem;
  white-space: pre-wrap;
  word-break: break-word;
}

.connect-input {
  border: 1px solid var(--border-primary);
  font-family: inherit;
}

.connect-actions {
  display: flex;
  align-items: center;
  gap: 0.75rem;
}

.connect-note,
.connect-result {
  color: var(--text-secondary);
  font-size: 0.8rem;
}

.connect-result {
  margin-top: 0.75rem;
  max-height: 12rem;
  overflow: auto;
}
</style>
