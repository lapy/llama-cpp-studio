<template>
  <Dialog
    :visible="visible"
    header="Connect"
    modal
    class="dialog-width-md"
    @update:visible="emit('update:visible', $event)"
  >
    <p class="connect-lead">Use this model through the local OpenAI-compatible endpoint.</p>
    <div class="connect-field">
      <span class="connect-label">API model ID</span>
      <code class="connect-value">{{ modelId || 'Unavailable until this model is registered with the proxy.' }}</code>
      <Button label="Copy" size="small" severity="secondary" outlined :disabled="!modelId" @click="copy(modelId)" />
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
        :disabled="!modelId || !model?.is_active"
        @click="sendTest"
      />
      <span v-if="!model?.is_active" class="connect-note">Start the model before sending a test.</span>
    </div>
    <pre v-if="testResult" class="connect-result" role="status">{{ testResult }}</pre>
  </Dialog>
</template>

<script setup>
import { computed, ref, watch } from 'vue'
import Dialog from 'primevue/dialog'
import Button from 'primevue/button'
import { useToast } from 'primevue/usetoast'
import { audioInferenceModelId } from '@/composables/useAudioInferenceClient'

const props = defineProps({
  visible: { type: Boolean, default: false },
  model: { type: Object, default: null },
  proxyPort: { type: [Number, String], default: 2000 },
})

const emit = defineEmits(['update:visible'])
const toast = useToast()
const testing = ref(false)
const testResult = ref('')

const modelId = computed(() => audioInferenceModelId(props.model, props.model?.config))
const port = computed(() => {
  const value = Number(props.proxyPort)
  return Number.isFinite(value) && value > 0 ? value : 2000
})
const endpoint = computed(() => {
  if (typeof window === 'undefined') return `http://127.0.0.1:${port.value}/v1/chat/completions`
  const { protocol, hostname } = window.location
  return `${protocol}//${hostname}:${port.value}/v1/chat/completions`
})
const requestBody = computed(() => ({
  model: modelId.value,
  messages: [{ role: 'user', content: 'Reply with the word pong.' }],
  max_tokens: 16,
}))
const requestExample = computed(() => JSON.stringify(requestBody.value, null, 2))

watch(() => props.visible, (open) => {
  if (open) testResult.value = ''
})

async function copy(text) {
  try {
    await navigator.clipboard.writeText(String(text || ''))
    toast.add({ severity: 'success', summary: 'Copied', life: 2000 })
  } catch {
    toast.add({ severity: 'warn', summary: 'Copy failed', detail: 'Select the text and copy it manually.', life: 3000 })
  }
}

async function sendTest() {
  testing.value = true
  testResult.value = ''
  try {
    const response = await fetch(endpoint.value, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(requestBody.value),
    })
    const text = await response.text()
    testResult.value = response.ok ? text : `HTTP ${response.status}\n${text}`
  } catch (error) {
    testResult.value = error?.message || 'The test request did not reach the endpoint.'
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
