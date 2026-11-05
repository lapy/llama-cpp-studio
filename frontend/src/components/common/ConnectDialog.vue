<template>
  <Dialog
    :visible="visible"
    header="Connect"
    modal
    class="dialog-width-md"
    @update:visible="requestVisibility"
  >
    <p v-if="copyFeedback" class="connect-note" role="status">{{ copyFeedback }}</p>
    <div v-if="manualCopyValue" class="connect-field">
      <label for="connect-manual-copy">Copy text manually</label
      ><textarea
        id="connect-manual-copy"
        ref="manualCopyField"
        :value="manualCopyValue"
        readonly
        rows="3"
        @focus="$event.target.select()"
      />
    </div>
    <p class="connect-lead">{{ lead }}</p>
    <div class="connect-field">
      <span class="connect-label">API model ID</span>
      <code class="connect-value">{{
        modelId || 'Unavailable until this model is registered with the proxy.'
      }}</code>
      <Button
        label="Copy model ID"
        size="small"
        severity="secondary"
        outlined
        :disabled="!modelId"
        @click="copy(modelId)"
      />
    </div>
    <div class="connect-field">
      <label class="connect-label" for="connect-public-url">Public URL</label>
      <input
        id="connect-public-url"
        v-model="publicUrlDraft"
        class="connect-input"
        type="url"
        placeholder="http://127.0.0.1:2000"
        :aria-invalid="Boolean(urlError)"
        :aria-describedby="urlError ? 'connect-url-help connect-url-error' : 'connect-url-help'"
      />
      <small id="connect-url-help"
        >Shared by all models. Set the URL clients use to reach this Studio proxy.</small
      >
      <Button
        label="Save public URL"
        size="small"
        severity="secondary"
        outlined
        :loading="savingUrl"
        @click="savePublicUrl"
      />
      <small v-if="urlSaved" role="status">Public URL saved.</small>
    </div>
    <div class="connect-field">
      <span class="connect-label">Endpoint</span>
      <code class="connect-value">{{ endpoint }}</code>
      <Button
        label="Copy endpoint"
        size="small"
        severity="secondary"
        outlined
        @click="copy(endpoint)"
      />
    </div>
    <div class="connect-request">
      <div class="connect-request__header">
        <span class="connect-label">Request example</span>
        <div class="connect-format" role="group" aria-label="Request format">
          <button
            type="button"
            :aria-pressed="requestFormat === 'curl'"
            @click="requestFormat = 'curl'"
          >
            cURL
          </button>
          <button
            type="button"
            :aria-pressed="requestFormat === 'json'"
            @click="requestFormat = 'json'"
          >
            JSON body
          </button>
        </div>
      </div>
      <pre
        class="connect-example"
        :aria-label="requestFormat === 'curl' ? 'Complete cURL request' : 'JSON request body'"
        >{{ requestFormat === 'curl' ? curlExample : requestExample }}</pre>
      <Button
        :label="requestFormat === 'curl' ? 'Copy cURL' : 'Copy JSON body'"
        size="small"
        severity="secondary"
        outlined
        @click="copy(requestFormat === 'curl' ? curlExample : requestExample)"
      />
    </div>
    <p v-if="urlError" id="connect-url-error" class="connect-error" role="alert">{{ urlError }}</p>
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
    <div class="connect-field connect-keys">
      <span class="connect-label">Access keys</span>
      <p v-if="keysLoading" role="status">Loading access keys…</p>
      <div v-else-if="keysError" class="connect-error" role="alert">
        {{ keysError }}
        <Button label="Retry keys" size="small" severity="secondary" @click="loadAccessKeys" />
      </div>
      <p class="connect-note">{{ accessPolicyLabel }}</p>
      <div class="connect-key-create">
        <input
          v-model="keyName"
          class="connect-input"
          type="text"
          maxlength="120"
          placeholder="Key name, for example Desktop"
          aria-label="New access key name"
        />
        <Button
          label="Create key"
          size="small"
          severity="secondary"
          outlined
          :loading="creatingKey"
          :disabled="!keyName.trim() || (Boolean(createdKey) && !keyAcknowledged)"
          @click="createAccessKey"
        />
      </div>
      <div v-if="createdKey" class="connect-key-secret">
        <span>This key is shown once. Store it now.</span>
        <code>{{ createdKey }}</code>
        <Button
          label="I saved this key"
          size="small"
          severity="secondary"
          outlined
          @click="keyAcknowledged = true"
        />
        <small v-if="keyAcknowledged">Key saved. You can close this dialog.</small>
        <Button
          label="Copy key"
          size="small"
          severity="secondary"
          outlined
          @click="copy(createdKey)"
        />
      </div>
      <ul v-if="accessKeys.length" class="connect-key-list">
        <li v-for="item in accessKeys" :key="item.id">
          <span
            >{{ item.name }} <code>{{ item.prefix }}…</code></span
          >
          <Button
            label="Revoke"
            size="small"
            severity="danger"
            text
            :loading="revokingKeyId === item.id"
            :disabled="Boolean(revokingKeyId)"
            @click="confirmRevoke(item)"
          />
        </li>
      </ul>
      <small v-if="!keysLoading && !keysError && !accessKeys.length"
        >No access keys configured.</small
      >
    </div>
    <p v-if="persistenceNotice" class="persistence-notice" role="status">
      <strong>{{ persistenceNotice.summary }}.</strong>
      {{ persistenceNotice.detail }}
      <button v-if="persistenceNotice.retry" type="button" @click="savePublicUrl">Try again</button>
      <button v-if="persistenceNotice.refresh" type="button" @click="refreshPublicUrl">
        Refresh
      </button>
    </p>
    <pre v-if="testResult" class="connect-result" role="status">{{ testResult }}</pre>
  </Dialog>
</template>

<script setup>
import { copyText } from '@/utils/clipboard'
import { computed, nextTick, ref, watch } from 'vue'
import Dialog from 'primevue/dialog'
import Button from 'primevue/button'
import { useToast } from 'primevue/usetoast'
import { useConfirm } from 'primevue/useconfirm'
import { requireSingleConfirmation } from '@/composables/singleConfirm'
import { modelRuntimeState } from '@/utils/modelPresentation'
import axios from 'axios'
import { audioInferenceModelId } from '@/composables/useAudioInferenceClient'
import {
  connectEndpoint,
  curlRequest,
  connectRequest,
  normalizePublicInferenceUrl,
} from '@/composables/connectTarget'
import { classifyPersistenceError } from '@/composables/persistenceOutcome'

const props = defineProps({
  visible: { type: Boolean, default: false },
  model: { type: Object, default: null },
  proxyPort: { type: [Number, String], default: 2000 },
  publicInferenceUrl: { type: String, default: '' },
})

const emit = defineEmits(['update:visible', 'update:publicInferenceUrl'])
const toast = useToast()
const confirm = useConfirm()
const keysLoading = ref(false)
const keysError = ref('')
const anonymousAllowed = ref(null)
const revokingKeyId = ref('')
const keyAcknowledged = ref(false)
const savingUrl = ref(false)
const urlSaved = ref(false)
const accessPolicyLabel = computed(() =>
  anonymousAllowed.value === true
    ? 'Anonymous inference is enabled. Clients may connect without a key.'
    : anonymousAllowed.value === false
      ? 'An access key is required for inference.'
      : 'Access policy could not be verified. Check Proxy settings.',
)
const copyFeedback = ref(''),
  manualCopyValue = ref(''),
  manualCopyField = ref(null)
const testing = ref(false)
const requestFormat = ref('curl')
const testResult = ref('')
const publicUrlDraft = ref('')
watch(
  publicUrlDraft,
  () => {
    urlSaved.value = false
  },
  { flush: 'sync' },
)
const persistenceNotice = ref(null)
const urlError = ref('')
const accessKeys = ref([])
const keyName = ref('')
const createdKey = ref('')
const creatingKey = ref(false)

const modelId = computed(() => audioInferenceModelId(props.model, props.model?.config))
const example = computed(() => connectRequest(props.model))
const endpoint = computed(() =>
  connectEndpoint(props.model, {
    publicUrl: publicUrlDraft.value,
    proxyPort: props.proxyPort,
  }),
)
const curlExample = computed(() =>
  curlRequest(endpoint.value, example.value.body, anonymousAllowed.value !== true),
)
const requestExample = computed(() => JSON.stringify(example.value.body, null, 2))
const lead = computed(() => {
  if (example.value.kind === 'embeddings')
    return 'This model serves embeddings. The example calls /v1/embeddings.'
  if (example.value.kind === 'audio')
    return 'This model is served by audio.cpp. Run it from the Audio page.'
  return 'This model serves chat completions. The example is a short text request.'
})
const canTest = computed(() =>
  Boolean(
    modelId.value && modelRuntimeState(props.model).actionable && example.value.kind !== 'audio',
  ),
)
const testNote = computed(() => {
  if (example.value.kind === 'audio') return 'Use the Audio page to run this model.'
  if (!props.model?.is_active) return 'Start the model before sending a test.'
  return 'The test is sent through Studio, not directly to the proxy port.'
})

watch(
  () => props.visible,
  (open) => {
    copyFeedback.value = ''
    manualCopyValue.value = ''
    if (!open) {
      createdKey.value = ''
      return
    }
    testResult.value = ''
    requestFormat.value = 'curl'
    urlError.value = ''
    publicUrlDraft.value = props.publicInferenceUrl || ''
    createdKey.value = ''
    keyAcknowledged.value = false
    urlSaved.value = false
    loadAccessKeys()
    loadAccessPolicy()
  },
  { immediate: true },
)

function requestVisibility(value) {
  if (value || !createdKey.value || keyAcknowledged.value) {
    emit('update:visible', value)
    return
  }
  requireSingleConfirmation(confirm, {
    header: 'Close without saving this key?',
    message: 'This key is shown only once. Copy it or store it before closing.',
    acceptLabel: 'Close dialog',
    rejectLabel: 'Keep key open',
    accept: () => emit('update:visible', false),
  })
}

async function loadAccessPolicy() {
  anonymousAllowed.value = null
  try {
    const { data } = await axios.get('/api/proxy')
    if (typeof data?.settings?.allow_anonymous === 'boolean')
      anonymousAllowed.value = data.settings.allow_anonymous
  } catch {
    /* Unknown policy is shown explicitly. */
  }
}

async function loadAccessKeys() {
  keysLoading.value = true
  keysError.value = ''
  try {
    const response = await axios.get('/api/inference/keys')
    if (!Array.isArray(response?.data?.keys)) throw new Error('Invalid key response')
    accessKeys.value = response.data.keys
  } catch {
    keysError.value = 'Could not load access keys. Your access policy has not been changed.'
  } finally {
    keysLoading.value = false
  }
}

async function createAccessKey() {
  const name = keyName.value.trim()
  if (!name) return
  creatingKey.value = true
  try {
    const { data } = await axios.post('/api/inference/keys', { name })
    createdKey.value = data?.key || ''
    keyAcknowledged.value = false
    keyName.value = ''
    await loadAccessKeys()
  } catch (error) {
    toast.add({
      severity: 'error',
      summary: 'Key not created',
      detail: error?.response?.data?.detail || error?.message || 'Could not create the access key.',
      life: 4000,
    })
  } finally {
    creatingKey.value = false
  }
}

function confirmRevoke(key) {
  requireSingleConfirmation(confirm, {
    header: `Revoke “${key.name}”?`,
    message: 'Clients using this key stop working immediately.',
    icon: 'pi pi-exclamation-triangle',
    acceptClass: 'p-button-danger',
    acceptLabel: 'Revoke key',
    rejectLabel: 'Keep key',
    accept: () => revokeAccessKey(key.id),
  })
}

async function revokeAccessKey(id) {
  if (revokingKeyId.value) return
  revokingKeyId.value = id
  try {
    await axios.delete(`/api/inference/keys/${encodeURIComponent(id)}`)
    accessKeys.value = accessKeys.value.filter((item) => item.id !== id)
  } catch (error) {
    toast.add({
      severity: 'error',
      summary: 'Key not revoked',
      detail: error?.response?.data?.detail || error?.message || 'Could not revoke the access key.',
      life: 4000,
    })
  } finally {
    revokingKeyId.value = ''
  }
}

watch(
  () => props.publicInferenceUrl,
  (value) => {
    if (!props.visible) publicUrlDraft.value = value || ''
  },
)

async function copy(text) {
  manualCopyValue.value = ''
  copyFeedback.value = ''
  if (text == null || String(text) === '') {
    copyFeedback.value = 'No text is available to copy.'
    return
  }
  try {
    await copyText(text)
    copyFeedback.value = 'Copied to clipboard.'
  } catch {
    manualCopyValue.value = String(text || '')
    copyFeedback.value =
      'Automatic copy is blocked. The text is selected; press Ctrl+C or Command+C, or use your device’s Copy action.'
    await nextTick()
    manualCopyField.value?.focus()
    manualCopyField.value?.select()
  }
}

async function loadStoredPublicUrl() {
  const { data } = await axios.get('/api/settings/inference')
  return data?.public_inference_url || ''
}

async function refreshPublicUrl() {
  const draft = publicUrlDraft.value
  try {
    publicUrlDraft.value = await loadStoredPublicUrl()
    emit('update:publicInferenceUrl', publicUrlDraft.value)
    persistenceNotice.value = null
  } catch {
    publicUrlDraft.value = draft
    persistenceNotice.value = {
      outcome: 'unknown',
      committed: 'unknown',
      summary: 'Outcome unknown',
      detail:
        'The saved URL could not be reloaded. Your edits are still here. Refresh before trying again.',
      refresh: true,
      retry: false,
      preserveEdits: true,
      success: false,
    }
  }
}

async function savePublicUrl() {
  if (savingUrl.value) return
  urlSaved.value = false
  const draft = publicUrlDraft.value
  const normalized = normalizePublicInferenceUrl(draft)
  if (draft.trim() && !normalized) {
    urlError.value = 'Use an http or https URL without a username or password.'
    return
  }
  urlError.value = ''
  savingUrl.value = true
  try {
    const { data } = await axios.put('/api/settings/inference', {
      public_inference_url: normalized,
    })
    publicUrlDraft.value = data?.public_inference_url || ''
    emit('update:publicInferenceUrl', publicUrlDraft.value)
    persistenceNotice.value = null
    urlSaved.value = true
  } catch (error) {
    publicUrlDraft.value = draft
    const outcome = classifyPersistenceError(error)
    if (!outcome) {
      toast.add({
        severity: 'error',
        summary: 'URL not saved',
        detail: error?.response?.data?.detail || error?.message || 'Could not save the public URL.',
        life: 4000,
      })
      return
    }
    persistenceNotice.value = outcome
    if (outcome.committed === true) {
      try {
        publicUrlDraft.value = await loadStoredPublicUrl()
        emit('update:publicInferenceUrl', publicUrlDraft.value)
        persistenceNotice.value = { ...outcome, refresh: false, retry: false }
      } catch {
        publicUrlDraft.value = draft
        persistenceNotice.value = {
          ...outcome,
          detail:
            'The document was replaced, but it could not be reloaded. Your edits are still here. Refresh before trying again.',
          refresh: true,
          retry: false,
        }
      }
    }
    toast.add({
      severity: outcome.committed === true ? 'warn' : 'error',
      summary: persistenceNotice.value.summary,
      detail: persistenceNotice.value.detail,
      life: 6000,
    })
  } finally {
    savingUrl.value = false
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
    testResult.value =
      error?.response?.data?.detail ||
      error?.message ||
      'The test request did not reach the endpoint.'
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

.connect-keys {
  align-items: flex-start;
  margin-top: 1.25rem;
  padding-top: 1.25rem;
  border-top: 1px solid var(--border-primary);
}

.connect-key-create,
.connect-key-secret,
.connect-key-list {
  flex: 1 1 20rem;
}

.connect-key-create {
  display: flex;
  gap: 0.5rem;
}

.connect-key-secret {
  display: grid;
  gap: 0.4rem;
  padding: 0.6rem;
  border: 1px solid var(--status-warning);
  border-radius: var(--radius-md);
  background: var(--bg-tertiary);
}

.connect-key-secret code {
  overflow-wrap: anywhere;
}

.connect-key-list {
  display: grid;
  gap: 0.35rem;
  margin: 0;
  padding: 0;
  list-style: none;
}

.connect-key-list li {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 0.5rem;
}

.persistence-notice {
  margin: 0 0 0.75rem;
  padding: 0.6rem 0.75rem;
  border-radius: var(--radius-md);
  background: var(--bg-tertiary);
}

.connect-actions {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 0.75rem;
}

.connect-error {
  margin: 0 0 0.75rem;
  color: var(--status-warning);
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

.connect-request {
  display: grid;
  gap: 0.75rem;
  margin-block: 1rem;
}
.connect-request__header,
.connect-format {
  display: flex;
  gap: 0.5rem;
  align-items: center;
  flex-wrap: wrap;
}
.connect-format button {
  border: 1px solid var(--border-secondary);
  border-radius: var(--radius-sm);
  padding: 0.4rem 0.65rem;
  background: var(--bg-secondary);
  color: var(--text-secondary);
  cursor: pointer;
}
.connect-format button[aria-pressed='true'] {
  color: var(--accent-primary);
  background: var(--nav-active-bg);
  border-color: var(--accent-primary);
}
.connect-request .connect-example {
  max-height: 14rem;
  overflow: auto;
  white-space: pre-wrap;
  margin: 0;
}
.connect-request > .p-button {
  justify-self: start;
}

@media (max-width: 600px) {
  .connect-field {
    display: grid;
    grid-template-columns: minmax(0, 1fr);
    gap: 0.65rem;
  }
  .connect-label {
    flex: none;
  }
  .connect-field > .p-button {
    justify-self: start;
  }
}
</style>
