<template>
  <Dialog
    :visible="visible"
    header="Hugging Face token"
    modal
    class="dialog-width-sm"
    @update:visible="requestVisibility"
  >
    <div class="dialog-stack">
      <PersistenceAlert :notice="notice" @retry="retryAction" @refresh="refreshTokenState" />
      <p class="page-section__description">
        A token unlocks gated models (for example Llama or Gemma). It is stored in your Studio data
        directory and shared by Search and the library.
      </p>
      <div class="page-form__field">
        <label class="page-form__label" for="hf-token">Token</label>
        <Password
          v-model="tokenInput"
          input-id="hf-token"
          placeholder="hf_…"
          :feedback="false"
          toggleMask
          class="w-full"
        />
        <p class="page-form__hint">
          Saved tokens are never shown again. Set a new token to replace the current one.
        </p>
      </div>
      <div v-if="modelStore.hasHuggingfaceToken" class="token-current">
        <i class="pi pi-check-circle token-current__icon" aria-hidden="true" />
        <span>Token set: {{ modelStore.huggingfaceToken || '••••••••' }}</span>
        <Button
          label="Clear token"
          severity="danger"
          text
          size="small"
          :disabled="saveHeld"
          @click="clearToken"
        />
      </div>
      <StateBanner v-if="saved" severity="success" message="Token saved." role="status" />
    </div>
    <template #footer>
      <div class="dialog-actions">
        <Button label="Cancel" severity="secondary" outlined @click="close" />
        <Button
          label="Save token"
          icon="pi pi-save"
          :disabled="!tokenInput || saveHeld"
          :loading="saving"
          @click="saveToken"
        />
      </div>
    </template>
  </Dialog>
</template>

<script setup>
import { computed, ref, watch } from 'vue'
import Dialog from 'primevue/dialog'
import Button from 'primevue/button'
import Password from 'primevue/password'
import { useToast } from 'primevue/usetoast'
import PersistenceAlert from '@/components/common/PersistenceAlert.vue'
import StateBanner from '@/components/common/StateBanner.vue'
import {
  classifyPersistenceError,
  noteDocumentSaveFailure,
  saveHeldForRefresh,
} from '@/composables/persistenceOutcome'
import { useModelStore } from '@/stores/models'

const props = defineProps({
  visible: { type: Boolean, default: false },
})

const emit = defineEmits(['update:visible'])
const toast = useToast()
const modelStore = useModelStore()

const tokenInput = ref('')
const notice = ref(null)
const action = ref('save')
const saving = ref(false)
const saved = ref(false)
const saveHeld = computed(() => saveHeldForRefresh(notice.value))

watch(
  () => props.visible,
  (open) => {
    if (open) {
      tokenInput.value = ''
      notice.value = null
      saved.value = false
    }
  },
)

function close() {
  saved.value = false
  notice.value = null
  emit('update:visible', false)
}

function requestVisibility(value) {
  if (value) {
    emit('update:visible', true)
    return
  }
  close()
}

async function refreshTokenState() {
  const draft = tokenInput.value
  try {
    await modelStore.fetchHuggingfaceTokenStatus()
    notice.value = null
    return true
  } catch {
    tokenInput.value = draft
    notice.value = classifyPersistenceError(new Error('reload failed'))
    return false
  }
}

async function saveToken() {
  saving.value = true
  saved.value = false
  const draft = tokenInput.value
  try {
    await modelStore.setHuggingfaceToken(tokenInput.value)
    tokenInput.value = ''
    notice.value = null
    saved.value = true
    toast.add({ severity: 'success', summary: 'Token saved', life: 3000 })
    emit('update:visible', false)
  } catch (error) {
    tokenInput.value = draft
    action.value = 'save'
    const outcome = noteDocumentSaveFailure(toast, error)
    if (!outcome) {
      toast.add({
        severity: 'error',
        summary: 'Token not saved',
        detail: error?.message || 'Could not save the token.',
        life: 4000,
      })
      return
    }
    notice.value = outcome
    if (outcome.committed !== true) return
    const reloaded = await refreshTokenState()
    notice.value = reloaded
      ? { ...outcome, refresh: false, retry: false }
      : {
          ...outcome,
          detail:
            'The document was replaced, but it could not be reloaded. Your edits are still here. Refresh before trying again.',
          refresh: true,
          retry: false,
        }
    if (reloaded) tokenInput.value = ''
  } finally {
    saving.value = false
  }
}

async function clearToken() {
  action.value = 'clear'
  saved.value = false
  try {
    await modelStore.clearHuggingfaceToken()
    notice.value = null
    toast.add({ severity: 'info', summary: 'Token cleared', life: 3000 })
  } catch (error) {
    const outcome = noteDocumentSaveFailure(toast, error)
    if (outcome) notice.value = outcome
    else {
      toast.add({
        severity: 'error',
        summary: 'Token not cleared',
        detail: error?.message || 'Could not clear the token.',
        life: 4000,
      })
    }
  }
}

function retryAction() {
  if (action.value === 'clear') void clearToken()
  else void saveToken()
}
</script>

<style scoped>
.token-current {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 0.5rem;
  padding: 0.5rem 0.75rem;
  border: 1px solid color-mix(in srgb, var(--status-success) 35%, transparent);
  border-radius: var(--radius-md);
  background: var(--status-success-soft);
  font-size: var(--text-sm);
}

.token-current__icon {
  color: var(--status-success);
}
</style>
