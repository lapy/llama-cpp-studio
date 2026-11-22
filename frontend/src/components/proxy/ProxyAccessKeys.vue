<template>
  <PageSection
    title="Client access keys"
    description="Set model access, request rates, token budgets and concurrency per client. Secrets are shown once."
  >
    <template #meta>
      <Tag :value="`${keys.length} configured`" severity="secondary" />
    </template>
    <template #actions>
      <Button label="New key" icon="pi pi-plus" size="small" @click="focusKeyForm" />
    </template>

    <Dialog
      v-model:visible="keyFormVisible"
      header="Create inference key"
      :closable="!creating"
      :close-on-escape="!creating"
      modal
      :style="{ width: '50rem', maxWidth: '95vw' }"
      @show="keyNameInput?.focus()"
    >
      <div class="dialog-stack">
        <p class="dialog-intro">
          Give each client its own credential and policy. The secret is shown once after creation.
        </p>
        <StateBanner v-if="keyCreateError" severity="error" icon="">
          {{ keyCreateError }}
        </StateBanner>
        <div class="page-form key-options">
          <label class="page-form__label"
            >Token accounting<select
              v-model="keyAccounting"
              class="field"
              aria-label="Token accounting"
            >
              <option value="auto">Engine counts when available; text estimate otherwise</option>
              <option value="exact">Require exact engine input counts</option>
            </select></label
          >
          <label class="page-form__label"
            >Key name
            <input
              ref="keyNameInput"
              v-model="keyName"
              class="field"
              maxlength="120"
              placeholder="for example Production"
              @keyup.enter="createKey"
            />
          </label>
          <label class="page-form__label"
            >Expires at
            <input v-model="keyExpires" class="field" type="datetime-local" />
          </label>
          <label class="page-form__label"
            >Requests per minute
            <input
              v-model.number="keyRateLimit"
              class="field"
              type="number"
              min="1"
              placeholder="Unlimited"
            />
          </label>
          <label class="page-form__label page-form__field--full"
            >Allowed model IDs
            <input
              v-model="keyModels"
              class="field"
              placeholder="All models, or comma-separated IDs"
            />
          </label>
          <label v-for="field in policyFields" :key="field.name" class="page-form__label"
            >{{ field.label }}
            <input
              v-model.number="keyPolicy[field.name]"
              class="field"
              type="number"
              min="1"
              step="1"
              placeholder="Unlimited"
            />
          </label>
        </div>
        <p class="page-form__hint">
          Token budgets use the engine’s input-token counter when supported and reserve the
          requested output allowance before generation. Auto mode uses a visible byte estimate for
          text when counting is unavailable; exact mode refuses that fallback. Actual engine usage
          replaces the reservation on completion; missing usage retains the estimate. Exact mode
          guarantees exact admission counting where the installed engine supports it; it cannot
          guarantee a final output count after an interrupted stream. Daily and monthly budgets
          reset at midnight UTC. Text inference only when token limits are enabled.
        </p>
      </div>
      <template #footer>
        <div class="dialog-actions">
          <Button
            label="Cancel"
            text
            severity="secondary"
            :disabled="creating"
            @click="keyFormVisible = false"
          />
          <Button
            label="Create key"
            icon="pi pi-key"
            :disabled="creating || !keyName.trim()"
            :loading="creating"
            @click="createKey"
          />
        </div>
      </template>
    </Dialog>

    <Dialog
      v-model:visible="secretVisible"
      header="Save your inference key"
      modal
      :style="{ width: '35rem', maxWidth: '95vw' }"
    >
      <div class="dialog-stack">
        <div class="secret">
          <strong>{{ createdKeyLabel }}</strong>
          <label class="page-form__label" for="inference-key-secret">Inference key</label>
          <textarea
            id="inference-key-secret"
            ref="secretField"
            autofocus
            class="secret-field field"
            :value="createdKey"
            readonly
            rows="2"
            spellcheck="false"
            autocomplete="off"
            @focus="$event.target.select()"
            @click="$event.target.select()"
          />
          <div class="dialog-actions">
            <Button
              :label="keyCopyState === 'copied' ? 'Copied' : 'Copy'"
              :disabled="keyCopyState === 'copying'"
              icon="pi pi-copy"
              size="small"
              severity="secondary"
              outlined
              @click="copyKey"
            />
          </div>
          <p v-if="keyCopyState === 'copied'" class="copy-feedback" role="status">
            Key copied to clipboard.
          </p>
          <p v-else-if="keyCopyState === 'failed'" class="copy-feedback" role="status">
            Automatic copy is blocked. The key is selected; press Ctrl+C or Command+C, or use your
            device’s Copy action.
          </p>
        </div>
      </div>
      <template #footer>
        <div class="dialog-actions">
          <Button label="Done" text severity="secondary" @click="secretVisible = false" />
        </div>
      </template>
    </Dialog>

    <div v-if="keys.length" class="key-list">
      <div v-for="key in keys" :key="key.id" class="key-row">
        <div class="key-summary">
          <span class="key-title"
            ><strong>{{ key.name }}</strong
            ><Tag :value="keyStatus(key)" :severity="keySeverity(key)"
          /></span>
          <code>{{ key.prefix }}…</code>
          <small
            >{{ key.models?.length ? key.models.join(', ') : 'All models' }} ·
            {{
              key.rate_limit_per_minute ? `${key.rate_limit_per_minute} req/min` : 'no rate limit'
            }}
            · {{ key.expires_at ? `expires ${formatDate(key.expires_at)}` : 'no expiry' }} ·
            {{
              key.usage?.last_used_at
                ? `last used ${formatTime(key.usage.last_used_at)}`
                : key.last_used_at
                  ? `last used ${formatAgo(key.last_used_at)}`
                  : 'never used'
            }}</small
          >
        </div>
        <dl class="key-usage" v-if="key.usage">
          <div>
            <dt>
              Today
              <i
                class="pi pi-info-circle metric-help"
                v-tooltip.top="
                  'Enforcement total = confirmed usage + active reservations + retained conservative charges.'
                "
                aria-hidden="true"
              />
            </dt>
            <dd>
              {{ key.usage.daily_tokens.toLocaleString() }} /
              {{ formatAllowance(key.daily_token_budget) }} tokens
            </dd>
            <small v-if="key.usage.daily_confirmed_tokens != null" class="ledger-breakdown">
              {{ number(key.usage.daily_confirmed_tokens) }} confirmed ·
              {{ number(key.usage.daily_reserved_tokens) }} reserved ·
              {{ number(key.usage.daily_retained_tokens) }} retained
            </small>
            <small
              >{{
                key.usage.daily_remaining == null
                  ? 'Unlimited'
                  : `${key.usage.daily_remaining.toLocaleString()} remaining`
              }}
              · resets {{ formatTime(key.usage.day_resets_at) }}</small
            ><ProgressBar
              v-if="key.daily_token_budget"
              :value="Math.min(100, (key.usage.daily_tokens / key.daily_token_budget) * 100)"
              :show-value="false"
            />
          </div>
          <div>
            <dt>This month</dt>
            <dd>
              {{ key.usage.monthly_tokens.toLocaleString() }} /
              {{ formatAllowance(key.monthly_token_budget) }} tokens
            </dd>
            <small v-if="key.usage.monthly_confirmed_tokens != null" class="ledger-breakdown">
              {{ number(key.usage.monthly_confirmed_tokens) }} confirmed ·
              {{ number(key.usage.monthly_reserved_tokens) }} reserved ·
              {{ number(key.usage.monthly_retained_tokens) }} retained
            </small>
            <small
              >{{
                key.usage.monthly_remaining == null
                  ? 'Unlimited'
                  : `${key.usage.monthly_remaining.toLocaleString()} remaining`
              }}
              · resets {{ formatTime(key.usage.month_resets_at) }}</small
            ><ProgressBar
              v-if="key.monthly_token_budget"
              :value="Math.min(100, (key.usage.monthly_tokens / key.monthly_token_budget) * 100)"
              :show-value="false"
            />
          </div>
          <div>
            <dt>Last minute</dt>
            <dd>
              {{ key.usage.requests_per_minute }} /
              {{ formatAllowance(key.rate_limit_per_minute) }} requests
            </dd>
            <small
              >{{ key.usage.input_tokens_per_minute }} /
              {{ formatAllowance(key.input_tokens_per_minute) }} input ·
              {{ key.usage.output_tokens_per_minute }} /
              {{ formatAllowance(key.output_tokens_per_minute) }} output tokens</small
            >
          </div>
          <div>
            <dt>In flight</dt>
            <dd>
              {{ key.usage.concurrent_requests }} /
              {{ formatAllowance(key.max_concurrent_requests) }}
            </dd>
            <small
              >Output cap: {{ formatAllowance(key.max_output_tokens) }} per request ·
              {{ key.usage.estimated_requests }} requests charged by estimate</small
            >
          </div>
        </dl>
        <span class="key-actions">
          <Button
            label="Edit policy"
            size="small"
            severity="secondary"
            text
            @click="editKey(key)"
          />
          <Button label="Rotate" size="small" severity="warn" text @click="confirmRotate(key)" />
          <Button label="Revoke" size="small" severity="danger" text @click="confirmRevoke(key)" />
        </span>
      </div>
    </div>
    <EmptyState
      v-else
      icon="pi pi-key"
      title="No access keys yet"
      description="Create a credential for each client. Anonymous access is controlled separately in Settings & routing."
    >
      <Button label="Create the first key" icon="pi pi-key" size="small" @click="focusKeyForm" />
    </EmptyState>

    <Dialog
      v-model:visible="editingKeyVisible"
      header="Edit key policy"
      modal
      :style="{ width: '62rem', maxWidth: '95vw' }"
    >
      <div class="dialog-stack">
        <p>Changes affect new requests immediately. The existing secret remains valid.</p>
        <div class="page-form">
          <label class="page-form__label"
            >Key name<input v-model="editPolicy.name" class="field" maxlength="120"
          /></label>
          <label class="page-form__label"
            >Expires at<input v-model="editPolicy.expires" class="field" type="datetime-local"
          /></label>
          <label class="page-form__label page-form__field--full"
            >Allowed model IDs<input
              v-model="editPolicy.models"
              class="field"
              placeholder="All models, or comma-separated IDs"
          /></label>
          <label class="page-form__label"
            >Requests per minute<input
              v-model.number="editPolicy.rate_limit_per_minute"
              class="field"
              type="number"
              min="1"
              placeholder="Unlimited"
          /></label>
          <label v-for="field in policyFields" :key="`edit-${field.name}`" class="page-form__label"
            >{{ field.label
            }}<input
              v-model.number="editPolicy[field.name]"
              class="field"
              type="number"
              min="1"
              placeholder="Unlimited"
          /></label>
          <label class="page-form__label page-form__field--full"
            >Token accounting<select
              v-model="editPolicy.token_accounting"
              class="field"
              aria-label="Token accounting"
            >
              <option value="auto">Engine counts when available; text estimate otherwise</option>
              <option value="exact">Require exact engine input counts</option>
            </select></label
          >
          <fieldset class="page-form__field--full scope-options">
            <legend>Permissions</legend>
            <label>
              <input v-model="editPolicy.scopes" type="checkbox" value="inference" /> Run inference
            </label>
            <label>
              <input v-model="editPolicy.scopes" type="checkbox" value="models:read" /> List models
            </label>
          </fieldset>
        </div>
        <StateBanner v-if="policyError" severity="error" icon="">{{ policyError }}</StateBanner>
      </div>
      <template #footer>
        <div class="dialog-actions">
          <Button label="Cancel" text severity="secondary" @click="editingKey = null" />
          <Button
            label="Save policy"
            :loading="savingPolicy"
            :disabled="!editPolicy.name?.trim()"
            @click="saveKeyPolicy"
          />
        </div>
      </template>
    </Dialog>
  </PageSection>
</template>

<script setup>
import { computed, reactive, ref, watch } from 'vue'
import axios from 'axios'
import { useConfirm } from 'primevue/useconfirm'
import { useToast } from 'primevue/usetoast'
import Button from 'primevue/button'
import Dialog from 'primevue/dialog'
import ProgressBar from 'primevue/progressbar'
import Tag from 'primevue/tag'
import EmptyState from '@/components/common/EmptyState.vue'
import PageSection from '@/components/common/PageSection.vue'
import StateBanner from '@/components/common/StateBanner.vue'
import { copyText } from '@/utils/clipboard'
import { requireSingleConfirmation } from '@/composables/singleConfirm'
import {
  formatAgo,
  formatAllowance,
  formatDate,
  formatTime,
  number,
} from '@/utils/proxyRequestPresentation'

defineProps({
  keys: { type: Array, default: () => [] },
})

const emit = defineEmits(['refresh'])

const toast = useToast()
const confirm = useConfirm()

const policyFields = [
  { name: 'input_tokens_per_minute', label: 'Input tokens per minute' },
  { name: 'output_tokens_per_minute', label: 'Output tokens per minute' },
  { name: 'daily_token_budget', label: 'Daily token budget (UTC)' },
  { name: 'monthly_token_budget', label: 'Monthly token budget (UTC)' },
  { name: 'max_output_tokens', label: 'Maximum output tokens per request' },
  { name: 'max_concurrent_requests', label: 'Concurrent requests' },
]

const keyFormVisible = ref(false)
const keyCreateError = ref('')
const creating = ref(false)
const keyAccounting = ref('auto')
const keyName = ref('')
const keyExpires = ref('')
const keyRateLimit = ref(null)
const keyModels = ref('')
const keyPolicy = reactive(Object.fromEntries(policyFields.map((field) => [field.name, null])))
const keyNameInput = ref(null)

const createdKey = ref('')
const createdKeyLabel = ref('Copy this key now — it will not be shown again.')
const secretField = ref(null)
const keyCopyState = ref('idle')
let copyAttempt = 0
const secretVisible = computed({
  get: () => Boolean(createdKey.value),
  set: (value) => {
    if (!value) createdKey.value = ''
  },
})
watch(
  () => createdKey.value,
  () => {
    copyAttempt++
    keyCopyState.value = 'idle'
  },
)

const editingKey = ref(null)
const editPolicy = reactive({ scopes: [] })
const policyError = ref('')
const savingPolicy = ref(false)
const editingKeyVisible = computed({
  get: () => Boolean(editingKey.value),
  set: (value) => {
    if (!value) editingKey.value = null
  },
})

function policyNumber(value) {
  return value === '' || value == null ? null : value
}

function keyStatus(key) {
  if (key.expires_at && new Date(key.expires_at) <= new Date()) return 'Expired'
  if (key.usage?.daily_remaining === 0 || key.usage?.monthly_remaining === 0)
    return 'Budget exhausted'
  return 'Active'
}

function keySeverity(key) {
  return { Expired: 'danger', 'Budget exhausted': 'warn', Active: 'success' }[keyStatus(key)]
}

function focusKeyForm() {
  keyCreateError.value = ''
  keyFormVisible.value = true
}

async function createKey() {
  if (creating.value || !keyName.value.trim()) return
  creating.value = true
  keyCreateError.value = ''
  try {
    const { data } = await axios.post('/api/inference/keys', {
      ...Object.fromEntries(
        policyFields.map((field) => [field.name, policyNumber(keyPolicy[field.name])]),
      ),
      token_accounting: keyAccounting.value,
      name: keyName.value.trim(),
      expires_at: keyExpires.value ? new Date(keyExpires.value).toISOString() : null,
      models: keyModels.value
        .split(',')
        .map((item) => item.trim())
        .filter(Boolean),
      scopes: ['inference', 'models:read'],
      rate_limit_per_minute: policyNumber(keyRateLimit.value),
    })
    keyFormVisible.value = false
    createdKey.value = data.key
    createdKeyLabel.value = 'Key created. Copy it now — it will not be shown again.'
    keyName.value = ''
    keyAccounting.value = 'auto'
    keyExpires.value = ''
    keyModels.value = ''
    keyRateLimit.value = null
    for (const field of policyFields) keyPolicy[field.name] = null
    emit('refresh')
  } catch (err) {
    keyCreateError.value = err?.response?.data?.detail || err.message
  } finally {
    creating.value = false
  }
}

function editKey(key) {
  editingKey.value = key.id
  Object.assign(editPolicy, {
    ...key,
    token_accounting: key.token_accounting || 'auto',
    models: (key.models || []).join(', '),
    scopes: [...(key.scopes || ['inference'])],
    expires: key.expires_at
      ? new Date(
          new Date(key.expires_at).getTime() - new Date(key.expires_at).getTimezoneOffset() * 60000,
        )
          .toISOString()
          .slice(0, 16)
      : '',
  })
  policyError.value = ''
}

async function saveKeyPolicy() {
  savingPolicy.value = true
  try {
    await axios.patch(`/api/inference/keys/${encodeURIComponent(editingKey.value)}`, {
      token_accounting: editPolicy.token_accounting || 'auto',
      name: editPolicy.name.trim(),
      models: editPolicy.models
        .split(',')
        .map((model) => model.trim())
        .filter(Boolean),
      scopes: editPolicy.scopes,
      expires_at: editPolicy.expires ? new Date(editPolicy.expires).toISOString() : null,
      rate_limit_per_minute: policyNumber(editPolicy.rate_limit_per_minute),
      ...Object.fromEntries(
        policyFields.map((field) => [field.name, policyNumber(editPolicy[field.name])]),
      ),
    })
    editingKey.value = null
    toast.add({ severity: 'success', summary: 'Key policy updated', life: 2500 })
    emit('refresh')
  } catch (err) {
    policyError.value = err?.response?.data?.detail || err.message
  } finally {
    savingPolicy.value = false
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
        emit('refresh')
      } catch (err) {
        reportFailure(`Revoking “${key.name}” failed`, err)
      }
    },
  })
}

function confirmRotate(key) {
  requireSingleConfirmation(confirm, {
    header: `Rotate “${key.name}”?`,
    message:
      'A new secret is issued and the old one stops working immediately. Update clients before rotating.',
    icon: 'pi pi-refresh',
    acceptClass: 'p-button-warning',
    accept: async () => {
      try {
        const { data } = await axios.post(
          `/api/inference/keys/${encodeURIComponent(key.id)}/rotate`,
        )
        createdKey.value = data.key
        createdKeyLabel.value = 'Key rotated. Copy the new secret — the old one no longer works.'
        emit('refresh')
      } catch (err) {
        reportFailure(`Rotating “${key.name}” failed`, err)
      }
    },
  })
}

async function copyKey() {
  if (!createdKey.value || keyCopyState.value === 'copying') return
  const key = createdKey.value,
    attempt = ++copyAttempt
  keyCopyState.value = 'copying'
  try {
    await copyText(key, { container: secretField.value?.closest('[role="dialog"]') })
    if (attempt === copyAttempt && createdKey.value === key) keyCopyState.value = 'copied'
  } catch {
    if (attempt !== copyAttempt || createdKey.value !== key) return
    keyCopyState.value = 'failed'
    secretField.value?.focus()
    secretField.value?.select()
  }
}
</script>

<style scoped>
.dialog-intro,
.copy-feedback {
  margin: 0;
  color: var(--text-secondary);
  font-size: var(--text-sm);
}

/* Labels wrap their control for the accessible name; the grid keeps the
   shared label/hint spacing without a page-local control size. */
.page-form > label {
  display: grid;
  gap: 0.35rem;
  align-content: start;
}

.page-form > label .field {
  font-weight: 400;
}

.secret {
  display: grid;
  gap: 0.7rem;
  padding: 1rem;
  border: 1px solid var(--status-warning);
  border-radius: var(--radius-md);
  justify-items: start;
}

.secret-field {
  width: 100%;
  min-width: 0;
  resize: vertical;
  font-family: ui-monospace, monospace;
}

.key-list {
  display: grid;
  gap: 0.6rem;
  min-width: 0;
}

.key-row {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  justify-content: space-between;
  gap: 1rem;
  padding: 0.7rem 0;
  border-top: 1px solid var(--border-primary);
}

.key-summary {
  display: grid;
  gap: 0.25rem;
  min-width: 0;
}

.key-summary code {
  font-size: var(--text-sm);
}

.key-summary small {
  color: var(--text-secondary);
}

.key-title {
  display: flex;
  align-items: center;
  gap: 0.6rem;
}

.key-usage {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(min(100%, 12rem), 1fr));
  width: 100%;
  gap: 1rem;
  margin: 0;
}

.key-usage dt,
.key-usage small {
  color: var(--text-secondary);
  font-size: var(--text-xs);
  overflow-wrap: anywhere;
}

.key-usage dd {
  margin: 0.3rem 0;
  font-variant-numeric: tabular-nums;
  overflow-wrap: anywhere;
}

.key-usage :deep(.p-progressbar) {
  height: 0.35rem;
  margin-top: 0.4rem;
}

.ledger-breakdown {
  display: block;
  margin-bottom: 0.15rem;
  font-variant-numeric: tabular-nums;
}

.metric-help {
  margin-left: 0.25rem;
  color: var(--text-secondary);
  font-size: var(--text-xs);
  cursor: help;
}

.key-actions {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 0.5rem;
}

.scope-options {
  flex-wrap: wrap;
  gap: 1rem;
  margin: 0;
  border: 1px solid var(--border-primary);
}

@media (max-width: 640px), (pointer: coarse) {
  .key-row {
    align-items: stretch;
  }

  .key-actions {
    justify-content: flex-start;
  }
}
</style>
