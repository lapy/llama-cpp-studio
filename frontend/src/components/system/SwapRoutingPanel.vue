<template>
  <div class="ev-system-layout">
    <p class="routing-lead">
      Define llama-swap <strong>selectors</strong> (virtual model IDs) and
      <strong>profiles</strong> (runtime pin maps). Save here, then apply the pending
      llama-swap config so the proxy rewrites
      <code>llama-swap-config.yaml</code>. Active profile switches only work for
      profiles already present in the applied config.
    </p>

    <div class="status-detail">
      <span class="detail-label">Active profile</span>
      <Dropdown
        v-model="activeProfile"
        :options="activeProfileOptions"
        option-label="label"
        option-value="value"
        placeholder="None"
        class="routing-active-select"
        :disabled="activeBusy || !proxyAvailable"
        aria-label="Active profile"
      />
      <Button
        label="Set"
        icon="pi pi-check"
        size="small"
        outlined
        :loading="activeBusy"
        :disabled="!proxyAvailable"
        @click="applyActiveProfile"
      />
      <Tag
        :value="proxyAvailable ? 'Proxy reachable' : 'Proxy offline'"
        :severity="proxyAvailable ? 'success' : 'warning'"
      />
      <Tag
        v-if="showApplyLlamaSwap"
        value="Apply pending"
        severity="warn"
      />
    </div>
    <small v-if="!proxyAvailable" class="form-hint">
      Start llama-swap (apply config or activate an engine) to switch profiles live.
    </small>
    <small v-else class="form-hint">
      Live profile switch uses the applied YAML; save + apply first if you just edited profiles.
    </small>

    <Message
      v-for="(error, idx) in formErrors"
      :key="`err-${idx}`"
      severity="error"
      :closable="false"
      class="routing-message"
    >
      {{ error }}
    </Message>
    <Message
      v-for="(warning, idx) in warnings"
      :key="`warn-${idx}`"
      severity="warn"
      :closable="false"
      class="routing-message"
    >
      {{ warning }}
    </Message>

    <div class="ev-subsection">
      <div class="routing-subsection-head">
        <h4>Selectors (virtual models)</h4>
        <Button
          label="Add selector"
          icon="pi pi-plus"
          size="small"
          text
          @click="addSelector"
        />
      </div>
      <small class="form-hint">
        Client-facing IDs resolved per request. Pick catalog models/aliases or type a peer FQN
        (<code>peer/model</code>).
      </small>

      <div v-if="!selectorRows.length" class="empty-state-mini">
        <i class="pi pi-info-circle" aria-hidden="true" />
        <span>No selectors yet. Add one to expose a virtual model ID to clients.</span>
      </div>

      <div
        v-for="(row, idx) in selectorRows"
        :key="row._key"
        class="routing-item"
      >
        <div class="form-row">
          <label>ID</label>
          <InputText
            v-model="row.id"
            placeholder="coding-model"
            class="form-input"
            aria-label="Selector id"
            @update:model-value="onFieldChange"
          />
          <Dropdown
            v-model="row.strategy"
            :options="strategyOptionValues"
            class="routing-strategy"
            aria-label="Selector strategy"
            @update:model-value="onFieldChange"
          />
          <Button
            icon="pi pi-trash"
            text
            severity="danger"
            size="small"
            aria-label="Remove selector"
            @click="removeSelector(idx)"
          />
        </div>
        <small class="form-hint form-hint--inset">{{ strategyHint(row.strategy) }}</small>
        <div class="form-row">
          <label>Name</label>
          <InputText
            v-model="row.name"
            placeholder="Optional display name"
            class="form-input"
            @update:model-value="onFieldChange"
          />
        </div>
        <div class="form-row">
          <label>Desc</label>
          <InputText
            v-model="row.description"
            placeholder="Optional description"
            class="form-input"
            @update:model-value="onFieldChange"
          />
        </div>
        <div class="form-row">
          <label>Targets</label>
          <AutoComplete
            v-model="row.targets"
            :suggestions="targetSuggestions"
            multiple
            :force-selection="false"
            dropdown
            class="form-input routing-autocomplete"
            placeholder="model, alias, or peer/model"
            aria-label="Selector targets"
            @complete="onTargetComplete"
            @update:model-value="onFieldChange"
          />
        </div>
        <div v-if="row.strategy === 'spillover'" class="form-row">
          <label>Spill</label>
          <InputNumber
            v-model="row.spillover"
            :min="1"
            show-buttons
            class="form-input-short"
            @update:model-value="onFieldChange"
          />
        </div>
        <div class="form-row">
          <label>Unlisted</label>
          <InputSwitch
            v-model="row.unlisted"
            aria-label="Selector unlisted"
            @update:model-value="onFieldChange"
          />
          <small class="form-hint form-hint--inline">
            Hide from <code>/v1/models</code> listings.
          </small>
        </div>
        <div
          v-for="(meta, metaIdx) in row.metadataRows"
          :key="`${row._key}-meta-${metaIdx}`"
          class="form-row"
        >
          <label>{{ metaIdx === 0 ? 'Meta' : '' }}</label>
          <InputText
            v-model="meta.key"
            placeholder="key"
            class="form-input-short"
            aria-label="Metadata key"
            @update:model-value="onFieldChange"
          />
          <InputText
            v-model="meta.value"
            placeholder="value"
            class="form-input"
            aria-label="Metadata value"
            @update:model-value="onFieldChange"
          />
          <Button
            icon="pi pi-times"
            text
            severity="secondary"
            size="small"
            aria-label="Remove metadata row"
            @click="removeMetadataRow(row, metaIdx)"
          />
        </div>
        <Button
          label="Add metadata"
          icon="pi pi-plus"
          size="small"
          text
          @click="addMetadataRow(row)"
        />
      </div>
    </div>

    <div class="ev-subsection">
      <div class="routing-subsection-head">
        <h4>Profiles</h4>
        <Button
          label="Add profile"
          icon="pi pi-plus"
          size="small"
          text
          @click="addProfile"
        />
      </div>
      <small class="form-hint">
        Named pin maps switched at runtime. Pin target may be a model, alias,
        selector, or empty to disable that client id.
      </small>

      <div v-if="!profileRows.length" class="empty-state-mini">
        <i class="pi pi-info-circle" aria-hidden="true" />
        <span>No profiles yet. Add one to pin client model ids to different backends.</span>
      </div>

      <div
        v-for="(row, idx) in profileRows"
        :key="row._key"
        class="routing-item"
      >
        <div class="form-row">
          <label>ID</label>
          <InputText
            v-model="row.id"
            placeholder="coding"
            class="form-input"
            aria-label="Profile id"
            @update:model-value="onFieldChange"
          />
          <Button
            icon="pi pi-trash"
            text
            severity="danger"
            size="small"
            aria-label="Remove profile"
            @click="removeProfile(idx)"
          />
        </div>
        <div class="form-row">
          <label>Desc</label>
          <InputText
            v-model="row.description"
            placeholder="Optional description"
            class="form-input"
            @update:model-value="onFieldChange"
          />
        </div>

        <div
          v-for="(pin, pinIdx) in row.pins"
          :key="`${row._key}-pin-${pinIdx}`"
          class="form-row"
        >
          <label>Pin</label>
          <InputText
            v-model="pin.key"
            placeholder="client model id"
            class="form-input-short"
            aria-label="Pin client id"
            @update:model-value="onFieldChange"
          />
          <AutoComplete
            v-model="pin.target"
            :suggestions="targetSuggestions"
            :force-selection="false"
            dropdown
            class="form-input routing-autocomplete"
            placeholder="target (empty = disable)"
            aria-label="Pin target"
            @complete="onTargetComplete"
            @update:model-value="onFieldChange"
          />
          <Button
            icon="pi pi-times"
            text
            severity="secondary"
            size="small"
            aria-label="Remove pin"
            @click="removePin(row, pinIdx)"
          />
        </div>
        <Button
          label="Add pin"
          icon="pi pi-plus"
          size="small"
          text
          @click="addPin(row)"
        />
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed, onMounted, ref } from 'vue'
import axios from 'axios'
import { useToast } from 'primevue/usetoast'
import Button from 'primevue/button'
import Dropdown from 'primevue/dropdown'
import InputText from 'primevue/inputtext'
import InputNumber from 'primevue/inputnumber'
import InputSwitch from 'primevue/inputswitch'
import AutoComplete from 'primevue/autocomplete'
import Tag from 'primevue/tag'
import Message from 'primevue/message'
import { useEnginesStore } from '@/stores/engines'
import { useModelStore } from '@/stores/models'
import {
  STRATEGY_OPTIONS,
  buildRoutingPayload,
  collectCatalogTargetIds,
  metadataToRows,
  normalizeTargetList,
  strategyHint,
  validateRoutingForm,
} from '@/components/system/swapRoutingForm'

const toast = useToast()
const enginesStore = useEnginesStore()
const modelStore = useModelStore()

const loading = ref(false)
const saving = ref(false)
const applying = ref(false)
const dirty = ref(false)
const activeBusy = ref(false)
const proxyAvailable = ref(false)
const warnings = ref([])
const formErrors = ref([])
const selectorRows = ref([])
const profileRows = ref([])
const activeProfile = ref(null)
const liveProfiles = ref([])
const targetSuggestions = ref([])

let keySeq = 0
function nextKey(prefix) {
  keySeq += 1
  return `${prefix}-${keySeq}`
}

const strategyOptionValues = STRATEGY_OPTIONS.map((opt) => opt.value)

const showApplyLlamaSwap = computed(() =>
  Boolean(
    enginesStore.swapConfigStale?.applicable &&
      enginesStore.swapConfigStale?.stale &&
      !dirty.value
  )
)

const catalogTargetIds = computed(() =>
  collectCatalogTargetIds(modelStore.allQuantizations || [])
)

const selectorIdSuggestions = computed(() =>
  selectorRows.value
    .map((row) => String(row.id || '').trim())
    .filter(Boolean)
)

const allTargetIds = computed(() => {
  const ids = new Set([...catalogTargetIds.value, ...selectorIdSuggestions.value])
  return [...ids].sort((a, b) => a.localeCompare(b))
})

const activeProfileOptions = computed(() => {
  const opts = [{ label: 'None', value: null }]
  const ids = new Set()
  for (const row of profileRows.value) {
    const id = String(row.id || '').trim()
    if (id) ids.add(id)
  }
  for (const profile of liveProfiles.value) {
    const id = String(profile?.id || '').trim()
    if (id) ids.add(id)
  }
  for (const id of [...ids].sort()) {
    opts.push({ label: id, value: id })
  }
  return opts
})

function onFieldChange() {
  dirty.value = true
  formErrors.value = []
}

function onTargetComplete(event) {
  const query = String(event?.query || '').trim().toLowerCase()
  const pool = allTargetIds.value
  targetSuggestions.value = query
    ? pool.filter((id) => id.toLowerCase().includes(query))
    : pool.slice(0, 40)
}

function selectorFromDoc(id, block) {
  const settings = block?.settings && typeof block.settings === 'object' ? block.settings : {}
  return {
    _key: nextKey('sel'),
    id: id || '',
    strategy: block?.strategy || 'warm',
    name: block?.name || '',
    description: block?.description || '',
    targets: normalizeTargetList(block?.targets),
    spillover: Number(settings.spillover) > 0 ? Number(settings.spillover) : 1,
    unlisted: Boolean(block?.unlisted),
    metadataRows: metadataToRows(block?.metadata),
  }
}

function profileFromDoc(id, block) {
  const pinsRaw = block?.pins && typeof block.pins === 'object' ? block.pins : {}
  const pins = Object.entries(pinsRaw).map(([key, target]) => ({
    key,
    target: target == null ? '' : String(target),
  }))
  if (!pins.length) pins.push({ key: '', target: '' })
  return {
    _key: nextKey('prof'),
    id: id || '',
    description: block?.description || '',
    pins,
  }
}

function loadFromDoc(doc) {
  const selectors = doc?.selectors && typeof doc.selectors === 'object' ? doc.selectors : {}
  const profiles = doc?.profiles && typeof doc.profiles === 'object' ? doc.profiles : {}
  selectorRows.value = Object.entries(selectors).map(([id, block]) =>
    selectorFromDoc(id, block)
  )
  profileRows.value = Object.entries(profiles).map(([id, block]) =>
    profileFromDoc(id, block)
  )
  warnings.value = Array.isArray(doc?.warnings) ? doc.warnings : []
  formErrors.value = []
  dirty.value = false
}

function addSelector() {
  selectorRows.value.push(selectorFromDoc('', { strategy: 'warm', targets: [] }))
  onFieldChange()
}

function removeSelector(idx) {
  selectorRows.value.splice(idx, 1)
  onFieldChange()
}

function addProfile() {
  profileRows.value.push(profileFromDoc('', { description: '', pins: { '': '' } }))
  onFieldChange()
}

function removeProfile(idx) {
  profileRows.value.splice(idx, 1)
  onFieldChange()
}

function addPin(row) {
  row.pins.push({ key: '', target: '' })
  onFieldChange()
}

function removePin(row, pinIdx) {
  row.pins.splice(pinIdx, 1)
  if (!row.pins.length) row.pins.push({ key: '', target: '' })
  onFieldChange()
}

function addMetadataRow(row) {
  row.metadataRows.push({ key: '', value: '' })
  onFieldChange()
}

function removeMetadataRow(row, metaIdx) {
  row.metadataRows.splice(metaIdx, 1)
  if (!row.metadataRows.length) row.metadataRows.push({ key: '', value: '' })
  onFieldChange()
}

async function fetchLiveProfiles() {
  try {
    const { data } = await axios.get('/api/llama-swap/profiles')
    liveProfiles.value = Array.isArray(data?.profiles) ? data.profiles : []
    activeProfile.value = data?.active ?? null
    proxyAvailable.value = true
  } catch {
    liveProfiles.value = []
    proxyAvailable.value = false
  }
}

async function reload() {
  loading.value = true
  try {
    if (!modelStore.allQuantizations?.length) {
      try {
        await modelStore.fetchModels()
      } catch {
        /* catalog suggestions are optional */
      }
    }
    const [{ data }] = await Promise.all([
      axios.get('/api/llama-swap/routing'),
      fetchLiveProfiles(),
      enginesStore.fetchSwapConfigStale(),
    ])
    loadFromDoc(data || {})
  } catch (err) {
    toast.add({
      severity: 'error',
      summary: 'Failed to load routing',
      detail: err?.response?.data?.detail || err.message,
      life: 4500,
    })
  } finally {
    loading.value = false
  }
}

async function save() {
  const errors = validateRoutingForm(selectorRows.value, profileRows.value)
  if (errors.length) {
    formErrors.value = errors
    toast.add({
      severity: 'warn',
      summary: 'Fix routing form',
      detail: errors[0],
      life: 4500,
    })
    return false
  }

  saving.value = true
  try {
    const payload = buildRoutingPayload(selectorRows.value, profileRows.value)
    const { data } = await axios.put('/api/llama-swap/routing', payload)
    loadFromDoc(data || {})
    enginesStore.markSwapConfigStaleLocal()
    toast.add({
      severity: 'success',
      summary: 'Routing saved',
      detail: 'Apply llama-swap config to publish profiles/selectors to the proxy.',
      life: 4000,
    })
    return true
  } catch (err) {
    toast.add({
      severity: 'error',
      summary: 'Save failed',
      detail: err?.response?.data?.detail || err.message,
      life: 5500,
    })
    return false
  } finally {
    saving.value = false
  }
}

async function applyConfig() {
  if (dirty.value) {
    const ok = await save()
    if (!ok) return
  }
  applying.value = true
  try {
    await enginesStore.applySwapConfig()
    await fetchLiveProfiles()
    toast.add({
      severity: 'success',
      summary: 'llama-swap applied',
      detail: 'Config rewritten and proxy reloaded. All loaded models were stopped.',
      life: 4500,
    })
  } catch (err) {
    toast.add({
      severity: 'error',
      summary: 'Apply failed',
      detail: err?.response?.data?.detail || err.message,
      life: 5500,
    })
  } finally {
    applying.value = false
  }
}

async function applyActiveProfile() {
  activeBusy.value = true
  try {
    const { data } = await axios.put('/api/llama-swap/profiles/active', {
      name: activeProfile.value,
    })
    activeProfile.value = data?.active ?? null
    proxyAvailable.value = true
    toast.add({
      severity: 'success',
      summary: 'Active profile updated',
      detail: activeProfile.value
        ? `Active profile is now «${activeProfile.value}».`
        : 'No profile active.',
      life: 3000,
    })
  } catch (err) {
    proxyAvailable.value = false
    toast.add({
      severity: 'error',
      summary: 'Could not set active profile',
      detail: err?.response?.data?.detail || err.message,
      life: 5000,
    })
  } finally {
    activeBusy.value = false
  }
}

defineExpose({
  reload,
  save,
  applyConfig,
  loading,
  saving,
  applying,
  dirty,
  showApplyLlamaSwap,
})

onMounted(() => {
  void reload()
})
</script>

<style scoped>
.routing-lead {
  margin: 0;
  color: var(--text-secondary);
  font-size: 0.875rem;
  line-height: 1.45;
}

.routing-active-select {
  min-width: 10rem;
  max-width: 16rem;
}

/* Local copies of EnginesView helpers (parent scoped styles do not reach here). */
.ev-system-layout {
  display: flex;
  flex-direction: column;
  gap: 1.25rem;
}

.ev-subsection h4 {
  font-size: 0.75rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: var(--text-secondary, #9ca3af);
  margin: 0;
}

.status-detail {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  font-size: 0.875rem;
  flex-wrap: wrap;
}

.detail-label {
  color: var(--text-secondary);
  flex-shrink: 0;
}

.form-hint {
  display: block;
  margin: -0.5rem 0 0;
  color: var(--text-secondary);
  font-size: 0.75rem;
  line-height: 1.4;
}

.form-hint--inset {
  margin: -0.25rem 0 0.5rem 5.75rem;
}

.form-hint--inline {
  margin: 0;
  flex: 1;
  min-width: 0;
}

.form-row {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  margin-bottom: 0.5rem;
}

.form-row label {
  font-size: 0.875rem;
  width: 88px;
  flex-shrink: 0;
  color: var(--text-secondary);
}

.form-input {
  flex: 1;
  min-width: 0;
}

.form-input-short {
  width: 140px;
  flex-shrink: 0;
}

.routing-autocomplete {
  flex: 1;
  min-width: 0;
}

.routing-autocomplete :deep(.p-autocomplete) {
  width: 100%;
}

.routing-autocomplete :deep(.p-autocomplete-multiple-container),
.routing-autocomplete :deep(.p-autocomplete-input) {
  width: 100%;
}

.empty-state-mini {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.75rem;
  color: var(--text-secondary);
  font-size: 0.875rem;
  margin: 0.5rem 0 0;
}

.empty-state-mini i {
  color: var(--text-muted);
}

.routing-message {
  margin: 0;
}

.routing-subsection-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 0.5rem;
  margin-bottom: 0.35rem;
}

.routing-item {
  margin-top: 0.75rem;
  padding-top: 0.75rem;
  border-top: 1px solid var(--border-primary);
}

.routing-item:first-of-type {
  border-top: none;
  padding-top: 0.35rem;
}

.routing-strategy {
  width: 8.5rem;
  flex-shrink: 0;
}
</style>
