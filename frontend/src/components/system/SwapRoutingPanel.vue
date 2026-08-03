<template>
  <section class="ev-section">
    <div class="ev-section-header">
      <button
        type="button"
        class="ev-section-header__toggle interactive-row"
        :aria-expanded="expanded"
        aria-controls="ev-section-routing-body"
        @click="expanded = !expanded"
      >
        <div class="ev-section-title">
          <i class="pi pi-sitemap" aria-hidden="true" />
          <h2>llama-swap routing</h2>
        </div>
        <i
          :class="['pi', 'ev-section-chevron', expanded ? 'pi-chevron-up' : 'pi-chevron-down']"
          aria-hidden="true"
        />
      </button>
      <div class="ev-section-actions">
        <Button
          icon="pi pi-refresh"
          text
          severity="secondary"
          size="small"
          :loading="loading"
          v-tooltip.top="'Reload routing'"
          aria-label="Reload routing"
          @click="reload"
        />
        <Button
          label="Save"
          icon="pi pi-save"
          size="small"
          :loading="saving"
          :disabled="!dirty || saving"
          @click="save"
        />
      </div>
    </div>

    <Transition name="ev-collapse">
      <div v-if="expanded" id="ev-section-routing-body" class="ev-section-body routing-body">
        <p class="routing-lead">
          Virtual model IDs and runtime pin sets written into
          <code>llama-swap-config.yaml</code>. Save here, then apply the pending
          llama-swap config so the proxy picks them up.
        </p>

        <div class="routing-panel">
          <div class="routing-panel__head">
            <div>
              <span class="routing-panel__title">Active profile</span>
              <span class="routing-panel__subtitle">
                Live on the running llama-swap process — no config rewrite
              </span>
            </div>
            <Tag
              :value="proxyAvailable ? 'Proxy reachable' : 'Proxy offline'"
              :severity="proxyAvailable ? 'success' : 'warning'"
            />
          </div>
          <div class="routing-panel__row routing-panel__row--active">
            <Dropdown
              v-model="activeProfile"
              :options="activeProfileOptions"
              option-label="label"
              option-value="value"
              placeholder="None"
              class="routing-control"
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
          </div>
          <p v-if="!proxyAvailable" class="routing-hint routing-hint--warn">
            Start llama-swap (apply config or activate an engine) to switch profiles live.
          </p>
        </div>

        <div v-if="formErrors.length" class="routing-banner routing-banner--error" role="alert">
          <div v-for="(error, idx) in formErrors" :key="`err-${idx}`">{{ error }}</div>
        </div>

        <div v-if="warnings.length" class="routing-banner routing-banner--warn" role="status">
          <div v-for="(warning, idx) in warnings" :key="`warn-${idx}`">{{ warning }}</div>
        </div>

        <div class="routing-block">
          <div class="routing-block__head">
            <h3>Selectors</h3>
            <Button
              label="Add selector"
              icon="pi pi-plus"
              size="small"
              text
              @click="addSelector"
            />
          </div>
          <p class="routing-hint">
            Client-facing IDs resolved per request.
            Strategies: <code>warm</code> (prefer loaded), <code>pin</code> (first target),
            <code>spillover</code> (fill then overflow).
          </p>

          <div v-if="!selectorRows.length" class="routing-empty">No selectors configured.</div>

          <div
            v-for="(row, idx) in selectorRows"
            :key="row._key"
            class="routing-card"
          >
            <div class="routing-card__toolbar">
              <span class="routing-card__label">Selector {{ idx + 1 }}</span>
              <Button
                icon="pi pi-trash"
                text
                severity="danger"
                rounded
                aria-label="Remove selector"
                @click="removeSelector(idx)"
              />
            </div>
            <div class="routing-grid routing-grid--2">
              <label class="routing-field">
                <span class="routing-field__label">ID</span>
                <InputText
                  v-model="row.id"
                  placeholder="coding-model"
                  class="w-full"
                  aria-label="Selector id"
                  @update:model-value="onFieldChange"
                />
              </label>
              <label class="routing-field">
                <span class="routing-field__label">Strategy</span>
                <Dropdown
                  v-model="row.strategy"
                  :options="strategyOptions"
                  class="w-full"
                  aria-label="Selector strategy"
                  @update:model-value="onFieldChange"
                />
              </label>
            </div>
            <div class="routing-grid routing-grid--2">
              <label class="routing-field">
                <span class="routing-field__label">Display name</span>
                <InputText
                  v-model="row.name"
                  placeholder="Optional"
                  class="w-full"
                  @update:model-value="onFieldChange"
                />
              </label>
              <label class="routing-field">
                <span class="routing-field__label">Description</span>
                <InputText
                  v-model="row.description"
                  placeholder="Optional"
                  class="w-full"
                  @update:model-value="onFieldChange"
                />
              </label>
            </div>
            <label class="routing-field">
              <span class="routing-field__label">Targets</span>
              <InputText
                v-model="row.targetsText"
                placeholder="model-a, model-b"
                class="w-full"
                aria-label="Selector targets"
                @update:model-value="onFieldChange"
              />
            </label>
            <label v-if="row.strategy === 'spillover'" class="routing-field routing-field--narrow">
              <span class="routing-field__label">Spillover threshold</span>
              <InputNumber
                v-model="row.spillover"
                :min="1"
                show-buttons
                class="w-full"
                @update:model-value="onFieldChange"
              />
            </label>
          </div>
        </div>

        <div class="routing-block">
          <div class="routing-block__head">
            <h3>Profiles</h3>
            <Button
              label="Add profile"
              icon="pi pi-plus"
              size="small"
              text
              @click="addProfile"
            />
          </div>
          <p class="routing-hint">
            Named pin maps switched at runtime. Pin target may be a model, alias,
            selector, or empty to disable that client id.
          </p>

          <div v-if="!profileRows.length" class="routing-empty">No profiles configured.</div>

          <div
            v-for="(row, idx) in profileRows"
            :key="row._key"
            class="routing-card"
          >
            <div class="routing-card__toolbar">
              <span class="routing-card__label">Profile {{ idx + 1 }}</span>
              <Button
                icon="pi pi-trash"
                text
                severity="danger"
                rounded
                aria-label="Remove profile"
                @click="removeProfile(idx)"
              />
            </div>
            <div class="routing-grid routing-grid--2">
              <label class="routing-field">
                <span class="routing-field__label">ID</span>
                <InputText
                  v-model="row.id"
                  placeholder="coding"
                  class="w-full"
                  aria-label="Profile id"
                  @update:model-value="onFieldChange"
                />
              </label>
              <label class="routing-field">
                <span class="routing-field__label">Description</span>
                <InputText
                  v-model="row.description"
                  placeholder="Optional"
                  class="w-full"
                  @update:model-value="onFieldChange"
                />
              </label>
            </div>

            <div class="routing-pins">
              <div class="routing-pins__head">
                <span class="routing-field__label">Pins</span>
                <Button
                  label="Add pin"
                  icon="pi pi-plus"
                  size="small"
                  text
                  @click="addPin(row)"
                />
              </div>
              <div
                v-for="(pin, pinIdx) in row.pins"
                :key="`${row._key}-pin-${pinIdx}`"
                class="routing-grid routing-grid--pins"
              >
                <InputText
                  v-model="pin.key"
                  placeholder="client model id"
                  class="w-full"
                  aria-label="Pin client id"
                  @update:model-value="onFieldChange"
                />
                <InputText
                  v-model="pin.target"
                  placeholder="target (empty = disable)"
                  class="w-full"
                  aria-label="Pin target"
                  @update:model-value="onFieldChange"
                />
                <Button
                  icon="pi pi-times"
                  text
                  severity="secondary"
                  rounded
                  aria-label="Remove pin"
                  @click="removePin(row, pinIdx)"
                />
              </div>
            </div>
          </div>
        </div>
      </div>
    </Transition>
  </section>
</template>

<script setup>
import { computed, onMounted, ref } from 'vue'
import axios from 'axios'
import { useToast } from 'primevue/usetoast'
import Button from 'primevue/button'
import Dropdown from 'primevue/dropdown'
import InputText from 'primevue/inputtext'
import InputNumber from 'primevue/inputnumber'
import Tag from 'primevue/tag'
import { useEnginesStore } from '@/stores/engines'
import {
  buildRoutingPayload,
  validateRoutingForm,
} from '@/components/system/swapRoutingForm'

const toast = useToast()
const enginesStore = useEnginesStore()

const expanded = ref(true)
const loading = ref(false)
const saving = ref(false)
const dirty = ref(false)
const activeBusy = ref(false)
const proxyAvailable = ref(false)
const warnings = ref([])
const formErrors = ref([])
const selectorRows = ref([])
const profileRows = ref([])
const activeProfile = ref(null)
const liveProfiles = ref([])

let keySeq = 0
function nextKey(prefix) {
  keySeq += 1
  return `${prefix}-${keySeq}`
}

const strategyOptions = ['warm', 'pin', 'spillover']

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

function selectorFromDoc(id, block) {
  const settings = block?.settings && typeof block.settings === 'object' ? block.settings : {}
  return {
    _key: nextKey('sel'),
    id: id || '',
    strategy: block?.strategy || 'warm',
    name: block?.name || '',
    description: block?.description || '',
    targetsText: Array.isArray(block?.targets) ? block.targets.join(', ') : '',
    spillover: Number(settings.spillover) > 0 ? Number(settings.spillover) : 1,
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
    const [{ data }] = await Promise.all([
      axios.get('/api/llama-swap/routing'),
      fetchLiveProfiles(),
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
    return
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
  } catch (err) {
    toast.add({
      severity: 'error',
      summary: 'Save failed',
      detail: err?.response?.data?.detail || err.message,
      life: 5500,
    })
  } finally {
    saving.value = false
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

onMounted(() => {
  void reload()
})
</script>

<style scoped>
.routing-body {
  display: flex;
  flex-direction: column;
  gap: 1rem;
}

.routing-lead,
.routing-hint {
  margin: 0;
  color: var(--text-secondary);
  font-size: 0.8125rem;
  line-height: 1.45;
}

.routing-hint--warn {
  color: var(--status-warning);
}

.routing-panel {
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
  padding: 0.9rem 1rem;
  background: var(--bg-surface);
  border: 1px solid var(--border-primary);
  border-radius: var(--radius-lg, 0.75rem);
}

.routing-panel__head {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 0.75rem;
}

.routing-panel__title {
  display: block;
  font-size: 0.7rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: var(--text-secondary);
  margin-bottom: 0.25rem;
}

.routing-panel__subtitle {
  display: block;
  font-size: 0.8125rem;
  line-height: 1.4;
  color: var(--text-secondary);
}

.routing-panel__row {
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
  align-items: center;
}

.routing-panel__row--active .routing-control {
  flex: 1 1 12rem;
  min-width: 10rem;
  max-width: 20rem;
}

.routing-banner {
  display: flex;
  flex-direction: column;
  gap: 0.3rem;
  padding: 0.55rem 0.75rem;
  border-radius: var(--radius-md, 0.5rem);
  font-size: 0.8125rem;
  line-height: 1.4;
}

.routing-banner--warn {
  background: var(--status-warning-soft);
  border: 1px solid rgba(245, 158, 11, 0.3);
  color: var(--status-warning);
}

.routing-banner--error {
  background: color-mix(in srgb, var(--status-error, #ef4444) 10%, transparent);
  border: 1px solid color-mix(in srgb, var(--status-error, #ef4444) 35%, transparent);
  color: var(--status-error, #ef4444);
}

.routing-block {
  display: flex;
  flex-direction: column;
  gap: 0.65rem;
}

.routing-block__head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 0.5rem;
}

.routing-block__head h3 {
  margin: 0;
  font-size: 0.95rem;
}

.routing-empty {
  color: var(--text-secondary);
  font-size: 0.8125rem;
  padding: 0.65rem 0.75rem;
  border: 1px dashed var(--border-primary);
  border-radius: var(--radius-md, 0.5rem);
  background: var(--bg-surface);
}

.routing-card {
  display: flex;
  flex-direction: column;
  gap: 0.65rem;
  padding: 0.9rem 1rem;
  border: 1px solid var(--border-primary);
  border-radius: var(--radius-lg, 0.75rem);
  background: var(--bg-surface);
}

.routing-card__toolbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 0.5rem;
}

.routing-card__label {
  font-size: 0.7rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: var(--text-secondary);
}

.routing-grid {
  display: grid;
  gap: 0.65rem;
  align-items: end;
}

.routing-grid--2 {
  grid-template-columns: repeat(2, minmax(0, 1fr));
}

.routing-grid--pins {
  grid-template-columns: minmax(0, 1fr) minmax(0, 1fr) auto;
  align-items: center;
}

.routing-field {
  display: flex;
  flex-direction: column;
  gap: 0.3rem;
  min-width: 0;
}

.routing-field--narrow {
  max-width: 12rem;
}

.routing-field__label {
  font-size: 0.7rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  color: var(--text-secondary);
}

.routing-pins {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}

.routing-pins__head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 0.5rem;
}

.w-full {
  width: 100%;
}

@media (max-width: 720px) {
  .routing-grid--2,
  .routing-grid--pins {
    grid-template-columns: 1fr;
  }

  .routing-panel__head {
    flex-direction: column;
  }

  .routing-field--narrow {
    max-width: none;
  }
}
</style>
