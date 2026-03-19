<template>
  <div class="model-config-view">

    <div v-if="loading" class="loading-state">
      <ProgressSpinner style="width:40px;height:40px" />
      <span>Loading configuration…</span>
    </div>

    <div v-else-if="!model" class="empty-state">
      <i class="pi pi-exclamation-circle" style="font-size:3rem;color:var(--text-secondary)" />
      <h3>Model not found</h3>
      <Button label="Back to Models" icon="pi pi-arrow-left" @click="$router.push('/models')" />
    </div>

    <template v-else>
      <!-- Header -->
      <div class="config-header">
        <Button icon="pi pi-arrow-left" text severity="secondary" @click="$router.push('/models')" />
        <div class="header-info">
          <h1>{{ model.display_name || model.base_model_name }}</h1>
          <div class="header-meta">
            <Tag :value="model.format || 'gguf'" severity="info" />
            <Tag v-if="model.quantization" :value="model.quantization" severity="secondary" />
            <a :href="`https://huggingface.co/${model.huggingface_id}`" target="_blank" class="hf-link">
              <i class="pi pi-external-link" /> {{ model.huggingface_id }}
            </a>
          </div>
        </div>
      </div>

      <!-- Engine Selector -->
      <div class="config-card">
        <div class="section-label">Engine</div>
        <div class="engine-selector">
          <div
            v-for="eng in engineOptions"
            :key="eng.value"
            class="engine-option"
            :class="{ selected: config.engine === eng.value }"
            @click="changeEngine(eng.value)"
          >
            <div class="engine-option-label">
              <span
                v-if="eng.value === 'llama_cpp'"
                class="engine-mark engine-mark--llama"
                aria-hidden="true"
              >L</span>
              <span
                v-else-if="eng.value === 'ik_llama'"
                class="engine-mark engine-mark--ik"
                aria-hidden="true"
              >IK</span>
              <i
                v-else-if="eng.value === 'lmdeploy'"
                class="pi pi-server engine-icon-lmdeploy"
                aria-hidden="true"
              />
              <span class="engine-name">{{ eng.label }}</span>
            </div>
          </div>
        </div>
      </div>

      <!-- Basic Parameters -->
      <div class="config-card">
        <div class="section-label">Basic Parameters</div>
        <div class="params-grid">
          <div v-for="param in basicParams" :key="param.key" class="param-field">
            <label :for="`param-${param.key}`">
              {{ param.label }}
              <i class="pi pi-info-circle param-info" v-tooltip.top="param.description" />
            </label>
            <!-- Context length / session length: slider + numeric input (soft max based on model metadata) -->
            <template v-if="param.type === 'int' && (param.key === 'ctx_size' || param.key === 'session_len')">
              <div class="param-slider-row">
                <Slider
                  v-model="config[param.key]"
                  :min="512"
                  :max="maxContextSuggestion || 131072"
                  :step="256"
                  class="param-slider"
                />
                <span v-if="maxContextSuggestion" class="param-hint">
                  Suggested max: {{ maxContextSuggestion.toLocaleString() }} tokens
                </span>
              </div>
              <InputNumber
                :id="`param-${param.key}`"
                v-model="config[param.key]"
                :placeholder="String(param.default ?? '')"
                class="param-input"
              />
            </template>
            <!-- GPU layers: slider guided by detected layer count, but value not clamped -->
            <template v-else-if="param.type === 'int' && param.key === 'n_gpu_layers'">
              <div class="param-slider-row">
                <Slider
                  v-model="config[param.key]"
                  :min="0"
                  :max="layerCountSuggestion || 128"
                  :step="1"
                  class="param-slider"
                />
                <span v-if="layerCountSuggestion" class="param-hint">
                  Detected layers: {{ layerCountSuggestion }}
                </span>
              </div>
              <InputNumber
                :id="`param-${param.key}`"
                v-model="config[param.key]"
                :placeholder="String(param.default ?? '')"
                class="param-input"
              />
            </template>
            <!-- Params with options: render as select -->
            <Dropdown
              v-else-if="param.options && param.options.length"
              :id="`param-${param.key}`"
              v-model="config[param.key]"
              :options="param.options"
              optionLabel="label"
              optionValue="value"
              :placeholder="param.default != null ? String(param.default) : ''"
              class="param-input"
            />
            <!-- Fallback: regular numeric / other inputs -->
            <InputNumber
              v-else-if="param.type === 'int'"
              :id="`param-${param.key}`"
              v-model="config[param.key]"
              :placeholder="String(param.default ?? '')"
              class="param-input"
            />
            <InputNumber
              v-else-if="param.type === 'float'"
              :id="`param-${param.key}`"
              v-model="config[param.key]"
              :minFractionDigits="1"
              :maxFractionDigits="4"
              :placeholder="String(param.default ?? '')"
              class="param-input"
            />
            <InputSwitch
              v-else-if="param.type === 'bool'"
              :id="`param-${param.key}`"
              v-model="config[param.key]"
            />
            <InputText
              v-else
              :id="`param-${param.key}`"
              v-model="config[param.key]"
              :placeholder="param.default != null ? String(param.default) : ''"
              class="param-input"
            />
          </div>
        </div>
      </div>

      <!-- Advanced Parameters -->
      <div class="config-card">
        <div class="section-label">
          Advanced Parameters
          <span class="section-count" v-if="activeAdvancedParams.length">
            {{ activeAdvancedParams.length }} active
          </span>
        </div>

        <div v-if="activeAdvancedParams.length" class="params-grid" style="margin-bottom:1rem">
          <div v-for="param in activeAdvancedParams" :key="param.key" class="param-field">
            <label :for="`adv-${param.key}`">
              {{ param.label }}
              <i class="pi pi-info-circle param-info" v-tooltip.top="param.description" />
              <Button
                icon="pi pi-times"
                text
                severity="danger"
                size="small"
                class="remove-param-btn"
                v-tooltip.top="'Remove parameter'"
                @click="removeAdvancedParam(param.key)"
              />
            </label>
            <Dropdown
              v-if="param.options && param.options.length"
              :id="`adv-${param.key}`"
              v-model="config[param.key]"
              :options="param.options"
              optionLabel="label"
              optionValue="value"
              :placeholder="param.default != null ? String(param.default) : ''"
              class="param-input"
            />
            <InputNumber
              v-else-if="param.type === 'int'"
              :id="`adv-${param.key}`"
              v-model="config[param.key]"
              :placeholder="String(param.default ?? '')"
              class="param-input"
            />
            <InputNumber
              v-else-if="param.type === 'float'"
              :id="`adv-${param.key}`"
              v-model="config[param.key]"
              :minFractionDigits="1"
              :maxFractionDigits="4"
              :placeholder="String(param.default ?? '')"
              class="param-input"
            />
            <InputSwitch
              v-else-if="param.type === 'bool'"
              :id="`adv-${param.key}`"
              v-model="config[param.key]"
            />
            <InputText
              v-else
              :id="`adv-${param.key}`"
              v-model="config[param.key]"
              :placeholder="param.default != null ? String(param.default) : ''"
              class="param-input"
            />
          </div>
        </div>

        <div class="add-param-row">
          <Dropdown
            v-model="selectedNewParam"
            :options="availableAdvancedParams"
            optionLabel="label"
            optionValue="key"
            placeholder="Add parameter…"
            filter
            :filterPlaceholder="'Search parameters…'"
            class="add-param-dropdown"
            @update:modelValue="onNewParamSelected"
          >
            <template #option="{ option }">
              <div class="param-option">
                <span class="param-option-label">{{ option.label }}</span>
                <span class="param-option-desc">{{ option.description }}</span>
              </div>
            </template>
          </Dropdown>
        </div>
      </div>

      <!-- Custom CLI Arguments -->
      <div class="config-card">
        <div class="section-label">
          Custom Arguments
          <small class="section-hint">Raw CLI flags appended to the server command</small>
        </div>
        <Textarea
          v-model="config.custom_args"
          rows="2"
          placeholder="e.g. --some-flag value --another-flag"
          style="width:100%;font-family:monospace;font-size:0.875rem"
          autoResize
        />
      </div>

      <!-- Actions -->
      <div class="config-actions">
        <Button
          label="Save Configuration"
          icon="pi pi-save"
          severity="success"
          :loading="saving"
          @click="saveConfig"
        />
        <Button
          label="Reset to Saved"
          icon="pi pi-refresh"
          severity="secondary"
          outlined
          @click="resetConfig"
        />
      </div>
    </template>
  </div>
</template>

<script setup>
import { ref, computed, watch, onMounted } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { useToast } from 'primevue/usetoast'
import axios from 'axios'
import Button from 'primevue/button'
import Tag from 'primevue/tag'
import InputText from 'primevue/inputtext'
import InputNumber from 'primevue/inputnumber'
import InputSwitch from 'primevue/inputswitch'
import Dropdown from 'primevue/dropdown'
import Textarea from 'primevue/textarea'
import ProgressSpinner from 'primevue/progressspinner'
import Slider from 'primevue/slider'
import { useModelStore } from '@/stores/models'

const route = useRoute()
const router = useRouter()
const toast = useToast()
const modelStore = useModelStore()

// ── State ──────────────────────────────────────────────────
const loading = ref(true)
const saving = ref(false)
const model = ref(null)
const config = ref({})
const savedConfig = ref({})          // for reset
const paramRegistry = ref({ basic: [], advanced: [] })
const selectedNewParam = ref(null)
const activeAdvancedKeys = ref([])   // keys of advanced params currently in the form
const modelLimits = ref(null)        // engine-agnostic: { max_context_length?, layer_count? } from /api/models/{id}/limits

const allEngineOptions = [
  { value: 'llama_cpp', label: 'llama.cpp', icon: 'pi-microchip' },
  { value: 'ik_llama',  label: 'ik_llama.cpp', icon: 'pi-microchip' },
  { value: 'lmdeploy',  label: 'LMDeploy', icon: 'pi-server' },
]

// GGUF is not compatible with LMDeploy; show LMDeploy only for safetensors
const engineOptions = computed(() => {
  const fmt = model.value?.format
  if (fmt === 'safetensors') return allEngineOptions
  return allEngineOptions.filter(eng => eng.value !== 'lmdeploy')
})

// ── Computed ───────────────────────────────────────────────
const basicParams = computed(() => paramRegistry.value.basic || [])
const allAdvancedParams = computed(() => paramRegistry.value.advanced || [])

const activeAdvancedParams = computed(() =>
  allAdvancedParams.value.filter(p => activeAdvancedKeys.value.includes(p.key))
)

const availableAdvancedParams = computed(() =>
  allAdvancedParams.value.filter(p => !activeAdvancedKeys.value.includes(p.key))
)

const maxContextSuggestion = computed(() => {
  if (!model.value) return null
  const limits = modelLimits.value
  const cfg = config.value || {}
  if (limits?.max_context_length != null && Number(limits.max_context_length) > 0) {
    return Number(limits.max_context_length)
  }
  if (cfg.session_len != null && Number(cfg.session_len) > 0) return Number(cfg.session_len)
  if (cfg.ctx_size != null && Number(cfg.ctx_size) > 0) return Number(cfg.ctx_size)
  return null
})

const layerCountSuggestion = computed(() => {
  const limits = modelLimits.value
  if (limits?.layer_count != null && Number(limits.layer_count) > 0) {
    return Number(limits.layer_count)
  }
  return null
})

// ── Helpers ────────────────────────────────────────────────
function findModelById(id) {
  for (const group of modelStore.models) {
    for (const q of group.quantizations || []) {
      if (q.id === id) return { ...q, base_model_name: group.base_model_name, huggingface_id: group.huggingface_id }
    }
  }
  // Fallback: search allQuantizations
  return modelStore.allQuantizations.find(m => m.id === id) ?? null
}

async function fetchParamRegistry(engine) {
  try {
    const { data } = await axios.get('/api/models/param-registry', { params: { engine } })
    paramRegistry.value = data
  } catch (e) {
    console.error('Failed to fetch param registry:', e)
    paramRegistry.value = { basic: [], advanced: [] }
  }
}

function detectActiveAdvancedKeys(cfg) {
  const basicKeys = new Set([
    ...(paramRegistry.value.basic || []).map(p => p.key),
    'engine', 'custom_args',
  ])
  return Object.keys(cfg).filter(
    k => !basicKeys.has(k) && cfg[k] != null && cfg[k] !== ''
  )
}

// ── Engine change ──────────────────────────────────────────
async function changeEngine(engine) {
  config.value.engine = engine
  await fetchParamRegistry(engine)
  // Recompute which advanced keys are active with new registry
  activeAdvancedKeys.value = detectActiveAdvancedKeys(config.value)
}

// ── Advanced param management ──────────────────────────────
function onNewParamSelected() {
  if (!selectedNewParam.value) return
  addAdvancedParam()
}

function addAdvancedParam() {
  if (!selectedNewParam.value) return
  const param = allAdvancedParams.value.find(p => p.key === selectedNewParam.value)
  if (!param) return
  if (!activeAdvancedKeys.value.includes(param.key)) {
    activeAdvancedKeys.value.push(param.key)
    if (config.value[param.key] == null) {
      config.value[param.key] = param.default ?? null
    }
  }
  selectedNewParam.value = null
}

function removeAdvancedParam(key) {
  activeAdvancedKeys.value = activeAdvancedKeys.value.filter(k => k !== key)
  delete config.value[key]
}

// ── Load ───────────────────────────────────────────────────
async function loadAll() {
  loading.value = true
  try {
    if (!modelStore.models.length) await modelStore.fetchModels()
    const found = findModelById(route.params.id)
    if (!found) { loading.value = false; return }
    model.value = found

    const [cfgResp, limitsResp] = await Promise.all([
      axios.get(`/api/models/${route.params.id}/config`),
      axios.get(`/api/models/${route.params.id}/limits`).catch(() => ({ data: null })),
    ])

    const cfg = cfgResp.data
    // Use saved config engine so param registry and dropdown match the selected engine
    let engine = cfg.engine ?? found.engine ?? 'llama_cpp'
    // LMDeploy can only run safetensors models; if the model format is not
    // safetensors, force engine back to llama_cpp.
    if (found.format !== 'safetensors' && engine === 'lmdeploy') {
      engine = 'llama_cpp'
    }
    await fetchParamRegistry(engine)

    const merged = { engine, ...cfg }
    config.value = merged
    savedConfig.value = JSON.parse(JSON.stringify(merged))
    activeAdvancedKeys.value = detectActiveAdvancedKeys(merged)
    modelLimits.value = limitsResp?.data ?? null
  } catch (e) {
    toast.add({ severity: 'error', summary: 'Failed to load config', detail: e.message, life: 4000 })
  } finally {
    loading.value = false
  }
}

// ── Save ───────────────────────────────────────────────────
async function saveConfig() {
  saving.value = true
  try {
    // Save *only* keys that belong to the current engine form.
    // This avoids leaving behind params from a previously selected engine.
    const basicKeys = new Set((paramRegistry.value.basic || []).map(p => p.key))
    const advancedKeys = new Set((paramRegistry.value.advanced || []).map(p => p.key))
    const activeAdvancedKeySet = new Set(activeAdvancedKeys.value || [])

    const toSave = {}

    // Always persist engine itself.
    toSave.engine = config.value.engine

    // Persist basic keys for the currently selected engine.
    for (const key of basicKeys) {
      if (Object.prototype.hasOwnProperty.call(config.value, key)) {
        toSave[key] = config.value[key]
      }
    }

    // Persist only advanced keys that are active AND present in the current registry.
    for (const key of advancedKeys) {
      if (activeAdvancedKeySet.has(key) && Object.prototype.hasOwnProperty.call(config.value, key)) {
        toSave[key] = config.value[key]
      }
    }

    // Drop any keys that are effectively "unset" (keep false/0).
    for (const [key, value] of Object.entries(toSave)) {
      if (value == null || value === '' || (typeof value === 'number' && Number.isNaN(value))) {
        delete toSave[key]
      }
    }
    await axios.put(`/api/models/${route.params.id}/config`, toSave)
    savedConfig.value = JSON.parse(JSON.stringify(toSave))
    toast.add({ severity: 'success', summary: 'Saved', detail: 'Configuration saved', life: 2000 })
  } catch (e) {
    toast.add({ severity: 'error', summary: 'Save failed', detail: e.message, life: 4000 })
  } finally {
    saving.value = false
  }
}

// ── Reset ──────────────────────────────────────────────────
function resetConfig() {
  config.value = JSON.parse(JSON.stringify(savedConfig.value))
  activeAdvancedKeys.value = detectActiveAdvancedKeys(config.value)
  toast.add({ severity: 'info', summary: 'Reset', detail: 'Config reset to saved values', life: 2000 })
}

// ── Lifecycle ──────────────────────────────────────────────
onMounted(loadAll)
</script>

<style scoped>
.model-config-view {
  max-width: 960px;
  margin: 0 auto;
  padding: var(--spacing-lg, 1.5rem);
  display: flex;
  flex-direction: column;
  gap: var(--spacing-lg, 1.5rem);
}

/* ── Loading / Empty ──────────────────────────────────── */
.loading-state,
.empty-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 1rem;
  padding: 4rem 0;
  color: var(--text-secondary, #9ca3af);
}

/* ── Header ───────────────────────────────────────────── */
.config-header {
  display: flex;
  align-items: flex-start;
  gap: 0.75rem;
}

.header-info { flex: 1; }

.header-info h1 {
  font-size: 1.25rem;
  font-weight: 700;
  margin: 0 0 0.4rem;
  line-height: 1.2;
}

.header-meta {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  flex-wrap: wrap;
}

.hf-link {
  font-size: 0.875rem;
  color: var(--accent-cyan, #22d3ee);
  text-decoration: none;
  display: flex;
  align-items: center;
  gap: 0.25rem;
}

.hf-link:hover { text-decoration: underline; }

/* ── Card ─────────────────────────────────────────────── */
.config-card {
  background: var(--bg-card, #161b2e);
  border: 1px solid var(--border-primary, #2a2f45);
  border-radius: var(--radius-lg, 0.75rem);
  padding: 1.25rem;
}

.section-label {
  font-size: 0.75rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--text-secondary, #9ca3af);
  margin-bottom: 0.875rem;
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.section-count {
  background: var(--accent-cyan, #22d3ee);
  color: #000;
  border-radius: 999px;
  padding: 0.1em 0.5em;
  font-size: 0.7rem;
  font-weight: 600;
}

.section-hint {
  font-weight: 400;
  text-transform: none;
  letter-spacing: normal;
  color: var(--text-secondary, #9ca3af);
  opacity: 0.7;
}

/* ── Engine selector ──────────────────────────────────── */
.engine-selector {
  display: flex;
  gap: 0.5rem;
  flex-wrap: wrap;
}

.engine-option {
  display: inline-flex;
  align-items: center;
  padding: 0.5rem 1rem;
  border-radius: var(--radius-md, 0.5rem);
  border: 1px solid var(--border-primary, #2a2f45);
  cursor: pointer;
  transition: all 0.15s;
  font-size: 0.875rem;
  user-select: none;
}

.engine-option:hover {
  border-color: var(--accent-cyan, #22d3ee);
  background: rgba(34, 211, 238, 0.05);
}

.engine-option.selected {
  border-color: var(--accent-cyan, #22d3ee);
  background: rgba(34, 211, 238, 0.1);
  color: var(--accent-cyan, #22d3ee);
  font-weight: 600;
}

.engine-option-label {
  display: inline-flex;
  align-items: center;
  gap: 0.5rem;
}

.engine-name {
  font-size: 0.875rem;
}

.engine-mark {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  min-width: 1.6rem;
  height: 1.6rem;
  padding: 0 0.4rem;
  border-radius: 999px;
  font-size: 0.7rem;
  font-weight: 700;
  line-height: 1;
  letter-spacing: 0.04em;
  color: #fff;
  box-shadow: inset 0 0 0 1px rgba(255, 255, 255, 0.1);
}

.engine-mark--llama {
  background: linear-gradient(135deg, #0ea5e9, #2563eb);
}

.engine-mark--ik {
  background: linear-gradient(135deg, #8b5cf6, #ec4899);
}

.engine-icon-lmdeploy {
  font-size: 1.1rem;
  color: var(--accent-cyan, #22d3ee);
}

/* ── Params grid ──────────────────────────────────────── */
.params-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(240px, 1fr));
  gap: 0.875rem;
}

.param-field {
  display: flex;
  flex-direction: column;
  gap: 0.25rem;
}

.param-field label {
  font-size: 0.8rem;
  font-weight: 500;
  color: var(--text-secondary, #9ca3af);
  display: flex;
  align-items: center;
  gap: 0.3rem;
}

.param-input { width: 100%; }

.param-info {
  font-size: 0.7rem;
  cursor: help;
  opacity: 0.6;
}

.remove-param-btn {
  margin-left: auto;
  padding: 0 !important;
  height: auto !important;
  width: auto !important;
}

/* ── Add param row ────────────────────────────────────── */
.add-param-row {
  display: flex;
  gap: 0.5rem;
  align-items: center;
}

.add-param-dropdown { flex: 1; }

.param-option {
  display: flex;
  flex-direction: column;
  gap: 0.15rem;
}

.param-option-label { font-size: 0.875rem; font-weight: 500; }
.param-option-desc  { font-size: 0.75rem; color: var(--text-secondary, #9ca3af); }

/* ── Actions ──────────────────────────────────────────── */
.config-actions {
  display: flex;
  gap: 0.75rem;
  justify-content: flex-end;
  padding-bottom: var(--spacing-lg, 1.5rem);
}

.param-slider-row {
  display: flex;
  flex-direction: column;
  gap: 0.25rem;
  margin-bottom: 0.25rem;
}

.param-slider {
  width: 100%;
  max-width: 15rem;
}

/* Align PrimeVue slider handle with the track bar */
.param-slider :deep(.p-slider) {
  height: 0.5rem;
}
.param-slider :deep(.p-slider-handle) {
  width: 1rem;
  height: 1rem;
  top: 50%;
  margin-top: -0.5rem;
}

.param-hint {
  font-size: 0.75rem;
  color: var(--text-secondary, #9ca3af);
}
</style>
