<template>
  <div class="model-config-view page-shell page-shell--relaxed">

    <LoadingState v-if="loading" message="Loading configuration…" />

    <EmptyState
      v-else-if="!model"
      icon="pi pi-exclamation-circle"
      title="Model not found"
    >
      <Button label="Back to Models" icon="pi pi-arrow-left" @click="$router.push('/models')" />
    </EmptyState>

    <template v-else>
      <PageHeader>
        <template #start>
          <Button icon="pi pi-arrow-left" text severity="secondary" @click="$router.push('/models')" />
        </template>
        <template #title>
          <div class="config-page-title">
            <h1 class="page-title">{{ model.display_name || model.base_model_name }}</h1>
            <div class="header-meta">
              <Tag :value="model.format || 'gguf'" severity="info" />
              <Tag v-if="model.quantization" :value="model.quantization" severity="secondary" />
              <a :href="`https://huggingface.co/${model.huggingface_id}`" target="_blank" class="hf-link">
                <i class="pi pi-external-link" /> {{ model.huggingface_id }}
              </a>
            </div>
          </div>
        </template>
        <template v-if="!loading && model && hasUnsavedChanges" #actions>
          <Tag value="Unsaved changes" severity="warning" class="unsaved-tag" />
        </template>
      </PageHeader>

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

      <Message
        v-if="paramRegistry.scan_error"
        severity="warn"
        :closable="false"
        class="config-scan-message"
      >
        Could not read engine CLI help: {{ paramRegistry.scan_error }}. Open Engines and use
        <strong>Rescan CLI parameters</strong> for this engine.
      </Message>
      <Message
        v-else-if="paramRegistry.scan_pending"
        severity="info"
        :closable="false"
        class="config-scan-message"
      >
        Parameter catalog not loaded yet. It will populate after the next engine install or when you rescan from the Engines page.
      </Message>

      <!-- Catalog-backed: filters + quick nav + collapsible sections -->
      <template v-if="catalogSections.length">
        <div class="config-card config-toolbar">
          <div class="config-toolbar__row">
            <span class="p-input-icon-left config-search-wrap">
              <i class="pi pi-search" aria-hidden="true" />
              <InputText
                v-model="paramSearchQuery"
                type="search"
                placeholder="Search by name, key, flag, or description…"
                class="config-search-input"
                aria-label="Filter parameters"
              />
            </span>
            <Button
              v-if="paramSearchQuery"
              icon="pi pi-times"
              text
              rounded
              severity="secondary"
              v-tooltip.top="'Clear search'"
              aria-label="Clear search"
              @click="paramSearchQuery = ''"
            />
          </div>
          <div class="config-toolbar__row config-toolbar__toggles">
            <div class="toggle-field">
              <InputSwitch v-model="hideUnsupportedParams" input-id="toggle-hide-unsupported" />
              <label for="toggle-hide-unsupported">Hide unsupported in this build</label>
            </div>
            <div class="toolbar-actions">
              <Button
                label="Expand all"
                text
                size="small"
                severity="secondary"
                @click="setAllSectionsExpanded(true)"
              />
              <Button
                label="Collapse all"
                text
                size="small"
                severity="secondary"
                @click="setAllSectionsExpanded(false)"
              />
            </div>
          </div>
          <nav
            v-if="catalogSections.length > 1"
            class="config-section-nav"
            aria-label="Jump to section"
          >
            <span class="config-section-nav__label">Jump to</span>
            <a
              v-for="sec in catalogSections"
              :key="`nav-${sec.id}`"
              :href="`#cfg-sec-${sec.id}`"
              class="config-section-nav__link"
              @click.prevent="scrollToSection(sec.id)"
            >
              {{ sec.label }}
            </a>
          </nav>
        </div>

        <Message
          v-if="filteredCatalogSections.length === 0"
          severity="secondary"
          :closable="false"
          class="config-scan-message"
        >
          No parameters match your filters. Try clearing the search or turning off “hide unsupported”.
        </Message>

        <div
          v-for="section in filteredCatalogSections"
          :key="section.id"
          :id="`cfg-sec-${section.id}`"
          class="config-card config-section-card"
        >
          <button
            type="button"
            class="section-header-btn"
            :aria-expanded="isSectionExpanded(section.id)"
            @click="toggleSectionExpanded(section.id)"
          >
            <i
              class="pi section-chevron"
              :class="isSectionExpanded(section.id) ? 'pi-chevron-down' : 'pi-chevron-right'"
              aria-hidden="true"
            />
            <span class="section-header-title">{{ section.label }}</span>
            <span class="section-header-meta">
              {{ section.params.length }}<template v-if="paramSearchQuery.trim() || hideUnsupportedParams">
                / {{ sectionParamCount(section.id) }}</template>
            </span>
          </button>
          <div v-show="isSectionExpanded(section.id)" class="params-grid section-params">
            <div
              v-for="param in section.params"
              :key="`${section.id}-${param.key}`"
              class="param-field"
              :class="{ 'param-field--unsupported': param.supported === false }"
            >
              <label :for="`p-${section.id}-${param.key}`">
                {{ param.label }}
                <code v-if="paramSearchQuery.trim()" class="param-key-hint">{{ param.key }}</code>
                <Tag
                  v-if="param.supported === false"
                  value="Not in this build"
                  severity="secondary"
                  class="param-supported-tag"
                />
                <i class="pi pi-info-circle param-info" v-tooltip.top="paramDescriptionTooltip(param)" />
              </label>
              <template v-if="param.type === 'int' && (param.key === 'ctx_size' || param.key === 'session_len')">
                <div class="param-slider-row">
                  <Slider
                    v-model="config[param.key]"
                    :min="512"
                    :max="maxContextSuggestion || 131072"
                    :step="256"
                    class="param-slider"
                    :disabled="param.supported === false"
                  />
                  <span v-if="maxContextSuggestion" class="param-hint">
                    Suggested max: {{ maxContextSuggestion.toLocaleString() }} tokens
                  </span>
                </div>
                <InputNumber
                  :id="`p-${section.id}-${param.key}`"
                  v-model="config[param.key]"
                  :placeholder="String(param.default ?? '')"
                  class="param-input"
                  :disabled="param.supported === false"
                />
              </template>
              <template v-else-if="param.type === 'int' && param.key === 'n_gpu_layers'">
                <div class="param-slider-row">
                  <Slider
                    v-model="config[param.key]"
                    :min="0"
                    :max="layerCountSuggestion || 128"
                    :step="1"
                    class="param-slider"
                    :disabled="param.supported === false"
                  />
                  <span v-if="layerCountSuggestion" class="param-hint">
                    Detected layers: {{ layerCountSuggestion }}
                  </span>
                </div>
                <InputNumber
                  :id="`p-${section.id}-${param.key}`"
                  v-model="config[param.key]"
                  :placeholder="String(param.default ?? '')"
                  class="param-input"
                  :disabled="param.supported === false"
                />
              </template>
              <Dropdown
                v-else-if="param.options && param.options.length"
                :id="`p-${section.id}-${param.key}`"
                v-model="config[param.key]"
                :options="param.options"
                optionLabel="label"
                optionValue="value"
                :placeholder="param.default != null ? String(param.default) : ''"
                class="param-input"
                :disabled="param.supported === false"
              />
              <InputNumber
                v-else-if="param.type === 'int'"
                :id="`p-${section.id}-${param.key}`"
                v-model="config[param.key]"
                :placeholder="String(param.default ?? '')"
                class="param-input"
                :disabled="param.supported === false"
              />
              <InputNumber
                v-else-if="param.type === 'float'"
                :id="`p-${section.id}-${param.key}`"
                v-model="config[param.key]"
                :minFractionDigits="1"
                :maxFractionDigits="4"
                :placeholder="String(param.default ?? '')"
                class="param-input"
                :disabled="param.supported === false"
              />
              <InputSwitch
                v-else-if="param.type === 'bool'"
                :id="`p-${section.id}-${param.key}`"
                v-model="config[param.key]"
                :disabled="param.supported === false"
              />
              <InputText
                v-else
                :id="`p-${section.id}-${param.key}`"
                v-model="config[param.key]"
                :placeholder="param.default != null ? String(param.default) : ''"
                class="param-input"
                :disabled="param.supported === false"
              />
            </div>
          </div>
        </div>
      </template>

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
          class="w-full textarea-cli"
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
import { ref, computed, onMounted, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { useToast } from 'primevue/usetoast'
import axios from 'axios'
import Button from 'primevue/button'
import Tag from 'primevue/tag'
import InputText from 'primevue/inputtext'
import InputNumber from 'primevue/inputnumber'
import InputSwitch from 'primevue/inputswitch'
import Dropdown from 'primevue/dropdown'
import Message from 'primevue/message'
import Textarea from 'primevue/textarea'
import Slider from 'primevue/slider'
import LoadingState from '@/components/common/LoadingState.vue'
import EmptyState from '@/components/common/EmptyState.vue'
import PageHeader from '@/components/common/PageHeader.vue'
import { useModelStore } from '@/stores/models'
import { useEnginesStore } from '@/stores/engines'

const route = useRoute()
const router = useRouter()
const toast = useToast()
const modelStore = useModelStore()
const enginesStore = useEnginesStore()

// ── State ──────────────────────────────────────────────────
const loading = ref(true)
const saving = ref(false)
const model = ref(null)
const config = ref({})
const savedConfig = ref({})          // for reset
const paramRegistry = ref({
  sections: [],
  scan_error: null,
  scan_pending: false,
})
const paramSearchQuery = ref('')
const hideUnsupportedParams = ref(false)
/** section id -> expanded (omitted / true = expanded, false = collapsed) */
const sectionExpanded = ref({})
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
const catalogSections = computed(() =>
  Array.isArray(paramRegistry.value.sections) ? paramRegistry.value.sections : [],
)

/** Flat list of all catalog params (sections order). */
const catalogParamList = computed(() => {
  const out = []
  for (const s of catalogSections.value) {
    for (const p of s.params || []) out.push(p)
  }
  return out
})

const hasUnsavedChanges = computed(() => {
  try {
    return JSON.stringify(config.value) !== JSON.stringify(savedConfig.value)
  } catch {
    return false
  }
})

function paramVisibleInFilters(param) {
  if (hideUnsupportedParams.value && param.supported === false) return false
  const q = paramSearchQuery.value.trim().toLowerCase()
  if (!q) return true
  const hay = [
    param.label || '',
    param.key || '',
    param.description || '',
    ...(param.flags || []),
  ]
    .join(' ')
    .toLowerCase()
  return hay.includes(q)
}

const filteredCatalogSections = computed(() =>
  catalogSections.value
    .map(sec => ({
      ...sec,
      params: (sec.params || []).filter(paramVisibleInFilters),
    }))
    .filter(sec => sec.params.length > 0),
)

function sectionParamCount(sectionId) {
  const sec = catalogSections.value.find(s => s.id === sectionId)
  return (sec?.params || []).length
}

function isSectionExpanded(sectionId) {
  return sectionExpanded.value[sectionId] !== false
}

function toggleSectionExpanded(sectionId) {
  sectionExpanded.value = {
    ...sectionExpanded.value,
    [sectionId]: !isSectionExpanded(sectionId),
  }
}

function setAllSectionsExpanded(expanded) {
  const next = {}
  for (const s of catalogSections.value) {
    next[s.id] = expanded
  }
  sectionExpanded.value = next
}

function scrollToSection(sectionId) {
  document.getElementById(`cfg-sec-${sectionId}`)?.scrollIntoView({
    behavior: 'smooth',
    block: 'start',
  })
  if (!isSectionExpanded(sectionId)) {
    toggleSectionExpanded(sectionId)
  }
}

function paramDescriptionTooltip(param) {
  const parts = [param.description].filter(Boolean)
  if (param.flags?.length) {
    parts.push(`CLI: ${param.flags.join(', ')}`)
  }
  return parts.join('\n\n') || param.label || param.key
}

watch(
  catalogSections,
  sections => {
    const total = sections.reduce((n, s) => n + (s.params?.length || 0), 0)
    const next = { ...sectionExpanded.value }
    const ids = new Set(sections.map(s => s.id))
    for (const k of Object.keys(next)) {
      if (!ids.has(k)) delete next[k]
    }
    for (const s of sections) {
      if (next[s.id] === undefined) {
        next[s.id] =
          total > 48 ? Boolean(s.studio_only || s.id === 'studio') : true
      }
    }
    sectionExpanded.value = next
  },
  { deep: true, immediate: true },
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
    const { data } = await axios.get('/api/models/param-registry', {
      params: { engine, dynamic: true },
    })
    paramRegistry.value = {
      sections: data.sections || [],
      scan_error: data.scan_error ?? null,
      scan_pending: Boolean(data.scan_pending),
    }
  } catch (e) {
    console.error('Failed to fetch param registry:', e)
    paramRegistry.value = {
      sections: [],
      scan_error: null,
      scan_pending: false,
    }
  }
}

function buildWorkingConfigFromApi(cfg) {
  const engines =
    cfg.engines && typeof cfg.engines === 'object'
      ? JSON.parse(JSON.stringify(cfg.engines))
      : {}
  const engine = cfg.engine ?? 'llama_cpp'
  const sec = engines[engine] || {}
  return {
    engine,
    engines,
    ...sec,
  }
}

function stashCurrentEngineIntoEngines(engineKey) {
  if (!engineKey) return
  const stash = {
    ...((config.value.engines && config.value.engines[engineKey]) || {}),
  }
  if (catalogSections.value.length) {
    const keys = new Set()
    for (const s of catalogSections.value) {
      for (const p of s.params || []) keys.add(p.key)
    }
    for (const key of keys) {
      if (Object.prototype.hasOwnProperty.call(config.value, key)) {
        stash[key] = config.value[key]
      }
    }
  } else {
    for (const key of Object.keys(config.value)) {
      if (key === 'engine' || key === 'engines') continue
      stash[key] = config.value[key]
    }
  }
  if (Object.prototype.hasOwnProperty.call(config.value, 'custom_args')) {
    stash.custom_args = config.value.custom_args
  }
  for (const [k, v] of Object.entries(stash)) {
    if (v == null || v === '' || (typeof v === 'number' && Number.isNaN(v))) {
      delete stash[k]
    }
  }
  if (!config.value.engines) config.value.engines = {}
  config.value.engines[engineKey] = stash
}

function applyEngineSectionToForm(engine) {
  const sec = (config.value.engines && config.value.engines[engine]) || {}
  const params = catalogParamList.value
  if (!params.length) {
    const eng = config.value.engine
    const engMap = config.value.engines
    for (const k of Object.keys(config.value)) {
      if (k !== 'engine' && k !== 'engines') delete config.value[k]
    }
    Object.assign(config.value, sec)
    config.value.engine = eng
    config.value.engines = engMap
    return
  }
  const allowed = new Set(['engine', 'engines', 'custom_args', ...params.map(p => p.key)])
  for (const k of Object.keys(config.value)) {
    if (!allowed.has(k)) delete config.value[k]
  }
  for (const p of params) {
    const v = sec[p.key]
    config.value[p.key] =
      v !== undefined && v !== null && v !== '' ? v : (p.default ?? null)
  }
}

// ── Engine change ──────────────────────────────────────────
async function changeEngine(engine) {
  if (engine === config.value.engine) return
  stashCurrentEngineIntoEngines(config.value.engine)
  config.value.engine = engine
  paramSearchQuery.value = ''
  hideUnsupportedParams.value = false
  await fetchParamRegistry(engine)
  applyEngineSectionToForm(engine)
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

    const merged = buildWorkingConfigFromApi({ ...cfg, engine })
    config.value = merged
    applyEngineSectionToForm(engine)
    savedConfig.value = JSON.parse(JSON.stringify(config.value))
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
    stashCurrentEngineIntoEngines(config.value.engine)
    const payload = {
      engine: config.value.engine,
      engines: JSON.parse(JSON.stringify(config.value.engines || {})),
    }
    const { data } = await axios.put(`/api/models/${route.params.id}/config`, payload)
    const merged = buildWorkingConfigFromApi(data)
    config.value = merged
    applyEngineSectionToForm(config.value.engine)
    savedConfig.value = JSON.parse(JSON.stringify(config.value))
    void enginesStore.fetchSwapConfigPending()
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
  toast.add({ severity: 'info', summary: 'Reset', detail: 'Config reset to saved values', life: 2000 })
}

// ── Lifecycle ──────────────────────────────────────────────
onMounted(loadAll)
</script>

<style scoped>
/* layout: .page-shell.page-shell--relaxed */

.config-scan-message {
  margin-bottom: 1rem;
}

.config-page-title {
  display: flex;
  flex-direction: column;
  align-items: flex-start;
  gap: 0.4rem;
  min-width: 0;
}

.header-meta {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  flex-wrap: wrap;
}

.textarea-cli {
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
  font-size: 0.875rem;
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

.param-field--unsupported {
  opacity: 0.88;
}

.param-supported-tag {
  margin-left: 0.35rem;
  vertical-align: middle;
  font-size: 0.65rem !important;
}

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

/* ── Actions (sticky bar) ───────────────────────────────── */
.config-actions {
  display: flex;
  gap: 0.75rem;
  justify-content: flex-end;
  padding-bottom: var(--spacing-lg, 1.5rem);
  position: sticky;
  bottom: 0;
  z-index: 10;
  background: linear-gradient(
    to top,
    var(--bg-primary, #0f111a) 65%,
    transparent
  );
  padding-top: 0.75rem;
  margin-top: 0.5rem;
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

/* ── Unsaved indicator ─────────────────────────────────── */
.unsaved-tag {
  font-size: 0.75rem;
}

/* ── Catalog toolbar (search, toggles, jump nav) ───────── */
.config-toolbar {
  margin-bottom: 1rem;
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
}

.config-toolbar__row {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 0.5rem;
}

.config-toolbar__toggles {
  justify-content: space-between;
  gap: 1rem;
}

.config-search-wrap {
  position: relative;
  flex: 1;
  min-width: min(100%, 12rem);
}

.config-search-wrap .pi-search {
  position: absolute;
  left: 0.75rem;
  top: 50%;
  transform: translateY(-50%);
  color: var(--text-secondary, #9ca3af);
  pointer-events: none;
  z-index: 1;
  font-size: 0.875rem;
}

.config-search-wrap :deep(.p-inputtext),
.config-search-wrap :deep(input.config-search-input) {
  width: 100%;
  padding-left: 2.35rem;
}

.toggle-field {
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.toggle-field label {
  font-size: 0.875rem;
  color: var(--text-secondary, #9ca3af);
  cursor: pointer;
  user-select: none;
}

.toolbar-actions {
  display: flex;
  flex-wrap: wrap;
  gap: 0.25rem;
  margin-left: auto;
}

.config-section-nav {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 0.35rem 0.6rem;
  padding-top: 0.5rem;
  border-top: 1px solid var(--border-primary, #2a2f45);
}

.config-section-nav__label {
  font-size: 0.7rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: var(--text-secondary, #9ca3af);
  margin-right: 0.15rem;
}

.config-section-nav__link {
  font-size: 0.8125rem;
  color: var(--accent-cyan, #22d3ee);
  text-decoration: none;
  opacity: 0.9;
}

.config-section-nav__link:hover {
  text-decoration: underline;
  opacity: 1;
}

/* ── Collapsible section cards ─────────────────────────── */
.config-section-card {
  scroll-margin-top: 5rem;
  margin-bottom: 0.75rem;
}

.section-header-btn {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  width: 100%;
  padding: 0 0 0.75rem;
  margin: 0;
  border: none;
  background: transparent;
  cursor: pointer;
  text-align: left;
  color: var(--text-primary, #e5e7eb);
  font-size: 1rem;
  border-radius: var(--radius-sm, 0.25rem);
}

.section-header-btn:focus-visible {
  outline: 2px solid var(--accent-cyan, #22d3ee);
  outline-offset: 2px;
}

.section-header-btn:hover .section-header-title {
  color: var(--accent-cyan, #22d3ee);
}

.section-chevron {
  flex-shrink: 0;
  font-size: 0.85rem;
  opacity: 0.75;
}

.section-header-title {
  flex: 1;
  font-weight: 600;
  min-width: 0;
}

.section-header-meta {
  flex-shrink: 0;
  font-size: 0.75rem;
  font-variant-numeric: tabular-nums;
  color: var(--text-secondary, #9ca3af);
}

.section-params {
  padding-top: 0.5rem;
  border-top: 1px solid var(--border-primary, #2a2f45);
}

.param-key-hint {
  margin-left: 0.35rem;
  padding: 0.1rem 0.35rem;
  font-size: 0.65rem;
  font-weight: 400;
  color: var(--text-secondary, #9ca3af);
  background: rgba(0, 0, 0, 0.25);
  border-radius: 0.25rem;
  vertical-align: middle;
}
</style>
