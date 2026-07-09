<template>
  <div class="audio-model-config">
    <Message
      v-if="paramRegistry.scan_error"
      severity="warn"
      :closable="false"
      class="audio-config-message"
    >
      <div class="audio-config-message__body">
        <strong>CLI parameters could not be loaded.</strong>
        {{ paramRegistry.scan_error }}
        <Button
          label="Rescan CLI parameters"
          icon="pi pi-refresh"
          size="small"
          severity="secondary"
          outlined
          :loading="rescanLoading"
          class="audio-config-message__action"
          @click="rescanCliParams"
        />
      </div>
    </Message>
    <Message
      v-else-if="paramRegistry.scan_pending"
      severity="info"
      :closable="false"
      class="audio-config-message"
    >
      <div class="audio-config-message__body">
        <strong>CLI parameters not indexed yet.</strong>
        Activate audio.cpp on the Engines page, then rescan.
        <Button
          label="Rescan now"
          icon="pi pi-refresh"
          size="small"
          severity="secondary"
          outlined
          :loading="rescanLoading"
          class="audio-config-message__action"
          @click="rescanCliParams"
        />
      </div>
    </Message>

    <div class="audio-header-strip config-card">
      <div class="audio-header-strip__main">
        <Tag
          :value="taskKindMeta.short"
          :severity="taskKindMeta.tagSeverity"
          class="audio-header-strip__kind"
        />
        <div>
          <div class="audio-header-strip__title">{{ taskKindMeta.label }} configuration</div>
          <div class="audio-header-strip__meta">
            <code>{{ apiEndpoint }}</code>
            <span v-if="config.family" class="audio-header-strip__sep">·</span>
            <span v-if="config.family">{{ config.family }}</span>
            <span v-if="config.task" class="audio-header-strip__sep">·</span>
            <span v-if="config.task">{{ config.task }}</span>
          </div>
        </div>
      </div>
      <div class="audio-header-strip__actions">
        <Button
          label="Defaults"
          icon="pi pi-sliders-h"
          size="small"
          severity="secondary"
          outlined
          type="button"
          @click="activeTab = 'api'"
        />
        <Button
          label="API example"
          icon="pi pi-book"
          size="small"
          severity="secondary"
          outlined
          type="button"
          @click="activeTab = 'reference'"
        />
      </div>
    </div>

    <nav class="audio-tabs" aria-label="Audio configuration sections">
      <button
        v-for="tab in tabs"
        :key="tab.id"
        type="button"
        class="audio-tab"
        :class="{ 'audio-tab--active': activeTab === tab.id }"
        :aria-selected="activeTab === tab.id"
        @click="activeTab = tab.id"
      >
        <i :class="tab.icon" aria-hidden="true" />
        <span class="audio-tab__label">{{ tab.label }}</span>
        <span v-if="tab.hint" class="audio-tab__hint">{{ tab.hint }}</span>
      </button>
    </nav>

    <!-- Overview -->
    <div v-show="activeTab === 'overview'" class="audio-tab-panel">
      <div v-if="taskProfile" class="config-card audio-profile-hero">
        <div class="audio-profile-hero__head">
          <div>
            <div class="section-label section-label--inline">
              {{ taskProfile.label || 'Model profile' }}
            </div>
            <p v-if="taskProfile.summary" class="config-muted-hint">{{ taskProfile.summary }}</p>
          </div>
          <Tag :value="`${setupProgress}% ready`" :severity="setupProgress === 100 ? 'success' : 'info'" />
        </div>
        <div v-if="taskWorkflowTags.length" class="audio-capability-tags">
          <Tag
            v-for="workflow in taskWorkflowTags"
            :key="workflow"
            :value="workflow"
            severity="secondary"
          />
        </div>
        <p v-if="taskProfile.api_hint" class="config-muted-hint">{{ taskProfile.api_hint }}</p>
      </div>

      <div v-if="audioInspectionSummary.length" class="config-card">
        <div class="section-label">
          Instpected bundle
          <small class="section-hint">Capabilities read from the installed package</small>
        </div>
        <div class="audio-capability-tags">
          <Tag
            v-for="item in audioInspectionSummary"
            :key="item"
            :value="item"
            severity="info"
          />
        </div>
      </div>

      <div class="config-card">
        <div class="section-label">
          Setup checklist
          <small class="section-hint">Complete these before saving and applying to llama-swap</small>
        </div>
        <ul class="audio-checklist">
          <li
            v-for="item in setupChecklist"
            :key="item.id"
            class="audio-checklist__item"
            :class="{ 'audio-checklist__item--done': item.done }"
          >
            <i
              class="pi"
              :class="item.done ? 'pi-check-circle' : 'pi-circle'"
              aria-hidden="true"
            />
            <div>
              <strong>{{ item.label }}</strong>
              <small>{{ item.detail }}</small>
            </div>
            <Button
              v-if="item.tab"
              :label="item.tab === 'api' ? 'Edit defaults' : 'Open'"
              size="small"
              text
              type="button"
              @click="activeTab = item.tab"
            />
          </li>
        </ul>
        <div class="audio-checklist__actions">
          <Button
            label="Configure runtime"
            icon="pi pi-server"
            size="small"
            severity="secondary"
            outlined
            @click="activeTab = 'server'"
          />
          <Button
            v-if="isProfiledAudioModel"
            label="Set defaults"
            icon="pi pi-sliders-h"
            size="small"
            severity="secondary"
            outlined
            @click="activeTab = 'api'"
          />
        </div>
      </div>

      <details class="config-card audio-guide-details">
        <summary class="audio-guide-details__summary">How configuration is applied</summary>
        <div class="audio-guide-grid">
          <div class="audio-guide-item audio-guide-item--server">
            <div class="audio-guide-item__badge">Sidecar</div>
            <p>Runtime, load/session options, and voice presets start with the audio.cpp server.</p>
          </div>
          <div class="audio-guide-item audio-guide-item--studio">
            <div class="audio-guide-item__badge">llama-swap setParams</div>
            <p>
              Saved defaults for <code>{{ requestDefaultsKey }}</code> are injected into JSON
              <code>{{ apiEndpoint }}</code> requests when you apply llama-swap config.
            </p>
          </div>
          <div class="audio-guide-item audio-guide-item--request">
            <div class="audio-guide-item__badge">Per request</div>
            <p>Input media and one-off overrides stay in each API call — see the API tab.</p>
          </div>
        </div>
      </details>
    </div>

    <!-- Server -->
    <div v-show="activeTab === 'server'" class="audio-tab-panel">
      <div class="config-card config-toolbar">
        <div class="config-toolbar__row">
          <span class="p-input-icon-left config-search-wrap">
            <i class="pi pi-search" aria-hidden="true" />
            <InputText
              v-model="serverSearchQuery"
              type="search"
              placeholder="Filter server parameters…"
              class="config-search-input"
              aria-label="Filter server parameters"
            />
          </span>
          <Button
            v-if="serverSearchQuery"
            icon="pi pi-times"
            text
            rounded
            severity="secondary"
            aria-label="Clear search"
            @click="serverSearchQuery = ''"
          />
        </div>
        <div class="config-toolbar__row config-toolbar__toggles">
          <div class="toggle-field">
            <InputSwitch v-model="hideUnsupportedParams" input-id="audio-hide-unsupported" />
            <label for="audio-hide-unsupported">Hide unsupported in this build</label>
          </div>
        </div>
      </div>

      <div
        v-for="group in visibleServerGroups"
        :key="group.id"
        class="config-card audio-server-group"
      >
        <button
          type="button"
          class="audio-group-toggle"
          :aria-expanded="expandedGroups[group.id]"
          @click="toggleGroup(group.id)"
        >
          <span class="audio-group-toggle__main">
            <i
              class="pi"
              :class="expandedGroups[group.id] ? 'pi-chevron-down' : 'pi-chevron-right'"
              aria-hidden="true"
            />
            <span>
              <span class="section-label section-label--inline">{{ group.label }}</span>
              <Tag :value="String(group.params.length)" severity="secondary" />
            </span>
          </span>
          <small class="section-hint audio-group-toggle__hint">{{ group.description }}</small>
        </button>

        <div v-show="expandedGroups[group.id]" class="audio-group-body">
          <div class="params-grid section-params">
            <div
              v-for="param in group.params"
              :key="`${param.scope}-${param.key}`"
              class="param-field"
              :class="{ 'param-field--unsupported': param.supported === false }"
            >
              <div class="param-field__head">
                <label :for="`audio-${param.scope}-${param.key}`" class="param-field__label">
                  {{ param.label }}
                  <code class="param-key-hint">{{ paramStorageKey(param) }}</code>
                  <Tag v-if="param.required" value="Required" severity="danger" />
                  <Tag v-if="param.asset_selector" value="Bundle asset" severity="secondary" />
                  <Tag value="Sidecar" severity="success" class="audio-sidecar-tag" />
                  <i
                    class="pi pi-info-circle param-info"
                    v-tooltip.top="paramDescriptionTooltip(param)"
                  />
                </label>
              </div>
              <p v-if="param.key === 'lazy_load'" class="field-inline-hint">
                {{ param.description }}
              </p>
              <AudioParamField
                :id="`audio-${param.scope}-${param.key}`"
                :param="param"
                :model-value="audioParamValue(param)"
                :options="audioParamOptions(param)"
                :disabled="param.supported === false"
                @update:model-value="(value) => setAudioParamValue(param, value)"
                @update:json="(value) => updateAudioJsonParam(param, value)"
              />
            </div>
          </div>
          <Message
            v-if="!group.params.length"
            severity="secondary"
            :closable="false"
            class="config-scan-message"
          >
            No parameters match your filter in this group.
          </Message>
        </div>
      </div>
    </div>

    <!-- API defaults -->
    <div v-show="activeTab === 'api'" class="audio-tab-panel">
      <div class="config-card">
        <div class="section-label">
          {{ requestDefaultsSectionTitle }}
          <small class="section-hint">
            Defaults for <code>{{ apiEndpoint }}</code>
          </small>
        </div>

        <p class="config-muted-hint audio-defaults-hint">{{ defaultsApplyHint }}</p>

        <div v-if="swapSetParamsPreview" class="audio-setparams-preview">
          <div class="audio-setparams-preview__label">
            llama-swap <code>filters.setParams</code> preview
          </div>
          <pre class="audio-setparams-preview__code">{{ JSON.stringify(swapSetParamsPreview, null, 2) }}</pre>
        </div>

        <template v-if="isProfiledAudioModel">
          <div v-if="supportsVoicePresets" class="audio-subsection">
            <div class="audio-subsection__head">
              <div>
                <span class="audio-subsection__title">Voice presets</span>
                <Tag value="Applied via sidecar" severity="success" />
              </div>
              <Button
                label="Add preset"
                icon="pi pi-plus"
                size="small"
                severity="secondary"
                outlined
                type="button"
                @click="addVoicePreset"
              />
            </div>
            <p class="config-muted-hint">
              Named voices clients can select with <code>"voice": "preset-name"</code>. Paths are
              relative to the bundle directory on the server.
            </p>
            <div v-if="!voicePresetRows.length" class="config-muted-hint">
              No voice presets yet — add one to enable named voices in API requests.
            </div>
            <div v-for="row in voicePresetRows" :key="row.name" class="voice-preset-card">
              <div class="voice-preset-card__head">
                <InputText
                  :model-value="row.name"
                  class="voice-preset-card__name"
                  placeholder="preset-name"
                  @update:model-value="(value) => renameVoicePreset(row.name, value)"
                />
                <Button
                  icon="pi pi-trash"
                  severity="danger"
                  text
                  rounded
                  type="button"
                  aria-label="Remove preset"
                  @click="removeVoicePreset(row.name)"
                />
              </div>
              <div class="voice-preset-card__grid">
                <div
                  v-for="field in voicePresetFieldDefs"
                  :key="`${row.name}-${field.key}`"
                  class="param-field"
                >
                  <label class="param-field__label">{{ field.label }}</label>
                  <small v-if="field.type === 'path'" class="field-inline-hint">
                    Relative to bundle root on the server
                  </small>
                  <Textarea
                    v-if="field.type === 'textarea'"
                    :model-value="row.preset[field.key] || ''"
                    :placeholder="field.placeholder || ''"
                    rows="2"
                    class="w-full textarea-cli param-input"
                    @update:model-value="(value) => setVoicePresetField(row.name, field.key, value)"
                  />
                  <InputText
                    v-else
                    :model-value="row.preset[field.key] || ''"
                    :placeholder="field.placeholder || ''"
                    class="param-input"
                    @update:model-value="(value) => setVoicePresetField(row.name, field.key, value)"
                  />
                </div>
              </div>
            </div>
            <div class="param-field">
              <label class="param-field__label">Default voice preset</label>
              <p class="field-inline-hint">
                Used when the client omits <code>voice</code> in speech requests.
              </p>
              <Dropdown
                :model-value="defaultVoicePresetSelection"
                :options="defaultVoicePresetOptions"
                optionLabel="label"
                optionValue="value"
                placeholder="Inline default or choose a named preset"
                showClear
                class="param-input"
                @update:model-value="setDefaultVoicePresetSelection"
              />
            </div>
          </div>

          <div v-if="requestFieldGroups.length" class="audio-subsection">
            <div class="audio-subsection__title">Request default fields</div>
            <div
              v-for="group in requestFieldGroups"
              :key="group.id"
              class="audio-field-group"
            >
              <div class="audio-field-group__label">{{ group.label }}</div>
              <p v-if="group.description" class="config-muted-hint">{{ group.description }}</p>
              <div class="params-grid section-params">
                <div
                  v-for="field in group.fields"
                  :key="`${group.id}-${field.key}`"
                  class="param-field"
                >
                  <label class="param-field__label">
                    {{ field.label }}
                    <Tag value="Proxy" severity="info" class="audio-proxy-tag" />
                    <code v-if="field.nested || field.options_key" class="param-key-hint">
                      options.{{ field.options_key || field.key }}
                    </code>
                  </label>
                  <small v-if="fieldStorageHint(field)" class="field-inline-hint">
                    {{ fieldStorageHint(field) }}
                  </small>
                  <InputSwitch
                    v-if="field.type === 'bool'"
                    :model-value="Boolean(requestDefaultValue(field))"
                    @update:model-value="(value) => setRequestDefaultValue(field, value)"
                  />
                  <InputNumber
                    v-else-if="field.type === 'int' || field.type === 'float'"
                    :model-value="requestDefaultValue(field)"
                    :minFractionDigits="field.type === 'float' ? 1 : 0"
                    :maxFractionDigits="field.type === 'float' ? 6 : 0"
                    class="param-input"
                    @update:model-value="(value) => setRequestDefaultValue(field, value)"
                  />
                  <Textarea
                    v-else-if="field.type === 'textarea'"
                    :model-value="requestDefaultValue(field) || ''"
                    :placeholder="field.placeholder || ''"
                    rows="2"
                    class="w-full textarea-cli param-input"
                    @update:model-value="(value) => setRequestDefaultValue(field, value)"
                  />
                  <InputText
                    v-else
                    :model-value="requestDefaultValue(field) || ''"
                    :placeholder="field.placeholder || ''"
                    class="param-input"
                    @update:model-value="(value) => setRequestDefaultValue(field, value)"
                  />
                </div>
              </div>
            </div>
          </div>
        </template>

        <Message v-else severity="secondary" :closable="false" class="config-scan-message">
          No curated profile exists for this family yet. Use the Reference tab to see available
          API request fields, and configure server settings in the Server tab.
        </Message>
      </div>
    </div>

    <!-- Reference -->
    <div v-show="activeTab === 'reference'" class="audio-tab-panel">
      <div class="config-card">
        <div class="audio-subsection__head">
          <div class="section-label section-label--inline">API example</div>
          <Button
            label="Copy curl"
            icon="pi pi-copy"
            size="small"
            severity="secondary"
            outlined
            type="button"
            @click="copyApiExample"
          />
        </div>
        <p v-if="apiExampleHint" class="config-muted-hint">{{ apiExampleHint }}</p>
        <p class="config-muted-hint">
          Endpoint: <code>{{ apiEndpoint }}</code> · Model id:
          <code>{{ config.model_alias || llamaSwapStableId || 'your-model-id' }}</code>
        </p>
        <Textarea
          :model-value="requestApiExample"
          readonly
          rows="14"
          class="w-full textarea-cli cmd-preview-textarea"
          autoResize
        />
      </div>

      <div v-if="audioRequestCapabilities.length" class="config-card">
        <div class="section-label">
          Request-only parameters
          <small class="section-hint">
            Send these in the JSON body — they are not saved as server startup settings.
          </small>
        </div>
        <div class="config-toolbar__row">
          <span class="p-input-icon-left config-search-wrap">
            <i class="pi pi-search" aria-hidden="true" />
            <InputText
              v-model="referenceSearchQuery"
              type="search"
              placeholder="Filter request parameters…"
              class="config-search-input"
              aria-label="Filter request parameters"
            />
          </span>
        </div>
        <div class="request-cap-grid" role="list">
          <div
            v-for="param in filteredRequestCapabilities"
            :key="`request-${param.key}`"
            class="request-cap-item"
            role="listitem"
          >
            <code class="request-cap-item__key">{{ param.key }}</code>
            <span class="request-cap-item__label">{{ param.label }}</span>
            <i
              class="pi pi-info-circle param-info request-cap-item__info"
              v-tooltip.top="paramDescriptionTooltip(param)"
            />
          </div>
        </div>
        <Message
          v-if="!filteredRequestCapabilities.length"
          severity="secondary"
          :closable="false"
          class="config-scan-message"
        >
          No request parameters match your filter.
        </Message>
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed, ref, reactive, watch } from 'vue'
import { useToast } from 'primevue/usetoast'
import Button from 'primevue/button'
import Tag from 'primevue/tag'
import InputText from 'primevue/inputtext'
import InputNumber from 'primevue/inputnumber'
import InputSwitch from 'primevue/inputswitch'
import Dropdown from 'primevue/dropdown'
import Message from 'primevue/message'
import Textarea from 'primevue/textarea'
import AudioParamField from '@/components/audio/AudioParamField.vue'
import {
  useAudioModelConfig,
  paramDescriptionTooltip,
  fieldStorageHint,
  AUDIO_NESTED_SCOPE_KEYS,
} from '@/composables/useAudioModelConfig'
import { useEnginesStore } from '@/stores/engines'

const props = defineProps({
  config: {
    type: Object,
    required: true,
  },
  paramRegistry: {
    type: Object,
    required: true,
  },
  llamaSwapStableId: {
    type: String,
    default: '',
  },
})

const emit = defineEmits(['rescan-complete'])

const toast = useToast()
const enginesStore = useEnginesStore()

const activeTab = ref('overview')
const serverSearchQuery = ref('')
const referenceSearchQuery = ref('')
const hideUnsupportedParams = ref(false)
const rescanLoading = ref(false)

const tabs = computed(() => [
  { id: 'overview', label: 'Overview', icon: 'pi pi-compass', hint: 'Setup' },
  { id: 'server', label: 'Runtime', icon: 'pi pi-server', hint: 'Sidecar' },
  { id: 'api', label: 'Defaults', icon: 'pi pi-sliders-h', hint: taskKindMeta.value.tabHint },
  { id: 'reference', label: 'API', icon: 'pi pi-code', hint: 'curl' },
])

const configRef = computed(() => props.config)
const registryRef = computed(() => props.paramRegistry)
const stableIdRef = computed(() => props.llamaSwapStableId)

const audio = useAudioModelConfig(configRef, registryRef, enginesStore, stableIdRef)

const {
  audioConfigGroups,
  audioRequestCapabilities,
  taskProfile,
  isProfiledAudioModel,
  requestFieldGroups,
  requestDefaultsKey,
  apiEndpoint,
  apiExampleHint,
  requestDefaultsSectionTitle,
  audioTaskKind,
  taskKindMeta,
  swapSetParamsPreview,
  configuredDefaultsCount,
  defaultsApplyHint,
  setupProgress,
  supportsVoicePresets,
  taskWorkflowTags,
  voicePresetFieldDefs,
  voicePresetRows,
  defaultVoicePresetOptions,
  defaultVoicePresetSelection,
  audioInspectionSummary,
  setupChecklist,
  requestApiExample,
  audioParamValue,
  audioParamOptions,
  setAudioParamValue,
  updateAudioJsonParam,
  requestDefaultValue,
  setRequestDefaultValue,
  addVoicePreset,
  removeVoicePreset,
  renameVoicePreset,
  setVoicePresetField,
  setDefaultVoicePresetSelection,
  filterGroupParams,
} = audio

const expandedGroups = reactive({})

watch(
  audioConfigGroups,
  (groups) => {
    for (const group of groups) {
      if (expandedGroups[group.id] === undefined) {
        expandedGroups[group.id] = group.defaultExpanded !== false
      }
    }
  },
  { immediate: true },
)

const visibleServerGroups = computed(() =>
  audioConfigGroups.value
    .map((group) => ({
      ...group,
      params: filterGroupParams(group.params, serverSearchQuery.value, hideUnsupportedParams.value),
    }))
    .filter((group) => group.params.length || !serverSearchQuery.value.trim()),
)

const filteredRequestCapabilities = computed(() => {
  const q = referenceSearchQuery.value.trim().toLowerCase()
  if (!q) return audioRequestCapabilities.value
  return audioRequestCapabilities.value.filter((param) => {
    const hay = [param.key, param.label, param.description].join(' ').toLowerCase()
    return hay.includes(q)
  })
})

function paramStorageKey(param) {
  const nestedKey = AUDIO_NESTED_SCOPE_KEYS[param.scope]
  if (nestedKey) return `${nestedKey}.${param.key}`
  return param.key
}

function toggleGroup(groupId) {
  expandedGroups[groupId] = !expandedGroups[groupId]
}

async function rescanCliParams() {
  rescanLoading.value = true
  try {
    const data = await enginesStore.scanEngineParams('audio_cpp')
    if (data?.ok) {
      toast.add({
        severity: 'success',
        summary: 'CLI parameters scanned',
        detail: `Indexed ${data.param_count ?? 0} options for audio.cpp.`,
        life: 3500,
      })
      emit('rescan-complete')
    } else {
      toast.add({
        severity: 'warn',
        summary: 'Scan failed',
        detail: data?.scan_error || 'Unknown error',
        life: 6000,
      })
    }
  } catch (error) {
    toast.add({
      severity: 'error',
      summary: 'Scan failed',
      detail: error?.message || String(error),
      life: 5000,
    })
  } finally {
    rescanLoading.value = false
  }
}

async function copyApiExample() {
  try {
    await navigator.clipboard.writeText(requestApiExample.value)
    toast.add({
      severity: 'success',
      summary: 'Copied',
      detail: 'API example copied to clipboard.',
      life: 2500,
    })
  } catch {
    toast.add({
      severity: 'warn',
      summary: 'Copy failed',
      detail: 'Select the text manually.',
      life: 4000,
    })
  }
}
</script>

<style scoped>
.audio-model-config {
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
}

.audio-header-strip {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 0.75rem;
  flex-wrap: wrap;
}

.audio-header-strip__main {
  display: flex;
  align-items: flex-start;
  gap: 0.65rem;
}

.audio-header-strip__title {
  font-size: 0.95rem;
  font-weight: 600;
}

.audio-header-strip__meta {
  margin-top: 0.15rem;
  font-size: 0.78rem;
  color: var(--text-secondary, #9ca3af);
}

.audio-header-strip__sep {
  margin: 0 0.25rem;
}

.audio-header-strip__actions {
  display: flex;
  gap: 0.45rem;
  flex-wrap: wrap;
}

.audio-profile-hero__head {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 0.75rem;
}

.audio-checklist__actions {
  display: flex;
  gap: 0.45rem;
  flex-wrap: wrap;
  margin-top: 0.75rem;
}

.audio-guide-details {
  padding-top: 0.5rem;
}

.audio-guide-details__summary {
  cursor: pointer;
  font-size: 0.82rem;
  font-weight: 600;
  color: var(--text-secondary, #9ca3af);
}

.audio-defaults-hint {
  margin: 0.5rem 0 0.75rem;
}

.audio-setparams-preview {
  margin-bottom: 0.85rem;
  border: 1px solid var(--border-primary, #2a2f45);
  border-radius: var(--radius-md, 0.5rem);
  overflow: hidden;
}

.audio-setparams-preview__label {
  padding: 0.45rem 0.65rem;
  font-size: 0.74rem;
  color: var(--text-secondary, #9ca3af);
  border-bottom: 1px solid var(--border-primary, #2a2f45);
  background: rgba(255, 255, 255, 0.02);
}

.audio-setparams-preview__code {
  margin: 0;
  padding: 0.65rem;
  font-size: 0.72rem;
  line-height: 1.4;
  overflow: auto;
  max-height: 12rem;
}

.audio-proxy-tag {
  font-size: 0.62rem;
}

.audio-config-message__body {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}

.audio-config-message__action {
  align-self: flex-start;
}

.audio-guide-grid {
  display: grid;
  gap: 0.75rem;
  margin-top: 0.5rem;
}

@media (min-width: 900px) {
  .audio-guide-grid {
    grid-template-columns: repeat(3, minmax(0, 1fr));
  }
}

.audio-guide-item {
  border: 1px solid var(--border-primary, #2a2f45);
  border-radius: var(--radius-md, 0.5rem);
  padding: 0.75rem;
  background: var(--bg-surface, rgba(255, 255, 255, 0.02));
}

.audio-guide-item__badge {
  display: inline-block;
  font-size: 0.72rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  margin-bottom: 0.35rem;
  color: var(--text-secondary, #9ca3af);
}

.audio-guide-item p {
  margin: 0;
  font-size: 0.82rem;
  line-height: 1.45;
  color: var(--text-primary, #e5e7eb);
}

.audio-guide-item--server {
  border-left: 3px solid #22c55e;
}

.audio-guide-item--studio {
  border-left: 3px solid #3b82f6;
}

.audio-guide-item--request {
  border-left: 3px solid #f59e0b;
}

.audio-tabs {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 0.45rem;
}

@media (min-width: 768px) {
  .audio-tabs {
    grid-template-columns: repeat(4, minmax(0, 1fr));
  }
}

.audio-tab {
  display: flex;
  flex-direction: column;
  align-items: flex-start;
  gap: 0.1rem;
  padding: 0.65rem 0.75rem;
  border: 1px solid var(--border-primary, #2a2f45);
  border-radius: var(--radius-md, 0.5rem);
  background: var(--bg-surface, rgba(255, 255, 255, 0.02));
  color: var(--text-primary, #e5e7eb);
  cursor: pointer;
  text-align: left;
  transition: border-color 0.15s, background 0.15s;
}

.audio-tab:hover {
  border-color: var(--accent-primary, #6366f1);
}

.audio-tab--active {
  border-color: var(--accent-primary, #6366f1);
  background: rgba(99, 102, 241, 0.08);
}

.audio-tab__label {
  font-size: 0.85rem;
  font-weight: 600;
}

.audio-tab__hint {
  font-size: 0.72rem;
  color: var(--text-secondary, #9ca3af);
}

.audio-tab-panel {
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
}

.audio-checklist {
  list-style: none;
  margin: 0.5rem 0 0;
  padding: 0;
  display: flex;
  flex-direction: column;
  gap: 0.55rem;
}

.audio-checklist__item {
  display: flex;
  align-items: flex-start;
  gap: 0.55rem;
  font-size: 0.82rem;
}

.audio-checklist__item > div {
  flex: 1;
}

.audio-checklist__item .pi {
  margin-top: 0.15rem;
  color: var(--text-secondary, #9ca3af);
}

.audio-checklist__item--done .pi {
  color: #22c55e;
}

.audio-checklist__item small {
  display: block;
  color: var(--text-secondary, #9ca3af);
  margin-top: 0.1rem;
}

.audio-checklist__cta {
  margin-top: 0.75rem;
}

.audio-server-group {
  padding-top: 0.65rem;
}

.audio-group-toggle {
  display: flex;
  flex-direction: column;
  align-items: flex-start;
  gap: 0.15rem;
  width: 100%;
  padding: 0;
  border: 0;
  background: transparent;
  color: inherit;
  text-align: left;
  cursor: pointer;
}

.audio-group-toggle__main {
  display: inline-flex;
  align-items: center;
  gap: 0.45rem;
}

.audio-group-toggle__hint {
  margin: 0 0 0 1.35rem;
}

.audio-group-body {
  margin-top: 0.75rem;
  padding-top: 0.75rem;
  border-top: 1px solid var(--border-primary, #2a2f45);
}

.audio-subsection {
  margin-top: 0.9rem;
  padding-top: 0.75rem;
  border-top: 1px solid var(--border-primary, #2a2f45);
}

.audio-subsection__head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 0.75rem;
  margin-bottom: 0.35rem;
}

.audio-subsection__title {
  font-size: 0.82rem;
  font-weight: 600;
  color: var(--text-primary, #e5e7eb);
}

.audio-field-group + .audio-field-group {
  margin-top: 0.75rem;
}

.audio-field-group__label {
  font-size: 0.78rem;
  font-weight: 600;
  margin-bottom: 0.2rem;
}

.audio-info-banner {
  margin: 0.65rem 0;
}

.field-inline-hint {
  margin: 0 0 0.25rem;
  font-size: 0.74rem;
  color: var(--text-secondary, #9ca3af);
}

.audio-sidecar-tag {
  font-size: 0.65rem;
}

.audio-capability-tags {
  display: flex;
  flex-wrap: wrap;
  gap: 0.45rem;
}

.voice-preset-card {
  border: 1px solid var(--border-primary, #2a2f45);
  border-radius: var(--radius-md, 0.5rem);
  padding: 0.65rem 0.75rem;
  margin-bottom: 0.65rem;
  background: var(--bg-surface, rgba(255, 255, 255, 0.02));
}

.voice-preset-card__head {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  margin-bottom: 0.55rem;
}

.voice-preset-card__name {
  flex: 1;
}

.voice-preset-card__grid {
  display: grid;
  gap: 0.65rem;
}

@media (min-width: 900px) {
  .voice-preset-card__grid {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
}

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

.param-field__head .param-field__label {
  font-size: 0.8rem;
  font-weight: 500;
  color: var(--text-secondary, #9ca3af);
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 0.3rem;
}

.param-input {
  width: 100%;
}

.param-key-hint {
  font-size: 0.68rem;
}

.param-info {
  font-size: 0.7rem;
  cursor: help;
  opacity: 0.6;
}

.param-field--unsupported {
  opacity: 0.55;
}

.request-cap-grid {
  display: grid;
  grid-template-columns: minmax(7.5rem, auto) minmax(0, 1fr) auto;
  gap: 0.2rem 0.65rem;
  margin-top: 0.55rem;
  font-size: 0.78rem;
  line-height: 1.25;
}

.request-cap-item {
  display: contents;
}

.request-cap-item__key {
  color: var(--text-secondary, #9ca3af);
  font-size: 0.74rem;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.request-cap-item__label {
  color: var(--text-primary, #e5e7eb);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.request-cap-item__info {
  justify-self: end;
}

.section-label--inline {
  margin: 0;
}
</style>
