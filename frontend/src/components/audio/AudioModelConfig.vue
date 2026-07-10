<template>
  <div class="audio-model-config">
    <Message
      v-if="paramRegistry.scan_error"
      severity="warn"
      :closable="false"
      class="config-scan-message"
    >
      <div class="config-message__body">
        <strong>CLI parameters could not be loaded.</strong>
        {{ paramRegistry.scan_error }}
        <Button
          label="Rescan CLI parameters"
          icon="pi pi-refresh"
          size="small"
          severity="secondary"
          outlined
          :loading="rescanLoading"
          class="config-message__action"
          @click="rescanCliParams"
        />
      </div>
    </Message>
    <Message
      v-else-if="paramRegistry.scan_pending"
      severity="info"
      :closable="false"
      class="config-scan-message"
    >
      <div class="config-message__body">
        <strong>CLI parameters not indexed yet.</strong>
        Activate audio.cpp on the Engines page, then rescan.
        <Button
          label="Rescan now"
          icon="pi pi-refresh"
          size="small"
          severity="secondary"
          outlined
          :loading="rescanLoading"
          class="config-message__action"
          @click="rescanCliParams"
        />
      </div>
    </Message>

    <div class="config-card config-card--compact">
      <div class="section-label section-label--inline">
        {{ taskKindMeta.label }} configuration
        <Tag
          :value="taskKindMeta.short"
          :severity="taskKindMeta.tagSeverity"
        />
      </div>
      <p class="config-muted-hint">
        Endpoint <code>{{ apiEndpoint }}</code>
        <template v-if="config.family"> · {{ config.family }}</template>
        <template v-if="config.task"> · {{ config.task }}</template>
      </p>
    </div>

    <div class="engine-selector" role="tablist" aria-label="Audio configuration sections">
      <button
        v-for="tab in tabs"
        :key="tab.id"
        type="button"
        role="tab"
        class="engine-option"
        :class="{ selected: activeTab === tab.id }"
        :aria-selected="activeTab === tab.id"
        @click="activeTab = tab.id"
      >
        <span class="engine-option-label">
          <i :class="tab.icon" aria-hidden="true" />
          <span class="engine-name">{{ tab.label }}</span>
        </span>
      </button>
    </div>

    <!-- Overview -->
    <div v-show="activeTab === 'overview'" class="config-tab-panel">
      <div v-if="taskProfile" class="config-card">
        <div class="config-profile-hero__head">
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
        <ul class="config-checklist">
          <li
            v-for="item in setupChecklist"
            :key="item.id"
            class="config-checklist__item"
            :class="{ 'config-checklist__item--done': item.done }"
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
        <div class="config-checklist__actions">
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

      <details class="config-card">
        <summary class="config-details-summary">How configuration is applied</summary>
        <ul class="config-guide-list">
          <li>
            <strong>Sidecar</strong> — Runtime, load/session options, and voice presets start with
            the audio.cpp server.
          </li>
          <li>
            <strong>llama-swap setParams</strong> — Saved defaults for
            <code>{{ requestDefaultsKey }}</code> are injected into JSON
            <code>{{ apiEndpoint }}</code> requests when you apply llama-swap config.
          </li>
          <li>
            <strong>Per request</strong> — Input media and one-off overrides stay in each API call.
            See the API tab for examples.
          </li>
        </ul>
      </details>
    </div>

    <!-- Server -->
    <div v-show="activeTab === 'server'" class="config-tab-panel">
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
        class="config-card"
      >
        <button
          type="button"
          class="request-cap-toggle"
          :aria-expanded="expandedGroups[group.id]"
          @click="toggleGroup(group.id)"
        >
          <span class="request-cap-toggle__title">
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
          <small class="section-hint request-cap-toggle__hint">{{ group.description }}</small>
        </button>

        <div v-show="expandedGroups[group.id]" class="config-group-body">
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
                  <Tag value="Sidecar" severity="success" class="param-supported-tag" />
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
    <div v-show="activeTab === 'api'" class="config-tab-panel">
      <div class="config-card">
        <div class="section-label">
          {{ requestDefaultsSectionTitle }}
          <small class="section-hint">
            Defaults for <code>{{ apiEndpoint }}</code>
          </small>
        </div>

        <p class="config-muted-hint">{{ defaultsApplyHint }}</p>

        <div v-if="swapSetParamsPreview" class="setparams-preview">
          <div class="setparams-preview__label">
            llama-swap <code>filters.setParams</code> preview
          </div>
          <pre class="setparams-preview__code">{{ JSON.stringify(swapSetParamsPreview, null, 2) }}</pre>
        </div>

        <div class="tts-subsection">
          <div class="tts-subsection__head">
            <div>
              <span class="tts-subsection__title">Reference audio library</span>
              <Tag value="Bundle refs/" severity="secondary" />
            </div>
            <div class="reference-audio-actions">
              <input
                ref="referenceUploadInput"
                type="file"
                accept=".wav,audio/wav,audio/x-wav"
                class="reference-audio-upload-input"
                @change="onReferenceAudioSelected"
              />
              <Button
                label="Upload WAV"
                icon="pi pi-upload"
                size="small"
                severity="secondary"
                outlined
                type="button"
                :loading="referenceAudioUploading"
                @click="openReferenceAudioUpload"
              />
              <Button
                icon="pi pi-refresh"
                size="small"
                severity="secondary"
                text
                rounded
                type="button"
                aria-label="Refresh reference audio list"
                :loading="referenceAudioLoading"
                @click="loadReferenceAudio"
              />
            </div>
          </div>
          <p class="config-muted-hint">
            Upload reference WAV clips into <code>refs/</code> under the model bundle. Use them in
            voice presets or request defaults — paths are relative to the bundle root.
          </p>
          <div v-if="referenceAudioLoading && !referenceAudioItems.length" class="config-muted-hint">
            Loading reference audio…
          </div>
          <div v-else-if="!referenceAudioItems.length" class="config-muted-hint">
            No reference audio uploaded yet.
          </div>
          <div v-else class="reference-audio-list">
            <div
              v-for="item in referenceAudioItems"
              :key="item.path"
              class="reference-audio-row"
            >
              <div class="reference-audio-row__meta">
                <code class="reference-audio-row__path">{{ item.path }}</code>
                <span class="reference-audio-row__size">{{ formatBytes(item.size_bytes) }}</span>
                <Tag
                  v-for="usage in item.used_by || []"
                  :key="`${item.path}-${usage}`"
                  :value="usage"
                  severity="info"
                />
              </div>
              <div class="reference-audio-row__actions">
                <Button
                  v-if="supportsVoicePresets && voicePresetRows.length"
                  label="Use in preset"
                  icon="pi pi-link"
                  size="small"
                  text
                  type="button"
                  @click="openUseReferenceInPreset(item)"
                />
                <Button
                  icon="pi pi-trash"
                  severity="danger"
                  text
                  rounded
                  type="button"
                  aria-label="Delete reference audio"
                  :loading="referenceAudioDeleting === item.filename"
                  @click="deleteReferenceAudioItem(item)"
                />
              </div>
            </div>
          </div>
        </div>

        <template v-if="isProfiledAudioModel">
          <div v-if="supportsVoicePresets" class="tts-subsection">
            <div class="tts-subsection__head">
              <div>
                <span class="tts-subsection__title">Voice presets</span>
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
                  <div v-if="field.type === 'path'" class="reference-path-field">
                    <Dropdown
                      :model-value="row.preset[field.key] || ''"
                      :options="referenceAudioPathOptions"
                      optionLabel="label"
                      optionValue="value"
                      placeholder="Choose uploaded clip or type below"
                      showClear
                      editable
                      class="param-input reference-path-field__dropdown"
                      @update:model-value="(value) => setVoicePresetField(row.name, field.key, value)"
                    />
                  </div>
                  <Textarea
                    v-else-if="field.type === 'textarea'"
                    :model-value="row.preset[field.key] || ''"
                    :placeholder="field.placeholder || ''"
                    rows="2"
                    class="w-full textarea-cli param-input"
                    @update:model-value="(value) => setVoicePresetField(row.name, field.key, value)"
                  />
                  <InputText
                    v-else-if="field.type !== 'path'"
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

          <div v-if="requestFieldGroups.length" class="tts-subsection">
            <div class="tts-subsection__title">Request default fields</div>
            <div
              v-for="group in requestFieldGroups"
              :key="group.id"
              class="tts-speech-group"
            >
              <div class="tts-speech-group__label">{{ group.label }}</div>
              <p v-if="group.description" class="config-muted-hint">{{ group.description }}</p>
              <div class="params-grid section-params">
                <div
                  v-for="field in group.fields"
                  :key="`${group.id}-${field.key}`"
                  class="param-field"
                >
                  <label class="param-field__label">
                    {{ field.label }}
                    <Tag value="Proxy" severity="info" class="param-supported-tag" />
                    <code v-if="field.nested || field.options_key" class="param-key-hint">
                      options.{{ field.options_key || field.key }}
                    </code>
                  </label>
                  <small v-if="fieldStorageHint(field)" class="field-inline-hint">
                    {{ fieldStorageHint(field) }}
                  </small>
                  <small v-if="field.hint" class="field-inline-hint field-inline-hint--warning">
                    {{ field.hint }}
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
    <div v-show="activeTab === 'reference'" class="config-tab-panel">
      <div class="config-card">
        <div class="tts-subsection__head">
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
import { computed, ref, reactive, watch, onMounted } from 'vue'
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
import { useModelStore } from '@/stores/models'

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
  modelId: {
    type: String,
    default: '',
  },
})

const emit = defineEmits(['rescan-complete'])

const toast = useToast()
const enginesStore = useEnginesStore()
const modelStore = useModelStore()

const activeTab = ref('overview')
const serverSearchQuery = ref('')
const referenceSearchQuery = ref('')
const hideUnsupportedParams = ref(false)
const rescanLoading = ref(false)
const referenceAudioItems = ref([])
const referenceAudioLoading = ref(false)
const referenceAudioUploading = ref(false)
const referenceAudioDeleting = ref('')
const referenceUploadInput = ref(null)

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

const referenceAudioPathOptions = computed(() =>
  referenceAudioItems.value.map((item) => ({
    label: item.path,
    value: item.path,
  })),
)

watch(
  () => props.modelId,
  (modelId) => {
    if (modelId) {
      void loadReferenceAudio()
    } else {
      referenceAudioItems.value = []
    }
  },
  { immediate: true },
)

watch(activeTab, (tab) => {
  if (tab === 'api' && props.modelId) {
    void loadReferenceAudio()
  }
})

onMounted(() => {
  if (props.modelId) {
    void loadReferenceAudio()
  }
})

function formatBytes(bytes) {
  const value = Number(bytes) || 0
  if (value < 1024) return `${value} B`
  if (value < 1024 * 1024) return `${(value / 1024).toFixed(1)} KB`
  return `${(value / (1024 * 1024)).toFixed(1)} MB`
}

async function loadReferenceAudio() {
  if (!props.modelId) return
  referenceAudioLoading.value = true
  try {
    referenceAudioItems.value = await modelStore.listReferenceAudio(props.modelId)
  } catch (error) {
    toast.add({
      severity: 'error',
      summary: 'Failed to load reference audio',
      detail: error?.response?.data?.detail || error?.message || String(error),
      life: 5000,
    })
  } finally {
    referenceAudioLoading.value = false
  }
}

function openReferenceAudioUpload() {
  referenceUploadInput.value?.click()
}

async function onReferenceAudioSelected(event) {
  const file = event.target.files?.[0]
  event.target.value = ''
  if (!file || !props.modelId) return
  referenceAudioUploading.value = true
  try {
    const saved = await modelStore.uploadReferenceAudio(props.modelId, file)
    await loadReferenceAudio()
    toast.add({
      severity: 'success',
      summary: 'Reference audio uploaded',
      detail: saved?.path ? `Saved as ${saved.path}` : undefined,
      life: 3500,
    })
  } catch (error) {
    toast.add({
      severity: 'error',
      summary: 'Upload failed',
      detail: error?.response?.data?.detail || error?.message || String(error),
      life: 5000,
    })
  } finally {
    referenceAudioUploading.value = false
  }
}

async function deleteReferenceAudioItem(item) {
  if (!props.modelId || !item?.filename) return
  referenceAudioDeleting.value = item.filename
  try {
    await modelStore.deleteReferenceAudio(props.modelId, item.filename)
    await loadReferenceAudio()
    toast.add({
      severity: 'success',
      summary: 'Reference audio deleted',
      detail: item.path,
      life: 3000,
    })
  } catch (error) {
    toast.add({
      severity: 'error',
      summary: 'Delete failed',
      detail: error?.response?.data?.detail || error?.message || String(error),
      life: 6000,
    })
  } finally {
    referenceAudioDeleting.value = ''
  }
}

function openUseReferenceInPreset(item) {
  const firstPreset = voicePresetRows.value[0]?.name
  if (!firstPreset) return
  setVoicePresetField(firstPreset, 'voice_ref', item.path)
  toast.add({
    severity: 'info',
    summary: 'Preset updated',
    detail: `Set voice_ref on "${firstPreset}" to ${item.path}. Save configuration to apply.`,
    life: 4500,
  })
}

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

.reference-audio-actions {
  display: flex;
  align-items: center;
  gap: 0.35rem;
}

.reference-audio-upload-input {
  display: none;
}

.reference-audio-list {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}

.reference-audio-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 0.75rem;
  padding: 0.55rem 0.65rem;
  border: 1px solid var(--border-primary, #2a2f45);
  border-radius: var(--radius-md, 0.5rem);
  background: var(--bg-surface, rgba(255, 255, 255, 0.02));
}

.reference-audio-row__meta {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 0.45rem;
  min-width: 0;
}

.reference-audio-row__path {
  font-size: 0.78rem;
}

.reference-audio-row__size {
  font-size: 0.75rem;
  color: var(--text-secondary, #9ca3af);
}

.reference-audio-row__actions {
  display: flex;
  align-items: center;
  gap: 0.25rem;
  flex-shrink: 0;
}

.reference-path-field {
  display: flex;
  flex-direction: column;
  gap: 0.35rem;
}

.reference-path-field__dropdown {
  width: 100%;
}
</style>
