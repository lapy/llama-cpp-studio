<template>
  <div class="audio-model-config">
    <StateBanner v-if="paramRegistry.scan_error" severity="warning" class="config-scan-message">
      <strong>Engine options could not be loaded.</strong>
      {{ paramRegistry.scan_error }}
      <template #actions>
        <Button
          label="Refresh engine options"
          icon="pi pi-refresh"
          size="small"
          severity="secondary"
          outlined
          :loading="rescanLoading"
          @click="rescanCliParams"
        />
      </template>
    </StateBanner>
    <StateBanner v-else-if="paramRegistry.scan_pending" severity="info" class="config-scan-message">
      <strong>Engine options are not indexed yet.</strong>
      Activate audio.cpp on the Engines page, then refresh.
      <template #actions>
        <Button
          label="Refresh now"
          icon="pi pi-refresh"
          size="small"
          severity="secondary"
          outlined
          :loading="rescanLoading"
          @click="rescanCliParams"
        />
      </template>
    </StateBanner>

    <AudioContractNotice
      :scan-error="paramRegistry.scan_error"
      :scan-pending="paramRegistry.scan_pending"
    />

    <StateBanner
      v-for="(warning, index) in contractDriftWarnings"
      :key="`contract-drift-${index}`"
      severity="warning"
      icon=""
      class="config-scan-message"
    >
      <strong>Engine contract drift.</strong>
      {{ warning }}
    </StateBanner>

    <SectionTabs
      v-if="!hideTabs"
      v-model="internalTab"
      :items="tabs"
      id-prefix="audio-config"
      variant="pills"
      aria-label="Audio configuration sections"
    />

    <!-- Server -->
    <div
      v-show="currentTab === 'server'"
      id="audio-config-panel-server"
      role="tabpanel"
      aria-labelledby="audio-config-tab-server"
      tabindex="0"
      class="config-tab-panel"
    >
      <div class="config-card">
        <div class="runtime-common-head">
          <h3 class="page-subheading">Common settings</h3>
          <div class="toggle-field runtime-advanced-toggle">
            <ToggleSwitch v-model="showAdvancedRuntime" input-id="audio-runtime-advanced" />
            <label for="audio-runtime-advanced">Advanced</label>
          </div>
        </div>

        <div
          v-if="commonRuntimeParams.length"
          class="params-grid section-params runtime-common-grid"
        >
          <div
            v-for="param in commonRuntimeParams"
            :key="`common-${param.scope}-${param.key}`"
            class="param-field"
            :class="{ 'param-field--unsupported': param.supported === false }"
          >
            <div class="param-field__head">
              <label :for="`audio-common-${param.scope}-${param.key}`" class="param-field__label">
                {{ param.label }}
                <Tag
                  v-if="param.required && !param.dependency"
                  value="Required"
                  severity="danger"
                />
                <Tag
                  v-if="param.supported === false"
                  value="Not in this build"
                  severity="secondary"
                  class="param-supported-tag"
                />
                <Tag
                  v-for="tag in dependencyFieldTags(param)"
                  :key="`common-dep-${param.key}-${tag.key}`"
                  :value="tag.label"
                  :severity="tag.severity"
                />
                <i
                  v-if="paramHasExtraInfo(param)"
                  class="pi pi-info-circle param-info"
                  v-tooltip.top="paramDescriptionTooltip(param)"
                />
              </label>
            </div>
            <AudioParamField
              :id="`audio-common-${param.scope}-${param.key}`"
              :param="param"
              :model-value="audioParamValue(param)"
              :options="audioParamOptions(param)"
              :disabled="param.supported === false"
              @update:model-value="(value) => setAudioParamValue(param, value)"
              @update:json="(value) => updateAudioJsonParam(param, value)"
            />
            <small
              v-if="param.install_hint && !audioParamHasExplicitValue(param)"
              class="param-install-hint"
              >{{ param.install_hint }}</small
            >
          </div>
        </div>
        <StateBanner v-else severity="info" icon="" class="runtime-empty-message">
          No common runtime settings were found in the current audio.cpp parameter index.
        </StateBanner>

        <template v-if="showAdvancedRuntime">
          <div class="audio-advanced">
            <div class="config-toolbar__row">
              <span class="p-input-icon-left config-search-wrap">
                <i class="pi pi-search" aria-hidden="true" />
                <InputText
                  v-model="serverSearchQuery"
                  type="search"
                  placeholder="Search to add a parameter…"
                  class="config-search-input"
                  aria-label="Search parameters to add"
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
              <div class="toggle-field">
                <ToggleSwitch v-model="hideUnsupportedParams" input-id="audio-hide-unsupported" />
                <label for="audio-hide-unsupported">Hide unsupported</label>
              </div>
            </div>

            <div v-if="serverSearchQuery.trim()" class="param-tag-cloud-wrap">
              <h3 class="page-subheading">Add parameter</h3>
              <div v-if="advancedSearchTagResults.length" class="param-tag-cloud" role="list">
                <button
                  v-for="param in advancedSearchTagResults"
                  :key="`tag-${advancedParamId(param)}`"
                  type="button"
                  class="param-search-tag"
                  role="listitem"
                  @click="addAdvancedParam(param)"
                >
                  <span class="param-search-tag__label">{{ param.label }}</span>
                  <code class="param-search-tag__key">{{ param.key }}</code>
                </button>
              </div>
              <StateBanner v-else severity="info" icon="">No parameters match.</StateBanner>
            </div>

            <h3 class="page-subheading">Parameters</h3>
            <StateBanner v-if="!advancedPaneParams.length" severity="info" icon="">
              Search above to add advanced options. Saved values appear here automatically.
            </StateBanner>
            <div v-else class="params-grid section-params">
              <div
                v-for="param in advancedPaneParams"
                :key="`adv-${advancedParamId(param)}`"
                class="param-field"
                :class="{ 'param-field--unsupported': param.supported === false }"
              >
                <div class="param-field__head">
                  <label :for="`audio-${param.scope}-${param.key}`" class="param-field__label">
                    {{ param.label }}
                    <Tag
                      v-if="param.required && !param.dependency"
                      value="Required"
                      severity="danger"
                    />
                    <Tag
                      v-if="param.supported === false"
                      value="Not in this build"
                      severity="secondary"
                      class="param-supported-tag"
                    />
                    <Tag
                      v-for="tag in dependencyFieldTags(param)"
                      :key="`dep-${param.key}-${tag.key}`"
                      :value="tag.label"
                      :severity="tag.severity"
                    />
                    <i
                      v-if="paramHasExtraInfo(param)"
                      class="pi pi-info-circle param-info"
                      v-tooltip.top="paramDescriptionTooltip(param)"
                    />
                  </label>
                  <Button
                    v-if="canRemoveAdvancedParam(param)"
                    type="button"
                    icon="pi pi-times"
                    text
                    rounded
                    severity="secondary"
                    class="param-remove-btn"
                    aria-label="Remove parameter (reset to default)"
                    v-tooltip.top="'Remove from pane (reset to default)'"
                    @click="removeAdvancedParam(param)"
                  />
                </div>
                <AudioParamField
                  :id="`audio-${param.scope}-${param.key}`"
                  :param="param"
                  :model-value="audioParamValue(param)"
                  :options="audioParamOptions(param)"
                  :disabled="param.supported === false"
                  @update:model-value="(value) => setAudioParamValue(param, value)"
                  @update:json="(value) => updateAudioJsonParam(param, value)"
                />
                <small
                  v-if="param.install_hint && !audioParamHasExplicitValue(param)"
                  class="param-install-hint"
                  >{{ param.install_hint }}</small
                >
              </div>
            </div>
          </div>
        </template>
      </div>
    </div>

    <!-- Assets -->
    <div
      v-show="currentTab === 'assets'"
      id="audio-config-panel-assets"
      role="tabpanel"
      aria-labelledby="audio-config-tab-assets"
      tabindex="0"
      class="config-tab-panel"
    >
      <div class="config-card">
        <div class="tts-subsection__head">
          <h3 class="page-subheading">Community preset voices</h3>
          <Button
            :label="
              communityVoicesInstalled ? 'Re-download preset voices' : 'Download preset voices'
            "
            icon="pi pi-download"
            size="small"
            severity="secondary"
            outlined
            type="button"
            :loading="communityVoicesInstalling"
            @click="installCommunityVoices"
          />
          <Button
            v-if="communityVoicesAnyInstalled"
            label="Remove preset voices"
            icon="pi pi-trash"
            size="small"
            severity="danger"
            outlined
            type="button"
            :loading="communityVoicesRemoving"
            @click="confirmRemoveCommunityVoices"
          />
        </div>
        <p class="config-muted-hint">
          Demo clips from audio.cpp. After download, use a voice name such as demo_1_man, then apply
          and restart so the server picks them up.
        </p>
        <div v-if="communityVoicesLoading && !communityVoiceItems.length" class="config-muted-hint">
          Loading community preset voices…
        </div>
        <div v-else-if="!communityVoiceItems.length" class="config-muted-hint">
          Community preset voices are unavailable.
        </div>
        <div v-else class="reference-audio-list">
          <div v-for="item in communityVoiceItems" :key="item.id" class="reference-audio-row">
            <div class="reference-audio-row__meta">
              <code class="reference-audio-row__path">{{ item.label || item.id }}</code>
              <span class="reference-audio-row__size">{{ item.reference_text }}</span>
              <Tag
                :value="item.installed ? 'Installed' : 'Not downloaded'"
                :severity="item.installed ? 'success' : 'secondary'"
              />
            </div>
          </div>
        </div>
      </div>

      <div class="config-card">
        <div class="tts-subsection__head">
          <h3 class="page-subheading">Reference audio</h3>
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
              v-tooltip.top="'Max 60 MB per file'"
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

        <div v-if="referenceAudioLoading && !referenceAudioItems.length" class="config-muted-hint">
          Loading reference audio…
        </div>
        <div v-else-if="!uploadedReferenceAudio.length" class="config-muted-hint">
          No reference audio uploaded yet.
        </div>
        <div v-else class="reference-audio-list">
          <div v-for="item in uploadedReferenceAudio" :key="item.path" class="reference-audio-row">
            <div class="reference-audio-row__meta">
              <code class="reference-audio-row__path">
                {{ item.display_path || item.relative_path || item.path }}
              </code>
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

      <div v-if="supportsVoicePresets" class="config-card">
        <div class="tts-subsection__head">
          <h3 class="page-subheading">Voice presets</h3>
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

        <div v-if="!voicePresetRows.length" class="config-muted-hint">
          {{ emptyVoicePresetsHint }}
        </div>
        <div v-for="row in voicePresetRows" :key="row.id" class="voice-preset-card">
          <div class="voice-preset-card__head">
            <InputText
              :model-value="voicePresetNameDraft(row.name)"
              class="voice-preset-card__name"
              placeholder="preset-name"
              @update:model-value="(value) => setVoicePresetNameDraft(row.name, value)"
              @blur="() => commitVoicePresetRename(row.name)"
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
              :key="`${row.id}-${field.key}`"
              class="param-field"
            >
              <label class="param-field__label">
                {{ field.label }}
              </label>
              <div v-if="field.type === 'path'" class="reference-path-field">
                <Select
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
              <Select
                v-else-if="fieldSelectOptions(field).length"
                :model-value="row.preset[field.key] || ''"
                :options="fieldSelectOptions(field)"
                optionLabel="label"
                optionValue="value"
                :placeholder="field.placeholder || 'Choose a packaged voice'"
                showClear
                editable
                class="param-input"
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
          <label class="param-field__label">
            {{ requiresSessionVoice ? 'Default voice preset (required)' : 'Default voice preset' }}
          </label>
          <Select
            :model-value="defaultVoicePresetSelection"
            :options="defaultVoicePresetOptions"
            optionLabel="label"
            optionValue="value"
            :placeholder="
              requiresSessionVoice ? 'Required for session prepare' : 'Use Defaults tab values'
            "
            showClear
            class="param-input"
            @update:model-value="setDefaultVoicePresetSelection"
          />
        </div>
      </div>
    </div>

    <!-- Defaults (+ API reference) -->
    <div
      v-show="currentTab === 'api'"
      id="audio-config-panel-api"
      role="tabpanel"
      aria-labelledby="audio-config-tab-api"
      tabindex="0"
      class="config-tab-panel"
    >
      <ModelConfigAudioProfiles
        :title="requestDefaultsSectionTitle"
        :endpoint="apiEndpoint"
        :guidance="instructionsPolicyGuidance"
        :profiled="isProfiledAudioModel"
        :groups="requestFieldGroups"
        :resolve-value="requestDefaultValue"
        :resolve-options="fieldSelectOptions"
        :defaults-preview="swapSetParamsPreview"
        :api-example="requestApiExample"
        :model-id="config.model_alias || stableModelId || 'your-model-id'"
        :capabilities="audioRequestCapabilities"
        @update-request-default="applyProfileDefault"
      />
    </div>
  </div>
</template>

<script setup>
import { computed, onMounted, ref, watch, onUnmounted } from 'vue'
import { useToast } from 'primevue/usetoast'
import { useConfirm } from 'primevue/useconfirm'
import { requireSingleConfirmation } from '@/composables/singleConfirm'
import Button from 'primevue/button'
import Tag from 'primevue/tag'
import InputText from 'primevue/inputtext'
import ToggleSwitch from 'primevue/toggleswitch'
import Select from 'primevue/select'
import Textarea from 'primevue/textarea'
import SectionTabs from '@/components/common/SectionTabs.vue'
import StateBanner from '@/components/common/StateBanner.vue'
import AudioParamField from '@/components/audio/AudioParamField.vue'
import AudioContractNotice from '@/components/audio/AudioContractNotice.vue'
import ModelConfigAudioProfiles from '@/components/models/ModelConfigAudioProfiles.vue'
import {
  useAudioModelConfig,
  paramDescriptionTooltip,
  dependencyFieldTags,
  paramMatchesSearch,
} from '@/composables/useAudioModelConfig'
import { fetchAudioContract } from '@/composables/useAudioContract'
import { useEnginesStore } from '@/stores/engines'
import { isAudioEngineId } from '@/engines/registry'
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
  stableModelId: {
    type: String,
    default: '',
  },
  modelId: {
    type: String,
    default: '',
  },
  hideTabs: {
    type: Boolean,
    default: false,
  },
  activeTab: {
    type: String,
    default: '',
  },
})

const emit = defineEmits(['rescan-complete'])

const toast = useToast()
const confirm = useConfirm()
const enginesStore = useEnginesStore()
const modelStore = useModelStore()

const internalTab = ref('server')
const currentTab = computed(() =>
  props.hideTabs && props.activeTab ? props.activeTab : internalTab.value,
)
const serverSearchQuery = ref('')
const hideUnsupportedParams = ref(true)
const showAdvancedRuntime = ref(false)
const rescanLoading = ref(false)
const referenceAudioItems = ref([])
const referenceAudioLoading = ref(false)
const referenceAudioUploading = ref(false)
const referenceAudioDeleting = ref('')
const communityVoiceItems = ref([])
const communityVoicesLoading = ref(false)
const communityVoicesInstalling = ref(false)
const communityVoicesRemoving = ref(false)
const referenceUploadInput = ref(null)
const REFERENCE_AUDIO_MAX_BYTES = 60 * 1024 * 1024

const COMMON_RUNTIME_PARAM_ORDER = [
  'family',
  'task',
  'mode',
  'config',
  'weight',
  'backend',
  'device',
  'threads',
  'lazy_load',
]
const COMMON_RUNTIME_PARAM_KEYS = new Set(COMMON_RUNTIME_PARAM_ORDER)
const COMMON_RUNTIME_ORDER = new Map(COMMON_RUNTIME_PARAM_ORDER.map((key, index) => [key, index]))
const COMMON_RUNTIME_SCOPE_PRIORITY = new Map([
  ['model', 0],
  ['process', 1],
  ['load_option', 2],
  ['session_option', 3],
])
const configRef = computed(() => props.config)
const registryRef = computed(() => props.paramRegistry)
const stableIdRef = computed(() => props.stableModelId)
const audioEngineId = computed(() => {
  const configured = String(configRef.value?.engine || '')
  if (isAudioEngineId(configured)) return configured
  return (
    (enginesStore.engineDescriptors || []).find((item) => isAudioEngineId(item.id))?.id ||
    configured
  )
})

const audio = useAudioModelConfig(configRef, registryRef, enginesStore, stableIdRef)

// Drift = engine observations disagree with the Studio operation table.
// Surfaced verbatim; never silently patched over.
const contractDriftWarnings = computed(() =>
  (props.paramRegistry?.contract_drift_warnings || []).map(String),
)

const {
  audioConfigGroups,
  audioRequestCapabilities,
  isProfiledAudioModel,
  requestFieldGroups,
  apiEndpoint,
  requestDefaultsSectionTitle,
  swapSetParamsPreview,
  instructionsPolicyGuidance,
  supportsVoicePresets,
  requiresSessionVoice,
  emptyVoicePresetsHint,
  voicePresetFieldDefs,
  fieldSelectOptions,
  voicePresetRows,
  voicePresetNameDraft,
  setVoicePresetNameDraft,
  commitVoicePresetRename,
  defaultVoicePresetOptions,
  defaultVoicePresetSelection,
  requestApiExample,
  audioParamValue,
  audioParamHasExplicitValue,
  audioParamOptions,
  setAudioParamValue,
  updateAudioJsonParam,
  requestDefaultValue,
  setRequestDefaultValue,
  addVoicePreset,
  removeVoicePreset,
  setVoicePresetField,
  setDefaultVoicePresetSelection,
} = audio

function applyProfileDefault({ field, value }) {
  setRequestDefaultValue(field, value)
}

const tabs = computed(() => [
  { id: 'server', label: 'Runtime', icon: 'pi pi-server' },
  { id: 'assets', label: 'Assets', icon: 'pi pi-folder-open' },
  { id: 'api', label: 'Defaults', icon: 'pi pi-sliders-h' },
])

function paramHasExtraInfo(param) {
  return Boolean(
    param?.description ||
    param?.install_hint ||
    param?.dependency ||
    param?.primary_flag ||
    param?.negative_flag ||
    (param?.flags && param.flags.length) ||
    param?.scope === 'load_option' ||
    param?.scope === 'session_option',
  )
}

const uploadedReferenceAudio = computed(() =>
  referenceAudioItems.value.filter((item) => item.storage !== 'community'),
)

const communityVoicesInstalled = computed(
  () =>
    communityVoiceItems.value.length > 0 &&
    communityVoiceItems.value.every((item) => item.installed),
)

const communityVoicesAnyInstalled = computed(() =>
  communityVoiceItems.value.some((item) => item.installed),
)

const referenceAudioPathOptions = computed(() =>
  referenceAudioItems.value.map((item) => ({
    label: item.display_path || item.relative_path || item.path,
    value: item.path,
  })),
)

const commonRuntimeParams = computed(() => {
  const chosen = new Map()
  for (const group of audioConfigGroups.value) {
    for (const param of group.params || []) {
      const scope = param.scope || 'process'
      const commonKey = COMMON_RUNTIME_PARAM_KEYS.has(param.key)
      const modelAsset = scope === 'model' && param.asset_selector
      const required = param.required === true
      if ((!commonKey && !modelAsset && !required) || param.supported === false) continue

      const existing = chosen.get(param.key)
      const existingScope = existing?.scope || 'process'
      const priority = COMMON_RUNTIME_SCOPE_PRIORITY.get(scope) ?? 99
      const existingPriority = COMMON_RUNTIME_SCOPE_PRIORITY.get(existingScope) ?? 99
      if (!existing || priority < existingPriority) {
        chosen.set(param.key, param)
      }
    }
  }
  return [...chosen.values()].sort((a, b) => {
    const aOrder = COMMON_RUNTIME_ORDER.get(a.key)
    const bOrder = COMMON_RUNTIME_ORDER.get(b.key)
    const aRank = aOrder ?? (a.required ? 50 : 99)
    const bRank = bOrder ?? (b.required ? 50 : 99)
    if (aRank !== bRank) return aRank - bRank
    return String(a.label || a.key).localeCompare(String(b.label || b.key))
  })
})

let referenceAudioEpoch = 0
let referenceAudioDisposed = false

onMounted(() => {
  // Shared module-level singleton; concurrent callers share one request.
  void fetchAudioContract()
})

onUnmounted(() => {
  referenceAudioDisposed = true
  referenceAudioEpoch += 1
})

watch(
  () => props.modelId,
  (modelId) => {
    if (modelId) {
      void loadReferenceAudio()
      void loadCommunityVoices()
    } else {
      referenceAudioEpoch += 1
      referenceAudioItems.value = []
      referenceAudioLoading.value = false
      communityVoiceItems.value = []
      communityVoicesLoading.value = false
    }
  },
  { immediate: true },
)

watch(currentTab, (tab) => {
  if (tab === 'assets' && props.modelId) {
    void loadReferenceAudio()
    void loadCommunityVoices()
  }
})

function formatBytes(bytes) {
  const value = Number(bytes) || 0
  if (value < 1024) return `${value} B`
  if (value < 1024 * 1024) return `${(value / 1024).toFixed(1)} KB`
  return `${(value / (1024 * 1024)).toFixed(1)} MB`
}

async function loadReferenceAudio() {
  if (!props.modelId || referenceAudioDisposed) return
  const epoch = ++referenceAudioEpoch
  referenceAudioLoading.value = true
  try {
    const items = await modelStore.listReferenceAudio(props.modelId)
    if (epoch !== referenceAudioEpoch || referenceAudioDisposed) return
    referenceAudioItems.value = Array.isArray(items) ? items : []
  } catch (error) {
    if (epoch !== referenceAudioEpoch || referenceAudioDisposed) return
    toast.add({
      severity: 'error',
      summary: 'Failed to load reference audio',
      detail: error?.response?.data?.detail || error?.message || String(error),
      life: 5000,
    })
  } finally {
    if (epoch === referenceAudioEpoch && !referenceAudioDisposed) {
      referenceAudioLoading.value = false
    }
  }
}

async function loadCommunityVoices() {
  if (typeof modelStore.listCommunityVoices !== 'function') return
  communityVoicesLoading.value = true
  try {
    const payload = await modelStore.listCommunityVoices()
    if (referenceAudioDisposed) return
    communityVoiceItems.value = Array.isArray(payload?.items) ? payload.items : []
  } catch (error) {
    if (referenceAudioDisposed) return
    toast.add({
      severity: 'error',
      summary: 'Failed to load community preset voices',
      detail: error?.response?.data?.detail || error?.message || String(error),
      life: 5000,
    })
  } finally {
    if (!referenceAudioDisposed) communityVoicesLoading.value = false
  }
}

async function installCommunityVoices() {
  if (typeof modelStore.installCommunityVoices !== 'function') return
  communityVoicesInstalling.value = true
  try {
    const payload = await modelStore.installCommunityVoices()
    if (referenceAudioDisposed) return
    communityVoiceItems.value = Array.isArray(payload?.items) ? payload.items : []
    await loadReferenceAudio()
    if (referenceAudioDisposed) return
    toast.add({
      severity: 'success',
      summary: 'Community preset voices installed',
      detail:
        'Apply the proxy config and restart the model so audio.cpp can use these voice names.',
      life: 5000,
    })
  } catch (error) {
    if (referenceAudioDisposed) return
    toast.add({
      severity: 'error',
      summary: 'Preset voice download failed',
      detail: error?.response?.data?.detail || error?.message || String(error),
      life: 5000,
    })
  } finally {
    if (!referenceAudioDisposed) communityVoicesInstalling.value = false
  }
}

async function removeCommunityVoices() {
  if (typeof modelStore.removeCommunityVoices !== 'function') return
  communityVoicesRemoving.value = true
  try {
    const payload = await modelStore.removeCommunityVoices()
    if (referenceAudioDisposed) return
    communityVoiceItems.value = Array.isArray(payload?.items) ? payload.items : []
    await loadReferenceAudio()
    if (referenceAudioDisposed) return
    toast.add({
      severity: 'success',
      summary: 'Community preset voices removed',
      detail: 'Voice presets pointing at community clips stop resolving after apply and restart.',
      life: 6000,
    })
  } catch (error) {
    if (referenceAudioDisposed) return
    toast.add({
      severity: 'error',
      summary: 'Preset voice removal failed',
      detail: error?.response?.data?.detail || error?.message || String(error),
      life: 6000,
    })
  } finally {
    if (!referenceAudioDisposed) communityVoicesRemoving.value = false
  }
}

function confirmRemoveCommunityVoices() {
  requireSingleConfirmation(confirm, {
    header: 'Remove community preset voices',
    message:
      'Delete the installed community demo clips from the Studio data root? ' +
      'Voice presets that reference them will need new voices. Uploaded reference audio is not affected.',
    accept: removeCommunityVoices,
    acceptLabel: 'Remove',
    rejectLabel: 'Cancel',
    acceptClass: 'p-button-danger',
  })
}

function openReferenceAudioUpload() {
  referenceUploadInput.value?.click()
}

async function onReferenceAudioSelected(event) {
  const file = event.target.files?.[0]
  event.target.value = ''
  if (!file || !props.modelId) return
  if (file.size > REFERENCE_AUDIO_MAX_BYTES) {
    toast.add({
      severity: 'warn',
      summary: 'Upload too large',
      detail: `Reference WAVs must be ${formatBytes(REFERENCE_AUDIO_MAX_BYTES)} or smaller.`,
      life: 5000,
    })
    return
  }
  referenceAudioUploading.value = true
  try {
    const saved = await modelStore.uploadReferenceAudio(props.modelId, file)
    if (referenceAudioDisposed) return
    await loadReferenceAudio()
    if (referenceAudioDisposed) return
    toast.add({
      severity: 'success',
      summary: 'Reference audio uploaded',
      detail: saved?.path
        ? `Saved as ${saved.display_path || saved.relative_path || saved.path}`
        : undefined,
      life: 3500,
    })
  } catch (error) {
    if (referenceAudioDisposed) return
    toast.add({
      severity: 'error',
      summary: 'Upload failed',
      detail: error?.response?.data?.detail || error?.message || String(error),
      life: 5000,
    })
  } finally {
    if (!referenceAudioDisposed) referenceAudioUploading.value = false
  }
}

async function deleteReferenceAudioItem(item) {
  if (!props.modelId || !item?.filename) return
  referenceAudioDeleting.value = item.filename
  try {
    await modelStore.deleteReferenceAudio(props.modelId, item.filename)
    if (referenceAudioDisposed) return
    await loadReferenceAudio()
    if (referenceAudioDisposed) return
    toast.add({
      severity: 'success',
      summary: 'Reference audio deleted',
      detail: item.display_path || item.relative_path || item.path,
      life: 3000,
    })
  } catch (error) {
    if (referenceAudioDisposed) return
    toast.add({
      severity: 'error',
      summary: 'Delete failed',
      detail: error?.response?.data?.detail || error?.message || String(error),
      life: 6000,
    })
  } finally {
    if (!referenceAudioDisposed) referenceAudioDeleting.value = ''
  }
}

function openUseReferenceInPreset(item) {
  const firstPreset = voicePresetRows.value[0]?.name
  if (!firstPreset) return
  setVoicePresetField(firstPreset, 'voice_ref', item.path)
  if (item.reference_text) {
    setVoicePresetField(firstPreset, 'reference_text', item.reference_text)
  }
  toast.add({
    severity: 'info',
    summary: 'Preset updated',
    detail: `Set voice_ref on "${firstPreset}" to ${item.display_path || item.path}. Save configuration to apply.`,
    life: 4500,
  })
}

const advancedParamKeys = ref([])

function advancedParamId(param) {
  return `${param.scope || 'process'}:${param.key}`
}

const commonParamKeys = computed(() => new Set(commonRuntimeParams.value.map((param) => param.key)))

const advancedCandidateParams = computed(() => {
  const seen = new Set()
  const out = []
  for (const group of audioConfigGroups.value) {
    for (const param of group.params || []) {
      if (commonParamKeys.value.has(param.key)) continue
      const id = advancedParamId(param)
      if (seen.has(id)) continue
      seen.add(id)
      out.push(param)
    }
  }
  return out
})

function advancedParamHasSavedValue(param) {
  if (!audioParamHasExplicitValue(param)) return false
  const value = audioParamValue(param)
  if (value == null || value === '') return false
  if (typeof value === 'number' && Number.isNaN(value)) return false
  if (Array.isArray(value) && value.length === 0) return false
  return true
}

function shouldPinAdvancedParam(param) {
  if (param.dependency || param.required) return true
  return advancedParamHasSavedValue(param)
}

function syncAdvancedParamKeys() {
  const candidates = advancedCandidateParams.value
  const candidateIds = new Set(candidates.map(advancedParamId))
  const next = new Set(advancedParamKeys.value.filter((id) => candidateIds.has(id)))
  for (const param of candidates) {
    if (shouldPinAdvancedParam(param)) next.add(advancedParamId(param))
  }
  advancedParamKeys.value = [...next]
}

watch(
  [advancedCandidateParams, configRef],
  () => {
    syncAdvancedParamKeys()
  },
  { immediate: true, deep: true },
)

const advancedSearchTagResults = computed(() => {
  const q = serverSearchQuery.value
  if (!q.trim()) return []
  const active = new Set(advancedParamKeys.value)
  const out = []
  for (const param of advancedCandidateParams.value) {
    if (active.has(advancedParamId(param))) continue
    if (paramMatchesSearch(param, q, hideUnsupportedParams.value)) out.push(param)
    if (out.length >= 100) break
  }
  return out
})

const advancedPaneParams = computed(() => {
  const byId = new Map(
    advancedCandidateParams.value.map((param) => [advancedParamId(param), param]),
  )
  return advancedParamKeys.value.map((id) => byId.get(id)).filter(Boolean)
})

function addAdvancedParam(param) {
  const id = advancedParamId(param)
  if (advancedParamKeys.value.includes(id)) return
  advancedParamKeys.value = [...advancedParamKeys.value, id]
  serverSearchQuery.value = ''
}

function canRemoveAdvancedParam(param) {
  return !(param.dependency || param.required)
}

function removeAdvancedParam(param) {
  const id = advancedParamId(param)
  advancedParamKeys.value = advancedParamKeys.value.filter((key) => key !== id)
  setAudioParamValue(param, null)
}

async function rescanCliParams() {
  rescanLoading.value = true
  try {
    const data = await enginesStore.scanEngineParams(audioEngineId.value, null, {
      modelId: props.modelId || undefined,
    })
    const profileFailed = Boolean(data?.profile_scan_error)
    if (data?.ok && !profileFailed) {
      toast.add({
        severity: 'success',
        summary: 'Parameters scanned',
        detail: props.modelId
          ? `Indexed engine options and refreshed this model's session/request profile. Open the progress card for the help scan log.`
          : `Indexed ${data.param_count ?? 0} options for audio.cpp. Open the progress card for the help scan log.`,
        life: 3500,
      })
      emit('rescan-complete')
    } else {
      toast.add({
        severity: 'warn',
        summary: 'Scan failed',
        detail: data?.profile_scan_error || data?.scan_error || 'Unknown error',
        life: 6000,
      })
      if (data?.ok) emit('rescan-complete')
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
</script>

<style scoped>
.audio-model-config {
  display: flex;
  flex-direction: column;
  gap: 0.55rem;
}

.audio-advanced {
  margin-top: 0.75rem;
  padding-top: 0.75rem;
  border-top: 1px solid var(--border-primary, #2a2f45);
  display: flex;
  flex-direction: column;
  gap: 0.55rem;
}

.param-install-hint {
  display: block;
  margin-top: 0.35rem;
  color: var(--text-color-secondary);
  line-height: 1.35;
}

.runtime-common-head {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 0.75rem;
}

.runtime-common-grid {
  margin-top: 0.85rem;
}

.runtime-advanced-toggle {
  flex-shrink: 0;
}

.runtime-empty-message {
  margin-top: 0.85rem;
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
  font-size: var(--text-xs);
}

.reference-audio-row__size {
  font-size: var(--text-xs);
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
