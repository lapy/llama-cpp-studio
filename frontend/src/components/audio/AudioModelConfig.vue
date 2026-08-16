<template>
  <div class="audio-model-config">
    <Message
      v-if="paramRegistry.scan_error"
      severity="warn"
      :closable="false"
      class="config-scan-message"
    >
      <div class="config-message__body">
        <strong>Engine options could not be loaded.</strong>
        {{ paramRegistry.scan_error }}
        <Button
          label="Refresh engine options"
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
        <strong>Engine options are not indexed yet.</strong>
        Activate audio.cpp on the Engines page, then refresh.
        <Button
          label="Refresh now"
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
      v-if="contractGradeBadge"
      :severity="contractGradeBadge.severity"
      :closable="false"
      class="config-scan-message"
    >
      <div class="config-message__body">
        <strong>{{ contractGradeBadge.title }}</strong>
        {{ contractGradeBadge.detail }}
      </div>
    </Message>

    <Message
      v-if="contractReviewRequired"
      severity="warn"
      :closable="false"
      class="config-scan-message"
    >
      <div class="config-message__body">
        <strong>This audio.cpp build changed — confirm this model’s settings</strong>
        <p class="config-muted-hint">
          Refresh engine options if needed, then clear outdated defaults once you’ve checked Runtime and Defaults.
        </p>
        <div class="config-message__actions">
          <Button
            label="Refresh engine options"
            icon="pi pi-refresh"
            size="small"
            severity="secondary"
            outlined
            :loading="rescanLoading"
            @click="rescanCliParams"
          />
          <Button
            label="Mark reviewed"
            icon="pi pi-check"
            size="small"
            severity="warning"
            :disabled="!contractFingerprint"
            @click="markContractReviewed"
          />
        </div>
      </div>
    </Message>

    <div class="config-section-tabs" role="tablist" aria-label="Audio configuration sections">
      <button
        v-for="tab in tabs"
        :key="tab.id"
        type="button"
        role="tab"
        class="config-section-tab"
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
      <div class="config-card">
        <div class="config-profile-hero__head">
          <div class="section-label section-label--inline">
            Setup
            <Tag
              v-if="taskProfile?.label"
              :value="taskProfile.label"
              severity="secondary"
            />
            <Tag
              :value="setupProgress === 100 ? 'Ready' : `${setupIncompleteCount} left`"
              :severity="setupProgress === 100 ? 'success' : 'info'"
            />
          </div>
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
              <small v-if="!item.done">{{ item.detail }}</small>
            </div>
            <Button
              v-if="item.tab && !item.done"
              :label="item.tab === 'api' ? 'Edit defaults' : 'Open'"
              size="small"
              text
              type="button"
              @click="activeTab = item.tab"
            />
          </li>
        </ul>
      </div>
    </div>

    <!-- Server -->
    <div v-show="activeTab === 'server'" class="config-tab-panel">
      <div class="config-card">
        <div class="runtime-common-head">
          <div class="section-label section-label--inline">
            Common settings
          </div>
          <div class="toggle-field runtime-advanced-toggle">
            <InputSwitch v-model="showAdvancedRuntime" input-id="audio-runtime-advanced" />
            <label for="audio-runtime-advanced">Advanced</label>
          </div>
        </div>

        <div v-if="commonRuntimeParams.length" class="params-grid section-params runtime-common-grid">
          <div
            v-for="param in commonRuntimeParams"
            :key="`common-${param.scope}-${param.key}`"
            class="param-field"
            :class="{ 'param-field--unsupported': param.supported === false }"
          >
            <div class="param-field__head">
              <label :for="`audio-common-${param.scope}-${param.key}`" class="param-field__label">
                {{ param.label }}
                <Tag v-if="param.required && !param.dependency" value="Required" severity="danger" />
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
            >{{ param.install_hint }}</small>
          </div>
        </div>
        <Message v-else severity="secondary" :closable="false" class="config-scan-message runtime-empty-message">
          No common runtime settings were found in the current audio.cpp parameter index.
        </Message>
      </div>

      <template v-if="showAdvancedRuntime">
        <div class="config-card config-toolbar">
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
          </div>
          <div class="config-toolbar__row config-toolbar__toggles">
            <div class="toggle-field">
              <InputSwitch v-model="hideUnsupportedParams" input-id="audio-hide-unsupported" />
              <label for="audio-hide-unsupported">Hide unsupported in this build</label>
            </div>
          </div>
        </div>

        <div v-if="serverSearchQuery.trim()" class="config-card config-search-tags-card">
          <div class="section-label">Add parameter</div>
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
          <Message v-else severity="secondary" :closable="false" class="config-scan-message">
            No parameters match.
          </Message>
        </div>

        <div class="config-card config-params-pane">
          <div class="section-label">
            Parameters
          </div>
          <Message
            v-if="!advancedPaneParams.length"
            severity="secondary"
            :closable="false"
            class="config-scan-message"
          >
            Search above to add advanced options. Saved values appear here automatically.
          </Message>
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
                  <Tag v-if="param.required && !param.dependency" value="Required" severity="danger" />
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
              >{{ param.install_hint }}</small>
            </div>
          </div>
        </div>
      </template>
    </div>

    <!-- Assets -->
    <div v-show="activeTab === 'assets'" class="config-tab-panel">
      <div class="config-card">
        <div class="tts-subsection__head">
          <div class="section-label section-label--inline">
            Reference audio
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
          <div class="section-label section-label--inline">
            Voice presets
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
              <Dropdown
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
          <Dropdown
            :model-value="defaultVoicePresetSelection"
            :options="defaultVoicePresetOptions"
            optionLabel="label"
            optionValue="value"
            :placeholder="requiresSessionVoice ? 'Required for session prepare' : 'Use Defaults tab values'"
            showClear
            class="param-input"
            @update:model-value="setDefaultVoicePresetSelection"
          />
        </div>
      </div>
    </div>

    <!-- Defaults (+ API reference) -->
    <div v-show="activeTab === 'api'" class="config-tab-panel">
      <div class="config-card">
        <div class="section-label section-label--inline">
          {{ requestDefaultsSectionTitle }}
          <small class="section-hint"><code>{{ apiEndpoint }}</code></small>
        </div>

        <Message
          v-if="instructionsPolicyGuidance"
          severity="info"
          :closable="false"
          class="config-scan-message"
        >
          {{ instructionsPolicyGuidance }}
        </Message>

        <div v-if="swapSetParamsPreview" class="setparams-preview">
          <button
            type="button"
            class="setparams-preview__label setparams-preview__toggle"
            :aria-expanded="showSetParamsPreview"
            @click="showSetParamsPreview = !showSetParamsPreview"
          >
            <span>Saved request defaults preview</span>
            <i
              class="pi"
              :class="showSetParamsPreview ? 'pi-chevron-up' : 'pi-chevron-down'"
              aria-hidden="true"
            />
          </button>
          <pre v-if="showSetParamsPreview" class="setparams-preview__code">{{ JSON.stringify(swapSetParamsPreview, null, 2) }}</pre>
        </div>

        <template v-if="isProfiledAudioModel">
          <div v-if="requestFieldGroups.length" class="tts-subsection">
            <div
              v-for="group in requestFieldGroups"
              :key="group.id"
              class="tts-speech-group"
            >
              <div
                v-if="requestFieldGroups.length > 1"
                class="tts-speech-group__label"
              >
                {{ group.label }}
              </div>
              <div class="params-grid section-params">
                <div
                  v-for="field in group.fields"
                  :key="`${group.id}-${field.key}`"
                  class="param-field"
                >
                  <label class="param-field__label">
                    {{ field.label }}
                    <i
                      v-if="field.description || field.hint"
                      class="pi pi-info-circle param-info"
                      v-tooltip.top="requestDefaultFieldTooltip(field)"
                      tabindex="0"
                      aria-label="About request default field"
                    />
                  </label>
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
                  <Dropdown
                    v-else-if="fieldSelectOptions(field).length"
                    :model-value="requestDefaultValue(field) || ''"
                    :options="fieldSelectOptions(field)"
                    optionLabel="label"
                    optionValue="value"
                    :placeholder="field.placeholder || 'Choose a packaged voice'"
                    showClear
                    editable
                    class="param-input"
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
          No request defaults profile is available for this audio model.
        </Message>
      </div>

      <details class="config-card config-details-block">
        <summary class="config-details-summary">API example</summary>
        <div class="config-details-body">
          <div class="tts-subsection__head">
            <p class="config-muted-hint">
              Model id:
              <code>{{ config.model_alias || llamaSwapStableId || 'your-model-id' }}</code>
            </p>
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
          <Textarea
            :model-value="requestApiExample"
            readonly
            rows="10"
            class="w-full textarea-cli cmd-preview-textarea"
            autoResize
          />
        </div>
      </details>

      <details
        v-if="audioRequestCapabilities.length"
        class="config-card config-details-block"
      >
        <summary class="config-details-summary">
          Request-only parameters
          <span class="config-details-summary__count">{{ audioRequestCapabilities.length }}</span>
        </summary>
        <div class="config-details-body">
          <p class="config-muted-hint">
            Available on requests, not as startup settings. Prefer Defaults above when a matching field exists.
          </p>
          <div class="config-toolbar__row">
            <span class="p-input-icon-left config-search-wrap">
              <i class="pi pi-search" aria-hidden="true" />
              <InputText
                v-model="referenceSearchQuery"
                type="search"
                placeholder="Filter…"
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
                v-if="param.description"
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
      </details>
    </div>
  </div>
</template>

<script setup>
import { computed, ref, watch, onMounted } from 'vue'
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
  dependencyFieldTags,
  paramMatchesSearch,
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
const hideUnsupportedParams = ref(true)
const showAdvancedRuntime = ref(false)
const showSetParamsPreview = ref(false)
const rescanLoading = ref(false)
const referenceAudioItems = ref([])
const referenceAudioLoading = ref(false)
const referenceAudioUploading = ref(false)
const referenceAudioDeleting = ref('')
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
const COMMON_RUNTIME_ORDER = new Map(
  COMMON_RUNTIME_PARAM_ORDER.map((key, index) => [key, index]),
)
const COMMON_RUNTIME_SCOPE_PRIORITY = new Map([
  ['model', 0],
  ['process', 1],
  ['load_option', 2],
  ['session_option', 3],
])

const configRef = computed(() => props.config)
const registryRef = computed(() => props.paramRegistry)
const stableIdRef = computed(() => props.llamaSwapStableId)

const audio = useAudioModelConfig(configRef, registryRef, enginesStore, stableIdRef)

const contractGradeBadge = computed(() => {
  const grade = String(
    props.paramRegistry?.contract_grade
    || enginesStore.audioCppStatus?.contract_grade
    || '',
  ).trim().toLowerCase()
  // Success/complete metadata is silent — only surface incomplete contracts.
  if (!grade || grade === 'full') return null
  const warnings = props.paramRegistry?.contract_warnings
    || enginesStore.audioCppStatus?.contract_warnings
    || []
  const detail = warnings.length
    ? String(warnings[0])
    : 'Some details were inferred. Core settings still work; companion model paths may need a quick check.'
  return {
    severity: grade === 'partial' ? 'warn' : 'secondary',
    title: grade === 'partial'
      ? 'Some model details were inferred'
      : 'Limited model metadata for this build',
    detail,
  }
})

const {
  audioConfigGroups,
  audioRequestCapabilities,
  taskProfile,
  isProfiledAudioModel,
  requestFieldGroups,
  apiEndpoint,
  requestDefaultsSectionTitle,
  swapSetParamsPreview,
  instructionsPolicyGuidance,
  contractReviewRequired,
  contractFingerprint,
  markContractReviewed,
  setupProgress,
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
  setupChecklist,
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

const tabs = computed(() => [
  { id: 'overview', label: 'Overview', icon: 'pi pi-compass' },
  { id: 'server', label: 'Runtime', icon: 'pi pi-server' },
  { id: 'assets', label: 'Assets', icon: 'pi pi-folder-open' },
  { id: 'api', label: 'Defaults', icon: 'pi pi-sliders-h' },
])

const setupIncompleteCount = computed(() =>
  setupChecklist.value.filter((item) => !item.done).length,
)

function paramHasExtraInfo(param) {
  return Boolean(
    param?.description
    || param?.install_hint
    || param?.dependency
    || param?.primary_flag
    || param?.negative_flag
    || (param?.flags && param.flags.length)
    || param?.scope === 'load_option'
    || param?.scope === 'session_option',
  )
}

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
  if (tab === 'assets' && props.modelId) {
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
    await loadReferenceAudio()
    toast.add({
      severity: 'success',
      summary: 'Reference audio uploaded',
      detail: saved?.path
        ? `Saved as ${saved.display_path || saved.relative_path || saved.path}`
        : undefined,
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
      detail: item.display_path || item.relative_path || item.path,
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

function shouldPinAdvancedParam(param) {
  if (param.dependency || param.required) return true
  return audioParamHasExplicitValue(param)
}

function syncAdvancedParamKeys() {
  const candidates = advancedCandidateParams.value
  const candidateIds = new Set(candidates.map(advancedParamId))
  const next = new Set(
    advancedParamKeys.value.filter((id) => candidateIds.has(id)),
  )
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
  const byId = new Map(advancedCandidateParams.value.map((param) => [advancedParamId(param), param]))
  return advancedParamKeys.value
    .map((id) => byId.get(id))
    .filter(Boolean)
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

const filteredRequestCapabilities = computed(() => {
  const q = referenceSearchQuery.value.trim().toLowerCase()
  if (!q) return audioRequestCapabilities.value
  return audioRequestCapabilities.value.filter((param) => {
    const hay = [param.key, param.label, param.description].join(' ').toLowerCase()
    return hay.includes(q)
  })
})

function requestDefaultFieldTooltip(field) {
  const modelHints = [field?.description, field?.hint].filter(Boolean)
  if (modelHints.length) return modelHints.join('\n\n')
  return 'Saved as a request default for this model.'
}

async function rescanCliParams() {
  rescanLoading.value = true
  try {
    const data = await enginesStore.scanEngineParams('audio_cpp', null, {
      modelId: props.modelId || undefined,
    })
    const profileFailed = Boolean(data?.profile_scan_error)
    if (data?.ok && !profileFailed) {
      toast.add({
        severity: 'success',
        summary: 'Parameters scanned',
        detail: props.modelId
          ? `Indexed engine options and refreshed this model's session/request profile.`
          : `Indexed ${data.param_count ?? 0} options for audio.cpp.`,
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

.param-install-hint {
  display: block;
  margin-top: 0.35rem;
  color: var(--text-color-secondary);
  line-height: 1.35;
}

.config-message__actions {
  display: flex;
  flex-wrap: wrap;
  gap: 0.4rem;
  margin-top: 0.65rem;
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

.config-details-block {
  padding-top: 0.85rem;
  padding-bottom: 0.85rem;
}

.config-details-block > .config-details-summary {
  list-style: none;
  cursor: pointer;
  display: flex;
  align-items: center;
  gap: 0.5rem;
  font-size: 0.75rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--text-secondary, #9ca3af);
}

.config-details-block > .config-details-summary::-webkit-details-marker {
  display: none;
}

.config-details-summary__count {
  font-weight: 600;
  text-transform: none;
  letter-spacing: normal;
  opacity: 0.8;
}

.config-details-body {
  margin-top: 0.75rem;
  display: flex;
  flex-direction: column;
  gap: 0.65rem;
}

.setparams-preview__toggle {
  width: 100%;
  border: 0;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 0.75rem;
  color: inherit;
  cursor: pointer;
  text-align: left;
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
