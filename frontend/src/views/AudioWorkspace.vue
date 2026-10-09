<template>
  <div class="audio-workspace page-shell page-shell--relaxed page-shell--wide">
    <PageHeader title="Audio">
      <template #meta>
        <Tag v-if="selectedConfig?.family" :value="selectedConfig.family" severity="secondary" />
        <Tag v-if="selectedConfig?.task" :value="selectedConfig.task" severity="info" />
        <Tag
          v-if="selectedModel"
          :value="selectedModel.is_active ? 'Running' : 'Stopped'"
          :severity="selectedModel.is_active ? 'success' : 'secondary'"
        />
        <Tag :value="proxyStatusLabel" :severity="proxyStatusSeverity" />
      </template>
      <template #actions>
        <Button
          icon="pi pi-refresh"
          text
          severity="secondary"
          :loading="refreshing"
          aria-label="Refresh audio models"
          v-tooltip.top="'Refresh'"
          @click="refreshWorkspace"
        />
        <Button
          v-if="selectedModelId"
          label="Configure"
          icon="pi pi-cog"
          size="small"
          severity="secondary"
          outlined
          @click="openConfig"
        />
        <Button
          v-if="selectedModel && !selectedModel.is_active"
          label="Start"
          icon="pi pi-play"
          size="small"
          :loading="starting"
          @click="startSelected"
        />
      </template>
    </PageHeader>

    <LoadingState v-if="bootLoading" message="Loading audio models…" />

    <EmptyState
      v-else-if="!audioModels.length"
      icon="pi pi-volume-up"
      title="No audio.cpp models installed"
      description="Download an audio model to transcribe files or generate speech and music."
    >
      <Button label="Search models" icon="pi pi-search" @click="$router.push('/search')" />
      <Button
        label="Engines"
        icon="pi pi-cog"
        severity="secondary"
        outlined
        @click="$router.push('/engines')"
      />
    </EmptyState>

    <template v-else>
      <div class="config-card config-card--compact">
        <div class="section-label">Model</div>
        <div class="audio-model-bar">
          <Select
            v-model="selectedModelId"
            :options="modelOptions"
            optionLabel="label"
            optionValue="value"
            placeholder="Select an audio model"
            class="audio-model-dropdown"
          />
          <code v-if="inferenceModelId" class="param-key-hint" :title="'API model id'">{{
            inferenceModelId
          }}</code>
        </div>
        <p v-if="needsReferenceHint" class="config-muted-hint audio-model-hint">
          <span v-if="!referenceAudioOptions.length">No reference audio yet.</span>
          <Button
            v-if="!communityVoicesReady"
            label="Download preset voices"
            size="small"
            severity="secondary"
            :loading="communityVoicesInstalling"
            @click="installCommunityVoices"
          />
          <Button
            label="Add reference audio"
            size="small"
            severity="secondary"
            :loading="referenceAudioUploading"
            @click="openReferenceUpload"
          />
          <input
            ref="referenceUploadInput"
            type="file"
            accept=".wav,audio/wav,audio/*"
            class="sr-only"
            aria-label="Reference audio file"
            @change="onReferenceAudioSelected"
          />
        </p>
      </div>

      <div class="config-section-tabs" role="tablist" aria-label="Audio tasks">
        <button
          v-for="tab in visibleTabs"
          :key="tab.id"
          type="button"
          :id="`audio-tab-${tab.id}`"
          role="tab"
          class="config-section-tab"
          :class="{ selected: activeTab === tab.id }"
          :aria-selected="activeTab === tab.id"
          :aria-controls="`audio-panel-${tab.id}`"
          :tabindex="activeTab === tab.id ? 0 : -1"
          @click="activeTab = tab.id"
          @keydown="onRovingTabKeydown"
        >
          <span class="engine-option-label">
            <i :class="tab.icon" aria-hidden="true" />
            <span class="engine-name">{{ tab.label }}</span>
          </span>
        </button>
      </div>

      <Message
        v-if="selectedModel && !selectedModel.is_active"
        severity="warn"
        :closable="false"
        class="config-scan-message"
      >
        Start this model before running inference.
      </Message>

      <!-- Speech -->
      <div
        v-if="activeTab === 'speech'"
        id="audio-panel-speech"
        role="tabpanel"
        aria-labelledby="audio-tab-speech"
        tabindex="0"
        class="config-tab-panel audio-task-layout"
      >
        <div class="config-card">
          <div class="section-label">Speech</div>
          <div class="param-field">
            <label class="param-field__label" for="audio-speech-text">Text</label>
            <Textarea
              id="audio-speech-text"
              v-model="speechText"
              rows="4"
              class="w-full textarea-cli"
              placeholder="Text to synthesize"
            />
          </div>
          <div class="params-grid section-params">
            <div class="param-field">
              <label class="param-field__label">Voice preset</label>
              <Select
                v-model="speechVoice"
                :options="voicePresetOptions"
                optionLabel="label"
                optionValue="value"
                showClear
                placeholder="Default"
                class="param-input"
              />
            </div>
            <div class="param-field">
              <label class="param-field__label">Reference audio</label>
              <Select
                v-model="speechVoiceRef"
                :options="referenceAudioOptions"
                optionLabel="label"
                optionValue="value"
                showClear
                placeholder="From Assets (optional)"
                class="param-input"
                :loading="referenceAudioLoading"
              />
            </div>
            <div class="param-field">
              <label class="param-field__label">Language</label>
              <InputText v-model="speechLanguage" class="param-input" placeholder="optional" />
            </div>
          </div>
          <div class="audio-actions">
            <Button
              label="Generate"
              icon="pi pi-volume-up"
              :loading="speechLoading"
              :disabled="!canRun || !speechText.trim()"
              @click="runSpeech"
            />
            <Button
              label="Edit defaults"
              icon="pi pi-cog"
              size="small"
              text
              severity="secondary"
              @click="openConfig"
            />
          </div>
        </div>
        <AudioResultPanel
          :error="speechError"
          :clips="resultAudioClips"
          @download="onDownloadClip"
        />
      </div>

      <!-- Transcribe -->
      <div
        v-if="activeTab === 'transcribe'"
        id="audio-panel-transcribe"
        role="tabpanel"
        aria-labelledby="audio-tab-transcribe"
        tabindex="0"
        class="config-tab-panel audio-task-layout"
      >
        <div class="config-card">
          <div class="section-label">Transcribe</div>
          <p class="config-muted-hint">
            Upload or record audio. Non-WAV formats are converted to WAV at the file's sample rate
            and channel count. Several files are sent together as one batch of up to 32.
          </p>
          <div class="param-field section-params">
            <label class="param-field__label">Audio</label>
            <input
              ref="asrFileInput"
              type="file"
              multiple
              accept="audio/*,.wav,.ogg,.opus,.mp3,.webm,.m4a"
              class="audio-file-input"
              @change="onAsrFile"
            />
            <div class="audio-actions audio-actions--flush">
              <Button
                label="Choose files"
                icon="pi pi-upload"
                size="small"
                severity="secondary"
                outlined
                @click="asrFileInput?.click()"
              />
              <Button
                :label="recording ? 'Stop' : 'Record'"
                :icon="recording ? 'pi pi-stop' : 'pi pi-microphone'"
                size="small"
                severity="secondary"
                outlined
                @click="toggleRecord"
              />
            </div>
            <ul v-if="asrFiles.length" class="asr-file-list">
              <li v-for="(file, index) in asrFiles" :key="`${file.name}-${index}`">
                <span>{{ file.name }}</span>
                <button type="button" @click="removeAsrFile(index)">Remove</button>
              </li>
            </ul>
          </div>
          <div class="params-grid section-params">
            <div class="param-field">
              <label class="param-field__label">Language</label>
              <InputText v-model="asrLanguage" class="param-input" placeholder="en" />
            </div>
            <div class="param-field">
              <label class="param-field__label">Prompt</label>
              <InputText v-model="asrPrompt" class="param-input" placeholder="optional context" />
            </div>
          </div>
          <div class="audio-actions">
            <label>
              <input v-model="asrDetails" type="checkbox" :disabled="asrFiles.length > 1" />
              Include timestamps and speaker labels when available
            </label>
            <span v-if="asrFiles.length > 1" class="param-key-hint">
              Timestamps stay on a single file.
            </span>
            <Button
              :label="asrFiles.length > 1 ? `Transcribe ${asrFiles.length}` : 'Transcribe'"
              icon="pi pi-file"
              :loading="asrLoading"
              :disabled="!canRun || !asrFiles.length"
              @click="runAsr"
            />
          </div>
        </div>
        <div v-if="asrError || asrText || asrBatchResults.length" class="config-card">
          <div class="section-label">Transcript</div>
          <Message v-if="asrError" severity="error" :closable="false" class="config-scan-message">
            {{ asrError }}
          </Message>
          <ul v-if="asrBatchResults.length" class="asr-batch-results">
            <li v-for="(row, index) in asrBatchResults" :key="`${row.name}-${index}`">
              <strong>{{ row.name }}</strong>
              <pre class="audio-transcript">{{ row.text }}</pre>
            </li>
          </ul>
          <pre v-else-if="asrText" class="audio-transcript">{{ asrText }}</pre>
          <pre v-if="asrMetadata" class="audio-transcript">{{ asrMetadata }}</pre>
        </div>
      </div>

      <div
        v-if="!['speech', 'transcribe'].includes(activeTab)"
        :id="`audio-panel-${activeTab}`"
        role="tabpanel"
        :aria-labelledby="`audio-tab-${activeTab}`"
        tabindex="0"
        class="config-tab-panel audio-task-layout"
      >
        <div class="config-card">
          <div class="section-label">{{ taskLabel }}</div>
          <Message v-if="!workspaceFields.length" severity="info" :closable="false">
            The active engine has not provided request fields for this model. Scan the model
            in Configure to discover its inputs.
          </Message>
          <div v-for="field in workspaceFields" :key="field.key" class="param-field">
            <label class="param-field__label" :for="`audio-task-${field.key}`">
              {{ field.label || field.key }}{{ field.required ? ' *' : '' }}
            </label>
            <AudioParamField
              :id="`audio-task-${field.key}`"
              v-model="requestValues[field.key]"
              :param="field"
              :options="field.options || []"
              :disabled="taskLoading"
            />
            <small v-if="field.description" class="config-muted-hint">{{ field.description }}</small>
          </div>
          <div class="audio-actions">
            <Button
              :label="taskAction"
              icon="pi pi-play"
              :loading="taskLoading"
              :disabled="!canRunTask"
              @click="runSchemaTask"
            />
          </div>
        </div>
        <AudioResultPanel
          :error="taskError"
          :clips="resultAudioClips"
          @download="onDownloadClip"
        />
        <pre v-if="taskResult" class="audio-transcript">{{ taskResult }}</pre>
      </div>
    </template>
  </div>
</template>

<script setup>
import { computed, onMounted, onUnmounted, ref, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { useToast } from 'primevue/usetoast'
import Button from 'primevue/button'
import Tag from 'primevue/tag'
import Select from 'primevue/select'
import InputText from 'primevue/inputtext'
import Textarea from 'primevue/textarea'
import Message from 'primevue/message'
import PageHeader from '@/components/common/PageHeader.vue'
import EmptyState from '@/components/common/EmptyState.vue'
import LoadingState from '@/components/common/LoadingState.vue'
import AudioResultPanel from '@/components/audio/AudioResultPanel.vue'
import AudioParamField from '@/components/audio/AudioParamField.vue'
import { onRovingTabKeydown } from '@/composables/useRovingTabs'
import { useModelStore } from '@/stores/models'
import { useEnginesStore } from '@/stores/engines'
import {
  audioInferenceModelId,
  isGenericTaskEndpoint,
  extractAudioClipsFromTaskResult,
  acceptedSpeechRateKeys,
  fetchAudioVoices,
  speechRateRequestFields,
  synthesizeSpeech,
  taskKindFromConfig,
  transcribeAudio,
  transcribeAudioBatch,
  runAudioTask,
} from '@/composables/useAudioInferenceClient'

const route = useRoute()
const router = useRouter()
const toast = useToast()
const modelStore = useModelStore()
const enginesStore = useEnginesStore()

const ALL_TABS = [
  { id: 'task', label: 'Run task', icon: 'pi pi-play', kinds: ['task'] },
  { id: 'speech', label: 'Speech', icon: 'pi pi-volume-up', kinds: ['speech'] },
  { id: 'transcribe', label: 'Transcribe', icon: 'pi pi-microphone', kinds: ['transcribe'] },
  { id: 'music', label: 'Music', icon: 'pi pi-headphones', kinds: ['music'] },
  { id: 'convert', label: 'Convert', icon: 'pi pi-sync', kinds: ['convert'] },
  { id: 'separate', label: 'Separate', icon: 'pi pi-filter', kinds: ['separate'] },
  { id: 'analyze', label: 'Analyze', icon: 'pi pi-chart-bar', kinds: ['analyze'] },
  { id: 'design', label: 'Design', icon: 'pi pi-palette', kinds: ['design'] },
]

const bootLoading = ref(true)
const refreshing = ref(false)
const selectedModelId = ref('')
const selectedConfig = ref(null)
const acceptedSpeechRates = ref([])
const activeTab = ref('speech')
const starting = ref(false)

const speechText = ref('')
const speechVoice = ref(null)
const speechVoiceRef = ref(null)
const speechLanguage = ref('')
const speechLoading = ref(false)
const speechError = ref('')
const referenceAudioItems = ref([])
const referenceAudioLoading = ref(false)
const referenceAudioUploading = ref(false)
const communityVoicesInstalling = ref(false)
const referenceUploadInput = ref(null)
const REFERENCE_AUDIO_MAX_BYTES = 60 * 1024 * 1024
const engineVoices = ref([])
let voicesController = null

const asrFiles = ref([])
const asrFileInput = ref(null)
const asrBatchResults = ref([])
const asrLanguage = ref('en')
const asrPrompt = ref('')
const asrLoading = ref(false)
const asrError = ref('')
const asrText = ref('')
const asrDetails = ref(false)
const asrMetadata = ref('')
const recording = ref(false)
let mediaRecorder = null
let recordChunks = []

const modelRegistry = ref(null)
const requestValues = ref({})
const workspaceFields = computed(() => modelRegistry.value?.workspace_request_fields || [])
const taskLabel = computed(() => visibleTabs.value[0]?.label || 'Run task')
const taskAction = computed(() => ({ music: 'Generate', design: 'Generate' })[activeTab.value] || taskLabel.value)
const canRunTask = computed(() => canRun.value && workspaceFields.value.length > 0 &&
  workspaceFields.value.every((field) => !field.required || (
    requestValues.value[field.key] != null && String(requestValues.value[field.key]).trim() !== ''
  )))
const taskLoading = ref(false)
const taskError = ref('')
const taskResult = ref('')
/** Shared playable outputs for speech, music, VC, separation, design, etc. */
const resultAudioClips = ref([])

const audioModels = computed(() =>
  modelStore.allQuantizations.filter((m) => {
    const engine = m.config?.engine || m.engine
    return engine === 'audio_cpp' || m.format === 'audio_cpp'
  }),
)

const modelOptions = computed(() =>
  audioModels.value.map((m) => ({
    value: m.id,
    label: `${m.display_name || m.base_model_name || m.id}${m.is_active ? ' · running' : ''}`,
  })),
)

const selectedModel = computed(
  () => audioModels.value.find((m) => m.id === selectedModelId.value) || null,
)

const inferenceModelId = computed(() =>
  audioInferenceModelId(selectedModel.value, selectedConfig.value),
)

const proxyStatus = computed(() => enginesStore.systemStatus?.proxy_status)
const proxyHealthy = computed(() => Boolean(proxyStatus.value?.healthy))
const proxyStatusLabel = computed(() => {
  if (!proxyStatus.value || proxyStatus.value.healthy == null) return 'llama-swap status unknown'
  return proxyHealthy.value ? 'llama-swap ready' : 'llama-swap offline'
})
const proxyStatusSeverity = computed(() => {
  if (!proxyStatus.value || proxyStatus.value.healthy == null) return 'secondary'
  return proxyHealthy.value ? 'success' : 'danger'
})

const canRun = computed(() => Boolean(selectedModel.value?.is_active && inferenceModelId.value))


const communityVoiceIds = computed(() =>
  (referenceAudioItems.value || [])
    .filter((item) => item.storage === 'community' && item.voice_id)
    .map((item) => item.voice_id),
)

const communityVoicesReady = computed(() => communityVoiceIds.value.length > 0)

const voicePresetOptions = computed(() => {
  const presets = selectedConfig.value?.voice_presets
  const names = presets && typeof presets === 'object' ? Object.keys(presets) : []
  return [...new Set([...engineVoices.value, ...names, ...communityVoiceIds.value])].map((name) => ({
    label: name,
    value: name,
  }))
})

watch([inferenceModelId, canRun], async ([modelId, running]) => {
  voicesController?.abort()
  engineVoices.value = []
  if (!modelId || !running) return
  const controller = new AbortController()
  voicesController = controller
  try {
    const voices = await fetchAudioVoices({ modelId, signal: controller.signal })
    if (!controller.signal.aborted) engineVoices.value = voices
  } catch {
    // Configured presets remain usable when an older engine lacks voices API.
  }
})
onUnmounted(() => voicesController?.abort())

const referenceAudioOptions = computed(() =>
  (referenceAudioItems.value || []).map((item) => ({
    label: item.display_path || item.relative_path || item.path,
    value: item.path,
  })),
)

const modelKind = computed(() => {
  const kind = taskKindFromConfig(selectedConfig.value || {})
  return ['speech', 'transcribe'].includes(kind) && isGenericTaskEndpoint(modelRegistry.value?.api_endpoint)
    ? 'task' : kind
})

const visibleTabs = computed(() => {
  const kind = modelKind.value
  const matched = ALL_TABS.filter((tab) => tab.kinds.includes(kind))
  return matched.length ? matched : ALL_TABS.filter((tab) => tab.id === 'task')
})

const needsReferenceHint = computed(() =>
  ['speech', 'convert', 'separate', 'analyze'].includes(activeTab.value),
)

function openReferenceUpload() {
  referenceUploadInput.value?.click()
}

function formatBytes(bytes) {
  const value = Number(bytes) || 0
  if (value < 1024) return `${value} B`
  if (value < 1024 * 1024) return `${(value / 1024).toFixed(1)} KB`
  return `${(value / (1024 * 1024)).toFixed(1)} MB`
}

async function onReferenceAudioSelected(event) {
  const file = event.target.files?.[0]
  event.target.value = ''
  if (!file || !selectedModelId.value) return
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
    await modelStore.uploadReferenceAudio(selectedModelId.value, file)
    await loadReferenceAudio(selectedModelId.value)
    toast.add({ severity: 'success', summary: 'Reference audio uploaded', life: 3000 })
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

async function installCommunityVoices() {
  if (typeof modelStore.installCommunityVoices !== 'function' || !selectedModelId.value) return
  communityVoicesInstalling.value = true
  try {
    await modelStore.installCommunityVoices()
    await loadReferenceAudio(selectedModelId.value)
    toast.add({
      severity: 'success',
      summary: 'Community preset voices installed',
      detail: 'Apply the proxy config and restart the model, then pick a voice name or a community clip.',
      life: 5000,
    })
  } catch (error) {
    toast.add({
      severity: 'error',
      summary: 'Preset voice download failed',
      detail: error?.response?.data?.detail || error?.message || String(error),
      life: 5000,
    })
  } finally {
    communityVoicesInstalling.value = false
  }
}

async function loadReferenceAudio(modelId) {
  referenceAudioItems.value = []
  if (!modelId) return
  referenceAudioLoading.value = true
  try {
    const items = await modelStore.listReferenceAudio(modelId)
    if (selectedModelId.value === modelId) referenceAudioItems.value = Array.isArray(items) ? items : []
  } catch {
    if (selectedModelId.value === modelId) referenceAudioItems.value = []
  } finally {
    if (selectedModelId.value === modelId) referenceAudioLoading.value = false
  }
}

async function loadAcceptedSpeechRates(modelId) {
  acceptedSpeechRates.value = []
  modelRegistry.value = null
  if (!modelId) return
  try {
    const response = await fetch(
      `/api/models/param-registry?engine=audio_cpp&model_id=${encodeURIComponent(modelId)}`,
    )
    if (!response.ok) return
    const registry = await response.json()
    if (selectedModelId.value !== modelId) return
    modelRegistry.value = registry
    acceptedSpeechRates.value = acceptedSpeechRateKeys(registry)
  } catch {
    if (selectedModelId.value === modelId) acceptedSpeechRates.value = []
  }
}

async function refreshWorkspace() {
  refreshing.value = true
  try {
    await Promise.all([
      modelStore.fetchModels().catch(() => null),
      enginesStore.fetchSystemStatus().catch(() => null),
    ])
    const modelId = selectedModelId.value
    if (modelId) {
      await Promise.all([
        modelStore
          .getModelConfig(modelId)
          .then((cfg) => {
            if (selectedModelId.value === modelId) selectedConfig.value = cfg
          })
          .catch(() => null),
        loadReferenceAudio(modelId),
        loadAcceptedSpeechRates(modelId),
      ])
      if (selectedModelId.value === modelId) initializeRequestValues(selectedConfig.value)
    }
  } finally {
    refreshing.value = false
  }
}

watch(selectedModelId, async (id) => {
  selectedConfig.value = null
  acceptedSpeechRates.value = []
  speechVoiceRef.value = null
  speechVoice.value = null
  speechLanguage.value = ''
  requestValues.value = {}
  modelRegistry.value = null
  clearOutputs()
  if (!id) return
  try {
    const [cfg] = await Promise.all([
      modelStore.getModelConfig(id),
      loadReferenceAudio(id),
      loadAcceptedSpeechRates(id),
    ])
    if (selectedModelId.value !== id) return
    selectedConfig.value = cfg
    const kind = modelKind.value
    const preferred = kind === 'design' ? 'design' : kind
    if (!route.query.tab || !visibleTabs.value.some((t) => t.id === route.query.tab)) {
      activeTab.value = preferred
    } else if (!visibleTabs.value.some((t) => t.id === activeTab.value)) {
      activeTab.value = visibleTabs.value[0]?.id || preferred
    }
    const defaults = selectedConfig.value?.speech_defaults || {}
    if (defaults.language && !speechLanguage.value) speechLanguage.value = defaults.language
    if (defaults.voice_ref) speechVoiceRef.value = defaults.voice_ref
    const tDefaults = selectedConfig.value?.transcription_defaults || {}
    if (tDefaults.language) asrLanguage.value = tDefaults.language
    if (selectedConfig.value?.default_voice_preset) {
      speechVoice.value = selectedConfig.value.default_voice_preset
    }
    initializeRequestValues(selectedConfig.value)
  } catch (error) {
    toast.add({
      severity: 'error',
      summary: 'Failed to load config',
      detail: error?.message || String(error),
      life: 4000,
    })
  }
})

watch(
  () => route.query,
  (query) => {
    if (query.model && audioModels.value.some((m) => m.id === query.model)) {
      selectedModelId.value = query.model
    }
    if (query.tab && ALL_TABS.some((t) => t.id === query.tab)) {
      activeTab.value = query.tab
    }
  },
  { immediate: true },
)

watch(visibleTabs, (tabs) => {
  if (!tabs.some((t) => t.id === activeTab.value)) {
    activeTab.value = tabs[0]?.id || 'speech'
  }
})

watch([selectedModelId, activeTab], ([model, tab]) => {
  if (!model) return
  const nextQuery = { model, tab }
  if (route.query.model === model && route.query.tab === tab) return
  router.replace({ name: 'audio', query: nextQuery })
})

watch(activeTab, () => {
  // Avoid showing Speech audio on Convert/Design after switching tabs.
  speechError.value = ''
  taskError.value = ''
})

onMounted(async () => {
  bootLoading.value = true
  try {
    await Promise.all([
      modelStore.fetchModels().catch(() => null),
      enginesStore.fetchSystemStatus().catch(() => null),
    ])

    if (route.query.model) {
      selectedModelId.value = String(route.query.model)
    } else if (!selectedModelId.value && audioModels.value.length) {
      const running = audioModels.value.find((m) => m.is_active)
      selectedModelId.value = running?.id || audioModels.value[0].id
    }
    if (route.query.tab && ALL_TABS.some((t) => t.id === route.query.tab)) {
      activeTab.value = String(route.query.tab)
    }
  } finally {
    bootLoading.value = false
  }
})

onUnmounted(() => {
  clearResultAudio()
  stopRecorder()
})

function clearResultAudio() {
  for (const clip of resultAudioClips.value) {
    if (clip?.url) URL.revokeObjectURL(clip.url)
  }
  resultAudioClips.value = []
}

function publishAudioClips(clips) {
  clearResultAudio()
  resultAudioClips.value = (clips || []).map((clip) => ({
    ...clip,
    url: URL.createObjectURL(clip.blob),
  }))
}

function setSpeechResult(blob, filename = 'speech.wav') {
  taskResult.value = ''
  if (!blob) {
    clearResultAudio()
    return
  }
  publishAudioClips([
    {
      id: 'audio',
      label: 'Audio',
      blob,
      filename,
    },
  ])
}

function setTaskResult(result, { defaultFilename = 'audio.wav' } = {}) {
  speechError.value = ''
  taskResult.value = ''
  if (result == null) {
    clearResultAudio()
    return
  }

  const { clips, meta } = extractAudioClipsFromTaskResult(result)
  publishAudioClips(
    clips.map((clip) => (clip.id === 'audio' ? { ...clip, filename: defaultFilename } : clip)),
  )

  if (clips.length && meta) {
    // Keep timing / text / segments, omit giant base64 payloads.
    taskResult.value = JSON.stringify(meta, null, 2)
  } else if (!clips.length) {
    taskResult.value = typeof result === 'string' ? result : JSON.stringify(result, null, 2)
  }
}

function onDownloadClip(clip) {
  downloadBlob(clip?.blob, clip?.filename || 'audio.wav')
}

function clearOutputs() {
  speechError.value = ''
  asrError.value = ''
  asrText.value = ''
  taskError.value = ''
  taskResult.value = ''
  clearResultAudio()
}

function openConfig() {
  if (!selectedModelId.value) return
  router.push({ name: 'model-config', params: { id: selectedModelId.value } })
}

async function startSelected() {
  if (!selectedModelId.value) return
  starting.value = true
  try {
    await modelStore.startModel(selectedModelId.value)
    await modelStore.fetchModels()
    toast.add({ severity: 'success', summary: 'Model starting', life: 2500 })
  } catch (error) {
    toast.add({
      severity: 'error',
      summary: 'Start failed',
      detail: error?.response?.data?.detail || error?.message || String(error),
      life: 5000,
    })
  } finally {
    starting.value = false
  }
}

async function runSpeech() {
  speechError.value = ''
  taskError.value = ''
  speechLoading.value = true
  setSpeechResult(null)
  try {
    const extras = {}
    if (speechLanguage.value) extras.language = speechLanguage.value
    const defaults = selectedConfig.value?.speech_defaults || {}
    Object.assign(extras, pickDefined(defaults, ['voice_ref', 'reference_text', 'instruct']))
    Object.assign(extras, speechRateRequestFields(defaults, acceptedSpeechRates.value))
    if (speechVoiceRef.value) {
      extras.voice_ref = speechVoiceRef.value
      const match = (referenceAudioItems.value || []).find((item) => item.path === speechVoiceRef.value)
      if (match?.reference_text && !extras.reference_text) {
        extras.reference_text = match.reference_text
      }
    }
    const { blob } = await synthesizeSpeech({
      modelId: inferenceModelId.value,
      input: speechText.value,
      voice: speechVoice.value || undefined,
      extras,
    })
    setSpeechResult(blob, 'speech.wav')
  } catch (error) {
    speechError.value = error?.message || String(error)
  } finally {
    speechLoading.value = false
  }
}

function onAsrFile(event) {
  const picked = Array.from(event.target.files || [])
  event.target.value = ''
  if (!picked.length) return
  if (picked.length > 32) {
    toast.add({
      severity: 'warn',
      summary: 'Batch limit',
      detail: 'Only the first 32 files were kept.',
      life: 4000,
    })
  }
  asrFiles.value = picked.slice(0, 32)
  asrBatchResults.value = []
}

function removeAsrFile(index) {
  asrFiles.value = asrFiles.value.filter((_, itemIndex) => itemIndex !== index)
}

async function toggleRecord() {
  if (recording.value) {
    stopRecorder()
    return
  }
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
    recordChunks = []
    mediaRecorder = new MediaRecorder(stream)
    mediaRecorder.ondataavailable = (ev) => {
      if (ev.data?.size) recordChunks.push(ev.data)
    }
    mediaRecorder.onstop = () => {
      stream.getTracks().forEach((t) => t.stop())
      const blob = new Blob(recordChunks, { type: mediaRecorder?.mimeType || 'audio/webm' })
      asrFiles.value = [new File([blob], 'recording.webm', { type: blob.type })]
      asrBatchResults.value = []
      mediaRecorder = null
    }
    mediaRecorder.start()
    recording.value = true
  } catch (error) {
    toast.add({
      severity: 'error',
      summary: 'Microphone unavailable',
      detail: error?.message || String(error),
      life: 4000,
    })
  }
}

function stopRecorder() {
  if (mediaRecorder && recording.value) {
    mediaRecorder.stop()
  }
  recording.value = false
}

async function runAsr() {
  asrError.value = ''
  asrText.value = ''
  asrMetadata.value = ''
  asrBatchResults.value = []
  asrLoading.value = true
  try {
    if (asrFiles.value.length > 1) {
      const result = await transcribeAudioBatch({
        modelId: inferenceModelId.value,
        files: asrFiles.value,
        language: asrLanguage.value || undefined,
        prompt: asrPrompt.value || undefined,
      })
      asrBatchResults.value = Array.isArray(result?.results) ? result.results : []
      if (!asrBatchResults.value.length && result?.text) {
        asrText.value = result.text
      }
      return
    }
    const file = asrFiles.value[0]
    const result = await transcribeAudio({
      modelId: inferenceModelId.value,
      file,
      filename: file?.name,
      language: asrLanguage.value || undefined,
      prompt: asrPrompt.value || undefined,
      details: asrDetails.value,
    })
    asrText.value = result?.text || JSON.stringify(result, null, 2)
    if (asrDetails.value && result && typeof result === 'object') {
      const { text: _text, ...metadata } = result
      asrMetadata.value = Object.keys(metadata).length ? JSON.stringify(metadata, null, 2) : ''
    }
  } catch (error) {
    asrError.value = error?.message || String(error)
  } finally {
    asrLoading.value = false
  }
}

async function runTask(task, input, { defaultFilename = 'audio.wav' } = {}) {
  taskError.value = ''
  setTaskResult(null)
  taskLoading.value = true
  try {
    const result = await runAudioTask({
      modelId: inferenceModelId.value,
      task,
      input,
      proxyPort: enginesStore.systemStatus?.proxy_status?.port,
    })
    setTaskResult(result, { defaultFilename })
  } catch (error) {
    taskError.value = error?.message || String(error)
  } finally {
    taskLoading.value = false
  }
}

function initializeRequestValues(config) {
  const defaults = config?.[modelRegistry.value?.request_defaults_key || 'task_defaults'] || {}
  requestValues.value = Object.fromEntries(workspaceFields.value.map((field) => [
    field.key,
    (field.nested ? defaults.options?.[field.key] : defaults[field.key]) ?? field.default ?? null,
  ]))
}

function runSchemaTask() {
  if (!canRunTask.value) return
  const input = {}
  for (const field of workspaceFields.value) {
    const value = requestValues.value[field.key]
    if (value == null || value === '') continue
    if (field.nested) (input.options ||= {})[field.key] = value
    else input[field.key] = value
  }
  return runTask(selectedConfig.value?.task, input)
}

function downloadBlob(blob, name) {
  if (!blob) return
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = name
  a.click()
  URL.revokeObjectURL(url)
}

function pickDefined(obj, keys) {
  const out = {}
  for (const key of keys) {
    if (obj?.[key] != null && obj[key] !== '') out[key] = obj[key]
  }
  return out
}
</script>

<style scoped>
.audio-model-bar {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 0.65rem;
}

.audio-model-dropdown {
  flex: 1 1 16rem;
  min-width: 12rem;
}

.audio-model-hint {
  margin-top: 0.65rem;
}

.audio-inline-link {
  padding: 0 !important;
  vertical-align: baseline;
}

.audio-actions {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 0.5rem;
  margin-top: 0.85rem;
}

.audio-actions--flush {
  margin-top: 0.35rem;
}

.asr-file-list,
.asr-batch-results {
  list-style: none;
  margin: 0.4rem 0 0;
  padding: 0;
  display: grid;
  gap: 0.25rem;
}

.asr-file-list li {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 0.5rem;
  min-height: 2rem;
  font-size: 0.85rem;
}

.asr-file-list button {
  border: 0;
  background: transparent;
  color: var(--text-secondary);
  cursor: pointer;
  font: inherit;
  min-height: 2rem;
}

.asr-batch-results strong {
  font-size: 0.85rem;
}

@media (max-width: 768px) {
  .audio-model-dropdown {
    flex: 1 1 100%;
    min-width: 0;
    max-width: none;
  }
}

@media (min-width: 900px) {
  .audio-task-layout:has(> :nth-child(2)) {
    display: grid;
    grid-template-columns: minmax(0, 1.15fr) minmax(0, 0.85fr);
    align-items: start;
    gap: 0.75rem;
  }
}

.audio-player {
  display: block;
  width: 100%;
}

.task-audio-clip + .task-audio-clip {
  margin-top: 1rem;
  padding-top: 0.85rem;
  border-top: 1px solid var(--border-primary, #2a2f45);
}

.task-audio-clip .audio-actions {
  margin-top: 0.5rem;
}

.audio-transcript {
  margin: 0;
  padding: 0.75rem;
  border-radius: var(--radius-md, 0.5rem);
  border: 1px solid var(--border-primary, #2a2f45);
  background: rgba(0, 0, 0, 0.2);
  white-space: pre-wrap;
  word-break: break-word;
  font-size: 0.85rem;
  line-height: 1.45;
  max-height: 20rem;
  overflow: auto;
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
}

.audio-file-input {
  display: none;
}

:deep(.page-empty__actions) {
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
}
</style>
