<template>
  <div class="audio-workspace page-shell page-shell--relaxed">
    <PageHeader title="Audio">
      <template #meta>
        <Tag
          v-if="selectedConfig?.family"
          :value="selectedConfig.family"
          severity="secondary"
        />
        <Tag
          v-if="selectedConfig?.task"
          :value="selectedConfig.task"
          severity="info"
        />
        <Tag
          v-if="selectedModel"
          :value="selectedModel.is_active ? 'Running' : 'Stopped'"
          :severity="selectedModel.is_active ? 'success' : 'secondary'"
        />
        <Tag
          :value="proxyHealthy ? 'llama-swap ready' : 'llama-swap offline'"
          :severity="proxyHealthy ? 'success' : 'danger'"
        />
      </template>
      <template #actions>
        <Button
          icon="pi pi-refresh"
          text
          severity="secondary"
          :loading="refreshing"
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
      description="Install an audio package from Search, then configure family/task and start it."
    >
      <Button label="Search models" icon="pi pi-search" @click="$router.push('/search')" />
      <Button label="Engines" icon="pi pi-cog" severity="secondary" outlined @click="$router.push('/engines')" />
    </EmptyState>

    <template v-else>
      <div class="config-card config-card--compact">
        <div class="section-label">Model</div>
        <div class="audio-model-bar">
          <Dropdown
            v-model="selectedModelId"
            :options="modelOptions"
            optionLabel="label"
            optionValue="value"
            placeholder="Select an audio model"
            class="audio-model-dropdown"
          />
          <code v-if="inferenceModelId" class="param-key-hint" :title="'API model id'">{{ inferenceModelId }}</code>
        </div>
        <p v-if="!referenceAudioOptions.length && needsReferenceHint" class="config-muted-hint audio-model-hint">
          No reference audio yet.
          <Button label="Add in Config → Assets" link size="small" class="audio-inline-link" @click="openConfig" />
        </p>
      </div>

      <div class="config-section-tabs" role="tablist" aria-label="Audio tasks">
        <button
          v-for="tab in visibleTabs"
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

      <Message
        v-if="selectedModel && !selectedModel.is_active"
        severity="warn"
        :closable="false"
        class="config-scan-message"
      >
        Start this model before running inference.
      </Message>

      <!-- Speech -->
      <div v-if="activeTab === 'speech'" class="config-tab-panel">
        <div class="config-card">
          <div class="section-label">Speech</div>
          <div class="param-field">
            <label class="param-field__label">Text</label>
            <Textarea
              v-model="speechText"
              rows="4"
              class="w-full textarea-cli"
              placeholder="Text to synthesize"
            />
          </div>
          <div class="params-grid section-params">
            <div class="param-field">
              <label class="param-field__label">Voice preset</label>
              <Dropdown
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
              <Dropdown
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
        <div v-if="speechError || speechUrl" class="config-card">
          <div class="section-label">Result</div>
          <Message v-if="speechError" severity="error" :closable="false" class="config-scan-message">
            {{ speechError }}
          </Message>
          <audio v-if="speechUrl" :src="speechUrl" controls class="audio-player" />
          <div v-if="speechUrl" class="audio-actions">
            <Button
              label="Download WAV"
              icon="pi pi-download"
              size="small"
              severity="secondary"
              outlined
              @click="downloadBlob(speechBlob, 'speech.wav')"
            />
          </div>
        </div>
      </div>

      <!-- Transcribe -->
      <div v-if="activeTab === 'transcribe'" class="config-tab-panel">
        <div class="config-card">
          <div class="section-label">Transcribe</div>
          <p class="config-muted-hint">
            Upload or record audio. Non-WAV formats are converted automatically before transcription.
          </p>
          <div class="param-field section-params">
            <label class="param-field__label">Audio</label>
            <input
              ref="asrFileInput"
              type="file"
              accept="audio/*,.wav,.ogg,.opus,.mp3,.webm,.m4a"
              class="audio-file-input"
              @change="onAsrFile"
            />
            <div class="audio-actions audio-actions--flush">
              <Button
                label="Choose file"
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
              <span v-if="asrFileName" class="param-key-hint">{{ asrFileName }}</span>
            </div>
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
            <Button
              label="Transcribe"
              icon="pi pi-file"
              :loading="asrLoading"
              :disabled="!canRun || !asrFile"
              @click="runAsr"
            />
          </div>
        </div>
        <div v-if="asrError || asrText" class="config-card">
          <div class="section-label">Transcript</div>
          <Message v-if="asrError" severity="error" :closable="false" class="config-scan-message">
            {{ asrError }}
          </Message>
          <pre v-if="asrText" class="audio-transcript">{{ asrText }}</pre>
        </div>
      </div>

      <!-- Music -->
      <div v-if="activeTab === 'music'" class="config-tab-panel">
        <div class="config-card">
          <div class="section-label">Music</div>
          <div class="param-field">
            <label class="param-field__label">Prompt</label>
            <Textarea
              v-model="musicPrompt"
              rows="3"
              class="w-full textarea-cli"
              placeholder="Style, instruments, mood"
            />
          </div>
          <div class="param-field">
            <label class="param-field__label">Lyrics <span class="section-hint">(optional)</span></label>
            <Textarea v-model="musicLyrics" rows="3" class="w-full textarea-cli" />
          </div>
          <div class="audio-actions">
            <Button
              label="Generate"
              icon="pi pi-play"
              :loading="taskLoading"
              :disabled="!canRun || !musicPrompt.trim()"
              @click="runMusic"
            />
          </div>
        </div>
        <div v-if="taskError || taskResult" class="config-card">
          <div class="section-label">Result</div>
          <Message v-if="taskError" severity="error" :closable="false" class="config-scan-message">
            {{ taskError }}
          </Message>
          <pre v-if="taskResult" class="audio-transcript">{{ taskResult }}</pre>
        </div>
      </div>

      <!-- Voice conversion -->
      <div v-if="activeTab === 'convert'" class="config-tab-panel">
        <div class="config-card">
          <div class="section-label">Voice conversion</div>
          <p class="config-muted-hint">
            Choose source and target from this model’s Assets, or enter a server path.
          </p>
          <div class="params-grid">
            <div class="param-field">
              <label class="param-field__label">Source audio</label>
              <Dropdown
                v-model="vcSource"
                :options="referenceAudioOptions"
                optionLabel="label"
                optionValue="value"
                editable
                placeholder="refs/… or server path"
                class="param-input"
              />
            </div>
            <div class="param-field">
              <label class="param-field__label">Target voice</label>
              <Dropdown
                v-model="vcTarget"
                :options="referenceAudioOptions"
                optionLabel="label"
                optionValue="value"
                editable
                showClear
                placeholder="refs/… or server path"
                class="param-input"
              />
            </div>
          </div>
          <div class="audio-actions">
            <Button
              label="Convert"
              icon="pi pi-sync"
              :loading="taskLoading || speechLoading"
              :disabled="!canRun || !vcSource"
              @click="runVc"
            />
          </div>
        </div>
        <div v-if="taskError || speechError || speechUrl || taskResult" class="config-card">
          <div class="section-label">Result</div>
          <Message
            v-if="taskError || speechError"
            severity="error"
            :closable="false"
            class="config-scan-message"
          >
            {{ taskError || speechError }}
          </Message>
          <audio v-if="speechUrl" :src="speechUrl" controls class="audio-player" />
          <pre v-if="taskResult" class="audio-transcript">{{ taskResult }}</pre>
        </div>
      </div>

      <!-- Separation -->
      <div v-if="activeTab === 'separate'" class="config-tab-panel">
        <div class="config-card">
          <div class="section-label">Source separation</div>
          <div class="params-grid">
            <div class="param-field">
              <label class="param-field__label">Audio path</label>
              <Dropdown
                v-model="sepPath"
                :options="referenceAudioOptions"
                optionLabel="label"
                optionValue="value"
                editable
                placeholder="refs/… or server-local WAV"
                class="param-input"
              />
            </div>
          </div>
          <div class="audio-actions">
            <Button
              label="Separate"
              icon="pi pi-filter"
              :loading="taskLoading"
              :disabled="!canRun || !sepPath"
              @click="runSep"
            />
          </div>
        </div>
        <div v-if="taskError || taskResult" class="config-card">
          <div class="section-label">Result</div>
          <Message v-if="taskError" severity="error" :closable="false" class="config-scan-message">
            {{ taskError }}
          </Message>
          <pre v-if="taskResult" class="audio-transcript">{{ taskResult }}</pre>
        </div>
      </div>

      <!-- Analysis -->
      <div v-if="activeTab === 'analyze'" class="config-tab-panel">
        <div class="config-card">
          <div class="section-label">Analysis</div>
          <div class="params-grid">
            <div class="param-field">
              <label class="param-field__label">Task</label>
              <Dropdown
                v-model="analyzeTask"
                :options="analyzeTaskOptions"
                optionLabel="label"
                optionValue="value"
                class="param-input"
              />
            </div>
            <div class="param-field">
              <label class="param-field__label">Audio path</label>
              <Dropdown
                v-model="analyzePath"
                :options="referenceAudioOptions"
                optionLabel="label"
                optionValue="value"
                editable
                placeholder="refs/… or server-local WAV"
                class="param-input"
              />
            </div>
          </div>
          <div class="audio-actions">
            <Button
              label="Analyze"
              icon="pi pi-chart-bar"
              :loading="taskLoading"
              :disabled="!canRun || !analyzePath"
              @click="runAnalyze"
            />
          </div>
        </div>
        <div v-if="taskError || taskResult" class="config-card">
          <div class="section-label">Result</div>
          <Message v-if="taskError" severity="error" :closable="false" class="config-scan-message">
            {{ taskError }}
          </Message>
          <pre v-if="taskResult" class="audio-transcript">{{ taskResult }}</pre>
        </div>
      </div>

      <!-- Voice design -->
      <div v-if="activeTab === 'design'" class="config-tab-panel">
        <div class="config-card">
          <div class="section-label">Voice design</div>
          <div class="param-field">
            <label class="param-field__label">Caption</label>
            <Textarea
              v-model="designCaption"
              rows="3"
              class="w-full textarea-cli"
              placeholder="Describe the target voice"
            />
          </div>
          <div class="param-field">
            <label class="param-field__label">Text to speak</label>
            <Textarea v-model="designText" rows="3" class="w-full textarea-cli" />
          </div>
          <div class="audio-actions">
            <Button
              label="Generate"
              icon="pi pi-palette"
              :loading="speechLoading"
              :disabled="!canRun || !designText.trim()"
              @click="runDesign"
            />
          </div>
        </div>
        <div v-if="speechError || speechUrl" class="config-card">
          <div class="section-label">Result</div>
          <Message v-if="speechError" severity="error" :closable="false" class="config-scan-message">
            {{ speechError }}
          </Message>
          <audio v-if="speechUrl" :src="speechUrl" controls class="audio-player" />
        </div>
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
import Dropdown from 'primevue/dropdown'
import InputText from 'primevue/inputtext'
import Textarea from 'primevue/textarea'
import Message from 'primevue/message'
import PageHeader from '@/components/common/PageHeader.vue'
import EmptyState from '@/components/common/EmptyState.vue'
import LoadingState from '@/components/common/LoadingState.vue'
import { useModelStore } from '@/stores/models'
import { useEnginesStore } from '@/stores/engines'
import {
  audioInferenceModelId,
  synthesizeSpeech,
  taskKindFromConfig,
  transcribeAudio,
  runAudioTask,
  usesSpeechForConversion,
} from '@/composables/useAudioInferenceClient'

const route = useRoute()
const router = useRouter()
const toast = useToast()
const modelStore = useModelStore()
const enginesStore = useEnginesStore()

const ALL_TABS = [
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
const activeTab = ref('speech')
const starting = ref(false)

const speechText = ref('')
const speechVoice = ref(null)
const speechVoiceRef = ref(null)
const speechLanguage = ref('')
const speechLoading = ref(false)
const speechError = ref('')
const speechBlob = ref(null)
const speechUrl = ref('')
const referenceAudioItems = ref([])
const referenceAudioLoading = ref(false)

const asrFile = ref(null)
const asrFileName = ref('')
const asrFileInput = ref(null)
const asrLanguage = ref('en')
const asrPrompt = ref('')
const asrLoading = ref(false)
const asrError = ref('')
const asrText = ref('')
const recording = ref(false)
let mediaRecorder = null
let recordChunks = []

const musicPrompt = ref('')
const musicLyrics = ref('')
const vcSource = ref('')
const vcTarget = ref('')
const sepPath = ref('')
const analyzeTask = ref('vad')
const analyzePath = ref('')
const designCaption = ref('')
const designText = ref('')
const taskLoading = ref(false)
const taskError = ref('')
const taskResult = ref('')

const analyzeTaskOptions = [
  { label: 'VAD', value: 'vad' },
  { label: 'Diarization', value: 'diar' },
  { label: 'Alignment', value: 'align' },
]

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

const selectedModel = computed(() =>
  audioModels.value.find((m) => m.id === selectedModelId.value) || null,
)

const inferenceModelId = computed(() =>
  audioInferenceModelId(selectedModel.value, selectedConfig.value),
)

const proxyHealthy = computed(() =>
  Boolean(enginesStore.systemStatus?.proxy_status?.healthy),
)

const vcUsesSpeech = computed(() =>
  usesSpeechForConversion(selectedConfig.value || {}, selectedModel.value),
)

const canRun = computed(() =>
  Boolean(selectedModel.value?.is_active && inferenceModelId.value),
)

const voicePresetOptions = computed(() => {
  const presets = selectedConfig.value?.voice_presets
  if (!presets || typeof presets !== 'object') return []
  return Object.keys(presets).map((name) => ({ label: name, value: name }))
})

const referenceAudioOptions = computed(() =>
  (referenceAudioItems.value || []).map((item) => ({
    label: item.display_path || item.relative_path || item.path,
    value: item.path,
  })),
)

const modelKind = computed(() => taskKindFromConfig(selectedConfig.value || {}))

const visibleTabs = computed(() => {
  const kind = modelKind.value
  const matched = ALL_TABS.filter((tab) => tab.kinds.includes(kind))
  return matched.length ? matched : ALL_TABS.filter((tab) => tab.id === 'speech')
})

const needsReferenceHint = computed(() =>
  ['speech', 'convert', 'separate', 'analyze'].includes(activeTab.value),
)

async function loadReferenceAudio(modelId) {
  referenceAudioItems.value = []
  if (!modelId) return
  referenceAudioLoading.value = true
  try {
    const items = await modelStore.listReferenceAudio(modelId)
    referenceAudioItems.value = Array.isArray(items) ? items : []
  } catch {
    referenceAudioItems.value = []
  } finally {
    referenceAudioLoading.value = false
  }
}

async function refreshWorkspace() {
  refreshing.value = true
  try {
    await Promise.all([
      modelStore.fetchModels().catch(() => null),
      enginesStore.fetchSystemStatus().catch(() => null),
    ])
    if (selectedModelId.value) {
      await Promise.all([
        modelStore.getModelConfig(selectedModelId.value).then((cfg) => {
          selectedConfig.value = cfg
        }).catch(() => null),
        loadReferenceAudio(selectedModelId.value),
      ])
    }
  } finally {
    refreshing.value = false
  }
}

watch(selectedModelId, async (id) => {
  selectedConfig.value = null
  speechVoiceRef.value = null
  clearOutputs()
  if (!id) return
  try {
    const [cfg] = await Promise.all([
      modelStore.getModelConfig(id),
      loadReferenceAudio(id),
    ])
    selectedConfig.value = cfg
    const kind = taskKindFromConfig(selectedConfig.value)
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

watch(
  [selectedModelId, activeTab],
  ([model, tab]) => {
    if (!model) return
    const nextQuery = { model, tab }
    if (route.query.model === model && route.query.tab === tab) return
    router.replace({ name: 'audio', query: nextQuery })
  },
)

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
  if (speechUrl.value) URL.revokeObjectURL(speechUrl.value)
  stopRecorder()
})

function clearOutputs() {
  speechError.value = ''
  asrError.value = ''
  asrText.value = ''
  taskError.value = ''
  taskResult.value = ''
  setSpeechResult(null)
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

function setSpeechResult(blob) {
  if (speechUrl.value) URL.revokeObjectURL(speechUrl.value)
  speechBlob.value = blob
  speechUrl.value = blob ? URL.createObjectURL(blob) : ''
}

async function runSpeech() {
  speechError.value = ''
  speechLoading.value = true
  setSpeechResult(null)
  try {
    const extras = {}
    if (speechLanguage.value) extras.language = speechLanguage.value
    const defaults = selectedConfig.value?.speech_defaults || {}
    Object.assign(extras, pickDefined(defaults, ['voice_ref', 'reference_text', 'instruct']))
    if (speechVoiceRef.value) extras.voice_ref = speechVoiceRef.value
    const { blob } = await synthesizeSpeech({
      modelId: inferenceModelId.value,
      input: speechText.value,
      voice: speechVoice.value || undefined,
      extras,
    })
    setSpeechResult(blob)
  } catch (error) {
    speechError.value = error?.message || String(error)
  } finally {
    speechLoading.value = false
  }
}

async function runDesign() {
  speechError.value = ''
  speechLoading.value = true
  setSpeechResult(null)
  try {
    const extras = {}
    if (designCaption.value) {
      extras.instruct = designCaption.value
      extras.caption = designCaption.value
    }
    const { blob } = await synthesizeSpeech({
      modelId: inferenceModelId.value,
      input: designText.value,
      extras,
    })
    setSpeechResult(blob)
  } catch (error) {
    speechError.value = error?.message || String(error)
  } finally {
    speechLoading.value = false
  }
}

function onAsrFile(event) {
  const file = event.target.files?.[0]
  event.target.value = ''
  if (!file) return
  asrFile.value = file
  asrFileName.value = file.name
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
      asrFile.value = new File([blob], 'recording.webm', { type: blob.type })
      asrFileName.value = 'recording.webm'
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
  asrLoading.value = true
  try {
    const result = await transcribeAudio({
      modelId: inferenceModelId.value,
      file: asrFile.value,
      filename: asrFileName.value,
      language: asrLanguage.value || undefined,
      prompt: asrPrompt.value || undefined,
    })
    asrText.value = result?.text || JSON.stringify(result, null, 2)
  } catch (error) {
    asrError.value = error?.message || String(error)
  } finally {
    asrLoading.value = false
  }
}

async function runTask(task, input) {
  taskError.value = ''
  taskResult.value = ''
  taskLoading.value = true
  try {
    const result = await runAudioTask({
      modelId: inferenceModelId.value,
      task,
      input,
    })
    taskResult.value = typeof result === 'string'
      ? result
      : JSON.stringify(result, null, 2)
  } catch (error) {
    taskError.value = error?.message || String(error)
  } finally {
    taskLoading.value = false
  }
}

function runMusic() {
  const config = selectedConfig.value || {}
  const defaults = config.task_defaults && typeof config.task_defaults === 'object'
    ? config.task_defaults
    : {}
  const family = String(config.family || '').toLowerCase().replace(/-/g, '_')
  const options = {
    ...(defaults.options && typeof defaults.options === 'object' ? defaults.options : {}),
  }
  if (defaults.task_route) options.task_route = defaults.task_route
  if (family === 'ace_step' && !options.task_route) {
    options.task_route = 'text2music'
  }
  const input = {
    ...pickDefined(defaults, [
      'language',
      'audio',
      'duration_seconds',
      'repaint_start',
      'repaint_end',
      'num_inference_steps',
      'guidance_scale',
      'seed',
      'tags',
    ]),
    // ACE-Step / Stable Audio / HeartMuLa use `text`, not OpenAI-style `prompt`.
    text: musicPrompt.value || defaults.text || undefined,
    lyrics: musicLyrics.value || defaults.lyrics || undefined,
  }
  if (Object.keys(options).length) input.options = options
  return runTask(config.task || 'gen', input)
}

async function runVc() {
  taskError.value = ''
  speechError.value = ''
  taskResult.value = ''
  if (vcUsesSpeech.value) {
    speechLoading.value = true
    setSpeechResult(null)
    try {
      const defaults = selectedConfig.value?.speech_defaults || {}
      const extras = {
        ...pickDefined(defaults, ['reference_text', 'instruct', 'instructions']),
        audio_path: vcSource.value,
      }
      if (vcTarget.value) extras.voice_ref = vcTarget.value
      const { blob } = await synthesizeSpeech({
        modelId: inferenceModelId.value,
        input: defaults.text || ' ',
        extras,
      })
      setSpeechResult(blob)
    } catch (error) {
      speechError.value = error?.message || String(error)
    } finally {
      speechLoading.value = false
    }
    return
  }
  return runTask(selectedConfig.value?.task || 'vc', {
    audio_path: vcSource.value,
    voice_ref: vcTarget.value || undefined,
  })
}

function runSep() {
  return runTask('sep', { audio_path: sepPath.value })
}

function runAnalyze() {
  return runTask(analyzeTask.value, { audio_path: analyzePath.value })
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
  max-width: 36rem;
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

.audio-player {
  display: block;
  width: 100%;
  max-width: 36rem;
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
