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
        <Button
          v-if="canRun && inferenceModelId"
          label="audio.cpp UI"
          icon="pi pi-external-link"
          size="small"
          severity="secondary"
          outlined
          @click="openUpstreamUi"
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
        <AudioResultPanel
          :error="speechError"
          :clips="resultAudioClips"
          @download="onDownloadClip"
        />
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
            <label class="param-field__label">
              Lyrics
              <span v-if="!musicTagsRequired" class="section-hint">(optional)</span>
            </label>
            <Textarea v-model="musicLyrics" rows="3" class="w-full textarea-cli" />
          </div>
          <div class="param-field">
            <label class="param-field__label">
              Style tags
              <span v-if="!musicTagsRequired" class="section-hint">(optional)</span>
            </label>
            <InputText
              v-model="musicTags"
              class="param-input w-full"
              placeholder="pop, bright, drums, female vocal"
            />
            <p v-if="musicTagsRequired" class="config-muted-hint">
              HeartMuLa expects comma-separated style tags (genre, mood, instruments).
            </p>
          </div>
          <div class="audio-actions">
            <Button
              label="Generate"
              icon="pi pi-play"
              :loading="taskLoading"
              :disabled="!canRunMusic"
              @click="runMusic"
            />
          </div>
        </div>
        <AudioResultPanel
          :error="taskError"
          :clips="resultAudioClips"
          :text="taskResult"
          @download="onDownloadClip"
        />
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
              :loading="taskLoading"
              :disabled="!canRun || !vcSource"
              @click="runVc"
            />
          </div>
        </div>
        <AudioResultPanel
          :error="taskError"
          :clips="resultAudioClips"
          :text="taskResult"
          @download="onDownloadClip"
        />
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
        <AudioResultPanel
          :error="taskError"
          :clips="resultAudioClips"
          :text="taskResult"
          @download="onDownloadClip"
        />
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
        <AudioResultPanel
          :error="taskError"
          :clips="resultAudioClips"
          :text="taskResult"
          @download="onDownloadClip"
        />
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
        <AudioResultPanel
          :error="speechError"
          :clips="resultAudioClips"
          @download="onDownloadClip"
        />
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
import AudioResultPanel from '@/components/audio/AudioResultPanel.vue'
import { useModelStore } from '@/stores/models'
import { useEnginesStore } from '@/stores/engines'
import {
  audioInferenceModelId,
  extractAudioClipsFromTaskResult,
  audioCppUpstreamUiUrl,
  synthesizeSpeech,
  taskKindFromConfig,
  transcribeAudio,
  runAudioTask,
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
const musicTags = ref('')
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
/** Shared playable outputs for speech, music, VC, separation, design, etc. */
const resultAudioClips = ref([])

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

const canRun = computed(() =>
  Boolean(selectedModel.value?.is_active && inferenceModelId.value),
)

const musicFamily = computed(() =>
  String(selectedConfig.value?.family || '').toLowerCase().replace(/-/g, '_'),
)

const musicTagsRequired = computed(() => musicFamily.value === 'heartmula')

const canRunMusic = computed(() => {
  if (!canRun.value || !musicPrompt.value.trim()) return false
  if (musicTagsRequired.value && !musicTags.value.trim()) return false
  return true
})

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
          syncMusicDefaultsFromConfig(cfg)
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
  musicPrompt.value = ''
  musicLyrics.value = ''
  musicTags.value = ''
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
    syncMusicDefaultsFromConfig(selectedConfig.value)
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
    clips.map((clip) => (
      clip.id === 'audio'
        ? { ...clip, filename: defaultFilename }
        : clip
    )),
  )

  if (clips.length && meta) {
    // Keep timing / text / segments, omit giant base64 payloads.
    taskResult.value = JSON.stringify(meta, null, 2)
  } else if (!clips.length) {
    taskResult.value = typeof result === 'string'
      ? result
      : JSON.stringify(result, null, 2)
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

function openUpstreamUi() {
  const modelId = inferenceModelId.value
  if (!modelId || typeof window === 'undefined') return
  const url = audioCppUpstreamUiUrl(
    modelId,
    enginesStore.systemStatus?.proxy_status?.port,
  )
  if (!url) return
  window.open(url, '_blank', 'noopener,noreferrer')
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
    if (speechVoiceRef.value) extras.voice_ref = speechVoiceRef.value
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

async function runDesign() {
  speechError.value = ''
  taskError.value = ''
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
    setSpeechResult(blob, 'voice-design.wav')
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

function syncMusicDefaultsFromConfig(config) {
  const defaults = config?.task_defaults && typeof config.task_defaults === 'object'
    ? config.task_defaults
    : {}
  const optionTags =
    defaults.options && typeof defaults.options === 'object' ? defaults.options.tags : undefined
  const tags = musicTags.value.trim()
    ? musicTags.value
    : (defaults.tags || optionTags || '')
  if (tags && !musicTags.value.trim()) musicTags.value = String(tags)
  if (!musicLyrics.value.trim() && defaults.lyrics) musicLyrics.value = String(defaults.lyrics)
  if (!musicPrompt.value.trim() && defaults.text) musicPrompt.value = String(defaults.text)
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
  const tags = (musicTags.value || defaults.tags || options.tags || '').toString().trim()
  if (tags) options.tags = tags
  else delete options.tags
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
    ]),
    // ACE-Step / Stable Audio / HeartMuLa use `text`, not OpenAI-style `prompt`.
    text: musicPrompt.value || defaults.text || undefined,
    lyrics: musicLyrics.value || defaults.lyrics || undefined,
  }
  if (Object.keys(options).length) input.options = options
  return runTask(config.task || 'gen', input, { defaultFilename: 'music.wav' })
}

async function runVc() {
  // audio.cpp VC/SVC/S2S always goes through llama-swap /audioapi/v1/tasks/run with source ``audio``
  // and target ``voice_ref`` (vevo2 also accepts source_audio / target_voice).
  const config = selectedConfig.value || {}
  const defaults = config.task_defaults && typeof config.task_defaults === 'object'
    ? config.task_defaults
    : {}
  const options = {
    ...(defaults.options && typeof defaults.options === 'object' ? defaults.options : {}),
  }
  if (defaults.task_route) options.task_route = defaults.task_route
  const input = {
    ...pickDefined(defaults, [
      'text',
      'target_text',
      'reference_text',
      'style_ref',
      'prosody_ref',
      'seed',
    ]),
    audio: vcSource.value,
    voice_ref: vcTarget.value || defaults.voice_ref || undefined,
    // VeVo2 option aliases (harmless for seed_vc / miocodec / chatterbox).
    source_audio: vcSource.value,
    target_voice: vcTarget.value || undefined,
  }
  if (Object.keys(options).length) input.options = options
  return runTask(config.task || 'vc', input, { defaultFilename: 'converted.wav' })
}

function runSep() {
  return runTask('sep', { audio: sepPath.value }, { defaultFilename: 'separated.wav' })
}

function runAnalyze() {
  return runTask(analyzeTask.value, { audio: analyzePath.value })
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
