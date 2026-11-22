<template>
  <div class="audio-workspace page-shell page-shell--relaxed page-shell--wide">
    <PageHeader
      title="Audio"
      description="Run speech, transcription, and audio tasks on an audio.cpp model, then play or download the result."
    >
      <template #meta>
        <Tag v-if="selectedConfig?.family" :value="selectedConfig.family" severity="secondary" />
        <Tag
          v-if="selectedConfig?.task"
          :value="taskDisplayLabel(selectedConfig.task)"
          severity="info"
        />
        <Tag
          v-if="selectedModel"
          :value="selectedRuntimeState.label"
          :severity="selectedRuntimeState.severity"
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
          @click="startSelected()"
        />
      </template>
    </PageHeader>

    <LoadingState v-if="bootLoading" message="Loading audio models…" />

    <EmptyState
      v-else-if="workspaceError && !audioModels.length"
      icon="pi pi-exclamation-circle"
      title="Could not load audio models"
      :description="workspaceError"
    >
      <Button label="Retry" icon="pi pi-refresh" @click="refreshWorkspace" />
    </EmptyState>
    <EmptyState
      v-else-if="audioEngineDisabled"
      icon="pi pi-power-off"
      title="audio.cpp is disabled"
      description="The audio.cpp engine is turned off by the AUDIO_CPP_ENABLED server setting. Enable it and restart Studio before installing audio models."
    >
      <Button label="Open Engines" icon="pi pi-cog" @click="$router.push('/engines')" />
    </EmptyState>
    <EmptyState
      v-else-if="audioEngineMissing"
      icon="pi pi-exclamation-circle"
      title="audio.cpp engine unavailable"
      description="This Studio build does not expose an audio.cpp engine. Enable or install audio.cpp on the Engines page before adding audio models."
    >
      <Button label="Open Engines" icon="pi pi-cog" @click="$router.push('/engines')" />
    </EmptyState>
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
      <StateBanner v-if="workspaceError" severity="error" icon="">
        {{ workspaceError }}
        <template #actions>
          <Button label="Retry" size="small" @click="refreshWorkspace" />
        </template>
      </StateBanner>
      <StateBanner v-if="configError" severity="error" icon="">
        {{ configError }}
        <template #actions>
          <Button label="Retry configuration" size="small" @click="refreshWorkspace" />
        </template>
      </StateBanner>
      <StateBanner v-if="contractError" severity="error" icon="">
        {{ contractError }}
        <template #actions>
          <Button label="Retry capabilities" size="small" @click="loadWorkspaceData(true)" />
        </template>
      </StateBanner>
      <StateBanner
        v-if="driftWarnings.length"
        severity="warning"
        role="status"
        class="contract-drift"
      >
        <details>
          <summary>Engine contract drift ({{ driftWarnings.length }})</summary>
          <ul>
            <li v-for="(warning, index) in driftWarnings" :key="`drift-${index}`">
              {{ warning }}
            </li>
          </ul>
        </details>
      </StateBanner>
      <StateBanner v-if="contractValidationHint" severity="info" role="status">
        {{ contractValidationHint }}
      </StateBanner>
      <StateBanner v-if="referenceAudioError" severity="error" icon="">
        {{ referenceAudioError }}
        <template #actions>
          <Button
            label="Retry reference audio"
            size="small"
            @click="loadReferenceAudio(selectedModelId)"
          />
        </template>
      </StateBanner>

      <PageSection
        variant="compact"
        title="Model context"
        description="The model, engine build, and API id this workspace runs against."
      >
        <div class="audio-model-bar">
          <Select
            v-model="selectedModelId"
            :options="modelOptions"
            optionLabel="label"
            optionValue="value"
            placeholder="Select an audio model"
            class="audio-model-dropdown"
            aria-label="Audio model"
          />
          <code v-if="inferenceModelId" class="param-key-hint" :title="'API model id'">{{
            inferenceModelId
          }}</code>
          <code
            v-if="contractEngineVersion"
            class="param-key-hint"
            data-testid="audio-engine-version"
            >audio.cpp {{ contractEngineVersion }}</code
          >
        </div>
        <PageToolbar v-if="needsReferenceHint" plain label="Reference audio">
          <template #summary>
            <span v-if="!referenceAudioOptions.length">No reference audio yet.</span>
            <span v-else>{{ referenceAudioOptions.length }} reference clips available.</span>
          </template>
          <template #actions>
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
          </template>
        </PageToolbar>
      </PageSection>

      <SectionTabs
        v-model="activeTab"
        :items="visibleTabs"
        id-prefix="audio"
        variant="pills"
        responsive-grid
        aria-label="Audio tasks"
      />

      <StateBanner
        v-if="selectedModel && !selectedModel.is_active && !selectedRuntimeState.actionable"
        severity="warning"
      >
        Start this model and wait for a verified running state before inference.
      </StateBanner>

      <!-- Speech -->
      <div
        v-if="activeTab === 'speech'"
        id="audio-panel-speech"
        role="tabpanel"
        aria-labelledby="audio-tab-speech"
        tabindex="0"
        class="config-tab-panel audio-task-layout"
      >
        <PageSection
          title="Speech"
          description="Synthesize text with the selected voice preset or a reference clip."
        >
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
              <span class="param-field__label" id="audio-speech-voice-label">Voice preset</span>
              <Select
                input-id="audio-speech-voice"
                aria-labelledby="audio-speech-voice-label"
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
              <span class="param-field__label" id="audio-speech-reference-label"
                >Reference audio</span
              >
              <Select
                input-id="audio-speech-reference"
                aria-labelledby="audio-speech-reference-label"
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
              <label class="param-field__label" for="audio-speech-language">Language</label>
              <InputText
                id="audio-speech-language"
                v-model="speechLanguage"
                class="param-input"
                placeholder="optional"
              />
            </div>
            <div v-if="speechStreamingAvailable" class="param-field">
              <label class="param-field__label" for="audio-stream-pcm">Play while generating</label>
              <ToggleSwitch
                v-model="speechStream"
                input-id="audio-stream-pcm"
                aria-label="Stream PCM"
              />
            </div>
          </div>
          <p v-if="speechStreamingAvailable && speechStream" class="page-form__hint">
            Play speech as it is generated. Enable streaming mode in the model configuration first.
          </p>
          <div class="page-form__actions">
            <Button
              :label="speechStream && speechStreamingAvailable ? 'Speak' : 'Generate'"
              icon="pi pi-volume-up"
              :loading="speechLoading"
              :disabled="!canRun || !speechText.trim()"
              @click="runSpeech"
            />
            <Button
              v-if="speechLoading"
              label="Cancel"
              icon="pi pi-stop"
              size="small"
              severity="secondary"
              outlined
              @click="cancelSpeech"
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
        </PageSection>
        <AudioResultPanel
          :error="speechError"
          :clips="resultAudioClips"
          :loading="speechLoading"
          @download="onDownloadClip"
        />
        <p v-if="speechStreamTiming" class="page-form__hint">{{ speechStreamTiming }}</p>
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
        <PageSection
          title="Transcribe"
          description="Upload or record audio. Non-WAV formats are converted to WAV at the file's sample rate and channel count. Several files are sent together as one batch of up to 32."
        >
          <div class="param-field section-params">
            <label class="param-field__label" for="audio-asr-file">Audio</label>
            <input
              ref="asrFileInput"
              id="audio-asr-file"
              type="file"
              multiple
              accept="audio/*,.wav,.ogg,.opus,.mp3,.webm,.m4a"
              class="sr-only"
              @change="onAsrFile"
            />
            <div class="audio-file-actions">
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
                :aria-pressed="recording ? 'true' : 'false'"
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
              <label class="param-field__label" for="audio-asr-language">Language</label>
              <InputText
                id="audio-asr-language"
                v-model="asrLanguage"
                class="param-input"
                placeholder="en"
              />
            </div>
            <div class="param-field">
              <label class="param-field__label" for="audio-asr-prompt">Prompt</label>
              <InputText
                id="audio-asr-prompt"
                v-model="asrPrompt"
                class="param-input"
                placeholder="optional context"
              />
            </div>
          </div>
          <div class="page-form__actions">
            <label class="audio-checkbox-field">
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
            <Button
              v-if="asrLoading"
              label="Cancel"
              icon="pi pi-stop"
              size="small"
              severity="secondary"
              outlined
              @click="cancelAsr"
            />
          </div>
        </PageSection>
        <AudioTranscriptPanel
          :error="asrError"
          :loading="asrLoading"
          :text="asrText"
          :metadata="asrMetadata"
          :batch-results="asrBatchResults"
        />
      </div>

      <div
        v-if="!['speech', 'transcribe'].includes(activeTab)"
        :id="`audio-panel-${activeTab}`"
        role="tabpanel"
        :aria-labelledby="`audio-tab-${activeTab}`"
        tabindex="0"
        class="config-tab-panel audio-task-layout"
      >
        <PageSection :title="taskLabel">
          <StateBanner v-if="!workspaceFields.length" severity="info">
            The active engine has not provided request fields for this model. Scan the model in
            Configure to discover its inputs.
            <template #actions>
              <Button label="Open configuration" size="small" @click="openConfig" />
            </template>
          </StateBanner>
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
            <small v-if="field.description" class="page-form__hint">{{ field.description }}</small>
          </div>
          <div class="page-form__actions">
            <Button
              :label="taskAction"
              icon="pi pi-play"
              :loading="taskLoading"
              :disabled="!canRunTask"
              @click="runSchemaTask"
            />
            <Button
              v-if="taskLoading"
              label="Cancel"
              icon="pi pi-stop"
              size="small"
              severity="secondary"
              outlined
              @click="cancelTask"
            />
          </div>
        </PageSection>
        <AudioResultPanel
          :error="taskError"
          :clips="resultAudioClips"
          :loading="taskLoading"
          @download="onDownloadClip"
        />
        <pre v-if="taskResult" class="audio-transcript">{{ taskResult }}</pre>
      </div>
    </template>
  </div>
</template>

<script setup>
import { modelRuntimeState, taskLabel as taskDisplayLabel } from '@/utils/modelPresentation'
import { isAudioPackage } from '@/utils/audioModel'
import { computed, onMounted, onUnmounted, ref, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { useConfirm } from 'primevue/useconfirm'
import { useToast } from 'primevue/usetoast'
import Button from 'primevue/button'
import Tag from 'primevue/tag'
import Select from 'primevue/select'
import ToggleSwitch from 'primevue/toggleswitch'
import InputText from 'primevue/inputtext'
import Textarea from 'primevue/textarea'
import EmptyState from '@/components/common/EmptyState.vue'
import LoadingState from '@/components/common/LoadingState.vue'
import PageHeader from '@/components/common/PageHeader.vue'
import PageSection from '@/components/common/PageSection.vue'
import PageToolbar from '@/components/common/PageToolbar.vue'
import SectionTabs from '@/components/common/SectionTabs.vue'
import StateBanner from '@/components/common/StateBanner.vue'
import AudioResultPanel from '@/components/audio/AudioResultPanel.vue'
import AudioParamField from '@/components/audio/AudioParamField.vue'
import AudioTranscriptPanel from '@/components/audio/AudioTranscriptPanel.vue'
import { requireSingleConfirmation } from '@/composables/singleConfirm'
import { withheldConfirmation } from '@/composables/actionConfirmation'
import { useModelStore } from '@/stores/models'
import { useEnginesStore } from '@/stores/engines'
import { DEFAULT_AUDIO_ENGINE_ID } from '@/engines/registry'
import {
  audioInferenceModelId,
  extractAudioClipsFromTaskResult,
  acceptedSpeechRateKeys,
  fetchAudioVoices,
  speechRateRequestFields,
  streamSpeech,
  synthesizeSpeech,
  taskKindFromConfig,
  transcribeAudio,
  transcribeAudioBatch,
  runAudioTask,
} from '@/composables/useAudioInferenceClient'
import {
  audioContract,
  audioContractError,
  audioEndpointIsGeneric,
  audioOperationForTask,
  audioTaskDefaultsKey,
  fetchAudioContract,
} from '@/composables/useAudioContract'

const route = useRoute()
const router = useRouter()
const toast = useToast()
const confirm = useConfirm()
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

const workspaceError = ref('')
const configError = ref('')
const contractError = ref('')
const referenceAudioError = ref('')
const engineDescriptorsLoaded = ref(false)
const audioEngineDescriptor = computed(
  () =>
    (enginesStore.engineDescriptors || []).find(
      (engine) => engine.id === DEFAULT_AUDIO_ENGINE_ID,
    ) || null,
)
const audioEngineDisabled = computed(() => audioEngineDescriptor.value?.enabled === false)
const audioEngineMissing = computed(
  () => engineDescriptorsLoaded.value && !audioEngineDescriptor.value,
)
const selectedRuntimeState = computed(() => modelRuntimeState(selectedModel.value))
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
const speechStream = ref(false)
const speechStreamTiming = ref('')
const SPEECH_STREAM_SAMPLE_RATE = 24000
let pcmPlayer = null
const referenceAudioItems = ref([])
const referenceAudioLoading = ref(false)
const referenceAudioUploading = ref(false)
const communityVoicesInstalling = ref(false)
const referenceUploadInput = ref(null)
const REFERENCE_AUDIO_MAX_BYTES = 60 * 1024 * 1024
const engineVoices = ref([])
let voicesController = null
let disposed = false
/** One controller per inference surface; Cancel aborts the current run. */
const runControllers = { speech: null, asr: null, task: null }

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
let recordStream = null
let recordChunks = []

const modelRegistry = ref(null)
const requestValues = ref({})
const workspaceFields = computed(() => modelRegistry.value?.workspace_request_fields || [])
const taskLabel = computed(() => visibleTabs.value[0]?.label || 'Run task')
const taskAction = computed(
  () => ({ music: 'Generate', design: 'Generate' })[activeTab.value] || taskLabel.value,
)
const canRunTask = computed(
  () =>
    canRun.value &&
    workspaceFields.value.length > 0 &&
    workspaceFields.value.every(
      (field) =>
        !field.required ||
        (requestValues.value[field.key] != null &&
          String(requestValues.value[field.key]).trim() !== ''),
    ),
)
const taskLoading = ref(false)
const taskError = ref('')
const taskResult = ref('')
/** Shared playable outputs for speech, music, VC, separation, design, etc. */
const resultAudioClips = ref([])

const audioModels = computed(() => modelStore.allQuantizations.filter((m) => isAudioPackage(m)))

const modelOptions = computed(() =>
  audioModels.value.map((m) => ({
    value: m.id,
    label: `${m.display_name || m.base_model_name || m.id} · ${modelRuntimeState(m).label}`,
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
  const name = proxyStatus.value?.backend === 'studio' ? 'Studio proxy' : 'Inference proxy'
  if (!proxyStatus.value || proxyStatus.value.healthy == null) return `${name} status unknown`
  return proxyHealthy.value ? `${name} ready` : `${name} offline`
})
const proxyStatusSeverity = computed(() => {
  if (!proxyStatus.value || proxyStatus.value.healthy == null) return 'secondary'
  return proxyHealthy.value ? 'success' : 'danger'
})

const canRun = computed(() =>
  Boolean(
    selectedRuntimeState.value.actionable &&
    inferenceModelId.value &&
    !contractError.value &&
    !configError.value,
  ),
)

const communityVoiceIds = computed(() =>
  (referenceAudioItems.value || [])
    .filter((item) => item.storage === 'community' && item.voice_id)
    .map((item) => item.voice_id),
)

const communityVoicesReady = computed(() => communityVoiceIds.value.length > 0)

const voicePresetOptions = computed(() => {
  const presets = selectedConfig.value?.voice_presets
  const names = presets && typeof presets === 'object' ? Object.keys(presets) : []
  return [...new Set([...engineVoices.value, ...names, ...communityVoiceIds.value])].map(
    (name) => ({
      label: name,
      value: name,
    }),
  )
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

const contractEngineVersion = computed(() =>
  String(audioContract.value.engine_version || '').trim(),
)

const driftWarnings = computed(() =>
  (audioContract.value.drift_warnings || []).map((warning) => String(warning)).filter(Boolean),
)

function versionParts(value) {
  return String(value || '')
    .trim()
    .replace(/^v/i, '')
    .split(/[.+-]/)
    .map((part) => {
      const parsed = Number.parseInt(part, 10)
      return Number.isFinite(parsed) ? parsed : 0
    })
}

/**
 * Compatibility workstream may add ``validated_through`` / ``engine_supported``
 * to the contract. When it does, a newer engine is a hint, never a block: the
 * payload still drives routing and the engine still runs.
 */
const engineNewerThanValidated = computed(() => {
  const contract = audioContract.value
  if (contract.engine_supported === false) return true
  if (!contract.validated_through) return false
  const engine = versionParts(contract.engine_version)
  const validated = versionParts(contract.validated_through)
  const length = Math.max(engine.length, validated.length)
  for (let index = 0; index < length; index += 1) {
    const enginePart = engine[index] || 0
    const validatedPart = validated[index] || 0
    if (enginePart !== validatedPart) return enginePart > validatedPart
  }
  return false
})

const contractValidationHint = computed(() => {
  if (!engineNewerThanValidated.value) return ''
  const contract = audioContract.value
  const engine = contract.engine_version || 'The active engine'
  const validated = contract.validated_through
    ? ` Studio validated the contract through ${contract.validated_through}.`
    : ''
  return `audio.cpp ${engine} is newer than the contract Studio validated.${validated} Some controls may be incomplete; the engine still runs.`
})

const configuredTask = computed(() =>
  String(selectedConfig.value?.task || '')
    .trim()
    .toLowerCase(),
)

/**
 * Tab presentation for a configured task. Dedicated contract operations map a
 * task to Speech/Transcribe/Analyze, but UI-only generic surfaces (Music,
 * Convert, Separate, Design) keep their schema-driven forms because the
 * contract has no operation that renders their request fields. Without a
 * contract this is the last-resort offline map.
 */
function workspaceKindForOperation(operationId, task) {
  const presentation = taskKindFromConfig({ task })
  if (!operationId) return presentation
  if (['music', 'convert', 'separate', 'analyze', 'design'].includes(presentation)) {
    return presentation
  }
  if (operationId === 'speech') return 'speech'
  if (['transcriptions', 'transcriptions_details', 'batch_transcriptions'].includes(operationId)) {
    return 'transcribe'
  }
  if (operationId === 'alignments') return 'analyze'
  if (operationId === 'tasks_run') {
    // A per-model policy can force a speech/ASR task through tasks/run.
    return ['speech', 'transcribe'].includes(presentation) ? 'task' : presentation
  }
  // A contract operation Studio has no dedicated form for: render the
  // schema-driven task form instead of guessing at a surface.
  return 'task'
}

const modelOperationId = computed(() => {
  const policy = modelRegistry.value?.request_policy || null
  if (policy?.operation_id) return policy.operation_id
  const endpoint = modelRegistry.value?.api_endpoint
  if (endpoint) {
    const byPath = (audioContract.value.operations || []).find(
      (operation) => operation.studio_path === endpoint,
    )
    if (byPath) return byPath.id
    if (audioEndpointIsGeneric(endpoint)) return 'tasks_run'
  }
  const task = configuredTask.value
  return task ? audioOperationForTask(task)?.id || '' : ''
})

function familyPresentationTask() {
  const family = String(selectedConfig.value?.family || '')
    .trim()
    .toLowerCase()
  if (!family) return ''
  const tasks = (audioContract.value.family_routing || {})[family]?.tasks || []
  return tasks.find((task) => taskKindFromConfig({ task }) !== 'task') || tasks[0] || ''
}

const modelKind = computed(() => {
  // Contract/model task routing chooses the tab; the configured task is the
  // input, and family routing fills in when a model has no task row yet.
  const task = configuredTask.value || familyPresentationTask()
  return workspaceKindForOperation(modelOperationId.value, task)
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
      detail:
        'Apply the proxy config and restart the model, then pick a voice name or a community clip.',
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
  referenceAudioError.value = ''
  if (modelId !== selectedModelId.value) return
  if (!modelId) return
  referenceAudioLoading.value = true
  try {
    const items = await modelStore.listReferenceAudio(modelId)
    if (selectedModelId.value === modelId)
      referenceAudioItems.value = Array.isArray(items) ? items : []
  } catch {
    if (selectedModelId.value === modelId)
      referenceAudioError.value =
        'Could not refresh reference audio. Previously loaded choices are retained.'
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
      `/api/models/param-registry?engine=${encodeURIComponent(DEFAULT_AUDIO_ENGINE_ID)}&model_id=${encodeURIComponent(modelId)}`,
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

async function loadWorkspaceData(force = false) {
  const descriptorRequest =
    typeof enginesStore.fetchEngineDescriptors === 'function'
      ? enginesStore.fetchEngineDescriptors()
      : Promise.resolve([])
  const results = await Promise.allSettled([
    modelStore.fetchModels(),
    enginesStore.fetchSystemStatus(),
    fetchAudioContract({ force }),
    descriptorRequest,
  ])
  engineDescriptorsLoaded.value = results[3].status === 'fulfilled'
  workspaceError.value =
    results[0].status === 'rejected'
      ? 'Could not refresh audio models. Check the connection and try again.'
      : ''
  contractError.value = audioContractError.value
    ? audioContractError.value
    : results[2].status === 'rejected'
      ? 'Audio capabilities could not be loaded. Retry before running inference.'
      : ''
}

async function refreshWorkspace() {
  refreshing.value = true
  try {
    await loadWorkspaceData(true)
    const modelId = selectedModelId.value
    if (modelId) {
      await Promise.all([
        modelStore
          .getModelConfig(modelId)
          .then((cfg) => {
            if (selectedModelId.value === modelId) {
              selectedConfig.value = cfg
              configError.value = ''
            }
          })
          .catch(() => {
            if (selectedModelId.value === modelId)
              configError.value =
                'Could not load the model configuration. Retry before running inference.'
          }),
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
  abortAllRuns()
  stopPcmPlayer()
  stopRecorder({ discard: true })
  referenceAudioItems.value = []
  referenceAudioError.value = ''
  selectedConfig.value = null
  configError.value = ''
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
    if (selectedModelId.value !== id) return
    configError.value = 'Could not load the model configuration. Retry before running inference.'
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
    await loadWorkspaceData()

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
  disposed = true
  abortAllRuns()
  clearResultAudio()
  stopRecorder({ discard: true })
  stopPcmPlayer()
})

function clearResultAudio() {
  for (const clip of resultAudioClips.value) {
    if (clip?.url) URL.revokeObjectURL(clip.url)
  }
  resultAudioClips.value = []
}

function publishAudioClips(clips) {
  if (disposed) return
  clearResultAudio()
  resultAudioClips.value = (clips || []).map((clip) => ({
    ...clip,
    url: URL.createObjectURL(clip.blob),
  }))
}

function stopPcmPlayer() {
  const player = pcmPlayer
  pcmPlayer = null
  if (player) player.stop()
}

function beginRun(kind) {
  runControllers[kind]?.abort()
  const controller = new AbortController()
  runControllers[kind] = controller
  return controller
}

function runWasCancelled(kind, controller) {
  return disposed || controller.signal.aborted || runControllers[kind] !== controller
}

function finishRun(kind, controller) {
  if (runControllers[kind] !== controller) return false
  runControllers[kind] = null
  return true
}

function abortAllRuns() {
  for (const kind of Object.keys(runControllers)) {
    const controller = runControllers[kind]
    if (!controller) continue
    controller.abort()
    runControllers[kind] = null
  }
  speechLoading.value = false
  asrLoading.value = false
  taskLoading.value = false
}

function cancelRun(kind) {
  const controller = runControllers[kind]
  if (!controller) return
  controller.abort()
  runControllers[kind] = null
  if (kind === 'speech') stopPcmPlayer()
  if (kind === 'speech') speechLoading.value = false
  if (kind === 'asr') asrLoading.value = false
  if (kind === 'task') taskLoading.value = false
}

function cancelSpeech() {
  cancelRun('speech')
}

function cancelAsr() {
  cancelRun('asr')
}

function cancelTask() {
  cancelRun('task')
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

function formatStartError(error) {
  const detail = error?.response?.data?.detail
  if (detail && typeof detail === 'object' && !Array.isArray(detail)) {
    return detail.message || detail.code || String(error)
  }
  return detail || error?.message || String(error)
}

async function startSelected(confirmation = null) {
  if (!selectedModelId.value) return
  starting.value = true
  try {
    await modelStore.startModel(selectedModelId.value, confirmation)
    await modelStore.fetchModels()
    toast.add({ severity: 'success', summary: 'Model starting', life: 2500 })
  } catch (error) {
    const withheld = !confirmation ? withheldConfirmation(error) : null
    if (withheld) {
      // An earlier start may still take effect; the backend withholds a
      // repeat until this exact operation and state are confirmed.
      await modelStore.fetchModels().catch(() => {})
      requireSingleConfirmation(confirm, {
        header: 'Confirm model start',
        message: `${withheld.message} Start this model again?`,
        icon: 'pi pi-exclamation-triangle',
        accept: () =>
          startSelected({
            confirm_operation_id: withheld.confirm_operation_id,
            confirm_state: withheld.confirm_state,
          }),
      })
      return
    }
    toast.add({
      severity: 'error',
      summary: 'Start failed',
      detail: formatStartError(error),
      life: 5000,
    })
  } finally {
    starting.value = false
  }
}

const speechStreamingAvailable = computed(() => selectedConfig.value?.mode === 'streaming')

/** Gapless queue for SSE PCM deltas: one AudioContext, scheduled by sample index. */
function createPcmPlayer(sampleRate) {
  const Ctx = window.AudioContext || window.webkitAudioContext
  const context = new Ctx()
  let nextTime = 0
  let sourceEnded = null
  let resolveSourceEnded = null
  return {
    push(pcmBytes) {
      const samples = new Int16Array(
        pcmBytes.buffer,
        pcmBytes.byteOffset,
        Math.floor(pcmBytes.byteLength / 2),
      )
      if (!samples.length) return
      // Declared-rate buffers are resampled by the context, so playback pitch
      // follows the model's stream rate even on 48 kHz output devices.
      const buffer = context.createBuffer(1, samples.length, sampleRate)
      const channel = buffer.getChannelData(0)
      for (let index = 0; index < samples.length; index += 1) {
        channel[index] = samples[index] / 32768
      }
      const source = context.createBufferSource()
      source.buffer = buffer
      source.connect(context.destination)
      sourceEnded = new Promise((resolve) => {
        resolveSourceEnded = resolve
      })
      source.onended = () => {
        resolveSourceEnded?.()
        resolveSourceEnded = null
      }
      const start = Math.max(context.currentTime + 0.02, nextTime)
      source.start(start)
      nextTime = start + buffer.duration
    },
    async finish() {
      if (sourceEnded) await sourceEnded.catch(() => {})
    },
    stop() {
      // A cancel can land while finish() is awaiting playback; resolve it so
      // the aborted run settles instead of leaving a pending promise.
      resolveSourceEnded?.()
      resolveSourceEnded = null
      context.close().catch(() => {})
    },
  }
}

async function runSpeech() {
  if (speechLoading.value) return
  speechError.value = ''
  taskError.value = ''
  speechStreamTiming.value = ''
  speechLoading.value = true
  setSpeechResult(null)
  stopPcmPlayer()
  const extras = {}
  const defaults = selectedConfig.value?.speech_defaults || {}
  Object.assign(extras, pickDefined(defaults, ['voice_ref', 'reference_text', 'instruct']))
  Object.assign(extras, speechRateRequestFields(defaults, acceptedSpeechRates.value))
  if (speechVoiceRef.value) {
    extras.voice_ref = speechVoiceRef.value
    const match = (referenceAudioItems.value || []).find(
      (item) => item.path === speechVoiceRef.value,
    )
    if (match?.reference_text && !extras.reference_text) {
      extras.reference_text = match.reference_text
    }
  }
  const controller = beginRun('speech')
  const task = configuredTask.value || 'tts'
  let player = null
  try {
    if (speechStreamingAvailable.value && speechStream.value) {
      player = createPcmPlayer(SPEECH_STREAM_SAMPLE_RATE)
      pcmPlayer = player
      const { blob, timing } = await streamSpeech({
        task,
        modelId: inferenceModelId.value,
        input: speechText.value,
        voice: speechVoice.value || undefined,
        sampleRate: SPEECH_STREAM_SAMPLE_RATE,
        extras,
        signal: controller.signal,
        onAudio: (bytes) => {
          if (!runWasCancelled('speech', controller)) player.push(bytes)
        },
      })
      if (runWasCancelled('speech', controller)) return
      await player.finish()
      if (runWasCancelled('speech', controller)) return
      if (timing?.ttft_ms != null) {
        speechStreamTiming.value = `First audio in ${Math.round(timing.ttft_ms)} ms`
      }
      setSpeechResult(blob, 'speech.wav')
    } else {
      const { blob } = await synthesizeSpeech({
        task,
        modelId: inferenceModelId.value,
        input: speechText.value,
        voice: speechVoice.value || undefined,
        extras,
        signal: controller.signal,
      })
      if (runWasCancelled('speech', controller)) return
      setSpeechResult(blob, 'speech.wav')
    }
  } catch (error) {
    if (runWasCancelled('speech', controller) || error?.name === 'AbortError') return
    speechError.value = error?.message || String(error)
  } finally {
    if (pcmPlayer === player) stopPcmPlayer()
    if (finishRun('speech', controller)) speechLoading.value = false
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
    if (disposed) {
      stream.getTracks().forEach((track) => track.stop())
      return
    }
    recordChunks = []
    recordStream = stream
    const recorder = new MediaRecorder(stream)
    recorder.__studioDiscard = false
    mediaRecorder = recorder
    recorder.ondataavailable = (ev) => {
      if (ev.data?.size) recordChunks.push(ev.data)
    }
    recorder.onstop = () => {
      recordStream?.getTracks().forEach((track) => track.stop())
      recordStream = null
      if (mediaRecorder === recorder) mediaRecorder = null
      if (recorder.__studioDiscard || disposed) return
      const blob = new Blob(recordChunks, { type: recorder.mimeType || 'audio/webm' })
      asrFiles.value = [new File([blob], 'recording.webm', { type: blob.type })]
      asrBatchResults.value = []
      recording.value = false
    }
    recorder.start()
    recording.value = true
  } catch (error) {
    if (disposed) return
    toast.add({
      severity: 'error',
      summary: 'Microphone unavailable',
      detail: error?.message || String(error),
      life: 4000,
    })
  }
}

function stopRecorder({ discard = false } = {}) {
  const recorder = mediaRecorder
  recording.value = false
  if (recorder) recorder.__studioDiscard = discard
  if (recorder && recorder.state !== 'inactive') {
    recorder.stop()
  } else if (!recorder) {
    recordStream?.getTracks().forEach((track) => track.stop())
    recordStream = null
  }
}

async function runAsr() {
  if (asrLoading.value) return
  asrError.value = ''
  asrText.value = ''
  asrMetadata.value = ''
  asrBatchResults.value = []
  asrLoading.value = true
  const controller = beginRun('asr')
  const task = configuredTask.value || 'asr'
  try {
    if (asrFiles.value.length > 1) {
      const result = await transcribeAudioBatch({
        task: 'batch_transcribe',
        modelId: inferenceModelId.value,
        files: asrFiles.value,
        language: asrLanguage.value || undefined,
        prompt: asrPrompt.value || undefined,
        signal: controller.signal,
      })
      if (runWasCancelled('asr', controller)) return
      asrBatchResults.value = Array.isArray(result?.results) ? result.results : []
      if (!asrBatchResults.value.length && result?.text) {
        asrText.value = result.text
      }
      return
    }
    const file = asrFiles.value[0]
    const result = await transcribeAudio({
      task,
      modelId: inferenceModelId.value,
      file,
      filename: file?.name,
      language: asrLanguage.value || undefined,
      prompt: asrPrompt.value || undefined,
      details: asrDetails.value,
      signal: controller.signal,
    })
    if (runWasCancelled('asr', controller)) return
    asrText.value = result?.text || JSON.stringify(result, null, 2)
    if (asrDetails.value && result && typeof result === 'object') {
      const { text: _text, ...metadata } = result
      asrMetadata.value = Object.keys(metadata).length ? JSON.stringify(metadata, null, 2) : ''
    }
  } catch (error) {
    if (runWasCancelled('asr', controller) || error?.name === 'AbortError') return
    asrError.value = error?.message || String(error)
  } finally {
    if (finishRun('asr', controller)) asrLoading.value = false
  }
}

async function runTask(input, { defaultFilename = 'audio.wav' } = {}) {
  if (taskLoading.value) return
  taskError.value = ''
  setTaskResult(null)
  taskLoading.value = true
  const controller = beginRun('task')
  try {
    const result = await runAudioTask({
      modelId: inferenceModelId.value,
      task: configuredTask.value,
      input,
      signal: controller.signal,
    })
    if (runWasCancelled('task', controller)) return
    setTaskResult(result, { defaultFilename })
  } catch (error) {
    if (runWasCancelled('task', controller) || error?.name === 'AbortError') return
    taskError.value = error?.message || String(error)
  } finally {
    if (finishRun('task', controller)) taskLoading.value = false
  }
}

function initializeRequestValues(config) {
  const defaultsKey =
    modelRegistry.value?.request_defaults_key || audioTaskDefaultsKey(selectedConfig.value?.task)
  const defaults = config?.[defaultsKey] || {}
  requestValues.value = Object.fromEntries(
    workspaceFields.value.map((field) => [
      field.key,
      (field.nested ? defaults.options?.[field.key] : defaults[field.key]) ?? field.default ?? null,
    ]),
  )
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
  return runTask(input)
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

/* The two-column task layout is an audio-specific exception: the engine form
   and its result panel read as a pair on desktop and stack on narrow screens. */
@media (min-width: 900px) {
  .audio-task-layout:has(> :nth-child(2)) {
    display: grid;
    grid-template-columns: minmax(0, 1.15fr) minmax(0, 0.85fr);
    align-items: start;
    gap: 0.75rem;
  }
}

.audio-file-actions {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 0.5rem;
}

.audio-checkbox-field {
  display: inline-flex;
  align-items: center;
  gap: 0.5rem;
  min-height: 2.125rem;
  font-size: var(--text-sm);
}

.asr-file-list {
  display: grid;
  gap: 0.25rem;
  margin: 0.4rem 0 0;
  padding: 0;
  list-style: none;
}

.asr-file-list li {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 0.5rem;
  min-height: 2rem;
  font-size: var(--text-sm);
}

.asr-file-list button {
  min-height: 2rem;
  border: 0;
  background: transparent;
  color: var(--text-secondary);
  font: inherit;
  cursor: pointer;
}

.audio-transcript {
  max-height: 20rem;
  margin: 0;
  padding: 0.75rem;
  overflow: auto;
  border: 1px solid var(--border-primary);
  border-radius: var(--radius-md);
  background: color-mix(in srgb, var(--bg-surface) 65%, transparent);
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
  font-size: var(--text-sm);
  line-height: 1.45;
  white-space: pre-wrap;
  word-break: break-word;
}

.contract-drift details {
  min-width: 0;
}

.contract-drift summary {
  cursor: pointer;
}

.contract-drift ul {
  margin: 0.35rem 0 0;
  padding-left: 1.1rem;
}

:deep(.page-empty__actions) {
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
}

@media (max-width: 768px) {
  .audio-model-dropdown {
    flex: 1 1 100%;
    min-width: 0;
    max-width: none;
  }
}

@media (max-width: 640px), (pointer: coarse) {
  .audio-checkbox-field {
    min-height: 2.75rem;
  }

  .asr-file-list button {
    min-height: 2.75rem;
  }
}
</style>
