import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { enableAutoUnmount, flushPromises, mount } from '@vue/test-utils'

import { audioContractFixture, audioParamRegistryFixture } from '@/test-fixtures/audioContract'
import { DEFAULT_AUDIO_ENGINE_ID } from '@/engines/registry'

const toastAdd = vi.fn()
const push = vi.fn()
const replace = vi.fn()
const fetchModels = vi.fn()
const fetchSystemStatus = vi.fn()
const fetchEngineDescriptors = vi.fn()
const getModelConfig = vi.fn()
const listReferenceAudio = vi.fn()
const startModel = vi.fn()
const runAudioTask = vi.fn()
const synthesizeSpeech = vi.fn()
const streamSpeech = vi.fn()
const transcribeAudio = vi.fn()
const transcribeAudioBatch = vi.fn()
const fetchAudioVoices = vi.fn()
const routeQuery = { model: undefined, tab: undefined }

vi.mock('vue-router', () => ({
  useRoute: () => ({ query: routeQuery }),
  useRouter: () => ({ push, replace }),
}))

vi.mock('primevue/usetoast', () => ({
  useToast: () => ({ add: toastAdd }),
}))

const confirmRequire = vi.fn()
vi.mock('primevue/useconfirm', () => ({
  useConfirm: () => ({ require: (...args) => confirmRequire(...args) }),
}))

const AUDIO_ENGINE = { id: DEFAULT_AUDIO_ENGINE_ID, label: 'audio.cpp', enabled: true }
let engineDescriptors = [AUDIO_ENGINE]
let systemStatus = { proxy_status: { healthy: true, port: 2000 } }
let audioQuantizations = []

vi.mock('@/stores/models', () => ({
  useModelStore: () => ({
    get allQuantizations() {
      return audioQuantizations
    },
    fetchModels: (...args) => fetchModels(...args),
    getModelConfig: (...args) => getModelConfig(...args),
    listReferenceAudio: (...args) => listReferenceAudio(...args),
    startModel: (...args) => startModel(...args),
  }),
}))

vi.mock('@/stores/engines', () => ({
  useEnginesStore: () => ({
    get engineDescriptors() {
      return engineDescriptors
    },
    get systemStatus() {
      return systemStatus
    },
    fetchEngineDescriptors: (...args) => fetchEngineDescriptors(...args),
    fetchSystemStatus: (...args) => fetchSystemStatus(...args),
  }),
}))

vi.mock('@/composables/useAudioInferenceClient', async (importOriginal) => {
  const actual = await importOriginal()
  return {
    ...actual,
    runAudioTask: (...args) => runAudioTask(...args),
    synthesizeSpeech: (...args) => synthesizeSpeech(...args),
    streamSpeech: (...args) => streamSpeech(...args),
    transcribeAudio: (...args) => transcribeAudio(...args),
    transcribeAudioBatch: (...args) => transcribeAudioBatch(...args),
    fetchAudioVoices: (...args) => fetchAudioVoices(...args),
  }
})

import AudioWorkspace from './AudioWorkspace.vue'
import {
  AUDIO_CONTRACT_PATH,
  AUDIO_CONTRACT_SCHEMA,
  resetAudioContract,
} from '@/composables/useAudioContract'

enableAutoUnmount(afterEach)

function makeAudioModels() {
  return [
    {
      id: 'audio/asr',
      display_name: 'ASR Demo',
      format: DEFAULT_AUDIO_ENGINE_ID,
      is_active: true,
      config: { engine: DEFAULT_AUDIO_ENGINE_ID },
    },
    {
      id: 'audio/ace',
      display_name: 'ACE-Step',
      format: DEFAULT_AUDIO_ENGINE_ID,
      is_active: true,
      config: { engine: DEFAULT_AUDIO_ENGINE_ID },
    },
    {
      id: 'audio/heartmula',
      display_name: 'HeartMuLa',
      format: DEFAULT_AUDIO_ENGINE_ID,
      is_active: true,
      config: { engine: DEFAULT_AUDIO_ENGINE_ID },
    },
    {
      id: 'audio/tts',
      display_name: 'TTS Demo',
      format: DEFAULT_AUDIO_ENGINE_ID,
      is_active: true,
      config: { engine: DEFAULT_AUDIO_ENGINE_ID },
    },
    {
      id: 'audio/sep',
      display_name: 'Demucs',
      format: DEFAULT_AUDIO_ENGINE_ID,
      is_active: true,
      config: { engine: DEFAULT_AUDIO_ENGINE_ID },
    },
  ]
}

const MUSIC_FIELDS = [
  { key: 'text', label: 'Prompt', type: 'textarea', nested: false, required: true },
  { key: 'lyrics', label: 'Lyrics', type: 'textarea', nested: false },
  { key: 'tags', label: 'Style tags', type: 'string', nested: true },
  { key: 'task_route', label: 'Route', type: 'string', nested: true },
]

let contractPayload
let contractFailure
let paramRegistryFor
let createObjectURL
let revokeObjectURL
let urlCounter
let microphoneStub

function jsonResponse(data) {
  return { ok: true, status: 200, json: async () => data }
}

function errorResponse(status, detail) {
  return { ok: false, status, json: async () => ({ detail }) }
}

function installFetchStub() {
  const fetchMock = vi.fn((input) => {
    const url = typeof input === 'string' ? input : input?.url || ''
    if (url === AUDIO_CONTRACT_PATH) {
      return Promise.resolve(contractFailure || jsonResponse(contractPayload))
    }
    if (url.startsWith('/api/models/param-registry')) {
      const match = url.match(/[?&]model_id=([^&]+)/)
      const modelId = match ? decodeURIComponent(match[1]) : ''
      return Promise.resolve(jsonResponse(paramRegistryFor(modelId)))
    }
    throw new Error(`Unexpected fetch in AudioWorkspace test: ${url}`)
  })
  vi.stubGlobal('fetch', fetchMock)
  return fetchMock
}

function stubMicrophone() {
  const track = { stop: vi.fn() }
  const stream = { getTracks: () => [track] }
  const getUserMedia = vi.fn().mockResolvedValue(stream)
  const descriptor = Object.getOwnPropertyDescriptor(navigator, 'mediaDevices')
  Object.defineProperty(navigator, 'mediaDevices', {
    configurable: true,
    value: { getUserMedia },
  })
  class FakeMediaRecorder {
    constructor(mediaStream) {
      this.stream = mediaStream
      this.mimeType = 'audio/webm'
      this.state = 'inactive'
      FakeMediaRecorder.instances.push(this)
    }
    start() {
      this.state = 'recording'
    }
    stop() {
      this.state = 'inactive'
      this.onstop?.()
    }
  }
  FakeMediaRecorder.instances = []
  vi.stubGlobal('MediaRecorder', FakeMediaRecorder)
  microphoneStub = {
    getUserMedia,
    track,
    restore: () => {
      if (descriptor) Object.defineProperty(navigator, 'mediaDevices', descriptor)
      else delete navigator.mediaDevices
    },
  }
  return microphoneStub
}

const mountStubs = {
  Button: {
    props: ['label', 'icon', 'loading', 'disabled', 'text', 'severity', 'outlined', 'size'],
    emits: ['click'],
    template:
      '<button type="button" :data-label="label" :disabled="disabled" :aria-pressed="$attrs[`aria-pressed`]" @click="$emit(`click`)">{{ label }}</button>',
  },
  Tag: true,
  Select: {
    props: [
      'modelValue',
      'options',
      'optionLabel',
      'optionValue',
      'editable',
      'showClear',
      'placeholder',
    ],
    emits: ['update:modelValue'],
    template: `
      <select
        :value="modelValue ?? ''"
        @change="$emit('update:modelValue', $event.target.value)"
      >
        <option v-for="opt in (options || [])" :key="opt.value ?? opt" :value="opt.value ?? opt">
          {{ opt.label ?? opt }}
        </option>
      </select>
    `,
  },
  InputNumber: {
    props: ['modelValue', 'placeholder', 'inputId'],
    emits: ['update:modelValue'],
    template:
      '<input :id="inputId" class="duration-input" type="number" :value="modelValue ?? ``" :placeholder="placeholder" @input="$emit(`update:modelValue`, $event.target.value === `` ? null : Number($event.target.value))" />',
  },
  InputText: {
    props: ['modelValue', 'placeholder', 'class', 'id'],
    emits: ['update:modelValue'],
    template:
      '<input :id="id" class="param-input" :value="modelValue ?? ``" :placeholder="placeholder" @input="$emit(`update:modelValue`, $event.target.value)" />',
  },
  Textarea: {
    props: ['modelValue', 'id'],
    emits: ['update:modelValue'],
    template:
      '<textarea :id="id" :value="modelValue ?? ``" @input="$emit(`update:modelValue`, $event.target.value)" />',
  },
  Message: {
    props: ['severity', 'closable'],
    template: '<div class="message"><slot /></div>',
  },
  LoadingState: true,
  PageHeader: {
    props: ['title', 'description'],
    template:
      '<header class="page-header"><h1 class="page-title">{{ title }}</h1><p v-if="description" class="page-header__description">{{ description }}</p><slot name="meta" /><slot name="actions" /></header>',
  },
  EmptyState: {
    props: ['title', 'description', 'icon'],
    template: '<div class="empty-state"><h2>{{ title }}</h2><p>{{ description }}</p><slot /></div>',
  },
  RouterLink: true,
}

function mountWorkspace() {
  return mount(AudioWorkspace, {
    attachTo: document.body,
    global: {
      directives: { tooltip: () => {} },
      stubs: mountStubs,
    },
  })
}

function deferred() {
  let resolve
  let reject
  const promise = new Promise((res, rej) => {
    resolve = res
    reject = rej
  })
  return { promise, resolve, reject }
}

function wavBase64(bytes = [0x52, 0x49, 0x46, 0x46]) {
  let binary = ''
  for (const byte of bytes) binary += String.fromCharCode(byte)
  return btoa(binary)
}

function abortableDeferred() {
  const gate = deferred()
  let signal = null
  return {
    gate,
    run: ({ signal: nextSignal }) => {
      signal = nextSignal
      nextSignal.addEventListener('abort', () => {
        const error = new Error('The operation was aborted.')
        error.name = 'AbortError'
        gate.reject(error)
      })
      return gate.promise
    },
    get signal() {
      return signal
    },
  }
}

async function selectAsrFile(wrapper, file) {
  const input = wrapper.get('#audio-asr-file')
  Object.defineProperty(input.element, 'files', { configurable: true, value: [file] })
  await input.trigger('change')
  await flushPromises()
}

describe('AudioWorkspace', () => {
  beforeEach(() => {
    routeQuery.model = undefined
    routeQuery.tab = undefined
    engineDescriptors = [AUDIO_ENGINE]
    systemStatus = { proxy_status: { healthy: true, port: 2000 } }
    audioQuantizations = makeAudioModels()
    contractPayload = audioContractFixture()
    contractFailure = null
    paramRegistryFor = () => audioParamRegistryFixture({ workspace_request_fields: MUSIC_FIELDS })
    urlCounter = 0
    createObjectURL = vi.fn(() => `blob:test-${(urlCounter += 1)}`)
    revokeObjectURL = vi.fn()

    resetAudioContract()
    fetchModels.mockReset().mockResolvedValue(undefined)
    fetchSystemStatus.mockReset().mockResolvedValue(undefined)
    fetchEngineDescriptors.mockReset().mockImplementation(async () => engineDescriptors)
    getModelConfig.mockReset().mockResolvedValue({ task: 'tts', model_alias: 'demo' })
    listReferenceAudio.mockReset().mockResolvedValue([])
    startModel.mockReset()
    runAudioTask.mockReset()
    synthesizeSpeech.mockReset()
    streamSpeech.mockReset()
    transcribeAudio.mockReset()
    transcribeAudioBatch.mockReset()
    fetchAudioVoices.mockReset().mockResolvedValue([])
    microphoneStub = null
    installFetchStub()
    vi.stubGlobal('URL', { createObjectURL, revokeObjectURL })
  })

  afterEach(() => {
    microphoneStub?.restore()
    delete window.AudioContext
    resetAudioContract()
    vi.unstubAllGlobals()
  })

  it('retains a visible retry action when the catalog or configuration cannot refresh', async () => {
    fetchModels.mockRejectedValue(new Error('offline'))
    getModelConfig.mockRejectedValue(new Error('configuration offline'))
    const wrapper = mountWorkspace()
    await flushPromises()

    expect(wrapper.text()).toContain('Could not refresh audio models')
    expect(wrapper.text()).toContain('Could not load the model configuration')
    expect(wrapper.find('button[data-label="Retry configuration"]').exists()).toBe(true)
  })

  it('renders Transcribe for an ASR model', async () => {
    routeQuery.model = 'audio/asr'
    routeQuery.tab = 'transcribe'
    getModelConfig.mockResolvedValue({
      family: 'qwen3-asr',
      task: 'asr',
      model_alias: 'asr-demo',
      transcription_defaults: { language: 'en' },
    })

    const wrapper = mountWorkspace()
    await flushPromises()

    expect(wrapper.text()).toContain('Audio')
    expect(fetchModels).toHaveBeenCalled()
    expect(fetchSystemStatus).toHaveBeenCalled()
    expect(fetchEngineDescriptors).toHaveBeenCalled()
    expect(getModelConfig).toHaveBeenCalledWith('audio/asr')
    const tabLabels = wrapper.findAll('.section-tabs__tab').map((node) => node.text())
    expect(tabLabels.some((label) => label.includes('Transcribe'))).toBe(true)
    expect(tabLabels.some((label) => label.includes('Speech'))).toBe(false)
    expect(wrapper.text()).toContain('Non-WAV formats are converted to WAV')
  })

  it('derives the Transcribe tab from contract task_routing for a details task', async () => {
    routeQuery.model = 'audio/asr'
    getModelConfig.mockResolvedValue({
      family: 'parakeet',
      task: 'transcribe_details',
      model_alias: 'parakeet-demo',
    })

    const wrapper = mountWorkspace()
    await flushPromises()

    // The local presentation table has no transcribe_details row; only the
    // contract's task_routing -> transcriptions_details operation maps it.
    const tabLabels = wrapper.findAll('.section-tabs__tab').map((node) => node.text())
    expect(tabLabels).toEqual(['Transcribe'])
    expect(wrapper.find('#audio-speech-text').exists()).toBe(false)
  })

  it('derives the tab from contract family routing when the config has no task', async () => {
    routeQuery.model = 'audio/asr'
    getModelConfig.mockResolvedValue({
      family: 'qwen3_asr',
      model_alias: 'asr-demo',
    })

    const wrapper = mountWorkspace()
    await flushPromises()

    const tabLabels = wrapper.findAll('.section-tabs__tab').map((node) => node.text())
    expect(tabLabels).toEqual(['Transcribe'])
  })

  it('keeps the schema-driven Design form for a vdes task even though it uses speech', async () => {
    routeQuery.model = 'audio/tts'
    getModelConfig.mockResolvedValue({
      family: 'omnivoice',
      task: 'vdes',
      model_alias: 'design-demo',
    })
    paramRegistryFor = () =>
      audioParamRegistryFixture({
        workspace_request_fields: [
          { key: 'text', label: 'Design prompt', type: 'string', nested: false, required: true },
        ],
      })

    const wrapper = mountWorkspace()
    await flushPromises()

    const tabLabels = wrapper.findAll('.section-tabs__tab').map((node) => node.text())
    expect(tabLabels).toEqual(['Design'])
    expect(wrapper.find('#audio-task-text').exists()).toBe(true)
    expect(wrapper.find('#audio-speech-text').exists()).toBe(false)
  })

  it('shows playable music result with download after generate', async () => {
    routeQuery.model = 'audio/ace'
    routeQuery.tab = 'music'
    getModelConfig.mockResolvedValue({
      family: 'ace_step',
      task: 'gen',
      model_alias: 'ace-demo',
      task_defaults: { options: { task_route: 'text2music' } },
    })
    runAudioTask.mockResolvedValue({
      audio: wavBase64(),
      sample_rate: 48000,
      channels: 2,
      timing: { wall_ms: 12, rtf: 0.5 },
    })

    const wrapper = mountWorkspace()
    await flushPromises()

    expect(wrapper.text()).toContain('Music')
    expect(wrapper.text()).toContain('Style tags')
    const prompt = wrapper.find('textarea')
    await prompt.setValue('lofi beat')
    await wrapper.find('button[data-label="Generate"]').trigger('click')
    await flushPromises()

    expect(runAudioTask).toHaveBeenCalledWith({
      modelId: 'ace-demo',
      task: 'gen',
      input: expect.objectContaining({
        text: 'lofi beat',
        options: expect.objectContaining({ task_route: 'text2music' }),
      }),
      signal: expect.any(AbortSignal),
    })
    expect(runAudioTask.mock.calls[0][0]).not.toHaveProperty('proxyPort')
    expect(wrapper.find('audio.audio-player').exists()).toBe(true)
    expect(wrapper.find('audio.audio-player').attributes('src')).toBe('blob:test-1')
    expect(wrapper.find('button[data-label="Download WAV"]').exists()).toBe(true)
    expect(wrapper.text()).toContain('wall_ms')
    expect(wrapper.text()).not.toContain(wavBase64())
  })

  it('sends HeartMuLa style tags from the music form', async () => {
    routeQuery.model = 'audio/heartmula'
    routeQuery.tab = 'music'
    getModelConfig.mockResolvedValue({
      family: 'heartmula',
      task: 'gen',
      model_alias: 'heartmula-demo',
      task_defaults: { options: { tags: 'piano,happy' } },
    })
    runAudioTask.mockResolvedValue({ audio: wavBase64(), sample_rate: 48000, channels: 2 })

    const wrapper = mountWorkspace()
    await flushPromises()

    const textareas = wrapper.findAll('textarea')
    await textareas[0].setValue('summer night')
    await textareas[1].setValue('[verse]\nhello')
    const tagsInput = wrapper.find('input.param-input')
    expect(tagsInput.element.value).toBe('piano,happy')
    await tagsInput.setValue('pop, bright, drums')
    await wrapper.find('button[data-label="Generate"]').trigger('click')
    await flushPromises()

    expect(runAudioTask).toHaveBeenCalledWith(
      expect.objectContaining({
        modelId: 'heartmula-demo',
        task: 'gen',
        input: expect.objectContaining({
          text: 'summer night',
          lyrics: '[verse]\nhello',
          options: expect.objectContaining({ tags: 'pop, bright, drums' }),
        }),
      }),
    )
  })

  it('shows playable speech result after synthesize', async () => {
    routeQuery.model = 'audio/tts'
    routeQuery.tab = 'speech'
    getModelConfig.mockResolvedValue({
      family: 'omnivoice',
      task: 'tts',
      model_alias: 'tts-demo',
      speech_defaults: {},
    })
    synthesizeSpeech.mockResolvedValue({
      blob: new Blob([new Uint8Array([1, 2, 3])], { type: 'audio/wav' }),
      contentType: 'audio/wav',
    })

    const wrapper = mountWorkspace()
    await flushPromises()

    await wrapper.find('#audio-speech-text').setValue('Hello from Studio')
    await wrapper.find('button[data-label="Generate"]').trigger('click')
    await flushPromises()

    expect(synthesizeSpeech).toHaveBeenCalledWith(
      expect.objectContaining({
        task: 'tts',
        modelId: 'tts-demo',
        input: 'Hello from Studio',
        signal: expect.any(AbortSignal),
      }),
    )
    expect(wrapper.find('audio.audio-player').exists()).toBe(true)
    expect(wrapper.find('button[data-label="Download WAV"]').exists()).toBe(true)
  })

  it('sends scanned speech speed on the speech body', async () => {
    routeQuery.model = 'audio/tts'
    routeQuery.tab = 'speech'
    paramRegistryFor = () =>
      audioParamRegistryFixture({
        sections: [
          {
            params: [
              { scope: 'request_option', key: 'speed' },
              { scope: 'session_option', key: 'speaking_rate' },
            ],
          },
        ],
        workspace_request_fields: [],
      })
    getModelConfig.mockResolvedValue({
      family: 'omnivoice',
      task: 'tts',
      model_alias: 'tts-demo',
      speech_defaults: { options: { speed: '1.1', speaking_rate: '0.9' } },
    })
    synthesizeSpeech.mockResolvedValue({
      blob: new Blob([new Uint8Array([1])], { type: 'audio/wav' }),
      contentType: 'audio/wav',
    })

    const wrapper = mountWorkspace()
    await flushPromises()
    await wrapper.find('#audio-speech-text').setValue('Hello')
    await wrapper.find('button[data-label="Generate"]').trigger('click')
    await flushPromises()

    const extras = synthesizeSpeech.mock.calls[0][0].extras
    expect(extras).toEqual({ speed: 1.1 })
  })

  it('streams speech through native PCM deltas when the model runs in streaming mode', async () => {
    class FakeAudioContext {
      constructor() {
        FakeAudioContext.instances.push(this)
        this.currentTime = 0
      }
      createBuffer(_channels, length, sampleRate) {
        return {
          length,
          sampleRate,
          duration: length / sampleRate,
          getChannelData: () => new Float32Array(length),
        }
      }
      createBufferSource() {
        const source = {
          connect: vi.fn(),
          start: vi.fn(() => {
            if (source.onended) source.onended()
          }),
          onended: null,
        }
        return source
      }
      close() {
        return Promise.resolve()
      }
    }
    FakeAudioContext.instances = []
    window.AudioContext = FakeAudioContext

    routeQuery.model = 'audio/tts'
    routeQuery.tab = 'speech'
    getModelConfig.mockResolvedValue({
      family: 'pocket_tts',
      task: 'tts',
      mode: 'streaming',
      model_alias: 'pocket-stream',
      speech_defaults: {},
    })
    streamSpeech.mockImplementation(async ({ onAudio }) => {
      onAudio(new Uint8Array([1, 0, 2, 0]), 0)
      return {
        blob: new Blob([new Uint8Array([1, 2])], { type: 'audio/wav' }),
        timing: { ttft_ms: 137 },
        text: '',
        fellBack: false,
      }
    })

    const wrapper = mountWorkspace()
    await flushPromises()

    const toggle = wrapper.findComponent({ name: 'ToggleSwitch' })
    expect(toggle.exists()).toBe(true)
    toggle.vm.$emit('update:modelValue', true)
    await flushPromises()

    await wrapper.find('#audio-speech-text').setValue('Streaming hello')
    await wrapper.find('button[data-label="Speak"]').trigger('click')
    await flushPromises()

    expect(streamSpeech).toHaveBeenCalledWith(
      expect.objectContaining({
        task: 'tts',
        modelId: 'pocket-stream',
        input: 'Streaming hello',
        sampleRate: 24000,
        signal: expect.any(AbortSignal),
      }),
    )
    expect(synthesizeSpeech).not.toHaveBeenCalled()
    expect(wrapper.text()).toContain('First audio in 137 ms')
    expect(wrapper.find('audio.audio-player').exists()).toBe(true)
  })

  it('does not add family-specific duration fields on the speech form', async () => {
    routeQuery.model = 'audio/tts'
    routeQuery.tab = 'speech'
    getModelConfig.mockResolvedValue({
      family: 'echo_tts',
      task: 'clon',
      model_alias: 'echo-demo',
      session_options: { 'echo_tts.reference_duration_sec': '15' },
      speech_defaults: {},
    })

    const wrapper = mountWorkspace()
    await flushPromises()

    expect(wrapper.find('#audio-speech-ref-duration').exists()).toBe(false)
    expect(wrapper.text()).not.toContain('Max generation duration (seconds)')
  })

  it('renders separation stems as multiple players', async () => {
    routeQuery.model = 'audio/sep'
    routeQuery.tab = 'separate'
    paramRegistryFor = () =>
      audioParamRegistryFixture({
        workspace_request_fields: [
          { key: 'audio', label: 'Audio', type: 'string', nested: false, required: true },
        ],
      })
    getModelConfig.mockResolvedValue({
      family: 'htdemucs',
      task: 'sep',
      model_alias: 'sep-demo',
    })
    listReferenceAudio.mockResolvedValue([{ path: '/data/mix.wav', display_path: 'mix.wav' }])
    const b64 = wavBase64()
    runAudioTask.mockResolvedValue({
      named_audio_outputs: [
        { id: 'drums', audio: b64, sample_rate: 44100, channels: 2 },
        { id: 'vocals', audio: b64, sample_rate: 44100, channels: 2 },
      ],
      timing: { wall_ms: 9 },
    })

    const wrapper = mountWorkspace()
    await flushPromises()

    expect(wrapper.text()).toContain('Separate')
    await wrapper.find('#audio-task-audio').setValue('/data/mix.wav')
    await wrapper.find('button[data-label="Separate"]').trigger('click')
    await flushPromises()

    expect(runAudioTask).toHaveBeenCalledWith({
      modelId: 'sep-demo',
      task: 'sep',
      input: { audio: '/data/mix.wav' },
      signal: expect.any(AbortSignal),
    })
    expect(wrapper.findAll('audio.audio-player')).toHaveLength(2)
    expect(wrapper.text()).toContain('drums')
    expect(wrapper.text()).toContain('vocals')
  })

  it('disables generic inference when the engine publishes no request schema', async () => {
    routeQuery.model = 'audio/ace'
    paramRegistryFor = () => audioParamRegistryFixture({ workspace_request_fields: [] })
    getModelConfig.mockResolvedValue({ task: 'music', model_alias: 'future' })
    const wrapper = mountWorkspace()
    await flushPromises()

    expect(wrapper.text()).toContain('has not provided request fields')
    expect(wrapper.find('button[data-label="Generate"]').element.disabled).toBe(true)
    expect(runAudioTask).not.toHaveBeenCalled()
  })

  it('uses a new engine schema without adding music or conversion fields', async () => {
    routeQuery.model = 'audio/ace'
    paramRegistryFor = () =>
      audioParamRegistryFixture({
        workspace_request_fields: [
          {
            key: 'source_audio',
            label: 'Source clip',
            type: 'string',
            required: true,
            nested: false,
          },
          { key: 'variation', label: 'Variation', type: 'float', default: 0, nested: true },
          { key: 'sample', label: 'Sample', type: 'bool', default: false, nested: true },
        ],
      })
    getModelConfig.mockResolvedValue({
      task: 'future_task',
      model_alias: 'future',
      task_defaults: { lyrics: 'stale', options: { obsolete: 123 } },
    })
    runAudioTask.mockResolvedValue({ text: 'done' })
    const wrapper = mountWorkspace()
    await flushPromises()

    expect(wrapper.text()).not.toContain('Style tags')
    expect(wrapper.find('button[data-label="Run task"]').element.disabled).toBe(true)
    await wrapper.find('#audio-task-source_audio').setValue('/new.wav')
    await wrapper.find('button[data-label="Run task"]').trigger('click')
    await flushPromises()

    expect(runAudioTask).toHaveBeenCalledWith({
      modelId: 'future',
      task: 'future_task',
      input: { source_audio: '/new.wav', options: { variation: 0, sample: false } },
      signal: expect.any(AbortSignal),
    })
  })

  it('honors an engine tasks endpoint for a selected speech task', async () => {
    routeQuery.model = 'audio/tts'
    paramRegistryFor = () =>
      audioParamRegistryFixture({
        api_endpoint: '/audioapi/v1/tasks/run',
        request_policy: {
          operation_id: 'tasks_run',
          api_endpoint: '/audioapi/v1/tasks/run',
          body_shape: 'tasks-run',
        },
        workspace_request_fields: [
          { key: 'text', label: 'Engine text', type: 'string', nested: false },
        ],
      })
    getModelConfig.mockResolvedValue({ task: 'tts', model_alias: 'future' })
    const wrapper = mountWorkspace()
    await flushPromises()

    expect(wrapper.find('#audio-task-text').exists()).toBe(true)
    expect(wrapper.find('#audio-speech-text').exists()).toBe(false)
  })

  it('cancels an in-flight speech run and ignores its rejection', async () => {
    routeQuery.model = 'audio/tts'
    routeQuery.tab = 'speech'
    getModelConfig.mockResolvedValue({ task: 'tts', model_alias: 'tts-demo' })
    const run = abortableDeferred()
    synthesizeSpeech.mockImplementation(run.run)

    const wrapper = mountWorkspace()
    await flushPromises()
    await wrapper.find('#audio-speech-text').setValue('Hello')
    await wrapper.find('button[data-label="Generate"]').trigger('click')
    await flushPromises()

    expect(run.signal).toBeInstanceOf(AbortSignal)
    expect(wrapper.find('button[data-label="Cancel"]').exists()).toBe(true)
    await wrapper.find('button[data-label="Cancel"]').trigger('click')
    await flushPromises()

    expect(run.signal.aborted).toBe(true)
    expect(wrapper.find('button[data-label="Cancel"]').exists()).toBe(false)
    expect(wrapper.text()).not.toContain('The operation was aborted')
    expect(wrapper.find('audio.audio-player').exists()).toBe(false)
  })

  it('cancels an in-flight generic task run', async () => {
    routeQuery.model = 'audio/ace'
    routeQuery.tab = 'music'
    getModelConfig.mockResolvedValue({ task: 'gen', model_alias: 'ace-demo' })
    const run = abortableDeferred()
    runAudioTask.mockImplementation(run.run)

    const wrapper = mountWorkspace()
    await flushPromises()
    await wrapper.find('textarea').setValue('lofi beat')
    await wrapper.find('button[data-label="Generate"]').trigger('click')
    await flushPromises()

    await wrapper.find('button[data-label="Cancel"]').trigger('click')
    await flushPromises()

    expect(run.signal.aborted).toBe(true)
    expect(runAudioTask).toHaveBeenCalledTimes(1)
    expect(wrapper.find('audio.audio-player').exists()).toBe(false)
  })

  it('ignores a late speech result after unmount', async () => {
    routeQuery.model = 'audio/tts'
    routeQuery.tab = 'speech'
    getModelConfig.mockResolvedValue({ task: 'tts', model_alias: 'tts-demo' })
    const gate = deferred()
    synthesizeSpeech.mockImplementation(() => gate.promise)

    const wrapper = mountWorkspace()
    await flushPromises()
    await wrapper.find('#audio-speech-text').setValue('Hello')
    await wrapper.find('button[data-label="Generate"]').trigger('click')
    await flushPromises()

    wrapper.unmount()
    gate.resolve({ blob: new Blob([new Uint8Array([1])], { type: 'audio/wav' }) })
    await flushPromises()

    expect(createObjectURL).not.toHaveBeenCalled()
  })

  it('revokes object URLs when a result is replaced and on unmount', async () => {
    routeQuery.model = 'audio/tts'
    routeQuery.tab = 'speech'
    getModelConfig.mockResolvedValue({ task: 'tts', model_alias: 'tts-demo' })
    synthesizeSpeech.mockResolvedValue({
      blob: new Blob([new Uint8Array([1])], { type: 'audio/wav' }),
    })

    const wrapper = mountWorkspace()
    await flushPromises()
    await wrapper.find('#audio-speech-text').setValue('Hello')
    await wrapper.find('button[data-label="Generate"]').trigger('click')
    await flushPromises()
    expect(createObjectURL).toHaveBeenCalledTimes(1)

    await wrapper.find('button[data-label="Generate"]').trigger('click')
    await flushPromises()
    expect(createObjectURL).toHaveBeenCalledTimes(2)
    expect(revokeObjectURL).toHaveBeenCalledWith('blob:test-1')

    wrapper.unmount()
    expect(revokeObjectURL).toHaveBeenCalledWith('blob:test-2')
  })

  it('renders engine version, collapsible drift warnings, and a non-blocking compatibility hint', async () => {
    contractPayload = audioContractFixture({
      engine_version: 'v0.9.7',
      drift_warnings: [
        'Engine advertises an endpoint Studio has no operation for: /v1/future/route',
      ],
      validated_through: 'v0.9.5',
    })
    routeQuery.model = 'audio/tts'
    routeQuery.tab = 'speech'
    getModelConfig.mockResolvedValue({ task: 'tts', model_alias: 'tts-demo' })

    const wrapper = mountWorkspace()
    await flushPromises()

    expect(wrapper.get('[data-testid="audio-engine-version"]').text()).toContain('v0.9.7')
    const drift = wrapper.get('.contract-drift')
    expect(drift.attributes('role')).toBe('status')
    expect(drift.get('summary').text()).toContain('Engine contract drift (1)')
    expect(drift.text()).toContain('/v1/future/route')
    expect(wrapper.text()).toContain('newer than the contract Studio validated')
    expect(wrapper.text()).toContain('v0.9.5')
  })

  it('shows the engine-disabled state instead of the model search CTA', async () => {
    engineDescriptors = [{ ...AUDIO_ENGINE, enabled: false }]
    audioQuantizations = []

    const wrapper = mountWorkspace()
    await flushPromises()

    expect(wrapper.text()).toContain('audio.cpp is disabled')
    expect(wrapper.text()).toContain('AUDIO_CPP_ENABLED')
    expect(wrapper.find('button[data-label="Search models"]').exists()).toBe(false)
    expect(wrapper.find('button[data-label="Open Engines"]').exists()).toBe(true)
  })

  it('shows a needs-setup state when the engine descriptor is missing', async () => {
    engineDescriptors = []
    audioQuantizations = []

    const wrapper = mountWorkspace()
    await flushPromises()

    expect(wrapper.text()).toContain('audio.cpp engine unavailable')
    expect(wrapper.find('button[data-label="Search models"]').exists()).toBe(false)
    expect(wrapper.find('button[data-label="Open Engines"]').exists()).toBe(true)
  })

  it('renders a 409 contract failure without claiming the workspace can run', async () => {
    contractFailure = errorResponse(409, 'No composed audio.cpp contract is available.')
    routeQuery.model = 'audio/tts'
    routeQuery.tab = 'speech'
    getModelConfig.mockResolvedValue({ task: 'tts', model_alias: 'tts-demo' })

    const wrapper = mountWorkspace()
    await flushPromises()

    expect(wrapper.text()).toContain('No composed audio.cpp contract is available')
    expect(wrapper.find('button[data-label="Retry capabilities"]').exists()).toBe(true)
    expect(wrapper.find('button[data-label="Generate"]').element.disabled).toBe(true)
  })

  it('renders an explicit contract schema mismatch', async () => {
    const mismatchedSchema = `${AUDIO_CONTRACT_SCHEMA}-v99`
    contractPayload = { ...audioContractFixture(), schema: mismatchedSchema }
    routeQuery.model = 'audio/tts'
    getModelConfig.mockResolvedValue({ task: 'tts', model_alias: 'tts-demo' })

    const wrapper = mountWorkspace()
    await flushPromises()

    expect(wrapper.text()).toContain('Audio contract schema mismatch')
    expect(wrapper.text()).toContain(mismatchedSchema)
  })

  it('renders the shared page header, one model-context section, and pill task tabs', async () => {
    routeQuery.model = 'audio/tts'
    routeQuery.tab = 'speech'
    getModelConfig.mockResolvedValue({ task: 'tts', model_alias: 'tts-demo' })

    const wrapper = mountWorkspace()
    await flushPromises()

    const header = wrapper.get('.page-header')
    expect(header.get('h1').text()).toBe('Audio')
    expect(header.get('.page-header__description').text()).toContain('audio.cpp model')

    const modelContext = wrapper
      .findAll('.page-section')
      .find((node) => node.text().includes('Model context'))
    expect(modelContext).toBeTruthy()
    expect(modelContext.find('.page-toolbar').exists()).toBe(true)

    const tablist = wrapper.get('[role="tablist"]')
    expect(tablist.attributes('aria-label')).toBe('Audio tasks')
    expect(tablist.classes()).toContain('section-tabs--pills')
    const tab = wrapper.get('[role="tab"]')
    expect(tab.attributes('aria-selected')).toBe('true')
    expect(tab.attributes('aria-controls')).toBe('audio-panel-speech')

    // Task panels are shared sections with a real heading hierarchy.
    const speechPanel = wrapper.get('#audio-panel-speech')
    expect(speechPanel.get('h2').text()).toBe('Speech')
  })

  it('wires tab aria-controls to the rendered panel and announces results', async () => {
    routeQuery.model = 'audio/tts'
    routeQuery.tab = 'speech'
    getModelConfig.mockResolvedValue({ task: 'tts', model_alias: 'tts-demo' })
    const gate = deferred()
    synthesizeSpeech.mockImplementation(() => gate.promise)

    const wrapper = mountWorkspace()
    await flushPromises()

    const tab = wrapper.get('[role="tab"]')
    const controls = tab.attributes('aria-controls')
    expect(controls).toBe('audio-panel-speech')
    expect(document.getElementById(controls)).not.toBeNull()
    const panel = document.getElementById(controls)
    expect(panel.getAttribute('role')).toBe('tabpanel')
    expect(panel.getAttribute('aria-labelledby')).toBe('audio-tab-speech')

    const resultRegion = wrapper.get('.audio-result-region')
    expect(resultRegion.attributes('role')).toBe('status')
    expect(resultRegion.attributes('aria-live')).toBe('polite')

    await wrapper.find('#audio-speech-text').setValue('Hello')
    await wrapper.find('button[data-label="Generate"]').trigger('click')
    await flushPromises()
    expect(resultRegion.attributes('aria-busy')).toBe('true')

    gate.resolve({ blob: new Blob([new Uint8Array([1])], { type: 'audio/wav' }) })
    await flushPromises()
    expect(resultRegion.attributes('aria-busy')).toBe('false')
  })

  it('labels the ASR file input and exposes the record toggle pressed state', async () => {
    routeQuery.model = 'audio/asr'
    routeQuery.tab = 'transcribe'
    getModelConfig.mockResolvedValue({ task: 'asr', model_alias: 'asr-demo' })
    const microphone = stubMicrophone()

    const wrapper = mountWorkspace()
    await flushPromises()

    const label = wrapper.get('label[for="audio-asr-file"]')
    const input = document.getElementById(label.attributes('for'))
    expect(input).not.toBeNull()
    expect(input.classList.contains('sr-only')).toBe(true)
    expect(input.classList.contains('audio-file-input')).toBe(false)

    const record = wrapper.get('button[data-label="Record"]')
    expect(record.attributes('aria-pressed')).toBe('false')
    await record.trigger('click')
    await flushPromises()
    expect(wrapper.get('button[data-label="Stop"]').attributes('aria-pressed')).toBe('true')
    expect(microphone.getUserMedia).toHaveBeenCalledWith({ audio: true })

    await wrapper.get('button[data-label="Stop"]').trigger('click')
    await flushPromises()
    expect(microphone.track.stop).toHaveBeenCalled()
  })

  it('cancels an in-flight transcription', async () => {
    routeQuery.model = 'audio/asr'
    routeQuery.tab = 'transcribe'
    getModelConfig.mockResolvedValue({ task: 'asr', model_alias: 'asr-demo' })
    const run = abortableDeferred()
    transcribeAudio.mockImplementation(run.run)

    const wrapper = mountWorkspace()
    await flushPromises()
    await selectAsrFile(wrapper, new File([new Uint8Array([1])], 'memo.wav', { type: 'audio/wav' }))
    await wrapper.find('button[data-label="Transcribe"]').trigger('click')
    await flushPromises()

    await wrapper.find('button[data-label="Cancel"]').trigger('click')
    await flushPromises()

    expect(run.signal.aborted).toBe(true)
    expect(wrapper.text()).not.toContain('The operation was aborted')
  })

  it('confirms and retries a start withheld by unresolved prior work', async () => {
    routeQuery.model = 'audio/tts'
    getModelConfig.mockResolvedValue({ task: 'tts', model_alias: 'tts-demo' })
    const tts = audioQuantizations.find((quantization) => quantization.id === 'audio/tts')
    tts.is_active = false
    startModel
      .mockReset()
      .mockRejectedValueOnce({
        response: {
          data: {
            detail: {
              code: 'ACTION_RETRY_WITHHELD',
              operation_id: 'start-unknown',
              state_token: 'tok-1',
              message:
                'Prior work may already have happened. Confirm this operation and its current state before trying again.',
            },
          },
        },
      })
      .mockResolvedValue({ model: 'tts-demo', state: 'loading' })
    confirmRequire.mockReset().mockImplementation((options) => options.accept())

    const wrapper = mountWorkspace()
    await flushPromises()

    expect(wrapper.get('.state-banner--warning').text()).toContain(
      'Start this model and wait for a verified running state',
    )
    await wrapper.find('button[data-label="Start"]').trigger('click')
    await flushPromises()

    expect(startModel).toHaveBeenCalledTimes(2)
    expect(startModel).toHaveBeenNthCalledWith(1, 'audio/tts', null)
    expect(startModel).toHaveBeenNthCalledWith(2, 'audio/tts', {
      confirm_operation_id: 'start-unknown',
      confirm_state: 'tok-1',
    })
    expect(confirmRequire).toHaveBeenCalledWith(
      expect.objectContaining({ header: 'Confirm model start' }),
    )
    expect(toastAdd).not.toHaveBeenCalledWith(expect.objectContaining({ summary: 'Start failed' }))
  })
})
