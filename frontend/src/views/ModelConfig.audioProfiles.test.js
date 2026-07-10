import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest'
import { mount, flushPromises } from '@vue/test-utils'

import ModelConfig from './ModelConfig.vue'

const toastAdd = vi.fn()
const fetchModels = vi.fn()
const listReferenceAudio = vi.fn().mockResolvedValue([])
const uploadReferenceAudio = vi.fn()
const deleteReferenceAudio = vi.fn()
const fetchSwapConfigStale = vi.fn()
const fetchGpuList = vi.fn()
const applySwapConfig = vi.fn()
const markSwapConfigStaleLocal = vi.fn()

vi.mock('axios', () => ({
  default: {
    get: vi.fn(),
    post: vi.fn(),
    put: vi.fn(),
  },
}))

vi.mock('vue-router', () => ({
  useRoute: () => ({ params: { id: 'audio-model-1' } }),
}))

vi.mock('primevue/usetoast', () => ({
  useToast: () => ({ add: toastAdd }),
}))

vi.mock('@/stores/models', () => ({
  useModelStore: () => ({
    models: [
      {
        base_model_name: 'Audio Model',
        huggingface_id: 'org/audio-model',
        quantizations: [
          {
            id: 'audio-model-1',
            display_name: 'Audio Model',
            base_model_name: 'Audio Model',
            huggingface_id: 'org/audio-model',
            format: 'audio_cpp',
          },
        ],
      },
    ],
    allQuantizations: [],
    fetchModels,
    listReferenceAudio,
    uploadReferenceAudio,
    deleteReferenceAudio,
  }),
}))

const fetchEngineDescriptors = vi.fn()

vi.mock('@/stores/engines', () => ({
  useEnginesStore: () => ({
    swapConfigStale: { applicable: false, stale: false },
    engineDescriptors: [
      {
        id: 'audio_cpp',
        label: 'audio.cpp',
        available_runtime_backends: ['cpu', 'cuda'],
      },
    ],
    fetchSwapConfigStale,
    fetchGpuList,
    fetchEngineDescriptors,
    applySwapConfig,
    markSwapConfigStaleLocal,
  }),
}))

import axios from 'axios'

const buttonStub = {
  props: ['label', 'icon', 'text', 'severity', 'loading', 'outlined', 'rounded'],
  emits: ['click'],
  template: '<button :data-label="label" @click="$emit(`click`)">{{ label }}</button>',
}

const textInputStub = {
  props: ['modelValue', 'placeholder', 'disabled', 'id'],
  emits: ['update:modelValue'],
  template:
    '<input :id="id" :value="modelValue ?? ``" :placeholder="placeholder" :disabled="disabled" @input="$emit(`update:modelValue`, $event.target.value)" />',
}

const textareaStub = {
  props: ['modelValue', 'rows', 'readonly'],
  emits: ['update:modelValue'],
  template:
    '<textarea :value="modelValue ?? ``" :readonly="readonly" @input="$emit(`update:modelValue`, $event.target.value)" />',
}

function selectStub(name = 'select') {
  return {
    props: ['modelValue', 'options', 'disabled', 'id'],
    emits: ['update:modelValue'],
    template: `<${name} :id="id" :disabled="disabled" :value="modelValue ?? ''" @change="$emit('update:modelValue', $event.target.value)">
      <option value=""></option>
      <option v-for="opt in options || []" :key="opt.value ?? opt" :value="opt.value ?? opt">{{ opt.label ?? opt }}</option>
    </${name}>`,
  }
}

function mountView() {
  return mount(ModelConfig, {
    global: {
      directives: {
        tooltip: () => {},
      },
      stubs: {
        Button: buttonStub,
        Tag: { props: ['value'], template: '<span>{{ value }}</span>' },
        InputText: textInputStub,
        InputNumber: textInputStub,
        InputSwitch: {
          props: ['modelValue', 'inputId'],
          emits: ['update:modelValue'],
          template:
            '<input :id="inputId" type="checkbox" :checked="Boolean(modelValue)" @change="$emit(`update:modelValue`, $event.target.checked)" />',
        },
        Dropdown: selectStub(),
        Chips: {
          props: ['modelValue', 'id'],
          emits: ['update:modelValue'],
          template:
            '<input :id="id" :value="Array.isArray(modelValue) ? modelValue.join(`,`) : ``" @input="$emit(`update:modelValue`, $event.target.value ? $event.target.value.split(`,`) : [])" />',
        },
        Message: { template: '<div><slot /></div>' },
        Textarea: textareaStub,
        MultiSelect: {
          props: ['modelValue'],
          emits: ['update:modelValue'],
          template: '<div class="multiselect-stub" />',
        },
        Slider: true,
        LoadingState: { template: '<div>loading</div>' },
        EmptyState: { template: '<div><slot /></div>' },
        PageHeader: { template: '<div><slot name="start" /><slot name="title" /><slot name="actions" /></div>' },
        Dialog: {
          props: ['visible', 'header'],
          emits: ['update:visible', 'show'],
          template:
            '<div v-if="visible" class="dialog-stub"><slot /><slot name="footer" /></div>',
        },
      },
    },
  })
}

function audioConfigResponse(audioSection = {}) {
  return {
    engine: 'audio_cpp',
    engines: {
      audio_cpp: {
        family: 'omnivoice',
        task: 'tts',
        mode: 'offline',
        backend: 'cuda',
        ...audioSection,
      },
    },
  }
}

function paramRegistryResponse(overrides = {}) {
  return {
    sections: [
      {
        id: 'identity',
        label: 'Identity',
        params: [
          {
            key: 'family',
            label: 'Family',
            type: 'string',
            scope: 'process',
            supported: true,
          },
          {
            key: 'task',
            label: 'Task',
            type: 'string',
            scope: 'process',
            supported: true,
          },
          {
            key: 'mode',
            label: 'Mode',
            type: 'string',
            scope: 'process',
            supported: true,
          },
          {
            key: 'backend',
            label: 'Backend',
            type: 'string',
            scope: 'process',
            supported: true,
          },
        ],
      },
    ],
    scan_error: null,
    scan_pending: false,
    task_profile: {
      label: 'OmniVoice',
      workflows: ['clone', 'design'],
      summary: 'Clone from reference audio or design a voice with instructions.',
      api_hint: 'Use voice presets for cloning.',
    },
    request_field_groups: [
      {
        id: 'voice',
        label: 'Voice & reference',
        fields: [
          { key: 'voice_ref', label: 'Reference audio', type: 'path', preset_field: true },
        ],
      },
    ],
    request_defaults_key: 'speech_defaults',
    api_endpoint: '/v1/audio/speech',
    api_example_hint: 'OpenAI-compatible speech synthesis request.',
    ...overrides,
  }
}

function setupAudioMocks(registryOverrides = {}, configSection = {}) {
  vi.mocked(axios.get).mockImplementation((url, config) => {
    if (url === '/api/models/audio-model-1/config') {
      return Promise.resolve({ data: audioConfigResponse(configSection) })
    }
    if (url === '/api/models/param-registry') {
      return Promise.resolve({
        data: paramRegistryResponse(registryOverrides),
      })
    }
    if (url === '/api/models/audio-model-1/saved-llama-swap-cmd') {
      return Promise.resolve({ data: { ok: true, cmd: 'saved-cmd' } })
    }
    if (url === '/api/gpu-list') {
      return Promise.resolve({
        data: { vendor: null, device_count: 0, gpus: [], cpu_only_mode: true },
      })
    }
    throw new Error(`Unexpected GET ${url} ${JSON.stringify(config?.params || {})}`)
  })

  vi.mocked(axios.put).mockResolvedValue({
    data: audioConfigResponse(configSection),
  })
}

describe('ModelConfig audio profiles', () => {
  beforeEach(() => {
    vi.useFakeTimers()
    toastAdd.mockReset()
    fetchModels.mockReset()
    fetchSwapConfigStale.mockReset()
    fetchGpuList.mockReset()
    fetchEngineDescriptors.mockReset()
    applySwapConfig.mockReset()
    markSwapConfigStaleLocal.mockReset()
    listReferenceAudio.mockReset()
    listReferenceAudio.mockResolvedValue([])
    uploadReferenceAudio.mockReset()
    deleteReferenceAudio.mockReset()
    vi.mocked(axios.get).mockReset()
    vi.mocked(axios.post).mockReset()
    vi.mocked(axios.put).mockReset()

    fetchGpuList.mockResolvedValue({
      vendor: null,
      device_count: 0,
      gpus: [],
      cpu_only_mode: true,
    })
    fetchEngineDescriptors.mockResolvedValue([
      {
        id: 'audio_cpp',
        label: 'audio.cpp',
        available_runtime_backends: ['cpu', 'cuda'],
      },
    ])
    vi.mocked(axios.post).mockImplementation((url, payload) => {
      if (url === '/api/models/audio-model-1/preview-llama-swap-cmd') {
        return Promise.resolve({ data: { ok: true, cmd: JSON.stringify(payload) } })
      }
      throw new Error(`Unexpected POST ${url}`)
    })
  })

  afterEach(() => {
    vi.useRealTimers()
  })

  it('renders speech profile card for TTS models', async () => {
    setupAudioMocks()
    const wrapper = mountView()
    await flushPromises()
    await vi.runAllTimersAsync()
    await flushPromises()

    expect(wrapper.text()).toContain('Speech synthesis defaults')
    expect(wrapper.text()).toContain('Clone from reference audio')
    expect(wrapper.text()).toContain('Voice presets')
    expect(wrapper.text()).toContain('/v1/audio/speech')
    expect(wrapper.text()).toContain('Text-to-Speech configuration')
    expect(wrapper.text()).toContain('llama-swap setParams')
  })

  it('renders transcription profile card for ASR models', async () => {
    setupAudioMocks(
      {
        task_profile: {
          label: 'Nemotron ASR',
          workflows: ['offline', 'streaming'],
          summary: 'RNNT ASR with language prompts.',
          api_hint: 'Supports streaming.',
        },
        request_field_groups: [
          {
            id: 'context',
            label: 'Prompt & language',
            fields: [{ key: 'language', label: 'Language', type: 'string' }],
          },
        ],
        request_defaults_key: 'transcription_defaults',
        api_endpoint: '/v1/audio/transcriptions',
        api_example_hint: 'JSON uses a server-local audio path.',
      },
      { family: 'nemotron_asr', task: 'asr' },
    )
    const wrapper = mountView()
    await flushPromises()
    await vi.runAllTimersAsync()
    await flushPromises()

    expect(wrapper.text()).toContain('Transcription defaults')
    expect(wrapper.text()).toContain('RNNT ASR with language prompts')
    expect(wrapper.text()).not.toContain('Add preset')
    expect(wrapper.text()).toContain('/v1/audio/transcriptions')
    expect(wrapper.text()).toContain('Speech-to-Text configuration')
  })

  it('renders generic task defaults card for generation models', async () => {
    setupAudioMocks(
      {
        task_profile: {
          label: 'ACE-Step',
          workflows: ['text2music', 'repaint'],
          summary: 'Generate and edit music from prompts.',
          api_hint: 'Routes control source audio requirements.',
        },
        request_field_groups: [
          {
            id: 'route',
            label: 'Route',
            fields: [{ key: 'task_route', label: 'Task route', type: 'string' }],
          },
        ],
        request_defaults_key: 'task_defaults',
        api_endpoint: '/v1/tasks/run',
        api_example_hint: 'Generic task request via /v1/tasks/run.',
      },
      { family: 'ace_step', task: 'gen' },
    )
    const wrapper = mountView()
    await flushPromises()
    await vi.runAllTimersAsync()
    await flushPromises()

    expect(wrapper.text()).toContain('Task request defaults')
    expect(wrapper.text()).toContain('Generate and edit music')
    expect(wrapper.text()).toContain('/v1/tasks/run')
    expect(wrapper.text()).toContain('Generic task request via /v1/tasks/run')
  })

  it('persists speech_defaults and voice presets on save', async () => {
    setupAudioMocks(
      {},
      {
        voice_presets: { assistant: { voice_id: 'M1' } },
        default_voice_preset: 'assistant',
        speech_defaults: { voice: 'assistant', language: 'en' },
      },
    )
    const wrapper = mountView()
    await flushPromises()
    await vi.runAllTimersAsync()
    await flushPromises()

    await wrapper.get('button[data-label="Save Configuration"]').trigger('click')
    await flushPromises()

    expect(axios.put).toHaveBeenCalledWith(
      '/api/models/audio-model-1/config',
      expect.objectContaining({
        engine: 'audio_cpp',
        engines: {
          audio_cpp: expect.objectContaining({
            voice_presets: { assistant: { voice_id: 'M1' } },
            default_voice_preset: 'assistant',
            speech_defaults: { voice: 'assistant', language: 'en' },
          }),
        },
      }),
    )
  })

  it('persists task_defaults on save for generic tasks', async () => {
    setupAudioMocks(
      {
        task_profile: {
          label: 'Seed-VC',
          workflows: ['v2_vc'],
          summary: 'Voice conversion.',
          api_hint: 'Default vc route is v2_vc.',
        },
        request_field_groups: [
          {
            id: 'audio',
            label: 'Audio inputs',
            fields: [{ key: 'audio', label: 'Input audio', type: 'path' }],
          },
        ],
        request_defaults_key: 'task_defaults',
        api_endpoint: '/v1/tasks/run',
      },
      {
        family: 'seed_vc',
        task: 'vc',
        task_defaults: {
          audio: 'source.wav',
          options: { task_route: 'v2_vc' },
        },
      },
    )
    const wrapper = mountView()
    await flushPromises()
    await vi.runAllTimersAsync()
    await flushPromises()

    await wrapper.get('button[data-label="Save Configuration"]').trigger('click')
    await flushPromises()

    expect(axios.put).toHaveBeenCalledWith(
      '/api/models/audio-model-1/config',
      expect.objectContaining({
        engines: {
          audio_cpp: expect.objectContaining({
            task_defaults: {
              audio: 'source.wav',
              options: { task_route: 'v2_vc' },
            },
          }),
        },
      }),
    )
  })

  it('does not warn about studio-managed audio.cpp keys', async () => {
    setupAudioMocks(
      {},
      {
        lazy_load: false,
        speech_defaults: { voice: 'assistant' },
        voice_presets: { assistant: { voice_id: 'M1' } },
        default_voice_preset: 'assistant',
      },
    )
    const wrapper = mountView()
    await flushPromises()
    await vi.runAllTimersAsync()
    await flushPromises()

    expect(wrapper.text()).not.toContain('Unrecognized audio.cpp keys')
  })

  it('shows fallback guidance when registry has no task_profile', async () => {
    setupAudioMocks({
      task_profile: null,
      request_field_groups: [],
      request_defaults_key: 'task_defaults',
      api_endpoint: '/v1/tasks/run',
    })
    const wrapper = mountView()
    await flushPromises()
    await vi.runAllTimersAsync()
    await flushPromises()

    expect(wrapper.text()).not.toContain('Speech synthesis defaults')
    expect(wrapper.text()).not.toContain('Transcription defaults')
    expect(wrapper.text()).toContain('No curated profile exists for this family yet')
    expect(listReferenceAudio).toHaveBeenCalledWith('audio-model-1')
  })

  it('loads reference audio library on the Defaults tab for profiled models', async () => {
    listReferenceAudio.mockResolvedValue([
      {
        path: 'refs/voice.wav',
        filename: 'voice.wav',
        size_bytes: 1024,
        used_by: [],
      },
    ])
    setupAudioMocks()
    const wrapper = mountView()
    await flushPromises()
    await vi.runAllTimersAsync()
    await flushPromises()

    const defaultsTab = wrapper.findAll('button').find((btn) => btn.text().includes('Defaults'))
    await defaultsTab.trigger('click')
    await flushPromises()

    expect(wrapper.text()).toContain('Reference audio library')
    expect(wrapper.text()).toContain('refs/voice.wav')
  })

  it('persists transcription_defaults on save for ASR models', async () => {
    setupAudioMocks(
      {
        task_profile: {
          label: 'Nemotron ASR',
          workflows: ['offline'],
          summary: 'RNNT ASR.',
          api_hint: 'Supports streaming.',
        },
        request_field_groups: [
          {
            id: 'context',
            label: 'Prompt & language',
            fields: [{ key: 'language', label: 'Language', type: 'string' }],
          },
        ],
        request_defaults_key: 'transcription_defaults',
        api_endpoint: '/v1/audio/transcriptions',
      },
      {
        family: 'nemotron_asr',
        task: 'asr',
        transcription_defaults: {
          language: 'en',
          prompt: 'Transcribe clearly.',
          options: { num_beams: 4 },
        },
      },
    )
    const wrapper = mountView()
    await flushPromises()
    await vi.runAllTimersAsync()
    await flushPromises()

    await wrapper.get('button[data-label="Save Configuration"]').trigger('click')
    await flushPromises()

    expect(axios.put).toHaveBeenCalledWith(
      '/api/models/audio-model-1/config',
      expect.objectContaining({
        engines: {
          audio_cpp: expect.objectContaining({
            transcription_defaults: {
              language: 'en',
              prompt: 'Transcribe clearly.',
              options: { num_beams: 4 },
            },
          }),
        },
      }),
    )
  })

  it('preserves unknown audio.cpp keys on save', async () => {
    setupAudioMocks(
      {},
      {
        custom_future_sidecar_key: { keep: true },
        speech_defaults: { voice: 'assistant' },
      },
    )
    const wrapper = mountView()
    await flushPromises()
    await vi.runAllTimersAsync()
    await flushPromises()

    await wrapper.get('button[data-label="Save Configuration"]').trigger('click')
    await flushPromises()

    expect(axios.put).toHaveBeenCalledWith(
      '/api/models/audio-model-1/config',
      expect.objectContaining({
        engines: {
          audio_cpp: expect.objectContaining({
            custom_future_sidecar_key: { keep: true },
          }),
        },
      }),
    )
  })

  it('hides sub-id variants section for audio engine', async () => {
    setupAudioMocks()
    const wrapper = mountView()
    await flushPromises()
    await vi.runAllTimersAsync()
    await flushPromises()

    expect(wrapper.text()).not.toContain('Sub-ID variants')
  })
})
