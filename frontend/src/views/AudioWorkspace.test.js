import { beforeEach, describe, expect, it, vi } from 'vitest'
import { mount, flushPromises } from '@vue/test-utils'

const toastAdd = vi.fn()
const push = vi.fn()
const replace = vi.fn()
const fetchModels = vi.fn().mockResolvedValue(undefined)
const fetchSystemStatus = vi.fn().mockResolvedValue(undefined)
const getModelConfig = vi.fn()
const listReferenceAudio = vi.fn().mockResolvedValue([])
const runAudioTask = vi.fn()
const synthesizeSpeech = vi.fn()
const transcribeAudio = vi.fn()
const routeQuery = { model: undefined, tab: undefined }

vi.mock('vue-router', () => ({
  useRoute: () => ({ query: routeQuery }),
  useRouter: () => ({ push, replace }),
}))

vi.mock('primevue/usetoast', () => ({
  useToast: () => ({ add: toastAdd }),
}))

vi.mock('@/stores/models', () => ({
  useModelStore: () => ({
    allQuantizations: [
      {
        id: 'audio/asr',
        display_name: 'ASR Demo',
        format: 'audio_cpp',
        engine: 'audio_cpp',
        is_active: true,
        config: { engine: 'audio_cpp' },
      },
      {
        id: 'audio/ace',
        display_name: 'ACE-Step',
        format: 'audio_cpp',
        engine: 'audio_cpp',
        is_active: true,
        config: { engine: 'audio_cpp' },
      },
      {
        id: 'audio/heartmula',
        display_name: 'HeartMuLa',
        format: 'audio_cpp',
        engine: 'audio_cpp',
        is_active: true,
        config: { engine: 'audio_cpp' },
      },
      {
        id: 'audio/tts',
        display_name: 'TTS Demo',
        format: 'audio_cpp',
        engine: 'audio_cpp',
        is_active: true,
        config: { engine: 'audio_cpp' },
      },
      {
        id: 'audio/sep',
        display_name: 'Demucs',
        format: 'audio_cpp',
        engine: 'audio_cpp',
        is_active: true,
        config: { engine: 'audio_cpp' },
      },
    ],
    fetchModels,
    getModelConfig,
    listReferenceAudio: (...args) => listReferenceAudio(...args),
    startModel: vi.fn(),
  }),
}))

vi.mock('@/stores/engines', () => ({
  useEnginesStore: () => ({
    systemStatus: { proxy_status: { healthy: true, port: 2000 } },
    fetchSystemStatus,
  }),
}))

vi.mock('@/composables/useAudioInferenceClient', async (importOriginal) => {
  const actual = await importOriginal()
  return {
    ...actual,
    runAudioTask: (...args) => runAudioTask(...args),
    synthesizeSpeech: (...args) => synthesizeSpeech(...args),
    transcribeAudio: (...args) => transcribeAudio(...args),
  }
})

import AudioWorkspace from './AudioWorkspace.vue'

function wavBase64(bytes = [0x52, 0x49, 0x46, 0x46]) {
  let binary = ''
  for (const byte of bytes) binary += String.fromCharCode(byte)
  return btoa(binary)
}

const mountStubs = {
  Button: {
    props: ['label', 'icon', 'loading', 'disabled', 'text', 'severity', 'outlined', 'size'],
    emits: ['click'],
    template:
      '<button type="button" :data-label="label" :disabled="disabled" @click="$emit(`click`)">{{ label }}</button>',
  },
  Tag: true,
  Dropdown: {
    props: ['modelValue', 'options', 'optionLabel', 'optionValue', 'editable', 'showClear', 'placeholder'],
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
  InputText: {
    props: ['modelValue', 'placeholder', 'class'],
    emits: ['update:modelValue'],
    template:
      '<input class="param-input" :value="modelValue ?? ``" :placeholder="placeholder" @input="$emit(`update:modelValue`, $event.target.value)" />',
  },
  Textarea: {
    props: ['modelValue'],
    emits: ['update:modelValue'],
    template:
      '<textarea :value="modelValue ?? ``" @input="$emit(`update:modelValue`, $event.target.value)" />',
  },
  Message: {
    props: ['severity', 'closable'],
    template: '<div class="message"><slot /></div>',
  },
  LoadingState: true,
  PageHeader: {
    props: ['title'],
    template: '<header><h1>{{ title }}</h1><slot name="meta" /><slot name="actions" /></header>',
  },
  EmptyState: true,
  RouterLink: true,
}

function mountWorkspace() {
  return mount(AudioWorkspace, {
    global: {
      directives: { tooltip: () => {} },
      stubs: mountStubs,
    },
  })
}

describe('AudioWorkspace', () => {
  beforeEach(() => {
    routeQuery.model = undefined
    routeQuery.tab = undefined
    fetchModels.mockReset().mockResolvedValue(undefined)
    fetchSystemStatus.mockReset().mockResolvedValue(undefined)
    getModelConfig.mockReset()
    listReferenceAudio.mockReset().mockResolvedValue([])
    runAudioTask.mockReset()
    synthesizeSpeech.mockReset()
    transcribeAudio.mockReset()
    vi.stubGlobal('URL', {
      createObjectURL: vi.fn(() => 'blob:test-audio'),
      revokeObjectURL: vi.fn(),
    })
  })

  it('renders Audio shell with Transcribe for an ASR model', async () => {
    getModelConfig.mockResolvedValue({
      engine: 'audio_cpp',
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
    expect(getModelConfig).toHaveBeenCalledWith('audio/asr')
    const tabLabels = wrapper.findAll('.config-section-tab').map((n) => n.text())
    expect(tabLabels.some((t) => t.includes('Transcribe'))).toBe(true)
    expect(tabLabels.some((t) => t.includes('Speech'))).toBe(false)
    expect(tabLabels.some((t) => t.includes('Music'))).toBe(false)
    expect(wrapper.text()).toContain('Transcribe')
    expect(wrapper.text()).toContain('audio.cpp UI')
  })

  it('shows playable music result with download after generate', async () => {
    routeQuery.model = 'audio/ace'
    routeQuery.tab = 'music'
    getModelConfig.mockResolvedValue({
      engine: 'audio_cpp',
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

    expect(runAudioTask).toHaveBeenCalledWith(
      expect.objectContaining({
        modelId: 'ace-demo',
        task: 'gen',
        input: expect.objectContaining({
          text: 'lofi beat',
          options: expect.objectContaining({ task_route: 'text2music' }),
        }),
      }),
    )
    expect(wrapper.find('audio.audio-player').exists()).toBe(true)
    expect(wrapper.find('audio.audio-player').attributes('src')).toBe('blob:test-audio')
    expect(wrapper.find('button[data-label="Download WAV"]').exists()).toBe(true)
    expect(wrapper.text()).toContain('wall_ms')
    expect(wrapper.text()).not.toContain(wavBase64())
  })

  it('sends HeartMuLa style tags from the music form', async () => {
    routeQuery.model = 'audio/heartmula'
    routeQuery.tab = 'music'
    getModelConfig.mockResolvedValue({
      engine: 'audio_cpp',
      family: 'heartmula',
      task: 'gen',
      model_alias: 'heartmula-demo',
      task_defaults: {
        options: { tags: 'piano,happy' },
      },
    })
    runAudioTask.mockResolvedValue({
      audio: wavBase64(),
      sample_rate: 48000,
      channels: 2,
    })

    const wrapper = mountWorkspace()
    await flushPromises()

    expect(wrapper.text()).toContain('HeartMuLa expects comma-separated')
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
      engine: 'audio_cpp',
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

    const prompt = wrapper.find('textarea')
    await prompt.setValue('Hello from Studio')
    await wrapper.find('button[data-label="Generate"]').trigger('click')
    await flushPromises()

    expect(synthesizeSpeech).toHaveBeenCalledWith(
      expect.objectContaining({
        modelId: 'tts-demo',
        input: 'Hello from Studio',
      }),
    )
    expect(wrapper.find('audio.audio-player').exists()).toBe(true)
    expect(wrapper.find('button[data-label="Download WAV"]').exists()).toBe(true)
  })

  it('renders separation stems as multiple players', async () => {
    routeQuery.model = 'audio/sep'
    routeQuery.tab = 'separate'
    getModelConfig.mockResolvedValue({
      engine: 'audio_cpp',
      family: 'htdemucs',
      task: 'sep',
      model_alias: 'sep-demo',
    })
    listReferenceAudio.mockResolvedValue([
      { path: '/data/mix.wav', display_path: 'mix.wav' },
    ])
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
    const pathSelect = wrapper.findAll('select').at(-1)
    await pathSelect.setValue('/data/mix.wav')
    await pathSelect.trigger('change')
    await wrapper.find('button[data-label="Separate"]').trigger('click')
    await flushPromises()

    expect(runAudioTask).toHaveBeenCalledWith(
      expect.objectContaining({
        modelId: 'sep-demo',
        task: 'sep',
        input: { audio: '/data/mix.wav' },
      }),
    )
    expect(wrapper.findAll('audio.audio-player')).toHaveLength(2)
    expect(wrapper.text()).toContain('drums')
    expect(wrapper.text()).toContain('vocals')
  })
})
