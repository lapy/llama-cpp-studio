import { describe, it, expect } from 'vitest'
import { ref } from 'vue'

import {
  AUDIO_NESTED_SCOPE_KEYS,
  AUDIO_STUDIO_KNOWN_KEYS,
  defaultValueForAudioParam,
  delimitedEnumSeparator,
  fieldStorageHint,
  isDelimitedEnumParam,
  jsonParamDisplay,
  normalizeCsvEnumValue,
  paramDescriptionTooltip,
  paramMatchesSearch,
  useAudioModelConfig,
} from './useAudioModelConfig.js'

describe('useAudioModelConfig pure helpers', () => {
  it('defaultValueForAudioParam handles repeatable, flag, and scalar defaults', () => {
    expect(defaultValueForAudioParam({ value_kind: 'repeatable', default: ['a'] })).toEqual(['a'])
    expect(defaultValueForAudioParam({ value_kind: 'repeatable' })).toEqual([])
    expect(defaultValueForAudioParam({ value_kind: 'flag', negative_flag: '--no-x' })).toBeNull()
    expect(defaultValueForAudioParam({ value_kind: 'flag' })).toBe(true)
    expect(defaultValueForAudioParam({ default: 3 })).toBe(3)
    expect(defaultValueForAudioParam({})).toBeNull()
  })

  it('normalizeCsvEnumValue splits csv and semicolon enums', () => {
    const csv = { value_kind: 'csv_enum' }
    const semi = { value_kind: 'semicolon_enum' }
    expect(normalizeCsvEnumValue('a, b , ,c', csv)).toEqual(['a', 'b', 'c'])
    expect(normalizeCsvEnumValue('a;b', semi)).toEqual(['a', 'b'])
    expect(normalizeCsvEnumValue(['x'], csv)).toEqual(['x'])
    expect(normalizeCsvEnumValue('', csv)).toEqual([])
  })

  it('isDelimitedEnumParam and delimitedEnumSeparator agree', () => {
    const param = { value_kind: 'semicolon_enum' }
    expect(isDelimitedEnumParam(param)).toBe(true)
    expect(delimitedEnumSeparator(param)).toBe(';')
  })

  it('jsonParamDisplay pretty-prints JSON and preserves invalid strings', () => {
    expect(jsonParamDisplay('{"a":1}')).toBe('{\n  "a": 1\n}')
    expect(jsonParamDisplay('not-json')).toBe('not-json')
    expect(jsonParamDisplay({ b: 2 })).toBe('{\n  "b": 2\n}')
    expect(jsonParamDisplay(null)).toBe('')
  })

  it('paramMatchesSearch requires all tokens and can hide unsupported', () => {
    const param = {
      key: 'temperature',
      label: 'Temperature',
      description: 'Sampling temperature',
      supported: false,
    }
    expect(paramMatchesSearch(param, 'temperature sampling')).toBe(true)
    expect(paramMatchesSearch(param, 'temperature missing')).toBe(false)
    expect(paramMatchesSearch(param, 'temperature', true)).toBe(false)
  })

  it('paramDescriptionTooltip includes flags', () => {
    const tooltip = paramDescriptionTooltip({
      label: 'Lazy load',
      description: 'Defer weights',
      primary_flag: '--lazy-load',
      negative_flag: '--no-lazy-load',
    })
    expect(tooltip).toContain('Defer weights')
    expect(tooltip).toContain('--lazy-load')
    expect(tooltip).toContain('--no-lazy-load')
  })

  it('fieldStorageHint distinguishes preset, nested, path, and proxy storage', () => {
    expect(fieldStorageHint({ key: 'voice_id', preset_field: true }, { presetField: true }))
      .toContain('sidecar')
    expect(fieldStorageHint({ key: 'speaker', nested: true, options_key: 'speaker' }))
      .toContain('options.speaker')
    expect(fieldStorageHint({ key: 'voice_ref', type: 'path' }))
      .toContain('bundle directory')
    expect(fieldStorageHint({ key: 'voice' }, { viaProxy: false }))
      .toBe('')
  })

  it('exports studio known keys used by ModelConfig preservation', () => {
    expect(AUDIO_STUDIO_KNOWN_KEYS).toContain('speech_defaults')
    expect(AUDIO_STUDIO_KNOWN_KEYS).toContain('transcription_defaults')
    expect(AUDIO_NESTED_SCOPE_KEYS.load_option).toBe('load_options')
  })
})

function makeComposable(overrides = {}) {
  const config = ref({
    family: 'omnivoice',
    task: 'tts',
    mode: 'offline',
    backend: 'cuda',
    speech_defaults: {},
    voice_presets: {},
    load_options: {},
    session_options: {},
    ...overrides.config,
  })
  const paramRegistry = ref({
    sections: [],
    task_profile: {
      label: 'OmniVoice',
      workflows: ['clone', 'design'],
      summary: 'Clone or design voices.',
      api_hint: 'Use presets.',
    },
    request_field_groups: [
      {
        id: 'voice',
        label: 'Voice',
        fields: [
          { key: 'voice_ref', label: 'Reference', type: 'path', preset_field: true },
          { key: 'instructions', label: 'Instructions', type: 'textarea' },
        ],
      },
    ],
    request_defaults_key: overrides.request_defaults_key || 'speech_defaults',
    api_endpoint: overrides.api_endpoint || '/v1/audio/speech',
    api_example_hint: 'OpenAI-compatible speech synthesis request.',
    inspection: {
      family: 'omnivoice',
      tasks: [{ task: 'tts', modes: ['offline', 'streaming'] }],
    },
    ...overrides.registry,
  })
  const enginesStore = {
    engineDescriptors: [
      { id: 'audio_cpp', available_runtime_backends: ['cpu', 'cuda'] },
    ],
  }
  const llamaSwapStableId = ref('audio-demo')
  return useAudioModelConfig(config, paramRegistry, enginesStore, llamaSwapStableId)
}

describe('useAudioModelConfig composable', () => {
  it('detects TTS task kind and section title from endpoint', () => {
    const tts = makeComposable()
    expect(tts.audioTaskKind.value).toBe('tts')
    expect(tts.requestDefaultsSectionTitle.value).toBe('Speech synthesis defaults')
    expect(tts.taskKindMeta.value.short).toBe('TTS')

    const asr = makeComposable({
      request_defaults_key: 'transcription_defaults',
      api_endpoint: '/v1/audio/transcriptions',
      registry: {
        task_profile: { label: 'Nemotron ASR', workflows: ['offline'], summary: 'ASR' },
      },
      config: { task: 'asr', family: 'nemotron_asr', transcription_defaults: {} },
    })
    expect(asr.audioTaskKind.value).toBe('asr')
    expect(asr.requestDefaultsSectionTitle.value).toBe('Transcription defaults')
  })

  it('supportsVoicePresets only for OpenAI speech tasks outside generic families', () => {
    const tts = makeComposable()
    expect(tts.supportsVoicePresets.value).toBe(true)

    const gen = makeComposable({
      request_defaults_key: 'task_defaults',
      api_endpoint: '/v1/tasks/run',
      config: { family: 'ace_step', task: 'gen', task_defaults: {} },
    })
    expect(gen.supportsVoicePresets.value).toBe(false)

    const asr = makeComposable({
      request_defaults_key: 'transcription_defaults',
      api_endpoint: '/v1/audio/transcriptions',
      config: { task: 'asr', family: 'nemotron_asr', transcription_defaults: {} },
    })
    expect(asr.supportsVoicePresets.value).toBe(false)
  })

  it('swapSetParamsPreview mirrors backend speech and transcription mapping', () => {
    const tts = makeComposable({
      config: {
        speech_defaults: {
          instructions: 'warm narrator',
          temperature: 0.7,
          options: { speaker: 'Vivian' },
        },
      },
    })
    expect(tts.swapSetParamsPreview.value).toEqual({
      instructions: 'warm narrator',
      temperature: 0.7,
      options: { speaker: 'Vivian' },
    })
    expect(tts.configuredDefaultsCount.value).toBe(3)

    const asr = makeComposable({
      request_defaults_key: 'transcription_defaults',
      api_endpoint: '/v1/audio/transcriptions',
      config: {
        task: 'asr',
        family: 'qwen3_asr',
        transcription_defaults: {
          language: 'en',
          prompt: 'Transcribe clearly.',
          options: { num_beams: 4 },
        },
      },
    })
    expect(asr.swapSetParamsPreview.value).toEqual({
      language: 'en',
      options: { num_beams: 4, text: 'Transcribe clearly.' },
    })
  })

  it('setAudioParamValue writes nested scopes and resets invalid mode on task change', () => {
    const config = ref({
      task: 'tts',
      mode: 'streaming',
      family: 'omnivoice',
      backend: 'cuda',
      load_options: {},
      session_options: {},
      speech_defaults: {},
      voice_presets: {},
    })
    const paramRegistry = ref({
      sections: [
        {
          params: [
            { key: 'language', scope: 'load_option', type: 'string' },
            { key: 'temperature', scope: 'session_option', type: 'float' },
            { key: 'task', scope: 'model', type: 'string' },
            { key: 'mode', scope: 'model', type: 'string' },
          ],
        },
      ],
      task_profile: { label: 'OmniVoice', workflows: [], summary: 'x' },
      request_field_groups: [],
      request_defaults_key: 'speech_defaults',
      api_endpoint: '/v1/audio/speech',
      inspection: {
        tasks: [
          { task: 'tts', modes: ['offline', 'streaming'] },
          { task: 'asr', modes: ['offline'] },
        ],
      },
    })
    const api = useAudioModelConfig(
      config,
      paramRegistry,
      { engineDescriptors: [{ id: 'audio_cpp', available_runtime_backends: ['cpu'] }] },
      ref('audio-demo'),
    )

    api.setAudioParamValue({ key: 'language', scope: 'load_option' }, 'en')
    api.setAudioParamValue({ key: 'temperature', scope: 'session_option' }, 0.5)
    expect(config.value.load_options.language).toBe('en')
    expect(config.value.session_options.temperature).toBe(0.5)

    api.setAudioParamValue({ key: 'task', scope: 'model' }, 'asr')
    expect(config.value.task).toBe('asr')
    expect(config.value.mode).toBe('offline')
  })

  it('setAudioParamValue stores null for cleared optional params', () => {
    const config = ref({
      task: 'tts',
      mode: 'offline',
      family: 'omnivoice',
      backend: 'cuda',
      load_options: { language: 'en' },
      session_options: { temperature: 0.5 },
      speech_defaults: { instructions: 'female' },
    })
    const paramRegistry = ref({
      sections: [
        {
          params: [
            { key: 'language', scope: 'load_option', type: 'string' },
            { key: 'temperature', scope: 'session_option', type: 'float' },
          ],
        },
      ],
      task_profile: { label: 'OmniVoice', workflows: [], summary: 'x' },
      request_field_groups: [
        {
          id: 'voice',
          label: 'Voice',
          fields: [{ key: 'instructions', label: 'Instructions', type: 'textarea' }],
        },
      ],
      request_defaults_key: 'speech_defaults',
      api_endpoint: '/v1/audio/speech',
      inspection: { tasks: [{ task: 'tts', modes: ['offline'] }] },
    })
    const api = useAudioModelConfig(
      config,
      paramRegistry,
      { engineDescriptors: [{ id: 'audio_cpp', available_runtime_backends: ['cpu'] }] },
      ref('audio-demo'),
    )

    api.setAudioParamValue({ key: 'language', scope: 'load_option' }, '')
    api.setAudioParamValue({ key: 'temperature', scope: 'session_option' }, null)
    api.setRequestDefaultValue(
      { key: 'instructions', label: 'Instructions', type: 'textarea' },
      '',
    )

    expect(config.value.load_options.language).toBeNull()
    expect(config.value.session_options.temperature).toBeNull()
    expect(config.value.speech_defaults.instructions).toBeNull()
    expect(api.audioParamValue({ key: 'language', scope: 'load_option' })).toBeNull()
    expect(api.swapSetParamsPreview.value).toBeNull()
  })

  it('voice preset CRUD dedupes names and updates default preset', () => {
    const config = ref({
      family: 'omnivoice',
      task: 'tts',
      mode: 'offline',
      backend: 'cuda',
      speech_defaults: {},
      voice_presets: { 'preset-1': {} },
      default_voice_preset: 'preset-1',
    })
    const paramRegistry = ref({
      task_profile: { label: 'OmniVoice', workflows: [], summary: 'x' },
      request_field_groups: [],
      request_defaults_key: 'speech_defaults',
      api_endpoint: '/v1/audio/speech',
      inspection: { tasks: [{ task: 'tts', modes: ['offline'] }] },
    })
    const api = useAudioModelConfig(
      config,
      paramRegistry,
      { engineDescriptors: [{ id: 'audio_cpp', available_runtime_backends: ['cpu'] }] },
      ref('audio-demo'),
    )

    api.addVoicePreset()
    expect(Object.keys(config.value.voice_presets)).toEqual(['preset-1', 'preset-2'])

    api.renameVoicePreset('preset-2', 'assistant')
    expect(config.value.voice_presets.assistant).toEqual({})
    expect(config.value.default_voice_preset).toBe('preset-1')

    api.setVoicePresetField('assistant', 'voice_id', 'M1')
    expect(config.value.voice_presets.assistant.voice_id).toBe('M1')

    api.removeVoicePreset('assistant')
    expect(config.value.voice_presets.assistant).toBeUndefined()
  })

  it('setRequestDefaultValue syncs prompt and nested option coercion', () => {
    const config = ref({
      family: 'omnivoice',
      task: 'tts',
      mode: 'offline',
      backend: 'cuda',
      speech_defaults: {},
      voice_presets: {},
    })
    const api = useAudioModelConfig(
      config,
      ref({
        task_profile: { label: 'OmniVoice', workflows: [], summary: 'x' },
        request_field_groups: [],
        request_defaults_key: 'speech_defaults',
        api_endpoint: '/v1/audio/speech',
        inspection: { tasks: [{ task: 'tts', modes: ['offline'] }] },
      }),
      { engineDescriptors: [{ id: 'audio_cpp', available_runtime_backends: ['cpu'] }] },
      ref('audio-demo'),
    )

    const promptField = { key: 'prompt', nested: true, options_key: 'text', type: 'textarea' }
    api.setRequestDefaultValue(promptField, '  meeting context  ')
    expect(config.value.speech_defaults.prompt).toBe('meeting context')

    const beamField = { key: 'num_beams', nested: true, options_key: 'num_beams', type: 'int' }
    api.setRequestDefaultValue(beamField, '4')
    expect(config.value.speech_defaults.options.num_beams).toBe(4)

    api.setRequestDefaultValue({ key: 'stream', type: 'bool' }, true)
    expect(config.value.speech_defaults.stream).toBe(true)
  })

  it('requestApiExample includes default voice preset and multipart hint', () => {
    const tts = makeComposable({
      config: {
        default_voice_preset: 'assistant',
        speech_defaults: { instructions: 'warm' },
      },
    })
    const example = tts.requestApiExample.value
    expect(example).toContain('/v1/audio/speech')
    expect(example).toContain('"voice": "assistant"')
    expect(example).toContain('"instructions": "warm"')

    const asr = makeComposable({
      request_defaults_key: 'transcription_defaults',
      api_endpoint: '/v1/audio/transcriptions',
      config: {
        task: 'asr',
        family: 'nemotron_asr',
        transcription_defaults: { language: 'en' },
      },
    })
    expect(asr.requestApiExample.value).toContain('Multipart')
    expect(asr.requestApiExample.value).toContain('speech.wav')
  })

  it('setupChecklist tracks progress for required and optional items', () => {
    const api = makeComposable({
      config: {
        family: 'omnivoice',
        task: 'tts',
        backend: 'cuda',
        voice_presets: { assistant: { voice_id: 'M1' } },
        speech_defaults: { instructions: 'warm' },
      },
    })
    expect(api.setupProgress.value).toBeGreaterThan(0)
    const ids = api.setupChecklist.value.map((item) => item.id)
    expect(ids).toContain('family')
    expect(ids).toContain('voice')
    expect(ids).toContain('defaults')
  })

  it('audioParamOptions derives mode and backend choices from inspection and descriptors', () => {
    const api = makeComposable()
    const modes = api.audioParamOptions({ key: 'mode' })
    expect(modes.map((item) => item.value)).toEqual(['offline', 'streaming'])
    const backends = api.audioParamOptions({ key: 'backend' })
    expect(backends.map((item) => item.value)).toEqual(['cpu', 'cuda'])
  })
})
