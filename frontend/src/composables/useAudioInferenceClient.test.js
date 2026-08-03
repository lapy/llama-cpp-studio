import { afterEach, describe, expect, it, vi } from 'vitest'

import {
  audioApiEndpoint,
  audioInferenceModelId,
  audioTabFromConfig,
  runAudioTask,
  studioAudioBaseUrl,
  synthesizeSpeech,
  taskKindFromConfig,
  transcribeAudio,
  usesSpeechForConversion,
} from './useAudioInferenceClient'

describe('useAudioInferenceClient', () => {
  afterEach(() => {
    vi.unstubAllGlobals()
    vi.restoreAllMocks()
  })

  it('resolves inference model id from alias then proxy ids', () => {
    expect(audioInferenceModelId(
      { id: 'pkg/demo', llama_swap_id: 'swap-id' },
      { model_alias: 'friendly' },
    )).toBe('friendly')
    expect(audioInferenceModelId({ id: 'pkg/demo', proxy_name: 'proxy' })).toBe('proxy')
    expect(audioInferenceModelId({ id: 'pkg/demo' })).toBe('pkg/demo')
  })

  it('maps config task to workspace tab kind', () => {
    expect(taskKindFromConfig({ task: 'asr' })).toBe('transcribe')
    expect(taskKindFromConfig({ task: 'tts' })).toBe('speech')
    expect(taskKindFromConfig({ task: 'clon' })).toBe('speech')
    expect(taskKindFromConfig({ task: 'gen' })).toBe('music')
    expect(taskKindFromConfig({ task: 'vc' })).toBe('convert')
    expect(audioTabFromConfig({ task: 'asr' })).toBe('transcribe')
  })

  it('routes chatterbox VC to speech and dedicated VC to tasks/run', () => {
    expect(audioApiEndpoint({ task: 'vc', family: 'chatterbox' })).toBe('/v1/audio/speech')
    expect(usesSpeechForConversion({ task: 'vc', family: 'chatterbox' })).toBe(true)
    expect(audioApiEndpoint({ task: 'vc', family: 'vevo2' })).toBe('/v1/tasks/run')
    expect(usesSpeechForConversion({ task: 'vc', family: 'vevo2' })).toBe(false)
    expect(audioApiEndpoint({ task: 'clon', family: 'chatterbox' })).toBe('/v1/audio/speech')
  })

  it('prefers inspect-recorded preferred_api_endpoint over family hardcodes', () => {
    expect(
      audioApiEndpoint(
        { task: 'vc', family: 'vevo2' },
        { preferred_api_endpoint: '/v1/audio/speech' },
      ),
    ).toBe('/v1/audio/speech')
    expect(
      usesSpeechForConversion(
        { task: 'vc', family: 'vevo2' },
        { preferred_api_endpoint: '/v1/audio/speech' },
      ),
    ).toBe(true)
  })

  it('posts speech JSON to Studio /v1/audio/speech', async () => {
    vi.stubGlobal('window', { location: { origin: 'http://studio.test' } })
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      headers: { get: () => 'audio/wav' },
      blob: async () => new Blob([new Uint8Array([1, 2, 3])], { type: 'audio/wav' }),
    })
    vi.stubGlobal('fetch', fetchMock)

    const { blob } = await synthesizeSpeech({
      modelId: 'tts-demo',
      input: 'hello',
      voice: 'default',
    })

    expect(studioAudioBaseUrl()).toBe('http://studio.test/v1/audio')
    expect(fetchMock).toHaveBeenCalledWith(
      'http://studio.test/v1/audio/speech',
      expect.objectContaining({ method: 'POST' }),
    )
    const body = JSON.parse(fetchMock.mock.calls[0][1].body)
    expect(body).toEqual({ model: 'tts-demo', input: 'hello', voice: 'default' })
    expect(blob.size).toBe(3)
  })

  it('posts multipart transcriptions to Studio /v1/audio/transcriptions', async () => {
    vi.stubGlobal('window', { location: { origin: 'http://studio.test' } })
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      headers: { get: () => 'application/json' },
      json: async () => ({ text: 'hi' }),
    })
    vi.stubGlobal('fetch', fetchMock)

    const file = new File([new Uint8Array([9])], 'memo.ogg', { type: 'audio/ogg' })
    const result = await transcribeAudio({
      modelId: 'asr-demo',
      file,
      language: 'en',
    })

    expect(result).toEqual({ text: 'hi' })
    expect(fetchMock.mock.calls[0][0]).toBe('http://studio.test/v1/audio/transcriptions')
    const form = fetchMock.mock.calls[0][1].body
    expect(form).toBeInstanceOf(FormData)
    expect(form.get('model')).toBe('asr-demo')
    expect(form.get('language')).toBe('en')
  })

  it('posts generic tasks to Studio /v1/audio/tasks/run', async () => {
    vi.stubGlobal('window', { location: { origin: 'http://studio.test' } })
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      headers: { get: () => 'application/json' },
      json: async () => ({ ok: true }),
    })
    vi.stubGlobal('fetch', fetchMock)

    const result = await runAudioTask({
      modelId: 'vad-demo',
      task: 'vad',
      input: { audio_path: '/a.wav' },
    })

    expect(result).toEqual({ ok: true })
    expect(fetchMock).toHaveBeenCalledWith(
      'http://studio.test/v1/audio/tasks/run',
      expect.objectContaining({ method: 'POST' }),
    )
    expect(JSON.parse(fetchMock.mock.calls[0][1].body)).toEqual({
      model: 'vad-demo',
      task: 'vad',
      input: { audio_path: '/a.wav' },
    })
  })
})
