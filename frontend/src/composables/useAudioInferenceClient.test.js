import { afterEach, describe, expect, it, vi } from 'vitest'

import {
  audioApiEndpoint,
  audioInferenceModelId,
  audioTabFromConfig,
  extractAudioClipsFromTaskResult,
  llamaSwapBaseUrl,
  LLAMA_SWAP_AUDIO_TASKS_PATH,
  runAudioTask,
  studioAudioBaseUrl,
  synthesizeSpeech,
  taskKindFromConfig,
  tasksRunRequestObject,
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

  it('routes conversion tasks to audioapi tasks/run and speech/clone to speech', () => {
    expect(audioApiEndpoint({ task: 'vc', family: 'chatterbox' })).toBe(LLAMA_SWAP_AUDIO_TASKS_PATH)
    expect(usesSpeechForConversion({ task: 'vc', family: 'chatterbox' })).toBe(false)
    expect(audioApiEndpoint({ task: 'vc', family: 'vevo2' })).toBe(LLAMA_SWAP_AUDIO_TASKS_PATH)
    expect(audioApiEndpoint({ task: 'svc', family: 'seed_vc' })).toBe(LLAMA_SWAP_AUDIO_TASKS_PATH)
    expect(audioApiEndpoint({ task: 'clon', family: 'chatterbox' })).toBe('/v1/audio/speech')
    expect(audioApiEndpoint({ task: 'tts', family: 'chatterbox' })).toBe('/v1/audio/speech')
  })

  it('prefers inspect-recorded preferred_api_endpoint for non-conversion tasks', () => {
    expect(
      audioApiEndpoint(
        { task: 'tts', family: 'demo' },
        { preferred_api_endpoint: '/v1/audio/speech' },
      ),
    ).toBe('/v1/audio/speech')
    // Conversion must not be overridden onto speech — speech has no source audio field.
    expect(
      audioApiEndpoint(
        { task: 'vc', family: 'vevo2' },
        { preferred_api_endpoint: '/v1/audio/speech' },
      ),
    ).toBe(LLAMA_SWAP_AUDIO_TASKS_PATH)
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

  it('posts generic tasks to llama-swap /audioapi/v1/tasks/run', async () => {
    vi.stubGlobal('window', { location: { hostname: 'studio.test', protocol: 'http:' } })
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      headers: { get: () => 'application/json' },
      json: async () => ({ ok: true }),
    })
    vi.stubGlobal('fetch', fetchMock)

    const result = await runAudioTask({
      modelId: 'vad-demo',
      input: { audio_path: '/a.wav' },
      proxyPort: 2000,
    })

    expect(result).toEqual({ ok: true })
    expect(llamaSwapBaseUrl(2000)).toBe('http://studio.test:2000')
    expect(fetchMock).toHaveBeenCalledWith(
      'http://studio.test:2000/audioapi/v1/tasks/run',
      expect.objectContaining({ method: 'POST' }),
    )
    expect(JSON.parse(fetchMock.mock.calls[0][1].body)).toEqual({
      model: 'vad-demo',
      request: { audio: '/a.wav' },
    })
  })

  it('maps VC aliases into the audio.cpp request object', () => {
    expect(tasksRunRequestObject({
      source_audio: '/data/src.wav',
      target_voice: '/data/ref.wav',
    })).toEqual({
      source_audio: '/data/src.wav',
      target_voice: '/data/ref.wav',
      audio: '/data/src.wav',
      voice_ref: '/data/ref.wav',
    })
  })

  it('decodes base64 WAV task results into playable blobs', async () => {
    // Minimal RIFF/WAV header bytes → base64 "UklGRgAAAABXQVZF..."
    const wavBytes = new Uint8Array([
      0x52, 0x49, 0x46, 0x46, 0x24, 0x00, 0x00, 0x00,
      0x57, 0x41, 0x56, 0x45, 0x66, 0x6d, 0x74, 0x20,
    ])
    let binary = ''
    for (const byte of wavBytes) binary += String.fromCharCode(byte)
    const b64 = btoa(binary)

    const { clips, meta } = extractAudioClipsFromTaskResult({
      audio: b64,
      sample_rate: 48000,
      channels: 2,
      timing: { wall_ms: 12, rtf: 0.5 },
      named_audio_outputs: [{ id: 'drums', audio: b64, sample_rate: 48000, channels: 2 }],
    })

    expect(clips).toHaveLength(2)
    expect(clips[0].filename).toBe('audio.wav')
    expect(clips[0].blob.type).toBe('audio/wav')
    expect(clips[1].id).toBe('drums')
    expect(clips[1].filename).toBe('drums.wav')
    expect(meta).toEqual({
      sample_rate: 48000,
      channels: 2,
      timing: { wall_ms: 12, rtf: 0.5 },
    })
    expect(await clips[0].blob.arrayBuffer()).toBeTruthy()
  })

  it('returns no clips for analysis-only task JSON', () => {
    const { clips, meta } = extractAudioClipsFromTaskResult({
      segments: [{ start_sample: 0, end_sample: 10, confidence: 0.9 }],
      timing: { wall_ms: 3 },
    })
    expect(clips).toEqual([])
    expect(meta).toEqual({
      segments: [{ start_sample: 0, end_sample: 10, confidence: 0.9 }],
      timing: { wall_ms: 3 },
    })
  })

  it('accepts a direct blob payload from speech-like callers', () => {
    const blob = new Blob([new Uint8Array([1, 2])], { type: 'audio/wav' })
    const { clips, meta } = extractAudioClipsFromTaskResult({ blob })
    expect(meta).toBeNull()
    expect(clips).toHaveLength(1)
    expect(clips[0].blob).toBe(blob)
    expect(clips[0].filename).toBe('audio.wav')
  })

  it('ignores empty or non-object task payloads', () => {
    expect(extractAudioClipsFromTaskResult(null)).toEqual({ clips: [], meta: null })
    expect(extractAudioClipsFromTaskResult('nope')).toEqual({ clips: [], meta: null })
    expect(extractAudioClipsFromTaskResult([])).toEqual({ clips: [], meta: null })
  })
})
