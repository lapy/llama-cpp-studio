import { afterEach, describe, expect, it, vi } from 'vitest'

import * as audioClient from './useAudioInferenceClient'
import {
  acceptedSpeechRateKeys,
  audioInferenceModelId,
  audioTabFromConfig,
  extractAudioClipsFromTaskResult,
  isGenericTaskEndpoint,
  llamaSwapBaseUrl,
  runAudioTask,
  speechRateRequestFields,
  studioAudioBaseUrl,
  synthesizeSpeech,
  taskKindFromConfig,
  tasksRunRequestObject,
  parseBatchTranscriptionPayload,
  transcribeAudio,
  transcribeAudioBatch,
} from './useAudioInferenceClient'
import {
  audioContract,
  audioEndpointIsGeneric,
  audioTaskBodyShape,
  audioTaskDefaultsKey,
  audioTaskEndpoint,
  OFFLINE_AUDIO_CONTRACT,
} from './useAudioContract'

describe('useAudioInferenceClient', () => {
  afterEach(() => {
    vi.unstubAllGlobals()
    vi.restoreAllMocks()
  })

  it('does not expose the removed native WebUI helper', () => {
    expect(audioClient.audioCppUpstreamUiUrl).toBeUndefined()
  })

  it('resolves inference model id from alias then proxy ids', () => {
    expect(audioInferenceModelId(
      { id: 'pkg/demo', llama_swap_id: 'swap-id' },
      { model_alias: 'friendly' },
    )).toBe('friendly')
    expect(audioInferenceModelId({ id: 'pkg/demo', proxy_name: 'proxy' })).toBe('proxy')
    expect(audioInferenceModelId({ id: 'pkg/demo' })).toBe('pkg/demo')
  })

  it('sends speech rate only for scanned request options', () => {
    const registry = {
      sections: [{
        params: [
          { scope: 'request_option', key: 'speed' },
          { scope: 'session_option', key: 'speaking_rate' },
        ],
      }],
    }
    expect(acceptedSpeechRateKeys(registry)).toEqual(['speed'])
    expect(speechRateRequestFields(
      { options: { speed: '1.25', speaking_rate: 0.8 } },
      acceptedSpeechRateKeys(registry),
    )).toEqual({ speed: 1.25 })
    expect(speechRateRequestFields(
      { options: { speed: '1.25' } },
      [],
    )).toEqual({})
  })

  it('maps config task to workspace tab kind', () => {
    expect(taskKindFromConfig({ task: 'asr' })).toBe('transcribe')
    expect(taskKindFromConfig({ task: 'tts' })).toBe('speech')
    expect(taskKindFromConfig({ task: 'clon' })).toBe('speech')
    expect(taskKindFromConfig({ task: 'gen' })).toBe('music')
    expect(taskKindFromConfig({ task: 'vc' })).toBe('convert')
    expect(audioTabFromConfig({ task: 'asr' })).toBe('transcribe')
  })

  it('falls through to the generic route before the contract has loaded', () => {
    // Offline floor: no operations table → every task hits the generic route.
    expect(audioTaskEndpoint('tts')).toBe(OFFLINE_AUDIO_CONTRACT.generic_studio_path)
    expect(audioTaskDefaultsKey('tts')).toBe('task_defaults')
    expect(isGenericTaskEndpoint(audioTaskEndpoint('vc'))).toBe(true)
    expect(isGenericTaskEndpoint('/v1/audio/speech')).toBe(false)
    expect(isGenericTaskEndpoint('')).toBe(false)
    expect(audioEndpointIsGeneric('/v1/tasks/run')).toBe(true)
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

  it('posts every selected file to /v1/batches/transcriptions', async () => {
    vi.stubGlobal('window', { location: { origin: 'http://studio.test' } })
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      headers: { get: () => 'application/json' },
      text: async () => JSON.stringify({ results: [{ filename: 'a.wav', text: 'one' }] }),
    })
    vi.stubGlobal('fetch', fetchMock)

    const result = await transcribeAudioBatch({
      modelId: 'meeting',
      files: [
        new File([new Uint8Array([1])], 'a.wav', { type: 'audio/wav' }),
        new File([new Uint8Array([2])], 'b.wav', { type: 'audio/wav' }),
      ],
      language: 'en',
    })

    expect(fetchMock.mock.calls[0][0]).toBe('http://studio.test/v1/batches/transcriptions')
    const form = fetchMock.mock.calls[0][1].body
    expect(form.getAll('file')).toHaveLength(2)
    expect(form.get('model')).toBe('meeting')
    expect(result.results[0]).toEqual({ name: 'a.wav', text: 'one' })
  })

  it('reads batch event-stream transcripts', () => {
    const parsed = parseBatchTranscriptionPayload(
      'text/event-stream',
      'data: {"text":"hi"}\n\ndata: [DONE]\n\n',
    )
    expect(parsed.results).toEqual([{ name: 'File 1', text: 'hi' }])
  })

  it('posts generic tasks to Studio /v1/tasks/run', async () => {
    vi.stubGlobal('window', { location: { origin: 'http://studio.test', hostname: 'studio.test', protocol: 'http:' } })
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      headers: { get: () => 'application/json' },
      json: async () => ({ ok: true }),
    })
    vi.stubGlobal('fetch', fetchMock)

    const result = await runAudioTask({
      modelId: 'vad-demo',
      input: { audio_path: '/a.wav' },
    })

    expect(result).toEqual({ ok: true })
    expect(llamaSwapBaseUrl(2000)).toBe('http://studio.test:2000')
    expect(fetchMock).toHaveBeenCalledWith(
      'http://studio.test/v1/tasks/run',
      expect.objectContaining({ method: 'POST' }),
    )
    expect(JSON.parse(fetchMock.mock.calls[0][1].body)).toEqual({
      model: 'vad-demo',
      request: { audio_path: '/a.wav' },
    })
  })

  it('routes tasks from the loaded contract payload', () => {
    const previous = audioContract.value
    audioContract.value = {
      ...OFFLINE_AUDIO_CONTRACT,
      defaults_keys: {
        ...OFFLINE_AUDIO_CONTRACT.defaults_keys,
        speech: 'speech_defaults',
        transcriptions: 'transcription_defaults',
        alignments: 'task_defaults',
      },
      task_routing: {
        tts: 'speech',
        asr: 'transcriptions',
        align: 'alignments',
        vc: 'tasks_run',
      },
      operations: [
        { id: 'speech', studio_path: '/v1/audio/speech', request_defaults_key: 'speech_defaults', body_shape: 'openai-json' },
        { id: 'transcriptions', studio_path: '/v1/audio/transcriptions', request_defaults_key: 'transcription_defaults', body_shape: 'multipart-form' },
        { id: 'alignments', studio_path: '/v1/audio/alignments', request_defaults_key: 'task_defaults', body_shape: 'multipart-form' },
        { id: 'tasks_run', studio_path: '/audioapi/v1/tasks/run', request_defaults_key: 'task_defaults', body_shape: 'tasks-run' },
      ],
    }
    try {
      expect(audioTaskEndpoint('align')).toBe('/v1/audio/alignments')
      expect(audioTaskEndpoint('tts')).toBe('/v1/audio/speech')
      expect(audioTaskEndpoint('asr')).toBe('/v1/audio/transcriptions')
      expect(audioTaskDefaultsKey('asr')).toBe('transcription_defaults')
      expect(audioTaskBodyShape('align')).toBe('multipart-form')
      expect(audioTaskBodyShape('vc')).toBe('tasks-run')
      // Unknown task ids fall through to the generic route like the backend.
      expect(audioTaskEndpoint('brand_new_task')).toBe('/audioapi/v1/tasks/run')
    } finally {
      audioContract.value = previous
    }
  })

  it('preserves engine input names without inventing conversion aliases', () => {
    expect(tasksRunRequestObject({
      source_audio: '/data/src.wav',
      target_voice: '/data/ref.wav',
    })).toEqual({
      source_audio: '/data/src.wav',
      target_voice: '/data/ref.wav',
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
