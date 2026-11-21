import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import * as audioClient from './useAudioInferenceClient'
import {
  acceptedSpeechRateKeys,
  audioInferenceModelId,
  audioTabFromConfig,
  extractAudioClipsFromTaskResult,
  runAudioTask,
  speechRateRequestFields,
  studioAudioBaseUrl,
  synthesizeSpeech,
  streamSpeech,
  pcmToWavBlob,
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
  audioTaskRequestPath,
  OFFLINE_AUDIO_CONTRACT,
  resetAudioContract,
} from './useAudioContract'
import { audioContractFixture } from '@/test-fixtures/audioContract'

describe('useAudioInferenceClient', () => {
  beforeEach(() => {
    resetAudioContract()
    audioContract.value = audioContractFixture()
  })

  afterEach(() => {
    vi.unstubAllGlobals()
    vi.restoreAllMocks()
    resetAudioContract()
  })

  it('does not expose the removed native WebUI helper', () => {
    expect(audioClient.audioCppUpstreamUiUrl).toBeUndefined()
  })

  it('resolves inference model id from alias then proxy ids', () => {
    expect(
      audioInferenceModelId(
        { id: 'pkg/demo', proxy_model_id: 'swap-id' },
        { model_alias: 'friendly' },
      ),
    ).toBe('friendly')
    expect(audioInferenceModelId({ id: 'pkg/demo', proxy_name: 'proxy' })).toBe('proxy')
    expect(audioInferenceModelId({ id: 'pkg/demo' })).toBe('pkg/demo')
  })

  it('sends speech rate only for scanned request options', () => {
    const registry = {
      sections: [
        {
          params: [
            { scope: 'request_option', key: 'speed' },
            { scope: 'session_option', key: 'speaking_rate' },
          ],
        },
      ],
    }
    expect(acceptedSpeechRateKeys(registry)).toEqual(['speed'])
    expect(
      speechRateRequestFields(
        { options: { speed: '1.25', speaking_rate: 0.8 } },
        acceptedSpeechRateKeys(registry),
      ),
    ).toEqual({ speed: 1.25 })
    expect(speechRateRequestFields({ options: { speed: '1.25' } }, [])).toEqual({})
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
    resetAudioContract()
    // Offline floor: no operations table → every task hits the generic route.
    expect(audioTaskEndpoint('tts')).toBe(OFFLINE_AUDIO_CONTRACT.generic_studio_path)
    expect(audioTaskDefaultsKey('tts')).toBe('task_defaults')
    expect(audioEndpointIsGeneric(audioTaskEndpoint('vc'))).toBe(true)
    expect(audioEndpointIsGeneric('/v1/audio/speech')).toBe(false)
    expect(audioEndpointIsGeneric('')).toBe(false)
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
    const signal = new AbortController().signal

    const { blob } = await synthesizeSpeech({
      modelId: 'tts-demo',
      input: 'hello',
      voice: 'default',
      signal,
    })

    expect(studioAudioBaseUrl()).toBe('http://studio.test/v1/audio')
    expect(fetchMock).toHaveBeenCalledWith(
      'http://studio.test/v1/audio/speech',
      expect.objectContaining({ method: 'POST', signal }),
    )
    const body = JSON.parse(fetchMock.mock.calls[0][1].body)
    expect(body).toEqual({ model: 'tts-demo', input: 'hello', voice: 'default' })
    expect(blob.size).toBe(3)
  })

  it('follows a contract-provided speech path instead of a literal', async () => {
    vi.stubGlobal('window', { location: { origin: 'http://studio.test' } })
    const payload = audioContractFixture()
    payload.operations = payload.operations.map((op) =>
      op.id === 'speech' ? { ...op, studio_path: '/v1/audio/speech-v2' } : op,
    )
    audioContract.value = payload
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      headers: { get: () => 'audio/wav' },
      blob: async () => new Blob([new Uint8Array([1])], { type: 'audio/wav' }),
    })
    vi.stubGlobal('fetch', fetchMock)

    await synthesizeSpeech({ modelId: 'tts-demo', input: 'hello' })

    expect(fetchMock.mock.calls[0][0]).toBe('http://studio.test/v1/audio/speech-v2')
  })

  it('maps a future /audioapi-prefixed operation onto the management origin', async () => {
    vi.stubGlobal('window', { location: { origin: 'http://studio.test' } })
    const payload = audioContractFixture()
    payload.operations = payload.operations.map((op) =>
      op.id === 'speech' ? { ...op, studio_path: '/audioapi/v1/audio/speech-v2' } : op,
    )
    audioContract.value = payload
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      headers: { get: () => 'audio/wav' },
      blob: async () => new Blob([new Uint8Array([1])], { type: 'audio/wav' }),
    })
    vi.stubGlobal('fetch', fetchMock)

    await synthesizeSpeech({ modelId: 'tts-demo', input: 'hello' })

    expect(fetchMock.mock.calls[0][0]).toBe('http://studio.test/v1/audio/speech-v2')
  })

  it('refuses to post a speech body through the offline generic floor', async () => {
    vi.stubGlobal('window', { location: { origin: 'http://studio.test' } })
    const fetchMock = vi.fn()
    vi.stubGlobal('fetch', fetchMock)
    resetAudioContract()

    await expect(synthesizeSpeech({ modelId: 'tts-demo', input: 'hello' })).rejects.toThrow(
      /needs the audio\.cpp integration contract/,
    )
    expect(fetchMock).not.toHaveBeenCalled()
  })

  function sseResponse(frames) {
    const encoder = new TextEncoder()
    const body = new ReadableStream({
      start(controller) {
        for (const frame of frames) controller.enqueue(encoder.encode(frame))
        controller.close()
      },
    })
    return {
      ok: true,
      headers: { get: () => 'text/event-stream; charset=utf-8' },
      body,
    }
  }

  it('streams speech SSE deltas with no buffering and assembles PCM', async () => {
    vi.stubGlobal('window', { location: { origin: 'http://studio.test' } })
    const pcmA = new Uint8Array([1, 0, 2, 0])
    const pcmB = new Uint8Array([3, 0, 4, 0])
    const b64 = (bytes) => btoa(String.fromCharCode(...bytes))
    const fetchMock = vi
      .fn()
      .mockResolvedValue(
        sseResponse([
          `data: {"type":"speech.audio.delta","audio":"${b64(pcmA)}"}\n\n`,
          `data: {"type":"speech.text.delta","delta":"Hi"}\ndata: {"type":"speech.audio.delta","audio":"${b64(pcmB)}"}\n\n`,
          'data: {"type":"speech.audio.done","timing":{"ttft_ms":42}}\n\n',
          'data: [DONE]\n\n',
        ]),
      )
    vi.stubGlobal('fetch', fetchMock)

    const frames = []
    const texts = []
    let timing = null
    const result = await streamSpeech({
      modelId: 'pocket-stream',
      input: 'hello',
      voice: 'alba',
      onAudio: (bytes, index) => frames.push([index, bytes.length]),
      onText: (delta) => texts.push(delta),
      onTiming: (value) => {
        timing = value
      },
    })

    const sent = JSON.parse(fetchMock.mock.calls[0][1].body)
    expect(fetchMock.mock.calls[0][0]).toBe('http://studio.test/v1/audio/speech')
    expect(sent).toMatchObject({
      model: 'pocket-stream',
      input: 'hello',
      voice: 'alba',
      stream: true,
      stream_format: 'sse',
      response_format: 'pcm',
    })
    expect(fetchMock.mock.calls[0][1].headers.get('Accept')).toBe('text/event-stream')
    expect(frames).toEqual([
      [0, 4],
      [1, 4],
    ])
    expect(texts).toEqual(['Hi'])
    expect(timing).toEqual({ ttft_ms: 42 })
    expect(Array.from(result.pcm)).toEqual([1, 0, 2, 0, 3, 0, 4, 0])
    expect(result.fellBack).toBe(false)
    const wav = new DataView(await result.blob.arrayBuffer())
    expect(wav.getUint32(24, true)).toBe(24000)
    expect(result.text).toBe('Hi')
  })

  it('falls back to a full blob when the engine answers speech without SSE', async () => {
    vi.stubGlobal('window', { location: { origin: 'http://studio.test' } })
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      headers: { get: () => 'audio/wav' },
      blob: async () => new Blob([new Uint8Array([82, 73, 70, 70])], { type: 'audio/wav' }),
    })
    vi.stubGlobal('fetch', fetchMock)

    const result = await streamSpeech({ modelId: 'offline-model', input: 'hi' })
    expect(result.fellBack).toBe(true)
    expect(result.blob.size).toBe(4)
  })

  it('wraps pcm bytes in a valid wav header', async () => {
    const pcm = new Uint8Array([100, 0, 200, 0])
    const blob = pcmToWavBlob(pcm, { sampleRate: 16000 })
    expect(blob.size).toBe(48)
    const view = new DataView(await blob.arrayBuffer())
    const ascii = (offset, length) =>
      String.fromCharCode(...new Uint8Array(view.buffer, offset, length))
    expect(ascii(0, 4)).toBe('RIFF')
    expect(ascii(8, 8)).toBe('WAVEfmt ')
    expect(view.getUint16(20, true)).toBe(1)
    expect(view.getUint16(22, true)).toBe(1)
    expect(view.getUint32(24, true)).toBe(16000)
    expect(view.getUint32(28, true)).toBe(32000)
    expect(ascii(36, 4)).toBe('data')
    expect(view.getUint32(40, true)).toBe(4)
  })

  it('posts multipart transcriptions to the contract transcription path', async () => {
    vi.stubGlobal('window', { location: { origin: 'http://studio.test' } })
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      headers: { get: () => 'application/json' },
      json: async () => ({ text: 'hi' }),
    })
    vi.stubGlobal('fetch', fetchMock)

    const file = new File([new Uint8Array([9])], 'memo.ogg', { type: 'audio/ogg' })
    const signal = new AbortController().signal
    const result = await transcribeAudio({
      modelId: 'asr-demo',
      file,
      language: 'en',
      signal,
    })

    expect(result).toEqual({ text: 'hi' })
    expect(fetchMock).toHaveBeenCalledWith(
      'http://studio.test/v1/audio/transcriptions',
      expect.objectContaining({ method: 'POST', signal }),
    )
    const form = fetchMock.mock.calls[0][1].body
    expect(form).toBeInstanceOf(FormData)
    expect(form.get('model')).toBe('asr-demo')
    expect(form.get('language')).toBe('en')
    expect(form.get('file').name).toBe('memo.ogg')
  })

  it('posts details requests to the contract transcriptions/details path', async () => {
    vi.stubGlobal('window', { location: { origin: 'http://studio.test' } })
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      headers: { get: () => 'application/json' },
      json: async () => ({ text: 'hi', words: [] }),
    })
    vi.stubGlobal('fetch', fetchMock)

    await transcribeAudio({
      modelId: 'asr-demo',
      file: new File([new Uint8Array([9])], 'memo.wav', { type: 'audio/wav' }),
      details: true,
    })

    expect(fetchMock.mock.calls[0][0]).toBe('http://studio.test/v1/audio/transcriptions/details')
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

  it('reads the native batch SSE shape without turning the done event into a row', () => {
    const parsed = parseBatchTranscriptionPayload(
      'text/event-stream',
      [
        'data: {"type":"batch.transcription.result","index":0,"filename":"meeting-a.wav","text":"one"}',
        'data: {"type":"batch.transcription.result","index":1,"filename":"meeting-b.wav","text":"two"}',
        'data: {"type":"batch.transcription.done","result_count":2,"timing":{"wall_ms":10}}',
        'data: [DONE]',
        '',
      ].join('\n\n'),
    )
    expect(parsed.results).toEqual([
      { name: 'meeting-a.wav', text: 'one' },
      { name: 'meeting-b.wav', text: 'two' },
    ])
  })

  it('posts generic tasks to the same-origin management /v1/tasks/run route', async () => {
    vi.stubGlobal('window', {
      location: { origin: 'http://studio.test', hostname: 'studio.test', protocol: 'http:' },
    })
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      headers: { get: () => 'application/json' },
      json: async () => ({ ok: true }),
    })
    vi.stubGlobal('fetch', fetchMock)
    const signal = new AbortController().signal

    const result = await runAudioTask({
      modelId: 'vad-demo',
      task: 'vad',
      input: { audio: '/a.wav' },
      signal,
    })

    expect(result).toEqual({ ok: true })
    // The contract's /audioapi/... path belongs to the proxy dataplane; the
    // browser management origin serves the same route as /v1/tasks/run.
    expect(fetchMock).toHaveBeenCalledWith(
      'http://studio.test/v1/tasks/run',
      expect.objectContaining({ method: 'POST', signal }),
    )
    expect(JSON.parse(fetchMock.mock.calls[0][1].body)).toEqual({
      model: 'vad-demo',
      request: { audio: '/a.wav' },
    })
  })

  it('builds a multipart body for a contract operation with multipart-form shape', async () => {
    vi.stubGlobal('window', { location: { origin: 'http://studio.test' } })
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      headers: { get: () => 'application/json' },
      json: async () => ({ words: [] }),
    })
    vi.stubGlobal('fetch', fetchMock)

    await runAudioTask({
      modelId: 'align-demo',
      task: 'align',
      input: { text: 'hello', language: 'en' },
    })

    expect(fetchMock.mock.calls[0][0]).toBe('http://studio.test/v1/audio/alignments')
    const form = fetchMock.mock.calls[0][1].body
    expect(form).toBeInstanceOf(FormData)
    expect(form.get('model')).toBe('align-demo')
    expect(form.get('text')).toBe('hello')
    expect(form.get('language')).toBe('en')
  })

  it('routes tasks from the loaded contract payload', () => {
    audioContract.value = audioContractFixture()

    expect(audioTaskEndpoint('align')).toBe('/v1/audio/alignments')
    expect(audioTaskEndpoint('tts')).toBe('/v1/audio/speech')
    expect(audioTaskEndpoint('asr')).toBe('/v1/audio/transcriptions')
    expect(audioTaskEndpoint('gen')).toBe('/audioapi/v1/tasks/run')
    expect(audioTaskRequestPath('tts')).toBe('/v1/audio/speech')
    expect(audioTaskRequestPath('asr')).toBe('/v1/audio/transcriptions')
    expect(audioTaskRequestPath('batch_transcribe')).toBe('/v1/batches/transcriptions')
    expect(audioTaskRequestPath('gen')).toBe('/v1/tasks/run')
    expect(audioTaskDefaultsKey('asr')).toBe('transcription_defaults')
    expect(audioTaskBodyShape('align')).toBe('multipart-form')
    expect(audioTaskBodyShape('vc')).toBe('tasks-run')
    // Unknown task ids fall through to the generic route like the backend.
    expect(audioTaskEndpoint('brand_new_task')).toBe('/audioapi/v1/tasks/run')
  })

  it('preserves engine input names without inventing conversion aliases', () => {
    expect(
      tasksRunRequestObject({
        source_audio: '/data/src.wav',
        target_voice: '/data/ref.wav',
      }),
    ).toEqual({
      source_audio: '/data/src.wav',
      target_voice: '/data/ref.wav',
    })
  })

  it('decodes base64 WAV task results into playable blobs', async () => {
    // Minimal RIFF/WAV header bytes → base64 "UklGRgAAAABXQVZF..."
    const wavBytes = new Uint8Array([
      0x52, 0x49, 0x46, 0x46, 0x24, 0x00, 0x00, 0x00, 0x57, 0x41, 0x56, 0x45, 0x66, 0x6d, 0x74,
      0x20,
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
