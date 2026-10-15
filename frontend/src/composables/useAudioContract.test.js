import { afterEach, describe, expect, it, vi } from 'vitest'

import {
  audioContract,
  audioContractError,
  audioContractLoaded,
  audioEndpointIsGeneric,
  audioOperationForTask,
  audioTaskBodyShape,
  audioTaskDefaultsKey,
  audioTaskEndpoint,
  fetchAudioContract,
  OFFLINE_AUDIO_CONTRACT,
} from './useAudioContract'

function mockContractResponse(payload, ok = true, status = 200) {
  vi.stubGlobal('fetch', vi.fn().mockResolvedValue({
    ok,
    status,
    json: async () => payload,
  }))
}

describe('useAudioContract', () => {
  afterEach(() => {
    vi.unstubAllGlobals()
    audioContract.value = { ...OFFLINE_AUDIO_CONTRACT }
    audioContractLoaded.value = false
    audioContractError.value = ''
  })

  it('starts on the offline generic-route floor', () => {
    expect(audioTaskEndpoint('tts')).toBe('/audioapi/v1/tasks/run')
    expect(audioTaskDefaultsKey('tts')).toBe('task_defaults')
    expect(audioOperationForTask('tts')).toBeNull()
    expect(audioTaskBodyShape('tts')).toBe('tasks-run')
  })

  it('loads the composed contract and routes from it', async () => {
    mockContractResponse({
      schema: 'audio-cpp.studio-contract/v0',
      engine_version: 'v0.9.1',
      generic_studio_path: '/audioapi/v1/tasks/run',
      generic_native_path: '/v1/tasks/run',
      defaults_keys: { tasks_run: 'task_defaults', speech: 'speech_defaults' },
      task_routing: { tts: 'speech', vc: 'tasks_run' },
      operations: [
        { id: 'speech', studio_path: '/v1/audio/speech', request_defaults_key: 'speech_defaults', body_shape: 'openai-json' },
        { id: 'tasks_run', studio_path: '/audioapi/v1/tasks/run', request_defaults_key: 'task_defaults', body_shape: 'tasks-run' },
      ],
      drift_warnings: [],
    })
    const contract = await fetchAudioContract({ force: true })
    expect(audioContractLoaded.value).toBe(true)
    expect(contract.engine_version).toBe('v0.9.1')
    expect(audioTaskEndpoint('tts')).toBe('/v1/audio/speech')
    expect(audioTaskDefaultsKey('tts')).toBe('speech_defaults')
    expect(audioTaskBodyShape('vc')).toBe('tasks-run')
    expect(audioEndpointIsGeneric('/audioapi/v1/tasks/run')).toBe(true)
    expect(audioEndpointIsGeneric('/v1/audio/speech')).toBe(false)
  })

  it('keeps the offline floor and records the error when no contract exists (409)', async () => {
    mockContractResponse({ detail: 'No composed audio.cpp contract is available.' }, false, 409)
    await fetchAudioContract({ force: true })
    expect(audioContractLoaded.value).toBe(false)
    expect(audioContractError.value).toContain('No composed audio.cpp contract')
    expect(audioTaskEndpoint('tts')).toBe(OFFLINE_AUDIO_CONTRACT.generic_studio_path)
  })

  it('deduplicates concurrent fetches', async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => ({ engine_version: 'v-dedup' }),
    })
    vi.stubGlobal('fetch', fetchMock)
    const [a, b] = await Promise.all([fetchAudioContract(), fetchAudioContract()])
    expect(a).toBe(b)
    expect(fetchMock).toHaveBeenCalledTimes(1)
  })
})
