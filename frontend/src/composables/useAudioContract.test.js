import { afterEach, describe, expect, it, vi } from 'vitest'

import {
  AUDIO_CONTRACT_PATH,
  AUDIO_CONTRACT_SCHEMA,
  audioContract,
  audioContractError,
  audioContractLoaded,
  audioContractSchemaMismatch,
  audioContractSchema,
  audioContractStatus,
  audioEndpointIsGeneric,
  audioOperationForTask,
  audioTaskBodyShape,
  audioTaskDefaultsKey,
  audioTaskEndpoint,
  audioTaskRequestPath,
  fetchAudioContract,
  managementAudioPath,
  OFFLINE_AUDIO_CONTRACT,
  resetAudioContract,
} from './useAudioContract'
import { audioContractFixture } from '@/test-fixtures/audioContract'

function mockJsonResponse(payload, { ok = true, status = 200 } = {}) {
  return vi.fn().mockResolvedValue({
    ok,
    status,
    json: async () => payload,
  })
}

describe('useAudioContract', () => {
  afterEach(() => {
    vi.unstubAllGlobals()
    resetAudioContract()
  })

  it('starts on the offline generic-route floor', () => {
    expect(audioTaskEndpoint('tts')).toBe(OFFLINE_AUDIO_CONTRACT.generic_studio_path)
    expect(audioTaskRequestPath('tts')).toBe('/v1/tasks/run')
    expect(audioTaskDefaultsKey('tts')).toBe('task_defaults')
    expect(audioOperationForTask('tts')).toBeNull()
    expect(audioTaskBodyShape('tts')).toBe('tasks-run')
    expect(audioContractLoaded.value).toBe(false)
  })

  it('maps dataplane /audioapi paths onto same-origin management paths', () => {
    audioContract.value = audioContractFixture()

    expect(audioTaskEndpoint('gen')).toBe('/audioapi/v1/tasks/run')
    expect(audioTaskRequestPath('gen')).toBe('/v1/tasks/run')
    expect(audioTaskRequestPath('brand_new_task')).toBe('/v1/tasks/run')
    // Every dedicated route is already a management path and must not change.
    expect(audioTaskRequestPath('tts')).toBe('/v1/audio/speech')
    expect(audioTaskRequestPath('asr')).toBe('/v1/audio/transcriptions')
    expect(audioTaskRequestPath('transcribe_details')).toBe('/v1/audio/transcriptions/details')
    expect(audioTaskRequestPath('batch_transcribe')).toBe('/v1/batches/transcriptions')
    expect(audioTaskRequestPath('align')).toBe('/v1/audio/alignments')
    expect(managementAudioPath('/audioapi/v1/audio/future')).toBe('/v1/audio/future')
    expect(managementAudioPath('/v1/audio/speech')).toBe('/v1/audio/speech')
  })

  it('loads the composed contract and routes from it', async () => {
    const payload = audioContractFixture()
    vi.stubGlobal('fetch', mockJsonResponse(payload))

    const contract = await fetchAudioContract({ force: true })

    expect(audioContractLoaded.value).toBe(true)
    expect(contract.engine_version).toBe('v0.9.5')
    expect(contract.schema).toBe(AUDIO_CONTRACT_SCHEMA)
    expect(audioTaskEndpoint('tts')).toBe('/v1/audio/speech')
    expect(audioTaskEndpoint('asr')).toBe('/v1/audio/transcriptions')
    expect(audioTaskEndpoint('clon')).toBe('/v1/audio/speech')
    expect(audioTaskEndpoint('gen')).toBe('/audioapi/v1/tasks/run')
    expect(audioTaskDefaultsKey('tts')).toBe('speech_defaults')
    expect(audioTaskDefaultsKey('asr')).toBe('transcription_defaults')
    expect(audioTaskBodyShape('vc')).toBe('tasks-run')
    expect(audioTaskBodyShape('transcribe_details')).toBe('multipart-form')
    expect(audioEndpointIsGeneric('/audioapi/v1/tasks/run')).toBe(true)
    expect(audioEndpointIsGeneric('/v1/tasks/run')).toBe(true)
    expect(audioEndpointIsGeneric('/v1/audio/speech')).toBe(false)
  })

  it('requests the composed contract with the session auth header', async () => {
    const fetchMock = mockJsonResponse(audioContractFixture())
    vi.stubGlobal('fetch', fetchMock)
    await fetchAudioContract({ force: true })

    expect(fetchMock).toHaveBeenCalledWith(
      AUDIO_CONTRACT_PATH,
      expect.objectContaining({ credentials: 'include' }),
    )
    expect(fetchMock.mock.calls[0][1].headers.get('Authorization')).toBe('Bearer local')
  })

  it('honors operations[].serves_tasks when task_routing is incomplete', () => {
    audioContract.value = audioContractFixture({ taskRouting: {} })
    expect(audioTaskEndpoint('tts')).toBe('/v1/audio/speech')
    expect(audioTaskBodyShape('asr')).toBe('multipart-form')
    expect(audioTaskDefaultsKey('asr')).toBe('transcription_defaults')
  })

  it('keeps the offline floor and records a schema mismatch', async () => {
    const mismatchedSchema = `${AUDIO_CONTRACT_SCHEMA}-v99`
    vi.stubGlobal(
      'fetch',
      mockJsonResponse({ ...audioContractFixture(), schema: mismatchedSchema }),
    )

    await fetchAudioContract({ force: true })

    expect(audioContractLoaded.value).toBe(false)
    expect(audioContractSchemaMismatch.value).toBe(true)
    expect(audioContractSchema.value).toBe(mismatchedSchema)
    expect(audioContractStatus.value).toBe(200)
    expect(audioContractError.value).toContain('schema mismatch')
    expect(audioTaskEndpoint('tts')).toBe(OFFLINE_AUDIO_CONTRACT.generic_studio_path)
  })

  it('keeps the offline floor and the 409 status when no contract exists', async () => {
    vi.stubGlobal(
      'fetch',
      mockJsonResponse(
        { detail: 'No composed audio.cpp contract is available.' },
        { ok: false, status: 409 },
      ),
    )

    await fetchAudioContract({ force: true })

    expect(audioContractLoaded.value).toBe(false)
    expect(audioContractStatus.value).toBe(409)
    expect(audioContractError.value).toContain('No composed audio.cpp contract')
    expect(audioTaskEndpoint('tts')).toBe(OFFLINE_AUDIO_CONTRACT.generic_studio_path)
  })

  it('distinguishes a 500 from a transport failure', async () => {
    vi.stubGlobal(
      'fetch',
      mockJsonResponse({ detail: 'contract composer crashed' }, { ok: false, status: 500 }),
    )
    await fetchAudioContract({ force: true })
    expect(audioContractStatus.value).toBe(500)
    expect(audioContractError.value).toBe('contract composer crashed')

    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new TypeError('Failed to fetch')))
    await fetchAudioContract({ force: true })
    expect(audioContractStatus.value).toBe(0)
    expect(audioContractError.value).toBe('Failed to fetch')
  })

  it('drops the previous composed routing when a forced refetch fails', async () => {
    vi.stubGlobal('fetch', mockJsonResponse(audioContractFixture()))
    await fetchAudioContract({ force: true })
    expect(audioTaskEndpoint('tts')).toBe('/v1/audio/speech')

    vi.stubGlobal(
      'fetch',
      mockJsonResponse({ detail: 'no active version' }, { ok: false, status: 409 }),
    )
    await fetchAudioContract({ force: true })

    // Stale task_routing/operations must not linger after the failed reload.
    expect(audioContractLoaded.value).toBe(false)
    expect(audioContractStatus.value).toBe(409)
    expect(audioContract.value.operations).toEqual([])
    expect(audioContract.value.task_routing).toEqual({})
    expect(audioTaskEndpoint('tts')).toBe(OFFLINE_AUDIO_CONTRACT.generic_studio_path)
  })

  it('deduplicates concurrent fetches', async () => {
    const fetchMock = mockJsonResponse(audioContractFixture({ engine_version: 'v-dedup' }))
    vi.stubGlobal('fetch', fetchMock)
    const [a, b] = await Promise.all([fetchAudioContract(), fetchAudioContract()])
    expect(a).toBe(b)
    expect(fetchMock).toHaveBeenCalledTimes(1)
  })

  it('resets every singleton between tests', async () => {
    vi.stubGlobal('fetch', mockJsonResponse(audioContractFixture()))
    await fetchAudioContract({ force: true })
    expect(audioContractLoaded.value).toBe(true)

    resetAudioContract()

    expect(audioContractLoaded.value).toBe(false)
    expect(audioContract.value.operations).toEqual([])
    expect(audioContractError.value).toBe('')
    expect(audioContractStatus.value).toBe(0)
    expect(audioContractSchemaMismatch.value).toBe(false)
    expect(audioTaskEndpoint('tts')).toBe(OFFLINE_AUDIO_CONTRACT.generic_studio_path)
  })
})
