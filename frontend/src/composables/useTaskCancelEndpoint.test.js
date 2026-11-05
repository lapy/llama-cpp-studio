import { describe, it, expect } from 'vitest'
import { cancelEndpointForTask } from './useTaskCancelEndpoint.js'
import { DECLARED_CANCEL_ENDPOINTS_BY_ENGINE } from '../engines/registry.js'
import { REAL_TASK_FIXTURES } from '@/test-fixtures/taskFixtures.js'

describe('cancelEndpointForTask', () => {
  it.each(REAL_TASK_FIXTURES)(
    'routes $label to $cancelEndpoint',
    ({ task, cancelEndpoint }) => {
      expect(cancelEndpointForTask(task)).toBe(cancelEndpoint)
    },
  )

  it('returns null when cancel is unavailable', () => {
    expect(cancelEndpointForTask({ type: 'unknown' })).toBeNull()
    expect(cancelEndpointForTask(null)).toBeNull()
  })

  it('declares bespoke cancel APIs on the descriptor, not in the composable', () => {
    // R79: audio.cpp is the only engine with its own cancel endpoint.
    expect(DECLARED_CANCEL_ENDPOINTS_BY_ENGINE).toEqual({
      audio_cpp: '/api/audio-cpp/cancel',
    })
    expect(
      cancelEndpointForTask({ type: 'build', metadata: { engine: 'audio_cpp' } }),
    ).toBe('/api/audio-cpp/cancel')
    // An engine with no declaration and no generic verb falls back to the
    // type-level legacy endpoint; a custom manager has no cancel URL at all.
    expect(
      cancelEndpointForTask({ type: 'build', metadata: { engine: 'custom_engine' } }),
    ).toBe('/api/llama-versions/build-cancel')
    expect(
      cancelEndpointForTask({ type: 'build', metadata: { manager: 'custom_manager' } }),
    ).toBe('/api/llama-versions/build-cancel')
  })
})
