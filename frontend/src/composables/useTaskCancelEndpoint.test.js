import { describe, it, expect } from 'vitest'
import { cancelEndpointForTask } from './useTaskCancelEndpoint.js'
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
})
