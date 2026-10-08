import { describe, expect, it } from 'vitest'
import { versionDeleteRetry } from './actionConfirmation'

function errorWith(detail) {
  return { response: { data: { detail } } }
}

describe('versionDeleteRetry', () => {
  it('confirms retirement of a retained launch generation', () => {
    const retry = versionDeleteRetry(errorWith({
      code: 'RETAINED_LAUNCH_REFERENCE',
      message: 'Confirm to retire those generations and delete the version.',
      confirm_operation_id: 'op-1',
      state_token: 'token-1',
    }))
    expect(retry.params).toEqual({
      retire_launch_references: true,
      confirm_operation_id: 'op-1',
      confirm_state: 'token-1',
    })
    expect(retry.message).toContain('retire')
  })

  it('sends the withheld token and still retires launch generations', () => {
    const retry = versionDeleteRetry(errorWith({
      code: 'ACTION_RETRY_WITHHELD',
      operation_id: 'op-2',
      state_token: 'token-2',
      message: 'Prior work may already have happened.',
    }))
    expect(retry.params).toEqual({
      retire_launch_references: true,
      confirm_operation_id: 'op-2',
      confirm_state: 'token-2',
    })
  })

  it('leaves an ordinary failure without another confirmation', () => {
    expect(versionDeleteRetry(errorWith('disk busy'))).toBeNull()
  })
})
