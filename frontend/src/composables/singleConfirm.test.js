import { describe, expect, it, vi } from 'vitest'

import { requireSingleConfirmation } from './singleConfirm'

describe('requireSingleConfirmation', () => {
  it('suppresses repeated requests until the active confirmation closes', () => {
    const confirm = { require: vi.fn() }

    expect(requireSingleConfirmation(confirm, { message: 'Delete it?' })).toBe(true)
    expect(requireSingleConfirmation(confirm, { message: 'Delete it?' })).toBe(false)
    expect(confirm.require).toHaveBeenCalledOnce()

    confirm.require.mock.calls[0][0].onHide()

    expect(requireSingleConfirmation(confirm, { message: 'A different action?' })).toBe(true)
    expect(confirm.require).toHaveBeenCalledTimes(2)
    confirm.require.mock.calls[1][0].onHide()
  })

  it('releases the guard before running accept', async () => {
    const accept = vi.fn()
    const confirm = { require: vi.fn() }
    requireSingleConfirmation(confirm, { message: 'Delete it?', accept })

    await confirm.require.mock.calls[0][0].accept()

    expect(accept).toHaveBeenCalledOnce()
    expect(requireSingleConfirmation(confirm, { message: 'Next action?' })).toBe(true)
    confirm.require.mock.calls[1][0].onHide()
  })
})
