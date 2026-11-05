import { describe, expect, it } from 'vitest'
import { groupRuntimeState, modelRuntimeState } from './modelPresentation'

describe('runtime presentation', () => {
  it.each(['stale', 'unreachable', 'unknown'])(
    'never presents a %s observation as ready',
    (quality) => {
      const state = modelRuntimeState({
        status: 'ready',
        is_active: true,
        runtime_quality: quality,
      })
      expect(state.actionable).toBe(false)
      expect(state.label).not.toBe('Ready')
      expect(state.severity).toBe('warn')
    },
  )
  it('surfaces incomplete observations ahead of a ready variant in a group', () => {
    expect(
      groupRuntimeState({
        quantizations: [{ status: 'ready', is_active: true }, { runtime_quality: 'unreachable' }],
      }).key,
    ).toBe('unknown')
  })
  it('distinguishes loading, ready and a stopped model', () => {
    expect(modelRuntimeState({ status: 'loading', is_active: true }).actionable).toBe(false)
    expect(modelRuntimeState({ status: 'ready', is_active: true }).actionable).toBe(true)
    expect(modelRuntimeState({ is_active: false }).label).toBe('Stopped')
  })
})
