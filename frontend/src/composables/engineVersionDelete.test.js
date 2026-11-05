import { describe, expect, it } from 'vitest'
import { activeVersionDeletePlan } from './engineVersionDelete'

describe('activeVersionDeletePlan', () => {
  it('blocks the active version when a model selects that engine', () => {
    const plan = activeVersionDeletePlan({
      version: 'main',
      is_active: true,
      selected_models: ['Demo'],
      dormant_models: [],
    })
    expect(plan.allowed).toBe(false)
    expect(plan.message).toContain('Demo')
  })

  it('allows the active version when saved settings belong to another engine', () => {
    const plan = activeVersionDeletePlan({
      version: 'main',
      is_active: true,
      selected_models: [],
      dormant_models: [{ name: 'Demo', engine: 'ik_llama' }],
    })
    expect(plan.allowed).toBe(true)
    expect(plan.message).toContain('ik_llama.cpp')
    expect(plan.message).toContain('Delete it anyway?')
  })

  it('allows an unused active version', () => {
    const plan = activeVersionDeletePlan({
      version: 'main',
      is_active: true,
      selected_models: [],
      dormant_models: [],
    })
    expect(plan.allowed).toBe(true)
    expect(plan.message).toContain('No model is using this engine')
  })
})
