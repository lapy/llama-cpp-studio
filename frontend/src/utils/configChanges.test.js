import { describe, it, expect } from 'vitest'
import { buildConfigChanges, countConfigChanges } from './configChanges'

describe('configChanges', () => {
  it('groups routing, runtime, and engine changes with readable values', () => {
    const current = {
      engine: 'test_engine',
      model_alias: 'new-name',
      set_params_by_id: [{ sub_id: 'thinking', params: { thinking_effort: 'low' } }],
      runtime_env: { CUDA_VISIBLE_DEVICES: '0', NEW: '1' },
      engines: {
        test_engine: {
          model_alias: 'new-name',
          set_params_by_id: [{ sub_id: 'thinking', params: { thinking_effort: 'low' } }],
          runtime_env: { CUDA_VISIBLE_DEVICES: '0', NEW: '1' },
          ctx_size: 16384,
          temperature: 0.2,
        },
      },
    }
    const saved = {
      engine: 'test_engine',
      model_alias: 'old-name',
      set_params_by_id: [],
      runtime_env: { CUDA_VISIBLE_DEVICES: '0' },
      engines: {
        test_engine: {
          model_alias: 'old-name',
          set_params_by_id: [],
          runtime_env: { CUDA_VISIBLE_DEVICES: '0' },
          ctx_size: 8192,
          temperature: 0.7,
        },
      },
    }
    const groups = buildConfigChanges(current, saved, { engineLabel: 'Test engine' })
    expect(groups.map((group) => group.id)).toEqual(['routing', 'runtime', 'params'])
    expect(countConfigChanges(groups)).toBe(5)
    const routing = groups[0].rows
    expect(routing.find((row) => row.key === 'model_alias')).toMatchObject({
      before: 'old-name',
      after: 'new-name',
      kind: 'changed',
    })
    const aliases = routing.find((row) => row.key === 'set_params_by_id')
    expect(aliases.before).toBe('None')
    expect(aliases.after).toContain('thinking: thinking_effort=low')
    const env = groups[1].rows.find((row) => row.key === 'runtime_env')
    expect(env.after).toContain('NEW=1')
    expect(groups[2].label).toBe('Engine settings · Test engine')
  })

  it('shows added and removed parameters against the default state', () => {
    const current = { engine: 'test_engine', engines: { test_engine: { ctx_size: 4096 } } }
    const saved = { engine: 'test_engine', engines: { test_engine: { temperature: 0.5 } } }
    const rows = buildConfigChanges(current, saved)[0].rows
    expect(rows.map((row) => [row.key, row.kind, row.before, row.after])).toEqual([
      ['ctx_size', 'added', 'Default', '4096'],
      ['temperature', 'removed', '0.5', 'Default'],
    ])
  })
})
