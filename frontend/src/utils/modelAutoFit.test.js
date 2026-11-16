import { describe, expect, it } from 'vitest'

import {
  AUTO_FIT_OVERRIDE_KEYS,
  buildAutoFitState,
  detectAutoFitOverrides,
  findAutoFitParam,
} from './modelAutoFit'

const canonicalParams = [
  {
    key: 'fit',
    label: 'Auto-fit',
    type: 'select',
    value_kind: 'enum',
    default: 'on',
    primary_flag: '--fit',
    flags: ['--fit'],
    options: [
      { value: 'on', label: 'on' },
      { value: 'off', label: 'off' },
    ],
    supported: true,
  },
  {
    key: 'fit_ctx',
    label: 'Fit context',
    type: 'int',
    value_kind: 'scalar',
    default: 4096,
    primary_flag: '--fit-ctx',
    flags: ['--fit-ctx'],
    supported: true,
  },
  {
    key: 'fit_target',
    label: 'Fit target',
    type: 'int',
    value_kind: 'scalar',
    default: 1024,
    primary_flag: '--fit-target',
    flags: ['--fit-target'],
    supported: true,
  },
  {
    key: 'ctx_size',
    label: 'Context size',
    type: 'int',
    value_kind: 'scalar',
    default: 0,
    primary_flag: '--ctx-size',
    flags: ['--ctx-size'],
    supported: true,
  },
  {
    key: 'n_gpu_layers',
    label: 'GPU layers',
    type: 'int',
    value_kind: 'scalar',
    default: 'auto',
    primary_flag: '--n-gpu-layers',
    flags: ['--gpu-layers', '--n-gpu-layers'],
    supported: true,
  },
  {
    key: 'flash_attn',
    label: 'Flash attention',
    type: 'select',
    value_kind: 'enum',
    default: 'auto',
    options: [
      { value: 'on', label: 'on' },
      { value: 'off', label: 'off' },
      { value: 'auto', label: 'auto' },
    ],
    supported: true,
  },
  {
    key: 'threads',
    label: 'Threads',
    type: 'int',
    value_kind: 'scalar',
    default: -1,
    supported: true,
  },
  { key: 'batch_size', label: 'Batch size', type: 'int', default: 2048, supported: true },
  { key: 'parallel', label: 'Parallel', type: 'int', default: -1, supported: true },
]

function overrides(config, parameters = canonicalParams) {
  return detectAutoFitOverrides({ parameters, config })
}

describe('findAutoFitParam', () => {
  it('detects auto-fit from the key, primary flag, or flags', () => {
    expect(findAutoFitParam(canonicalParams)?.key).toBe('fit')
    expect(
      findAutoFitParam([{ key: 'engine_fit', primary_flag: '--fit', label: 'Fit' }])?.key,
    ).toBe('engine_fit')
    expect(findAutoFitParam([{ key: 'fitter', flags: ['--fit', '-fit'], label: 'Fit' }])?.key).toBe(
      'fitter',
    )
  })

  it('returns null when the parameter list does not advertise fit', () => {
    expect(findAutoFitParam([{ key: 'ctx_size' }])).toBeNull()
    expect(findAutoFitParam([])).toBeNull()
    expect(findAutoFitParam(null)).toBeNull()
  })
})

describe('buildAutoFitState', () => {
  it('unsupported engines get no state, no overrides, and no defaults', () => {
    const state = buildAutoFitState({
      parameters: [{ key: 'ctx_size' }, { key: 'threads' }],
      config: { ctx_size: 8192, threads: 8 },
    })

    expect(state).toEqual({
      supported: false,
      fitParam: null,
      overrides: [],
      active: false,
      defaults: { fit: null, fitCtx: null, fitTarget: null },
    })
  })

  it('is active when no override defeats auto-fit', () => {
    const state = buildAutoFitState({
      parameters: canonicalParams,
      config: { ctx_size: 0, n_gpu_layers: 'auto', flash_attn: 'auto', threads: 8 },
    })

    expect(state.supported).toBe(true)
    expect(state.active).toBe(true)
    expect(state.overrides).toEqual([])
    expect(state.defaults).toEqual({ fit: 'on', fitCtx: 4096, fitTarget: 1024 })
  })
})

describe('detectAutoFitOverrides', () => {
  it('detects a non-zero context override', () => {
    expect(overrides({ ctx_size: 8192 })).toEqual([
      expect.objectContaining({
        key: 'ctx_size',
        current: 8192,
        autoLabel: 'Auto (0)',
        restore: { mode: 'set', value: 0 },
      }),
    ])
    expect(overrides({ ctx_size: 0 })).toEqual([])
    expect(overrides({ ctx_size: '0' })).toEqual([])
    expect(overrides({ ctx_size: null })).toEqual([])
    expect(overrides({ ctx_size: 'auto' })).toEqual([])
  })

  it('detects an explicit numeric GPU layer override, but not auto/all', () => {
    for (const value of [32, 0, -1, '48']) {
      expect(overrides({ n_gpu_layers: value })).toEqual([
        expect.objectContaining({
          key: 'n_gpu_layers',
          current: value,
          autoLabel: 'Auto',
          restore: { mode: 'clear' },
        }),
      ])
    }
    for (const value of ['auto', 'all', 'AUTO', null, '']) {
      expect(overrides({ n_gpu_layers: value })).toEqual([])
    }
  })

  it('ignores flash attention because --fit does not own it', () => {
    for (const value of ['on', 'off', true, 'auto', 'AUTO', null, '']) {
      expect(overrides({ flash_attn: value })).toEqual([])
    }
  })

  it('reports exactly the three auto-fit keys in a stable order', () => {
    const rows = overrides({ fit: 'off', ctx_size: 8192, n_gpu_layers: 32 })

    expect(rows.map((row) => row.key)).toEqual([...AUTO_FIT_OVERRIDE_KEYS])
    expect(rows.map((row) => row.key)).toEqual(['fit', 'ctx_size', 'n_gpu_layers'])
  })

  it('never proposes thread, batch, or parallel changes', () => {
    const rows = overrides({ threads: 8, batch_size: 512, parallel: 4 })
    expect(rows).toEqual([])
  })

  it('only considers parameters the engine actually exposes', () => {
    const parameters = canonicalParams.filter(
      (param) => !['fit', 'ctx_size', 'n_gpu_layers'].includes(param.key),
    )
    expect(overrides({ fit: 'off', ctx_size: 8192, n_gpu_layers: 32 }, parameters)).toEqual([])
  })

  it('uses auto strings only when the control accepts them', () => {
    const withStringLayers = canonicalParams.map((param) =>
      param.key === 'n_gpu_layers' ? { ...param, type: 'string' } : param,
    )
    expect(overrides({ n_gpu_layers: 32 }, withStringLayers)[0].restore).toEqual({
      mode: 'set',
      value: 'auto',
    })

    const withSelectLayers = canonicalParams.map((param) =>
      param.key === 'n_gpu_layers'
        ? {
            ...param,
            type: 'select',
            options: [
              { value: 'auto', label: 'auto' },
              { value: 'all', label: 'all' },
            ],
          }
        : param,
    )
    expect(overrides({ n_gpu_layers: 32 }, withSelectLayers)[0].restore).toEqual({
      mode: 'set',
      value: 'auto',
    })
  })

  it('restores fit to on when the control offers it', () => {
    expect(overrides({ fit: 'off' })[0].restore).toEqual({ mode: 'set', value: 'on' })
  })

  it('explains why each override defeats auto-fit without inventing numbers', () => {
    for (const row of overrides({ fit: 'off', ctx_size: 8192, n_gpu_layers: 32 })) {
      expect(row.reason.length).toBeGreaterThan(30)
      expect(row.reason).toMatch(/auto-fit/i)
      expect(row.reason).not.toMatch(/GiB|VRAM|estimate|recommend/i)
    }
  })
})
