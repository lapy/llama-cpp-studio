import { describe, expect, it } from 'vitest'
import { FALLBACK_BUILD_DEFAULTS, FALLBACK_BUILD_TYPES } from '../buildDefaults'

describe('generated build defaults', () => {
  it('carries the backend llama defaults', () => {
    // The authoritative values live in the backend llama build-option catalog;
    // regenerate with `python scripts/export-build-defaults.py`.
    expect(FALLBACK_BUILD_DEFAULTS).toMatchObject({
      build_type: 'Release',
      cuda: false,
      cuda_fa: true,
      build_server: true,
      build_examples: true,
      blas_vendor: 'OpenBLAS',
      ccache: true,
    })
    expect(Object.isFrozen(FALLBACK_BUILD_DEFAULTS)).toBe(true)
  })

  it('keeps the panel build-type order with the default first', () => {
    expect(FALLBACK_BUILD_TYPES).toEqual(['Release', 'Debug', 'RelWithDebInfo', 'MinSizeRel'])
    expect(Object.isFrozen(FALLBACK_BUILD_TYPES)).toBe(true)
  })
})
