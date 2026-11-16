import { describe, expect, it } from 'vitest'
import {
  FALLBACK_AUDIO_BUILD_DEFAULTS,
  FALLBACK_AUDIO_BUILD_TYPES,
  FALLBACK_BUILD_DEFAULTS,
  FALLBACK_BUILD_TYPES,
} from '../buildDefaults'

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

  it('carries the backend audio.cpp defaults', () => {
    expect(FALLBACK_AUDIO_BUILD_DEFAULTS).toMatchObject({
      build_type: 'RelWithDebInfo',
      cuda: false,
      cuda_graphs: true,
      llamafile: true,
      model_set: 'full',
      native_model_manager: false,
      server_frontend_modules: '',
      server_frontends_dir: 'external/audio.cpp-server-frontends',
      backend: 'cpu',
    })
    // Every upstream backend toggle exists and is pinned off by default.
    for (const backend of ['hip', 'vulkan', 'webgpu', 'sycl', 'metal']) {
      expect(FALLBACK_AUDIO_BUILD_DEFAULTS[backend]).toBe(false)
    }
    expect(Object.isFrozen(FALLBACK_AUDIO_BUILD_DEFAULTS)).toBe(true)
  })

  it('orders the audio.cpp build types with its default first', () => {
    expect(FALLBACK_AUDIO_BUILD_TYPES).toEqual(['RelWithDebInfo', 'Debug', 'Release', 'MinSizeRel'])
    expect(Object.isFrozen(FALLBACK_AUDIO_BUILD_TYPES)).toBe(true)
  })
})
