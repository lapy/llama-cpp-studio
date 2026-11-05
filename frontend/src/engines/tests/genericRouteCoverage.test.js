/**
 * R103 pin: the frontend no longer depends on any legacy per-engine prefix.
 *
 * The generated backend inventory (docs/legacy-route-inventory.md) lists the
 * legacy compatibility routers. This pin asserts the FRONTEND side of the cutover
 * claim: every python-venv descriptor routes through `/api/engines/{id}/{verb}`,
 * and every native engine with a legacy module is served through
 * NATIVE_GENERIC_VERBS.
 */

import { describe, expect, it } from 'vitest'
import {
  ENGINE_DESCRIPTORS,
  NATIVE_GENERIC_VERBS,
  VENV_ENGINE_IDS,
  VENV_DESCRIPTORS,
  venvApiPath,
} from '../registry'

describe('generic route coverage', () => {
  it('routes every venv engine through the generic engine API', () => {
    expect(VENV_ENGINE_IDS.length).toBeGreaterThan(0)
    for (const engineId of VENV_ENGINE_IDS) {
      expect(ENGINE_DESCRIPTORS[engineId].genericApi, engineId).toBe(true)
      expect(venvApiPath(engineId, 'status'), engineId).toBe(
        `/api/engines/${engineId}/status`,
      )
      expect(VENV_DESCRIPTORS[engineId].apiSegment, engineId).toBeTruthy()
    }
  })

  it('serves the native engines with legacy modules through declared verbs', () => {
    for (const engineId of ['llama_cpp', 'ik_llama', 'unsloth_llama']) {
      expect(NATIVE_GENERIC_VERBS[engineId], engineId).toBeTruthy()
      expect(NATIVE_GENERIC_VERBS[engineId].length, engineId).toBeGreaterThan(0)
    }
  })

  it('keeps 1cat-vllm on the generic path despite its legacy prefix', () => {
    expect(ENGINE_DESCRIPTORS['1cat_vllm'].genericApi).toBe(true)
    expect(venvApiPath('1cat_vllm', 'install')).toBe('/api/engines/1cat_vllm/install')
  })
})
