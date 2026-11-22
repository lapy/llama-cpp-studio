import { describe, expect, it } from 'vitest'
import { vramEstimatePairs, vramNotice, vramSummary } from './vramAdmission'

describe('vramEstimatePairs', () => {
  it('keeps positive rounded estimates with a model id and drops incomplete rows', () => {
    expect(
      vramEstimatePairs([
        { model: ' demo ', mib: '8192' },
        { model: 'rounds', mib: 1024.6 },
        { model: '', mib: 512 },
        { model: 'zero', mib: 0 },
        { model: 'missing', mib: null },
      ]),
    ).toEqual({ demo: 8192, rounds: 1025 })
  })

  it('tolerates missing rows', () => {
    expect(vramEstimatePairs()).toEqual({})
    expect(vramEstimatePairs([])).toEqual({})
  })
})

describe('vramSummary', () => {
  it('reports observed and estimated memory for a running generation', () => {
    expect(
      vramSummary({
        observed_mib: 8000,
        estimate_mib: 8192,
        estimate_source: 'settings',
        quality: 'observed',
      }),
    ).toBe('8,000 MiB observed · 8,192 MiB settings')
  })

  it('marks contended samples and unknown observations', () => {
    expect(vramSummary({ observed_mib: 100, quality: 'contended' })).toBe(
      '100 MiB observed · contended sample',
    )
    expect(vramSummary(null)).toBe('Not observed')
    expect(vramSummary({})).toBe('Not observed')
  })
})

describe('vramNotice', () => {
  const warned = {
    last_admission: {
      decision: 'warned',
      model: 'demo',
      estimate_mib: 16000,
      estimate_source: 'file_size',
      available_mib: 7168,
      reason: 'the weights-only bound is not conclusive',
    },
  }

  it('is shown for a warned or rejected decision only', () => {
    expect(vramNotice(warned)).toEqual({
      severity: 'warn',
      text: 'demo needs 16,000 MiB file_size, more than the 7,168 MiB available for the selected GPU(s). the weights-only bound is not conclusive',
    })
    expect(
      vramNotice({ ...warned, last_admission: { ...warned.last_admission, decision: 'rejected' } })
        .severity,
    ).toBe('error')
    for (const decision of ['admitted', 'unknown', 'skipped']) {
      expect(vramNotice({ last_admission: { ...warned.last_admission, decision } })).toBeNull()
    }
    expect(vramNotice({})).toBeNull()
    expect(vramNotice(null)).toBeNull()
  })

  it('states an unknown estimate and available memory truthfully', () => {
    expect(
      vramNotice({
        last_admission: { decision: 'warned', model: 'demo', reason: 'no estimate' },
      }).text,
    ).toBe(
      'demo needs an unknown estimate, more than the unknown MiB available for the selected GPU(s). no estimate',
    )
  })
})
