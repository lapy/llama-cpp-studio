import { describe, expect, it } from 'vitest'

import {
  cudaCompatibilityWarnings,
  gpuComputeCapability,
  minimumComputeCapability,
} from './cudaCompatibility'

describe('minimumComputeCapability', () => {
  it.each([
    ['11.8.0', 3.5],
    ['11.0', 3.5],
    ['12.9', 5.0],
    ['12.8.2', 5.0],
    ['12.7.1', 5.0],
    ['13.0', 7.5],
    ['13.5.0', 7.5],
  ])('maps CUDA %s to compute capability %s', (version, minimum) => {
    expect(minimumComputeCapability(version)).toBe(minimum)
  })

  it('returns null for release lines outside the documented map', () => {
    expect(minimumComputeCapability('10.2')).toBeNull()
    expect(minimumComputeCapability('14.0')).toBeNull()
    expect(minimumComputeCapability('99.9.9')).toBeNull()
  })

  it('returns null for missing or malformed versions', () => {
    expect(minimumComputeCapability(null)).toBeNull()
    expect(minimumComputeCapability(undefined)).toBeNull()
    expect(minimumComputeCapability('')).toBeNull()
    expect(minimumComputeCapability('latest')).toBeNull()
    expect(minimumComputeCapability('12x')).toBeNull()
  })
})

describe('gpuComputeCapability', () => {
  it('parses the detector string into a number', () => {
    expect(gpuComputeCapability({ compute_capability: '7.0' })).toBe(7)
    expect(gpuComputeCapability({ compute_capability: '8.9' })).toBe(8.9)
    expect(gpuComputeCapability({ compute_capability: '12.0' })).toBe(12)
    expect(gpuComputeCapability({ compute_capability: 7.5 })).toBe(7.5)
  })

  it('tolerates a missing or malformed capability', () => {
    expect(gpuComputeCapability({})).toBeNull()
    expect(gpuComputeCapability(null)).toBeNull()
    expect(gpuComputeCapability(undefined)).toBeNull()
    expect(gpuComputeCapability({ compute_capability: '' })).toBeNull()
    expect(gpuComputeCapability({ compute_capability: 'sm_70' })).toBeNull()
    expect(gpuComputeCapability({ compute_capability: 'unknown' })).toBeNull()
    expect(gpuComputeCapability({ compute_capability: '7.0 (approximate)' })).toBeNull()
  })
})

describe('cudaCompatibilityWarnings', () => {
  const voltaGpu = { name: 'Tesla Volta-SXM2-32GB', compute_capability: '7.0' }
  const rtx4090 = { name: 'RTX 4090', compute_capability: '8.9' }

  it('warns when a CUDA 13 version cannot target a Volta GPU', () => {
    const warnings = cudaCompatibilityWarnings(['13.0.1', '12.9'], [voltaGpu])

    expect(warnings).toHaveLength(1)
    expect(warnings[0]).toMatchObject({
      version: '13.0.1',
      minimum: 7.5,
      gpus: ['Tesla Volta-SXM2-32GB'],
    })
    expect(warnings[0].message).toBe(
      'CUDA 13.x needs Turing (7.5)+; Tesla Volta-SXM2-32GB is 7.0. Choose 12.9 or older.',
    )
  })

  it('warns when only one GPU in a mixed fleet is below the minimum', () => {
    const warnings = cudaCompatibilityWarnings(
      ['13.5.0'],
      [voltaGpu, rtx4090, { name: 'RTX 3090 Ti', compute_capability: '8.6' }],
    )

    expect(warnings).toHaveLength(1)
    expect(warnings[0].gpus).toEqual(['Tesla Volta-SXM2-32GB'])
    expect(warnings[0].message).toContain('Tesla Volta-SXM2-32GB is 7.0')
    expect(warnings[0].message).not.toContain('RTX 4090')
  })

  it('lists every incompatible GPU and omits advice when no compatible version is offered', () => {
    const warnings = cudaCompatibilityWarnings(
      ['13.0'],
      [voltaGpu, { name: 'Tesla P100-PCIE-16GB', compute_capability: '6.0' }],
    )

    expect(warnings).toHaveLength(1)
    expect(warnings[0].gpus).toEqual(['Tesla Volta-SXM2-32GB', 'Tesla P100-PCIE-16GB'])
    expect(warnings[0].message).toContain('Tesla P100-PCIE-16GB (6.0) are below that')
    expect(warnings[0].message).not.toContain('Choose')
  })

  it('does not warn when the GPUs meet the minimum', () => {
    expect(cudaCompatibilityWarnings(['12.9'], [voltaGpu])).toEqual([])
    expect(cudaCompatibilityWarnings(['13.0.1'], [rtx4090])).toEqual([])
    expect(
      cudaCompatibilityWarnings(['11.8.0'], [{ name: 'GTX 980', compute_capability: '5.2' }]),
    ).toEqual([])
  })

  it('does not warn when the GPU capability is unknown or missing', () => {
    const unknown = [
      { name: 'Tesla Volta-SXM2-32GB' },
      { name: 'Mystery GPU', compute_capability: '' },
      { name: 'Broken GPU', compute_capability: 'not-a-number' },
    ]
    expect(cudaCompatibilityWarnings(['13.0.1', '12.9'], unknown)).toEqual([])
    expect(cudaCompatibilityWarnings(['13.0.1'], [])).toEqual([])
    expect(cudaCompatibilityWarnings(['13.0.1'], undefined)).toEqual([])
  })

  it('judges the known GPUs when some capabilities are unknown', () => {
    const warnings = cudaCompatibilityWarnings(['13.0.1'], [{ name: 'Mystery GPU' }, voltaGpu])

    expect(warnings).toHaveLength(1)
    expect(warnings[0].gpus).toEqual(['Tesla Volta-SXM2-32GB'])
  })

  it('returns one warning per incompatible version, newest first and deduplicated', () => {
    const warnings = cudaCompatibilityWarnings(
      ['11.8.0', '13.0.1', '12.9', '13.0.1'],
      [{ name: 'Tesla K40', compute_capability: '3.5' }],
    )

    expect(warnings.map((warning) => warning.version)).toEqual(['13.0.1', '12.9'])
    expect(warnings.map((warning) => warning.minimum)).toEqual([7.5, 5])
  })

  it('skips versions outside the documented release lines', () => {
    expect(cudaCompatibilityWarnings(['10.2', 'latest', '13.0.1'], [voltaGpu])).toHaveLength(1)
    expect(cudaCompatibilityWarnings(['10.2', 'latest'], [voltaGpu])).toEqual([])
  })

  it('uses a stable GPU name fallback when the detector reported no name', () => {
    const warnings = cudaCompatibilityWarnings(['13.0.1'], [{ compute_capability: '7.0' }])

    expect(warnings[0].gpus).toEqual(['GPU 1'])
    expect(warnings[0].message).toContain('GPU 1 is 7.0')
  })
})
