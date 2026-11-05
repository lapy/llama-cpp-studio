import { describe, expect, it } from 'vitest'

import { lmdeployCompatibilityFor } from './lmdeployCompatibility'

describe('lmdeployCompatibilityFor', () => {
  it('shows the install quirk only when an SM70 device is present', () => {
    const compatibility = lmdeployCompatibilityFor({
      gpus: [
        { name: 'Tesla V100-SXM2-32GB', compute_capability: '7.0' },
        { name: 'RTX 3090 Ti', compute_capability: '8.6' },
      ],
    })

    expect(compatibility.show).toBe(true)
    expect(compatibility.note).toContain('Tesla V100-SXM2-32GB')
    expect(compatibility.note).toContain('runtime parameters')
    expect(compatibility.options.map((option) => option.key)).toEqual(['compat_sm70'])
  })

  it('hides the quirk on non-SM70 hardware', () => {
    const compatibility = lmdeployCompatibilityFor({
      gpus: [{ name: 'RTX 4090', compute_capability: '8.9' }],
    })

    expect(compatibility).toEqual({ show: false, options: [] })
  })

  it('hides the quirk when hardware detection has no GPUs', () => {
    expect(lmdeployCompatibilityFor({ gpus: [] })).toEqual({ show: false, options: [] })
    expect(lmdeployCompatibilityFor({})).toEqual({ show: false, options: [] })
  })
})
