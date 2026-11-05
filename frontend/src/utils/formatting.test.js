import { describe, it, expect } from 'vitest'
import {
  formatBytes,
  formatBytesIEC,
  formatDate,
  formatNumber,
} from './formatting.js'

describe('formatBytes', () => {
  it('formats zero and small values', () => {
    expect(formatBytes(0)).toBe('0 B')
    expect(formatBytes(500)).toBe('500 B')
  })

  it('uses decimal SI units', () => {
    expect(formatBytes(1500)).toMatch(/1\.5 KB/)
    expect(formatBytes(2_000_000)).toMatch(/2\.0 MB/)
  })

  it('handles invalid input', () => {
    expect(formatBytes(NaN)).toBe('Unknown size')
    expect(formatBytes(null)).toBe('Unknown size')
  })
})

describe('formatBytesIEC', () => {
  it('uses base-1024 units', () => {
    expect(formatBytesIEC(1024)).toBe('1.0 KiB')
    expect(formatBytesIEC(0)).toBe('0 B')
  })
})

describe('formatDate', () => {
  it('returns Unknown for empty', () => {
    expect(formatDate(null)).toBe('Unknown')
    expect(formatDate('')).toBe('Unknown')
  })

  it('formats ISO strings', () => {
    const s = formatDate('2020-01-15T12:00:00.000Z')
    expect(s).toBeTruthy()
    expect(s).not.toBe('Unknown')
  })
})

describe('formatNumber', () => {
  it('handles nullish and NaN', () => {
    expect(formatNumber(null)).toBe('0')
    expect(formatNumber(undefined)).toBe('0')
    expect(formatNumber(NaN)).toBe('0')
  })

  it('formats integers with locale', () => {
    expect(formatNumber(1234)).toMatch(/1/)
  })
})
