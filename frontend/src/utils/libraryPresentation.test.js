import { describe, expect, it } from 'vitest'

import {
  buildLibraryLines,
  formatLibraryBytes,
  formatLibraryDate,
  groupEngine,
  groupIsRunning,
  groupTitle,
  groupTotalFileSize,
  isStandaloneGroup,
  lineIsRunning,
  primaryQuant,
  quantStatus,
} from './libraryPresentation'

describe('library presentation', () => {
  it('formats bytes with decimal units and dates as relative days', () => {
    expect(formatLibraryBytes(0)).toBe('')
    expect(formatLibraryBytes(1500)).toBe('1.5 KB')
    expect(formatLibraryBytes(4000)).toBe('4.0 KB')
    expect(formatLibraryDate('')).toBe('')
    expect(formatLibraryDate(new Date(Date.now() - 2 * 86400000).toISOString())).toMatch(
      /2 days ago/,
    )
  })

  it('derives group identity, engine and disk totals', () => {
    const group = {
      huggingface_id: 'org/model',
      base_model_name: 'Model',
      quantizations: [
        { id: 'a', file_size: 1000, config: { engine: 'llama.cpp' } },
        { id: 'b', file_size: 2500 },
        { id: 'c' },
      ],
    }
    expect(groupTitle(group)).toBe('Model')
    expect(groupTitle({ huggingface_id: 'org/only' })).toBe('org/only')
    expect(primaryQuant(group)).toBe(group.quantizations[0])
    expect(primaryQuant({ quantizations: [] })).toBeNull()
    expect(groupEngine(group)).toBe('llama.cpp')
    expect(groupEngine({ quantizations: [{ id: 'x' }] })).toBe('')
    expect(groupTotalFileSize(group)).toBe(3500)
    expect(groupTotalFileSize({})).toBe(0)
  })

  it('detects standalone cards and running quantizations', () => {
    expect(isStandaloneGroup({ quantizations: [{ id: 'one' }] })).toBe(true)
    expect(isStandaloneGroup({ quantizations: [] })).toBe(false)
    expect(isStandaloneGroup(null)).toBe(false)
    expect(
      isStandaloneGroup({
        quantizations: [
          { id: 'a', format: 'gguf' },
          { id: 'b', format: 'gguf' },
        ],
      }),
    ).toBe(false)
    expect(
      isStandaloneGroup({
        quantizations: [
          { id: 'a', format: 'safetensors' },
          { id: 'b', format: 'safetensors' },
        ],
      }),
    ).toBe(true)
    expect(
      isStandaloneGroup({
        quantizations: [
          { id: 'a', artifact: { package_kind: 'prepared_bundle' } },
          { id: 'b', artifact: { package_kind: 'builtin' } },
        ],
      }),
    ).toBe(true)

    expect(quantStatus({ status: 'READY' })).toBe('ready')
    expect(quantStatus({ run_state: 'loading' })).toBe('loading')
    expect(lineIsRunning({ is_active: true })).toBe(true)
    expect(lineIsRunning({ status: 'ready' })).toBe(true)
    expect(lineIsRunning({ status: 'stopped' })).toBe(false)
    expect(lineIsRunning({ status: 'ready', runtime_quality: 'unreachable' })).toBe(false)
    expect(groupIsRunning({ quantizations: [{ status: 'stopped' }, { status: 'loading' }] })).toBe(
      true,
    )
    expect(groupIsRunning({})).toBe(false)
  })

  it('flattens grouped and standalone models into list rows', () => {
    const groups = [
      {
        huggingface_id: 'org/group',
        base_model_name: 'Group Model',
        quantizations: [
          {
            id: 'g-q4',
            quantization: 'Q4_K_M',
            file_size: 1500,
            is_active: false,
            status: 'stopped',
          },
          { id: 'g-q8', quantization: 'Q8_0', file_size: 2500, is_active: true, status: 'ready' },
        ],
      },
      {
        huggingface_id: 'org/single',
        base_model_name: 'Single Model',
        quantizations: [{ id: 's1', file_size: 1000, is_active: false, status: 'stopped' }],
      },
    ]

    const lines = buildLibraryLines(groups)

    expect(lines.map((line) => [line.kind, line.key, line.title])).toEqual([
      ['group', 'group:org/group', 'Group Model'],
      ['variant', 'g-q4', 'Q4_K_M'],
      ['variant', 'g-q8', 'Q8_0'],
      ['single', 's1', 'Single Model'],
    ])
    expect(lines[0]).toMatchObject({ size: '4.0 KB', running: true, quant: null })
    expect(lines[1]).toMatchObject({ size: '1.5 KB', running: false })
    expect(lines[2]).toMatchObject({ size: '2.5 KB', running: true, status: 'Ready' })
    expect(lines[3].meta).toContain('org/single')
  })
})
