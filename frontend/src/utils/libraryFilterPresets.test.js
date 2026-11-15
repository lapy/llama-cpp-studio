import { describe, expect, it } from 'vitest'
import {
  LIBRARY_PRESETS_KEY,
  LIBRARY_PRESETS_VERSION,
  MAX_LIBRARY_PRESETS,
  deletePreset,
  describePresetFilters,
  loadPresets,
  normalizePresetFilters,
  parsePresets,
  savePreset,
} from './libraryFilterPresets'

function memoryStorage(initial = {}) {
  const data = new Map(Object.entries(initial))
  return {
    getItem: (key) => (data.has(key) ? data.get(key) : null),
    setItem: (key, value) => data.set(key, String(value)),
    removeItem: (key) => data.delete(key),
    read: (key) => data.get(key),
  }
}

const FILTERS = {
  q: '  llama  ',
  status: 'attention',
  engine: ' llama.cpp ',
  task: 'text-generation',
  sort: 'recent',
}

describe('library filter presets', () => {
  it('saves normalized filters under a versioned key', () => {
    const storage = memoryStorage()
    const { preset, presets } = savePreset('Needs attention', FILTERS, {
      storage,
      id: 'preset-1',
      now: '2026-11-15T00:00:00.000Z',
    })

    expect(preset).toEqual({
      id: 'preset-1',
      name: 'Needs attention',
      filters: {
        q: 'llama',
        status: 'attention',
        engine: 'llama.cpp',
        task: 'text-generation',
        sort: 'recent',
      },
      createdAt: '2026-11-15T00:00:00.000Z',
    })
    expect(presets).toHaveLength(1)
    expect(JSON.parse(storage.read(LIBRARY_PRESETS_KEY))).toEqual({
      version: LIBRARY_PRESETS_VERSION,
      presets,
    })
    expect(loadPresets(storage)).toEqual(presets)
  })

  it('defaults invalid filters and drops unknown fields', () => {
    expect(
      normalizePresetFilters({
        q: 'x'.repeat(500),
        status: 'broken',
        engine: '',
        task: 42,
        sort: 'newest',
        layout: 'list',
      }),
    ).toEqual({
      q: 'x'.repeat(200),
      status: 'all',
      engine: 'all',
      task: 'all',
      sort: 'name',
    })
  })

  it('tolerates malformed JSON, foreign versions and invalid entries', () => {
    expect(parsePresets('{not json')).toEqual([])
    expect(parsePresets(JSON.stringify({ version: 999, presets: [{ name: 'Old' }] }))).toEqual([])
    expect(parsePresets(JSON.stringify({ version: 1, presets: 'nope' }))).toEqual([])
    const parsed = parsePresets(
      JSON.stringify({
        version: 1,
        presets: [
          null,
          { name: '  ' },
          { name: 'Good', filters: { status: 'running' } },
          { name: 'good', filters: { status: 'stopped' } },
          { name: 'Broken', filters: null },
        ],
      }),
    )
    expect(parsed.map((preset) => preset.name)).toEqual(['Good', 'Broken'])
    expect(parsed[0].filters.status).toBe('running')
    expect(parsed[1].filters.status).toBe('all')
    expect(parsed.every((preset) => typeof preset.id === 'string' && preset.id)).toBe(true)
  })

  it('rejects empty and duplicate names without writing', () => {
    const storage = memoryStorage()
    expect(() => savePreset('   ', FILTERS, { storage })).toThrow('Enter a name for the preset.')
    savePreset('Focused', FILTERS, { storage })
    expect(() => savePreset(' focused ', FILTERS, { storage })).toThrow('already exists')
    expect(loadPresets(storage)).toHaveLength(1)
  })

  it('caps the number of stored presets', () => {
    const storage = memoryStorage()
    for (let index = 0; index < MAX_LIBRARY_PRESETS; index += 1) {
      savePreset(`Preset ${index}`, FILTERS, { storage })
    }
    expect(() => savePreset('One too many', FILTERS, { storage })).toThrow('up to 20 presets')
  })

  it('deletes by id and leaves unrelated presets alone', () => {
    const storage = memoryStorage()
    savePreset('Keep', FILTERS, { storage, id: 'keep' })
    savePreset('Drop', FILTERS, { storage, id: 'drop' })

    expect(deletePreset('drop', { storage }).map((preset) => preset.name)).toEqual(['Keep'])
    expect(loadPresets(storage).map((preset) => preset.name)).toEqual(['Keep'])
    expect(deletePreset('missing', { storage }).map((preset) => preset.name)).toEqual(['Keep'])
  })

  it('reports unavailable or full storage instead of losing a save silently', () => {
    expect(
      loadPresets({
        getItem: () => {
          throw new Error('blocked')
        },
      }),
    ).toEqual([])
    expect(() =>
      savePreset('No storage', FILTERS, {
        storage: {
          getItem: () => null,
          setItem: () => {
            throw new Error('quota')
          },
        },
      }),
    ).toThrow('Browser storage may be full')
  })

  it('describes restored filters for the preset list', () => {
    expect(describePresetFilters(FILTERS)).toBe(
      'Name: llama · Status: Needs attention · Engine: llama.cpp · Task: text-generation · Sort: Recent use',
    )
    expect(describePresetFilters({})).toBe('No filters')
  })

  it('uses the browser storage when none is injected', () => {
    localStorage.removeItem(LIBRARY_PRESETS_KEY)
    savePreset('Browser', FILTERS)
    expect(loadPresets().map((preset) => preset.name)).toEqual(['Browser'])
    localStorage.removeItem(LIBRARY_PRESETS_KEY)
  })
})
