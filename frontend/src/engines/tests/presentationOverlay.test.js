import { describe, expect, it } from 'vitest'

import { presentationForEngine, ENGINE_DESCRIPTORS } from '../registry'

describe('presentationForEngine', () => {
  it('falls back to the declared ui_kind for a server-only engine', () => {
    const presentation = presentationForEngine('brand_new_engine', [
      { id: 'brand_new_engine', label: 'Brand New', ui_kind: 'native' },
    ])
    expect(presentation.label).toBe('Brand New')
    expect(presentation.generic).toBe(true)
    expect(presentation.card.subtitle).toBe('Native GGUF server.')
    expect(presentation.card.mark.icon).toBe('pi pi-microchip')
  })

  it('uses the generic mark when the server has no ui_kind', () => {
    const presentation = presentationForEngine('another_engine', [
      { id: 'another_engine', label: 'Another' },
    ])
    expect(presentation.label).toBe('Another')
    expect(presentation.icon).toBe('pi-server')
    expect(presentation.generic).toBe(true)
  })

  it('keeps the local descriptor for a shipped engine', () => {
    expect(presentationForEngine('llama_cpp', [])).toBe(ENGINE_DESCRIPTORS.llama_cpp)
  })
})
