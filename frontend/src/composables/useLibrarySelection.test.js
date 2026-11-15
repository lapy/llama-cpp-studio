import { describe, expect, it } from 'vitest'
import { effectScope, nextTick, ref } from 'vue'

import {
  pruneSelection,
  selectionSummary,
  useLibrarySelection,
} from '@/composables/useLibrarySelection'

function setupSelection({ visible = ['a', 'b', 'c'], filterKey = '' } = {}) {
  const visibleIds = ref(visible)
  const key = ref(filterKey)
  const scope = effectScope()
  const selection = scope.run(() => useLibrarySelection({ visibleIds, filterKey: key }))
  return { selection, visibleIds, filterKey: key, stop: () => scope.stop() }
}

describe('library selection', () => {
  it('summarises select-all and indeterminate states from visible models only', () => {
    expect(selectionSummary(new Set(), ['a', 'b'])).toEqual({
      count: 0,
      all: false,
      some: false,
      visible: 2,
    })
    expect(selectionSummary(new Set(['a']), ['a', 'b'])).toMatchObject({
      count: 1,
      all: false,
      some: true,
    })
    expect(selectionSummary(new Set(['a', 'b']), ['a', 'b'])).toMatchObject({
      count: 2,
      all: true,
      some: false,
    })
    expect(selectionSummary(new Set(['a', 'b']), [])).toMatchObject({
      count: 0,
      all: false,
      some: false,
    })
    expect(pruneSelection(new Set(['a', 'gone']), ['a', 'b'])).toEqual(new Set(['a']))
  })

  it('toggles one model, selects all visible and clears when all are selected again', async () => {
    const { selection, stop } = setupSelection()

    selection.toggleSelectAll()
    expect(selection.summary.value).toMatchObject({ count: 3, all: true, some: false })

    selection.toggle('b')
    expect(selection.summary.value).toMatchObject({ count: 2, all: false, some: true })
    expect(selection.isSelected('b')).toBe(false)

    selection.toggleSelectAll()
    expect(selection.summary.value).toMatchObject({ count: 3, all: true })

    selection.toggleSelectAll()
    expect(selection.selectedIds.value).toEqual([])
    await nextTick()
    stop()
  })

  it('drops selections whose models disappear from the visible list', async () => {
    const { selection, visibleIds, stop } = setupSelection()
    selection.toggleSelectAll()

    visibleIds.value = ['a', 'c']
    await nextTick()
    expect(selection.selectedIds.value.sort()).toEqual(['a', 'c'])
    expect(selection.summary.value).toMatchObject({ count: 2, all: true })

    visibleIds.value = []
    await nextTick()
    expect(selection.selectedIds.value).toEqual([])
    stop()
  })

  it('clears the selection whenever the filter key changes', async () => {
    const { selection, filterKey, stop } = setupSelection()
    selection.toggleSelectAll()

    filterKey.value = 'attention|all|all|all'
    await nextTick()
    expect(selection.selectedIds.value).toEqual([])
    expect(selection.summary.value.all).toBe(false)
    stop()
  })

  it('deduplicates visible ids and ignores blank model ids', () => {
    const { selection, stop } = setupSelection({ visible: ['a', 'a', '', null, 'b'] })

    expect(selection.visibleIds.value).toEqual(['a', 'b'])
    selection.toggle('')
    expect(selection.selectedIds.value).toEqual([])
    selection.clearSelection()
    expect(selection.selectedIds.value).toEqual([])
    stop()
  })
})
