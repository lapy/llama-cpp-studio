import { computed, ref, watch } from 'vue'

function selectionId(value) {
  return String(value ?? '')
}

/**
 * Count selected models that are currently visible. Header controls use `all`
 * for checked and `some` for the indeterminate state.
 */
export function selectionSummary(selectedIds, visibleIds) {
  const selected = new Set([...(selectedIds || [])].map(selectionId))
  const visible = [...new Set([...(visibleIds || [])].map(selectionId).filter(Boolean))]
  const count = visible.reduce((total, id) => total + (selected.has(id) ? 1 : 0), 0)
  return {
    count,
    all: visible.length > 0 && count === visible.length,
    some: count > 0 && count < visible.length,
    visible: visible.length,
  }
}

/** Keep only selected ids that still exist in the visible model list. */
export function pruneSelection(selectedIds, visibleIds) {
  const visible = new Set([...(visibleIds || [])].map(selectionId))
  return new Set([...(selectedIds || [])].map(selectionId).filter((id) => visible.has(id)))
}

/**
 * Multi-select state for the model library.
 *
 * Selection is cleared when the filter key changes and pruned when visible
 * models disappear, so bulk actions can only ever target models the user can
 * currently see.
 */
export function useLibrarySelection({ visibleIds, filterKey } = {}) {
  const selected = ref(new Set())
  const visible = computed(() => {
    const values = visibleIds?.value ?? []
    return [...new Set(values.map(selectionId).filter(Boolean))]
  })
  const summary = computed(() => selectionSummary(selected.value, visible.value))
  const selectedIds = computed(() => [...selected.value])

  function isSelected(id) {
    return selected.value.has(selectionId(id))
  }

  function toggle(id) {
    const key = selectionId(id)
    if (!key) return
    const next = new Set(selected.value)
    if (next.has(key)) next.delete(key)
    else next.add(key)
    selected.value = next
  }

  function toggleSelectAll() {
    if (!visible.value.length) return
    selected.value = summary.value.all ? new Set() : new Set(visible.value)
  }

  function clearSelection() {
    if (selected.value.size) selected.value = new Set()
  }

  watch(visible, (ids) => {
    const pruned = pruneSelection(selected.value, ids)
    if (pruned.size !== selected.value.size) selected.value = pruned
  })

  if (filterKey) {
    watch(filterKey, () => {
      selected.value = new Set()
    })
  }

  return {
    selectedIds,
    visibleIds: visible,
    summary,
    isSelected,
    toggle,
    toggleSelectAll,
    clearSelection,
  }
}
