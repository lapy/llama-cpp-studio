import { computed, ref } from 'vue'
import { defineStore } from 'pinia'
import { applyStorageCleanup, fetchStorageInventory, previewStorageCleanup } from '@/api/storage'

function failureMessage(error, fallback) {
  const detail = error?.response?.data?.detail
  if (typeof detail === 'string') return detail
  if (detail && typeof detail.message === 'string') return detail.message
  return fallback
}

function outcomeStatus(result) {
  if (result.outcome === 'completed') {
    return 'Cleanup completed. The selected storage was removed.'
  }
  if (result.outcome === 'partial') {
    return 'Cleanup was partial. Some units were removed and the rest stayed.'
  }
  return 'Nothing was removed. Resolve the listed refusals and preview again.'
}

function isInventoryPayload(value) {
  return (
    value &&
    typeof value === 'object' &&
    value.filesystem &&
    typeof value.filesystem === 'object' &&
    value.scan &&
    typeof value.scan === 'object' &&
    Array.isArray(value.categories)
  )
}

export const useStorageStore = defineStore('storage', () => {
  const inventory = ref(null)
  const loading = ref(false)
  const error = ref('')
  const status = ref('')
  const previewing = ref(false)
  const applying = ref(false)
  const plan = ref(null)
  const result = ref(null)

  const busy = computed(() => loading.value || previewing.value || applying.value)
  const categories = computed(() =>
    Array.isArray(inventory.value?.categories) ? inventory.value.categories : [],
  )
  const measuredBytes = computed(() =>
    categories.value.reduce((total, category) => total + Number(category.bytes || 0), 0),
  )

  async function loadInventory({ refresh = false } = {}) {
    if (loading.value) return
    loading.value = true
    error.value = ''
    try {
      const payload = await fetchStorageInventory({ refresh })
      if (payload == null) {
        inventory.value = null
        return
      }
      if (!isInventoryPayload(payload)) {
        throw new Error('Storage inventory response was invalid')
      }
      inventory.value = payload
    } catch (requestError) {
      error.value = failureMessage(
        requestError,
        'Storage usage could not be loaded. Retry the scan.',
      )
    } finally {
      loading.value = false
    }
  }

  async function previewCleanup(category, unitIds = [], keepHistory = null) {
    if (!category || previewing.value || applying.value) return
    previewing.value = true
    error.value = ''
    status.value = ''
    result.value = null
    const payload = { category }
    if (Array.isArray(unitIds) && unitIds.length) payload.unit_ids = unitIds
    if (keepHistory != null) payload.keep_history = keepHistory
    try {
      plan.value = await previewStorageCleanup(payload)
      status.value = plan.value.notice
    } catch (requestError) {
      plan.value = null
      error.value = failureMessage(
        requestError,
        'A cleanup plan could not be prepared for that category.',
      )
    } finally {
      previewing.value = false
    }
  }

  async function applyPlan() {
    if (!plan.value || applying.value) return
    applying.value = true
    error.value = ''
    status.value = ''
    try {
      const applied = await applyStorageCleanup(plan.value.plan_id)
      result.value = applied
      plan.value = null
      status.value = outcomeStatus(applied)
      await loadInventory({ refresh: true })
    } catch (requestError) {
      const code = requestError?.response?.data?.detail?.code
      if (code === 'PLAN_STALE') {
        plan.value = null
        await loadInventory({ refresh: true })
        error.value =
          'The storage inventory changed after the preview. Refresh and review the cleanup again.'
      } else {
        error.value = failureMessage(
          requestError,
          'The cleanup outcome could not be established. Refresh the storage inventory before trying again.',
        )
      }
    } finally {
      applying.value = false
    }
  }

  function clearPlan() {
    if (applying.value) return
    plan.value = null
    status.value = ''
  }

  function clearResult() {
    result.value = null
  }

  return {
    inventory,
    loading,
    error,
    status,
    previewing,
    applying,
    plan,
    result,
    busy,
    categories,
    measuredBytes,
    loadInventory,
    previewCleanup,
    applyPlan,
    clearPlan,
    clearResult,
  }
})
