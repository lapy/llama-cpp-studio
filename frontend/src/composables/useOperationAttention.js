import { computed } from 'vue'
import axios from 'axios'
import { useToast } from 'primevue/usetoast'
import { useProgressStore } from '@/stores/progress'

/**
 * Interrupted/unknown operations that need attention, plus the only actions
 * this surface may take: a read-only log view and an acknowledge/dismiss for
 * work the backend classified as never started. Anything unproven stays for
 * the page that owns the original action and its confirmation contract.
 */
export function useOperationAttention() {
  const progressStore = useProgressStore()
  const toast = useToast()

  const operations = computed(() => progressStore.attentionOperations)
  const loading = computed(() => progressStore.attentionLoading)
  const error = computed(() => progressStore.attentionError)

  async function refresh() {
    try {
      return await progressStore.fetchAttentionOperations()
    } catch (failure) {
      toast.add({
        severity: 'error',
        summary: 'Could not check interrupted operations',
        detail:
          progressStore.attentionError ||
          failure?.message ||
          'The operation list could not be read.',
        life: 5000,
      })
      return null
    }
  }

  function canAcknowledge(operation) {
    return Array.isArray(operation?.actions) && operation.actions.includes('dismiss')
  }

  async function acknowledge(operation) {
    const operationId = String(operation?.operation_id || '')
    if (!operationId) return false
    try {
      await axios.post('/api/tasks/dismiss', { task_id: operationId })
      progressStore.forgetAttentionOperation(operationId)
      progressStore.removeTask(operationId)
      toast.add({
        severity: 'info',
        summary: 'Acknowledged',
        detail: 'This interrupted operation will no longer be listed.',
        life: 4000,
      })
      return true
    } catch (failure) {
      toast.add({
        severity: 'error',
        summary: 'Could not acknowledge operation',
        detail:
          failure?.response?.data?.detail ||
          failure?.message ||
          'The operation is still in progress or could not be dismissed.',
        life: 5000,
      })
      return false
    }
  }

  return {
    operations,
    loading,
    error,
    refresh,
    canAcknowledge,
    acknowledge,
    progressStore,
  }
}
