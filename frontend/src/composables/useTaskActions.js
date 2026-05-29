import { ref } from 'vue'
import { useToast } from 'primevue/usetoast'
import axios from 'axios'
import { useProgressStore } from '@/stores/progress'
import { cancelEndpointForTask } from '@/composables/useTaskCancelEndpoint'

export function useTaskActions() {
  const progressStore = useProgressStore()
  const toast = useToast()
  const stopTaskId = ref(null)

  function canStopTask(task) {
    return task?.status === 'running' && Boolean(cancelEndpointForTask(task))
  }

  async function requestStopTask(task) {
    const id = task?.task_id
    const endpoint = cancelEndpointForTask(task)
    if (!id || !endpoint) return
    stopTaskId.value = id
    try {
      const res = await axios.post(endpoint, { task_id: id })
      if (res?.data?.ok === false) {
        toast.add({
          severity: 'warn',
          summary: 'Stop task',
          detail: res?.data?.message || 'Could not cancel this task.',
          life: 5000,
        })
        return
      }
      toast.add({
        severity: 'info',
        summary: 'Stop requested',
        detail: res?.data?.message || 'The task will be stopped shortly.',
        life: 4000,
      })
    } catch (e) {
      toast.add({
        severity: 'error',
        summary: 'Stop task failed',
        detail: e?.response?.data?.detail || e?.message || 'Request failed',
        life: 5000,
      })
    } finally {
      stopTaskId.value = null
    }
  }

  function getTaskLogs(task) {
    return progressStore.getTaskLogs(task?.task_id)
  }

  function dismissTask(taskId, expandedLogsRef, logPreEls) {
    if (!taskId) return
    progressStore.removeTask(taskId)
    if (expandedLogsRef?.value) {
      const { [taskId]: _, ...restExp } = expandedLogsRef.value
      expandedLogsRef.value = restExp
    }
    if (logPreEls) delete logPreEls[taskId]
  }

  return {
    stopTaskId,
    canStopTask,
    requestStopTask,
    getTaskLogs,
    dismissTask,
    progressStore,
  }
}
