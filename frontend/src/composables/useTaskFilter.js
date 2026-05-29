import { computed, unref } from 'vue'
import { useProgressStore } from '@/stores/progress'

function normalizeTypes(type) {
  if (type == null) return null
  return Array.isArray(type) ? type : [type]
}

export function useTaskFilter(options = {}) {
  const progressStore = useProgressStore()

  const filteredTasks = computed(() => {
    const type = unref(options.type)
    const metadataKey = unref(options.metadataKey)
    const metadataValue = unref(options.metadataValue)
    const taskId = unref(options.taskId)
    const showCompleted = unref(options.showCompleted) ?? false

    const types = normalizeTypes(type)
    const allTasks = Object.values(progressStore.tasks)

    return allTasks.filter((t) => {
      const taskIdMatch = taskId == null || taskId === '' || t.task_id === taskId
      const typeMatch = !types || types.length === 0 || types.includes(t.type)
      const metadataMatch = !metadataKey || t?.metadata?.[metadataKey] === metadataValue
      const statusMatch =
        t.status === 'running'
        || (showCompleted && t.status === 'completed')
        || t.status === 'failed'
      return taskIdMatch && typeMatch && metadataMatch && statusMatch
    })
  })

  return { filteredTasks, progressStore }
}
