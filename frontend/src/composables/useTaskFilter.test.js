import { describe, it, expect, beforeEach } from 'vitest'
import { ref } from 'vue'
import { setActivePinia, createPinia } from 'pinia'
import { useTaskFilter } from './useTaskFilter.js'
import { useProgressStore } from '@/stores/progress'

describe('useTaskFilter', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
  })

  it('returns running, completed, and failed tasks when showCompleted is true', () => {
    const store = useProgressStore()
    store.tasks = {
      run: { task_id: 'run', type: 'download', status: 'running', progress: 10, description: 'Running' },
      done: { task_id: 'done', type: 'download', status: 'completed', progress: 100, description: 'Done' },
      fail: { task_id: 'fail', type: 'build', status: 'failed', progress: 50, description: 'Failed' },
    }

    const { filteredTasks } = useTaskFilter({ showCompleted: ref(true) })
    expect(filteredTasks.value.map((t) => t.task_id).sort()).toEqual(['done', 'fail', 'run'])
  })

  it('filters by type and metadata', () => {
    const store = useProgressStore()
    store.tasks = {
      a: {
        task_id: 'a',
        type: 'install',
        status: 'running',
        progress: 20,
        description: 'CUDA',
        metadata: { manager: 'cuda' },
      },
      b: {
        task_id: 'b',
        type: 'install',
        status: 'running',
        progress: 30,
        description: 'LMDeploy',
        metadata: { manager: 'lmdeploy' },
      },
    }

    const { filteredTasks } = useTaskFilter({
      type: ref('install'),
      metadataKey: ref('manager'),
      metadataValue: ref('cuda'),
      showCompleted: ref(false),
    })

    expect(filteredTasks.value).toHaveLength(1)
    expect(filteredTasks.value[0].task_id).toBe('a')
  })
})
