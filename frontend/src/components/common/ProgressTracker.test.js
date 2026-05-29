import { describe, it, expect, beforeEach, vi } from 'vitest'
import { mount, flushPromises } from '@vue/test-utils'
import ProgressTracker from './ProgressTracker.vue'

describe('ProgressTracker', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('passes task-id through to TaskDetailPanel for sync-style tracking', async () => {
    const wrapper = mount(ProgressTracker, {
      props: {
        taskId: 'build_sync_source-main_1700000000',
        sectionTitle: 'Sync progress',
        showCompleted: true,
        dismissible: false,
      },
      global: {
        stubs: {
          TaskDetailPanel: {
            props: ['taskId', 'sectionTitle', 'showCompleted', 'dismissible'],
            template: '<div data-testid="detail-panel">{{ taskId }}</div>',
          },
        },
      },
    })
    await flushPromises()

    expect(wrapper.find('[data-testid="detail-panel"]').text()).toBe(
      'build_sync_source-main_1700000000',
    )
  })
})
