import { flushPromises, mount } from '@vue/test-utils'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { reactive } from 'vue'

const store = reactive({
  systemStatus: {
    system: {
      cpu_percent: 12,
      memory: { total: 100, used: 40, available: 60 },
      disk: { total: 100, used: 50, free: 50 },
    },
  },
  cudaStatus: {},
  gpuInfo: { gpus: [] },
  fetchSystemStatus: vi.fn().mockResolvedValue({}),
  fetchSystemStats: vi.fn().mockResolvedValue({}),
  fetchCudaStatus: vi.fn().mockResolvedValue({}),
  fetchGpuInfo: vi.fn().mockResolvedValue({}),
})

vi.mock('@/stores/engines', () => ({ useEnginesStore: () => store }))
vi.mock('primevue/usetoast', () => ({ useToast: () => ({ add: vi.fn() }) }))
vi.mock('primevue/useconfirm', () => ({ useConfirm: () => ({ require: vi.fn() }) }))

import HardwarePanel from './HardwarePanel.vue'

describe('HardwarePanel', () => {
  beforeEach(() => {
    vi.useFakeTimers()
    store.fetchSystemStatus.mockClear()
    store.fetchSystemStats.mockClear()
    store.fetchCudaStatus.mockClear()
    store.fetchGpuInfo.mockClear()
  })

  function mountPanel() {
    return mount(HardwarePanel, {
      global: {
        stubs: {
          Button: { template: '<button><slot /></button>' },
          Dialog: { template: '<div><slot /></div>' },
          Select: { template: '<select />' },
          Tag: { template: '<span><slot /></span>' },
          ProgressBar: { template: '<div class="bar" />' },
        },
      },
    })
  }

  it('refreshes system stats every second while mounted and stops on unmount', async () => {
    const wrapper = mountPanel()
    await flushPromises()
    expect(store.fetchSystemStatus).toHaveBeenCalledTimes(1)
    expect(store.fetchSystemStats).not.toHaveBeenCalled()

    await vi.advanceTimersByTimeAsync(1000)
    expect(store.fetchSystemStats).toHaveBeenCalledTimes(1)

    await vi.advanceTimersByTimeAsync(1000)
    expect(store.fetchSystemStats).toHaveBeenCalledTimes(2)

    wrapper.unmount()
    await vi.advanceTimersByTimeAsync(5000)
    expect(store.fetchSystemStats).toHaveBeenCalledTimes(2)
    vi.useRealTimers()
  })
})
