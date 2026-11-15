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
  activateCuda: vi.fn().mockResolvedValue({}),
  fetchGpuInfo: vi.fn().mockResolvedValue({}),
})

const { toastAdd } = vi.hoisted(() => ({ toastAdd: vi.fn() }))

vi.mock('@/stores/engines', () => ({ useEnginesStore: () => store }))
vi.mock('primevue/usetoast', () => ({ useToast: () => ({ add: toastAdd }) }))
vi.mock('primevue/useconfirm', () => ({ useConfirm: () => ({ require: vi.fn() }) }))

import HardwarePanel from './HardwarePanel.vue'

describe('HardwarePanel', () => {
  beforeEach(() => {
    vi.useFakeTimers()
    store.fetchSystemStatus.mockClear()
    store.fetchSystemStats.mockClear()
    store.fetchCudaStatus.mockClear()
    store.activateCuda.mockClear()
    store.fetchGpuInfo.mockClear()
    toastAdd.mockClear()
    store.cudaStatus = {}
  })

  function mountPanel() {
    return mount(HardwarePanel, {
      global: {
        stubs: {
          Button: {
            props: ['label'],
            emits: ['click'],
            template: '<button @click="$emit(`click`)"><slot />{{ label }}</button>',
          },
          Dialog: { template: '<div><slot /></div>' },
          Select: { inheritAttrs: false, template: '<select />' },
          Tag: { template: '<span><slot /></span>' },
          ProgressBar: { template: '<div class="bar" />' },
        },
        directives: {
          tooltip: () => {},
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

  it('uses patch-aware CUDA versions supplied by the backend', async () => {
    store.cudaStatus = { available_versions: ['13.0.1', '12.8.2', '12.7.1'] }
    const wrapper = mount(HardwarePanel, {
      global: {
        stubs: {
          Button: { template: '<button><slot /></button>' },
          Dialog: { template: '<div><slot /></div>' },
          Select: {
            props: ['options'],
            template: '<div class="cuda-options">{{ options.join(",") }}</div>',
          },
          Tag: { template: '<span><slot /></span>' },
          ProgressBar: { template: '<div class="bar" />' },
        },
      },
    })
    await flushPromises()

    expect(wrapper.get('.cuda-options').text()).toBe('13.0.1,12.8.2,12.7.1')
    wrapper.unmount()
    store.cudaStatus = {}
    vi.useRealTimers()
  })

  it('activates an installed CUDA version that is not current', async () => {
    store.cudaStatus = {
      installed: true,
      version: '12.8',
      installed_versions: [
        { version: '12.8', is_current: true },
        { version: '12.8.2', is_current: false },
      ],
    }
    const wrapper = mountPanel()
    await flushPromises()

    const activate = wrapper.findAll('button').find((button) => button.text() === 'Activate')
    expect(activate).toBeTruthy()
    await activate.trigger('click')
    await flushPromises()

    expect(store.activateCuda).toHaveBeenCalledWith({ version: '12.8.2' })
  })

  it('shows the typed CUDA error message instead of the generic request text', async () => {
    store.cudaStatus = {
      installed: true,
      version: '12.8',
      installed_versions: [
        { version: '12.8', is_current: true },
        { version: '12.8.2', is_current: false },
      ],
    }
    store.activateCuda.mockRejectedValueOnce({
      message: 'Request failed with status code 404',
      response: {
        data: {
          detail: {
            code: 'cuda_version_not_installed',
            message: 'CUDA 12.8.2 installation not found in state',
          },
        },
      },
    })
    const wrapper = mountPanel()
    await flushPromises()

    const activate = wrapper.findAll('button').find((button) => button.text() === 'Activate')
    await activate.trigger('click')
    await flushPromises()

    expect(toastAdd).toHaveBeenCalledWith(
      expect.objectContaining({
        severity: 'error',
        detail: 'CUDA 12.8.2 installation not found in state',
      }),
    )
    wrapper.unmount()
    vi.useRealTimers()
  })
})
