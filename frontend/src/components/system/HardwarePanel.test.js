import { flushPromises, mount } from '@vue/test-utils'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { reactive } from 'vue'

const { toastAdd, confirmRequire } = vi.hoisted(() => ({
  toastAdd: vi.fn(),
  confirmRequire: vi.fn(),
}))

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
  cleanupCudaInstallers: vi.fn().mockResolvedValue({ freed_bytes: 0, removed: 0 }),
  fetchGpuInfo: vi.fn().mockResolvedValue({}),
})

vi.mock('@/stores/engines', () => ({ useEnginesStore: () => store }))
vi.mock('primevue/usetoast', () => ({ useToast: () => ({ add: toastAdd }) }))
vi.mock('primevue/useconfirm', () => ({ useConfirm: () => ({ require: confirmRequire }) }))

import HardwarePanel from './HardwarePanel.vue'

describe('HardwarePanel', () => {
  beforeEach(() => {
    vi.useFakeTimers()
    store.fetchSystemStatus.mockClear()
    store.fetchSystemStats.mockClear()
    store.fetchCudaStatus.mockClear()
    store.activateCuda.mockClear()
    store.cleanupCudaInstallers.mockReset()
    store.cleanupCudaInstallers.mockResolvedValue({ freed_bytes: 0, removed: 0 })
    store.fetchGpuInfo.mockClear()
    store.cudaStatus = {}
    store.gpuInfo = { gpus: [] }
    toastAdd.mockClear()
    confirmRequire.mockClear()
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

  it('renders one hardware page section with a refresh action', async () => {
    const wrapper = mountPanel()
    await flushPromises()

    const section = wrapper.get('#hardware')
    expect(section.classes()).toContain('page-section')
    expect(section.get('.page-section__title').text()).toBe('Hardware resources')
    const refresh = section
      .get('.page-section__actions')
      .findAll('button')
      .find((button) => button.text() === 'Refresh')
    expect(refresh).toBeTruthy()

    await refresh.trigger('click')
    await flushPromises()
    expect(store.fetchSystemStatus).toHaveBeenCalledTimes(2)
    wrapper.unmount()
    vi.useRealTimers()
  })

  it('offers a retry when part of the hardware refresh fails', async () => {
    store.fetchSystemStatus.mockRejectedValueOnce(new Error('offline'))
    const wrapper = mountPanel()
    await flushPromises()

    const banner = wrapper.get('.state-banner--warning')
    expect(banner.text()).toContain('Some hardware information could not be refreshed')
    const retry = banner.findAll('button').find((button) => button.text() === 'Retry')
    await retry.trigger('click')
    await flushPromises()

    expect(store.fetchSystemStatus).toHaveBeenCalledTimes(2)
    expect(wrapper.find('.state-banner--warning').exists()).toBe(false)
    wrapper.unmount()
    vi.useRealTimers()
  })

  it('renders the metric summary and GPU memory', async () => {
    store.gpuInfo = {
      gpus: [{ index: 0, name: 'RTX 4090', memory: { used: 512, total: 1024 } }],
    }
    const wrapper = mountPanel()
    await flushPromises()

    expect(wrapper.text()).toContain('CPU, memory and disk')
    expect(wrapper.text()).toContain('12.0%')
    expect(wrapper.text()).toContain('40 B')
    expect(wrapper.text()).toContain('(40%)')
    expect(wrapper.text()).toContain('(50%)')
    expect(wrapper.get('#hardware-gpu-heading').text()).toBe('GPU memory')
    expect(wrapper.text()).toContain('512 B / 1.0 KiB VRAM')
    wrapper.unmount()
    vi.useRealTimers()
  })

  it('renders the CUDA toolkit as a subsection with its actions', async () => {
    const wrapper = mountPanel()
    await flushPromises()

    const heading = wrapper.get('#cuda-toolkit-heading')
    expect(heading.element.tagName).toBe('H3')
    expect(heading.text()).toBe('CUDA Toolkit')
    expect(heading.classes()).toContain('page-subheading')
    expect(wrapper.findAll('button').map((button) => button.text())).toContain('Install CUDA')
    wrapper.unmount()
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

  it('renders installed version sizes and the installer cache size', async () => {
    store.cudaStatus = {
      installed: true,
      version: '12.8',
      installed_versions: [{ version: '12.8', is_current: true, size_bytes: 1536 }],
      installer_cache: { bytes: 5 * 1024 * 1024, files: 2, directory: '/tmp/cuda_installers' },
    }
    const wrapper = mountPanel()
    await flushPromises()

    expect(wrapper.get('.version-size').text()).toBe('1.5 KiB')
    expect(wrapper.get('.cuda-toolkit-cache__label').text()).toContain(
      'Installer cache: 5.0 MiB (2 files)',
    )
    expect(
      wrapper.findAll('button').some((button) => button.text() === 'Clean installer cache'),
    ).toBe(true)
    wrapper.unmount()
    vi.useRealTimers()
  })

  it('confirms before cleaning the installer cache and refreshes on success', async () => {
    store.cudaStatus = {
      installed_versions: [],
      installer_cache: { bytes: 2048, files: 1, directory: '/tmp/cuda_installers' },
    }
    store.cleanupCudaInstallers.mockImplementation(async () => {
      store.cudaStatus = {
        ...store.cudaStatus,
        installer_cache: { bytes: 0, files: 0, directory: '/tmp/cuda_installers' },
      }
      return { freed_bytes: 2048, removed: 1 }
    })
    const wrapper = mountPanel()
    await flushPromises()

    const clean = wrapper
      .findAll('button')
      .find((button) => button.text() === 'Clean installer cache')
    await clean.trigger('click')
    await flushPromises()

    // Clicking only asks for confirmation; nothing is deleted yet.
    expect(store.cleanupCudaInstallers).not.toHaveBeenCalled()
    expect(confirmRequire).toHaveBeenCalledTimes(1)

    const options = confirmRequire.mock.calls[0][0]
    expect(options.header).toBe('Clean Installer Cache')
    await options.accept()
    await flushPromises()

    expect(store.cleanupCudaInstallers).toHaveBeenCalledTimes(1)
    expect(wrapper.find('.cuda-toolkit-cache').exists()).toBe(false)
    expect(toastAdd).toHaveBeenCalledWith(expect.objectContaining({ severity: 'success' }))
    wrapper.unmount()
    vi.useRealTimers()
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

  it('warns that a CUDA 13 install cannot target a Volta GPU', async () => {
    store.cudaStatus = { available_versions: ['13.0.1', '12.9'] }
    store.gpuInfo = {
      gpus: [{ name: 'Tesla Volta-SXM2-32GB', compute_capability: '7.0' }],
    }
    const wrapper = mountPanel()
    await flushPromises()

    const warnings = wrapper.findAll('.cuda-compat-warning')
    expect(warnings).toHaveLength(1)
    expect(warnings[0].attributes('role')).toBe('status')
    expect(warnings[0].text()).toContain('CUDA 13.x needs Turing (7.5)+')
    expect(warnings[0].text()).toContain('Tesla Volta-SXM2-32GB is 7.0')
    expect(warnings[0].text()).toContain('Choose 12.9 or older.')

    wrapper.unmount()
    vi.useRealTimers()
  })

  it('warns on an installed CUDA version without blocking activation', async () => {
    store.cudaStatus = {
      installed: true,
      version: '13.0.1',
      installed_versions: [{ version: '13.0.1', is_current: false }],
    }
    store.gpuInfo = {
      gpus: [{ name: 'Tesla Volta-SXM2-32GB', compute_capability: '7.0' }],
    }
    const wrapper = mountPanel()
    await flushPromises()

    const rowWarning = wrapper.get('.ev-version-row .cuda-compat-warning')
    expect(rowWarning.text()).toContain('CUDA 13.x needs Turing (7.5)+')
    // The warning is guidance: the row still offers activation.
    expect(wrapper.findAll('button').some((button) => button.text() === 'Activate')).toBe(true)

    wrapper.unmount()
    vi.useRealTimers()
  })

  it('does not warn when the GPU meets the minimum for the offered versions', async () => {
    store.cudaStatus = { available_versions: ['12.9'] }
    store.gpuInfo = { gpus: [{ name: 'Tesla Volta-SXM2-32GB', compute_capability: '7.0' }] }
    const wrapper = mountPanel()
    await flushPromises()

    expect(wrapper.find('.cuda-compat-warning').exists()).toBe(false)

    wrapper.unmount()
    vi.useRealTimers()
  })

  it('does not warn when the GPU is newer than the CUDA 13 minimum', async () => {
    store.cudaStatus = { available_versions: ['13.0.1'] }
    store.gpuInfo = { gpus: [{ name: 'RTX 4090', compute_capability: '8.9' }] }
    const wrapper = mountPanel()
    await flushPromises()

    expect(wrapper.find('.cuda-compat-warning').exists()).toBe(false)

    wrapper.unmount()
    vi.useRealTimers()
  })

  it('does not warn when the GPU compute capability is unknown', async () => {
    store.cudaStatus = { available_versions: ['13.0.1'] }
    store.gpuInfo = { gpus: [{ name: 'Unknown GPU' }] }
    const wrapper = mountPanel()
    await flushPromises()

    expect(wrapper.find('.cuda-compat-warning').exists()).toBe(false)

    wrapper.unmount()
    vi.useRealTimers()
  })
})
