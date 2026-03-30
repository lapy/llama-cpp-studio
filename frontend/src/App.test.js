import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest'
import { mount, flushPromises } from '@vue/test-utils'
import { reactive } from 'vue'

const toastAdd = vi.fn()
const initTheme = vi.fn()
const fetchSystemStatus = vi.fn()
const fetchSwapConfigStale = vi.fn()
const connect = vi.fn()
const disconnect = vi.fn()
const afterEachHook = vi.fn()
const removeAfterEach = vi.fn()
const subscribe = vi.fn()
const unsubscribeTaskUpdated = vi.fn()
const unsubscribeNotifications = vi.fn()

const systemStore = reactive({
  systemStatus: { proxy_status: { healthy: true } },
  fetchSystemStatus,
  fetchSwapConfigStale,
})

const progressStore = reactive({
  isConnected: true,
  connect,
  disconnect,
  subscribe,
})

let routeHook = null
const subscriptions = new Map()

vi.mock('primevue/usetoast', () => ({
  useToast: () => ({ add: toastAdd }),
}))

vi.mock('vue-router', () => ({
  useRouter: () => ({
    afterEach: afterEachHook,
  }),
}))

vi.mock('@/stores/engines', () => ({
  useEnginesStore: () => systemStore,
}))

vi.mock('@/stores/progress', () => ({
  useProgressStore: () => progressStore,
}))

vi.mock('@/composables/useTheme', () => ({
  useTheme: () => ({ initTheme }),
}))

import App from './App.vue'

function mountApp() {
  return mount(App, {
    global: {
      stubs: {
        ConfirmDialog: { template: '<div class="confirm-dialog-stub" />' },
        Toast: { template: '<div class="toast-stub" />' },
        AppHeader: {
          props: ['llamaSwapStatus'],
          template: '<div class="header-stub">{{ llamaSwapStatus?.healthy ? "healthy" : "offline" }}</div>',
        },
        AppNavigation: { template: '<div class="nav-stub" />' },
        AppFooter: { template: '<div class="footer-stub" />' },
        RouterView: { template: '<div class="route-view-stub" />' },
      },
    },
  })
}

describe('App', () => {
  let visibilityState = 'visible'

  beforeEach(() => {
    toastAdd.mockReset()
    initTheme.mockReset()
    fetchSystemStatus.mockReset()
    fetchSwapConfigStale.mockReset()
    connect.mockReset()
    disconnect.mockReset()
    afterEachHook.mockReset()
    removeAfterEach.mockReset()
    subscribe.mockReset()
    unsubscribeTaskUpdated.mockReset()
    unsubscribeNotifications.mockReset()
    subscriptions.clear()

    visibilityState = 'visible'
    Object.defineProperty(document, 'visibilityState', {
      configurable: true,
      get: () => visibilityState,
    })

    systemStore.systemStatus = { proxy_status: { healthy: true } }
    fetchSystemStatus.mockResolvedValue(undefined)
    fetchSwapConfigStale.mockResolvedValue({ applicable: true, stale: false })
    afterEachHook.mockImplementation((cb) => {
      routeHook = cb
      return removeAfterEach
    })
    subscribe.mockImplementation((eventType, cb) => {
      subscriptions.set(eventType, cb)
      return eventType === 'task_updated'
        ? unsubscribeTaskUpdated
        : unsubscribeNotifications
    })
  })

  afterEach(() => {
    routeHook = null
  })

  it('boots the app, reacts to refresh triggers, and cleans up subscriptions', async () => {
    const wrapper = mountApp()
    await flushPromises()

    expect(initTheme).toHaveBeenCalledTimes(1)
    expect(connect).toHaveBeenCalledTimes(1)
    expect(fetchSystemStatus).toHaveBeenCalledTimes(1)
    expect(fetchSwapConfigStale).toHaveBeenCalledTimes(1)
    expect(wrapper.text()).toContain('healthy')

    routeHook()
    subscriptions.get('task_updated')({ status: 'completed' })
    document.dispatchEvent(new Event('visibilitychange'))
    await flushPromises()

    expect(fetchSwapConfigStale).toHaveBeenCalledTimes(4)

    subscriptions.get('notification')({
      title: 'Config updated',
      message: 'Applied cleanly',
      type: 'warning',
    })
    await flushPromises()

    expect(toastAdd).toHaveBeenCalledWith(
      expect.objectContaining({
        severity: 'warn',
        summary: 'Config updated',
        detail: 'Applied cleanly',
      }),
    )

    wrapper.unmount()

    expect(unsubscribeTaskUpdated).toHaveBeenCalledTimes(1)
    expect(unsubscribeNotifications).toHaveBeenCalledTimes(1)
    expect(removeAfterEach).toHaveBeenCalledTimes(1)
    expect(disconnect).toHaveBeenCalledTimes(1)

    fetchSwapConfigStale.mockClear()
    document.dispatchEvent(new Event('visibilitychange'))
    expect(fetchSwapConfigStale).not.toHaveBeenCalled()
  })

  it('shows a toast when the initial status refresh fails and maps danger notifications to error', async () => {
    fetchSystemStatus.mockRejectedValue(new Error('status down'))

    mountApp()
    await flushPromises()

    expect(toastAdd).toHaveBeenCalledWith(
      expect.objectContaining({
        severity: 'error',
        summary: 'Failed to refresh system status',
        detail: 'status down',
      }),
    )

    subscriptions.get('notification')({
      summary: 'Proxy failed',
      detail: 'llama-swap exited',
      notification_type: 'danger',
    })
    await flushPromises()

    expect(toastAdd).toHaveBeenLastCalledWith(
      expect.objectContaining({
        severity: 'error',
        summary: 'Proxy failed',
        detail: 'llama-swap exited',
      }),
    )
  })
})
