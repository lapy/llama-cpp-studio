import { beforeEach, describe, expect, it, vi } from 'vitest'
import { mount } from '@vue/test-utils'
import { nextTick, reactive } from 'vue'

const route = reactive({ hash: '' })

vi.mock('vue-router', () => ({
  useRoute: () => route,
}))

vi.mock('axios', () => ({
  default: {
    get: vi.fn().mockResolvedValue({ data: [] }),
    post: vi.fn().mockResolvedValue({ data: {} }),
    defaults: {},
    interceptors: {
      request: { use: vi.fn() },
      response: { use: vi.fn() },
    },
  },
}))

const retryConnection = vi.fn()
const progressStore = reactive({
  isConnected: true,
  connectionState: 'live',
  reconnectAttempt: 0,
  lastEventAt: null,
  retryConnection,
})

const systemStore = reactive({
  systemStatus: {
    proxy_status: { health_observed_at: new Date(Date.now() - 12_000).toISOString() },
    runtime_observation: {
      quality: 'unreachable',
      observed_at: null,
      detail: 'No successful running-model observation yet.',
    },
    persistence: {
      saturated: true,
      pending_store_writes: 32,
      max_pending_store_writes: 32,
      latest_failure: {
        code: 'STORE_WRITE_FAILED',
        committed: false,
        message: 'disk full hf_FOOTERSECRET',
        description: 'hf_FOOTERSECRET',
      },
    },
  },
})

vi.mock('@/stores/progress', () => ({
  useProgressStore: () => progressStore,
}))

vi.mock('@/stores/engines', () => ({
  useEnginesStore: () => systemStore,
}))

import AppFooter from './AppFooter.vue'

function mountFooter() {
  return mount(AppFooter, {
    global: { stubs: { RouterLink: { props: ['to'], template: '<a :href="to"><slot /></a>' } } },
  })
}

describe('AppFooter', () => {
  beforeEach(() => {
    vi.useFakeTimers()
    vi.setSystemTime(new Date('2026-11-15T12:00:00Z'))
    retryConnection.mockReset()
    progressStore.isConnected = true
    progressStore.connectionState = 'live'
    progressStore.reconnectAttempt = 0
    progressStore.lastEventAt = null
  })

  it('keeps the footer to service status and shows a live age without a link', async () => {
    progressStore.lastEventAt = Date.now() - 2_000
    const wrapper = mountFooter()

    expect(wrapper.text()).toContain('Live · updated 2s ago')
    expect(wrapper.text()).toContain('Needs attention')
    expect(wrapper.find('a').exists()).toBe(false)
    expect(wrapper.find('#config-backup').exists()).toBe(false)
    expect(wrapper.find('button').exists()).toBe(false)
    expect(wrapper.text()).not.toContain('hf_FOOTERSECRET')

    await vi.advanceTimersByTimeAsync(3_000)
    await nextTick()
    expect(wrapper.text()).toContain('Live · updated 5s ago')
    wrapper.unmount()
  })

  it('shows the reconnect attempt and retries on demand', async () => {
    progressStore.isConnected = false
    progressStore.connectionState = 'connecting'
    progressStore.reconnectAttempt = 3
    progressStore.lastEventAt = Date.now() - 8_000
    const wrapper = mountFooter()

    expect(wrapper.text()).toContain('Reconnecting (attempt 3)…')
    const retry = wrapper.get('button')
    expect(retry.text()).toBe('Retry')
    expect(retry.attributes('aria-label')).toBe('Retry live updates')
    expect(wrapper.get('.sr-only').attributes('aria-live')).toBe('polite')
    expect(wrapper.get('.sr-only').text()).toBe('Live updates reconnecting')

    await retry.trigger('click')
    expect(retryConnection).toHaveBeenCalledTimes(1)
    wrapper.unmount()
  })

  it('offers an explicit offline retry after repeated failures', async () => {
    progressStore.isConnected = false
    progressStore.connectionState = 'offline'
    progressStore.reconnectAttempt = 5
    progressStore.lastEventAt = Date.now() - 61_000
    const wrapper = mountFooter()

    expect(wrapper.get('button').text()).toBe('Offline — Retry')
    expect(wrapper.get('.live-status .pi').classes()).toContain('pi-exclamation-circle')
    expect(wrapper.get('.sr-only').text()).toBe('Live updates offline. Retry available.')

    await wrapper.get('button').trigger('click')
    expect(retryConnection).toHaveBeenCalledTimes(1)
    wrapper.unmount()
  })
})
