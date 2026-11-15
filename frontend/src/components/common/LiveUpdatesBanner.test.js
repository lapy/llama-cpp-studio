import { beforeEach, describe, expect, it, vi } from 'vitest'
import { mount } from '@vue/test-utils'
import { nextTick, reactive } from 'vue'
import { LIVE_UPDATES_PAUSED_DELAY_MS } from '@/utils/liveUpdates'

const retryConnection = vi.fn()
const progressStore = reactive({
  connectionState: 'live',
  hasActiveTasks: false,
  disconnectedSince: null,
  retryConnection,
})

vi.mock('@/stores/progress', () => ({
  useProgressStore: () => progressStore,
}))

import LiveUpdatesBanner from './LiveUpdatesBanner.vue'

function mountBanner() {
  return mount(LiveUpdatesBanner)
}

describe('LiveUpdatesBanner', () => {
  beforeEach(() => {
    retryConnection.mockReset()
    progressStore.connectionState = 'live'
    progressStore.hasActiveTasks = false
    progressStore.disconnectedSince = null
    vi.useFakeTimers()
    vi.setSystemTime(new Date('2026-11-15T12:00:00Z'))
  })

  it('stays hidden while live, even with active tasks', async () => {
    progressStore.hasActiveTasks = true
    progressStore.disconnectedSince = Date.now() - 60_000
    const wrapper = mountBanner()

    await vi.advanceTimersByTimeAsync(15_000)
    await nextTick()

    expect(wrapper.find('.live-updates-banner').exists()).toBe(false)
    expect(wrapper.text()).not.toContain('Live updates paused')
    wrapper.unmount()
  })

  it('stays hidden while disconnected but without active tasks', async () => {
    progressStore.connectionState = 'connecting'
    progressStore.disconnectedSince = Date.now()
    const wrapper = mountBanner()

    await vi.advanceTimersByTimeAsync(LIVE_UPDATES_PAUSED_DELAY_MS + 5_000)
    await nextTick()

    expect(wrapper.find('.live-updates-banner').exists()).toBe(false)
    wrapper.unmount()
  })

  it('appears after the threshold with active tasks and offers Retry and Dismiss', async () => {
    progressStore.connectionState = 'connecting'
    progressStore.hasActiveTasks = true
    progressStore.disconnectedSince = Date.now()
    const wrapper = mountBanner()

    await vi.advanceTimersByTimeAsync(LIVE_UPDATES_PAUSED_DELAY_MS - 1)
    await nextTick()
    expect(wrapper.find('.live-updates-banner').exists()).toBe(false)

    await vi.advanceTimersByTimeAsync(1)
    await nextTick()

    const banner = wrapper.get('.live-updates-banner')
    expect(banner.attributes('role')).toBe('status')
    expect(banner.attributes('aria-live')).toBe('polite')
    expect(banner.text()).toContain('Live updates paused — retrying. Tasks may still be running.')
    expect(banner.findAll('button').map((button) => button.text())).toEqual(['Retry', 'Dismiss'])

    await banner.findAll('button')[0].trigger('click')
    expect(retryConnection).toHaveBeenCalledTimes(1)
    wrapper.unmount()
  })

  it('dismisses until the next successful reconnect', async () => {
    progressStore.connectionState = 'connecting'
    progressStore.hasActiveTasks = true
    progressStore.disconnectedSince = Date.now()
    const wrapper = mountBanner()

    await vi.advanceTimersByTimeAsync(LIVE_UPDATES_PAUSED_DELAY_MS)
    await nextTick()
    await wrapper.get('.live-updates-banner__dismiss').trigger('click')
    expect(wrapper.find('.live-updates-banner').exists()).toBe(false)

    await vi.advanceTimersByTimeAsync(30_000)
    expect(wrapper.find('.live-updates-banner').exists()).toBe(false)

    progressStore.connectionState = 'live'
    await nextTick()
    progressStore.connectionState = 'connecting'
    progressStore.disconnectedSince = Date.now()
    await vi.advanceTimersByTimeAsync(LIVE_UPDATES_PAUSED_DELAY_MS)
    await nextTick()

    expect(wrapper.find('.live-updates-banner').exists()).toBe(true)
    wrapper.unmount()
  })

  it('clears immediately when the connection comes back', async () => {
    progressStore.connectionState = 'connecting'
    progressStore.hasActiveTasks = true
    progressStore.disconnectedSince = Date.now()
    const wrapper = mountBanner()

    await vi.advanceTimersByTimeAsync(LIVE_UPDATES_PAUSED_DELAY_MS)
    await nextTick()
    expect(wrapper.find('.live-updates-banner').exists()).toBe(true)

    progressStore.connectionState = 'live'
    await nextTick()

    expect(wrapper.find('.live-updates-banner').exists()).toBe(false)
    wrapper.unmount()
  })
})
