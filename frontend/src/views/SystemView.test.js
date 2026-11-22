import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { enableAutoUnmount, flushPromises, mount } from '@vue/test-utils'
import { reactive } from 'vue'
import { createMemoryHistory, createRouter } from 'vue-router'

const systemStore = reactive({
  systemStatus: {
    persistence: {
      latest_failure: { code: 'STORE_WRITE_FAILED', committed: false, message: 'SECRET' },
    },
  },
  fetchSystemStatus: vi.fn().mockResolvedValue({}),
})
vi.mock('@/stores/engines', () => ({ useEnginesStore: () => systemStore }))

const retryConnection = vi.fn()
const progressStore = reactive({
  isConnected: true,
  connectionState: 'live',
  reconnectAttempt: 0,
  lastEventAt: null,
  retryConnection,
})
vi.mock('@/stores/progress', () => ({ useProgressStore: () => progressStore }))

import SystemView from './SystemView.vue'

enableAutoUnmount(afterEach)

const PANEL_STUBS = {
  Button: {
    props: ['label', 'disabled', 'loading'],
    template: '<button :disabled="disabled">{{ label }}</button>',
  },
  HardwarePanel: { template: '<div class="hardware-panel-stub">Hardware panel</div>' },
  StoragePanel: { template: '<div class="storage-panel-stub">Storage panel</div>' },
  ConfigBackupPanel: { template: '<div class="backup-panel-stub">Backup panel</div>' },
  ConfigHistoryPanel: { template: '<div class="history-panel-stub">History panel</div>' },
  NeedsAttentionPanel: {
    template: '<div class="attention-panel-stub">Needs attention panel</div>',
  },
}

async function createTestRouter(path = '/system') {
  const router = createRouter({
    history: createMemoryHistory(),
    routes: [
      { path: '/system', name: 'system', component: { template: '<div />' } },
      { path: '/:pathMatch(.*)*', component: { template: '<div />' } },
    ],
  })
  await router.push(path)
  await router.isReady()
  return router
}

async function mountSystem(path = '/system') {
  const router = await createTestRouter(path)
  const wrapper = mount(SystemView, {
    global: {
      plugins: [router],
      stubs: PANEL_STUBS,
    },
  })
  await flushPromises()
  return { wrapper, router }
}

async function goTo(router, path) {
  await router.push(path)
  await flushPromises()
}

beforeEach(() => {
  systemStore.fetchSystemStatus.mockClear()
  retryConnection.mockReset()
  progressStore.isConnected = true
  progressStore.connectionState = 'live'
  progressStore.reconnectAttempt = 0
  progressStore.lastEventAt = null
})

describe('SystemView', () => {
  it('renders one page header and the shared section tabs', async () => {
    const { wrapper } = await mountSystem()

    expect(wrapper.classes()).toContain('page-shell')
    const headings = wrapper.findAll('h1')
    expect(headings).toHaveLength(1)
    expect(headings[0].text()).toBe('System')
    expect(wrapper.find('.page-header__help').exists()).toBe(true)

    const nav = wrapper.get('nav[aria-label="System sections"]')
    const links = nav.findAll('a.section-tabs__tab')
    expect(links.map((link) => link.text())).toEqual([
      'Hardware',
      'Storage',
      'Diagnostics',
      'Backup & restore',
      'Change history',
    ])
    expect(links.map((link) => link.attributes('href'))).toEqual([
      '/system#hardware',
      '/system#storage',
      '/system',
      '/system#config-backup',
      '/system#history',
    ])
    expect(nav.get('[aria-current="page"]').text()).toBe('Diagnostics')
  })

  it('switches the visible section from the URL hash', async () => {
    const { wrapper, router } = await mountSystem()

    expect(wrapper.find('.attention-panel-stub').exists()).toBe(true)
    expect(wrapper.get('#service-health').classes()).toContain('page-section')

    await goTo(router, '/system#hardware')
    expect(wrapper.find('.hardware-panel-stub').exists()).toBe(true)
    expect(wrapper.find('.storage-panel-stub').exists()).toBe(false)
    expect(wrapper.get('nav[aria-label="System sections"] [aria-current="page"]').text()).toBe(
      'Hardware',
    )

    await goTo(router, '/system#storage')
    expect(wrapper.find('.storage-panel-stub').exists()).toBe(true)
    expect(wrapper.find('.hardware-panel-stub').exists()).toBe(false)

    await goTo(router, '/system#config-backup')
    expect(wrapper.find('.backup-panel-stub').exists()).toBe(true)
    expect(wrapper.find('.history-panel-stub').exists()).toBe(false)

    await goTo(router, '/system#history')
    expect(wrapper.find('.history-panel-stub').exists()).toBe(true)
    expect(wrapper.find('.backup-panel-stub').exists()).toBe(false)
  })

  it('keeps persistence details redacted and offers a status refresh', async () => {
    const { wrapper } = await mountSystem()

    const banner = wrapper.get('.state-banner--error')
    expect(banner.get('.state-banner__message').text()).toContain('The save was not stored')
    expect(banner.text()).not.toContain('SECRET')
    expect(wrapper.text()).toContain('Needs attention panel')

    const refresh = wrapper
      .get('#service-health .page-section__actions')
      .findAll('button')
      .find((button) => button.text() === 'Refresh status')
    expect(refresh).toBeTruthy()
    await refresh.trigger('click')
    await flushPromises()

    expect(systemStore.fetchSystemStatus).toHaveBeenCalledTimes(1)
  })

  it('renders the diagnostics service-health section with a download action', async () => {
    const { wrapper } = await mountSystem()

    const section = wrapper.get('#service-health')
    expect(section.get('.page-section__title').text()).toBe('Service health')
    const download = section.get('a.page-action-link')
    expect(download.attributes('href')).toBe('/api/diagnostics/bundle')
    expect(download.text()).toBe('Download diagnostics')
    expect(section.text()).toContain('Proxy health')
    expect(section.text()).toContain('Model observation')
    expect(section.text()).toContain('Settings writes')
  })

  it('renders live-update state, last event age, attempt count, and a retry action', async () => {
    progressStore.isConnected = false
    progressStore.connectionState = 'connecting'
    progressStore.reconnectAttempt = 3
    progressStore.lastEventAt = Date.now() - 5_000
    const { wrapper } = await mountSystem()

    expect(wrapper.text()).toContain('Reconnecting')
    expect(wrapper.text()).toContain('Last event 5s ago')
    expect(wrapper.text()).toContain('Retry attempts: 3')
    const retry = wrapper.get('.live-updates-health__retry')
    expect(retry.attributes('aria-label')).toBe('Retry live updates')
    await retry.trigger('click')
    expect(retryConnection).toHaveBeenCalledTimes(1)
  })
})
