import { beforeEach, it, expect, vi } from 'vitest'
import { mount, flushPromises } from '@vue/test-utils'
import { reactive } from 'vue'
const route = reactive({ hash: '' })
vi.mock('vue-router', () => ({ useRoute: () => route }))
const store = reactive({
  systemStatus: {
    persistence: {
      latest_failure: { code: 'STORE_WRITE_FAILED', committed: false, message: 'SECRET' },
    },
  },
})
const retryConnection = vi.fn()
const progressStore = reactive({
  isConnected: true,
  connectionState: 'live',
  reconnectAttempt: 0,
  lastEventAt: null,
  retryConnection,
})
vi.mock('@/stores/engines', () => ({ useEnginesStore: () => store }))
vi.mock('@/stores/progress', () => ({ useProgressStore: () => progressStore }))
vi.mock('@/components/system/NeedsAttentionPanel.vue', () => ({
  default: { template: '<div class="needs-attention-stub">Needs attention panel</div>' },
}))
import SystemView from './SystemView.vue'

function mountSystem() {
  return mount(SystemView, {
    global: {
      stubs: {
        RouterLink: { template: '<a><slot /></a>' },
        ConfigBackupPanel: { template: '<div>Backup panel</div>' },
        ConfigHistoryPanel: { template: '<div>History panel</div>' },
      },
    },
  })
}

beforeEach(() => {
  route.hash = ''
  retryConnection.mockReset()
  progressStore.isConnected = true
  progressStore.connectionState = 'live'
  progressStore.reconnectAttempt = 0
  progressStore.lastEventAt = null
})

it('separates diagnostics, backup and history and keeps persistence details redacted', async () => {
  const wrapper = mountSystem()
  expect(wrapper.text()).toContain('The save was not stored')
  expect(wrapper.text()).not.toContain('SECRET')
  expect(wrapper.text()).toContain('Needs attention panel')
  expect(wrapper.text()).not.toContain('Backup panel')
  route.hash = '#config-backup'
  await flushPromises()
  expect(wrapper.text()).toContain('Backup panel')
  expect(wrapper.text()).not.toContain('History panel')
  expect(wrapper.text()).not.toContain('Needs attention panel')
  route.hash = '#history'
  await flushPromises()
  expect(wrapper.text()).toContain('History panel')
  expect(wrapper.text()).not.toContain('Backup panel')
  wrapper.unmount()
})

it('renders live-update state, last event age, attempt count, and a retry action', async () => {
  progressStore.isConnected = false
  progressStore.connectionState = 'connecting'
  progressStore.reconnectAttempt = 3
  progressStore.lastEventAt = Date.now() - 5_000
  const wrapper = mountSystem()

  expect(wrapper.text()).toContain('Reconnecting')
  expect(wrapper.text()).toContain('Last event 5s ago')
  expect(wrapper.text()).toContain('Retry attempts: 3')
  const retry = wrapper.get('.live-updates-health__retry')
  expect(retry.attributes('aria-label')).toBe('Retry live updates')
  await retry.trigger('click')
  expect(retryConnection).toHaveBeenCalledTimes(1)
  wrapper.unmount()
})
