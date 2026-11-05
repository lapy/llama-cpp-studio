import { it, expect, vi } from 'vitest'
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
vi.mock('@/stores/engines', () => ({ useEnginesStore: () => store }))
vi.mock('@/stores/progress', () => ({ useProgressStore: () => ({ isConnected: true }) }))
import SystemView from './SystemView.vue'
it('separates diagnostics, backup and history and keeps persistence details redacted', async () => {
  const wrapper = mount(SystemView, {
    global: {
      stubs: {
        RouterLink: { template: '<a><slot /></a>' },
        ConfigBackupPanel: { template: '<div>Backup panel</div>' },
        ConfigHistoryPanel: { template: '<div>History panel</div>' },
      },
    },
  })
  expect(wrapper.text()).toContain('The save was not stored')
  expect(wrapper.text()).not.toContain('SECRET')
  expect(wrapper.text()).not.toContain('Backup panel')
  route.hash = '#config-backup'
  await flushPromises()
  expect(wrapper.text()).toContain('Backup panel')
  expect(wrapper.text()).not.toContain('History panel')
  route.hash = '#history'
  await flushPromises()
  expect(wrapper.text()).toContain('History panel')
  expect(wrapper.text()).not.toContain('Backup panel')
  wrapper.unmount()
})
