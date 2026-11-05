import { describe, it, expect, vi } from 'vitest'
import { mount } from '@vue/test-utils'
import { reactive, nextTick } from 'vue'

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

const progressStore = reactive({
  isConnected: true,
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

describe('AppFooter', () => {
  it('keeps the footer to service status and a System link', async () => {
    const wrapper = mount(AppFooter, {
      global: { stubs: { RouterLink: { props: ['to'], template: '<a :href="to"><slot /></a>' } } },
    })
    expect(wrapper.text()).toContain('Live')
    expect(wrapper.text()).toContain('Needs attention')
    expect(wrapper.get('a').attributes('href')).toBe('/system')
    expect(wrapper.find('#config-backup').exists()).toBe(false)
    progressStore.isConnected = false
    await nextTick()
    expect(wrapper.text()).toContain('Reconnecting…')
    expect(wrapper.text()).not.toContain('hf_FOOTERSECRET')
    wrapper.unmount()
  })
})
