import { describe, it, expect, vi } from 'vitest'
import { mount } from '@vue/test-utils'
import { reactive, nextTick } from 'vue'

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
  it('shows live status and updates when the SSE connection drops', async () => {
    const wrapper = mount(AppFooter)

    expect(wrapper.text()).toContain('llama.cpp Studio v')
    expect(wrapper.text()).toContain('Live')
    expect(wrapper.find('.footer-status--ok').exists()).toBe(true)

    progressStore.isConnected = false
    await nextTick()

    expect(wrapper.text()).toContain('Reconnecting…')
    expect(wrapper.find('.footer-status--warn').exists()).toBe(true)
    expect(wrapper.text()).toContain('Queue full (32/32)')
    expect(wrapper.text()).toContain('The save was not stored. The previous state is unchanged.')
    expect(wrapper.text()).not.toContain('hf_FOOTERSECRET')
    expect(wrapper.text()).not.toContain('OSError')
    expect(wrapper.text()).toContain('Proxy health')
    expect(wrapper.text()).toContain('No running-model observation')

    systemStore.systemStatus.persistence.latest_failure = {
      code: 'STORE_WRITE_FAILED',
      committed: true,
      message: 'hf_ACKSECRET',
    }
    await nextTick()
    expect(wrapper.text()).toContain('acknowledgement failed')
    expect(wrapper.text()).not.toContain('hf_ACKSECRET')

    systemStore.systemStatus.persistence.latest_failure = {
      code: 'STORE_WRITE_FAILED',
      committed: 'unknown',
    }
    await nextTick()
    expect(wrapper.text()).toContain('could not be established')

    systemStore.systemStatus.persistence.latest_failure = {
      code: 'PERSISTENCE_FAILED',
      message: 'cmake -DSECRET=hf_UNKNOWNSECRET',
    }
    await nextTick()
    expect(wrapper.text()).toContain('Its details were not exported.')
    expect(wrapper.text()).not.toContain('hf_UNKNOWNSECRET')
  })
})
