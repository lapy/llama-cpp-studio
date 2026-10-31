import { describe, expect, it } from 'vitest'
import { defineComponent, ref } from 'vue'
import { flushPromises, mount } from '@vue/test-utils'
import { createMemoryHistory, createRouter } from 'vue-router'
import { useQueryState } from './useQueryState'

describe('view query state', () => {
  it('restores whitelisted controls, writes changes and handles browser history', async () => {
    const value = ref('all')
    const View = defineComponent({
      setup() {
        useQueryState({
          task: {
            get: () => value.value,
            set: (v) => {
              value.value = v
            },
            parse: (v) => (['audio', 'text'].includes(v) ? v : 'all'),
            default: 'all',
          },
        })
        return {}
      },
      template: '<div />',
    })
    const router = createRouter({
      history: createMemoryHistory(),
      routes: [{ path: '/', component: View }],
    })
    await router.push('/?task=audio&unrelated=keep')
    const wrapper = mount({ template: '<router-view />' }, { global: { plugins: [router] } })
    await flushPromises()
    expect(value.value).toBe('audio')
    value.value = 'text'
    await flushPromises()
    expect(router.currentRoute.value.query).toEqual({ task: 'text', unrelated: 'keep' })
    await router.push('/?task=invalid')
    await flushPromises()
    expect(value.value).toBe('all')
    router.back()
    await flushPromises()
    expect(value.value).toBe('text')
    wrapper.unmount()
  })
})
