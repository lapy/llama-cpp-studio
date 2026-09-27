import { describe, it, expect, beforeEach } from 'vitest'
import { defineComponent, ref } from 'vue'
import { mount, flushPromises } from '@vue/test-utils'
import { createMemoryHistory, createRouter } from 'vue-router'
import { clearDraft, readDraft, useDraftGuard, writeDraft } from './useDraftGuard'

describe('config drafts', () => {
  beforeEach(() => {
    sessionStorage.clear()
  })

  it('stores and clears a draft for one model and engine', () => {
    writeDraft('model-1', 'llama_cpp', { config: { temperature: 0.2 }, engine: 'llama_cpp' })
    expect(readDraft('model-1', 'llama_cpp')?.config.temperature).toBe(0.2)
    expect(readDraft('model-1', 'ik_llama')).toBeNull()
    clearDraft('model-1', 'llama_cpp')
    expect(readDraft('model-1', 'llama_cpp')).toBeNull()
  })

  it('lets the user stay, and leaves after the form is clean', async () => {
    const dirty = ref(true)
    const Guard = defineComponent({
      setup() {
        const guard = useDraftGuard(() => dirty.value)
        return { guard }
      },
      template: '<div />',
    })
    const router = createRouter({
      history: createMemoryHistory(),
      routes: [
        { path: '/', component: Guard },
        { path: '/models', component: { template: '<div>models</div>' } },
      ],
    })
    const wrapper = mount({ template: '<router-view />' }, { global: { plugins: [router] } })
    await router.push('/')
    await router.isReady()
    await flushPromises()

    const page = wrapper.getComponent(Guard)
    const pending = router.push('/models')
    await flushPromises()
    expect(router.currentRoute.value.path).toBe('/')
    expect(page.vm.guard.leavePromptVisible.value).toBe(true)

    page.vm.guard.finishLeave(false)
    await pending
    await flushPromises()
    expect(router.currentRoute.value.path).toBe('/')

    dirty.value = false
    await router.push('/models')
    await flushPromises()
    expect(router.currentRoute.value.path).toBe('/models')
    wrapper.unmount()
  })
})
