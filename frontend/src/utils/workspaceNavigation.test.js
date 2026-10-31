import { afterEach, describe, expect, it } from 'vitest'
import { flushPromises } from '@vue/test-utils'
import { createRouter, createMemoryHistory } from 'vue-router'
import { installWorkspaceNavigation } from './workspaceNavigation'
afterEach(() => {
  document.body.innerHTML = ''
})
describe('workspace navigation', () => {
  it('sets meaningful page titles, preserves initial focus and query edits, and focuses real destination changes', async () => {
    document.body.innerHTML =
      '<button id="launcher">Navigate</button><main id="main-content" tabindex="-1"></main>'
    const launcher = document.getElementById('launcher')
    launcher.focus()
    const router = createRouter({
      history: createMemoryHistory(),
      routes: [
        { path: '/models', meta: { title: 'Models' }, component: { template: '<div />' } },
        { path: '/system', meta: { title: 'System' }, component: { template: '<div />' } },
      ],
    })
    const remove = installWorkspaceNavigation(router)
    await router.push('/models')
    await flushPromises()
    expect(document.title).toBe('Models — llama.cpp Studio')
    expect(document.activeElement).toBe(launcher)
    await router.push('/system#config-backup')
    await flushPromises()
    expect(document.title).toBe('Backup & restore — llama.cpp Studio')
    expect(document.activeElement.id).toBe('main-content')
    launcher.focus()
    await router.push('/system?q=filter#config-backup')
    await flushPromises()
    expect(document.activeElement).toBe(launcher)
    remove()
  })
})
