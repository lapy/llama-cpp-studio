import { mount } from '@vue/test-utils'
import { describe, expect, it } from 'vitest'
import { createMemoryHistory, createRouter } from 'vue-router'
import NotFoundView from './NotFoundView.vue'

async function mountNotFound() {
  const router = createRouter({
    history: createMemoryHistory(),
    routes: [
      { path: '/models', component: { template: '<div />' } },
      { path: '/:pathMatch(.*)*', component: NotFoundView },
    ],
  })
  await router.push('/missing-workspace')
  await router.isReady()
  return mount(NotFoundView, { global: { plugins: [router] } })
}

describe('NotFoundView', () => {
  it('uses the shared page shell, PageHeader and EmptyState next action', async () => {
    const wrapper = await mountNotFound()

    expect(wrapper.get('.page-shell').exists()).toBe(true)
    expect(wrapper.get('.page-header').exists()).toBe(true)
    expect(wrapper.get('h1.page-title').text()).toBe('Page not found')
    expect(wrapper.get('.page-empty').exists()).toBe(true)
    expect(wrapper.get('.page-empty__title').text()).toBe(
      'This address does not match a Studio workspace.',
    )
    expect(wrapper.get('.page-empty .page-action-link').attributes('href')).toBe('/models')
    expect(wrapper.get('.page-empty .page-action-link').text()).toContain('Return to Models')
  })
})
