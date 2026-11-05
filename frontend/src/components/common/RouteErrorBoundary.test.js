import { describe, expect, it } from 'vitest'
import { mount } from '@vue/test-utils'
import { defineComponent, h } from 'vue'
import RouteErrorBoundary from './RouteErrorBoundary.vue'

describe('route rendering recovery', () => {
  it('contains rendering failures and remounts the view when retried', async () => {
    let broken = true
    const Child = defineComponent({
      setup() {
        if (broken) throw new Error('private diagnostic')
        return () => h('p', 'Recovered view')
      },
    })
    const wrapper = mount(RouteErrorBoundary, {
      slots: { default: () => h(Child) },
      global: {
        stubs: {
          RouterLink: true,
          Button: { props: ['label'], template: '<button>{{ label }}</button>' },
        },
      },
    })
    await wrapper.vm.$nextTick()
    expect(wrapper.text()).toContain('This view could not be displayed')
    expect(wrapper.text()).not.toContain('private diagnostic')
    broken = false
    await wrapper.get('button').trigger('click')
    expect(wrapper.text()).toContain('Recovered view')
    wrapper.unmount()
  })
})
