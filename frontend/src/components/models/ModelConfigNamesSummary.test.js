import { describe, expect, it } from 'vitest'
import { mount } from '@vue/test-utils'

import ModelConfigNamesSummary from './ModelConfigNamesSummary.vue'

const RouterLinkStub = {
  props: ['to'],
  template: '<a class="router-link-stub" :data-to="JSON.stringify(to)"><slot /></a>',
}

const ButtonStub = {
  props: ['label', 'icon', 'size', 'text', 'disabled'],
  emits: ['click'],
  template:
    '<button type="button" :data-label="label" :disabled="disabled" @click="$emit(`click`)">{{ label }}</button>',
}

function mountSummary(props = {}) {
  return mount(ModelConfigNamesSummary, {
    props: {
      model: {
        id: 'org/model-GGUF',
        display_name: 'Assistant Model',
        base_model_name: 'assistant-base',
        huggingface_id: 'org/model',
      },
      config: {},
      stableModelId: '',
      ...props,
    },
    global: {
      stubs: { RouterLink: RouterLinkStub, Button: ButtonStub },
    },
  })
}

describe('ModelConfigNamesSummary', () => {
  it('explains the catalog identity, stable runtime id and routing names in one panel', () => {
    const wrapper = mountSummary({
      config: {
        model_alias: 'Assistant',
        swap_aliases: ['chat', 'CHAT', 'app'],
        set_params_by_id: [
          { sub_id: 'thinking', params: { thinking_effort: 'low', temperature: 0.2 } },
          { sub_id: '', params: {} },
        ],
      },
      stableModelId: 'studio-assistant',
    })

    expect(wrapper.get('h2').text()).toBe('Names & routing')
    const text = wrapper.text()
    expect(text).toContain('Assistant Model')
    expect(text).toContain('org/model-GGUF')
    expect(text).toContain('studio-assistant')
    // The saved alias is normalized to the name clients receive.
    expect(text).toContain('assistant')
    expect(text).toContain('Ordinary aliases')
    expect(text).toContain('chat')
    expect(text).toContain('app')
    expect(wrapper.findAll('.names-summary__chip').map((chip) => chip.text())).toEqual(
      expect.arrayContaining([
        expect.stringContaining('assistant:thinking'),
        expect.stringContaining('2 overrides'),
      ]),
    )

    const proxyLink = wrapper
      .findAll('a.router-link-stub')
      .find((link) => link.text() === 'Open routing settings')
    expect(JSON.parse(proxyLink.attributes('data-to'))).toEqual({
      path: '/proxy',
      query: { tab: 'settings' },
    })
    expect(wrapper.text()).toContain('multi-target aliases dispatch round-robin')
  })

  it('offers the next step when no primary API name or presets exist yet', () => {
    const wrapper = mountSummary()

    expect(wrapper.text()).toContain('Assigned the first time this model starts')
    expect(wrapper.text()).toContain('Not set — clients use the stable runtime id')
    expect(wrapper.text()).toContain('No additional ordinary aliases are saved')
    expect(wrapper.text()).toContain('No request presets are saved')
    expect(wrapper.text()).toContain('No routable name is published yet')
    expect(wrapper.text()).toContain('Add primary name')
    expect(wrapper.text()).toContain('Edit request presets')
    expect(wrapper.text()).toContain('Open in Models')
  })

  it('routes each edit action to its owning section or dialog', async () => {
    const wrapper = mountSummary({
      config: {
        model_alias: 'assistant',
        swap_aliases: ['chat'],
        set_params_by_id: [{ sub_id: 'fast', params: {} }],
      },
      stableModelId: 'studio-assistant',
    })

    const button = (label) =>
      wrapper.findAll('button').find((entry) => entry.attributes('data-label') === label)

    await button('Edit primary name').trigger('click')
    await button('Edit request presets').trigger('click')
    await button('Edit ordinary aliases').trigger('click')

    expect(wrapper.emitted('edit-section')).toEqual([['launch'], ['launch']])
    expect(wrapper.emitted('manage-templates')).toHaveLength(1)
  })
})
