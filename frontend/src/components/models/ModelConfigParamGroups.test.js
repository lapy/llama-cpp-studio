import { enableAutoUnmount, flushPromises, mount } from '@vue/test-utils'
import { afterEach, describe, expect, it } from 'vitest'
import { reactive } from 'vue'

import ModelConfigParamGroups from './ModelConfigParamGroups.vue'

enableAutoUnmount(afterEach)

const ButtonStub = {
  props: ['label', 'icon', 'text', 'severity', 'outlined', 'rounded', 'size', 'type', 'disabled'],
  emits: ['click'],
  template:
    '<button type="button" :data-label="label" :disabled="disabled" @click="$emit(`click`)">{{ label }}</button>',
}

const TextInputStub = {
  props: ['modelValue', 'placeholder', 'disabled', 'id', 'inputId', 'type'],
  emits: ['update:modelValue'],
  template:
    '<input :id="inputId || id" :type="type || `text`" :value="modelValue ?? ``" :placeholder="placeholder" :disabled="disabled" @input="$emit(`update:modelValue`, $event.target.value)" />',
}

const ToggleSwitchStub = {
  props: ['modelValue', 'inputId'],
  emits: ['update:modelValue'],
  template:
    '<input :id="inputId" type="checkbox" :checked="Boolean(modelValue)" @change="$emit(`update:modelValue`, $event.target.checked)" />',
}

const SelectStub = {
  props: ['modelValue', 'options', 'optionLabel', 'optionValue', 'disabled', 'inputId'],
  emits: ['update:modelValue'],
  template:
    '<select :id="inputId" :disabled="disabled" :value="modelValue ?? ``" @change="$emit(`update:modelValue`, $event.target.value)"><option v-for="option in options || []" :key="option.value ?? option" :value="option.value ?? option">{{ option.label ?? option }}</option></select>',
}

const InputTagsStub = {
  props: ['modelValue', 'inputId', 'disabled', 'delimiter'],
  emits: ['update:modelValue'],
  template:
    '<input :id="inputId" :disabled="disabled" :value="Array.isArray(modelValue) ? modelValue.join(`,`) : ``" @input="$emit(`update:modelValue`, $event.target.value ? $event.target.value.split(`,`) : [])" />',
}

const TextareaStub = {
  props: ['modelValue', 'rows', 'disabled', 'id', 'placeholder'],
  emits: ['update:modelValue'],
  template:
    '<textarea :id="id" :disabled="disabled" :value="modelValue ?? ``" @input="$emit(`update:modelValue`, $event.target.value)" />',
}

const slots = {
  Button: ButtonStub,
  Tag: { props: ['value'], template: '<span>{{ value }}</span>' },
  InputText: TextInputStub,
  InputNumber: TextInputStub,
  ToggleSwitch: ToggleSwitchStub,
  Select: SelectStub,
  InputTags: InputTagsStub,
  Message: { template: '<div><slot /></div>' },
  Textarea: TextareaStub,
  Slider: true,
  MultiSelect: {
    props: ['modelValue'],
    emits: ['update:modelValue'],
    template: '<div class="multiselect-stub" />',
  },
}

const baseParams = [
  {
    key: 'ctx_size',
    label: 'Context Size',
    type: 'int',
    value_kind: 'scalar',
    default: 4096,
    sectionId: 'core',
    supported: true,
  },
  {
    key: 'n_gpu_layers',
    label: 'GPU Layers',
    type: 'int',
    value_kind: 'scalar',
    default: 0,
    sectionId: 'core',
    supported: true,
  },
  {
    key: 'temperature',
    label: 'Temperature',
    type: 'float',
    value_kind: 'scalar',
    default: 0.8,
    sectionId: 'sampling',
    supported: true,
  },
  {
    key: 'top_p',
    label: 'Top P',
    type: 'float',
    value_kind: 'scalar',
    default: 0.95,
    sectionId: 'sampling',
    supported: true,
  },
  {
    key: 'host',
    label: 'Host',
    type: 'string',
    value_kind: 'scalar',
    default: '127.0.0.1',
    sectionId: 'network',
    supported: true,
  },
]

function hasValue(value) {
  if (value == null || value === '') return false
  return !(Array.isArray(value) && value.length === 0)
}

function mountGroups(props = {}) {
  const config = reactive({ temperature: 0.7, top_p: 0.9, ...(props.config || {}) })
  const wrapper = mount(ModelConfigParamGroups, {
    props: {
      params: baseParams,
      addableParams: [],
      isModified: (param) => hasValue(config[param.key]),
      isDelimitedEnum: () => false,
      ...props,
      config,
    },
    global: {
      directives: { tooltip: () => {} },
      stubs: slots,
    },
  })
  return { wrapper, config }
}

function group(wrapper, id) {
  return wrapper.get(`[data-group="${id}"]`)
}

describe('ModelConfigParamGroups', () => {
  it('renders ordered groups with counts and default open state', () => {
    const { wrapper } = mountGroups()

    const groups = wrapper.findAll('.param-group')
    expect(groups.map((element) => element.attributes('data-group'))).toEqual([
      'context',
      'memory',
      'sampling',
      'server',
    ])

    expect(group(wrapper, 'context').get('.param-group__toggle').attributes('aria-expanded')).toBe(
      'true',
    )
    expect(group(wrapper, 'memory').get('.param-group__toggle').attributes('aria-expanded')).toBe(
      'true',
    )
    expect(group(wrapper, 'sampling').get('.param-group__badge').text()).toBe(
      '2 parameters · 2 changed',
    )
    expect(group(wrapper, 'context').get('.param-group__badge').text()).toBe(
      '1 parameter · 0 changed',
    )

    // A group holding a modified parameter opens even though it is not a
    // default-open group.
    expect(group(wrapper, 'sampling').get('.param-group__toggle').attributes('aria-expanded')).toBe(
      'true',
    )
    expect(group(wrapper, 'server').get('.param-group__toggle').attributes('aria-expanded')).toBe(
      'false',
    )
    expect(wrapper.get('.param-groups__summary').text()).toBe('2 changed')
  })

  it('keeps the existing parameter row selectors and ids', () => {
    const { wrapper } = mountGroups()

    const row = wrapper
      .findAll('.param-field')
      .find((field) => field.text().includes('Context Size'))
    expect(row).toBeTruthy()
    expect(row.get('#p-core-ctx_size').exists()).toBe(true)
    expect(row.get('.param-key-hint').text()).toBe('ctx_size')
    expect(row.text()).toContain('Default')
    expect(row.text()).toContain('Engine default: 4096')
    expect(row.get('.param-remove-btn').exists()).toBe(true)

    const changed = wrapper
      .findAll('.param-field')
      .find((field) => field.text().includes('Temperature'))
    expect(changed.text()).toContain('Override')
    expect(changed.get('.param-reset').text()).toBe('Use engine default')
  })

  it('preserves legacy basic ids for pinned params', () => {
    const { wrapper } = mountGroups({ pinnedKeys: ['ctx_size'] })

    const row = wrapper
      .findAll('.param-field')
      .find((field) => field.text().includes('Context Size'))
    expect(row.get('#basic-ctx_size').exists()).toBe(true)
    expect(row.get('#basic-help-ctx_size').text()).toContain('Engine default: 4096')
  })

  it('updates changed counts when the parent config changes', async () => {
    const { wrapper, config } = mountGroups()
    expect(wrapper.get('.param-groups__summary').text()).toBe('2 changed')

    config.top_p = null
    await flushPromises()

    expect(wrapper.get('.param-groups__summary').text()).toBe('1 changed')
    expect(group(wrapper, 'sampling').get('.param-group__badge').text()).toBe(
      '2 parameters · 1 changed',
    )
  })

  it('filters to changed parameters only', async () => {
    const { wrapper } = mountGroups()

    const changedButton = wrapper
      .findAll('.param-filter__option')
      .find((button) => button.text() === 'Changed')
    await changedButton.trigger('click')

    expect(
      wrapper.findAll('.param-group').map((element) => element.attributes('data-group')),
    ).toEqual(['sampling'])
    expect(wrapper.text()).toContain('Temperature')
    expect(wrapper.text()).not.toContain('Context Size')

    const allButton = wrapper
      .findAll('.param-filter__option')
      .find((button) => button.text() === 'All')
    await allButton.trigger('click')
    expect(wrapper.findAll('.param-group')).toHaveLength(4)
  })

  it('auto-expands search matches and hides non-matching rows', async () => {
    const { wrapper } = mountGroups()

    // Collapse a default-open group first so the auto-expand is observable.
    await group(wrapper, 'context').get('.param-group__toggle').trigger('click')
    expect(group(wrapper, 'context').get('.param-group__toggle').attributes('aria-expanded')).toBe(
      'false',
    )

    await wrapper.setProps({ searchQuery: 'context' })
    await flushPromises()

    expect(
      wrapper.findAll('.param-group').map((element) => element.attributes('data-group')),
    ).toEqual(['context'])
    expect(group(wrapper, 'context').get('.param-group__toggle').attributes('aria-expanded')).toBe(
      'true',
    )
    expect(wrapper.text()).toContain('Context Size')
    expect(wrapper.text()).not.toContain('Temperature')
  })

  it('explains how to clear a search with no matches', async () => {
    const { wrapper } = mountGroups()
    await wrapper.setProps({ searchQuery: 'nothing-matches-this' })
    await flushPromises()

    const empty = wrapper.get('.param-groups__empty')
    expect(empty.text()).toContain('No parameters match')
    expect(empty.text()).toContain('Clear the search')

    await empty
      .get('button[aria-label="Clear search"], button[data-label="Clear search"]')
      .trigger('click')
    expect(wrapper.emitted('update:searchQuery')?.at(-1)).toEqual([''])
  })

  it('expands and collapses all groups', async () => {
    const { wrapper } = mountGroups()

    const collapse = wrapper.get('button[data-label="Collapse all"]')
    await collapse.trigger('click')
    for (const id of ['context', 'memory', 'sampling', 'server']) {
      expect(group(wrapper, id).get('.param-group__toggle').attributes('aria-expanded')).toBe(
        'false',
      )
      expect(group(wrapper, id).get('.param-group__body').attributes('style')).toContain(
        'display: none',
      )
    }

    const expand = wrapper.get('button[data-label="Expand all"]')
    await expand.trigger('click')
    for (const id of ['context', 'memory', 'sampling', 'server']) {
      expect(group(wrapper, id).get('.param-group__toggle').attributes('aria-expanded')).toBe(
        'true',
      )
      expect(group(wrapper, id).get('.param-group__body').attributes('style') || '').not.toContain(
        'display: none',
      )
    }
  })

  it('emits row edits, resets, and removals', async () => {
    const { wrapper } = mountGroups()

    const temperature = wrapper
      .findAll('.param-field')
      .find((field) => field.text().includes('Temperature'))
    await temperature.get('input').setValue('0.5')
    expect(wrapper.emitted('update:param')?.at(-1)).toEqual(['temperature', '0.5'])

    await temperature.get('.param-reset').trigger('click')
    expect(wrapper.emitted('reset:param')?.at(-1)?.[0]?.key).toBe('temperature')

    await temperature.get('.param-remove-btn').trigger('click')
    expect(wrapper.emitted('remove:param')?.at(-1)).toEqual(['temperature'])
  })

  it('emits JSON updates through the JSON textarea', async () => {
    const { wrapper } = mountGroups({
      config: { chat_template_kwargs: { enable_thinking: false } },
      params: [
        ...baseParams,
        {
          key: 'chat_template_kwargs',
          label: 'Chat template kwargs',
          type: 'json',
          value_kind: 'scalar',
          default: null,
          sectionId: 'network',
          supported: true,
        },
      ],
    })

    const row = wrapper
      .findAll('.param-field')
      .find((field) => field.text().includes('Chat template kwargs'))
    expect(row).toBeTruthy()
    await row.get('textarea').setValue('{"enable_thinking":true}')
    expect(wrapper.emitted('update:json')?.at(-1)).toEqual([
      'chat_template_kwargs',
      '{"enable_thinking":true}',
    ])
  })

  it('adds a parameter from the search tag cloud', async () => {
    const { wrapper } = mountGroups({
      addableParams: [
        {
          key: 'seed',
          label: 'Seed',
          type: 'int',
          value_kind: 'scalar',
          default: -1,
          supported: true,
        },
      ],
    })
    await wrapper.setProps({ searchQuery: 'seed' })
    await flushPromises()

    const tag = wrapper.get('.param-search-tag')
    expect(tag.text()).toContain('Seed')
    await tag.trigger('click')
    expect(wrapper.emitted('add:param')?.at(-1)).toEqual(['seed'])
  })

  it('emits hide-unsupported updates and drops unsupported rows when hidden', async () => {
    const params = [
      ...baseParams.filter((param) => param.key !== 'host'),
      {
        key: 'host',
        label: 'Host',
        type: 'string',
        value_kind: 'scalar',
        default: '127.0.0.1',
        sectionId: 'network',
        supported: false,
      },
    ]
    const { wrapper } = mountGroups({ params })

    expect(group(wrapper, 'server').text()).toContain('Not in this build')
    await wrapper.get('#toggle-hide-unsupported').setValue(true)
    expect(wrapper.emitted('update:hideUnsupported')?.at(-1)).toEqual([true])

    await wrapper.setProps({ hideUnsupported: true })
    expect(wrapper.find('[data-group="server"]').exists()).toBe(false)
  })

  it('offers one auto-fit restore action when overrides are present', async () => {
    const { wrapper } = mountGroups()
    expect(wrapper.find('[data-auto-fit-action]').exists()).toBe(false)

    await wrapper.setProps({
      autoFitOverrides: [
        { key: 'ctx_size', label: 'Context size' },
        { key: 'n_gpu_layers', label: 'GPU layers' },
      ],
    })
    const action = wrapper.get('[data-auto-fit-action]')
    expect(action.text()).toBe('Restore auto settings')

    await action.trigger('click')
    expect(wrapper.emitted('restore-auto-fit')).toHaveLength(1)
  })
})
