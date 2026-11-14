import { describe, it, expect } from 'vitest'
import { mount } from '@vue/test-utils'
import RequestAliasesEditor from './RequestAliasesEditor.vue'

const Button = {
  props: ['label', 'icon', 'severity', 'text', 'size', 'outlined', 'loading'],
  emits: ['click'],
  template: '<button :data-label="label" @click="$emit(`click`)">{{ label }}</button>',
}

const InputText = {
  props: ['modelValue', 'placeholder', 'ariaLabel', 'inputmode'],
  emits: ['update:modelValue'],
  template:
    '<input :aria-label="ariaLabel" :value="modelValue ?? ``" @input="$emit(`update:modelValue`, $event.target.value)" />',
}

const Textarea = {
  props: ['modelValue', 'rows', 'ariaLabel'],
  emits: ['update:modelValue'],
  template:
    '<textarea :aria-label="ariaLabel" :value="modelValue ?? ``" @input="$emit(`update:modelValue`, $event.target.value)" />',
}

const Select = {
  props: ['modelValue', 'options', 'optionLabel', 'optionValue', 'ariaLabel'],
  emits: ['update:modelValue'],
  computed: {
    normalizedOptions() {
      return (this.options || []).map((option) =>
        typeof option === 'object' ? option : { label: option, value: option },
      )
    },
  },
  template: `<select :aria-label="ariaLabel" :value="modelValue ?? ''" @change="$emit('update:modelValue', $event.target.value)">
    <option v-for="option in normalizedOptions" :key="option.value" :value="option.value">{{ option.label }}</option>
  </select>`,
}

function mountEditor(modelValue = []) {
  return mount(RequestAliasesEditor, {
    props: { modelValue, baseName: 'assistant' },
    global: { stubs: { Button, InputText, Textarea, Select } },
  })
}

describe('RequestAliasesEditor', () => {
  it('renders existing nested aliases as structured rows, not a JSON blob', () => {
    const wrapper = mountEditor([
      {
        sub_id: 'thinking',
        params: {
          'chat_template_kwargs.enable_thinking': false,
          thinking_effort: 'low',
        },
      },
    ])
    expect(wrapper.text()).toContain('chat_template_kwargs.')
    expect(wrapper.find('textarea').exists()).toBe(false)
    const names = wrapper.findAll('input[aria-label="Override name for alias 1"]')
    expect(names.map((input) => input.element.value)).toEqual([
      'enable_thinking',
      'thinking_effort',
    ])
    const scopeSelects = wrapper.findAll('select[aria-label="Override target"]')
    expect(scopeSelects.map((select) => select.element.value)).toEqual(['chat_template', 'request'])
    expect(wrapper.get('select[aria-label="Boolean value"]').element.value).toBe('false')
  })

  it('adds a chat-template override through the preset control', async () => {
    const wrapper = mountEditor([])
    await wrapper.get('[data-label="Add alias"]').trigger('click')
    let value = wrapper.emitted('update:modelValue').at(-1)[0]
    expect(value).toEqual([{ sub_id: 'thinking', params: {} }])
    await wrapper.setProps({ modelValue: value })
    await wrapper.get('[data-label="Add chat template"]').trigger('click')
    value = wrapper.emitted('update:modelValue').at(-1)[0]
    expect(value).toEqual([
      {
        sub_id: 'thinking',
        params: { 'chat_template_kwargs.enable_thinking': true },
      },
    ])
  })

  it('preserves a false boolean and rejects invalid JSON without publishing', async () => {
    const wrapper = mountEditor([{ sub_id: 'fast', params: { stream: false } }])
    expect(wrapper.get('select[aria-label="Boolean value"]').element.value).toBe('false')
    const published = wrapper.emitted('update:modelValue')?.length || 0
    await wrapper.get('select[aria-label="Override value type"]').setValue('json')
    const textarea = wrapper.get('textarea[aria-label="JSON value"]')
    expect(textarea.element.value).toBe('false')
    const publishedAfterTypeChange = wrapper.emitted('update:modelValue')?.length || 0
    await textarea.setValue('{broken')
    expect(wrapper.emitted('validity').at(-1)[0]).toBe(false)
    expect(wrapper.emitted('update:modelValue')?.length || 0).toBe(publishedAfterTypeChange)
    expect(publishedAfterTypeChange).toBeGreaterThanOrEqual(published)
  })
})
