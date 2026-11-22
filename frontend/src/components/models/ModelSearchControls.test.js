import { mount } from '@vue/test-utils'
import { describe, expect, it, vi } from 'vitest'

import ModelSearchControls from './ModelSearchControls.vue'

const ButtonStub = {
  props: ['label', 'ariaLabel'],
  emits: ['click'],
  template: '<button :aria-label="ariaLabel" @click="$emit(`click`)">{{ label }}</button>',
}

const InputStub = {
  props: ['modelValue', 'ariaLabel'],
  emits: ['update:modelValue', 'keyup'],
  template:
    '<input :aria-label="ariaLabel" :value="modelValue" @input="$emit(`update:modelValue`, $event.target.value)" @keyup="$emit(`keyup`, $event)" />',
}

const SelectStub = {
  props: ['modelValue', 'options', 'ariaLabel'],
  emits: ['update:modelValue', 'change'],
  template:
    '<select :aria-label="ariaLabel" :value="modelValue" @change="$emit(`update:modelValue`, $event.target.value); $emit(`change`, $event)"><option v-for="option in options" :key="option.value ?? option" :value="option.value ?? option">{{ option.label ?? option }}</option></select>',
}

function mountControls(props = {}) {
  return mount(ModelSearchControls, {
    props: {
      query: 'llama',
      taskFilterOptions: [{ label: 'Text', value: 'text' }],
      formatOptions: [{ label: 'GGUF', value: 'gguf' }],
      engineFilterOptions: [{ label: 'Example engine', value: 'example_engine' }],
      inputModalityOptions: ['text'],
      outputModalityOptions: ['text'],
      providerOptions: [{ label: 'Hugging Face', value: 'huggingface' }],
      installMethodOptions: [{ label: 'Download', value: 'download' }],
      ...props,
    },
    global: { stubs: { Button: ButtonStub, InputText: InputStub, Select: SelectStub } },
  })
}

describe('ModelSearchControls', () => {
  it('forwards search intent and controlled filter values', async () => {
    const wrapper = mountControls({ filtersOpen: true, showFormatSelect: true })
    expect(wrapper.get('.search-bar').classes()).toContain('compact-toolbar')

    await wrapper.get('input').setValue('vision')
    await wrapper.get('input').trigger('keyup', { key: 'Enter' })
    const task = wrapper.get('select[aria-label="Task"]')
    await task.setValue('text')

    expect(wrapper.emitted('update:query')?.at(-1)).toEqual(['vision'])
    expect(wrapper.emitted('update:taskFilter')?.at(-1)).toEqual(['text'])
    expect(wrapper.emitted('search')).toHaveLength(2)
  })

  it('exposes secondary controls and delegates chip clearing', async () => {
    const clearChip = vi.fn()
    const wrapper = mountControls({
      filtersOpen: false,
      secondaryFilterCount: 2,
      activeFilterChips: [{ key: 'engine', label: 'Engine: llama.cpp', clear: clearChip }],
    })

    await wrapper.get('button[aria-expanded="false"]').trigger('click')
    await wrapper.get('.search-chip').trigger('click')

    expect(wrapper.emitted('update:filtersOpen')).toEqual([[true]])
    expect(clearChip).toHaveBeenCalledOnce()
    expect(wrapper.emitted('import-audio')).toBeUndefined()
    expect(wrapper.text()).toContain('Filters (2)')
    wrapper.unmount()
  })

  it('keeps one primary Search action and renders the filter controls in the shared toolbar', () => {
    const wrapper = mountControls({
      filtersOpen: true,
      showFormatSelect: true,
      secondaryFilterCount: 2,
    })

    expect(wrapper.get('.page-toolbar').attributes('aria-label')).toBe('Model search')
    expect(wrapper.get('.page-toolbar__search').exists()).toBe(true)
    expect(wrapper.get('.page-toolbar__actions').exists()).toBe(true)
    expect(wrapper.findAll('.page-toolbar__actions button').map((button) => button.text())).toEqual(
      ['Search', 'Hide filters', 'Reset filters'],
    )
    wrapper.unmount()
  })
})
