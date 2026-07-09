import { describe, it, expect } from 'vitest'
import { mount } from '@vue/test-utils'

import AudioParamField from './AudioParamField.vue'

const dropdownStub = {
  props: ['modelValue', 'options', 'disabled'],
  template: '<select class="dropdown-stub" :disabled="disabled" @change="$emit(`update:modelValue`, $event.target.value)"><option /></select>',
}

const inputNumberStub = {
  props: ['modelValue', 'disabled', 'minFractionDigits', 'maxFractionDigits'],
  template: '<input type="number" class="number-stub" :disabled="disabled" :value="modelValue" @input="$emit(`update:modelValue`, Number($event.target.value))" />',
}

const inputSwitchStub = {
  props: ['modelValue', 'disabled'],
  template: '<input type="checkbox" class="switch-stub" :disabled="disabled" :checked="modelValue" @change="$emit(`update:modelValue`, $event.target.checked)" />',
}

const chipsStub = {
  props: ['modelValue', 'disabled'],
  template: '<input class="chips-stub" :disabled="disabled" :value="(modelValue || []).join(`,`) " @input="$emit(`update:modelValue`, $event.target.value.split(`,`))" />',
}

const textareaStub = {
  props: ['modelValue', 'disabled'],
  emits: ['update:modelValue'],
  template: '<textarea class="textarea-stub" :disabled="disabled" :value="modelValue" @input="$emit(`update:modelValue`, $event.target.value)" />',
}

const inputTextStub = {
  props: ['modelValue', 'placeholder', 'disabled'],
  template: '<input class="text-stub" :disabled="disabled" :placeholder="placeholder" :value="modelValue" @input="$emit(`update:modelValue`, $event.target.value)" />',
}

function mountField(param, props = {}) {
  return mount(AudioParamField, {
    props: {
      id: 'field-1',
      param,
      modelValue: props.modelValue ?? null,
      options: props.options || [],
      disabled: props.disabled || false,
    },
    global: {
      stubs: {
        Dropdown: dropdownStub,
        InputNumber: inputNumberStub,
        InputSwitch: inputSwitchStub,
        Chips: chipsStub,
        Textarea: textareaStub,
        InputText: inputTextStub,
      },
    },
  })
}

describe('AudioParamField', () => {
  it('renders Dropdown when options are provided', () => {
    const wrapper = mountField(
      { key: 'language', type: 'select' },
      { options: [{ label: 'English', value: 'en' }] },
    )
    expect(wrapper.find('.dropdown-stub').exists()).toBe(true)
  })

  it('renders InputNumber for int and float types', () => {
    const intField = mountField({ key: 'threads', type: 'int' }, { modelValue: 4 })
    expect(intField.find('.number-stub').exists()).toBe(true)

    const floatField = mountField({ key: 'temperature', type: 'float' }, { modelValue: 0.8 })
    expect(floatField.find('.number-stub').exists()).toBe(true)
  })

  it('renders InputSwitch for bool type', () => {
    const wrapper = mountField({ key: 'lazy_load', type: 'bool' }, { modelValue: true })
    expect(wrapper.find('.switch-stub').exists()).toBe(true)
  })

  it('renders Chips for repeatable value_kind', () => {
    const wrapper = mountField(
      { key: 'ctx', type: 'string', value_kind: 'repeatable' },
      { modelValue: ['a', 'b'] },
    )
    expect(wrapper.find('.chips-stub').exists()).toBe(true)
  })

  it('renders Textarea for json type and emits update:json', async () => {
    const wrapper = mountField(
      { key: 'payload', type: 'json' },
      { modelValue: { a: 1 } },
    )
    const textarea = wrapper.find('.textarea-stub')
    expect(textarea.exists()).toBe(true)
    expect(textarea.element.value).toContain('"a": 1')
    await textarea.setValue('{"b":2}')
    expect(wrapper.emitted('update:json')?.[0]).toEqual(['{"b":2}'])
  })

  it('falls back to InputText with default placeholder', () => {
    const wrapper = mountField({ key: 'host', type: 'string', default: '127.0.0.1' })
    const input = wrapper.find('.text-stub')
    expect(input.exists()).toBe(true)
    expect(input.attributes('placeholder')).toBe('127.0.0.1')
  })

  it('respects disabled prop across control types', () => {
    const wrapper = mountField({ key: 'lazy_load', type: 'bool' }, { disabled: true })
    expect(wrapper.find('.switch-stub').attributes('disabled')).toBeDefined()
  })
})
