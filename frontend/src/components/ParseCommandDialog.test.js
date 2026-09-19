import { describe, expect, it } from 'vitest'
import { mount, flushPromises } from '@vue/test-utils'

import ParseCommandDialog from './ParseCommandDialog.vue'

const catalogParams = [
  {
    key: 'temperature',
    label: 'Temperature',
    type: 'float',
    scalar_type: 'float',
    value_kind: 'scalar',
    primary_flag: '--temperature',
    flags: ['--temperature', '--temp'],
    supported: true,
  },
  {
    key: 'ctx_size',
    label: 'Context Size',
    type: 'int',
    scalar_type: 'int',
    value_kind: 'scalar',
    primary_flag: '--ctx-size',
    flags: ['--ctx-size'],
    supported: true,
  },
  {
    key: 'host',
    label: 'Host',
    type: 'string',
    value_kind: 'scalar',
    primary_flag: '--host',
    flags: ['--host'],
    reserved: false,
    supported: true,
  },
  {
    key: 'port',
    label: 'Port',
    type: 'int',
    value_kind: 'scalar',
    primary_flag: '--port',
    flags: ['--port'],
    reserved: true,
    supported: true,
  },
  {
    key: 'mirostat',
    label: 'Mirostat',
    type: 'int',
    value_kind: 'scalar',
    primary_flag: '--mirostat',
    flags: ['--mirostat'],
    supported: false,
  },
]

const buttonStub = {
  props: ['label', 'icon', 'text', 'severity', 'loading', 'outlined', 'rounded', 'disabled'],
  emits: ['click'],
  template: '<button :data-label="label" :disabled="disabled" @click="$emit(`click`)">{{ label }}</button>',
}

function mountDialog(props = {}) {
  return mount(ParseCommandDialog, {
    props: {
      visible: true,
      catalogParams,
      currentValues: {},
      customArgs: '',
      ...props,
    },
    global: {
      stubs: {
        Button: buttonStub,
        Dialog: {
          props: ['visible', 'header'],
          emits: ['update:visible'],
          template: '<div class="dialog-stub"><slot /><slot name="footer" /></div>',
        },
        Checkbox: {
          props: ['modelValue', 'inputId', 'binary'],
          emits: ['update:modelValue'],
          template:
            '<input type="checkbox" :id="inputId" :checked="Boolean(modelValue)" @change="$emit(`update:modelValue`, $event.target.checked)" />',
        },
        ToggleSwitch: {
          props: ['modelValue', 'inputId'],
          emits: ['update:modelValue'],
          template:
            '<input :id="inputId" type="checkbox" :checked="Boolean(modelValue)" @change="$emit(`update:modelValue`, $event.target.checked)" />',
        },
        Tag: { props: ['value'], template: '<span>{{ value }}</span>' },
        Message: { template: '<div><slot /></div>' },
        Textarea: {
          props: ['modelValue', 'id'],
          emits: ['update:modelValue'],
          template:
            '<textarea :id="id" :value="modelValue ?? ``" @input="$emit(`update:modelValue`, $event.target.value)" />',
        },
      },
    },
  })
}

describe('ParseCommandDialog', () => {
  it('live-previews importable params and excludes Studio-owned host/port', async () => {
    const wrapper = mountDialog()
    await wrapper.get('#parse-command-text').setValue(
      'llama-server --host 0.0.0.0 --port 8081 --ctx-size 8192 --temp 0.4',
    )
    await flushPromises()

    expect(wrapper.text()).toContain('Context Size')
    expect(wrapper.text()).toContain('Temperature')
    expect(wrapper.text()).toContain('Skipped — Studio managed')
    expect(wrapper.text()).toContain('--host 0.0.0.0')
    expect(wrapper.text()).toContain('--port 8081')
    expect(wrapper.text()).not.toContain('import-host')
    expect(wrapper.find('#import-host').exists()).toBe(false)
    expect(wrapper.find('#import-port').exists()).toBe(false)
  })

  it('lists unknown and deprecated flags without selecting them', async () => {
    const wrapper = mountDialog()
    await wrapper.get('#parse-command-text').setValue('--temp 0.2 --mirostat 2 --fancy-old-flag 1')
    await flushPromises()

    expect(wrapper.text()).toContain('Unrecognized')
    expect(wrapper.text()).toContain('--fancy-old-flag 1')
    expect(wrapper.text()).toContain('Skipped — not in this build')
    expect(wrapper.get('#import-unsup-mirostat').element.checked).toBe(false)
    expect(wrapper.get('#import-temperature').element.checked).toBe(true)
  })

  it('emits only confirmed catalog values on apply', async () => {
    const wrapper = mountDialog()
    await wrapper.get('#parse-command-text').setValue(
      '--host 127.0.0.1 --port 9 --temp 0.9 --mirostat 1 --unknown-x',
    )
    await flushPromises()

    await wrapper.get('button[data-label="Apply 1 parameter"]').trigger('click')
    const apply = wrapper.emitted('apply')?.[0]?.[0]
    expect(apply.params).toEqual([{ key: 'temperature', value: 0.9 }])
    expect(apply.customArgs).toBeUndefined()
  })

  it('can append unrecognized tokens to custom args when opted in', async () => {
    const wrapper = mountDialog({ customArgs: '--keep-me' })
    await wrapper.get('#parse-command-text').setValue('--temp 0.1 --unknown-x 3')
    await flushPromises()
    await wrapper.get('#parse-append-unknown').setValue(true)
    await wrapper.get('button[data-label="Apply 1 parameter"]').trigger('click')
    const apply = wrapper.emitted('apply')?.[0]?.[0]
    expect(apply.params).toEqual([{ key: 'temperature', value: 0.1 }])
    expect(apply.customArgs).toBe('--keep-me --unknown-x 3')
  })
})
