import { enableAutoUnmount, mount } from '@vue/test-utils'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import ModelConfigAudioProfiles from './ModelConfigAudioProfiles.vue'

enableAutoUnmount(afterEach)

const { toastAdd, copyText } = vi.hoisted(() => ({
  toastAdd: vi.fn(),
  copyText: vi.fn(),
}))

vi.mock('primevue/usetoast', () => ({ useToast: () => ({ add: toastAdd }) }))
vi.mock('@/utils/clipboard', () => ({ copyText }))

const ButtonStub = {
  props: ['label', 'icon', 'size', 'severity', 'outlined', 'type'],
  emits: ['click'],
  template: '<button :data-label="label" @click="$emit(`click`)">{{ label }}</button>',
}

const InputStub = {
  props: ['modelValue', 'placeholder', 'readonly', 'type'],
  emits: ['update:modelValue'],
  template:
    '<input :type="type || `text`" :value="modelValue ?? ``" :placeholder="placeholder" :readonly="readonly" @input="$emit(`update:modelValue`, $event.target.value)" />',
}

const NumberStub = {
  props: ['modelValue', 'minFractionDigits', 'maxFractionDigits', 'min', 'max'],
  emits: ['update:modelValue'],
  template:
    '<input type="number" :value="modelValue ?? ``" @input="$emit(`update:modelValue`, $event.target.value === `` ? null : Number($event.target.value))" />',
}

const ToggleStub = {
  props: ['modelValue'],
  emits: ['update:modelValue'],
  template:
    '<input type="checkbox" :checked="Boolean(modelValue)" @change="$emit(`update:modelValue`, $event.target.checked)" />',
}

const SelectStub = {
  props: [
    'modelValue',
    'options',
    'optionLabel',
    'optionValue',
    'placeholder',
    'showClear',
    'editable',
  ],
  emits: ['update:modelValue'],
  template: `<select :value="modelValue ?? ''" @change="$emit('update:modelValue', $event.target.value)">
    <option v-for="option in options || []" :key="option.value ?? option" :value="option.value ?? option">{{ option.label ?? option }}</option>
  </select>`,
}

const TextareaStub = {
  props: ['modelValue', 'rows', 'readonly', 'autoResize'],
  emits: ['update:modelValue'],
  template:
    '<textarea :value="modelValue ?? ``" :readonly="readonly" @input="$emit(`update:modelValue`, $event.target.value)" />',
}

const MessageStub = {
  template: '<div class="message-stub"><slot /></div>',
}

const groups = [
  {
    id: 'voice',
    label: 'Voice & reference',
    fields: [
      { key: 'voice_ref', label: 'Reference audio', type: 'path', hint: 'Server-local WAV path.' },
      { key: 'speed', label: 'Speed', type: 'float', minimum: 0.5, maximum: 2 },
    ],
  },
  {
    id: 'sampling',
    label: 'Sampling',
    fields: [
      { key: 'seed', label: 'Seed', type: 'int' },
      { key: 'temperature', label: 'Temperature', type: 'string' },
      { key: 'notes', label: 'Notes', type: 'textarea' },
      { key: 'denoise', label: 'Denoise', type: 'bool' },
    ],
  },
]

const fieldValues = {
  voice_ref: 'samples/reference.wav',
  speed: 1.25,
  seed: 7,
  temperature: 'warm',
  notes: 'Read this.',
  denoise: true,
}

const fieldOptions = {
  temperature: [
    { label: 'Warm', value: 'warm' },
    { label: 'Cold', value: 'cold' },
  ],
}

function resolveValue(field) {
  return fieldValues[field.key]
}

function resolveOptions(field) {
  return fieldOptions[field.key] || []
}

function mountProfiles(overrides = {}) {
  return mount(ModelConfigAudioProfiles, {
    props: {
      title: 'Speech synthesis defaults',
      endpoint: '/v1/audio/speech',
      guidance: '',
      profiled: true,
      groups,
      resolveValue,
      resolveOptions,
      defaultsPreview: null,
      apiExample: 'curl -X POST http://localhost/v1/audio/speech',
      modelId: 'my-model',
      capabilities: [],
      ...overrides,
    },
    global: {
      directives: {
        tooltip: (el, binding) => {
          el.setAttribute('data-tooltip', String(binding.value || ''))
        },
      },
      stubs: {
        Button: ButtonStub,
        InputText: InputStub,
        InputNumber: NumberStub,
        ToggleSwitch: ToggleStub,
        Select: SelectStub,
        Textarea: TextareaStub,
        Message: MessageStub,
      },
    },
  })
}

describe('ModelConfigAudioProfiles', () => {
  beforeEach(() => {
    toastAdd.mockReset()
    copyText.mockReset()
    copyText.mockResolvedValue(undefined)
  })

  it('renders the profile heading, endpoint, guidance and grouped fields', () => {
    const wrapper = mountProfiles({ guidance: 'Prefer tags in the instructions.' })

    expect(wrapper.text()).toContain('Speech synthesis defaults')
    expect(wrapper.text()).toContain('/v1/audio/speech')
    expect(wrapper.text()).toContain('Prefer tags in the instructions.')
    expect(wrapper.findAll('.tts-speech-group')).toHaveLength(2)
    expect(wrapper.text()).toContain('Voice & reference')
    expect(wrapper.text()).toContain('Sampling')
    expect(wrapper.text()).toContain('Reference audio')
    expect(wrapper.get('i[data-tooltip]').attributes('data-tooltip')).toContain(
      'Server-local WAV path.',
    )
  })

  it('renders current values with the field-type control', () => {
    const wrapper = mountProfiles()
    const rows = wrapper.findAll('.param-field')

    expect(rows[0].get('input').element.value).toBe('samples/reference.wav')
    expect(rows[1].get('input[type="number"]').element.value).toBe('1.25')
    expect(rows[2].get('input[type="number"]').element.value).toBe('7')
    expect(rows[3].get('select').element.value).toBe('warm')
    expect(rows[4].get('textarea').element.value).toBe('Read this.')
    expect(rows[5].get('input[type="checkbox"]').element.checked).toBe(true)
  })

  it('forwards edits as update-request-default events', async () => {
    const wrapper = mountProfiles()
    const rows = wrapper.findAll('.param-field')

    await rows[0].get('input').setValue('other.wav')
    expect(wrapper.emitted('update-request-default')?.at(-1)).toEqual([
      { field: groups[0].fields[0], value: 'other.wav' },
    ])

    await rows[5].get('input[type="checkbox"]').setValue(false)
    expect(wrapper.emitted('update-request-default')?.at(-1)).toEqual([
      { field: groups[1].fields[3], value: false },
    ])
  })

  it('shows the saved defaults preview on demand', async () => {
    const wrapper = mountProfiles({ defaultsPreview: { voice_id: 'M1', speed: 1 } })

    expect(wrapper.find('.setparams-preview__code').exists()).toBe(false)
    await wrapper.get('.setparams-preview__toggle').trigger('click')
    expect(wrapper.get('.setparams-preview__code').text()).toContain('"voice_id": "M1"')
    expect(wrapper.get('.setparams-preview__toggle').attributes('aria-expanded')).toBe('true')
  })

  it('renders and filters request-only parameters', async () => {
    const wrapper = mountProfiles({
      capabilities: [
        { key: 'temperature', label: 'Temperature', description: 'Sampling randomness.' },
        { key: 'language', label: 'Language' },
      ],
    })

    expect(wrapper.text()).toContain('Request-only parameters')
    expect(wrapper.findAll('.request-cap-item')).toHaveLength(2)

    await wrapper.get('input[aria-label="Filter request parameters"]').setValue('zzz')
    expect(wrapper.findAll('.request-cap-item')).toHaveLength(0)
    expect(wrapper.text()).toContain('No request parameters match your filter.')

    await wrapper.get('input[aria-label="Filter request parameters"]').setValue('lang')
    expect(wrapper.findAll('.request-cap-item')).toHaveLength(1)
    expect(wrapper.text()).toContain('Language')
  })

  it('shows the fallback when the model has no request profile', () => {
    const wrapper = mountProfiles({ profiled: false })

    expect(wrapper.text()).toContain('No request defaults profile is available')
    expect(wrapper.find('.tts-subsection').exists()).toBe(false)
    expect(wrapper.text()).toContain('API example')
  })

  it('copies the API example and reports through the toast', async () => {
    const wrapper = mountProfiles()
    const copy = wrapper.findAll('button').find((button) => button.text() === 'Copy curl')
    await copy.trigger('click')
    await Promise.resolve()

    expect(copyText).toHaveBeenCalledWith('curl -X POST http://localhost/v1/audio/speech')
    expect(toastAdd).toHaveBeenCalledWith(
      expect.objectContaining({ severity: 'success', summary: 'Copied' }),
    )
  })
})
