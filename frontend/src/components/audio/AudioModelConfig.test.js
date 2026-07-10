import { describe, it, expect, beforeEach, vi } from 'vitest'
import { mount, flushPromises } from '@vue/test-utils'

const toastAdd = vi.fn()
const listReferenceAudio = vi.fn()
const uploadReferenceAudio = vi.fn()
const deleteReferenceAudio = vi.fn()
const scanEngineParams = vi.fn()

vi.mock('primevue/usetoast', () => ({
  useToast: () => ({ add: toastAdd }),
}))

vi.mock('@/stores/models', () => ({
  useModelStore: () => ({
    listReferenceAudio,
    uploadReferenceAudio,
    deleteReferenceAudio,
  }),
}))

vi.mock('@/stores/engines', () => ({
  useEnginesStore: () => ({
    scanEngineParams,
  }),
}))

import AudioModelConfig from './AudioModelConfig.vue'

const buttonStub = {
  props: ['label', 'icon', 'text', 'severity', 'loading', 'outlined', 'rounded'],
  emits: ['click'],
  template:
    '<button :data-label="label" :aria-label="$attrs[`aria-label`]" @click="$emit(`click`)">{{ label }}</button>',
}

const textInputStub = {
  props: ['modelValue', 'placeholder'],
  emits: ['update:modelValue'],
  template:
    '<input :value="modelValue ?? ``" :placeholder="placeholder" @input="$emit(`update:modelValue`, $event.target.value)" />',
}

const textareaStub = {
  props: ['modelValue', 'readonly', 'rows'],
  emits: ['update:modelValue'],
  template:
    '<textarea :value="modelValue ?? ``" :readonly="readonly" @input="$emit(`update:modelValue`, $event.target.value)" />',
}

function selectStub(name = 'select') {
  return {
    props: ['modelValue', 'options', 'placeholder', 'editable', 'showClear'],
    emits: ['update:modelValue'],
    template: `<${name} :value="modelValue ?? ''" @change="$emit('update:modelValue', $event.target.value)">
      <option value=""></option>
      <option v-for="opt in options || []" :key="opt.value ?? opt" :value="opt.value ?? opt">{{ opt.label ?? opt }}</option>
    </${name}>`,
  }
}

function makeProps(overrides = {}) {
  return {
    config: {
      engine: 'audio_cpp',
      family: 'omnivoice',
      task: 'tts',
      voice_presets: {
        assistant: { voice_ref: '', reference_text: '' },
      },
      default_voice_preset: 'assistant',
      ...overrides.config,
    },
    paramRegistry: {
      sections: [],
      scan_error: null,
      scan_pending: false,
      task_profile: {
        label: 'OmniVoice',
        workflows: ['clone'],
        summary: 'Clone from reference audio.',
      },
      request_field_groups: [],
      request_defaults_key: 'speech_defaults',
      api_endpoint: '/v1/audio/speech',
      voice_preset_field_defs: [
        { key: 'voice_ref', label: 'Reference audio (WAV)', type: 'path' },
        { key: 'reference_text', label: 'Reference transcript', type: 'textarea' },
      ],
      ...overrides.paramRegistry,
    },
    llamaSwapStableId: 'audio-demo',
    modelId: 'audio/demo',
    ...overrides,
  }
}

function mountComponent(overrides = {}) {
  return mount(AudioModelConfig, {
    props: makeProps(overrides),
    global: {
      directives: {
        tooltip: () => {},
      },
      stubs: {
        Button: buttonStub,
        Tag: { props: ['value'], template: '<span>{{ value }}</span>' },
        InputText: textInputStub,
        InputNumber: textInputStub,
        InputSwitch: {
          props: ['modelValue'],
          emits: ['update:modelValue'],
          template:
            '<input type="checkbox" :checked="Boolean(modelValue)" @change="$emit(`update:modelValue`, $event.target.checked)" />',
        },
        Dropdown: selectStub(),
        Message: { template: '<div><slot /></div>' },
        Textarea: textareaStub,
        AudioParamField: { template: '<div class="audio-param-field-stub" />' },
      },
    },
  })
}

function openDefaultsTab(wrapper) {
  const tab = wrapper.findAll('button').find((button) => button.text().includes('Defaults'))
  if (!tab) {
    throw new Error('Defaults tab button not found')
  }
  return tab.trigger('click')
}

describe('AudioModelConfig reference audio', () => {
  beforeEach(() => {
    toastAdd.mockReset()
    listReferenceAudio.mockReset()
    uploadReferenceAudio.mockReset()
    deleteReferenceAudio.mockReset()
    scanEngineParams.mockReset()
    listReferenceAudio.mockResolvedValue([
      {
        path: 'refs/voice.wav',
        filename: 'voice.wav',
        size_bytes: 2048,
        used_by: ['voice_presets.assistant.voice_ref'],
      },
    ])
    uploadReferenceAudio.mockResolvedValue({
      path: 'refs/new.wav',
      filename: 'new.wav',
      size_bytes: 1024,
    })
    deleteReferenceAudio.mockResolvedValue(undefined)
  })

  it('loads and renders reference audio library on the Defaults tab', async () => {
    const wrapper = mountComponent()
    await flushPromises()

    expect(listReferenceAudio).toHaveBeenCalledWith('audio/demo')

    await openDefaultsTab(wrapper)
    await flushPromises()

    expect(wrapper.text()).toContain('Reference audio library')
    expect(wrapper.text()).toContain('refs/voice.wav')
    expect(wrapper.text()).toContain('2.0 KB')
    expect(wrapper.text()).toContain('voice_presets.assistant.voice_ref')
  })

  it('uploads a WAV through the hidden file input', async () => {
    const wrapper = mountComponent()
    await flushPromises()
    await openDefaultsTab(wrapper)
    await flushPromises()

    const file = new File(['wav'], 'new.wav', { type: 'audio/wav' })
    const input = wrapper.find('input[type="file"]')
    Object.defineProperty(input.element, 'files', { value: [file] })
    await input.trigger('change')
    await flushPromises()

    expect(uploadReferenceAudio).toHaveBeenCalledWith('audio/demo', file)
    expect(listReferenceAudio.mock.calls.length).toBeGreaterThanOrEqual(2)
    expect(toastAdd).toHaveBeenCalledWith(
      expect.objectContaining({ severity: 'success', summary: 'Reference audio uploaded' }),
    )
  })

  it('deletes a reference clip from the library', async () => {
    const wrapper = mountComponent()
    await flushPromises()
    await openDefaultsTab(wrapper)
    await flushPromises()

    const deleteButton = wrapper
      .findAll('button')
      .find((button) => button.attributes('aria-label') === 'Delete reference audio')
    expect(deleteButton).toBeTruthy()
    await deleteButton.trigger('click')
    await flushPromises()

    expect(deleteReferenceAudio).toHaveBeenCalledWith('audio/demo', 'voice.wav')
    expect(toastAdd).toHaveBeenCalledWith(
      expect.objectContaining({ severity: 'success', summary: 'Reference audio deleted' }),
    )
  })

  it('sets voice_ref on the first preset when Use in preset is clicked', async () => {
    const config = makeProps().config
    const wrapper = mountComponent({ config })
    await flushPromises()
    await openDefaultsTab(wrapper)
    await flushPromises()

    const useButton = wrapper.find('button[data-label="Use in preset"]')
    await useButton.trigger('click')

    expect(config.voice_presets.assistant.voice_ref).toBe('refs/voice.wav')
    expect(toastAdd).toHaveBeenCalledWith(
      expect.objectContaining({ severity: 'info', summary: 'Preset updated' }),
    )
  })
})
