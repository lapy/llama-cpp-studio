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
  const { config, paramRegistry, ...rest } = overrides
  return {
    config: {
      engine: 'audio_cpp',
      family: 'omnivoice',
      task: 'tts',
      voice_presets: {
        assistant: { voice_ref: '', reference_text: '' },
      },
      default_voice_preset: 'assistant',
      ...config,
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
      instructions_policy: 'soft_tags',
      supports_voice_presets: true,
      voice_preset_field_defs: [
        { key: 'voice_ref', label: 'Reference audio (WAV)', type: 'path' },
        { key: 'reference_text', label: 'Reference transcript', type: 'textarea' },
      ],
      ...paramRegistry,
    },
    llamaSwapStableId: 'audio-demo',
    modelId: 'audio/demo',
    ...rest,
  }
}

function mountComponent(overrides = {}) {
  return mount(AudioModelConfig, {
    props: makeProps(overrides),
    global: {
      directives: {
        tooltip: (el, binding) => {
          el.setAttribute('data-tooltip', String(binding.value || ''))
        },
      },
      stubs: {
        Button: buttonStub,
        Tag: { props: ['value'], template: '<span>{{ value }}</span>' },
        InputText: textInputStub,
        InputNumber: textInputStub,
        InputSwitch: {
          props: ['modelValue', 'inputId'],
          emits: ['update:modelValue'],
          template:
            '<input :id="inputId" type="checkbox" :checked="Boolean(modelValue)" @change="$emit(`update:modelValue`, $event.target.checked)" />',
        },
        Dropdown: selectStub(),
        Message: { template: '<div><slot /></div>' },
        Textarea: textareaStub,
        AudioParamField: { template: '<div class="audio-param-field-stub" />' },
      },
    },
  })
}

function openAssetsTab(wrapper) {
  const tab = wrapper.findAll('button').find((button) => button.text().includes('Assets'))
  if (!tab) {
    throw new Error('Assets tab button not found')
  }
  return tab.trigger('click')
}

function openRuntimeTab(wrapper) {
  const tab = wrapper.findAll('button').find((button) => button.text().includes('Runtime'))
  if (!tab) {
    throw new Error('Runtime tab button not found')
  }
  return tab.trigger('click')
}

function openDefaultsTab(wrapper) {
  const tab = wrapper.findAll('button').find((button) => button.text().includes('Defaults'))
  if (!tab) {
    throw new Error('Defaults tab button not found')
  }
  return tab.trigger('click')
}

function openApiTab(wrapper) {
  const tab = wrapper.findAll('button').find((button) => button.text().includes('API'))
  if (!tab) {
    throw new Error('API tab button not found')
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
        path: '/app/data/models/audio-cpp/reference-audio/audio-demo/refs/voice.wav',
        relative_path: 'refs/voice.wav',
        display_path: 'refs/voice.wav',
        filename: 'voice.wav',
        size_bytes: 2048,
        used_by: ['voice_presets.assistant.voice_ref'],
      },
    ])
    uploadReferenceAudio.mockResolvedValue({
      path: '/app/data/models/audio-cpp/reference-audio/audio-demo/refs/new.wav',
      relative_path: 'refs/new.wav',
      display_path: 'refs/new.wav',
      filename: 'new.wav',
      size_bytes: 1024,
    })
    deleteReferenceAudio.mockResolvedValue(undefined)
  })

  it('loads and renders reference audio library on the Assets tab', async () => {
    const wrapper = mountComponent()
    await flushPromises()

    expect(listReferenceAudio).toHaveBeenCalledWith('audio/demo')

    await openAssetsTab(wrapper)
    await flushPromises()

    expect(wrapper.text()).toContain('Reference audio')
    expect(wrapper.text()).toContain('Max upload: 60 MB')
    expect(wrapper.text()).toContain('refs/voice.wav')
    expect(wrapper.text()).toContain('2.0 KB')
    expect(wrapper.text()).toContain('voice_presets.assistant.voice_ref')
  })

  it('uploads a WAV through the hidden file input', async () => {
    const wrapper = mountComponent()
    await flushPromises()
    await openAssetsTab(wrapper)
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

  it('rejects oversized WAV uploads before calling the API', async () => {
    const wrapper = mountComponent()
    await flushPromises()
    await openAssetsTab(wrapper)
    await flushPromises()

    const file = { name: 'huge.wav', size: 60 * 1024 * 1024 + 1, type: 'audio/wav' }
    const input = wrapper.find('input[type="file"]')
    Object.defineProperty(input.element, 'files', { value: [file] })
    await input.trigger('change')
    await flushPromises()

    expect(uploadReferenceAudio).not.toHaveBeenCalled()
    expect(toastAdd).toHaveBeenCalledWith(
      expect.objectContaining({ severity: 'warn', summary: 'Upload too large' }),
    )
  })

  it('keeps preset name and voice_id inputs editable across keystrokes', async () => {
    const config = {
      engine: 'audio_cpp',
      family: 'kokoro',
      task: 'tts',
      voice_presets: {
        assistant: { voice_id: '' },
      },
      default_voice_preset: 'assistant',
    }
    const wrapper = mountComponent({
      config,
      paramRegistry: {
        sections: [],
        scan_error: null,
        scan_pending: false,
        task_profile: {
          label: 'Kokoro',
          workflows: ['builtin'],
          summary: 'Built-in voices.',
        },
        request_field_groups: [
          {
            id: 'voice',
            label: 'Voice',
            fields: [
              {
                key: 'voice_id',
                label: 'Built-in voice id',
                type: 'string',
                placeholder: 'alba',
                preset_field: true,
              },
            ],
          },
        ],
        request_defaults_key: 'speech_defaults',
        api_endpoint: '/v1/audio/speech',
      },
    })
    await flushPromises()
    await openAssetsTab(wrapper)
    await flushPromises()

    const nameInput = wrapper.find('input[placeholder="preset-name"]')
    const voiceIdInput = wrapper.find('input[placeholder="alba"]')
    expect(nameInput.exists()).toBe(true)
    expect(voiceIdInput.exists()).toBe(true)

    const nameNode = nameInput.element
    const voiceIdNode = voiceIdInput.element

    await nameInput.setValue('assist')
    await nameInput.setValue('assistant-2')
    expect(nameInput.element).toBe(nameNode)
    expect(config.voice_presets.assistant).toBeTruthy()

    await nameInput.trigger('blur')
    await flushPromises()
    expect(config.voice_presets['assistant-2']).toEqual({ voice_id: '' })
    expect(config.voice_presets.assistant).toBeUndefined()

    const voiceIdAfterRename = wrapper.find('input[placeholder="alba"]')
    await voiceIdAfterRename.setValue('a')
    await voiceIdAfterRename.setValue('al')
    await voiceIdAfterRename.setValue('alba')
    expect(voiceIdAfterRename.element).toBe(voiceIdNode)
    expect(config.voice_presets['assistant-2'].voice_id).toBe('alba')
  })

  it('rescan passes modelId so the model profile is force-scanned', async () => {
    scanEngineParams.mockResolvedValue({
      ok: true,
      param_count: 4,
      profile_fingerprint: 'abc',
    })
    const wrapper = mountComponent({
      modelId: 'audio-cpp--demo',
      paramRegistry: {
        sections: [],
        scan_error: 'previous scan failed',
        scan_pending: false,
      },
    })
    await flushPromises()

    const rescan = wrapper
      .findAll('button')
      .find((button) => (button.attributes('data-label') || button.text()).includes('Rescan'))
    expect(rescan).toBeTruthy()
    await rescan.trigger('click')
    await flushPromises()

    expect(scanEngineParams).toHaveBeenCalledWith('audio_cpp', null, {
      modelId: 'audio-cpp--demo',
    })
    expect(toastAdd).toHaveBeenCalledWith(
      expect.objectContaining({ severity: 'success', summary: 'Parameters scanned' }),
    )
  })

  it('deletes a reference clip from the library', async () => {
    const wrapper = mountComponent()
    await flushPromises()
    await openAssetsTab(wrapper)
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
    await openAssetsTab(wrapper)
    await flushPromises()

    const useButton = wrapper.find('button[data-label="Use in preset"]')
    await useButton.trigger('click')

    expect(config.voice_presets.assistant.voice_ref).toBe(
      '/app/data/models/audio-cpp/reference-audio/audio-demo/refs/voice.wav',
    )
    expect(toastAdd).toHaveBeenCalledWith(
      expect.objectContaining({ severity: 'info', summary: 'Preset updated' }),
    )
  })

  it('shows common runtime settings first and hides advanced fields by default', async () => {
    const wrapper = mountComponent({
      paramRegistry: {
        sections: [
          {
            params: [
              { key: 'family', label: 'Family', type: 'string', scope: 'model', required: true },
              { key: 'task', label: 'Task', type: 'string', scope: 'model', required: true },
              { key: 'mode', label: 'Mode', type: 'string', scope: 'model', required: true },
              { key: 'backend', label: 'Backend', type: 'string', scope: 'process' },
              { key: 'threads', label: 'Threads', type: 'int', scope: 'process' },
              { key: 'log_file', label: 'Log file', type: 'string', scope: 'process' },
            ],
          },
        ],
      },
    })
    await flushPromises()
    await openRuntimeTab(wrapper)

    expect(wrapper.text()).toContain('Common settings')
    expect(wrapper.text()).toContain('Family')
    expect(wrapper.text()).toContain('Backend')
    expect(wrapper.text()).toContain('Threads')
    expect(wrapper.text()).not.toContain('Log file')

    await wrapper.get('input#audio-runtime-advanced').setValue(true)
    await flushPromises()

    expect(wrapper.text()).toContain('Runtime & startup')
    expect(wrapper.text()).toContain('Log file')
  })

  it('collapses the raw setParams preview until requested', async () => {
    const wrapper = mountComponent({
      config: {
        speech_defaults: { voice: 'assistant' },
      },
    })
    await flushPromises()
    await openDefaultsTab(wrapper)

    const toggle = wrapper.get('button.setparams-preview__toggle')
    expect(toggle.attributes('aria-expanded')).toBe('false')
    expect(wrapper.text()).toContain('filters.setParams')
    expect(wrapper.text()).not.toContain('"voice"')

    await toggle.trigger('click')
    await flushPromises()

    expect(toggle.attributes('aria-expanded')).toBe('true')
    expect(wrapper.text()).toContain('"voice": "assistant"')
    expect(wrapper.text()).toContain('Relative reference paths are resolved')
  })

  it('shows defaults apply hint and instructions policy guidance', async () => {
    const wrapper = mountComponent({
      paramRegistry: {
        request_field_groups: [
          {
            id: 'voice',
            label: 'Voice',
            fields: [{ key: 'instructions', label: 'Instructions', type: 'textarea' }],
          },
        ],
        instructions_policy: 'soft_tags',
        supports_voice_presets: true,
      },
    })
    await flushPromises()
    await openDefaultsTab(wrapper)

    expect(wrapper.text()).toContain('setParams')
    expect(wrapper.text()).toContain('comma-separated voice attributes')
  })

  it('surfaces required non-common params in Common settings', async () => {
    const wrapper = mountComponent({
      paramRegistry: {
        sections: [
          {
            params: [
              { key: 'family', label: 'Family', type: 'string', scope: 'model', required: true },
              { key: 'task', label: 'Task', type: 'string', scope: 'model', required: true },
              {
                key: 'package_flag',
                label: 'Package flag',
                type: 'string',
                scope: 'load_option',
                required: true,
              },
              { key: 'log_file', label: 'Log file', type: 'string', scope: 'process' },
            ],
          },
        ],
      },
    })
    await flushPromises()
    await openRuntimeTab(wrapper)

    expect(wrapper.text()).toContain('Package flag')
    expect(wrapper.text()).not.toContain('Log file')
  })

  it('links request-only docs to the Defaults tab', async () => {
    const wrapper = mountComponent({
      paramRegistry: {
        task_profile: {
          label: 'OmniVoice',
          workflows: ['clone'],
          summary: 'Clone from reference audio.',
        },
        sections: [
          {
            params: [
              {
                key: 'temperature',
                label: 'Temperature',
                type: 'float',
                scope: 'request_option',
                read_only: true,
                supported: true,
              },
            ],
          },
        ],
        request_field_groups: [
          {
            id: 'sampling',
            label: 'Sampling',
            fields: [{ key: 'temperature', label: 'Temperature', type: 'float' }],
          },
        ],
        supports_voice_presets: true,
      },
    })
    await flushPromises()
    await openApiTab(wrapper)

    expect(wrapper.text()).toContain('Request-only parameters')
    const editDefaults = wrapper
      .findAll('button')
      .find((button) => button.text().includes('Edit Defaults'))
    expect(editDefaults).toBeTruthy()
    await editDefaults.trigger('click')
    await flushPromises()

    expect(wrapper.text()).toContain('Speech synthesis defaults')
    expect(wrapper.text()).toContain('Sampling')
  })
})
