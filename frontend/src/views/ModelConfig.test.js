import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest'
import { mount, flushPromises } from '@vue/test-utils'

import ModelConfig from './ModelConfig.vue'

const toastAdd = vi.fn()
const fetchModels = vi.fn()
const fetchSwapConfigStale = vi.fn()
const applySwapConfig = vi.fn()
const markSwapConfigStaleLocal = vi.fn()

vi.mock('axios', () => ({
  default: {
    get: vi.fn(),
    post: vi.fn(),
    put: vi.fn(),
  },
}))

vi.mock('vue-router', () => ({
  useRoute: () => ({ params: { id: 'model-1' } }),
}))

vi.mock('primevue/usetoast', () => ({
  useToast: () => ({ add: toastAdd }),
}))

vi.mock('@/stores/models', () => ({
  useModelStore: () => ({
    models: [
      {
        base_model_name: 'Test Model',
        huggingface_id: 'org/model',
        quantizations: [
          {
            id: 'model-1',
            display_name: 'Test Model',
            base_model_name: 'Test Model',
            huggingface_id: 'org/model',
            quantization: 'Q4_K_M',
            format: 'gguf',
          },
        ],
      },
    ],
    allQuantizations: [],
    fetchModels,
  }),
}))

vi.mock('@/stores/engines', () => ({
  useEnginesStore: () => ({
    swapConfigStale: { applicable: false, stale: false },
    fetchSwapConfigStale,
    applySwapConfig,
    markSwapConfigStaleLocal,
  }),
}))

import axios from 'axios'

const buttonStub = {
  props: ['label', 'icon', 'text', 'severity', 'loading', 'outlined', 'rounded'],
  emits: ['click'],
  template: '<button :data-label="label" @click="$emit(`click`)">{{ label }}</button>',
}

const textInputStub = {
  props: ['modelValue', 'placeholder', 'disabled', 'id'],
  emits: ['update:modelValue'],
  template:
    '<input :id="id" :value="modelValue ?? ``" :placeholder="placeholder" :disabled="disabled" @input="$emit(`update:modelValue`, $event.target.value)" />',
}

const textareaStub = {
  props: ['modelValue', 'rows', 'readonly'],
  emits: ['update:modelValue'],
  template:
    '<textarea :value="modelValue ?? ``" :readonly="readonly" @input="$emit(`update:modelValue`, $event.target.value)" />',
}

function selectStub(name = 'select') {
  return {
    props: ['modelValue', 'options', 'disabled', 'id'],
    emits: ['update:modelValue'],
    template: `<${name} :id="id" :disabled="disabled" :value="modelValue ?? ''" @change="$emit('update:modelValue', $event.target.value)">
      <option value=""></option>
      <option v-for="opt in options || []" :key="opt.value ?? opt" :value="opt.value ?? opt">{{ opt.label ?? opt }}</option>
    </${name}>`,
  }
}

function mountView() {
  return mount(ModelConfig, {
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
          props: ['modelValue', 'inputId'],
          emits: ['update:modelValue'],
          template:
            '<input :id="inputId" type="checkbox" :checked="Boolean(modelValue)" @change="$emit(`update:modelValue`, $event.target.checked)" />',
        },
        Dropdown: selectStub(),
        Chips: {
          props: ['modelValue', 'id'],
          emits: ['update:modelValue'],
          template:
            '<input :id="id" :value="Array.isArray(modelValue) ? modelValue.join(`,`) : ``" @input="$emit(`update:modelValue`, $event.target.value ? $event.target.value.split(`,`) : [])" />',
        },
        Message: { template: '<div><slot /></div>' },
        Textarea: textareaStub,
        MultiSelect: {
          props: ['modelValue'],
          emits: ['update:modelValue'],
          template: '<div class="multiselect-stub" />',
        },
        Slider: true,
        LoadingState: { template: '<div>loading</div>' },
        EmptyState: { template: '<div><slot /></div>' },
        PageHeader: { template: '<div><slot name="start" /><slot name="title" /><slot name="actions" /></div>' },
      },
    },
  })
}

describe('ModelConfig', () => {
  beforeEach(() => {
    vi.useFakeTimers()
    toastAdd.mockReset()
    fetchModels.mockReset()
    fetchSwapConfigStale.mockReset()
    applySwapConfig.mockReset()
    markSwapConfigStaleLocal.mockReset()
    vi.mocked(axios.get).mockReset()
    vi.mocked(axios.post).mockReset()
    vi.mocked(axios.put).mockReset()

    vi.mocked(axios.get).mockImplementation((url) => {
      if (url === '/api/models/model-1/config') {
        return Promise.resolve({
          data: {
            engine: 'llama_cpp',
            engines: {
              llama_cpp: {
                temperature: 0.7,
                legacy_temp: 0.7,
              },
            },
          },
        })
      }
      if (url === '/api/models/model-1/limits') {
        return Promise.resolve({ data: null })
      }
      if (url === '/api/models/param-registry') {
        return Promise.resolve({
          data: {
            sections: [
              {
                id: 'sampling',
                label: 'Sampling',
                params: [
                  {
                    key: 'temperature',
                    label: 'Temperature',
                    type: 'float',
                    scalar_type: 'float',
                    value_kind: 'scalar',
                    default: 0.8,
                    primary_flag: '--temperature',
                    flags: ['--temperature', '--temp'],
                    supported: true,
                  },
                ],
              },
            ],
            scan_error: null,
            scan_pending: false,
          },
        })
      }
      if (url === '/api/models/model-1/saved-llama-swap-cmd') {
        return Promise.resolve({ data: { ok: true, cmd: 'saved-cmd' } })
      }
      if (url === '/api/gpu-info') {
        return Promise.resolve({
          data: {
            vendor: null,
            device_count: 0,
            gpus: [],
            cpu_only_mode: true,
          },
        })
      }
      throw new Error(`Unexpected GET ${url}`)
    })

    vi.mocked(axios.post).mockImplementation((url, payload) => {
      if (url === '/api/models/model-1/preview-llama-swap-cmd') {
        return Promise.resolve({ data: { ok: true, cmd: JSON.stringify(payload) } })
      }
      throw new Error(`Unexpected POST ${url}`)
    })

    vi.mocked(axios.put).mockResolvedValue({
      data: {
        engine: 'llama_cpp',
        engines: {
          llama_cpp: {
            temperature: 0.7,
          },
        },
      },
    })
  })

  afterEach(() => {
    vi.useRealTimers()
  })

  it('drops unrecognized saved keys from preview and save payloads', async () => {
    const wrapper = mountView()
    await flushPromises()
    await vi.runAllTimersAsync()
    await flushPromises()

    expect(wrapper.text()).toContain('legacy_temp')
    expect(axios.post).toHaveBeenLastCalledWith('/api/models/model-1/preview-llama-swap-cmd', {
      engine: 'llama_cpp',
      engines: {
        llama_cpp: {
          temperature: 0.7,
          swap_env: {},
        },
      },
    })

    await wrapper.get('button[data-label="Save Configuration"]').trigger('click')
    await flushPromises()

    expect(axios.put).toHaveBeenCalledWith('/api/models/model-1/config', {
      engine: 'llama_cpp',
      engines: {
        llama_cpp: {
          temperature: 0.7,
          swap_env: {},
        },
      },
    })
    expect(markSwapConfigStaleLocal).toHaveBeenCalled()
  })

  it('shows unsaved changes immediately after editing a parameter', async () => {
    const wrapper = mountView()
    await flushPromises()
    await vi.runAllTimersAsync()
    await flushPromises()

    expect(wrapper.text()).not.toContain('Unsaved changes')

    const temperatureInput = wrapper.get('input[placeholder="0.8"]')
    await temperatureInput.setValue('0.9')
    await flushPromises()

    expect(wrapper.text()).toContain('Unsaved changes')
  })
})
