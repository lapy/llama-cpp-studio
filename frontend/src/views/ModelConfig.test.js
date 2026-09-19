import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest'
import { mount, flushPromises } from '@vue/test-utils'

import ModelConfig from './ModelConfig.vue'

const toastAdd = vi.fn()
const fetchModels = vi.fn()
const fetchSwapConfigStale = vi.fn()
const fetchGpuList = vi.fn()
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
  useRouter: () => ({ push: vi.fn(), replace: vi.fn() }),
}))

vi.mock('primevue/usetoast', () => ({
  useToast: () => ({ add: toastAdd }),
}))

const { storeQuantization } = vi.hoisted(() => ({
  storeQuantization: {
    id: 'model-1',
    display_name: 'Test Model',
    base_model_name: 'Test Model',
    huggingface_id: 'org/model',
    quantization: 'Q4_K_M',
    format: 'gguf',
    compatible_engines: undefined,
    artifact: {},
  },
}))

vi.mock('@/stores/models', () => ({
  useModelStore: () => ({
    models: [
      {
        base_model_name: 'Test Model',
        huggingface_id: 'org/model',
        quantizations: [storeQuantization],
      },
    ],
    allQuantizations: [],
    fetchModels,
  }),
}))

const fetchEngineDescriptors = vi.fn()

vi.mock('@/stores/engines', () => ({
  useEnginesStore: () => ({
    swapConfigStale: { applicable: false, stale: false },
    engineDescriptors: [],
    fetchSwapConfigStale,
    fetchGpuList,
    fetchEngineDescriptors,
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
        ToggleSwitch: {
          props: ['modelValue', 'inputId'],
          emits: ['update:modelValue'],
          template:
            '<input :id="inputId" type="checkbox" :checked="Boolean(modelValue)" @change="$emit(`update:modelValue`, $event.target.checked)" />',
        },
        Select: selectStub(),
        InputTags: {
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
        Checkbox: {
          props: ['modelValue', 'inputId', 'binary'],
          emits: ['update:modelValue'],
          template:
            '<input type="checkbox" :id="inputId" :checked="Boolean(modelValue)" @change="$emit(`update:modelValue`, $event.target.checked)" />',
        },
        Slider: true,
        LoadingState: { template: '<div>loading</div>' },
        EmptyState: { template: '<div><slot /></div>' },
        PageHeader: { template: '<div><slot name="start" /><slot name="title" /><slot name="actions" /></div>' },
        Dialog: {
          props: ['visible', 'header'],
          emits: ['update:visible', 'show'],
          template:
            '<div v-if="visible" class="dialog-stub"><slot /><slot name="footer" /></div>',
        },
      },
    },
  })
}

describe('ModelConfig', () => {
  beforeEach(() => {
    vi.useFakeTimers()
    storeQuantization.format = 'gguf'
    storeQuantization.quantization = 'Q4_K_M'
    storeQuantization.compatible_engines = undefined
    storeQuantization.artifact = {}
    toastAdd.mockReset()
    fetchModels.mockReset()
    fetchSwapConfigStale.mockReset()
    fetchGpuList.mockReset()
    fetchEngineDescriptors.mockReset()
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
                  {
                    key: 'ctx_size',
                    label: 'Context Size',
                    type: 'int',
                    scalar_type: 'int',
                    value_kind: 'scalar',
                    default: 0,
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
      if (url === '/api/gpu-list') {
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
    fetchGpuList.mockResolvedValue({
      vendor: null,
      device_count: 0,
      gpus: [],
      cpu_only_mode: true,
    })
    fetchEngineDescriptors.mockResolvedValue([
      {
        id: 'llama_cpp',
        label: 'llama.cpp',
        artifact_formats: ['gguf'],
        enabled: true,
      },
    ])

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

    await wrapper.get('button[data-label="Live preview"]').trigger('click')
    await flushPromises()
    await vi.runAllTimersAsync()
    await flushPromises()

    expect(axios.post).toHaveBeenLastCalledWith(
      '/api/models/model-1/preview-llama-swap-cmd',
      {
        engine: 'llama_cpp',
        engines: {
          llama_cpp: {
            temperature: 0.7,
            swap_env: {},
          },
        },
      },
      expect.objectContaining({ signal: expect.any(AbortSignal) }),
    )

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

  it('imports parsed command flags into the form and skips Studio-owned host/port', async () => {
    const wrapper = mountView()
    await flushPromises()
    await vi.runAllTimersAsync()
    await flushPromises()

    await wrapper.get('button[data-label="Import command"]').trigger('click')
    await flushPromises()

    const commandInput = wrapper.get('#parse-command-text')
    await commandInput.setValue('--host 0.0.0.0 --port 8081 --temp 0.55 --ctx-size 4096')
    await flushPromises()

    expect(wrapper.text()).toContain('Skipped — Studio managed')
    expect(wrapper.find('#import-host').exists()).toBe(false)
    expect(wrapper.find('#import-port').exists()).toBe(false)

    await wrapper.get('button[data-label="Apply 2 parameters"]').trigger('click')
    await flushPromises()

    expect(wrapper.text()).toContain('Unsaved changes')
    expect(wrapper.get('input[placeholder="0.8"]').element.value).toBe('0.55')
    expect(wrapper.vm).toBeTruthy()
    expect(wrapper.text()).toContain('ctx_size')
    expect(wrapper.text()).not.toContain('0.0.0.0')
  })

  it('imports parsed environment variables into llama-swap env and skips Studio-owned names', async () => {
    const wrapper = mountView()
    await flushPromises()
    await vi.runAllTimersAsync()
    await flushPromises()

    await wrapper.get('button[data-label="Import command"]').trigger('click')
    await flushPromises()
    await wrapper.get('#parse-command-text').setValue(
      'FLASHINFER_DISABLE_VERSION_CHECK=1 PORT=9 LLAMA_STUDIO_MODEL_PATH=/x sglang serve',
    )
    await flushPromises()

    expect(wrapper.find('#import-env-FLASHINFER_DISABLE_VERSION_CHECK').exists()).toBe(true)
    expect(wrapper.find('#import-env-PORT').exists()).toBe(false)
    await wrapper.get('button[data-label="Apply 1 env var"]').trigger('click')
    await flushPromises()

    expect(wrapper.text()).toContain('Unsaved changes')
    const envKeys = wrapper.findAll('input[placeholder="VAR_NAME"]').map((input) => input.element.value)
    expect(envKeys).toContain('FLASHINFER_DISABLE_VERSION_CHECK')
    expect(envKeys).not.toContain('PORT')
    expect(envKeys).not.toContain('LLAMA_STUDIO_MODEL_PATH')
    const envValues = wrapper.findAll('input[placeholder="value"]').map((input) => input.element.value)
    expect(envValues).toContain('1')
  })

  it('keeps vLLM and SGLang selectable for safetensors with a stale engine list', async () => {
    storeQuantization.format = 'safetensors'
    storeQuantization.quantization = ''
    storeQuantization.compatible_engines = ['lmdeploy', '1cat_vllm']
    storeQuantization.artifact = { package_kind: 'hf_snapshot', format: 'safetensors' }

    const wrapper = mountView()
    await flushPromises()
    await vi.runAllTimersAsync()
    await flushPromises()

    const options = wrapper.vm.engineOptions
    const byId = Object.fromEntries(options.map((option) => [option.value, option]))
    expect(byId.vllm.disabled).toBe(false)
    expect(byId.sglang.disabled).toBe(false)
    expect(byId.sglang_v100.disabled).toBe(false)
    expect(byId.lmdeploy.disabled).toBe(false)
    expect(byId.vllm.disabledReason).toBe('')
    expect(byId.sglang.disabledReason).toBe('')
    expect(byId.sglang_v100.disabledReason).toBe('')
    expect(byId.llama_cpp.disabled).toBe(true)
    expect(byId.audio_cpp.disabled).toBe(true)
  })
})
