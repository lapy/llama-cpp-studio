import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest'
import { enableAutoUnmount, mount, flushPromises } from '@vue/test-utils'

enableAutoUnmount(afterEach)

import ModelConfig from './ModelConfig.vue'

const toastAdd = vi.fn()
const fetchModels = vi.fn()
const startModel = vi.fn()
const fetchRuntimePlanState = vi.fn()
const fetchGpuList = vi.fn()
const invalidateRuntimePlan = vi.fn()

vi.mock('axios', () => ({
  default: {
    defaults: {},
    get: vi.fn(),
    post: vi.fn(),
    put: vi.fn(),
    interceptors: { request: { use: vi.fn() }, response: { use: vi.fn() } },
  },
}))

vi.mock('vue-router', () => ({
  useRoute: () => ({ params: { id: 'model-1' } }),
  useRouter: () => ({ push: vi.fn(), replace: vi.fn() }),
  onBeforeRouteLeave: () => {},
  RouterLink: { props: ['to'], template: '<a><slot /></a>' },
}))

vi.mock('primevue/usetoast', () => ({
  useToast: () => ({ add: toastAdd }),
}))

vi.mock('primevue/useconfirm', () => ({
  useConfirm: () => ({ require: vi.fn() }),
}))

const { storeQuantization, engineDescriptors, engineRuntime } = vi.hoisted(() => ({
  engineDescriptors: [],
  engineRuntime: {
    runtimePlanState: { applicable: false, stale: false },
    runtimePlan: { applicable: false, pending: false, changes: [], reason: null },
  },
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
    startModel,
  }),
}))

const fetchEngineDescriptors = vi.fn()

vi.mock('@/stores/engines', () => ({
  useEnginesStore: () => ({
    runtimePlanState: engineRuntime.runtimePlanState,
    runtimePlan: engineRuntime.runtimePlan,
    engineDescriptors,
    fetchRuntimePlanState,
    fetchRuntimePlan: vi.fn().mockResolvedValue(undefined),
    fetchGpuList,
    fetchEngineDescriptors,
    invalidateRuntimePlan,
  }),
}))

import axios from 'axios'
import { ENGINE_DESCRIPTORS } from '@/engines/registry'

const [PRIMARY_ENGINE_ID, SECONDARY_ENGINE_ID] = Object.keys(ENGINE_DESCRIPTORS)

const buttonStub = {
  props: ['label', 'icon', 'text', 'severity', 'loading', 'outlined', 'rounded', 'disabled'],
  emits: ['click'],
  template:
    '<button :data-label="label" :disabled="disabled" @click="!disabled && $emit(`click`)">{{ label }}</button>',
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
        PageHeader: {
          template: '<div><slot name="start" /><slot name="title" /><slot name="actions" /></div>',
        },
        Dialog: {
          props: ['visible', 'header'],
          emits: ['update:visible', 'show'],
          template: '<div v-if="visible" class="dialog-stub"><slot /><slot name="footer" /></div>',
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
    engineDescriptors.splice(0, engineDescriptors.length)
    engineRuntime.runtimePlanState.applicable = false
    engineRuntime.runtimePlanState.stale = false
    delete engineRuntime.runtimePlan.models
    delete engineRuntime.runtimePlan.launch_manifests
    toastAdd.mockReset()
    fetchModels.mockReset()
    startModel.mockReset()
    startModel.mockResolvedValue({ status: 'running' })
    fetchRuntimePlanState.mockReset()
    fetchGpuList.mockReset()
    fetchEngineDescriptors.mockReset()
    invalidateRuntimePlan.mockReset()
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
      if (url === '/api/models/model-1/saved-command') {
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
      if (url === '/api/models/model-1/preview-command') {
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

  it('removes an explicit override from the saved engine section when restoring its default', async () => {
    const wrapper = mountView()
    await flushPromises()
    await vi.runAllTimersAsync()
    await flushPromises()
    const basic = wrapper
      .findAll('.param-field')
      .find((field) => field.text().includes('Context Size'))
    await basic.get('input').setValue('8192')
    await basic.get('.param-reset').trigger('click')
    await wrapper.get('button[data-label="Save changes"]').trigger('click')
    await flushPromises()
    const payload = vi.mocked(axios.put).mock.calls.at(-1)[1]
    expect(payload.engines.llama_cpp).not.toHaveProperty('ctx_size')
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
      '/api/models/model-1/preview-command',
      {
        engine: 'llama_cpp',
        engines: {
          llama_cpp: {
            temperature: 0.7,
            runtime_env: {},
          },
        },
      },
      expect.objectContaining({ signal: expect.any(AbortSignal) }),
    )

    await wrapper.get('button[data-label="Save changes"]').trigger('click')
    await flushPromises()

    expect(axios.put).toHaveBeenCalledWith('/api/models/model-1/config', {
      engine: 'llama_cpp',
      engines: {
        llama_cpp: {
          temperature: 0.7,
          runtime_env: {},
        },
      },
    })
    expect(invalidateRuntimePlan).toHaveBeenCalled()
  })

  it('keeps the form and waits for a deliberate retry when the save queue is full', async () => {
    vi.mocked(axios.put).mockRejectedValue(
      Object.assign(new Error('busy'), {
        response: { status: 503, data: { code: 'STORE_QUEUE_FULL', committed: false } },
      }),
    )
    const wrapper = mountView()
    await flushPromises()
    await vi.runAllTimersAsync()
    await flushPromises()

    await wrapper.get('button[data-label="Save changes"]').trigger('click')
    await flushPromises()

    expect(wrapper.text()).toContain('Your edits are still here')
    expect(toastAdd).not.toHaveBeenCalledWith(expect.objectContaining({ summary: 'Saved' }))
    expect(axios.put).toHaveBeenCalledTimes(1)

    vi.mocked(axios.put).mockResolvedValue({
      data: { engine: 'llama_cpp', engines: { llama_cpp: { temperature: 0.7 } } },
    })
    const retry = wrapper.findAll('button').find((button) => button.text() === 'Try again')
    await retry.trigger('click')
    await flushPromises()
    expect(axios.put).toHaveBeenCalledTimes(2)
  })

  it('refreshes a replaced configuration and does not save it again automatically', async () => {
    vi.mocked(axios.put).mockRejectedValue(
      Object.assign(new Error('ack'), {
        response: { status: 500, data: { code: 'STORE_WRITE_FAILED', committed: true } },
      }),
    )
    const wrapper = mountView()
    await flushPromises()
    await vi.runAllTimersAsync()
    await flushPromises()

    await wrapper.get('button[data-label="Save changes"]').trigger('click')
    await flushPromises()

    expect(axios.put).toHaveBeenCalledTimes(1)
    expect(wrapper.text()).toContain('acknowledgement failed')
    expect(wrapper.findAll('button').some((button) => button.text() === 'Try again')).toBe(false)
    expect(toastAdd).not.toHaveBeenCalledWith(expect.objectContaining({ summary: 'Saved' }))
  })

  it('keeps the form when reloading a replaced configuration fails', async () => {
    const wrapper = mountView()
    await flushPromises()
    await vi.runAllTimersAsync()
    await flushPromises()

    vi.mocked(axios.get).mockRejectedValue(new Error('offline'))
    vi.mocked(axios.put).mockRejectedValue(
      Object.assign(new Error('ack'), {
        response: { status: 500, data: { code: 'STORE_WRITE_FAILED', committed: true } },
      }),
    )
    await wrapper.get('button[data-label="Save changes"]').trigger('click')
    await flushPromises()

    expect(axios.put).toHaveBeenCalledTimes(1)
    expect(wrapper.text()).toContain('could not be reloaded')
    expect(wrapper.text()).toContain('Your edits are still here')
  })

  it('keeps saved environment and inherit GPU mode when saving without edits', async () => {
    vi.mocked(axios.get).mockImplementation((url) => {
      if (url === '/api/models/model-1/config') {
        return Promise.resolve({
          data: {
            engine: 'llama_cpp',
            engines: {
              llama_cpp: {
                temperature: 0.7,
                gpu_mode: 'inherit',
                gpu_devices: [],
                runtime_env: {
                  FOO: 'bar',
                  EMPTY: '',
                  CUDA_VISIBLE_DEVICES: 'GPU-keep',
                },
                runtime_env_unset: ['REMOVED_VAR'],
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
                    flags: ['--temperature'],
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
      if (url === '/api/models/model-1/saved-command') {
        return Promise.resolve({ data: { ok: true, cmd: 'saved-cmd' } })
      }
      throw new Error(`Unexpected GET ${url}`)
    })
    fetchGpuList.mockResolvedValue({
      vendor: 'nvidia',
      device_count: 2,
      cpu_only_mode: false,
      gpus: [
        { index: 0, uuid: 'GPU-0', name: 'A' },
        { index: 1, uuid: 'GPU-1', name: 'B' },
      ],
    })

    const wrapper = mountView()
    await flushPromises()
    await vi.runAllTimersAsync()
    await flushPromises()

    const envKeys = wrapper
      .findAll('input[placeholder="VAR_NAME"]')
      .map((input) => input.element.value)
    const envValues = wrapper
      .findAll('input[placeholder="value"]')
      .map((input) => input.element.value)
    expect(envKeys).toEqual(expect.arrayContaining(['FOO', 'EMPTY', 'REMOVED_VAR']))
    expect(envValues).toContain('bar')
    expect(wrapper.text()).not.toContain('2 GPUs selected')

    await wrapper.get('input[placeholder="0.8"]').setValue('0.9')
    await wrapper.get('button[data-label="Save changes"]').trigger('click')
    await flushPromises()

    const payload = vi.mocked(axios.put).mock.calls.at(-1)[1]
    const section = payload.engines.llama_cpp
    expect(section.runtime_env).toEqual({
      FOO: 'bar',
      EMPTY: '',
      CUDA_VISIBLE_DEVICES: 'GPU-keep',
    })
    expect(section.runtime_env_unset).toEqual(['REMOVED_VAR'])
    expect(section.gpu_mode).toBe('inherit')
    expect(section.gpu_devices ?? []).toEqual([])
    expect(JSON.stringify(section)).not.toContain('GPU-0')
    expect(JSON.stringify(section)).not.toContain('GPU-1')
  })

  it('switching to inherit keeps other environment variables and does not pin every GPU', async () => {
    vi.mocked(axios.get).mockImplementation((url) => {
      if (url === '/api/models/model-1/config') {
        return Promise.resolve({
          data: {
            engine: 'llama_cpp',
            engines: {
              llama_cpp: {
                temperature: 0.7,
                gpu_mode: 'selected',
                gpu_devices: ['GPU-0'],
                runtime_env: {
                  FOO: 'bar',
                  CUDA_VISIBLE_DEVICES: 'GPU-0',
                },
                runtime_env_unset: ['REMOVED_VAR'],
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
                    flags: ['--temperature'],
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
      if (url === '/api/models/model-1/saved-command') {
        return Promise.resolve({ data: { ok: true, cmd: 'saved-cmd' } })
      }
      throw new Error(`Unexpected GET ${url}`)
    })
    fetchGpuList.mockResolvedValue({
      vendor: 'nvidia',
      device_count: 2,
      cpu_only_mode: false,
      gpus: [
        { index: 0, uuid: 'GPU-0', name: 'A' },
        { index: 1, uuid: 'GPU-1', name: 'B' },
      ],
    })

    const wrapper = mountView()
    await flushPromises()
    await vi.runAllTimersAsync()
    await flushPromises()

    const gpuMode = wrapper
      .findAll('#gpu-mode button')
      .find((button) => button.text() === 'All GPUs')
    await gpuMode.trigger('click')
    await flushPromises()

    await wrapper.get('button[data-label="Save changes"]').trigger('click')
    await flushPromises()

    const payload = vi.mocked(axios.put).mock.calls.at(-1)[1]
    const section = payload.engines.llama_cpp
    expect(section.gpu_mode).toBe('inherit')
    expect(section.gpu_devices ?? []).toEqual([])
    expect(section.runtime_env).toEqual({ FOO: 'bar' })
    expect(section.runtime_env_unset).toEqual(['REMOVED_VAR'])
    expect(section.runtime_env.CUDA_VISIBLE_DEVICES).toBeUndefined()
    expect(JSON.stringify(section)).not.toContain('GPU-1')
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

  it('imports parsed environment variables into Studio proxy env and skips Studio-owned names', async () => {
    const wrapper = mountView()
    await flushPromises()
    await vi.runAllTimersAsync()
    await flushPromises()

    await wrapper.get('button[data-label="Import command"]').trigger('click')
    await flushPromises()
    await wrapper
      .get('#parse-command-text')
      .setValue('FLASHINFER_DISABLE_VERSION_CHECK=1 PORT=9 LLAMA_STUDIO_MODEL_PATH=/x sglang serve')
    await flushPromises()

    expect(wrapper.find('#import-env-FLASHINFER_DISABLE_VERSION_CHECK').exists()).toBe(true)
    expect(wrapper.find('#import-env-PORT').exists()).toBe(false)
    await wrapper.get('button[data-label="Apply 1 env var"]').trigger('click')
    await flushPromises()

    expect(wrapper.text()).toContain('Unsaved changes')
    const envKeys = wrapper
      .findAll('input[placeholder="VAR_NAME"]')
      .map((input) => input.element.value)
    expect(envKeys).toContain('FLASHINFER_DISABLE_VERSION_CHECK')
    expect(envKeys).not.toContain('PORT')
    expect(envKeys).not.toContain('LLAMA_STUDIO_MODEL_PATH')
    const envValues = wrapper
      .findAll('input[placeholder="value"]')
      .map((input) => input.element.value)
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
    expect(wrapper.findAll('.engine-name').map((node) => node.text())).not.toContain('llama.cpp')
    expect(wrapper.text()).not.toContain('Show other engines')
  })

  it('hides engines that are not installed', async () => {
    engineDescriptors.push(
      {
        id: 'llama_cpp',
        label: 'llama.cpp',
        enabled: true,
        installed_versions: 0,
        artifact_formats: ['gguf'],
      },
      {
        id: 'ik_llama',
        label: 'ik_llama.cpp',
        enabled: true,
        installed_versions: 1,
        artifact_formats: ['gguf'],
      },
    )
    const wrapper = mountView()
    await flushPromises()
    await vi.runAllTimersAsync()
    await flushPromises()

    expect(wrapper.text()).not.toContain('Show other engines')
    expect(wrapper.findAll('.engine-name').map((node) => node.text())).toEqual(['ik_llama.cpp'])
    expect(wrapper.text()).toContain('llama.cpp is saved for this model')
  })

  it('does not mark this model pending because another model changed', async () => {
    engineRuntime.runtimePlanState.applicable = true
    engineRuntime.runtimePlanState.stale = true
    engineRuntime.runtimePlan.launch_manifests = true
    engineRuntime.runtimePlan.models = [
      { catalog_id: 'other-model', model_id: 'other', action: 'restart_now', running: true },
      { catalog_id: 'model-1', model_id: 'model-1', action: 'none', running: false },
    ]

    const wrapper = mountView()
    await flushPromises()
    await vi.runAllTimersAsync()
    await flushPromises()

    expect(wrapper.text()).toContain(
      'The model is stopped, so they are used the next time it starts.',
    )
    expect(wrapper.text()).not.toContain('Pending changes for this model')
  })

  it('describes a stopped model with unpublished settings as pending for this model only', async () => {
    engineRuntime.runtimePlan.launch_manifests = true
    engineRuntime.runtimePlan.models = [
      { catalog_id: 'model-1', model_id: 'model-1', action: 'publish_next_start', running: false },
    ]

    const wrapper = mountView()
    await flushPromises()
    await vi.runAllTimersAsync()
    await flushPromises()

    expect(wrapper.text()).toContain('Saved changes for this model are not published yet')
    expect(wrapper.text()).toContain('Saved · needs apply')
  })

  it('renders the overview identity, engine selector, at-a-glance facts and review', async () => {
    const wrapper = mountView()
    await flushPromises()
    await vi.runAllTimersAsync()
    await flushPromises()

    const overview = wrapper.get('.config-overview')
    expect(wrapper.find('.config-workbench').exists()).toBe(false)
    expect(overview.text()).toContain('Test Model')
    expect(overview.text()).toContain('org/model')
    expect(overview.text()).toContain('GGUF')
    expect(overview.text()).toContain('Q4_K_M')
    expect(overview.text()).toContain('llama.cpp')

    const engineGroup = overview.get('[role="radiogroup"][aria-label="Engine"]')
    const engineRadios = engineGroup.findAll('[role="radio"]')
    expect(engineRadios.length).toBeGreaterThan(0)
    expect(engineRadios.some((radio) => radio.attributes('aria-checked') === 'true')).toBe(true)

    expect(overview.text()).toContain('At a glance')
    expect(overview.text()).toContain('Context window')
    expect(overview.text()).toContain('GPU & offload')
    expect(overview.text()).toContain('Endpoint')
    expect(overview.text()).toContain('Apply state')
    expect(overview.text()).toContain('Review')
    expect(overview.text()).toContain('Ready to start')
  })

  it('switches engines from the overview identity card', async () => {
    engineDescriptors.push(
      {
        id: PRIMARY_ENGINE_ID,
        label: ENGINE_DESCRIPTORS[PRIMARY_ENGINE_ID].label,
        enabled: true,
        installed_versions: 1,
        artifact_formats: ['gguf'],
      },
      {
        id: SECONDARY_ENGINE_ID,
        label: ENGINE_DESCRIPTORS[SECONDARY_ENGINE_ID].label,
        enabled: true,
        installed_versions: 1,
        artifact_formats: ['gguf'],
      },
    )
    const wrapper = mountView()
    await flushPromises()
    await vi.runAllTimersAsync()
    await flushPromises()

    const option = wrapper
      .get('.config-overview [role="radiogroup"][aria-label="Engine"]')
      .findAll('[role="radio"]')
      .find((radio) => radio.text().includes(ENGINE_DESCRIPTORS[SECONDARY_ENGINE_ID].label))
    await option.trigger('click')
    await flushPromises()

    expect(wrapper.vm.config.engine).toBe(SECONDARY_ENGINE_ID)
    expect(wrapper.vm.savedConfig.engine).toBe(PRIMARY_ENGINE_ID)
    expect(axios.get).toHaveBeenCalledWith(
      '/api/models/param-registry',
      expect.objectContaining({
        params: expect.objectContaining({ engine: SECONDARY_ENGINE_ID }),
      }),
    )
  })

  it('shows launch-plan reasons in the overview with a link to the fix section', async () => {
    engineRuntime.runtimePlan.launch_manifests = true
    engineRuntime.runtimePlan.models = [
      {
        catalog_id: 'model-1',
        model_id: 'model-1',
        action: 'invalid',
        running: false,
        reasons: ['Context size exceeds the model limit.'],
      },
    ]

    const wrapper = mountView()
    await flushPromises()
    await vi.runAllTimersAsync()
    await flushPromises()

    const overview = wrapper.get('.config-overview')
    expect(overview.text()).toContain('Configuration cannot be compiled')
    expect(overview.text()).toContain('Context size exceeds the model limit.')
    const openRuntime = overview
      .findAll('button')
      .find((button) => button.text() === 'Open Runtime')
    expect(openRuntime).toBeDefined()

    await openRuntime.trigger('click')
    await flushPromises()
    expect(wrapper.vm.pageTab).toBe('runtime')
  })

  it('starts the model from the action bar only when the configuration is clean', async () => {
    const wrapper = mountView()
    await flushPromises()
    await vi.runAllTimersAsync()
    await flushPromises()

    const startButton = () => wrapper.get('.config-action-bar button[data-label="Start model"]')
    expect(startButton().attributes('disabled')).toBeUndefined()

    await startButton().trigger('click')
    await flushPromises()

    expect(startModel).toHaveBeenCalledWith('model-1')
    expect(toastAdd).toHaveBeenCalledWith(expect.objectContaining({ summary: 'Model starting' }))
  })

  it('disables Start while the form is dirty and never calls the store', async () => {
    const wrapper = mountView()
    await flushPromises()
    await vi.runAllTimersAsync()
    await flushPromises()

    await wrapper.get('input[placeholder="0.8"]').setValue('0.9')
    await flushPromises()

    const startButton = wrapper.get('.config-action-bar button[data-label="Start model"]')
    expect(startButton.attributes('disabled')).toBeDefined()
    expect(wrapper.text()).toContain('Save or discard these edits before starting the model.')

    await startButton.trigger('click')
    await flushPromises()

    expect(startModel).not.toHaveBeenCalled()
    expect(
      wrapper.get('.config-action-bar button[data-label="Save changes"]').attributes('disabled'),
    ).toBeUndefined()
  })

  it('opens the apply flow from the action bar when saved settings need applying', async () => {
    engineRuntime.runtimePlan.launch_manifests = true
    engineRuntime.runtimePlan.models = [
      { catalog_id: 'model-1', model_id: 'model-1', action: 'publish_next_start', running: false },
    ]

    const wrapper = mountView()
    await flushPromises()
    await vi.runAllTimersAsync()
    await flushPromises()

    const apply = wrapper.get('.config-action-bar button[data-label="Use on next start"]')
    await apply.trigger('click')
    await flushPromises()

    expect(wrapper.vm.applyImpactVisible).toBe(true)
  })

  it('restores only the auto-fit overrides in the draft without saving', async () => {
    const engineId = PRIMARY_ENGINE_ID
    const autoFitRegistry = {
      sections: [
        {
          id: 'performance',
          label: 'Performance',
          params: [
            {
              key: 'fit',
              label: 'Auto-fit',
              type: 'select',
              value_kind: 'enum',
              default: 'on',
              primary_flag: '--fit',
              flags: ['--fit'],
              options: [
                { value: 'on', label: 'on' },
                { value: 'off', label: 'off' },
              ],
              supported: true,
            },
            {
              key: 'fit_ctx',
              label: 'Fit context',
              type: 'int',
              value_kind: 'scalar',
              default: 4096,
              primary_flag: '--fit-ctx',
              flags: ['--fit-ctx'],
              supported: true,
            },
            {
              key: 'fit_target',
              label: 'Fit target',
              type: 'int',
              value_kind: 'scalar',
              default: 1024,
              primary_flag: '--fit-target',
              flags: ['--fit-target'],
              supported: true,
            },
            {
              key: 'ctx_size',
              label: 'Context Size',
              type: 'int',
              value_kind: 'scalar',
              default: 0,
              primary_flag: '--ctx-size',
              flags: ['--ctx-size'],
              supported: true,
            },
            {
              key: 'n_gpu_layers',
              label: 'GPU Layers',
              type: 'int',
              value_kind: 'scalar',
              default: 'auto',
              primary_flag: '--n-gpu-layers',
              flags: ['--gpu-layers', '--n-gpu-layers'],
              supported: true,
            },
            {
              key: 'flash_attn',
              label: 'Flash Attention',
              type: 'select',
              value_kind: 'enum',
              default: 'auto',
              options: [
                { value: 'on', label: 'on' },
                { value: 'off', label: 'off' },
                { value: 'auto', label: 'auto' },
              ],
              supported: true,
            },
            {
              key: 'threads',
              label: 'Threads',
              type: 'int',
              value_kind: 'scalar',
              default: -1,
              supported: true,
            },
            {
              key: 'batch_size',
              label: 'Batch Size',
              type: 'int',
              value_kind: 'scalar',
              default: 2048,
              supported: true,
            },
            {
              key: 'parallel',
              label: 'Parallel',
              type: 'int',
              value_kind: 'scalar',
              default: -1,
              supported: true,
            },
          ],
        },
      ],
    }
    const baseGet = vi.mocked(axios.get).getMockImplementation()
    vi.mocked(axios.get).mockImplementation((url, config) => {
      if (url === '/api/models/model-1/config') {
        return Promise.resolve({
          data: {
            engine: engineId,
            engines: {
              [engineId]: {
                fit: 'off',
                ctx_size: 8192,
                n_gpu_layers: 32,
                threads: 8,
                batch_size: 512,
                parallel: 2,
              },
            },
          },
        })
      }
      if (url === '/api/models/param-registry') {
        return Promise.resolve({ data: autoFitRegistry })
      }
      return baseGet(url, config)
    })

    const wrapper = mountView()
    await flushPromises()
    await vi.runAllTimersAsync()
    await flushPromises()

    expect(wrapper.vm.autoFitState.supported).toBe(true)
    expect(wrapper.vm.autoFitState.overrides.map((row) => row.key)).toEqual([
      'fit',
      'ctx_size',
      'n_gpu_layers',
    ])
    const card = wrapper.get('[data-auto-fit]')
    expect(card.text()).toContain('Auto-fit is overridden')
    expect(card.findAll('[data-auto-fit-key]')).toHaveLength(3)
    expect(card.text()).not.toMatch(/VRAM|GiB|recommended for this machine/i)
    expect(wrapper.find('[data-auto-fit-action]').exists()).toBe(true)

    await card.get('[data-auto-fit-review]').trigger('click')
    await flushPromises()
    expect(wrapper.vm.pageTab).toBe('runtime')

    await card.get('[data-auto-fit-restore]').trigger('click')
    await flushPromises()

    expect(wrapper.vm.config.fit).toBe('on')
    expect(wrapper.vm.config.ctx_size).toBe(0)
    expect(wrapper.vm.config).not.toHaveProperty('n_gpu_layers')
    // Auto-fit restoration must not touch unrelated parameters.
    expect(wrapper.vm.config.threads).toBe(8)
    expect(wrapper.vm.config.batch_size).toBe(512)
    expect(wrapper.vm.config.parallel).toBe(2)

    expect(axios.put).not.toHaveBeenCalled()
    expect(toastAdd).toHaveBeenCalledWith(
      expect.objectContaining({ summary: 'Auto settings restored' }),
    )
    expect(wrapper.vm.autoFitState.overrides).toEqual([])
    expect(wrapper.get('[data-auto-fit]').text()).toContain('Auto-fit is active')
    expect(wrapper.find('[data-auto-fit-action]').exists()).toBe(false)

    const paramRows = wrapper.vm.configChangeGroups
      .find((group) => group.id === 'params')
      .rows.map((row) => row.key)
    expect(paramRows.sort()).toEqual(['ctx_size', 'fit', 'n_gpu_layers'])
  })

  it('explains a missing engine parameter catalog and keeps saved values visible', async () => {
    const baseGet = vi.mocked(axios.get).getMockImplementation()
    vi.mocked(axios.get).mockImplementation((url, config) => {
      if (url === '/api/models/param-registry') {
        return Promise.resolve({
          data: { sections: [], scan_error: 'engine help timed out', scan_pending: false },
        })
      }
      return baseGet(url, config)
    })

    const wrapper = mountView()
    await flushPromises()
    await vi.runAllTimersAsync()
    await flushPromises()
    wrapper.vm.pageTab = 'runtime'
    await flushPromises()

    expect(wrapper.text()).toContain('No engine parameters yet')
    expect(wrapper.text()).toContain('engine help timed out')
    expect(wrapper.text()).toContain('Saved settings')
    expect(wrapper.text()).toContain('Temperature')
    expect(wrapper.find('[data-runtime-rescan]').exists()).toBe(true)

    await wrapper.get('[data-runtime-rescan]').trigger('click')
    await flushPromises()
    expect(axios.get).toHaveBeenCalledWith(
      '/api/models/param-registry',
      expect.objectContaining({
        params: expect.objectContaining({ rescan: true }),
      }),
    )
  })

  it('hides auto-fit when the engine scan has no fit parameter', async () => {
    const wrapper = mountView()
    await flushPromises()
    await vi.runAllTimersAsync()
    await flushPromises()

    expect(wrapper.vm.autoFitState.supported).toBe(false)
    expect(wrapper.find('[data-auto-fit]').exists()).toBe(false)
    expect(wrapper.find('[data-auto-fit-action]').exists()).toBe(false)
  })
})
