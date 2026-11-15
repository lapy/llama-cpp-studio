import { enableAutoUnmount, flushPromises, mount } from '@vue/test-utils'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import ModelConfigLaunchSettings from './ModelConfigLaunchSettings.vue'

enableAutoUnmount(afterEach)

const { toastAdd, fetchModelCompanions, fetchModels, updateModelProjector } = vi.hoisted(() => ({
  toastAdd: vi.fn(),
  fetchModelCompanions: vi.fn(),
  fetchModels: vi.fn(),
  updateModelProjector: vi.fn(),
}))

vi.mock('primevue/usetoast', () => ({ useToast: () => ({ add: toastAdd }) }))
vi.mock('@/stores/models', () => ({
  useModelStore: () => ({ fetchModelCompanions, fetchModels, updateModelProjector }),
}))

const ButtonStub = {
  props: ['label', 'icon', 'size', 'severity', 'outlined', 'text', 'rounded', 'loading', 'type'],
  emits: ['click'],
  template: '<button :data-label="label" @click="$emit(`click`)">{{ label }}</button>',
}

const InputStub = {
  props: ['modelValue', 'placeholder', 'readonly'],
  emits: ['update:modelValue'],
  template:
    '<input :value="modelValue ?? ``" :placeholder="placeholder" :readonly="readonly" @input="$emit(`update:modelValue`, $event.target.value)" />',
}

const SelectStub = {
  props: ['modelValue', 'options', 'optionLabel', 'optionValue'],
  emits: ['update:modelValue'],
  template:
    '<select :value="modelValue" @change="$emit(`update:modelValue`, $event.target.value)"><option v-for="option in options || []" :key="option.value ?? option" :value="option.value ?? option">{{ option.label ?? option }}</option></select>',
}

const TextareaStub = {
  props: ['modelValue', 'rows'],
  emits: ['update:modelValue'],
  template:
    '<textarea :value="modelValue" @input="$emit(`update:modelValue`, $event.target.value)" />',
}

const RequestAliasesEditorStub = {
  props: ['modelValue', 'baseName'],
  emits: ['update:modelValue', 'validity'],
  template: '<div class="request-aliases-stub" />',
}

const model = {
  id: 'model-1',
  format: 'gguf',
  huggingface_id: 'org/model',
  mmproj_filename: '',
  mtp_filename: '',
  dflash_filename: '',
  dspark_filename: '',
}

function mountSettings(props = {}) {
  return mount(ModelConfigLaunchSettings, {
    props: {
      model,
      engine: 'llama_cpp',
      stableModelId: 'studio-model',
      modelAlias: 'my-alias',
      setParamsById: [],
      customArgs: '',
      canImportCommand: true,
      envRows: [{ row: { key: 'FOO', value: 'bar', mode: 'set' }, originalIndex: 0 }],
      cmdPreview: { dialogVisible: false, mode: 'unsaved', loading: false, unsavedLoading: false },
      ...props,
    },
    global: {
      directives: { tooltip: () => {} },
      stubs: {
        Button: ButtonStub,
        InputText: InputStub,
        Select: SelectStub,
        Textarea: TextareaStub,
        RequestAliasesEditor: RequestAliasesEditorStub,
      },
    },
  })
}

describe('ModelConfigLaunchSettings', () => {
  beforeEach(() => {
    toastAdd.mockReset()
    fetchModelCompanions.mockReset()
    fetchModels.mockReset()
    updateModelProjector.mockReset()
    fetchModelCompanions.mockResolvedValue({
      mmproj_files: [{ filename: 'mmproj-F16.gguf', size: 7 }],
      mtp_files: [],
      dflash_files: [],
      dspark_files: [],
      engram_files: [],
    })
    fetchModels.mockResolvedValue(undefined)
    updateModelProjector.mockResolvedValue({ applied: true, message: 'Applied' })
  })

  it('renders the launch form and forwards edits through emits', async () => {
    const wrapper = mountSettings()
    await flushPromises()

    expect(wrapper.get('input[aria-label="Stable Studio proxy model ID"]').element.value).toBe(
      'studio-model',
    )
    expect(wrapper.get('#config-mmproj').text()).toContain('F16')

    await wrapper.get('input[placeholder="e.g. my-app-model"]').setValue('renamed-model')
    expect(wrapper.emitted('update:modelAlias')?.at(-1)).toEqual(['renamed-model'])

    await wrapper.get('input[placeholder="VAR_NAME"]').setValue('BAR')
    expect(wrapper.emitted('update-env-row')?.at(-1)).toEqual([
      { originalIndex: 0, field: 'key', value: 'BAR' },
    ])

    const addVariable = wrapper.findAll('button').find((button) => button.text() === 'Add variable')
    await addVariable.trigger('click')
    expect(wrapper.emitted('add-env-row')).toHaveLength(1)
  })

  it('applies a companion selection through the model store', async () => {
    const wrapper = mountSettings()
    await flushPromises()

    await wrapper.get('#config-mmproj').setValue('mmproj-F16.gguf')
    const applyProjector = wrapper
      .findAll('button')
      .find((button) => button.attributes('data-label') === 'Apply')
    await applyProjector.trigger('click')
    await flushPromises()

    expect(updateModelProjector).toHaveBeenCalledWith('model-1', 'mmproj-F16.gguf', 7)
    expect(fetchModels).toHaveBeenCalled()
    expect(wrapper.emitted('refresh-model')).toHaveLength(1)
  })
})
