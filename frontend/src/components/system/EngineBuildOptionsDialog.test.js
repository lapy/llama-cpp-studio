import { describe, it, expect } from 'vitest'
import { mount } from '@vue/test-utils'
import EngineBuildOptionsDialog from './EngineBuildOptionsDialog.vue'

const fields = [
  {
    key: 'repository_url',
    label: 'Repo URL',
    placeholder: 'https://github.com/0xShug0/audio.cpp.git',
    hint: 'Official repo or any fork with the same layout.',
  },
  {
    key: 'source_ref',
    label: 'Ref (tag / branch / commit)',
    placeholder: 'main',
    hint: 'Used when you Build now.',
  },
  {
    key: 'versionSuffix',
    label: 'Build Name Suffix',
    optional: true,
    placeholder: 'e.g. my-build',
    hint: 'Appended to version name.',
  },
]

const categories = [
  {
    id: 'backends',
    label: 'GPU & compute backends',
    layout: 'grid',
    note: 'Studio exposes CUDA.',
    options: [
      {
        key: 'cuda',
        type: 'bool',
        label: 'CUDA',
        desc: 'ENGINE_ENABLE_CUDA',
        primary: true,
        disabled: true,
        disabledReason: 'No CUDA runtime found.',
      },
      {
        key: 'vulkan',
        type: 'bool',
        label: 'Vulkan',
        desc: 'ENGINE_ENABLE_VULKAN',
        primary: false,
      },
    ],
  },
  {
    id: 'cuda',
    label: 'CUDA options',
    layout: 'advanced',
    note: 'Only used when CUDA is on.',
    options: [
      { key: 'cuda_graphs', type: 'bool', label: 'CUDA graphs', desc: 'ENGINE_CUDA_GRAPHS' },
      { key: 'jobs', type: 'int', label: 'Parallel jobs', desc: '0 = automatic' },
      {
        key: 'model_set',
        type: 'enum',
        label: 'Model set',
        desc: 'AUDIOCPP_MODEL_SET',
        enum_values: ['full', 'core'],
      },
      { key: 'models', type: 'str', label: 'Custom models', desc: 'Comma separated' },
    ],
  },
]

function makeForm(overrides = {}) {
  return {
    repository_url: 'https://github.com/0xShug0/audio.cpp.git',
    source_ref: 'main',
    versionSuffix: '',
    build_config: {
      build_type: 'RelWithDebInfo',
      cuda: true,
      vulkan: false,
      cuda_graphs: true,
      jobs: 0,
      model_set: 'full',
      models: '',
      custom_cmake_args: '',
      cflags: '',
      cxxflags: '',
    },
    ...overrides,
  }
}

function mountDialog(props = {}) {
  return mount(EngineBuildOptionsDialog, {
    props: {
      visible: true,
      header: 'Build settings — audio.cpp',
      form: makeForm(),
      fields,
      categories,
      buildTypes: [
        { label: 'RelWithDebInfo', value: 'RelWithDebInfo' },
        { label: 'Release', value: 'Release' },
      ],
      ...props,
    },
    global: {
      stubs: {
        Dialog: {
          props: ['visible', 'header'],
          template: `
            <div class="dialog-stub" :data-header="header">
              <div class="dialog-header">{{ header }}</div>
              <slot />
              <div class="dialog-footer"><slot name="footer" /></div>
            </div>
          `,
        },
        Button: {
          props: ['label', 'icon', 'severity', 'loading', 'disabled', 'outlined'],
          emits: ['click'],
          template:
            '<button type="button" :data-label="label" :disabled="disabled" @click="$emit(\'click\')">{{ label }}</button>',
        },
        InputText: {
          props: ['modelValue', 'placeholder'],
          emits: ['update:modelValue'],
          template:
            '<input class="input-stub" :value="modelValue" :placeholder="placeholder" @input="$emit(\'update:modelValue\', $event.target.value)" />',
        },
        InputNumber: {
          props: ['modelValue', 'min', 'max'],
          emits: ['update:modelValue'],
          template:
            '<input class="number-stub" type="number" :value="modelValue" :min="min" :max="max" @input="$emit(\'update:modelValue\', Number($event.target.value))" />',
        },
        Select: {
          props: ['modelValue', 'options'],
          emits: ['update:modelValue'],
          template: `
            <select class="select-stub" :value="modelValue" @change="$emit('update:modelValue', $event.target.value)">
              <option v-for="option in options" :key="option.value ?? option" :value="option.value ?? option">
                {{ option.label ?? option }}
              </option>
            </select>
          `,
        },
        ToggleSwitch: {
          props: ['modelValue', 'disabled'],
          emits: ['update:modelValue'],
          template:
            '<input class="toggle-stub" type="checkbox" :checked="modelValue" :disabled="disabled" @change="$emit(\'update:modelValue\', $event.target.checked)" />',
        },
      },
    },
  })
}

describe('EngineBuildOptionsDialog schema rendering', () => {
  it('renders form fields, build type, categories and custom flags from props', () => {
    const wrapper = mountDialog()
    const text = wrapper.text()

    expect(wrapper.get('[data-header]').attributes('data-header')).toBe(
      'Build settings — audio.cpp',
    )
    expect(text).toContain('Repo URL')
    expect(text).toContain('Ref (tag / branch / commit)')
    expect(text).toContain('Build Name Suffix')
    expect(text).toContain('Build type')
    expect(text).toContain('GPU & compute backends')
    expect(text).toContain('Studio exposes CUDA.')
    expect(text).toContain('CUDA')
    expect(text).toContain('No CUDA runtime found.')
    expect(text).toContain('CUDA options')
    expect(text).toContain('CUDA graphs')
    expect(text).toContain('Parallel jobs')
    expect(text).toContain('Model set')
    expect(text).toContain('Custom models')
    expect(text).toContain('Custom flags')
    expect(text).toContain('Custom CMake args')
    expect(text).toContain('CFLAGS / CXXFLAGS')
  })

  it('splits primary backends from the More backends details and honours disabled state', () => {
    const wrapper = mountDialog()
    const grid = wrapper
      .findAll('.form-field')
      .find((node) => node.text().includes('GPU & compute backends'))
    const primaryToggles = grid.findAll('.toggle-grid')[0].findAll('.toggle-stub')
    expect(primaryToggles).toHaveLength(1)
    expect(primaryToggles[0].attributes('disabled')).toBeDefined()

    const moreBackends = wrapper.findAll('.build-options-details')[0]
    expect(moreBackends.text()).toContain('More backends')
    expect(moreBackends.text()).toContain('Vulkan')
  })

  it('renders int and enum option controls with their schema bounds', () => {
    const wrapper = mountDialog()
    const number = wrapper.get('.number-stub')
    expect(number.attributes('min')).toBe('0')
    expect(number.attributes('max')).toBe('256')
    expect(wrapper.findAll('.select-stub')[1].text()).toContain('core')
  })

  it('emits field and option updates instead of owning form state', async () => {
    const wrapper = mountDialog()
    const inputs = wrapper.findAll('.input-stub')
    await inputs[0].setValue('https://github.com/me/audio.cpp.git')
    expect(wrapper.emitted('field')?.[0]).toEqual([
      'repository_url',
      'https://github.com/me/audio.cpp.git',
    ])

    const toggles = wrapper.findAll('.toggle-stub')
    await toggles[1].setValue(true)
    expect(wrapper.emitted('option')?.[0]).toEqual(['vulkan', true])

    const selects = wrapper.findAll('.select-stub')
    await selects[1].setValue('core')
    expect(wrapper.emitted('option')?.[1]).toEqual(['model_set', 'core'])

    await selects[0].setValue('Release')
    expect(wrapper.emitted('option')?.[2]).toEqual(['build_type', 'Release'])
  })

  it('renders notes from the caller with their tone and role', () => {
    const wrapper = mountDialog({
      notes: [
        { text: 'Could not check CUDA.', role: 'status' },
        { text: 'CUDA: available.', tone: 'info', role: 'status' },
      ],
    })
    const notes = wrapper.findAll('.build-note')
    expect(notes[0].text()).toBe('Could not check CUDA.')
    expect(notes[0].classes()).not.toContain('build-note--info')
    expect(notes[0].attributes('role')).toBe('status')
    expect(notes[1].text()).toBe('CUDA: available.')
    expect(notes[1].classes()).toContain('build-note--info')
  })

  it('shows the loading note while the catalog is being fetched', () => {
    const wrapper = mountDialog({ optionsLoading: true })
    expect(wrapper.text()).toContain('Loading build options…')
  })
})

describe('EngineBuildOptionsDialog footer modes', () => {
  it('offers Save settings and Build now for global build settings', async () => {
    const wrapper = mountDialog()
    const labels = wrapper.findAll('button').map((button) => button.text())
    expect(labels).toEqual(['Cancel', 'Save settings', 'Build now'])

    const build = wrapper.findAll('button').find((button) => button.text() === 'Build now')
    await build.trigger('click')
    expect(wrapper.emitted('build')).toHaveLength(1)

    const save = wrapper.findAll('button').find((button) => button.text() === 'Save settings')
    await save.trigger('click')
    expect(wrapper.emitted('save')).toHaveLength(1)
  })

  it('freezes global fields and offers Save plus Save & retry for a failed version', async () => {
    const wrapper = mountDialog({
      frozen: true,
      frozenVersion: 'source-fail',
      actions: {
        save: { loading: true, disabled: true },
        rebuild: { label: 'Save & retry', loading: true },
      },
    })
    const text = wrapper.text()
    expect(text).toContain('These CMake options are frozen to')
    expect(text).toContain('source-fail')
    expect(text).not.toContain('Repo URL')
    expect(text).not.toContain('Build Name Suffix')
    expect(text).toContain('Build type')

    const labels = wrapper.findAll('button').map((button) => button.text())
    expect(labels).toEqual(['Cancel', 'Save', 'Save & retry'])
    expect(labels).not.toContain('Build now')

    const save = wrapper.findAll('button').find((button) => button.text() === 'Save')
    expect(save.attributes('disabled')).toBeDefined()

    const rebuild = wrapper.findAll('button').find((button) => button.text() === 'Save & retry')
    await rebuild.trigger('click')
    expect(wrapper.emitted('rebuild')).toHaveLength(1)
  })

  it('omits the rebuild button when no rebuild action is available', () => {
    const wrapper = mountDialog({
      frozen: true,
      frozenVersion: 'source-main',
      actions: { save: { label: 'Save' } },
    })
    const labels = wrapper.findAll('button').map((button) => button.text())
    expect(labels).toEqual(['Cancel', 'Save'])
  })

  it('emits update:visible false when cancelled', async () => {
    const wrapper = mountDialog()
    const cancel = wrapper.findAll('button').find((button) => button.text() === 'Cancel')
    await cancel.trigger('click')
    expect(wrapper.emitted('update:visible')?.[0]).toEqual([false])
  })
})
