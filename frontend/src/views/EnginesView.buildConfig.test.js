import { describe, it, expect, beforeEach, vi } from 'vitest'
import { mount, flushPromises } from '@vue/test-utils'
import { reactive } from 'vue'
import { readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { dirname, resolve } from 'node:path'
import EnginesView from './EnginesView.vue'

const toastAdd = vi.fn()
const fetchAll = vi.fn().mockResolvedValue(undefined)
const fetchBuildOptions = vi.fn()
const fetchAudioCppBuildOptions = vi.fn()
const saveVersionBuildConfig = vi.fn()
const syncVersion = vi.fn()
const retryVersion = vi.fn()
const checkLlamaCppUpdates = vi.fn().mockResolvedValue(null)
const checkIkLlamaUpdates = vi.fn().mockResolvedValue(null)
const checkLmdeployUpdates = vi.fn().mockResolvedValue(null)
const checkOnecatVllmUpdates = vi.fn().mockResolvedValue(null)
const checkAudioCppUpdates = vi.fn().mockResolvedValue(null)
const fetchLlamaVersions = vi.fn().mockResolvedValue(undefined)
const fetchAudioCppStatus = vi.fn().mockResolvedValue(undefined)
const fetchSystemStatus = vi.fn().mockResolvedValue(undefined)
const fetchCudaStatus = vi.fn().mockResolvedValue(undefined)
const fetchLmdeployStatus = vi.fn().mockResolvedValue(undefined)
const fetchOnecatVllmStatus = vi.fn().mockResolvedValue(undefined)

vi.mock('vue-router', () => ({
  useRoute: () => ({ hash: '' }),
}))

vi.mock('primevue/useconfirm', () => ({
  useConfirm: () => ({ require: vi.fn() }),
}))

vi.mock('primevue/usetoast', () => ({
  useToast: () => ({ add: toastAdd }),
}))

vi.mock('@/stores/progress', () => ({
  useProgressStore: () => ({
    subscribe: () => () => {},
  }),
}))

vi.mock('@/stores/engines', () => ({
  useEnginesStore: () => enginesStore,
}))

const enginesStore = reactive({
  llamaVersions: [],
  ikLlamaVersions: [],
  lmdeployVersions: [],
  onecatVllmVersions: [],
  audioCppVersions: [],
  audioCppStatus: {
    supported_build_backends: ['cpu', 'cuda', 'vulkan'],
    tracking_ref: 'main',
    active: null,
  },
  engineDescriptors: [{ id: 'audio_cpp', enabled: true, maturity_surfaces: {} }],
  cudaStatus: {},
  lmdeployStatus: {},
  onecatVllmStatus: {},
  systemStatus: {},
  loading: false,
  fetchAll,
  fetchCudaStatus,
  fetchLmdeployStatus,
  fetchOnecatVllmStatus,
  fetchLlamaVersions,
  fetchSystemStatus,
  fetchEngineDescriptors: vi.fn().mockResolvedValue([]),
  fetchAudioCppStatus,
  checkLlamaCppUpdates,
  checkIkLlamaUpdates,
  checkLmdeployUpdates,
  checkOnecatVllmUpdates,
  checkAudioCppUpdates,
  fetchBuildOptions,
  fetchAudioCppBuildOptions,
  fetchBuildSettings: vi.fn().mockResolvedValue({}),
  fetchAudioCppBuildSettings: vi.fn().mockResolvedValue({}),
  saveVersionBuildConfig,
  syncVersion,
  retryVersion,
  scanEngineParams: vi.fn(),
})

function mountEnginesView() {
  return mount(EnginesView, {
    global: {
      directives: { tooltip: () => {} },
      stubs: {
        Button: {
          props: ['label', 'icon', 'text', 'severity', 'loading', 'outlined', 'disabled'],
          emits: ['click'],
          template:
            '<button type="button" :data-label="label" :disabled="disabled" @click="$emit(\'click\')">{{ label }}</button>',
        },
        Tag: { template: '<span><slot /></span>' },
        ProgressBar: true,
        Dialog: {
          props: ['visible', 'header'],
          template: `
            <div v-if="visible !== false" class="dialog-stub" :data-header="header">
              <div class="dialog-header">{{ header }}</div>
              <slot />
              <div class="dialog-footer"><slot name="footer" /></div>
            </div>
          `,
        },
        Select: true,
        InputText: true,
        ToggleSwitch: true,
        Checkbox: { template: '<input type="checkbox" />' },
        InputNumber: { template: '<input type="number" />' },
        EngineDialogHeader: true,
        EngineCheckUpdatesCta: true,
        EngineBuildSettingsHint: true,
        VersionTable: {
          props: ['versions'],
          emits: ['sync', 'retry', 'edit-config', 'activate', 'delete'],
          template:
            '<button type="button" data-testid="edit-config-btn" @click="$emit(\'edit-config\', versions?.[0]?.id)">Edit</button>',
        },
        SwapRoutingPanel: true,
      },
    },
  })
}

describe('EnginesView per-build CMake editor', () => {
  beforeEach(() => {
    toastAdd.mockReset()
    fetchBuildOptions.mockReset()
    fetchAudioCppBuildOptions.mockReset()
    saveVersionBuildConfig.mockReset()
    syncVersion.mockReset()
    retryVersion.mockReset()
    fetchBuildOptions.mockResolvedValue({
      categories: [{ id: 'backends', label: 'GPU', collapsed: false, options: [] }],
      defaults: { cuda: false, build_type: 'Release' },
    })
    fetchAudioCppBuildOptions.mockResolvedValue({
      categories: [{ id: 'backends', label: 'GPU', options: [] }],
      defaults: { cuda: false, build_type: 'RelWithDebInfo' },
    })
    saveVersionBuildConfig.mockResolvedValue({ ok: true })
    syncVersion.mockResolvedValue({ task_id: 'sync-1' })
    retryVersion.mockResolvedValue({ retry: true })

    enginesStore.llamaVersions = [
      {
        id: 'llama_cpp:source-main',
        version: 'source-main',
        type: 'source',
        install_type: 'source',
        source_ref: 'main',
        source_branch: 'main',
        source_ref_type: 'branch',
        cmake_editable: true,
        build_config: { enable_cuda: true, build_type: 'Release' },
        is_active: true,
      },
    ]
    enginesStore.ikLlamaVersions = []
    enginesStore.audioCppVersions = [
      {
        id: 'audio_cpp:source-main',
        version: 'source-main',
        type: 'source',
        install_type: 'source',
        source_branch: 'main',
        cmake_editable: true,
        build_config: { cuda: false, backend: 'cpu', build_type: 'Release' },
        is_active: true,
      },
    ]
  })

  it('opens frozen llama cmake settings without global ref fields', async () => {
    const wrapper = mountEnginesView()
    await flushPromises()
    await wrapper.get('.engine-card').trigger('click')
    await flushPromises()
    await wrapper.get('[data-testid="edit-config-btn"]').trigger('click')
    await flushPromises()

    const cmakeDialog = wrapper.findAll('.dialog-stub').find((node) =>
      String(node.attributes('data-header') || '').includes('CMake settings — llama.cpp'),
    )
    expect(cmakeDialog).toBeTruthy()
    expect(cmakeDialog.text()).toContain('source-main')
    expect(cmakeDialog.text()).toContain('frozen to')
    expect(cmakeDialog.text()).not.toContain('Ref (tag / branch / commit)')
    expect(cmakeDialog.text()).not.toContain('Build Name Suffix')
    const labels = cmakeDialog.findAll('button').map((btn) => btn.text())
    expect(labels).toContain('Save')
    expect(labels).toContain('Save & rebuild')
    expect(labels).not.toContain('Build now')
  })

  it('saves only the selected version config', async () => {
    const wrapper = mountEnginesView()
    await flushPromises()
    await wrapper.get('.engine-card').trigger('click')
    await wrapper.get('[data-testid="edit-config-btn"]').trigger('click')
    await flushPromises()

    const cmakeDialog = wrapper.findAll('.dialog-stub').find((node) =>
      String(node.attributes('data-header') || '').includes('CMake settings'),
    )
    const save = cmakeDialog.findAll('button').find((btn) => btn.text() === 'Save')
    await save.trigger('click')
    await flushPromises()

    expect(saveVersionBuildConfig).toHaveBeenCalledTimes(1)
    expect(saveVersionBuildConfig.mock.calls[0][0]).toBe('llama_cpp:source-main')
    expect(saveVersionBuildConfig.mock.calls[0][1]).toMatchObject({ cuda: true })
    expect(syncVersion).not.toHaveBeenCalled()
    expect(retryVersion).not.toHaveBeenCalled()
    expect(toastAdd).toHaveBeenCalledWith(
      expect.objectContaining({ summary: 'Build config saved' }),
    )
  })

  it('save and rebuild syncs a ready branch install', async () => {
    const wrapper = mountEnginesView()
    await flushPromises()
    await wrapper.get('.engine-card').trigger('click')
    await wrapper.get('[data-testid="edit-config-btn"]').trigger('click')
    await flushPromises()

    const cmakeDialog = wrapper.findAll('.dialog-stub').find((node) =>
      String(node.attributes('data-header') || '').includes('CMake settings'),
    )
    const rebuild = cmakeDialog.findAll('button').find((btn) => btn.text() === 'Save & rebuild')
    await rebuild.trigger('click')
    await flushPromises()

    expect(saveVersionBuildConfig).toHaveBeenCalledTimes(1)
    expect(syncVersion).toHaveBeenCalledWith('llama_cpp:source-main')
    expect(retryVersion).not.toHaveBeenCalled()
  })

  it('save and retry rebuilds a failed version in place', async () => {
    enginesStore.llamaVersions = [
      {
        id: 'llama_cpp:source-fail',
        version: 'source-fail',
        type: 'source',
        install_type: 'source',
        source_branch: 'main',
        build_status: 'failed',
        retryable: true,
        cmake_editable: true,
        build_config: { cuda: false },
        is_active: false,
      },
    ]
    const wrapper = mountEnginesView()
    await flushPromises()
    await wrapper.get('.engine-card').trigger('click')
    await wrapper.get('[data-testid="edit-config-btn"]').trigger('click')
    await flushPromises()

    const cmakeDialog = wrapper.findAll('.dialog-stub').find((node) =>
      String(node.attributes('data-header') || '').includes('CMake settings'),
    )
    const labels = cmakeDialog.findAll('button').map((btn) => btn.text())
    expect(labels).toContain('Save & retry')
    expect(labels).not.toContain('Save & rebuild')
    const retry = cmakeDialog.findAll('button').find((btn) => btn.text() === 'Save & retry')
    await retry.trigger('click')
    await flushPromises()

    expect(saveVersionBuildConfig).toHaveBeenCalledWith(
      'llama_cpp:source-fail',
      expect.any(Object),
    )
    expect(retryVersion).toHaveBeenCalledWith('llama_cpp:source-fail')
    expect(syncVersion).not.toHaveBeenCalled()
  })

  it('opens frozen audio.cpp cmake settings from the audio engine modal', async () => {
    const wrapper = mountEnginesView()
    await flushPromises()
    const audioCard = wrapper.findAll('.engine-card').find((card) => card.text().includes('audio.cpp'))
    await audioCard.trigger('click')
    await flushPromises()
    await wrapper.get('[data-testid="edit-config-btn"]').trigger('click')
    await flushPromises()

    const cmakeDialog = wrapper.findAll('.dialog-stub').find((node) =>
      String(node.attributes('data-header') || '').includes('CMake settings — audio.cpp'),
    )
    expect(cmakeDialog).toBeTruthy()
    expect(cmakeDialog.text()).not.toContain('Repo URL')
    expect(cmakeDialog.text()).not.toContain('Build Name Suffix')
    const save = cmakeDialog.findAll('button').find((btn) => btn.text() === 'Save')
    await save.trigger('click')
    await flushPromises()
    expect(saveVersionBuildConfig).toHaveBeenCalledWith(
      'audio_cpp:source-main',
      expect.objectContaining({ backend: 'cpu' }),
    )
  })
})

describe('EnginesView build dialog theming', () => {
  it('uses app theme tokens instead of PrimeVue surface grays', () => {
    const src = readFileSync(
      resolve(dirname(fileURLToPath(import.meta.url)), './EnginesView.vue'),
      'utf8',
    )
    expect(src).toContain('background: var(--bg-tertiary)')
    expect(src).toContain('background: var(--status-info-soft)')
    expect(src).not.toMatch(/var\(--surface-50/)
    expect(src).not.toMatch(/background: var\(--surface-100\)/)
  })
})
