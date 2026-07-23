import { describe, it, expect, beforeEach, vi } from 'vitest'
import { mount, flushPromises } from '@vue/test-utils'
import { reactive, ref } from 'vue'
import EnginesView from './EnginesView.vue'

const toastAdd = vi.fn()
const fetchAll = vi.fn()
const fetchCudaStatus = vi.fn()
const fetchLmdeployStatus = vi.fn()
const fetchOnecatVllmStatus = vi.fn()
const fetchLlamaVersions = vi.fn()
const fetchSystemStatus = vi.fn()
const syncVersion = vi.fn()
const buildAudioCppSource = vi.fn()
const checkAudioCppUpdates = vi.fn()
const updateAudioCpp = vi.fn()
const fetchAudioCppBuildSettings = vi.fn()
const fetchAudioCppStatus = vi.fn()
const migrateAudioCppDefaults = vi.fn()
const scanEngineParams = vi.fn()
const routerPush = vi.fn()

const taskUpdatedCallbacks = []

vi.mock('primevue/useconfirm', () => ({
  useConfirm: () => ({ require: vi.fn() }),
}))

vi.mock('primevue/usetoast', () => ({
  useToast: () => ({ add: toastAdd }),
}))

vi.mock('@/stores/engines', () => ({
  useEnginesStore: () => enginesStore,
}))

vi.mock('@/stores/progress', () => ({
  useProgressStore: () => ({
    subscribe: (eventType, cb) => {
      if (eventType === 'task_updated') taskUpdatedCallbacks.push(cb)
      return () => {}
    },
  }),
}))

const enginesStore = reactive({
  llamaVersions: [
    {
      id: 'llama_cpp:source-main',
      version: 'source-main',
      repository_source: 'llama.cpp',
      source_ref: 'main',
      source_branch: 'main',
      source_ref_type: 'branch',
    },
  ],
  ikLlamaVersions: [],
  lmdeployVersions: [],
  onecatVllmVersions: [],
  audioCppVersions: [],
  audioCppStatus: {
    supported_build_backends: ['cpu', 'cuda', 'vulkan'],
    tracking_ref: 'release-0.3',
    active: { version: 'source-release-0.3' },
    contract_changed: false,
    families: [],
    tasks: [],
  },
  engineDescriptors: [{
    id: 'audio_cpp',
    label: 'audio.cpp',
    enabled: true,
    maturity_surfaces: {
      speech_asr: 'stable',
      generic_tasks: 'limited',
      catalog_json: 'stable',
      heuristic_discovery: 'experimental',
    },
  }],
  cudaStatus: {},
  lmdeployStatus: {},
  onecatVllmStatus: {},
  systemStatus: {},
  fetchAll,
  fetchCudaStatus,
  fetchLmdeployStatus,
  fetchOnecatVllmStatus,
  fetchLlamaVersions,
  fetchSystemStatus,
  fetchEngineDescriptors: vi.fn().mockResolvedValue([]),
  fetchAudioCppStatus,
  checkAudioCppUpdates,
  fetchAudioCppBuildSettings,
  updateAudioCpp,
  buildAudioCppSource,
  migrateAudioCppDefaults,
  scanEngineParams,
  syncVersion,
})

function mountEnginesView() {
  return mount(EnginesView, {
    global: {
      directives: {
        tooltip: () => {},
      },
      mocks: {
        $router: { push: routerPush },
      },
      stubs: {
        Button: {
          props: ['label', 'icon', 'text', 'severity', 'loading', 'outlined', 'disabled'],
          emits: ['click'],
          template:
            '<button :data-label="label" :disabled="disabled" @click="$emit(`click`)">{{ label }}<slot /></button>',
        },
        Tag: { template: '<span><slot /></span>' },
        ProgressBar: true,
        Dialog: {
          props: ['visible'],
          template: '<div class="dialog-stub"><slot /></div>',
        },
        Dropdown: true,
        InputText: true,
        InputSwitch: true,
        Checkbox: { template: '<input type="checkbox" />' },
        InputNumber: { template: '<input type="number" />' },
        EngineDialogHeader: true,
        EngineCheckUpdatesCta: {
          emits: ['check'],
          template:
            '<button data-testid="audio-check-updates" @click="$emit(\'check\')">Check</button>',
        },
        EngineBuildSettingsHint: true,
        VersionTable: {
          emits: ['sync'],
          template: '<button data-testid="sync-btn" @click="$emit(\'sync\', \'llama_cpp:source-main\')">Sync</button>',
        },
      },
    },
  })
}

describe('EnginesView task integration', () => {
  beforeEach(() => {
    toastAdd.mockReset()
    fetchAll.mockReset()
    fetchCudaStatus.mockReset()
    fetchLmdeployStatus.mockReset()
    fetchOnecatVllmStatus.mockReset()
    fetchLlamaVersions.mockReset()
    fetchSystemStatus.mockReset()
    syncVersion.mockReset()
    buildAudioCppSource.mockReset()
    checkAudioCppUpdates.mockReset()
    updateAudioCpp.mockReset()
    fetchAudioCppBuildSettings.mockReset()
    fetchAudioCppStatus.mockReset()
    scanEngineParams.mockReset()
    routerPush.mockReset()
    taskUpdatedCallbacks.length = 0

    enginesStore.audioCppStatus = {
      supported_build_backends: ['cpu', 'cuda', 'vulkan'],
      tracking_ref: 'release-0.3',
      active: { version: 'source-release-0.3' },
      contract_changed: false,
      families: [],
      tasks: [],
    }

    fetchAll.mockResolvedValue(undefined)
    fetchCudaStatus.mockResolvedValue(undefined)
    fetchLmdeployStatus.mockResolvedValue(undefined)
    fetchOnecatVllmStatus.mockResolvedValue(undefined)
    fetchLlamaVersions.mockResolvedValue(undefined)
    fetchSystemStatus.mockResolvedValue(undefined)
    fetchAudioCppStatus.mockResolvedValue(enginesStore.audioCppStatus)
    checkAudioCppUpdates.mockResolvedValue(null)
    fetchAudioCppBuildSettings.mockResolvedValue({
      tracking_ref: 'release-0.3',
      repository_url: 'https://github.com/0xShug0/audio.cpp.git',
      backend: 'cpu',
    })
    updateAudioCpp.mockResolvedValue({
      task_id: 'build_sync_audio_1',
      sync: true,
      source_ref: 'release-0.3',
      source_ref_type: 'branch',
      version_name: 'source-release-0.3',
    })
    syncVersion.mockResolvedValue({ task_id: 'build_sync_source-main_1700000000' })
    buildAudioCppSource.mockResolvedValue({
      task_id: 'build_audio_cpp_source-release-0.2_1700000000',
      version_name: 'source-release-0.2-1234',
      source_ref: 'release-0.2',
      source_ref_type: 'branch',
    })
    scanEngineParams.mockResolvedValue({ ok: true, param_count: 12 })
  })

  it('starts sync and points users at notifications toast', async () => {
    const wrapper = mountEnginesView()
    await flushPromises()

    const syncButtons = wrapper.findAll('[data-testid="sync-btn"]')
    expect(syncButtons.length).toBeGreaterThan(0)
    await syncButtons[0].trigger('click')
    await flushPromises()

    expect(syncVersion).toHaveBeenCalledWith('llama_cpp:source-main')
    expect(toastAdd).toHaveBeenCalledWith(
      expect.objectContaining({
        severity: 'success',
        summary: 'Sync started',
        detail: 'Track progress in notifications',
      }),
    )
    expect(wrapper.find('[data-testid="sync-tracker"]').exists()).toBe(false)
  })

  it('refreshes CUDA status when cuda install task completes', async () => {
    mountEnginesView()
    await flushPromises()

    expect(taskUpdatedCallbacks).toHaveLength(1)
    await taskUpdatedCallbacks[0]({
      task_id: 'install_cuda_install_1',
      type: 'install',
      status: 'completed',
      metadata: { manager: 'cuda' },
    })
    await flushPromises()

    expect(fetchCudaStatus).toHaveBeenCalledTimes(1)
  })

  it('refreshes LMDeploy state when lmdeploy task completes', async () => {
    mountEnginesView()
    await flushPromises()

    await taskUpdatedCallbacks[0]({
      task_id: 'install_lmdeploy_install_1',
      type: 'install',
      status: 'completed',
      metadata: { manager: 'lmdeploy' },
    })
    await flushPromises()

    expect(fetchLmdeployStatus).toHaveBeenCalledTimes(1)
    expect(fetchLlamaVersions).toHaveBeenCalledTimes(1)
  })

  it('shows error toast when lmdeploy install task fails', async () => {
    mountEnginesView()
    await flushPromises()

    await taskUpdatedCallbacks[0]({
      task_id: 'install_lmdeploy_install_1',
      type: 'install',
      status: 'failed',
      message: 'pip failed',
      metadata: { manager: 'lmdeploy' },
    })
    await flushPromises()

    expect(toastAdd).toHaveBeenCalledWith(
      expect.objectContaining({
        severity: 'error',
        summary: 'LMDeploy install failed',
        detail: 'pip failed',
      }),
    )
  })

  it('refreshes engine versions when build task completes', async () => {
    mountEnginesView()
    await flushPromises()

    await taskUpdatedCallbacks[0]({
      task_id: 'build_source-main_1',
      type: 'build',
      status: 'completed',
    })
    await flushPromises()

    expect(fetchLlamaVersions).toHaveBeenCalledTimes(1)
    expect(fetchSystemStatus).toHaveBeenCalledTimes(1)
  })

  it('refreshes audio.cpp status when audio build task completes', async () => {
    mountEnginesView()
    await flushPromises()

    await taskUpdatedCallbacks[0]({
      task_id: 'build_audio_cpp_source-release-0.2_1',
      type: 'build',
      status: 'completed',
      metadata: { engine: 'audio_cpp' },
    })
    await flushPromises()

    expect(fetchAudioCppStatus).toHaveBeenCalledTimes(1)
  })

  it('shows tracking ref and runs update from audio.cpp modal', async () => {
    checkAudioCppUpdates.mockResolvedValue({
      update_available: true,
      tracking_ref: 'release-0.3',
      latest_version: 'abcdef0123456789',
      latest_commit: { html_url: 'https://example.test/commit/abcdef' },
    })

    const wrapper = mountEnginesView()
    await flushPromises()
    wrapper.vm.openEngineModal('audio_cpp')
    await flushPromises()

    expect(checkAudioCppUpdates).toHaveBeenCalled()
    expect(wrapper.text()).toContain('release-0.3')
    expect(wrapper.text()).toContain('abcdef01')

    const banner = wrapper.find('.update-banner')
    expect(banner.exists()).toBe(true)
    await banner.find('button').trigger('click')
    await flushPromises()

    expect(fetchAudioCppBuildSettings).toHaveBeenCalled()
    expect(updateAudioCpp).toHaveBeenCalledWith(
      expect.objectContaining({
        source_ref: 'release-0.3',
        repository_url: 'https://github.com/0xShug0/audio.cpp.git',
      }),
    )
    expect(toastAdd).toHaveBeenCalledWith(
      expect.objectContaining({
        severity: 'success',
        summary: 'audio.cpp sync started',
      }),
    )
  })

  it('shows contract-changed warning in audio.cpp modal', async () => {
    enginesStore.audioCppStatus = {
      ...enginesStore.audioCppStatus,
      contract_changed: true,
      active: { version: 'source-release-0.3' },
      tracking_ref: 'release-0.3',
      families: ['qwen3_tts', 'omnivoice'],
      tasks: ['tts', 'asr'],
      capability_delta: {
        added_families: ['omnivoice'],
        removed_families: [],
        added_tasks: ['asr'],
        removed_tasks: [],
      },
      affected_models: [
        { id: 'audio-omnivoice', name: 'OmniVoice', family: 'omnivoice', task: 'tts' },
      ],
    }

    const wrapper = mountEnginesView()
    await flushPromises()
    wrapper.vm.openEngineModal('audio_cpp')
    await flushPromises()

    expect(wrapper.text()).toContain('contract fingerprint changed')
    expect(wrapper.text()).toContain('2 families')
    expect(wrapper.text()).toContain('tts, asr')
    expect(wrapper.text()).toContain('Added families: omnivoice')

    await wrapper.get('button[data-label="Rescan CLI"]').trigger('click')
    await flushPromises()
    expect(scanEngineParams).toHaveBeenCalledWith('audio_cpp')

    await wrapper.get('button[data-label="Review affected models"]').trigger('click')
    await flushPromises()
    expect(wrapper.text()).toContain('OmniVoice')

    migrateAudioCppDefaults.mockResolvedValue({ migrated_count: 1 })
    await wrapper.get('button[data-label="Migrate defaults"]').trigger('click')
    await flushPromises()
    expect(migrateAudioCppDefaults).toHaveBeenCalledWith({
      model_ids: ['audio-omnivoice'],
      mark_reviewed: true,
    })

    await wrapper.get('button[data-label="Open Models"]').trigger('click')
    expect(routerPush).toHaveBeenCalledWith('/models')
  })

  it('explains that commit builds do not change the tracking ref', async () => {
    const wrapper = mountEnginesView()
    await flushPromises()
    wrapper.vm.openEngineModal('audio_cpp')
    await flushPromises()
    await wrapper.vm.openAudioCppBuildDialog()
    await flushPromises()

    expect(wrapper.text()).toContain('Building a commit installs that tip')
    expect(wrapper.text()).toContain('will not follow the detached commit')
  })
})
