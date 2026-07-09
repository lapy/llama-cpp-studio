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
  audioCppStatus: { supported_build_backends: ['cpu', 'cuda', 'vulkan'] },
  engineDescriptors: [{ id: 'audio_cpp', label: 'audio.cpp', enabled: true }],
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
  fetchAudioCppStatus: vi.fn().mockResolvedValue({}),
  checkAudioCppUpdates: vi.fn().mockResolvedValue(null),
  buildAudioCppSource,
  syncVersion,
})

function mountEnginesView() {
  return mount(EnginesView, {
    global: {
      directives: {
        tooltip: () => {},
      },
      stubs: {
        Button: { template: '<button><slot /></button>' },
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
        ProgressTracker: {
          props: ['taskId'],
          template: '<div data-testid="sync-tracker">{{ taskId || "" }}</div>',
        },
        EngineDialogHeader: true,
        EngineCheckUpdatesCta: true,
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
    taskUpdatedCallbacks.length = 0

    fetchAll.mockResolvedValue(undefined)
    fetchCudaStatus.mockResolvedValue(undefined)
    fetchLmdeployStatus.mockResolvedValue(undefined)
    fetchOnecatVllmStatus.mockResolvedValue(undefined)
    fetchLlamaVersions.mockResolvedValue(undefined)
    fetchSystemStatus.mockResolvedValue(undefined)
    syncVersion.mockResolvedValue({ task_id: 'build_sync_source-main_1700000000' })
    buildAudioCppSource.mockResolvedValue({
      task_id: 'build_audio_cpp_source-release-0.2_1700000000',
      version_name: 'source-release-0.2-1234',
      source_ref: 'release-0.2',
      source_ref_type: 'branch',
    })
  })

  it('binds sync dialog ProgressTracker to API task_id', async () => {
    const wrapper = mountEnginesView()
    await flushPromises()

    const syncButtons = wrapper.findAll('[data-testid="sync-btn"]')
    expect(syncButtons.length).toBeGreaterThan(0)
    await syncButtons[0].trigger('click')
    await flushPromises()

    expect(syncVersion).toHaveBeenCalledWith('llama_cpp:source-main')
    expect(wrapper.find('[data-testid="sync-tracker"]').text()).toBe(
      'build_sync_source-main_1700000000',
    )
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
    const fetchAudioCppStatus = enginesStore.fetchAudioCppStatus
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
})
