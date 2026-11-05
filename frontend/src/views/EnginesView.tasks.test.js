import { describe, it, expect, beforeEach, vi } from 'vitest'
import { mount, flushPromises } from '@vue/test-utils'
import { reactive } from 'vue'
import EnginesView from './EnginesView.vue'
import EngineUpdateBanner from '@/components/system/EngineUpdateBanner.vue'
import VersionTable from '@/components/system/VersionTable.vue'
import { engineStoreAccessors } from '@/test-fixtures/engineStoreAccessors'

const toastAdd = vi.fn()
const fetchAll = vi.fn()
const fetchCudaStatus = vi.fn()
const fetchLmdeployStatus = vi.fn()
const fetchOnecatVllmStatus = vi.fn()
const fetchSglangStatus = vi.fn()
const fetchLlamaVersions = vi.fn()
const fetchSystemStatus = vi.fn()
const syncVersion = vi.fn()
const buildAudioCppSource = vi.fn()
const checkAudioCppUpdates = vi.fn()
const updateAudioCpp = vi.fn()
const fetchAudioCppBuildSettings = vi.fn()
const fetchAudioCppStatus = vi.fn()
const checkLlamaCppUpdates = vi.fn()
const checkIkLlamaUpdates = vi.fn()
const checkUnslothLlamaUpdates = vi.fn()
const checkLmdeployUpdates = vi.fn()
const checkOnecatVllmUpdates = vi.fn()
const scanEngineParams = vi.fn()
const routerPush = vi.fn()

const taskUpdatedCallbacks = []

vi.mock('vue-router', () => ({
  useRoute: () => ({ hash: '' }),
}))

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
  ...engineStoreAccessors(),
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
  unslothLlamaVersions: [],
  lmdeployVersions: [],
  onecatVllmVersions: [],
  sglangVersions: [],
  sglangV100Versions: [],
  vllmVersions: [],
  audioCppVersions: [],
  engineStatusById: {
    audio_cpp: {
      supported_build_backends: ['cpu', 'cuda'],
      tracking_ref: 'release-0.3',
      active: { version: 'source-release-0.3' },
      contract_changed: false,
      families: [],
      tasks: [],
    },
  },
  engineDescriptors: [
    {
      id: 'audio_cpp',
      label: 'audio.cpp',
      enabled: true,
      maturity_surfaces: {
        speech_asr: 'stable',
        generic_tasks: 'stable',
        catalog_json: 'stable',
        heuristic_discovery: 'experimental',
      },
    },
  ],
  cudaStatus: {},
  systemStatus: {},
  fetchAll,
  fetchCudaStatus,
  fetchLmdeployStatus,
  fetchOnecatVllmStatus,
  fetchSglangStatus,
  fetchLlamaVersions,
  fetchSystemStatus,
  fetchEngineDescriptors: vi.fn().mockResolvedValue([]),
  fetchAudioCppStatus,
  checkAudioCppUpdates,
  checkLlamaCppUpdates,
  checkIkLlamaUpdates,
  checkUnslothLlamaUpdates,
  checkLmdeployUpdates,
  checkOnecatVllmUpdates,
  fetchAudioCppBuildSettings,
  updateAudioCpp,
  buildAudioCppSource,
  scanEngineParams,
  syncVersion,
})

function mountEnginesView({ withHeader = false } = {}) {
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
          template: withHeader
            ? '<div class="dialog-stub"><header class="dialog-header-stub"><slot name="header" /></header><slot /></div>'
            : '<div class="dialog-stub"><slot /></div>',
        },
        Select: true,
        InputText: true,
        ToggleSwitch: true,
        Checkbox: { template: '<input type="checkbox" />' },
        InputNumber: { template: '<input type="number" />' },
        EngineDialogHeader: withHeader
          ? {
              props: ['title'],
              template:
                '<div class="eh-stub" :data-title="title"><slot name="leading" /><slot name="tags" /><slot name="actions" /><slot name="more" /></div>',
            }
          : true,
        EngineCheckUpdatesCta: {
          props: ['hint'],
          emits: ['check'],
          template:
            '<button data-testid="audio-check-updates" :data-hint="hint" @click="$emit(\'check\')">Check</button>',
        },
        EngineBuildSettingsHint: true,
        VersionTable: {
          // Declares the pinned props like the real component so prop
          // semantics (Boolean casting, kebab-case mapping) match.
          props: ['canVerify', 'emptyMessage'],
          emits: ['sync'],
          template:
            '<button data-testid="sync-btn" @click="$emit(\'sync\', \'llama_cpp:source-main\')">Sync</button>',
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
    fetchSglangStatus.mockReset()
    fetchLlamaVersions.mockReset()
    fetchSystemStatus.mockReset()
    syncVersion.mockReset()
    buildAudioCppSource.mockReset()
    checkAudioCppUpdates.mockReset()
    updateAudioCpp.mockReset()
    fetchAudioCppBuildSettings.mockReset()
    fetchAudioCppStatus.mockReset()
    checkLlamaCppUpdates.mockReset()
    checkIkLlamaUpdates.mockReset()
    checkUnslothLlamaUpdates.mockReset()
    checkLmdeployUpdates.mockReset()
    checkOnecatVllmUpdates.mockReset()
    scanEngineParams.mockReset()
    routerPush.mockReset()
    taskUpdatedCallbacks.length = 0

    enginesStore.engineStatusById = {
      ...enginesStore.engineStatusById,
      audio_cpp: {
        supported_build_backends: ['cpu', 'cuda'],
        tracking_ref: 'release-0.3',
        active: { version: 'source-release-0.3' },
        contract_changed: false,
        families: [],
        tasks: [],
      },
    }

    fetchAll.mockResolvedValue(undefined)
    fetchCudaStatus.mockResolvedValue(undefined)
    fetchLmdeployStatus.mockResolvedValue(undefined)
    fetchOnecatVllmStatus.mockResolvedValue(undefined)
    fetchSglangStatus.mockResolvedValue(undefined)
    fetchLlamaVersions.mockResolvedValue(undefined)
    fetchSystemStatus.mockResolvedValue(undefined)
    fetchAudioCppStatus.mockResolvedValue(enginesStore.engineStatusById.audio_cpp)
    checkAudioCppUpdates.mockResolvedValue(null)
    checkLlamaCppUpdates.mockResolvedValue(null)
    checkIkLlamaUpdates.mockResolvedValue(null)
    checkUnslothLlamaUpdates.mockResolvedValue({})
    checkLmdeployUpdates.mockResolvedValue({})
    checkOnecatVllmUpdates.mockResolvedValue({})
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

  it('refreshes SGLang V100 state when an install task completes', async () => {
    mountEnginesView()
    await flushPromises()

    await taskUpdatedCallbacks[0]({
      task_id: 'install_sglang_v100_install_source_1',
      type: 'install',
      status: 'completed',
      metadata: { manager: 'sglang_v100' },
    })
    await flushPromises()

    expect(fetchSglangStatus).toHaveBeenCalledWith('sglang_v100')
    expect(fetchLlamaVersions).toHaveBeenCalledTimes(1)
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
      latest_version: 'release-0.5.1',
      update_channel: 'release',
      latest_release: {
        tag_name: 'release-0.5.1',
        html_url: 'https://github.com/0xShug0/audio.cpp/releases/tag/release-0.5.1',
      },
      latest_commit: { html_url: 'https://example.test/commit/abcdef' },
    })

    const wrapper = mountEnginesView()
    await flushPromises()
    wrapper.vm.openEngineModal('audio_cpp')
    await flushPromises()

    expect(checkAudioCppUpdates).toHaveBeenCalled()
    expect(wrapper.text()).toContain('release-0.5.1')

    const banner = wrapper.find('.engine-update-banner')
    expect(banner.exists()).toBe(true)
    await banner.find('button').trigger('click')
    await flushPromises()

    expect(fetchAudioCppBuildSettings).toHaveBeenCalled()
    expect(updateAudioCpp).toHaveBeenCalledWith(
      expect.objectContaining({
        from_release: true,
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

  it('does not warn about contract fingerprint changes after an automatic scan', async () => {
    const current = enginesStore.engineStatusById
    enginesStore.engineStatusById = {
      ...current,
      audio_cpp: {
        ...current.audio_cpp,
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
      },
    }

    const wrapper = mountEnginesView()
    await flushPromises()
    wrapper.vm.openEngineModal('audio_cpp')
    await flushPromises()

    expect(wrapper.text()).not.toContain('contract fingerprint changed')
    expect(wrapper.text()).not.toContain('Review affected models')
    expect(wrapper.text()).not.toContain('Added families: omnivoice')
  })

  it('explains that commit builds do not change the tracking ref', async () => {
    const wrapper = mountEnginesView()
    await flushPromises()
    wrapper.vm.openEngineModal('audio_cpp')
    await flushPromises()
    await wrapper.vm.openAudioCppBuildSettings()
    await flushPromises()

    expect(wrapper.text()).toContain('Building a commit installs that tip')
    expect(wrapper.text()).toContain('will not follow the detached commit')
  })

  it('openEngineModal dispatches only the target engine family probe', async () => {
    // Single mount (unmounting the heavy view mid-transition crashes the
    // test harness); opening a dialog mounts family panels that run their
    // own probes, so this pins DISPATCH (the target probe fired) and
    // CROSS-ABSENCE (no other family's probe fired) — what openEngineModal
    // owns — instead of exact counts.
    const wrapper = mountEnginesView()
    await flushPromises()
    const OWN_PROBES = {
      llama_cpp: () => expect(checkLlamaCppUpdates).toHaveBeenCalled(),
      ik_llama: () => expect(checkIkLlamaUpdates).toHaveBeenCalled(),
      unsloth_llama: () => expect(checkUnslothLlamaUpdates).toHaveBeenCalled(),
      lmdeploy: () => expect(checkLmdeployUpdates).toHaveBeenCalled(),
      '1cat_vllm': () => expect(checkOnecatVllmUpdates).toHaveBeenCalled(),
      sglang: () => expect(fetchSglangStatus).toHaveBeenCalledWith('sglang'),
      sglang_v100: () => expect(fetchSglangStatus).toHaveBeenCalledWith('sglang_v100'),
      vllm: () => expect(fetchSglangStatus).toHaveBeenCalledWith('vllm'),
      audio_cpp: () => {
        expect(fetchAudioCppStatus).toHaveBeenCalled()
        expect(checkAudioCppUpdates).toHaveBeenCalled()
      },
    }
    const ALL_PROBES = {
      checkLlamaCppUpdates,
      checkIkLlamaUpdates,
      checkUnslothLlamaUpdates,
      checkLmdeployUpdates,
      checkOnecatVllmUpdates,
      checkAudioCppUpdates,
      fetchSglangStatus,
      fetchAudioCppStatus,
    }
    const FAMILY = {
      llama_cpp: ['checkLlamaCppUpdates'],
      ik_llama: ['checkIkLlamaUpdates'],
      unsloth_llama: ['checkUnslothLlamaUpdates'],
      lmdeploy: ['checkLmdeployUpdates'],
      '1cat_vllm': ['checkOnecatVllmUpdates'],
      sglang: ['fetchSglangStatus'],
      sglang_v100: ['fetchSglangStatus'],
      vllm: ['fetchSglangStatus'],
      audio_cpp: ['fetchAudioCppStatus', 'checkAudioCppUpdates'],
    }
    for (const [engineKey, ownProbe] of Object.entries(OWN_PROBES)) {
      for (const fn of Object.values(ALL_PROBES)) fn.mockClear()
      wrapper.vm.openEngineModal(engineKey)
      await flushPromises()
      ownProbe()
      for (const [name, fn] of Object.entries(ALL_PROBES)) {
        if (!FAMILY[engineKey].includes(name)) {
          expect(fn, `${engineKey}: ${name} must not fire`).not.toHaveBeenCalled()
        }
      }
    }
  })

  it('task-completion handling carries byte-exact labels, fallbacks and timings per family', async () => {
    mountEnginesView()
    await flushPromises()
    const cb = taskUpdatedCallbacks[0]

    await cb({
      task_id: 'install_onecat_vllm_1',
      type: 'install',
      status: 'failed',
      metadata: { manager: 'onecat_vllm' },
    })
    expect(toastAdd).toHaveBeenCalledWith(
      expect.objectContaining({
        // manager id onecat_vllm renders as the ENGINE label
        summary: '1Cat-vLLM install failed',
        detail: '1Cat-vLLM operation failed',
        life: 5000,
      }),
    )
    expect(fetchOnecatVllmStatus).toHaveBeenCalledTimes(1)

    toastAdd.mockClear()
    fetchOnecatVllmStatus.mockClear()
    await cb({
      task_id: 'install_sglang_1',
      type: 'install',
      status: 'failed',
      metadata: { manager: 'sglang' },
    })
    expect(toastAdd).toHaveBeenCalledWith(
      expect.objectContaining({
        summary: 'SGLang install failed',
        detail: 'Engine operation failed',
        life: 6000,
      }),
    )
    expect(fetchSglangStatus).toHaveBeenCalledWith('sglang')

    toastAdd.mockClear()
    fetchSglangStatus.mockClear()
    await cb({
      task_id: 'install_vllm_1',
      type: 'install',
      status: 'failed',
      message: 'boom',
      metadata: { manager: 'vllm' },
    })
    expect(toastAdd).toHaveBeenCalledWith(
      expect.objectContaining({ summary: 'vLLM install failed', detail: 'boom', life: 6000 }),
    )

    toastAdd.mockClear()
    fetchLlamaVersions.mockClear()
    await cb({
      task_id: 'install_unsloth_llama_1',
      type: 'install',
      status: 'failed',
      metadata: { manager: 'unsloth_llama' },
    })
    expect(toastAdd).toHaveBeenCalledWith(
      expect.objectContaining({
        summary: 'Unsloth llama.cpp install failed',
        detail: 'Engine operation failed',
        life: 6000,
      }),
    )
    expect(fetchLlamaVersions).toHaveBeenCalledTimes(1)

    toastAdd.mockClear()
    fetchSglangStatus.mockClear()
    await cb({
      task_id: 'install_lmdeploy_1',
      type: 'install',
      status: 'failed',
      metadata: { manager: 'lmdeploy' },
    })
    expect(toastAdd).toHaveBeenCalledWith(
      expect.objectContaining({
        summary: 'LMDeploy install failed',
        detail: 'LMDeploy operation failed',
        life: 5000,
      }),
    )
  })

  it('lmdeploy dialog renders the PyPI banner, install buttons and no V100 note', async () => {
    checkLmdeployUpdates.mockResolvedValue({ latest_version: '0.9.0' })
    const wrapper = mountEnginesView()
    await flushPromises()
    wrapper.vm.openEngineModal('lmdeploy')
    await flushPromises()

    const body = wrapper.find('.engine-modal-body')
    expect(body.exists()).toBe(true)
    expect(body.find('[data-label="From PyPI"]').exists()).toBe(true)
    expect(body.find('[data-label="From source"]').exists()).toBe(true)
    expect(body.find('[data-label="From release"]').exists()).toBe(false)
    const banner = body.find('.engine-update-banner')
    expect(banner.exists()).toBe(true)
    expect(banner.text()).toContain('v0.9.0')
    expect(banner.find('a').attributes('href')).toBe('https://pypi.org/project/lmdeploy/')
    expect(banner.find('a').text()).toBe('View on PyPI')
    expect(wrapper.text()).not.toContain('Tesla V100 (SM70)')
  })

  it('1Cat dialog renders the GitHub banner, release button and the V100 note', async () => {
    checkOnecatVllmUpdates.mockResolvedValue({ latest_version: '0.9.0' })
    const wrapper = mountEnginesView()
    await flushPromises()
    wrapper.vm.openEngineModal('1cat_vllm')
    await flushPromises()

    const body = wrapper.find('.engine-modal-body')
    expect(body.exists()).toBe(true)
    expect(body.find('[data-label="From release"]').exists()).toBe(true)
    expect(body.find('[data-label="From source"]').exists()).toBe(true)
    expect(body.find('[data-label="From PyPI"]').exists()).toBe(false)
    const banner = body.find('.engine-update-banner')
    expect(banner.exists()).toBe(true)
    expect(banner.text()).toContain('v0.9.0')
    expect(banner.find('a').attributes('href')).toBe(
      'https://github.com/1CatAI/1Cat-vLLM/releases/latest',
    )
    expect(banner.find('a').text()).toBe('View release')
    expect(wrapper.text()).toContain('Tesla V100 (SM70)')
  })

  it('dialog headers render the per-engine title, icon and action labels', async () => {
    const CASES = {
      lmdeploy: {
        title: 'LMDeploy',
        icon: 'pi-server',
        settings: 'Build settings',
        refreshAria: true,
      },
      '1cat_vllm': {
        title: '1Cat-vLLM',
        icon: 'pi-bolt',
        settings: 'Build settings',
        refreshAria: true,
      },
      // The panel trio: same header SHAPE but no aria/tooltip on the
      // Reload/Rescan buttons — a real divergence the merge must preserve.
      sglang: {
        title: 'SGLang',
        icon: 'pi-sparkles',
        settings: 'Install settings',
        refreshAria: false,
      },
      sglang_v100: {
        title: 'SGLang V100',
        icon: 'pi-bolt',
        settings: 'Install settings',
        refreshAria: false,
      },
      // vllm's header icon is pi-sparkles, NOT its picker pi-server — real
      // per-surface divergence, pinned so refactors cannot "unify" it away.
      vllm: {
        title: 'vLLM',
        icon: 'pi-sparkles',
        settings: 'Install settings',
        refreshAria: false,
      },
    }
    const wrapper = mountEnginesView({ withHeader: true })
    await flushPromises()
    for (const [engineKey, expected] of Object.entries(CASES)) {
      wrapper.vm.openEngineModal(engineKey)
      await flushPromises()
      const header = wrapper.find('.eh-stub')
      expect(header.exists(), engineKey).toBe(true)
      expect(header.attributes('data-title'), engineKey).toBe(expected.title)
      expect(header.find('i').classes(), engineKey).toContain(expected.icon)
      expect(
        header.find(`[aria-label="${expected.settings}"]`).exists(),
        `${engineKey}: ${expected.settings}`,
      ).toBe(true)
      expect(
        header.find('[aria-label="Reload versions and status"]').exists(),
        `${engineKey}: reload aria`,
      ).toBe(expected.refreshAria)
      expect(
        header.find('[aria-label="Rescan CLI parameters"]').exists(),
        `${engineKey}: rescan aria`,
      ).toBe(expected.refreshAria)
      expect(header.text()).toContain('Reload versions and status')
      expect(header.text()).toContain('Rescan CLI parameters')
    }
  })

  it('native dialog headers render engine marks, settings presence and full a11y', async () => {
    const CASES = {
      llama_cpp: {
        title: 'llama.cpp',
        markClass: 'engine-mark--llama',
        markText: 'L',
        settings: true,
      },
      ik_llama: {
        title: 'ik_llama.cpp',
        markClass: 'engine-mark--ik',
        markText: 'IK',
        settings: true,
      },
      // Unsloth carries NO settings action at all — real divergence.
      unsloth_llama: {
        title: 'Unsloth llama.cpp',
        markClass: 'engine-mark--unsloth',
        markText: 'US',
        settings: false,
      },
    }
    const wrapper = mountEnginesView({ withHeader: true })
    await flushPromises()
    for (const [engineKey, expected] of Object.entries(CASES)) {
      wrapper.vm.openEngineModal(engineKey)
      await flushPromises()
      const header = wrapper.find('.eh-stub')
      expect(header.exists(), engineKey).toBe(true)
      expect(header.attributes('data-title'), engineKey).toBe(expected.title)
      const mark = header.find('.engine-mark')
      expect(mark.exists(), engineKey).toBe(true)
      expect(mark.classes(), engineKey).toContain(expected.markClass)
      expect(mark.text(), engineKey).toBe(expected.markText)
      // engine-mark spans, not pi icons (unlike the venv-family headers)
      expect(header.find('i').exists(), engineKey).toBe(false)
      expect(header.find('[aria-label="Build settings"]').exists(), engineKey).toBe(
        expected.settings,
      )
      // The native trio DO carry full aria (unlike the sglang trio)
      expect(header.find('[aria-label="Reload versions"]').exists(), engineKey).toBe(true)
      expect(header.find('[aria-label="Reload versions and status"]').exists(), engineKey).toBe(
        false,
      )
      expect(header.find('[aria-label="Rescan CLI parameters"]').exists(), engineKey).toBe(true)
      expect(header.text(), engineKey).toContain('Reload versions')
      expect(header.text(), engineKey).not.toContain('Reload versions and status')
      expect(header.text(), engineKey).toContain('Rescan CLI parameters')
    }
  })

  it('native dialog bodies render per-engine flows and preserve divergences', async () => {
    checkLlamaCppUpdates.mockResolvedValue({
      latest_release: {
        tag_name: 'b2000',
        html_url: 'https://github.com/ggml-org/llama.cpp/releases/tag/b2000',
      },
    })
    checkIkLlamaUpdates.mockResolvedValue({
      latest_commit: { sha: 'a'.repeat(40) },
    })
    checkUnslothLlamaUpdates.mockResolvedValue({
      update_available: true,
      latest_version: 'b7777',
      release_url: 'https://github.com/unslothai/llama.cpp/releases/tag/b7777',
    })
    const wrapper = mountEnginesView()
    await flushPromises()

    wrapper.vm.openEngineModal('llama_cpp')
    await flushPromises()
    let body = wrapper.find('.engine-modal-body')
    expect(body.exists()).toBe(true)
    expect(body.find('engine-build-settings-hint-stub').exists(), 'llama hint').toBe(true)
    expect(body.find('.build-note--info').exists(), 'llama no note').toBe(false)
    expect(body.find('[data-testid="audio-check-updates"]').attributes('data-hint')).toBe(
      'Compare the installed ref to the latest stable release (vX.Y.Z).',
    )
    await body.find('[data-testid="audio-check-updates"]').trigger('click')
    await flushPromises()
    body = wrapper.find('.engine-modal-body')
    let banner = body.findComponent(EngineUpdateBanner)
    expect(banner.props('latestVersion'), 'llama raw latest (no v prefix)').toBe('b2000')
    expect(banner.props('updateTooltip')).toBe('Update using saved build settings')
    expect(body.find('.engine-update-banner a').text()).toBe('View release')
    expect(body.find('[data-label="From release"]').exists()).toBe(true)
    expect(body.find('[data-label="From source"]').exists()).toBe(true)
    expect(body.find('[data-label="From tip"]').exists()).toBe(false)
    const table = body.findComponent(VersionTable)
    expect(table.props('canVerify'), 'llama can-verify').toBe(true)
    expect(table.props('emptyMessage')).toBe(
      'No versions yet. Install one using the options above.',
    )

    wrapper.vm.openEngineModal('ik_llama')
    await flushPromises()
    body = wrapper.find('.engine-modal-body')
    await body.find('[data-testid="audio-check-updates"]').trigger('click')
    await flushPromises()
    body = wrapper.find('.engine-modal-body')
    expect(body.find('engine-build-settings-hint-stub').exists(), 'ik hint').toBe(true)
    expect(body.find('[data-testid="audio-check-updates"]').attributes('data-hint')).toBe(
      'Compare the installed tip to the latest commit on main.',
    )
    banner = body.findComponent(EngineUpdateBanner)
    expect(banner.props('latestVersion')).toBe('a'.repeat(8))
    expect(banner.props('updateTooltip')).toBe('Rebuild tip of main using saved build settings')
    expect(body.find('.engine-update-banner a').text()).toBe('View commit')
    expect(body.find('[data-label="From tip"]').exists()).toBe(true)
    expect(body.find('[data-label="From source"]').exists()).toBe(true)
    expect(body.find('[data-label="From release"]').exists()).toBe(false)

    wrapper.vm.openEngineModal('unsloth_llama')
    await flushPromises()
    body = wrapper.find('.engine-modal-body')
    expect(body.exists()).toBe(true)
    expect(body.find('engine-build-settings-hint-stub').exists(), 'unsloth no hint').toBe(false)
    expect(body.find('.build-note--info').exists(), 'unsloth note').toBe(true)
    expect(body.find('.build-note--info').text()).toContain('not Unsloth Studio')
    expect(body.find('[data-testid="audio-check-updates"]').attributes('data-hint')).toBe(
      'Compare the active mix tag to the latest unslothai/llama.cpp prebuilt.',
    )
    await body.find('[data-testid="audio-check-updates"]').trigger('click')
    await flushPromises()
    body = wrapper.find('.engine-modal-body')
    banner = body.findComponent(EngineUpdateBanner)
    expect(banner.props('latestVersion')).toBe('b7777')
    expect(banner.props('updateTooltip')).toBe('Install the latest Unsloth llama-server mix')
    expect(body.find('[data-label="Install prebuilt"]').exists()).toBe(true)
    expect(body.find('[data-label="From source"]').exists()).toBe(false)
    expect(body.findComponent(VersionTable).props('canVerify'), 'unsloth no verify').toBe(false)
    expect(body.findComponent(VersionTable).props('emptyMessage')).toBe(
      'No versions yet. Install a prebuilt using the option above.',
    )
  })
})
