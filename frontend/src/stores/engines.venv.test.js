/**
 * Venv lifecycle parity matrix — one implementation per verb in the store,
 * exercised through the historical per-engine wrappers (unchanged names and
 * side effects: nullSafeStatus assignment, remove's status+versions refresh
 * ORDER). URLs are pinned as LITERALS in the registry tests below; the
 * store-side expectations here track whichever surface the descriptors
 * select (generic /api/engines/{id}/{verb} since the v2 migration).
 * Generic == legacy is NOT assumed — it is pinned server-side by
 * backend/tests/test_generic_engine_router.py.
 */

import { describe, it, expect, beforeEach, vi } from 'vitest'
import { setActivePinia, createPinia } from 'pinia'
import axios from 'axios'
import { useEnginesStore } from './engines.js'
import {
  CMAKE_ENGINE_IDS,
  ENGINE_DESCRIPTORS,
  ENGINE_IDS_BY_FALLBACK_FORMAT,
  ENGINE_LABELS,
  FALLBACK_ENGINE_OPTIONS,
  SG_PANEL_ENGINE_IDS,
  SGLANG_RUNTIME_UI_IDS,
  VENV_CANCEL_ENDPOINTS_BY_MANAGER,
  VENV_DESCRIPTORS,
  VENV_ENGINE_IDS,
  engineBuildHint,
  legacyVenvApiPath,
  NATIVE_CANCEL_ENDPOINTS_BY_ENGINE,
  NATIVE_GENERIC_VERBS,
  nativeGenericApiPath,
  venvApiPath,
  venvDescriptor,
} from '../engines/registry.js'

vi.mock('axios', () => ({
  default: {
    get: vi.fn(),
    post: vi.fn(),
    put: vi.fn(),
    delete: vi.fn(),
  },
}))

const SG_FAMILY = ['sglang', 'sglang_v100', 'sglang_sxm2', 'sglang_sm70', 'vllm']

// Per-engine entry points exactly as the store exports them today.
const CALLS = {
  lmdeploy: {
    check: (s) => s.checkLmdeployUpdates(),
    status: (s) => s.fetchLmdeployStatus(),
    getSettings: (s) => s.fetchLmdeployBuildSettings(),
    putSettings: (s, v) => s.saveLmdeployBuildSettings(v),
    install: (s, p) => s.installLmdeploy(p),
    installSource: (s, p) => s.installLmdeployFromSource(p),
    remove: (s, p) => s.removeLmdeploy(p),
  },
  '1cat_vllm': {
    check: (s) => s.checkOnecatVllmUpdates(),
    status: (s) => s.fetchOnecatVllmStatus(),
    getSettings: (s) => s.fetchOnecatVllmBuildSettings(),
    putSettings: (s, v) => s.saveOnecatVllmBuildSettings(v),
    install: (s, p) => s.installOnecatVllm(p),
    installSource: (s, p) => s.installOnecatVllmFromSource(p),
    remove: (s, p) => s.removeOnecatVllm(p),
  },
  sglang: sgFamilyCalls('sglang'),
  sglang_v100: sgFamilyCalls('sglang_v100'),
  sglang_sxm2: sgFamilyCalls('sglang_sxm2'),
  sglang_sm70: sgFamilyCalls('sglang_sm70'),
  vllm: sgFamilyCalls('vllm'),
}

function sgFamilyCalls(engine) {
  return {
    check: (s) => s.checkSglangUpdates(engine),
    status: (s) => s.fetchSglangStatus(engine),
    getSettings: (s) => s.fetchSglangBuildSettings(engine),
    putSettings: (s, v) => s.saveSglangBuildSettings(engine, v),
    install: (s, p) => s.installSglang(engine, p),
    installSource: (s, p) => s.installSglangFromSource(engine, p),
    // No store-level cancel: task cancellation routes through
    // useTaskCancelEndpoint -> VENV_CANCEL_ENDPOINTS_BY_MANAGER (pinned
    // below and exercised by useTaskActions/TaskNotifications suites).
  }
}

describe('engines/registry', () => {
  it('routes genericApi engines through the generic router, by literal', () => {
    expect([...VENV_ENGINE_IDS].sort()).toEqual(
      [
        '1cat_vllm',
        'lmdeploy',
        'sglang',
        'sglang_v100',
        'sglang_sxm2',
        'sglang_sm70',
        'vllm',
      ].sort(),
    )
    expect(Object.keys(VENV_DESCRIPTORS).sort()).toEqual([...VENV_ENGINE_IDS].sort())
    expect(venvApiPath('sglang_v100', 'status')).toBe('/api/engines/sglang_v100/status')
    expect(venvApiPath('1cat_vllm', 'install')).toBe('/api/engines/1cat_vllm/install')
    expect(venvApiPath('lmdeploy', 'check-updates')).toBe('/api/engines/lmdeploy/check-updates')
    // All five flipped onto the generic surface is the deliberate state.
    for (const engineId of VENV_ENGINE_IDS) {
      expect(venvDescriptor(engineId).genericApi).toBe(true)
    }
  })

  it('mirrors the backend native verb capabilities', () => {
    expect(NATIVE_GENERIC_VERBS).toEqual({
      llama_cpp: ['check-updates', 'build-settings', 'install', 'install-source', 'cancel'],
      ik_llama: ['check-updates', 'build-settings', 'install', 'install-source', 'cancel'],
      unsloth_llama: ['check-updates', 'install', 'cancel'],
    })
    expect(nativeGenericApiPath('ik_llama', 'check-updates')).toBe(
      '/api/engines/ik_llama/check-updates',
    )
    // Verbs the backend answers 405 for never get a URL built client-side.
    expect(() => nativeGenericApiPath('unsloth_llama', 'build-settings')).toThrow(
      'Generic API not available for engine: unsloth_llama/build-settings',
    )
    expect(() => nativeGenericApiPath('unsloth_llama', 'status')).toThrow(
      'Generic API not available for engine: unsloth_llama/status',
    )
    expect(() => nativeGenericApiPath('audio_cpp', 'check-updates')).toThrow(
      'Generic API not available for engine: audio_cpp/check-updates',
    )
  })

  it('keeps the legacy prefix segments as the rollback path', () => {
    expect(legacyVenvApiPath('sglang_v100', 'status')).toBe('/api/sglang-v100/status')
    expect(legacyVenvApiPath('1cat_vllm', 'install')).toBe('/api/1cat-vllm/install')
    expect(legacyVenvApiPath('lmdeploy', 'check-updates')).toBe('/api/lmdeploy/check-updates')
    expect(legacyVenvApiPath('vllm', 'status')).toBe('/api/vllm/status')
    expect(legacyVenvApiPath('sglang', 'cancel')).toBe('/api/sglang/cancel')
  })

  it('maps task-cancel managers to the generic cancel route', () => {
    expect(VENV_CANCEL_ENDPOINTS_BY_MANAGER).toEqual({
      lmdeploy: '/api/engines/lmdeploy/cancel',
      vllm: '/api/engines/vllm/cancel',
      sglang: '/api/engines/sglang/cancel',
      sglang_v100: '/api/engines/sglang_v100/cancel',
      sglang_sxm2: '/api/engines/sglang_sxm2/cancel',
      sglang_sm70: '/api/engines/sglang_sm70/cancel',
      // manager id differs from the engine id for 1Cat-vLLM
      onecat_vllm: '/api/engines/1cat_vllm/cancel',
    })
  })

  it('maps native task-cancel engines to the generic cancel route', () => {
    expect(NATIVE_CANCEL_ENDPOINTS_BY_ENGINE).toEqual({
      llama_cpp: '/api/engines/llama_cpp/cancel',
      ik_llama: '/api/engines/ik_llama/cancel',
      unsloth_llama: '/api/engines/unsloth_llama/cancel',
    })
  })

  it('v3: one descriptor per engine replaces the six parallel records', () => {
    expect(Object.keys(ENGINE_DESCRIPTORS)).toHaveLength(11)
    // ENGINE_LABELS (engineVersionDelete)
    expect(ENGINE_LABELS).toEqual({
      llama_cpp: 'llama.cpp',
      ik_llama: 'ik_llama.cpp',
      unsloth_llama: 'Unsloth llama.cpp',
      lmdeploy: 'LMDeploy',
      '1cat_vllm': '1Cat-vLLM',
      sglang: 'SGLang',
      sglang_v100: 'SGLang V100',
      sglang_sxm2: 'SGLang SXM2',
      sglang_sm70: 'SGLang SM70',
      vllm: 'vLLM',
      audio_cpp: 'audio.cpp',
    })
    // fallbackEngineOptions (ModelConfig) — picker order included
    expect(FALLBACK_ENGINE_OPTIONS.map((o) => o.value)).toEqual([
      'llama_cpp',
      'ik_llama',
      'unsloth_llama',
      'lmdeploy',
      '1cat_vllm',
      'sglang',
      'sglang_v100',
      'sglang_sxm2',
      'sglang_sm70',
      'vllm',
      'audio_cpp',
    ])
    expect(FALLBACK_ENGINE_OPTIONS.find((o) => o.value === 'sglang_v100')).toEqual({
      value: 'sglang_v100',
      label: 'SGLang V100',
      icon: 'pi-sparkles',
    })
    // inferredEnginesByFormat (ModelConfig) — feeds a Set, order-free
    expect([...ENGINE_IDS_BY_FALLBACK_FORMAT.gguf].sort()).toEqual(
      ['llama_cpp', 'ik_llama', 'unsloth_llama'].sort(),
    )
    expect([...ENGINE_IDS_BY_FALLBACK_FORMAT.safetensors].sort()).toEqual(
      [
        'lmdeploy',
        '1cat_vllm',
        'vllm',
        'sglang',
        'sglang_v100',
        'sglang_sxm2',
        'sglang_sm70',
      ].sort(),
    )
    // CMAKE_ENGINES (VersionTable)
    expect([...CMAKE_ENGINE_IDS].sort()).toEqual(['audio_cpp', 'ik_llama', 'llama_cpp'].sort())
    // repoSource — the repository_source display names (second identity
    // system) now owned by the registry, incl. the llama.cpp untagged-row
    // fallback that partitions GET /api/llama-versions
    expect(
      Object.fromEntries(Object.entries(ENGINE_DESCRIPTORS).map(([id, d]) => [id, d.repoSource])),
    ).toEqual({
      llama_cpp: 'llama.cpp',
      ik_llama: 'ik_llama.cpp',
      unsloth_llama: 'Unsloth llama.cpp',
      lmdeploy: 'LMDeploy',
      '1cat_vllm': '1Cat-vLLM',
      sglang: 'SGLang',
      sglang_v100: 'SGLang-V100',
      sglang_sxm2: 'SGLang-SXM2',
      sglang_sm70: 'SGLang-SM70',
      vllm: 'vLLM',
      audio_cpp: 'audio.cpp',
    })
    expect(ENGINE_DESCRIPTORS.llama_cpp.untaggedSourceOwner).toBe(true)
    // venv dialog composition fields — incl. 1Cat's header icon diverging
    // from its picker icon (real per-surface difference)
    expect(ENGINE_DESCRIPTORS.lmdeploy.versionsKey).toBe('lmdeployVersions')
    expect(ENGINE_DESCRIPTORS.lmdeploy.dialogIcon).toBe('pi-server')
    expect(ENGINE_DESCRIPTORS['1cat_vllm'].versionsKey).toBe('onecatVllmVersions')
    expect(ENGINE_DESCRIPTORS['1cat_vllm'].dialogIcon).toBe('pi-bolt')
    expect(ENGINE_DESCRIPTORS['1cat_vllm'].icon).toBe('pi-server')
    // Dialog-header icons of the panel trio (vllm's differs from its
    // picker icon; V100 reuses the bolt)
    expect(ENGINE_DESCRIPTORS.sglang.dialogIcon).toBe('pi-sparkles')
    expect(ENGINE_DESCRIPTORS.sglang_v100.dialogIcon).toBe('pi-bolt')
    expect(ENGINE_DESCRIPTORS.vllm.dialogIcon).toBe('pi-sparkles')
    expect(ENGINE_DESCRIPTORS.vllm.icon).toBe('pi-server')
    // Two DISTINCT sglang families — the v3 trap deliberately pinned:
    expect([...SGLANG_RUNTIME_UI_IDS].sort()).toEqual(
      ['sglang', 'sglang_v100', 'sglang_sxm2', 'sglang_sm70'].sort(),
    )
    expect([...SG_PANEL_ENGINE_IDS].sort()).toEqual(
      ['sglang', 'sglang_v100', 'sglang_sxm2', 'sglang_sm70', 'vllm'].sort(),
    )
    // EngineBuildSettingsHint copy, incl. the unsloth no-hint hole
    expect(engineBuildHint('vllm')).toContain('CUDA toolkit')
    expect(engineBuildHint('sglang_v100')).toContain('SM70')
    expect(engineBuildHint('unsloth_llama')).toBe('')
    expect(engineBuildHint('nope')).toBe('')
  })

  it('marks the null-safe status families the survey found', () => {
    for (const engine of SG_FAMILY) {
      expect(venvDescriptor(engine).nullSafeStatus).toBe(true)
    }
    expect(venvDescriptor('lmdeploy').nullSafeStatus).toBeUndefined()
    expect(venvDescriptor('1cat_vllm').nullSafeStatus).toBeUndefined()
  })

  it('rejects non-venv engines', () => {
    for (const id of ['llama_cpp', 'ik_llama', 'audio_cpp', 'unsloth_llama', 'nope']) {
      expect(() => venvDescriptor(id)).toThrow(`Unknown venv engine: ${id}`)
    }
  })
})

describe('version status refresh fan-out (activate/delete)', () => {
  // Pins the EXACT ordered refresh sequences produced by the hand-rolled
  // versionId substring chains in activateVersion/deleteVersion — including
  // the legacy quirk that 1Cat ids contain "vllm" and therefore also
  // refresh vLLM status, and that audio refreshes on activate only.
  const STATUS_URLS = [
    '/api/engines/lmdeploy/status',
    '/api/engines/1cat_vllm/status',
    '/api/engines/sglang_v100/status',
    '/api/engines/sglang_sxm2/status',
    '/api/engines/sglang_sm70/status',
    '/api/engines/sglang/status',
    '/api/engines/vllm/status',
    '/api/audio-cpp/status',
  ]

  function mockGraph() {
    const statusCalls = []
    setActivePinia(createPinia())
    axios.post.mockResolvedValue({ data: { ok: true } })
    axios.delete.mockResolvedValue({ data: { ok: true } })
    axios.get.mockImplementation(async (url) => {
      if (STATUS_URLS.includes(url)) statusCalls.push(url)
      return { data: url === '/api/llama-versions' ? [] : {} }
    })
    return statusCalls
  }

  const CASES = [
    ['lmdeploy-v1', ['/api/engines/lmdeploy/status'], ['/api/engines/lmdeploy/status']],
    [
      '1cat_vllm:v1',
      // The quirk: 1Cat status first, then the vllm branch of the exclusive
      // chain also matches the same id.
      ['/api/engines/1cat_vllm/status', '/api/engines/vllm/status'],
      ['/api/engines/1cat_vllm/status', '/api/engines/vllm/status'],
    ],
    [
      'sglang_v100:2026-01-02',
      ['/api/engines/sglang_v100/status'],
      ['/api/engines/sglang_v100/status'],
    ],
    [
      'sglang_sxm2:2026-01-02',
      ['/api/engines/sglang_sxm2/status'],
      ['/api/engines/sglang_sxm2/status'],
    ],
    [
      'sglang_sm70:2026-01-02',
      ['/api/engines/sglang_sm70/status'],
      ['/api/engines/sglang_sm70/status'],
    ],
    ['sglang:abc', ['/api/engines/sglang/status'], ['/api/engines/sglang/status']],
    ['vllm:0.17.0', ['/api/engines/vllm/status'], ['/api/engines/vllm/status']],
    ['llama_cpp:b7600', [], []],
    ['audio_cpp:source-main', ['/api/audio-cpp/status'], []],
  ]

  for (const [versionId, activateExpected, deleteExpected] of CASES) {
    it(`activate ${versionId} refreshes exactly the pinned statuses`, async () => {
      const calls = mockGraph()
      await useEnginesStore().activateVersion(versionId)
      expect(calls).toEqual(activateExpected)
    })
    it(`delete ${versionId} refreshes exactly the pinned statuses`, async () => {
      const calls = mockGraph()
      await useEnginesStore().deleteVersion(versionId)
      expect(calls).toEqual(deleteExpected)
    })
  }
})

describe('venv store lifecycle parity', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    for (const fn of [axios.get, axios.post, axios.put]) fn.mockReset()
  })

  function mockOk(data) {
    for (const fn of [axios.get, axios.post, axios.put]) fn.mockResolvedValue({ data })
  }

  for (const engine of [...VENV_ENGINE_IDS].sort()) {
    it(`${engine}: check-updates hits the legacy prefix`, async () => {
      mockOk({ latest_version: '1.2.3' })
      const store = useEnginesStore()
      await expect(CALLS[engine].check(store)).resolves.toEqual({ latest_version: '1.2.3' })
      expect(axios.get).toHaveBeenCalledWith(venvApiPath(engine, 'check-updates'))
    })

    it(`${engine}: status fills its keyed map entry`, async () => {
      mockOk({ installed: true })
      const store = useEnginesStore()
      await CALLS[engine].status(store)
      expect(axios.get).toHaveBeenCalledWith(venvApiPath(engine, 'status'))
      expect(store.engineStatusById[engine]).toEqual({ installed: true })
    })

    it(`${engine}: build-settings round trip`, async () => {
      mockOk({ source_branch: 'dev' })
      const store = useEnginesStore()
      await CALLS[engine].getSettings(store)
      expect(axios.get).toHaveBeenCalledWith(venvApiPath(engine, 'build-settings'))
      await CALLS[engine].putSettings(store, { source_branch: 'dev' })
      expect(axios.put).toHaveBeenCalledWith(venvApiPath(engine, 'build-settings'), {
        source_branch: 'dev',
      })
    })

    it(`${engine}: install and install-source forward payloads`, async () => {
      mockOk({ task_id: 't-1' })
      const store = useEnginesStore()
      await CALLS[engine].install(store, { version: '9.9.9' })
      expect(axios.post).toHaveBeenCalledWith(venvApiPath(engine, 'install'), { version: '9.9.9' })
      await CALLS[engine].installSource(store, { repo_url: 'https://x/y', branch: 'b' })
      expect(axios.post).toHaveBeenCalledWith(venvApiPath(engine, 'install-source'), {
        repo_url: 'https://x/y',
        branch: 'b',
      })
    })

    if (CALLS[engine].remove) {
      it(`${engine}: remove refreshes status then version rows`, async () => {
        const calls = []
        axios.post.mockImplementation(async (url) => {
          calls.push(`post ${url}`)
          return { data: {} }
        })
        axios.get.mockImplementation(async (url) => {
          calls.push(`get ${url}`)
          return { data: [] }
        })
        const store = useEnginesStore()
        await CALLS[engine].remove(store, {})
        expect(calls).toEqual([
          `post ${venvApiPath(engine, 'remove')}`,
          `get ${venvApiPath(engine, 'status')}`,
          'get /api/llama-versions',
        ])
      })
    }
  }

  it('preserves the status null-handling split', async () => {
    // SGLang-family producers coerce a nullish payload to {} for the keyed
    // map AND return; lmdeploy/1Cat assign the raw payload. (matrix §6 preserve).
    for (const fn of [axios.get, axios.post, axios.put]) fn.mockResolvedValue({ data: undefined })
    const store = useEnginesStore()
    await store.fetchSglangStatus('sglang')
    expect(store.engineStatusById.sglang).toEqual({})
    const out = await store.fetchLmdeployStatus()
    expect(out).toBeUndefined()
    expect(store.engineStatusById.lmdeploy).toBeUndefined()
  })
})

describe('native read verbs on the generic router', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    for (const fn of [axios.get, axios.post, axios.put]) fn.mockReset()
  })

  function mockOk(data) {
    for (const fn of [axios.get, axios.post, axios.put]) fn.mockResolvedValue({ data })
  }

  it('routes the check-updates trio through the generic router', async () => {
    mockOk({ latest_version: '1' })
    const store = useEnginesStore()
    await expect(store.checkLlamaCppUpdates()).resolves.toEqual({ latest_version: '1' })
    expect(axios.get).toHaveBeenLastCalledWith('/api/engines/llama_cpp/check-updates')
    await store.checkIkLlamaUpdates()
    // The ?source=ik_llama query dies: the engine travels in the path.
    expect(axios.get).toHaveBeenLastCalledWith('/api/engines/ik_llama/check-updates')
    await store.checkUnslothLlamaUpdates()
    expect(axios.get).toHaveBeenLastCalledWith('/api/engines/unsloth_llama/check-updates')
  })

  it('routes llama-family build-settings through the generic router', async () => {
    mockOk({ build_type: 'Release' })
    const store = useEnginesStore()
    await store.fetchBuildSettings('ik_llama')
    expect(axios.get).toHaveBeenLastCalledWith('/api/engines/ik_llama/build-settings')
    await store.saveBuildSettings('llama_cpp', { cuda: true })
    expect(axios.put).toHaveBeenLastCalledWith('/api/engines/llama_cpp/build-settings', {
      cuda: true,
    })
  })

  it('keeps the fallback for engines without a generic build-settings verb', async () => {
    // Callers only ever pass llama_cpp/ik_llama today; any other engine id
    // must still ride the legacy engine-parameter route unchanged.
    mockOk({})
    const store = useEnginesStore()
    await store.fetchBuildSettings('audio_cpp')
    expect(axios.get).toHaveBeenLastCalledWith('/api/llama-versions/build-settings', {
      params: { engine: 'audio_cpp' },
    })
    await store.saveBuildSettings('audio_cpp', { x: 1 })
    expect(axios.put).toHaveBeenLastCalledWith(
      '/api/llama-versions/build-settings',
      { x: 1 },
      {
        params: { engine: 'audio_cpp' },
      },
    )
  })

  it('write verbs ride the generic router (deliberate flip)', async () => {
    axios.post.mockResolvedValue({ data: { task_id: 't' } })
    axios.get.mockResolvedValue({ data: [] })
    const store = useEnginesStore()
    await store.updateEngine('ik_llama')
    expect(axios.post).toHaveBeenLastCalledWith('/api/engines/ik_llama/install', {
      engine: 'ik_llama',
    })
    expect(axios.get).toHaveBeenLastCalledWith('/api/llama-versions') // refresh unchanged
    await store.buildSource({ commit_sha: 'x', repository_source: 'llama.cpp' })
    expect(axios.post).toHaveBeenLastCalledWith('/api/engines/llama_cpp/install-source', {
      commit_sha: 'x',
      repository_source: 'llama.cpp',
    })
    await store.buildSource({ commit_sha: 'y', repository_source: 'ik_llama.cpp' })
    expect(axios.post).toHaveBeenLastCalledWith('/api/engines/ik_llama/install-source', {
      commit_sha: 'y',
      repository_source: 'ik_llama.cpp',
    })
    // Absent repository_source keeps building llama.cpp (legacy quirk),
    // byte-identically: same body reaches the same build_source function.
    await store.buildSource({ commit_sha: 'z' })
    expect(axios.post).toHaveBeenLastCalledWith('/api/engines/llama_cpp/install-source', {
      commit_sha: 'z',
    })
    await store.cancelSourceBuild('t1')
    expect(axios.post).toHaveBeenLastCalledWith('/api/engines/llama_cpp/cancel', { task_id: 't1' })
    await store.installUnslothLlama({ tag_name: 'b1' })
    expect(axios.post).toHaveBeenLastCalledWith('/api/engines/unsloth_llama/install', {
      tag_name: 'b1',
    })
    await store.cancelUnslothLlama('t2')
    expect(axios.post).toHaveBeenLastCalledWith('/api/engines/unsloth_llama/cancel', {
      task_id: 't2',
    })
  })
})
