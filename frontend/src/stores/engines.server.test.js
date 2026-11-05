/**
 * YAML campaign S5 — server-backed engine overlay (descriptor-driven UI).
 *
 * Proves a YAML-only engine (fixture id 'yamlcraft' — a TEST-ONLY string;
 * the G5 guard bans it from app code) behaves end-to-end in the client
 * from the live GET /api/engines payload ALONE: identity labels, generic
 * polling, version rows, verbs, and task-cancel routing — with zero
 * entries in the local engine table.
 */
import { describe, it, expect, beforeEach, vi } from 'vitest'
import { setActivePinia, createPinia } from 'pinia'
import axios from 'axios'
import { useEnginesStore } from './engines.js'
import {
  applyServerDescriptors,
  ENGINE_DESCRIPTORS,
  engineLabel,
  serverBackedDescriptor,
  serverBackedEngines,
} from '../engines/registry.js'
import { engineLabel as deleteEngineLabel } from '../composables/engineVersionDelete.js'
import { cancelEndpointForTask } from '../composables/useTaskCancelEndpoint.js'

vi.mock('axios', () => ({
  default: {
    get: vi.fn(),
    post: vi.fn(),
    put: vi.fn(),
  },
}))

// Shape mirrors the LIVE backend payload (backend/routes/engines.py):
// generic_lifecycle + generic_verbs are added for YAML-onboarded ids only.
const LIVE_DESCRIPTORS = [
  {
    id: 'lmdeploy',
    label: 'LMDeploy',
    repository_source: 'LMDeploy',
    install_kind: 'python_venv',
    artifact_formats: ['safetensors'],
    runnable: true,
    installed_versions: 2,
  },
  {
    id: 'yamlcraft',
    label: 'Crafty',
    repository_source: 'Crafty Runtime',
    install_kind: 'python_venv',
    artifact_formats: ['fixture_only'],
    runnable: false,
    installed_versions: 1,
    generic_lifecycle: true,
    generic_verbs: [
      'check-updates',
      'status',
      'build-settings',
      'install',
      'install-source',
      'remove',
      'cancel',
    ],
  },
]

describe('engines/registry — server-backed overlay', () => {
  it('never adds entries to the local table (nine stay nine)', () => {
    expect(Object.keys(ENGINE_DESCRIPTORS)).toHaveLength(9)
    expect(ENGINE_DESCRIPTORS.yamlcraft).toBeUndefined()
  })

  it('synthesizes a store descriptor from the payload alone', () => {
    const descriptor = serverBackedDescriptor('yamlcraft', LIVE_DESCRIPTORS)
    expect(descriptor).toMatchObject({
      label: 'Crafty',
      repoSource: 'Crafty Runtime',
      icon: 'pi-server',
      venv: true,
      genericApi: true,
      managerKey: 'yamlcraft',
      serverBacked: true,
    })
    // R81/R82: no per-engine statusRef — statuses live in one keyed map.
    expect(descriptor.statusRef).toBeUndefined()
    expect(descriptor.versionsKey).toBe('yamlcraft')
    expect(descriptor.genericVerbs).toContain('install')
    // Generated engines are NOT server-backed: legacy paths untouched.
    expect(serverBackedDescriptor('lmdeploy', LIVE_DESCRIPTORS)).toBeNull()
    expect(serverBackedDescriptor('nope', LIVE_DESCRIPTORS)).toBeNull()
    expect(serverBackedEngines(undefined)).toEqual([])
  })

  it('server descriptors win engine labels; table is the fallback', () => {
    applyServerDescriptors(LIVE_DESCRIPTORS)
    try {
      expect(engineLabel('yamlcraft')).toBe('Crafty')
      expect(deleteEngineLabel('yamlcraft')).toBe('Crafty')
      // The nine are byte-stable through either source (identity pins).
      expect(engineLabel('lmdeploy')).toBe('LMDeploy')
      expect(deleteEngineLabel('ik_llama')).toBe('ik_llama.cpp')
    } finally {
      applyServerDescriptors([])
    }
    expect(engineLabel('yamlcraft')).toBe('yamlcraft') // offline fallback
  })

  it('routes task cancel for server-backed managers via the generic router', () => {
    applyServerDescriptors(LIVE_DESCRIPTORS)
    try {
      expect(cancelEndpointForTask({ task_id: 't1', metadata: { manager: 'yamlcraft' } })).toBe(
        '/api/engines/yamlcraft/cancel',
      )
      expect(
        cancelEndpointForTask({ task_id: 't2', type: 'build', metadata: { engine: 'yamlcraft' } }),
      ).toBe('/api/engines/yamlcraft/cancel')
      // Unknown ids still resolve to nothing (no fabricated URLs).
      expect(cancelEndpointForTask({ task_id: 't3', metadata: { manager: 'ghost' } })).toBeNull()
    } finally {
      applyServerDescriptors([])
    }
  })
})

describe('engines store — server-backed engine flows with zero local entries', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    vi.mocked(axios.get).mockReset()
    vi.mocked(axios.post).mockReset()
    vi.mocked(axios.put).mockReset()
  })

  it('fetchEngineDescriptors refreshes the label overlay', async () => {
    vi.mocked(axios.get).mockImplementation(async (url) =>
      url === '/api/engines'
        ? { data: { schema_version: 1, engines: LIVE_DESCRIPTORS } }
        : { data: {} },
    )
    const store = useEnginesStore()
    await store.fetchEngineDescriptors()
    try {
      expect(engineLabel('yamlcraft')).toBe('Crafty')
    } finally {
      applyServerDescriptors([])
    }
  })

  it('polls the YAML engine status generically and keeps it out of the named refs', async () => {
    vi.mocked(axios.get).mockImplementation(async (url) => {
      if (url === '/api/engines') {
        return { data: { schema_version: 1, engines: LIVE_DESCRIPTORS } }
      }
      if (url === '/api/engines/yamlcraft/status') {
        return { data: { installed: true, version: '9.9.0' } }
      }
      if (url === '/api/llama-versions') return { data: [] }
      return { data: {} }
    })
    const store = useEnginesStore()
    await store.fetchEngineDescriptors()
    try {
      await store.fetchServerBackedStatuses()
      expect(axios.get).toHaveBeenCalledWith('/api/engines/yamlcraft/status')
      expect(store.serverStatuses.yamlcraft).toEqual({ installed: true, version: '9.9.0' })
      // Generated engines were NOT touched by the generic poll.
      expect(store.statusForEngine('lmdeploy')).toEqual({})
    } finally {
      applyServerDescriptors([])
    }
  })

  it('partitions version rows for the YAML engine by server repository_source', async () => {
    vi.mocked(axios.get).mockImplementation(async (url) => {
      if (url === '/api/engines') {
        return { data: { schema_version: 1, engines: LIVE_DESCRIPTORS } }
      }
      if (url === '/api/llama-versions') {
        return {
          data: [
            { id: 'lm-a', repository_source: 'LMDeploy', version: '1' },
            { id: 'craft-a', repository_source: 'Crafty Runtime', version: '9.9.0' },
          ],
        }
      }
      return { data: {} }
    })
    const store = useEnginesStore()
    await store.fetchEngineDescriptors()
    try {
      await store.fetchLlamaVersions()
      expect(store.serverVersions.yamlcraft).toEqual([
        { id: 'craft-a', repository_source: 'Crafty Runtime', version: '9.9.0' },
      ])
      expect(store.lmdeployVersions).toHaveLength(1) // nine paths untouched
    } finally {
      applyServerDescriptors([])
    }
  })

  it('drives install/build-settings/remove through the generic router for YAML engines', async () => {
    vi.mocked(axios.get).mockImplementation(async (url) => {
      if (url === '/api/engines') {
        return { data: { schema_version: 1, engines: LIVE_DESCRIPTORS } }
      }
      return { data: {} }
    })
    vi.mocked(axios.post).mockResolvedValue({ data: { message: 'Crafty installation started' } })
    vi.mocked(axios.put).mockResolvedValue({ data: {} })

    const store = useEnginesStore()
    await store.fetchEngineDescriptors()
    try {
      await store.venvInstall('yamlcraft', { version: '9.9.0' })
      expect(axios.post).toHaveBeenCalledWith('/api/engines/yamlcraft/install', {
        version: '9.9.0',
      })
      await store.venvGetBuildSettings('yamlcraft')
      expect(axios.get).toHaveBeenCalledWith('/api/engines/yamlcraft/build-settings')
      await store.venvSaveBuildSettings('yamlcraft', { pip_version: '9.9.0' })
      expect(axios.put).toHaveBeenCalledWith('/api/engines/yamlcraft/build-settings', {
        pip_version: '9.9.0',
      })
      // Unknown NON-server engine STILL throws (no fabricated routing).
      await expect(store.venvInstall('nope', {})).rejects.toThrow('Unknown venv engine: nope')
    } finally {
      applyServerDescriptors([])
    }
  })
})
