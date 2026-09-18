import { beforeEach, describe, expect, it, vi } from 'vitest'
import { mount } from '@vue/test-utils'

const { confirmRequire, store, toastAdd } = vi.hoisted(() => ({
  confirmRequire: vi.fn(),
  toastAdd: vi.fn(),
  store: {
    sglangVersions: [],
    sglangV100Versions: [],
    vllmVersions: [],
    sglangStatus: {},
    sglangV100Status: {},
    vllmStatus: {},
    fetchLlamaVersions: vi.fn().mockResolvedValue([]),
    fetchSglangStatus: vi.fn().mockResolvedValue({}),
    checkSglangUpdates: vi.fn().mockResolvedValue({}),
    retryVersion: vi.fn().mockResolvedValue({}),
    deleteVersion: vi.fn().mockResolvedValue({}),
  },
}))

vi.mock('@/stores/engines', () => ({
  useEnginesStore: () => store,
}))

vi.mock('primevue/useconfirm', () => ({
  useConfirm: () => ({ require: confirmRequire }),
}))

vi.mock('primevue/usetoast', () => ({
  useToast: () => ({ add: toastAdd }),
}))

import SglangEnginePanel from './SglangEnginePanel.vue'

function mountPanel() {
  return mount(SglangEnginePanel, {
    props: { engineId: 'sglang_v100' },
    global: {
      directives: { tooltip: () => {} },
      stubs: {
        Button: true,
        Dialog: true,
        InputText: true,
        EngineActiveStatus: true,
        EngineBuildSettingsHint: true,
        EngineCheckUpdatesCta: true,
        EngineInstallPanel: { template: '<div><slot /></div>' },
        EngineNote: true,
        EngineUpdateBanner: true,
        EngineVersionsBlock: { template: '<div><slot /></div>' },
        VersionTable: {
          props: ['versions'],
          emits: ['retry', 'delete'],
          template: `
            <div>
              <button data-testid="retry" @click="$emit('retry', versions[0].id ?? versions[0].version)">Retry</button>
              <button data-testid="delete" @click="$emit('delete', versions[0].id ?? versions[0].version)">Delete</button>
            </div>
          `,
        },
      },
    },
  })
}

describe('SglangEnginePanel version actions', () => {
  beforeEach(() => {
    confirmRequire.mockReset()
    toastAdd.mockReset()
    store.retryVersion.mockClear()
    store.deleteVersion.mockClear()
    store.sglangV100Versions = [{
      id: 'sglang_v100:20260918-165120-source',
      version: '20260918-165120-source',
      build_status: 'failed',
      retryable: true,
    }]
  })

  it('passes the emitted failed-version id through retry and delete', async () => {
    const wrapper = mountPanel()

    await wrapper.get('[data-testid="retry"]').trigger('click')
    expect(store.retryVersion).toHaveBeenCalledWith('sglang_v100:20260918-165120-source')

    await wrapper.get('[data-testid="delete"]').trigger('click')
    expect(confirmRequire).toHaveBeenCalledOnce()
    expect(confirmRequire.mock.calls[0][0].message).toContain('20260918-165120-source')

    await confirmRequire.mock.calls[0][0].accept()
    expect(store.deleteVersion).toHaveBeenCalledWith('sglang_v100:20260918-165120-source')
  })

  it('falls back to the version name when a failed row has no id', async () => {
    store.sglangV100Versions = [{
      version: '20260918-165120-source',
      build_status: 'failed',
      retryable: true,
    }]
    const wrapper = mountPanel()

    await wrapper.get('[data-testid="retry"]').trigger('click')
    expect(store.retryVersion).toHaveBeenCalledWith('20260918-165120-source')

    await wrapper.get('[data-testid="delete"]').trigger('click')
    await confirmRequire.mock.calls[0][0].accept()
    expect(store.deleteVersion).toHaveBeenCalledWith('20260918-165120-source')
  })
})
