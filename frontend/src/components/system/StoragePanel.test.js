import { beforeEach, describe, expect, it, vi } from 'vitest'
import { flushPromises, mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'

const { fetchStorageInventory, previewStorageCleanup, applyStorageCleanup } = vi.hoisted(() => ({
  fetchStorageInventory: vi.fn(),
  previewStorageCleanup: vi.fn(),
  applyStorageCleanup: vi.fn(),
}))

vi.mock('@/api/storage', () => ({
  fetchStorageInventory,
  previewStorageCleanup,
  applyStorageCleanup,
}))

import StoragePanel from './StoragePanel.vue'

function unit(overrides = {}) {
  return {
    id: 'unit-1',
    label: 'voice-models/orphan',
    kind: 'model_payload',
    bytes: 4096,
    cleanable: true,
    blocked: null,
    manage_path: null,
    outside_data_root: false,
    detail: {},
    ...overrides,
  }
}

function category(overrides = {}) {
  return {
    id: 'models',
    label: 'Model files',
    description: 'Payload directories in the model store.',
    bytes: 8192,
    unmanaged_bytes: 0,
    unit_count: 2,
    cleanable: true,
    cleanable_bytes: 4096,
    blocked: null,
    manage_path: '/models',
    outside_data_root: false,
    available: true,
    truncated: false,
    notes: [],
    units: [
      unit(),
      unit({
        id: 'unit-2',
        label: 'voice-models/keep',
        bytes: 4096,
        cleanable: false,
        blocked: {
          code: 'CATALOG_MODEL',
          message: 'A catalog model uses this directory. Remove the model from Models.',
        },
        manage_path: '/models',
      }),
    ],
    ...overrides,
  }
}

function inventoryFixture(categories = [category()]) {
  return {
    schema_version: 1,
    generated_at: '2026-11-22T20:00:00+00:00',
    cached: false,
    filesystem: {
      data_root: '/srv/studio/data',
      total_bytes: 1000,
      used_bytes: 400,
      free_bytes: 600,
    },
    scan: {
      quality: 'complete',
      cached: false,
      entries_scanned: 12,
      max_scan_entries: 100000,
      max_units_per_category: 200,
      truncated: false,
      category_count: categories.length,
      notes: [],
    },
    categories,
  }
}

function mountPanel() {
  return mount(StoragePanel, {
    global: {
      stubs: {
        RouterLink: {
          props: ['to'],
          template: '<a :href="to"><slot /></a>',
        },
        LoadingState: true,
      },
    },
  })
}

describe('StoragePanel', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    vi.clearAllMocks()
  })

  it('renders measured categories, blocked reasons, manage links and unit disclosure', async () => {
    fetchStorageInventory.mockResolvedValueOnce(
      inventoryFixture([
        category(),
        category({
          id: 'engine_installs',
          label: 'Engine installs',
          description: 'Versioned engine installations.',
          bytes: 2048,
          unit_count: 1,
          cleanable: false,
          cleanable_bytes: 0,
          blocked: {
            code: 'REPORT_ONLY',
            message: 'This storage is managed by another Studio surface.',
          },
          manage_path: '/engines',
          units: [
            unit({
              id: 'engine-1',
              label: 'native/demo',
              kind: 'engine_install',
              cleanable: false,
              blocked: {
                code: 'REPORT_ONLY',
                message: 'This storage is managed by another Studio surface.',
              },
              manage_path: '/engines',
            }),
          ],
        }),
      ]),
    )
    const wrapper = mountPanel()
    await flushPromises()

    expect(wrapper.get('#storage-heading').text()).toBe('Storage')
    expect(wrapper.get('.storage-path').text()).toBe('/srv/studio/data')
    expect(wrapper.text()).toContain('Model files')
    expect(wrapper.text()).toContain('Engine installs')
    expect(wrapper.text()).toContain('4.0 KiB reclaimable')
    expect(wrapper.text()).toContain('A catalog model uses this directory')
    expect(wrapper.text()).toContain('This storage is managed by another Studio surface.')
    expect(wrapper.text()).toContain('Report only')
    const links = wrapper.findAll('a').map((link) => link.attributes('href'))
    expect(links).toContain('/models')
    expect(links).toContain('/engines')
    expect(wrapper.text()).toContain('Show 2 units for Model files')
    expect(wrapper.get('.storage-share').attributes('aria-label')).toContain('Model files is')
  })

  it('previews, requires the confirmation checkbox and reports a partial cleanup', async () => {
    fetchStorageInventory
      .mockResolvedValueOnce(inventoryFixture())
      .mockResolvedValueOnce(inventoryFixture())
    previewStorageCleanup.mockResolvedValueOnce({
      plan_id: 'a'.repeat(64),
      category: 'models',
      category_label: 'Model files',
      keep_history: null,
      units: [{ id: 'unit-1', label: 'voice-models/orphan', bytes: 4096 }],
      refused: [
        {
          id: 'unit-2',
          label: 'voice-models/keep',
          bytes: 4096,
          code: 'CATALOG_MODEL',
          message: 'A catalog model uses this directory.',
        },
      ],
      reclaimable_bytes: 4096,
      confirmation_phrase: 'DELETE 1 MODEL DIRECTORIES',
      scope_cap_units: 200,
      expires_at: '2026-11-22T20:05:00+00:00',
      notice: 'Review the selected units.',
    })
    applyStorageCleanup.mockResolvedValueOnce({
      plan_id: 'a'.repeat(64),
      category: 'models',
      category_label: 'Model files',
      outcome: 'partial',
      freed_bytes: 4096,
      removed: [{ id: 'unit-1', label: 'voice-models/orphan', bytes: 4096 }],
      failed: [
        {
          id: 'unit-2',
          label: 'voice-models/keep',
          bytes: 4096,
          code: 'CATALOG_MODEL',
          message: 'A catalog model uses this directory.',
        },
      ],
      notice: 'Some units were removed; the rest stayed.',
    })
    const wrapper = mountPanel()
    await flushPromises()

    await wrapper.get('button[aria-label="Review cleanup for Model files"]').trigger('click')
    await flushPromises()

    expect(previewStorageCleanup).toHaveBeenCalledWith({ category: 'models' })
    expect(wrapper.text()).toContain('DELETE 1 MODEL DIRECTORIES')
    expect(wrapper.text()).toContain('These units are refused and will stay')
    const deleteButton = wrapper.get(
      'button[aria-label="Delete the selected units from Model files"]',
    )
    expect(deleteButton.attributes('disabled')).toBeDefined()

    await wrapper.get('.storage-confirm input[type="checkbox"]').setValue(true)
    await deleteButton.trigger('click')
    await flushPromises()

    expect(applyStorageCleanup).toHaveBeenCalledWith('a'.repeat(64))
    expect(wrapper.text()).toContain('Cleanup partial')
    expect(wrapper.text()).toContain('4.0 KiB freed')
    expect(wrapper.text()).toContain('Removed voice-models/orphan')
    expect(wrapper.text()).toContain('voice-models/keep: A catalog model uses this directory.')
    expect(wrapper.text()).toContain('Some units were removed; the rest stayed.')
  })

  it('passes the history keep count to the preview', async () => {
    fetchStorageInventory.mockResolvedValueOnce(
      inventoryFixture([
        category({
          id: 'history_backups',
          label: 'Configuration history',
          description: 'Local pre-change configuration revisions.',
          bytes: 500,
          unit_count: 20,
          cleanable: true,
          cleanable_bytes: 300,
          manage_path: null,
          units: [unit({ id: 'history-1', label: 'history-1.yaml', bytes: 10 })],
        }),
      ]),
    )
    previewStorageCleanup.mockResolvedValueOnce({
      plan_id: 'b'.repeat(64),
      category: 'history_backups',
      category_label: 'Configuration history',
      keep_history: 25,
      units: [],
      refused: [],
      reclaimable_bytes: 0,
      confirmation_phrase: 'NOTHING TO PRUNE FROM CONFIGURATION REVISIONS',
      scope_cap_units: 200,
      expires_at: '2026-11-22T20:05:00+00:00',
      notice: 'Nothing is selected; the plan cannot be applied.',
    })
    const wrapper = mountPanel()
    await flushPromises()

    await wrapper.get('#storage-keep-history').setValue('25')
    await wrapper
      .get('button[aria-label="Review cleanup for Configuration history"]')
      .trigger('click')
    await flushPromises()

    expect(previewStorageCleanup).toHaveBeenCalledWith({
      category: 'history_backups',
      keep_history: 25,
    })
  })

  it('shows a stable error when the inventory cannot be loaded', async () => {
    fetchStorageInventory.mockRejectedValueOnce(new Error('private filesystem detail'))
    const wrapper = mountPanel()
    await flushPromises()

    expect(wrapper.get('[role="alert"]').text()).toBe(
      'Storage usage could not be loaded. Retry the scan.',
    )
    expect(wrapper.text()).not.toContain('private filesystem detail')
  })
})
