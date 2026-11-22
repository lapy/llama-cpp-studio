import { beforeEach, describe, expect, it, vi } from 'vitest'
import { flushPromises, mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { nextTick } from 'vue'

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

function inventoryFixture(categories = [category()], overrides = {}) {
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
    ...overrides,
  }
}

function planFixture(overrides = {}) {
  return {
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
    ...overrides,
  }
}

function mountPanel(options = {}) {
  return mount(StoragePanel, {
    ...options,
    global: {
      stubs: {
        Button: {
          props: ['label', 'disabled', 'loading'],
          template: '<button :disabled="disabled">{{ label }}</button>',
        },
        RouterLink: {
          props: ['to'],
          template: '<a :href="to"><slot /></a>',
        },
        Dialog: {
          props: ['visible', 'header', 'modal'],
          emits: ['hide', 'update:visible'],
          template:
            '<div v-if="visible" role="dialog" :aria-label="header"><slot /><slot name="footer" /></div>',
        },
        InputText: {
          inheritAttrs: false,
          props: ['modelValue'],
          emits: ['update:modelValue'],
          template:
            '<input v-bind="$attrs" :value="modelValue" @input="$emit(`update:modelValue`, $event.target.value)" />',
        },
        Checkbox: {
          props: ['modelValue', 'binary', 'disabled'],
          emits: ['update:modelValue'],
          template:
            '<input type="checkbox" :checked="modelValue" :disabled="disabled" @change="$emit(`update:modelValue`, $event.target.checked)" />',
        },
        Tag: {
          props: ['value', 'severity'],
          template: '<span class="p-tag">{{ value }}</span>',
        },
        LoadingState: true,
      },
      ...(options.global || {}),
    },
  })
}

describe('StoragePanel', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    vi.clearAllMocks()
  })

  it('renders one page section with the shared refresh action', async () => {
    fetchStorageInventory.mockResolvedValueOnce(inventoryFixture())
    const wrapper = mountPanel()
    await flushPromises()

    const section = wrapper.get('#storage')
    expect(section.classes()).toContain('page-section')
    expect(section.get('.page-section__title').text()).toBe('Storage')
    const refresh = section
      .get('.page-section__actions')
      .findAll('button')
      .find((button) => button.text() === 'Refresh storage')
    expect(refresh).toBeTruthy()

    fetchStorageInventory.mockResolvedValueOnce(inventoryFixture())
    await refresh.trigger('click')
    await flushPromises()
    expect(fetchStorageInventory).toHaveBeenCalledTimes(2)
    expect(fetchStorageInventory).toHaveBeenLastCalledWith({ refresh: true })
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

    expect(wrapper.get('.storage-category h3').text()).toBe('Model files')
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
    expect(wrapper.get('.state-banner--warning').text()).toContain(
      'This storage is managed by another Studio surface.',
    )
  })

  it('previews, requires the confirmation checkbox and reports a partial cleanup', async () => {
    fetchStorageInventory
      .mockResolvedValueOnce(inventoryFixture())
      .mockResolvedValueOnce(inventoryFixture())
    previewStorageCleanup.mockResolvedValueOnce(planFixture())
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
    const dialog = wrapper.get('[role="dialog"]')
    expect(dialog.attributes('aria-label')).toBe('Confirm cleanup for Model files')
    expect(dialog.text()).toContain('DELETE 1 MODEL DIRECTORIES')
    expect(dialog.text()).toContain('These units are refused and will stay')
    const deleteButton = dialog.get(
      'button[aria-label="Delete the selected units from Model files"]',
    )
    expect(deleteButton.attributes('disabled')).toBeDefined()

    await wrapper.get('.storage-confirm input[type="checkbox"]').setValue(true)
    await nextTick()
    expect(deleteButton.attributes('disabled')).toBeUndefined()
    await deleteButton.trigger('click')
    await flushPromises()

    expect(applyStorageCleanup).toHaveBeenCalledWith('a'.repeat(64))
    expect(wrapper.text()).toContain('Cleanup partial')
    expect(wrapper.text()).toContain('4.0 KiB freed')
    expect(wrapper.text()).toContain('Removed voice-models/orphan')
    expect(wrapper.text()).toContain('voice-models/keep: A catalog model uses this directory.')
    expect(wrapper.text()).toContain('Some units were removed; the rest stayed.')
  })

  it('shows an apply failure inside the confirmation dialog', async () => {
    fetchStorageInventory.mockResolvedValueOnce(inventoryFixture())
    previewStorageCleanup.mockResolvedValueOnce(planFixture({ plan_id: 'c'.repeat(64) }))
    applyStorageCleanup.mockRejectedValueOnce({
      response: { data: { detail: { message: 'The cleanup could not start.' } } },
    })
    const wrapper = mountPanel()
    await flushPromises()

    await wrapper.get('button[aria-label="Review cleanup for Model files"]').trigger('click')
    await flushPromises()
    await wrapper.get('.storage-confirm input[type="checkbox"]').setValue(true)
    await wrapper
      .get('button[aria-label="Delete the selected units from Model files"]')
      .trigger('click')
    await flushPromises()

    const dialog = wrapper.get('[role="dialog"]')
    expect(dialog.get('[role="alert"]').get('.state-banner__message').text()).toBe(
      'The cleanup could not start.',
    )
  })

  it('returns focus to the review action after the confirmation closes', async () => {
    fetchStorageInventory.mockResolvedValueOnce(inventoryFixture())
    previewStorageCleanup.mockResolvedValueOnce(planFixture())
    const wrapper = mountPanel({ attachTo: document.body })
    await flushPromises()

    const review = wrapper.get('button[aria-label="Review cleanup for Model files"]')
    review.element.focus()
    await review.trigger('click')
    await flushPromises()
    expect(wrapper.find('[role="dialog"]').exists()).toBe(true)

    const cancel = wrapper
      .get('[role="dialog"]')
      .findAll('button')
      .find((button) => button.text() === 'Cancel cleanup')
    await cancel.trigger('click')
    await flushPromises()
    await nextTick()

    expect(wrapper.find('[role="dialog"]').exists()).toBe(false)
    expect(document.activeElement).toBe(review.element)
    wrapper.unmount()
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

  it('offers a scan from the empty state', async () => {
    fetchStorageInventory.mockResolvedValueOnce(undefined).mockResolvedValueOnce(inventoryFixture())
    const wrapper = mountPanel()
    await flushPromises()

    const empty = wrapper.get('.page-empty')
    expect(empty.text()).toContain('No storage inventory yet')
    const scan = empty.findAll('button').find((button) => button.text() === 'Scan storage')
    expect(scan).toBeTruthy()

    await scan.trigger('click')
    await flushPromises()
    expect(fetchStorageInventory).toHaveBeenLastCalledWith({ refresh: true })
    expect(wrapper.find('.page-empty').exists()).toBe(false)
    expect(wrapper.text()).toContain('Model files')
  })

  it('warns when the scan is partial and offers a rescan', async () => {
    const partial = inventoryFixture([category()])
    partial.scan.quality = 'partial'
    fetchStorageInventory.mockResolvedValueOnce(partial)
    const wrapper = mountPanel()
    await flushPromises()

    const warning = wrapper.get('.state-banner--warning')
    expect(warning.text()).toContain('This scan is partial')
    expect(warning.text()).toContain('Refresh scan')
  })

  it('shows a stable error with retry when the inventory cannot be loaded', async () => {
    fetchStorageInventory.mockRejectedValueOnce(new Error('private filesystem detail'))
    const wrapper = mountPanel()
    await flushPromises()

    const banner = wrapper.get('[role="alert"]')
    expect(banner.get('.state-banner__message').text()).toBe(
      'Storage usage could not be loaded. Retry the scan.',
    )
    expect(wrapper.text()).not.toContain('private filesystem detail')

    fetchStorageInventory.mockResolvedValueOnce(inventoryFixture())
    const retry = banner.findAll('button').find((button) => button.text() === 'Retry')
    await retry.trigger('click')
    await flushPromises()

    expect(fetchStorageInventory).toHaveBeenCalledTimes(2)
    expect(wrapper.find('[role="alert"]').exists()).toBe(false)
    expect(wrapper.text()).toContain('Model files')
  })
})
