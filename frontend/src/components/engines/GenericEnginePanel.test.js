/**
 * YAML campaign S6 — GenericEnginePanel: the install/manage journey for
 * SERVER-BACKED (YAML-onboarded) engines renders and acts from the live
 * descriptor payload alone (fixture id below is test payload data).
 */
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { flushPromises, mount } from '@vue/test-utils'

const { confirmRequire, toastAdd, store } = vi.hoisted(() => ({
  confirmRequire: vi.fn(),
  toastAdd: vi.fn(),
  store: {
    serverStatuses: {},
    serverVersions: {},
    venvFetchStatus: vi.fn().mockResolvedValue({}),
    venvGetBuildSettings: vi.fn().mockResolvedValue({
      source_repo: 'https://github.com/example-org/yamlcraft.git',
      source_branch: 'main',
      pip_version: '',
    }),
    venvSaveBuildSettings: vi.fn().mockResolvedValue({}),
    venvInstall: vi.fn().mockResolvedValue({ message: 'Crafty installation started' }),
    venvInstallFromSource: vi.fn().mockResolvedValue({ message: 'Crafty installation started' }),
    venvRemove: vi.fn().mockResolvedValue({}),
    venvCancel: vi.fn().mockResolvedValue({}),
    activateVersion: vi.fn().mockResolvedValue({}),
    venvCheckUpdates: vi
      .fn()
      .mockResolvedValue({ update_available: true, latest_version: '0.2.0' }),
    fetchLlamaVersions: vi.fn().mockResolvedValue([]),
  },
}))

vi.mock('@/stores/engines', () => ({ useEnginesStore: () => store }))
vi.mock('primevue/useconfirm', () => ({ useConfirm: () => ({ require: confirmRequire }) }))
vi.mock('primevue/usetoast', () => ({ useToast: () => ({ add: toastAdd }) }))

import GenericEnginePanel from './GenericEnginePanel.vue'

const CRAFTY_DESCRIPTOR = {
  id: 'yamlcraft',
  label: 'Crafty',
  repository_source: 'Crafty Runtime',
  install_kind: 'python_venv',
  artifact_formats: ['fixture_only'],
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
}

function mountPanel() {
  return mount(GenericEnginePanel, {
    props: { descriptor: CRAFTY_DESCRIPTOR },
    global: {
      stubs: {
        EngineCard: {
          props: ['title', 'count', 'cta'],
          emits: ['open'],
          template: `
            <div data-testid="card" :data-title="title" :data-count="count" :data-cta="cta" @click="$emit('open')">
              <slot name="badge" /><slot />
            </div>
          `,
        },
        Dialog: { template: '<div><slot /><slot name="footer" /></div>' },
        Button: {
          props: ['label', 'disabled'],
          emits: ['click'],
          template:
            '<button :data-label="label" :disabled="disabled" @click="$emit(\'click\')">{{ label }}</button>',
        },
        InputText: {
          props: ['modelValue'],
          emits: ['update:modelValue'],
          template:
            '<input :value="modelValue" @input="$emit(\'update:modelValue\', $event.target.value)" />',
        },
      },
    },
  })
}

beforeEach(() => {
  confirmRequire.mockReset()
  toastAdd.mockReset()
  for (const fn of Object.values(store)) if (fn?.mockReset) fn.mockReset()
  store.serverStatuses = {}
  store.serverVersions = {}
  store.venvGetBuildSettings.mockResolvedValue({
    source_repo: 'https://github.com/example-org/yamlcraft.git',
    source_branch: 'main',
    pip_version: '',
  })
  store.venvInstall.mockResolvedValue({ message: 'Crafty installation started' })
  store.venvCheckUpdates.mockResolvedValue({ update_available: true, latest_version: '0.2.0' })
})

describe('GenericEnginePanel', () => {
  it('renders a card for a YAML-only engine from the descriptor alone', () => {
    const wrapper = mountPanel()
    const card = wrapper.get('[data-testid="card"]')
    expect(card.attributes('data-title')).toBe('Crafty')
    expect(card.attributes('data-count')).toBe('0')
    expect(card.attributes('data-cta')).toBe('Install')
    expect(wrapper.text()).toContain('Not installed')
  })

  it('shows installed state and Manage cta from the dynamic maps', () => {
    store.serverStatuses = { yamlcraft: { installed: true, version: '0.1.0' } }
    store.serverVersions = { yamlcraft: [{ id: 'c1', version: '0.1.0', build_status: 'ready' }] }
    const wrapper = mountPanel()
    expect(wrapper.get('[data-testid="card"]').attributes('data-cta')).toBe('Manage')
    expect(wrapper.text()).toContain('Installed: 0.1.0')
  })

  it('opens the manage dialog and prefills source settings', async () => {
    const wrapper = mountPanel()
    await wrapper.get('[data-testid="card"]').trigger('click')
    await flushPromises()
    expect(store.venvGetBuildSettings).toHaveBeenCalledWith('yamlcraft')
    const repo = wrapper
      .findAll('input')
      .find((input) => input.attributes('placeholder') === 'repository URL')
    expect(repo.element.value).toBe('https://github.com/example-org/yamlcraft.git')
  })

  it('installs a release through the generic verb and reports the engine message', async () => {
    const wrapper = mountPanel()
    await wrapper.get('[data-testid="card"]').trigger('click')
    await flushPromises()
    const versionField = wrapper
      .findAll('input')
      .find((input) => (input.attributes('placeholder') || '').startsWith('version'))
    await versionField.setValue('0.1.0')
    await wrapper
      .findAll('button')
      .find((b) => b.attributes('data-label') === 'Install')
      .trigger('click')
    await flushPromises()
    expect(store.venvInstall).toHaveBeenCalledWith('yamlcraft', { version: '0.1.0' })
    expect(toastAdd).toHaveBeenCalledWith(
      expect.objectContaining({
        severity: 'success',
        detail: 'Crafty installation started',
      }),
    )
  })

  it('prefills the version from check-updates before install', async () => {
    const wrapper = mountPanel()
    await wrapper.get('[data-testid="card"]').trigger('click')
    await flushPromises()
    await wrapper
      .findAll('button')
      .find((b) => b.attributes('data-label') === 'Check updates')
      .trigger('click')
    await flushPromises()
    expect(store.venvCheckUpdates).toHaveBeenCalledWith('yamlcraft')
    expect(wrapper.text()).toContain('latest_version: 0.2.0')
    await wrapper
      .findAll('button')
      .find((b) => b.attributes('data-label') === 'Install')
      .trigger('click')
    await flushPromises()
    expect(store.venvInstall).toHaveBeenCalledWith('yamlcraft', { version: '0.2.0' })
  })

  it('installs from source with repo defaults carried through', async () => {
    const wrapper = mountPanel()
    await wrapper.get('[data-testid="card"]').trigger('click')
    await flushPromises()
    await wrapper
      .findAll('button')
      .find((b) => b.attributes('data-label') === 'Install source')
      .trigger('click')
    await flushPromises()
    expect(store.venvInstallFromSource).toHaveBeenCalledWith('yamlcraft', {
      repo_url: 'https://github.com/example-org/yamlcraft.git',
      branch: 'main',
    })
  })

  it('exposes a running operation with cancel wired to the generic verb', async () => {
    // REAL status contract (_stateful_status): `operation` is the running
    // operation NAME (string); the cancel handle is the separate
    // `progress_task_id` field. An earlier draft of this test invented a
    // nested-object shape and passed against the same invention — the
    // panel now reads the real bytes the server sends.
    store.serverStatuses = {
      yamlcraft: {
        installed: false,
        operation: 'install_release',
        progress_task_id: 'install_yamlcraft',
      },
    }
    const wrapper = mountPanel()
    expect(wrapper.text()).toContain('Operation running: install_release')
    await wrapper.get('[data-testid="card"]').trigger('click')
    await flushPromises()
    const install = wrapper.findAll('button').find((b) => b.attributes('data-label') === 'Install')
    expect(install.attributes('disabled')).toBeDefined()
    await wrapper
      .findAll('button')
      .find((b) => b.attributes('data-label') === 'Cancel')
      .trigger('click')
    await flushPromises()
    expect(store.venvCancel).toHaveBeenCalledWith('yamlcraft', 'install_yamlcraft')
  })

  it('renders nothing actionable for an operation it cannot see (no nested-object guessing)', () => {
    // A stale nested object must NOT masquerade as a running operation.
    store.serverStatuses = { yamlcraft: { installed: false, operation: { task_id: 't9' } } }
    const wrapper = mountPanel()
    expect(wrapper.text()).not.toContain('Operation running')
    expect(wrapper.text()).toContain('Not installed')
  })

  it('hides every action the engine lifecycle does not advertise', async () => {
    const restricted = mount(GenericEnginePanel, {
      props: { descriptor: { ...CRAFTY_DESCRIPTOR, generic_verbs: ['status'] } },
      global: {
        stubs: {
          EngineCard: {
            props: ['title', 'count', 'cta'],
            emits: ['open'],
            template: '<div data-testid="card" @click="$emit(\'open\')"><slot /></div>',
          },
          Dialog: { template: '<div><slot /><slot name="footer" /></div>' },
          Button: {
            props: ['label', 'disabled'],
            emits: ['click'],
            template: '<button :data-label="label">{{ label }}</button>',
          },
          InputText: { props: ['modelValue'], template: '<input />' },
        },
      },
    })
    await restricted.get('[data-testid="card"]').trigger('click')
    await flushPromises()
    const labels = restricted.findAll('button').map((b) => b.attributes('data-label'))
    expect(labels).not.toContain('Install')
    expect(labels).not.toContain('Install source')
    expect(labels).not.toContain('Check updates')
    expect(labels).not.toContain('Save settings')
    expect(labels).not.toContain('Remove engine')
    expect(labels).not.toContain('Cancel')
    expect(store.venvGetBuildSettings).not.toHaveBeenCalled() // status-only: no settings load
  })

  it('removes the engine behind confirmation', async () => {
    const wrapper = mountPanel()
    await wrapper.get('[data-testid="card"]').trigger('click')
    await flushPromises()
    await wrapper
      .findAll('button')
      .find((b) => b.attributes('data-label') === 'Remove engine')
      .trigger('click')
    expect(confirmRequire).toHaveBeenCalledTimes(1)
    const options = confirmRequire.mock.calls[0][0]
    expect(options.message).toContain('ACTIVE installed version')
    await options.accept()
    expect(store.venvRemove).toHaveBeenCalledWith('yamlcraft', {})
  })

  it('saves generic build settings through the settings verb', async () => {
    const wrapper = mountPanel()
    await wrapper.get('[data-testid="card"]').trigger('click')
    await flushPromises()
    const pinField = wrapper
      .findAll('input')
      .find((input) => input.attributes('placeholder') === 'pinned pip version')
    await pinField.setValue('0.3.1')
    await wrapper
      .findAll('button')
      .find((b) => b.attributes('data-label') === 'Save settings')
      .trigger('click')
    await flushPromises()
    expect(store.venvSaveBuildSettings).toHaveBeenCalledWith('yamlcraft', {
      pip_version: '0.3.1',
      source_repo: 'https://github.com/example-org/yamlcraft.git',
      source_branch: 'main',
    })
  })

  it('activates an inactive version through the shared version route', async () => {
    store.serverStatuses = { yamlcraft: { installed: true, version: '0.2.0' } }
    store.serverVersions = {
      yamlcraft: [
        { id: 'yamlcraft:0.1.0', version: '0.1.0', build_status: 'ready', is_active: false },
        { id: 'yamlcraft:0.2.0', version: '0.2.0', build_status: 'ready', is_active: true },
      ],
    }
    const wrapper = mountPanel()
    await wrapper.get('[data-testid="card"]').trigger('click')
    await flushPromises()
    const use = wrapper.findAll('button').find((b) => b.attributes('data-label') === 'Use')
    expect(use).toBeTruthy()
    await use.trigger('click')
    await flushPromises()
    expect(store.activateVersion).toHaveBeenCalledWith('yamlcraft:0.1.0')
    expect(store.venvFetchStatus).toHaveBeenCalledWith('yamlcraft')
  })

  it('normalizes structured error details instead of showing [object Object]', async () => {
    store.venvInstall.mockRejectedValueOnce({
      response: {
        data: {
          detail: {
            code: 'ACTION_RETRY_WITHHELD',
            message: 'Confirm the prior install attempt',
          },
        },
      },
    })
    const wrapper = mountPanel()
    await wrapper.get('[data-testid="card"]').trigger('click')
    await flushPromises()
    await wrapper
      .findAll('button')
      .find((b) => b.attributes('data-label') === 'Install')
      .trigger('click')
    await flushPromises()
    expect(toastAdd).toHaveBeenCalledWith(
      expect.objectContaining({ detail: 'Confirm the prior install attempt' }),
    )
  })

  it('confirms retirement when a launch reference blocks removal', async () => {
    store.venvRemove.mockRejectedValueOnce({
      response: {
        data: {
          detail: {
            code: 'RETAINED_LAUNCH_REFERENCE',
            message: 'This install is referenced by a launch generation',
          },
        },
      },
    })
    const wrapper = mountPanel()
    await wrapper.get('[data-testid="card"]').trigger('click')
    await flushPromises()
    await wrapper
      .findAll('button')
      .find((b) => b.attributes('data-label') === 'Remove engine')
      .trigger('click')
    const first = confirmRequire.mock.calls[0][0]
    await first.accept()
    await flushPromises()
    expect(confirmRequire).toHaveBeenCalledTimes(2)
    const second = confirmRequire.mock.calls[1][0]
    expect(second.message).toMatch(/launch generation/)
    await second.accept()
    expect(store.venvRemove).toHaveBeenLastCalledWith('yamlcraft', {
      retire_launch_references: true,
    })
  })
})
