import { describe, expect, it } from 'vitest'
import { mount } from '@vue/test-utils'

import ModelConfigActionBar from './ModelConfigActionBar.vue'

const ButtonStub = {
  props: ['label', 'disabled', 'loading', 'severity', 'outlined', 'size', 'icon', 'ariaLabel'],
  emits: ['click'],
  template: `
    <button
      type="button"
      :data-label="label"
      :disabled="disabled || undefined"
      :data-loading="loading ? 'true' : undefined"
      :data-severity="severity"
      :data-outlined="outlined ? 'true' : undefined"
      @click="!disabled && $emit('click')"
    >{{ label }}</button>
  `,
}

const TagStub = {
  props: ['value', 'severity'],
  template: '<span class="tag-stub" :data-severity="severity">{{ value }}</span>',
}

function status(overrides = {}) {
  return {
    key: 'applied',
    label: 'Applied',
    severity: 'success',
    detail: 'These settings are published. The model is stopped.',
    ...overrides,
  }
}

function mountBar(props = {}) {
  return mount(ModelConfigActionBar, {
    props: {
      modelLabel: 'Test Model',
      quantization: 'Q4_K_M',
      format: 'GGUF',
      engineLabel: 'llama.cpp',
      status: status(),
      dirty: false,
      canSave: false,
      saveHint: 'No changes to save.',
      saving: false,
      running: false,
      starting: false,
      canStart: true,
      startDisabledReason: '',
      canReview: false,
      reviewLabel: 'Review changes',
      showImportCommand: false,
      ...props,
    },
    global: {
      directives: { tooltip: () => {} },
      stubs: { Button: ButtonStub, Tag: TagStub },
    },
  })
}

function buttonByLabel(wrapper, label) {
  return wrapper.findAll('button').find((button) => button.attributes('data-label') === label)
}

describe('ModelConfigActionBar', () => {
  it('shows the model identity, engine, status pill and human status detail', () => {
    const wrapper = mountBar()

    const identity = wrapper.get('.config-action-bar__identity').text()
    expect(identity).toContain('Test Model')
    expect(identity).toContain('Q4_K_M')
    expect(identity).toContain('GGUF')
    expect(identity).toContain('llama.cpp')

    expect(wrapper.get('[data-status-key="applied"]').text()).toBe('Applied')
    expect(wrapper.get('.config-action-bar__status').attributes('role')).toBe('status')
    expect(wrapper.get('.runtime-state__detail').text()).toContain('published')
    expect(wrapper.get('.config-action-bar').attributes('role')).toBe('region')
    expect(wrapper.get('.config-action-bar').attributes('aria-label')).toBe('Configuration actions')
  })

  it('keeps Save disabled and offers Start when the configuration is clean', () => {
    const wrapper = mountBar()

    const save = buttonByLabel(wrapper, 'Save changes')
    const start = buttonByLabel(wrapper, 'Start model')
    expect(save.attributes('disabled')).toBeDefined()
    expect(start.attributes('disabled')).toBeUndefined()
    expect(buttonByLabel(wrapper, 'Apply settings')).toBeUndefined()
    expect(buttonByLabel(wrapper, 'Open in library')).toBeUndefined()
  })

  it('enables Save and blocks Start while there are unsaved changes', async () => {
    const wrapper = mountBar({
      dirty: true,
      canSave: true,
      saveHint: 'Save this form.',
      status: status({ key: 'dirty', label: 'Unsaved changes', severity: 'warn' }),
      canStart: false,
      startDisabledReason: 'Save or discard these edits before starting the model.',
      canReview: true,
    })

    const save = buttonByLabel(wrapper, 'Save changes')
    const start = buttonByLabel(wrapper, 'Start model')
    expect(save.attributes('disabled')).toBeUndefined()
    expect(start.attributes('disabled')).toBeDefined()
    expect(start.attributes('aria-describedby')).toBe('config-action-bar-start-reason')
    expect(wrapper.get('#config-action-bar-start-reason').text()).toContain('Save or discard')
    expect(wrapper.get('[data-status-key="dirty"]').text()).toBe('Unsaved changes')

    await save.trigger('click')
    await start.trigger('click')
    expect(wrapper.emitted('save')).toHaveLength(1)
    expect(wrapper.emitted('start')).toBeUndefined()
  })

  it('shows the saving state on Save and keeps the remaining actions disabled', () => {
    const wrapper = mountBar({
      dirty: true,
      canSave: false,
      saving: true,
      saveHint: 'Saving changes…',
      status: status({ key: 'dirty', label: 'Unsaved changes', severity: 'warn' }),
      canStart: false,
      startDisabledReason: 'Save or discard these edits before starting the model.',
    })

    const save = buttonByLabel(wrapper, 'Save changes')
    expect(save.attributes('data-loading')).toBe('true')
    expect(save.attributes('disabled')).toBeDefined()
  })

  it('shows Apply with the runtime label and blocks Start while an apply is pending', async () => {
    const wrapper = mountBar({
      showApply: true,
      applyLabel: 'Restart this model',
      applyHint: 'Restart only this model.',
      status: status({ key: 'pending', label: 'Saved · needs apply', severity: 'warn' }),
      canStart: false,
      startDisabledReason: 'Apply the saved settings before starting.',
    })

    const apply = buttonByLabel(wrapper, 'Restart this model')
    expect(apply).toBeDefined()
    expect(apply.attributes('disabled')).toBeUndefined()
    expect(buttonByLabel(wrapper, 'Start model').attributes('disabled')).toBeDefined()
    expect(wrapper.get('[data-status-key="pending"]').text()).toBe('Saved · needs apply')

    await apply.trigger('click')
    expect(wrapper.emitted('apply')).toHaveLength(1)
  })

  it('keeps Save and Start disabled when persistence holds the page', () => {
    const wrapper = mountBar({
      dirty: true,
      canSave: false,
      saveHint: 'Saving is paused.',
      canStart: false,
      startDisabledReason: 'Resolve the configuration problem before starting the model.',
    })

    expect(buttonByLabel(wrapper, 'Save changes').attributes('disabled')).toBeDefined()
    expect(buttonByLabel(wrapper, 'Start model').attributes('disabled')).toBeDefined()
  })

  it('replaces Start with Open in library while the model is running', async () => {
    const wrapper = mountBar({
      running: true,
      canStart: false,
      status: status({ key: 'running', label: 'Running', severity: 'success' }),
    })

    expect(buttonByLabel(wrapper, 'Start model')).toBeUndefined()
    const open = buttonByLabel(wrapper, 'Open in library')
    expect(open).toBeDefined()
    expect(wrapper.get('[data-status-key="running"]').text()).toBe('Running')

    await open.trigger('click')
    expect(wrapper.emitted('open-library')).toHaveLength(1)
  })

  it('shows Starting with a loading Start action that cannot be clicked', () => {
    const wrapper = mountBar({
      starting: true,
      canStart: false,
      startDisabledReason: 'The model is already starting.',
      status: status({ key: 'starting', label: 'Starting', severity: 'warn' }),
    })

    const start = buttonByLabel(wrapper, 'Start model')
    expect(start.attributes('data-loading')).toBe('true')
    expect(start.attributes('disabled')).toBeDefined()
    expect(wrapper.get('[data-status-key="starting"]').text()).toBe('Starting')
  })

  it('shows Failed after a start error and keeps Start available for a retry', async () => {
    const wrapper = mountBar({
      canStart: true,
      status: status({
        key: 'failed',
        label: 'Failed',
        severity: 'danger',
        detail: 'Port already in use.',
      }),
    })

    expect(wrapper.get('[data-status-key="failed"]').text()).toBe('Failed')
    expect(wrapper.get('.runtime-state__detail').text()).toContain('Port already in use')
    const start = buttonByLabel(wrapper, 'Start model')
    expect(start.attributes('disabled')).toBeUndefined()

    await start.trigger('click')
    expect(wrapper.emitted('start')).toHaveLength(1)
  })

  it('keeps the running pill visible when the form is also dirty', () => {
    const wrapper = mountBar({
      running: true,
      dirty: true,
      canSave: true,
      status: status({ key: 'dirty', label: 'Unsaved changes', severity: 'warn' }),
    })

    expect(wrapper.get('[data-status-key="dirty"]').text()).toBe('Unsaved changes')
    expect(wrapper.get('[data-status-key="running-extra"]').text()).toBe('Running')
  })

  it('routes secondary actions through the overflow without hiding them from assistive tech', async () => {
    const wrapper = mountBar({
      canReview: true,
      reviewLabel: 'Review changes (2)',
      showImportCommand: true,
    })

    await buttonByLabel(wrapper, 'Review changes (2)').trigger('click')
    await buttonByLabel(wrapper, 'Templates').trigger('click')
    await buttonByLabel(wrapper, 'Import command').trigger('click')
    await buttonByLabel(wrapper, 'Reset to saved').trigger('click')

    expect(wrapper.emitted('review')).toHaveLength(1)
    expect(wrapper.emitted('templates')).toHaveLength(1)
    expect(wrapper.emitted('import-command')).toHaveLength(1)
    expect(wrapper.emitted('reset')).toHaveLength(1)
  })

  it('does not render Import command when the engine has no command import', () => {
    const wrapper = mountBar({ showImportCommand: false })
    expect(buttonByLabel(wrapper, 'Import command')).toBeUndefined()
  })
})
