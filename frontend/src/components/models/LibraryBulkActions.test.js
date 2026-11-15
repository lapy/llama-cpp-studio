import { beforeEach, describe, expect, it, vi } from 'vitest'
import { flushPromises, mount } from '@vue/test-utils'

import LibraryBulkActions from './LibraryBulkActions.vue'

const modelStore = { startModel: vi.fn(), stopModel: vi.fn() }
vi.mock('@/stores/models', () => ({ useModelStore: () => modelStore }))

const toastAdd = vi.fn()
vi.mock('primevue/usetoast', () => ({ useToast: () => ({ add: (...args) => toastAdd(...args) }) }))

const ButtonStub = {
  props: ['label', 'disabled', 'loading', 'severity', 'size', 'text', 'icon'],
  emits: ['click'],
  template:
    '<button :disabled="disabled" :data-loading="loading ? `true` : undefined" @click="$emit(`click`)">{{ label }}</button>',
}

function mountBulk(props = {}) {
  return mount(LibraryBulkActions, {
    props: { models: [], busyIds: [], ...props },
    global: { stubs: { Button: ButtonStub } },
  })
}

function deferred() {
  let resolve
  let reject
  const promise = new Promise((resolvePromise, rejectPromise) => {
    resolve = resolvePromise
    reject = rejectPromise
  })
  return { promise, resolve, reject }
}

function buttonByText(wrapper, text) {
  return wrapper.findAll('button').find((button) => button.text() === text)
}

const STOPPED_A = { id: 'a', quantization: 'Q4_A', is_active: false, status: 'stopped' }
const STOPPED_B = { id: 'b', quantization: 'Q8_B', is_active: false, status: 'stopped' }
const RUNNING = { id: 'running', quantization: 'Q5_R', is_active: true, status: 'ready' }

describe('LibraryBulkActions', () => {
  beforeEach(() => {
    modelStore.startModel.mockReset()
    modelStore.stopModel.mockReset()
    toastAdd.mockReset()
  })

  it('renders nothing without a selection', () => {
    const wrapper = mountBulk()
    expect(wrapper.find('.library-bulk').exists()).toBe(false)
  })

  it('offers only actions eligible for selected models and ignores in-flight ids', async () => {
    const wrapper = mountBulk({
      models: [STOPPED_A, RUNNING, { ...STOPPED_B, id: 'c' }],
      busyIds: ['c'],
    })

    expect(wrapper.get('.library-bulk__count').text()).toBe('3 selected · 1 can start · 1 can stop')
    expect(buttonByText(wrapper, 'Start selected').attributes('disabled')).toBeUndefined()
    expect(buttonByText(wrapper, 'Stop selected').attributes('disabled')).toBeUndefined()

    await buttonByText(wrapper, 'Start selected').trigger('click')
    await flushPromises()
    expect(modelStore.startModel).toHaveBeenCalledWith('a')
    expect(wrapper.emitted('complete')?.at(-1)[0]).toMatchObject({
      action: 'start',
      total: 1,
      done: 1,
    })
  })

  it('stops each active selected model through the store action', async () => {
    const wrapper = mountBulk({ models: [RUNNING] })

    await buttonByText(wrapper, 'Stop selected').trigger('click')
    await flushPromises()

    expect(modelStore.stopModel).toHaveBeenCalledWith('running')
    expect(wrapper.emitted('complete')?.at(-1)[0]).toMatchObject({
      action: 'stop',
      total: 1,
      done: 1,
      failed: 0,
    })
    expect(wrapper.get('.library-bulk__result').text()).toContain('Stopped 1 of 1 selected model.')
  })

  it('runs sequentially and reports progress and final counts', async () => {
    const first = deferred()
    const second = deferred()
    modelStore.startModel
      .mockImplementationOnce(() => first.promise)
      .mockImplementationOnce(() => second.promise)
    const wrapper = mountBulk({ models: [STOPPED_A, STOPPED_B] })

    await buttonByText(wrapper, 'Start selected').trigger('click')

    expect(modelStore.startModel).toHaveBeenCalledTimes(1)
    expect(modelStore.startModel).toHaveBeenNthCalledWith(1, 'a')
    expect(wrapper.get('.library-bulk__progress').text()).toBe('Starting 1 of 2…')

    first.resolve()
    await flushPromises()
    expect(modelStore.startModel).toHaveBeenCalledTimes(2)
    expect(modelStore.startModel).toHaveBeenNthCalledWith(2, 'b')

    second.resolve()
    await flushPromises()

    expect(wrapper.emitted('complete')?.at(-1)[0]).toEqual({
      action: 'start',
      total: 2,
      done: 2,
      failed: 0,
      skipped: 0,
    })
    expect(wrapper.get('.library-bulk__result').text()).toContain('Started 2 of 2 selected models.')
    expect(toastAdd).toHaveBeenLastCalledWith(
      expect.objectContaining({ severity: 'success', summary: 'Bulk start finished' }),
    )
  })

  it('continues after a failed model and toasts that model', async () => {
    modelStore.startModel
      .mockRejectedValueOnce({ response: { data: { detail: 'Engine refused to start' } } })
      .mockResolvedValueOnce(undefined)
    const wrapper = mountBulk({ models: [STOPPED_A, STOPPED_B] })

    await buttonByText(wrapper, 'Start selected').trigger('click')
    await flushPromises()

    expect(modelStore.startModel).toHaveBeenCalledTimes(2)
    expect(toastAdd).toHaveBeenCalledWith(
      expect.objectContaining({
        severity: 'error',
        summary: 'Could not start Q4_A',
        detail: 'Engine refused to start',
      }),
    )
    expect(wrapper.emitted('complete')?.at(-1)[0]).toMatchObject({ done: 1, failed: 1 })
    expect(wrapper.get('.library-bulk__result').text()).toContain('1 failed')
    expect(toastAdd).toHaveBeenLastCalledWith(expect.objectContaining({ severity: 'warn' }))
  })

  it('skips models that disappear before their turn', async () => {
    const first = deferred()
    modelStore.startModel.mockImplementationOnce(() => first.promise)
    const wrapper = mountBulk({ models: [STOPPED_A, STOPPED_B] })

    await buttonByText(wrapper, 'Start selected').trigger('click')
    await wrapper.setProps({ models: [] })
    first.resolve()
    await flushPromises()

    expect(modelStore.startModel).toHaveBeenCalledTimes(1)
    expect(wrapper.emitted('complete')?.at(-1)[0]).toEqual({
      action: 'start',
      total: 2,
      done: 1,
      failed: 0,
      skipped: 1,
    })
    expect(wrapper.get('.library-bulk__result').text()).toContain('1 skipped')
  })

  it('disables both actions when no selected model can run', () => {
    const wrapper = mountBulk({
      models: [{ id: 'loading', is_active: false, status: 'loading' }],
    })

    expect(wrapper.get('.library-bulk__count').text()).toBe('1 selected · no eligible actions')
    expect(buttonByText(wrapper, 'Start selected').attributes('disabled')).toBeDefined()
    expect(buttonByText(wrapper, 'Stop selected').attributes('disabled')).toBeDefined()
  })

  it('emits clear when the user dismisses the selection', async () => {
    const wrapper = mountBulk({ models: [STOPPED_A] })

    await buttonByText(wrapper, 'Clear selection').trigger('click')

    expect(wrapper.emitted('clear')).toHaveLength(1)
  })
})
