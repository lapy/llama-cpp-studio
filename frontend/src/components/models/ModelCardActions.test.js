import { describe, expect, it, vi } from 'vitest'
import { flushPromises, mount } from '@vue/test-utils'

import ModelCardActions from './ModelCardActions.vue'
import { AUDIO_PACKAGE } from '@/utils/audioModel'

vi.mock('primevue/usetoast', () => ({ useToast: () => ({ add: () => {} }) }))

function mountActions(model = {}, props = {}) {
  return mount(ModelCardActions, {
    attachTo: document.body,
    props: {
      model: {
        id: 'm1',
        quantization: 'Q4_K_M',
        is_active: false,
        status: 'stopped',
        ...model,
      },
      name: 'Q4_K_M',
      ...props,
    },
    global: {
      directives: { tooltip: () => {} },
      stubs: { ModelWarmupButton: true },
    },
  })
}

function button(wrapper, ariaLabel) {
  return wrapper.get(`button[aria-label="${ariaLabel}"]`)
}

describe('ModelCardActions state matrix', () => {
  it('stopped: one prominent Start plus Configure, with no Stop', async () => {
    const wrapper = mountActions()
    const start = button(wrapper, 'Start Q4_K_M')
    expect(start.text()).toBe('Start')
    expect(wrapper.findAll('button[aria-label="Stop Q4_K_M"]')).toHaveLength(0)

    await start.trigger('click')
    expect(wrapper.emitted('start')?.[0]).toEqual(['m1'])

    await button(wrapper, 'Configure Q4_K_M').trigger('click')
    expect(wrapper.emitted('configure')?.[0]).toEqual(['m1'])
    wrapper.unmount()
  })

  it('starting from a request: disabled Starting… with a cancel Stop', async () => {
    const wrapper = mountActions({}, { isStarting: true })
    const primary = button(wrapper, 'Q4_K_M is starting')
    expect(primary.text()).toContain('Starting…')
    expect(primary.attributes('disabled')).toBeDefined()
    expect(primary.attributes('aria-busy')).toBe('true')

    const stop = button(wrapper, 'Stop Q4_K_M')
    expect(stop.attributes('disabled')).toBeUndefined()
    await stop.trigger('click')
    expect(wrapper.emitted('stop')?.[0]).toEqual(['m1'])
    wrapper.unmount()
  })

  it('loading reported by the server: Starting… and an always-available Stop', async () => {
    const wrapper = mountActions({ is_active: true, status: 'loading' })
    expect(button(wrapper, 'Q4_K_M is starting').text()).toContain('Starting…')
    expect(button(wrapper, 'Stop Q4_K_M')).toBeTruthy()
    wrapper.unmount()
  })

  it('running: Connect primary and an icon-only Stop with an accessible label', async () => {
    const wrapper = mountActions({ is_active: true, status: 'ready' })
    const connect = button(wrapper, 'Connect Q4_K_M')
    expect(connect.text()).toBe('Connect')
    await connect.trigger('click')
    expect(wrapper.emitted('connect')?.[0]?.[0]).toMatchObject({ id: 'm1' })

    const stop = button(wrapper, 'Stop Q4_K_M')
    expect(stop.text()).toBe('')
    expect(stop.attributes('aria-label')).toBe('Stop Q4_K_M')
    wrapper.unmount()
  })

  it('error: Retry start primary with truncated inline last_error and full text', async () => {
    const message = 'CUDA out of memory while loading 7B weights on the selected device'
    const wrapper = mountActions({ status: 'failed', last_error: message })
    const retry = button(wrapper, 'Retry start for Q4_K_M')
    expect(retry.text()).toBe('Retry start')

    const error = wrapper.get('.model-actions__error')
    expect(error.attributes('title')).toBe(message)
    expect(error.attributes('role')).toBe('alert')
    expect(error.text()).toBe(message)
    expect(retry.attributes('aria-describedby')).toBe(error.attributes('id'))

    await retry.trigger('click')
    expect(wrapper.emitted('start')?.[0]).toEqual(['m1'])
    wrapper.unmount()
  })

  it.each([
    ['stale', { runtime_quality: 'stale', status: 'ready', is_active: false }],
    ['unknown', { runtime_quality: 'unreachable', status: null, is_active: false }],
  ])('%s: Refresh status is the primary action', async (_label, model) => {
    const wrapper = mountActions(model)
    const refresh = button(wrapper, 'Refresh status for Q4_K_M')
    expect(refresh.text()).toBe('Refresh status')
    await refresh.trigger('click')
    expect(wrapper.emitted('refresh')?.[0]).toEqual(['m1'])
    expect(wrapper.findAll('button[aria-label="Start Q4_K_M"]')).toHaveLength(0)
    wrapper.unmount()
  })

  it('audio: Open Audio is primary while running and start/stop still apply', async () => {
    const wrapper = mountActions({ format: AUDIO_PACKAGE, is_active: true, status: 'ready' })
    expect(wrapper.findAll('button[aria-label="Connect Q4_K_M"]')).toHaveLength(0)
    const audio = button(wrapper, 'Open audio for Q4_K_M')
    expect(audio.text()).toBe('Open Audio')
    await audio.trigger('click')
    expect(wrapper.emitted('audio')?.[0]).toEqual(['m1'])
    expect(button(wrapper, 'Stop Q4_K_M')).toBeTruthy()
    wrapper.unmount()
  })

  it('audio: a stopped audio model keeps its audio action alongside Start', async () => {
    const wrapper = mountActions({ format: AUDIO_PACKAGE })
    expect(button(wrapper, 'Start Q4_K_M')).toBeTruthy()
    await button(wrapper, 'Open audio for Q4_K_M').trigger('click')
    expect(wrapper.emitted('audio')?.[0]).toEqual(['m1'])
    wrapper.unmount()
  })

  it('stopping: shows a disabled Stopping… indicator instead of a duplicate Stop', async () => {
    const wrapper = mountActions({ is_active: true, status: 'ready' }, { isStopping: true })
    const primary = button(wrapper, 'Q4_K_M is stopping')
    expect(primary.text()).toContain('Stopping…')
    expect(primary.attributes('disabled')).toBeDefined()
    expect(wrapper.findAll('button[aria-label="Stop Q4_K_M"]')).toHaveLength(0)
    wrapper.unmount()
  })

  it('keeps focus on the transition action: Start → cancel Stop → Connect', async () => {
    const wrapper = mountActions()
    await button(wrapper, 'Start Q4_K_M').trigger('click')
    await wrapper.setProps({ isStarting: true })
    await flushPromises()
    expect(document.activeElement).toBe(button(wrapper, 'Stop Q4_K_M').element)

    await wrapper.setProps({
      isStarting: false,
      model: { id: 'm1', quantization: 'Q4_K_M', is_active: true, status: 'ready' },
    })
    await flushPromises()
    expect(document.activeElement).toBe(button(wrapper, 'Connect Q4_K_M').element)
    wrapper.unmount()
  })
})
