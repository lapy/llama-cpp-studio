import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { enableAutoUnmount, flushPromises, mount } from '@vue/test-utils'
import axios from 'axios'
import ConfigRestore from './ConfigRestore.vue'

vi.mock('axios', () => ({ default: { get: vi.fn(), post: vi.fn() } }))
enableAutoUnmount(afterEach)
const plan = { applicable: true, plan_id: 'plan-a', items: [], notice: 'Saved only' }
function deferred() {
  let resolve
  const promise = new Promise(r => { resolve = r })
  return { promise, resolve }
}
function button(wrapper, label) {
  return wrapper.findAll('button').find(b => b.text() === label)
}
async function selectBackup(wrapper) {
  const input = wrapper.get('input[type=file]')
  Object.defineProperty(input.element, 'files', {
    configurable: true,
    value: [{ size: 2, text: async () => '{}' }],
  })
  await input.trigger('change')
  await flushPromises()
}
beforeEach(() => {
  axios.get.mockReset().mockResolvedValue({ data: [] })
  axios.post.mockReset().mockResolvedValue({ data: plan })
})
describe('configuration restore recovery', () => {
  it('admits one apply while pending and invalidates its completed preview', async () => {
    const pending = deferred()
    const wrapper = mount(ConfigRestore)
    await selectBackup(wrapper)
    axios.post.mockImplementation((url) => url.endsWith('/apply') ? pending.promise : Promise.resolve({ data: plan }))
    const restore = button(wrapper, 'Restore saved settings')
    await restore.trigger('click')
    await restore.trigger('click')
    expect(axios.post.mock.calls.filter(([url]) => url.endsWith('/apply'))).toHaveLength(1)
    expect(restore.attributes('disabled')).toBeDefined()
    expect(wrapper.get('input').attributes('disabled')).toBeDefined()
    pending.resolve({ data: { outcome: 'completed' } })
    await flushPromises()
    expect(restore.attributes('disabled')).toBeDefined()
    expect(wrapper.text()).toContain('Restore completed')
  })
  it('does not let a new file bypass uncertainty and requires a fresh preview after idle reconciliation', async () => {
    const wrapper = mount(ConfigRestore)
    await selectBackup(wrapper)
    axios.post.mockRejectedValueOnce(new Error('connection lost'))
    await button(wrapper, 'Restore saved settings').trigger('click')
    await flushPromises()
    expect(button(wrapper, 'Restore saved settings')).toBeUndefined()
    expect(wrapper.get('input').attributes('disabled')).toBeDefined()
    expect(button(wrapper, 'Cancel restore').attributes('disabled')).toBeDefined()
    axios.post.mockResolvedValueOnce({ data: { outcome: 'idle' } })
    await button(wrapper, 'Reconcile').trigger('click')
    await flushPromises()
    expect(button(wrapper, 'Restore saved settings').attributes('disabled')).toBeDefined()
    expect(wrapper.text()).toContain('No restore is pending')
    expect(axios.post.mock.calls.filter(([url]) => url.endsWith('/apply'))).toHaveLength(1)
  })
  it('locks mappings while previewing so a late result cannot authorize different decisions', async () => {
    const wrapper = mount(ConfigRestore)
    await selectBackup(wrapper)
    const pending = deferred()
    axios.post.mockReturnValueOnce(pending.promise)
    await button(wrapper, 'Update preview').trigger('click')
    expect(button(wrapper, 'Cancel restore').attributes('disabled')).toBeDefined()
    expect(wrapper.get('input').attributes('disabled')).toBeDefined()
    pending.resolve({ data: plan })
    await flushPromises()
    expect(button(wrapper, 'Restore saved settings').attributes('disabled')).toBeUndefined()
  })
})
