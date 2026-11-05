import { describe, expect, it, vi, beforeEach } from 'vitest'
import { flushPromises, mount } from '@vue/test-utils'
import axios from 'axios'
import ConnectDialog from './ConnectDialog.vue'

const toastAdd = vi.fn()
const confirmRequire = vi.fn()

vi.mock('axios', () => ({
  default: {
    get: vi.fn(),
    put: vi.fn(),
    post: vi.fn(),
    delete: vi.fn(),
    defaults: {},
    interceptors: {
      request: { use: vi.fn() },
      response: { use: vi.fn() },
    },
  },
}))

vi.mock('primevue/useconfirm', () => ({ useConfirm: () => ({ require: confirmRequire }) }))

vi.mock('primevue/usetoast', () => ({
  useToast: () => ({ add: toastAdd }),
}))

function failure(data, status = 500) {
  return Object.assign(new Error('request failed'), { response: { status, data } })
}

function mountDialog() {
  return mount(ConnectDialog, {
    props: {
      visible: true,
      model: { id: 'model-1', config: {} },
      proxyPort: 2000,
      publicInferenceUrl: 'http://old.example',
    },
    global: {
      stubs: {
        Dialog: { name: 'Dialog', props: ['visible'], emits: ['update:visible'], template: '<div><slot /></div>' },
        Button: { props: ['label'], template: '<button>{{ label }}<slot /></button>' },
      },
    },
  })
}

async function editUrl(wrapper, value) {
  const input = wrapper.get('#connect-public-url')
  await input.setValue(value)
  await wrapper.findAll('button').find((button) => button.text() === 'Save public URL').trigger('click')
  await flushPromises()
}

describe('ConnectDialog persistence', () => {
  it('requires a deliberate save and does not persist an input blur', async () => {
    const wrapper = mountDialog()
    await wrapper.get('#connect-public-url').setValue('http://preview.example')
    await flushPromises()
    expect(axios.put).not.toHaveBeenCalled()
    wrapper.unmount()
  })

  it('keeps key loading failures distinct from an empty list and access policy', async () => {
    axios.get.mockImplementation((path) => path === '/api/proxy'
      ? Promise.resolve({ data: { settings: { allow_anonymous: false } } })
      : Promise.reject(new Error('offline')))
    const wrapper = mountDialog()
    await flushPromises()
    expect(wrapper.text()).toContain('Could not load access keys')
    expect(wrapper.text()).toContain('An access key is required')
    expect(wrapper.text()).not.toContain('No access keys configured')
    wrapper.unmount()
  })

  it('only revokes a key after confirmation and protects an unacknowledged secret', async () => {
    axios.get.mockResolvedValue({ data: { keys: [{ id: 'desktop', name: 'Desktop' }] } })
    axios.delete.mockResolvedValue({ data: {} })
    axios.post.mockResolvedValue({ data: { key: 'sk-once' } })
    const wrapper = mountDialog()
    await flushPromises()
    await wrapper.findAll('button').find((button) => button.text() === 'Revoke').trigger('click')
    expect(axios.delete).not.toHaveBeenCalled()
    const revocation = confirmRequire.mock.calls.at(-1)[0]
    expect(revocation.header).toContain('Desktop')
    await revocation.accept()
    expect(axios.delete).toHaveBeenCalledWith('/api/inference/keys/desktop')
    await wrapper.get('input[aria-label="New access key name"]').setValue('New desktop')
    await wrapper.findAll('button').find((button) => button.text() === 'Create key').trigger('click')
    await flushPromises()
    wrapper.findComponent({ name: 'Dialog' }).vm.$emit('update:visible', false)
    expect(wrapper.emitted('update:visible')).toBeUndefined()
    const closure = confirmRequire.mock.calls.at(-1)[0]
    expect(closure.header).toContain('Close without saving')
    closure.reject()
    await wrapper.findAll('button').find((button) => button.text() === 'I saved this key').trigger('click')
    wrapper.findComponent({ name: 'Dialog' }).vm.$emit('update:visible', false)
    expect(wrapper.emitted('update:visible')).toEqual([[false]])
    wrapper.unmount()
  })

  beforeEach(() => {
    toastAdd.mockReset()
    confirmRequire.mockReset()
    axios.delete.mockReset()
    vi.mocked(axios.put).mockReset()
    vi.mocked(axios.get).mockReset()
  })

  it('keeps the typed URL and waits for a deliberate retry when the queue is full', async () => {
    vi.mocked(axios.put).mockRejectedValue(failure({
      code: 'STORE_QUEUE_FULL',
      committed: false,
    }, 503))
    const wrapper = mountDialog()
    await editUrl(wrapper, 'http://new.example/path')
    expect(wrapper.get('#connect-public-url').element.value).toBe('http://new.example/path')
    expect(wrapper.text()).toContain('Your edits are still here')
    expect(toastAdd).not.toHaveBeenCalledWith(expect.objectContaining({ severity: 'success' }))
    expect(axios.put).toHaveBeenCalledTimes(1)

    vi.mocked(axios.put).mockResolvedValue({ data: { public_inference_url: 'http://new.example/path' } })
    const retry = wrapper.findAll('button').find((button) => button.text() === 'Try again')
    await retry.trigger('click')
    await flushPromises()
    expect(axios.put).toHaveBeenCalledTimes(2)
  })

  it('does not show a saved URL when the write was not stored', async () => {
    vi.mocked(axios.put).mockRejectedValue(failure({
      code: 'STORE_WRITE_FAILED',
      committed: false,
    }))
    const wrapper = mountDialog()
    await editUrl(wrapper, 'http://new.example')
    expect(wrapper.text()).toContain('Not saved')
    expect(wrapper.get('#connect-public-url').element.value).toBe('http://new.example')
    expect(wrapper.emitted('update:publicInferenceUrl')).toBeUndefined()
  })

  it('refreshes after a known replacement and does not save again until asked', async () => {
    vi.mocked(axios.put).mockRejectedValue(failure({
      code: 'STORE_WRITE_FAILED',
      committed: true,
    }))
    vi.mocked(axios.get).mockResolvedValue({
      data: { public_inference_url: 'http://stored.example' },
    })
    const wrapper = mountDialog()
    await editUrl(wrapper, 'http://new.example')
    expect(axios.get).toHaveBeenCalledWith('/api/settings/inference')
    expect(axios.put).toHaveBeenCalledTimes(1)
    expect(wrapper.get('#connect-public-url').element.value).toBe('http://stored.example')
    expect(wrapper.text()).toContain('acknowledgement failed')
    expect(wrapper.findAll('button').some((button) => button.text() === 'Try again')).toBe(false)
  })

  it('keeps the typed URL when reloading a replaced document fails', async () => {
    vi.mocked(axios.put).mockRejectedValue(failure({
      code: 'STORE_WRITE_FAILED',
      committed: true,
    }))
    vi.mocked(axios.get).mockRejectedValue(new Error('offline'))
    const wrapper = mountDialog()
    await editUrl(wrapper, 'http://new.example')
    expect(wrapper.get('#connect-public-url').element.value).toBe('http://new.example')
    expect(wrapper.text()).toContain('Your edits are still here')
    expect(wrapper.text()).toContain('Refresh')
    expect(axios.put).toHaveBeenCalledTimes(1)
  })
})
