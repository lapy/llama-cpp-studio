import { beforeEach, describe, expect, it, vi } from 'vitest'
import { flushPromises, mount } from '@vue/test-utils'
import { reactive } from 'vue'

const toastAdd = vi.fn()

vi.mock('primevue/usetoast', () => ({ useToast: () => ({ add: toastAdd }) }))

const modelStore = reactive({
  hasHuggingfaceToken: false,
  huggingfaceToken: '',
  fetchHuggingfaceTokenStatus: vi.fn(),
  setHuggingfaceToken: vi.fn(),
  clearHuggingfaceToken: vi.fn(),
})

vi.mock('@/stores/models', () => ({ useModelStore: () => modelStore }))

import HuggingFaceTokenDialog from './HuggingFaceTokenDialog.vue'

const PasswordStub = {
  props: ['modelValue', 'inputId'],
  emits: ['update:modelValue'],
  template:
    '<input :id="inputId" :value="modelValue" @input="$emit(`update:modelValue`, $event.target.value)" />',
}

const DialogStub = {
  props: ['visible', 'header'],
  emits: ['update:visible'],
  template:
    '<div v-if="visible" class="dialog-stub" :data-header="header"><slot /><slot name="footer" /></div>',
}

function mountDialog() {
  return mount(HuggingFaceTokenDialog, {
    props: { visible: true },
    global: { stubs: { Dialog: DialogStub, Password: PasswordStub } },
  })
}

beforeEach(() => {
  toastAdd.mockReset()
  modelStore.hasHuggingfaceToken = false
  modelStore.setHuggingfaceToken.mockReset().mockResolvedValue(undefined)
  modelStore.clearHuggingfaceToken.mockReset().mockResolvedValue(undefined)
})

describe('HuggingFaceTokenDialog', () => {
  it('uses the shared dialog structure and saves a token', async () => {
    const wrapper = mountDialog()

    expect(wrapper.get('.dialog-stub').attributes('data-header')).toBe('Hugging Face token')
    expect(wrapper.get('.dialog-stack').exists()).toBe(true)
    expect(wrapper.get('.dialog-actions').text()).toContain('Save token')

    await wrapper.get('#hf-token').setValue('hf_test_token')
    await wrapper
      .findAll('button')
      .find((button) => button.text() === 'Save token')
      .trigger('click')
    await flushPromises()

    expect(modelStore.setHuggingfaceToken).toHaveBeenCalledWith('hf_test_token')
    expect(wrapper.emitted('update:visible')).toEqual([[false]])
    wrapper.unmount()
  })

  it('shows the current token state and clears it without closing the dialog', async () => {
    modelStore.hasHuggingfaceToken = true
    modelStore.huggingfaceToken = 'hf_existing'
    const wrapper = mountDialog()

    expect(wrapper.text()).toContain('Token set: hf_existing')
    await wrapper
      .findAll('button')
      .find((button) => button.text() === 'Clear token')
      .trigger('click')
    await flushPromises()

    expect(modelStore.clearHuggingfaceToken).toHaveBeenCalled()
    expect(wrapper.emitted('update:visible')).toBeUndefined()
    wrapper.unmount()
  })
})
