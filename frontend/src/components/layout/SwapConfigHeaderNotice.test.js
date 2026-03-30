import { describe, it, expect, beforeEach, vi } from 'vitest'
import { mount, flushPromises } from '@vue/test-utils'
import { reactive } from 'vue'

import SwapConfigHeaderNotice from './SwapConfigHeaderNotice.vue'

const toastAdd = vi.fn()
const fetchSwapConfigPending = vi.fn()
const fetchSwapConfigStale = vi.fn()
const applySwapConfig = vi.fn()

const storeState = reactive({
  swapConfigStale: { applicable: true, stale: true },
  swapConfigPending: {
    applicable: true,
    pending: true,
    changes: ['Update model «q»'],
    reason: null,
  },
})

vi.mock('primevue/usetoast', () => ({
  useToast: () => ({ add: toastAdd }),
}))

vi.mock('@/stores/engines', () => ({
  useEnginesStore: () => ({
    get swapConfigStale() {
      return storeState.swapConfigStale
    },
    get swapConfigPending() {
      return storeState.swapConfigPending
    },
    fetchSwapConfigPending,
    fetchSwapConfigStale,
    applySwapConfig,
  }),
}))

const ButtonStub = {
  props: ['label', 'icon', 'severity', 'outlined', 'loading', 'disabled'],
  emits: ['click'],
  template: '<button :data-label="label" :disabled="disabled" @click="$emit(`click`)">{{ label }}</button>',
}

const DialogStub = {
  props: ['visible', 'header'],
  emits: ['update:visible', 'show'],
  watch: {
    visible(value) {
      if (value) this.$emit('show')
    },
  },
  template: `
    <div v-if="visible">
      <div class="dialog-header">{{ header }}</div>
      <div class="dialog-body"><slot /></div>
      <div class="dialog-footer"><slot name="footer" /></div>
    </div>
  `,
}

function mountNotice() {
  return mount(SwapConfigHeaderNotice, {
    global: {
      directives: {
        tooltip: () => {},
      },
      stubs: {
        Button: ButtonStub,
        Dialog: DialogStub,
        Message: { template: '<div><slot /></div>' },
      },
    },
  })
}

describe('SwapConfigHeaderNotice', () => {
  beforeEach(() => {
    toastAdd.mockReset()
    fetchSwapConfigPending.mockReset()
    fetchSwapConfigStale.mockReset()
    applySwapConfig.mockReset()
    storeState.swapConfigStale = { applicable: true, stale: true }
    storeState.swapConfigPending = {
      applicable: true,
      pending: true,
      changes: ['Update model «q»'],
      reason: null,
    }
    fetchSwapConfigPending.mockResolvedValue(storeState.swapConfigPending)
    fetchSwapConfigStale.mockResolvedValue(storeState.swapConfigStale)
    applySwapConfig.mockResolvedValue({ message: 'ok' })
  })

  it('opens the dialog and loads pending configuration details', async () => {
    const wrapper = mountNotice()

    await wrapper.get('button').trigger('click')
    await flushPromises()

    expect(fetchSwapConfigPending).toHaveBeenCalledTimes(1)
    expect(fetchSwapConfigStale).toHaveBeenCalledTimes(1)
    expect(wrapper.text()).toContain('Summary of differences')
    expect(wrapper.text()).toContain('Update model «q»')
  })

  it('applies configuration through the store and shows a success toast', async () => {
    const wrapper = mountNotice()

    await wrapper.get('button').trigger('click')
    await flushPromises()
    await wrapper.get('button[data-label="Apply configuration"]').trigger('click')
    await flushPromises()

    expect(applySwapConfig).toHaveBeenCalledTimes(1)
    expect(toastAdd).toHaveBeenCalledWith(
      expect.objectContaining({
        severity: 'success',
        summary: 'Configuration applied',
      }),
    )
  })
})
