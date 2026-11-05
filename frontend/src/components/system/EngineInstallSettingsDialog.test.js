import { describe, expect, it } from 'vitest'
import { mount } from '@vue/test-utils'

import EngineInstallSettingsDialog from './EngineInstallSettingsDialog.vue'

const baseProps = {
  visible: true,
  title: 'Build settings',
  note: 'Saved defaults.',
  versionKey: 'pip_version',
  versionLabel: 'Version',
  form: {
    source_repo: 'https://example.test/repo.git',
    source_branch: 'main',
    pip_version: '',
    compat_sm70: false,
  },
}

describe('EngineInstallSettingsDialog compatibility options', () => {
  it('renders declared compatibility choices only when compatibility is shown', () => {
    const wrapper = mount(EngineInstallSettingsDialog, {
      props: {
        ...baseProps,
        compatibility: {
          show: true,
          note: 'Detected SM70 hardware.',
          options: [
            { key: 'compat_sm70', label: 'Use CUDA 12 PyTorch', hint: 'sm70 kernels' },
          ],
        },
      },
      global: {
        stubs: {
          Dialog: { template: '<div><slot /></div>' },
          Checkbox: { template: '<input type="checkbox" />' },
          InputText: { template: '<input />' },
          Button: { template: '<button />' },
        },
      },
    })

    expect(wrapper.text()).toContain('Compatibility')
    expect(wrapper.text()).toContain('Detected SM70 hardware.')
    expect(wrapper.text()).toContain('Use CUDA 12 PyTorch')
  })

  it('does not render the compatibility block when hardware is not relevant', () => {
    const wrapper = mount(EngineInstallSettingsDialog, {
      props: {
        ...baseProps,
        compatibility: { show: false, options: [] },
      },
      global: {
        stubs: {
          Dialog: { template: '<div><slot /></div>' },
          Checkbox: { template: '<input type="checkbox" />' },
          InputText: { template: '<input />' },
          Button: { template: '<button />' },
        },
      },
    })

    expect(wrapper.text()).not.toContain('Compatibility')
  })
})
