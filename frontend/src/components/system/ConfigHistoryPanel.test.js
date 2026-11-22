import { flushPromises, mount } from '@vue/test-utils'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import ConfigHistoryPanel from './ConfigHistoryPanel.vue'

const { listConfigurationHistory, configurationHistoryDiff, restoreConfigurationHistory } =
  vi.hoisted(() => ({
    listConfigurationHistory: vi.fn(),
    configurationHistoryDiff: vi.fn(),
    restoreConfigurationHistory: vi.fn(),
  }))

vi.mock('@/api/configuration', () => ({
  listConfigurationHistory,
  configurationHistoryDiff,
  restoreConfigurationHistory,
}))

const entry = {
  id: 'revision-1',
  created_at: '2026-10-07T10:00:00Z',
  document: 'models.yaml',
  reason: 'before model update',
}

function mountPanel() {
  return mount(ConfigHistoryPanel, {
    global: {
      stubs: {
        Button: {
          props: ['label', 'disabled', 'loading'],
          template: '<button :disabled="disabled">{{ label }}</button>',
        },
        LoadingState: true,
      },
    },
  })
}

describe('ConfigHistoryPanel', () => {
  beforeEach(() => {
    listConfigurationHistory.mockReset().mockResolvedValue([entry])
    configurationHistoryDiff.mockReset().mockResolvedValue({
      ...entry,
      current_revision: 'current-1',
      changes: [{ path: 'models.demo.config.ctx_size', before: 4096, after: 8192 }],
    })
    restoreConfigurationHistory.mockReset().mockResolvedValue({
      outcome: 'completed',
      notice: 'Model settings restored.',
      revision: 'current-2',
    })
  })

  it('renders one page section with a refresh action', async () => {
    const wrapper = mountPanel()
    await flushPromises()

    const section = wrapper.get('#history')
    expect(section.classes()).toContain('page-section')
    expect(section.get('.page-section__title').text()).toBe('Recent changes')
    const refresh = section
      .get('.page-section__actions')
      .findAll('button')
      .find((button) => button.text() === 'Refresh history')
    expect(refresh).toBeTruthy()
    await refresh.trigger('click')
    await flushPromises()
    expect(listConfigurationHistory).toHaveBeenCalledTimes(2)
  })

  it('compares a revision and selectively restores the changed model', async () => {
    const wrapper = mountPanel()
    await flushPromises()

    expect(listConfigurationHistory).toHaveBeenCalledTimes(1)
    await wrapper.get('.history-list button').trigger('click')
    await flushPromises()

    expect(configurationHistoryDiff).toHaveBeenCalledWith('revision-1')
    expect(wrapper.text()).toContain('models.demo.config.ctx_size')

    await wrapper.get('.history-scopes button').trigger('click')
    await flushPromises()

    expect(restoreConfigurationHistory).toHaveBeenCalledWith(
      'revision-1',
      { kind: 'model', item_id: 'demo' },
      'current-1',
    )
    expect(wrapper.get('[role="status"]').text()).toContain('restored')
  })

  it('shows an empty state with a next action when nothing is recorded', async () => {
    listConfigurationHistory.mockResolvedValueOnce([])
    const wrapper = mountPanel()
    await flushPromises()

    expect(wrapper.text()).toContain('No configuration changes recorded yet')
    const scan = wrapper.get('.page-empty').find('button')
    expect(scan.text()).toBe('Refresh history')
    await scan.trigger('click')
    await flushPromises()
    expect(listConfigurationHistory).toHaveBeenCalledTimes(2)
    expect(wrapper.text()).toContain('Model settings')
  })

  it('shows a stable error with a retry action when history cannot be loaded', async () => {
    listConfigurationHistory.mockRejectedValueOnce(new Error('private filesystem detail'))
    const wrapper = mountPanel()
    await flushPromises()

    const banner = wrapper.get('[role="alert"]')
    expect(banner.get('.state-banner__message').text()).toBe(
      'Configuration history could not be loaded.',
    )
    expect(wrapper.text()).not.toContain('private filesystem detail')

    listConfigurationHistory.mockResolvedValueOnce([entry])
    const retry = banner.findAll('button').find((button) => button.text() === 'Retry')
    await retry.trigger('click')
    await flushPromises()
    expect(listConfigurationHistory).toHaveBeenCalledTimes(2)
    expect(wrapper.find('[role="alert"]').exists()).toBe(false)
  })
})
