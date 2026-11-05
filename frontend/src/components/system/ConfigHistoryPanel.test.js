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

  it('compares a revision and selectively restores the changed model', async () => {
    const wrapper = mount(ConfigHistoryPanel)
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

  it('shows a stable error when history cannot be loaded', async () => {
    listConfigurationHistory.mockRejectedValueOnce(new Error('private filesystem detail'))
    const wrapper = mount(ConfigHistoryPanel)
    await flushPromises()

    expect(wrapper.get('[role="alert"]').text()).toBe('Configuration history could not be loaded.')
  })
})
