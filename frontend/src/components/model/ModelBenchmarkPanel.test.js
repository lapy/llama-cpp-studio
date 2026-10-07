import { flushPromises, mount } from '@vue/test-utils'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import ModelBenchmarkPanel from './ModelBenchmarkPanel.vue'

const { listModelBenchmarks, runModelBenchmark } = vi.hoisted(() => ({
  listModelBenchmarks: vi.fn(),
  runModelBenchmark: vi.fn(),
}))

vi.mock('@/api/configuration', () => ({
  listModelBenchmarks,
  runModelBenchmark,
}))

const result = {
  id: 'benchmark-1',
  created_at: 1_799_000_000,
  model_id: 'demo',
  config_revision: 'revision-1',
  config_fingerprint: 'abc123',
  time_to_first_token_ms: 125,
  total_seconds: 2,
  completion_tokens: 20,
  tokens_per_second: 10,
  peak_observed_gpu_memory_bytes: 2 * 1024 ** 3,
  output_preview: 'A clear blue sky.',
}

describe('ModelBenchmarkPanel', () => {
  beforeEach(() => {
    listModelBenchmarks.mockReset().mockResolvedValue([])
    runModelBenchmark.mockReset().mockResolvedValue(result)
  })

  it('requires a running model before invoking inference', async () => {
    const wrapper = mount(ModelBenchmarkPanel, {
      props: { modelId: 'demo', running: false },
    })
    await flushPromises()

    const runButton = wrapper
      .findAll('button')
      .find((button) => button.text().includes('Start the model'))
    expect(runButton.attributes('disabled')).toBeDefined()
    await runButton.trigger('click')
    expect(runModelBenchmark).not.toHaveBeenCalled()
  })

  it('runs a benchmark and renders locally stored measurements', async () => {
    const wrapper = mount(ModelBenchmarkPanel, {
      props: { modelId: 'demo', running: true },
    })
    await flushPromises()

    const runButton = wrapper.findAll('button').find((button) => button.text() === 'Run benchmark')
    await runButton.trigger('click')
    await flushPromises()

    expect(runModelBenchmark).toHaveBeenCalledWith(
      'demo',
      'Reply with a short description of the sky.',
      64,
    )
    expect(wrapper.text()).toContain('10.00 tok/s')
    expect(wrapper.text()).toContain('2.00 GiB total GPU use')
    expect(wrapper.text()).toContain('A clear blue sky.')
    expect(wrapper.get('[role="status"]').text()).toContain('completed')
  })
})
