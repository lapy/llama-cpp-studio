import { describe, expect, it } from 'vitest'
import { engineCardCta, engineCardOrder, engineMatchesFilters, runnableEngineIds } from './engineReadiness'

describe('engine readiness', () => {
  it('treats only runnable descriptors as ready, including Unsloth', () => {
    const descriptors = [
      { id: 'llama_cpp', runnable: false, installed_versions: 2, enabled: true },
      { id: 'unsloth_llama', runnable: true, installed_versions: 1, enabled: true },
    ]
    expect(runnableEngineIds(descriptors)).toEqual(['unsloth_llama'])
  })

  it('orders runnable engines before installed and missing ones', () => {
    expect(engineCardOrder({ runnable: true, installed_versions: 1 })).toBe(0)
    expect(engineCardOrder({ runnable: false, installed_versions: 2 })).toBe(1)
    expect(engineCardOrder({ runnable: false, installed_versions: 0 })).toBe(2)
  })

  it('labels an active version Manage and an inactive install Activate', () => {
    expect(engineCardCta({ active_version: 'b1', installed_versions: 1 })).toBe('Manage')
    expect(engineCardCta({ active_version: null, installed_versions: 1 })).toBe('Activate')
    expect(engineCardCta({ installed_versions: 0 })).toBe('Install')
  })

  it('narrows by task and hardware without hiding engines when the filter is open', () => {
    const audio = { id: 'audio_cpp', tasks: ['asr'], artifact_formats: ['gguf'] }
    const llama = { id: 'llama_cpp', tasks: ['text-generation', 'embeddings'], artifact_formats: ['gguf'], supports_embeddings: true }
    const vllm = { id: 'vllm', tasks: ['text-generation'], artifact_formats: ['safetensors'], supports_embeddings: true }
    expect(engineMatchesFilters(vllm, { task: 'all', hardware: 'all' })).toBe(true)
    expect(engineMatchesFilters(audio, { task: 'text' })).toBe(false)
    expect(engineMatchesFilters(llama, { task: 'embeddings' })).toBe(true)
    expect(engineMatchesFilters(vllm, { hardware: 'cpu' })).toBe(false)
    expect(engineMatchesFilters({ id: 'sglang_v100' }, { hardware: 'sm70' })).toBe(true)
  })
})
