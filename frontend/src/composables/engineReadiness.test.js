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
    expect(engineMatchesFilters({ id: 'sglang_v100', gpu_archs: ['sm70'] }, { hardware: 'sm70' })).toBe(true)
  })

  it('derives audio/text/cpu membership from descriptor data, not engine ids', () => {
    // audio.cpp: audio family via declared tasks; CPU support declared.
    const audio = {
      id: 'audio_cpp',
      tasks: ['asr'],
      artifact_formats: ['safetensors', 'mixed', 'original', 'builtin'],
      cpu_supported: true,
    }
    // vLLM: safetensors only, no CPU support.
    const vllm = {
      id: 'vllm',
      tasks: ['text-generation'],
      artifact_formats: ['safetensors'],
      cpu_supported: false,
    }
    // A YAML-onboarded engine that declares neither gguf nor cpu_supported.
    const fixture = { id: 'yamlcraft', tasks: ['tts'], artifact_formats: ['safetensors'] }
    // An unknown audio-family engine matches by its declared tasks.
    const custom = { id: 'custom_audio', tasks: ['vc'], artifact_formats: ['safetensors'] }

    expect(engineMatchesFilters(audio, { task: 'audio' })).toBe(true)
    expect(engineMatchesFilters(audio, { task: 'text' })).toBe(false)
    expect(engineMatchesFilters(audio, { hardware: 'cpu' })).toBe(true)
    expect(engineMatchesFilters(vllm, { hardware: 'cpu' })).toBe(false)
    expect(engineMatchesFilters(fixture, { hardware: 'cpu' })).toBe(false)
    expect(engineMatchesFilters(custom, { task: 'audio' })).toBe(true)
  })

  it('matches the sm70 filter from the declared gpu_archs, not engine ids', () => {
    const v100 = { id: 'sglang_v100', artifact_formats: ['safetensors'], gpu_archs: ['sm70'] }
    const onecat = { id: '1cat_vllm', artifact_formats: ['safetensors'], gpu_archs: ['sm70'] }
    const vllm = { id: 'vllm', artifact_formats: ['safetensors'], gpu_archs: [] }
    const llama = { id: 'llama_cpp', artifact_formats: ['gguf'] }
    expect(engineMatchesFilters(v100, { hardware: 'sm70' })).toBe(true)
    expect(engineMatchesFilters(onecat, { hardware: 'sm70' })).toBe(true)
    expect(engineMatchesFilters(vllm, { hardware: 'sm70' })).toBe(false)
    expect(engineMatchesFilters(llama, { hardware: 'sm70' })).toBe(false)
    // An engine with no declaration never matches the sm70 filter.
    expect(engineMatchesFilters({ id: 'custom' }, { hardware: 'sm70' })).toBe(false)
  })
})
