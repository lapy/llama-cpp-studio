/**
 * R101 pin: the audio CONFIG PANEL family is descriptor data, not an engine-id
 * comparison scattered across ModelConfig.
 */

import { describe, expect, it } from 'vitest'
import { ENGINE_DESCRIPTORS, isAudioEngineId } from '../registry'

describe('isAudioEngineId', () => {
  it('declares the audio panel on the audio engine only', () => {
    expect(ENGINE_DESCRIPTORS.audio_cpp.configPanel).toBe('audio')
    expect(isAudioEngineId('audio_cpp')).toBe(true)
  })

  it('is false for the text/venv/native engines and for unknown ids', () => {
    for (const engineId of ['llama_cpp', 'ik_llama', 'unsloth_llama', 'lmdeploy', 'vllm']) {
      expect(isAudioEngineId(engineId), engineId).toBe(false)
    }
    expect(isAudioEngineId('not-an-engine')).toBe(false)
    expect(isAudioEngineId(undefined)).toBe(false)
  })

  it('every descriptor claiming the audio panel is the audio engine', () => {
    const panels = Object.entries(ENGINE_DESCRIPTORS)
      .filter(([, descriptor]) => descriptor.configPanel === 'audio')
      .map(([engineId]) => engineId)
    expect(panels).toEqual(['audio_cpp'])
  })
})
