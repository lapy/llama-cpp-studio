import { describe, expect, it } from 'vitest'
import { isAudioPackage } from '../audioModel'

describe('isAudioPackage', () => {
  it('recognises audio packages by engine, format, or config engine', () => {
    expect(isAudioPackage({ config: { engine: 'audio_cpp' } })).toBe(true)
    expect(isAudioPackage({ engine: 'audio_cpp' })).toBe(true)
    expect(isAudioPackage({ format: 'audio_cpp' })).toBe(true)
    expect(isAudioPackage({ artifact: { format: 'audio_cpp' } })).toBe(true)
  })

  it('does not classify non-audio models', () => {
    expect(isAudioPackage({ config: { engine: 'llama_cpp' }, format: 'gguf' })).toBe(false)
    expect(isAudioPackage({ engine: 'vllm', format: 'safetensors' })).toBe(false)
    expect(isAudioPackage({})).toBe(false)
    expect(isAudioPackage(null)).toBe(false)
    expect(isAudioPackage(undefined)).toBe(false)
  })
})
