import { beforeEach, describe, expect, it, vi } from 'vitest'
import { mount, flushPromises } from '@vue/test-utils'

const toastAdd = vi.fn()
const push = vi.fn()
const replace = vi.fn()
const fetchModels = vi.fn().mockResolvedValue(undefined)
const fetchSystemStatus = vi.fn().mockResolvedValue(undefined)
const getModelConfig = vi.fn().mockResolvedValue({
  engine: 'audio_cpp',
  family: 'qwen3-asr',
  task: 'asr',
  model_alias: 'asr-demo',
  transcription_defaults: { language: 'en' },
})
const listReferenceAudio = vi.fn().mockResolvedValue([])

vi.mock('vue-router', () => ({
  useRoute: () => ({ query: {} }),
  useRouter: () => ({ push, replace }),
}))

vi.mock('primevue/usetoast', () => ({
  useToast: () => ({ add: toastAdd }),
}))

vi.mock('@/stores/models', () => ({
  useModelStore: () => ({
    allQuantizations: [
      {
        id: 'audio/asr',
        display_name: 'ASR Demo',
        format: 'audio_cpp',
        engine: 'audio_cpp',
        is_active: true,
        config: { engine: 'audio_cpp' },
      },
    ],
    fetchModels,
    getModelConfig,
    listReferenceAudio: (...args) => listReferenceAudio(...args),
    startModel: vi.fn(),
  }),
}))

vi.mock('@/stores/engines', () => ({
  useEnginesStore: () => ({
    systemStatus: { proxy_status: { healthy: true, port: 2000 } },
    fetchSystemStatus,
  }),
}))

import AudioWorkspace from './AudioWorkspace.vue'

describe('AudioWorkspace', () => {
  beforeEach(() => {
    fetchModels.mockReset()
    fetchSystemStatus.mockReset()
    getModelConfig.mockReset()
    listReferenceAudio.mockReset()
    fetchModels.mockResolvedValue(undefined)
    fetchSystemStatus.mockResolvedValue(undefined)
    listReferenceAudio.mockResolvedValue([])
    getModelConfig.mockResolvedValue({
      engine: 'audio_cpp',
      family: 'qwen3-asr',
      task: 'asr',
      model_alias: 'asr-demo',
      transcription_defaults: { language: 'en' },
    })
  })

  it('renders Audio shell with Transcribe for an ASR model', async () => {
    const wrapper = mount(AudioWorkspace, {
      global: {
        directives: { tooltip: () => {} },
        stubs: {
          Button: true,
          Tag: true,
          Dropdown: true,
          InputText: true,
          Textarea: true,
          Message: true,
          LoadingState: true,
          PageHeader: {
            props: ['title'],
            template: '<header><h1>{{ title }}</h1><slot name="meta" /><slot name="actions" /></header>',
          },
          EmptyState: true,
          RouterLink: true,
        },
      },
    })

    await flushPromises()

    expect(wrapper.text()).toContain('Audio')
    expect(fetchModels).toHaveBeenCalled()
    expect(fetchSystemStatus).toHaveBeenCalled()
    expect(getModelConfig).toHaveBeenCalledWith('audio/asr')
    const tabLabels = wrapper.findAll('.config-section-tab').map((n) => n.text())
    expect(tabLabels.some((t) => t.includes('Transcribe'))).toBe(true)
    expect(tabLabels.some((t) => t.includes('Speech'))).toBe(false)
    expect(tabLabels.some((t) => t.includes('Music'))).toBe(false)
    expect(wrapper.text()).toContain('Transcribe')
  })
})
