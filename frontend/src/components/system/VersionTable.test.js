import { describe, it, expect } from 'vitest'
import { mount } from '@vue/test-utils'

import VersionTable from './VersionTable.vue'

function mountTable(versions) {
  return mount(VersionTable, {
    props: {
      versions,
      activating: null,
      syncing: null,
    },
    global: {
      directives: {
        tooltip: () => {},
      },
      stubs: {
        Button: {
          props: ['label', 'disabled'],
          emits: ['click'],
          template: '<button :disabled="disabled" @click="$emit(\'click\')">{{ label || \'btn\' }}</button>',
        },
        Tag: {
          props: ['value', 'severity'],
          template: '<span class="tag" :data-severity="severity">{{ value }}</span>',
        },
      },
    },
  })
}

describe('VersionTable fork labeling', () => {
  it('shows fork badge with warning severity', () => {
    const wrapper = mountTable([
      {
        id: 'llama_cpp:source-main',
        version: 'source-main',
        type: 'fork',
        install_type: 'source',
        is_fork: true,
        source_repo: 'https://github.com/alice/llama.cpp.git',
        source_branch: 'main',
        repository_source: 'llama.cpp',
      },
    ])

    const tags = wrapper.findAll('.tag').map((node) => node.text())
    expect(tags).toContain('fork')
    const forkTag = wrapper.findAll('.tag').find((node) => node.text() === 'fork')
    expect(forkTag.attributes('data-severity')).toBe('warning')
  })

  it('emits sync for fork source installs with a branch', async () => {
    const wrapper = mountTable([
      {
        id: 'llama_cpp:source-main',
        version: 'source-main',
        type: 'fork',
        install_type: 'source',
        is_fork: true,
        source_branch: 'main',
        is_active: true,
      },
    ])

    // Active fork: Sync + Edit CMake + Delete (no Activate). First button is sync.
    const buttons = wrapper.findAll('button')
    expect(buttons.length).toBe(3)
    await buttons[0].trigger('click')
    expect(wrapper.emitted('sync')?.[0]).toEqual(['llama_cpp:source-main'])
  })

  it('emits sync for local branch checkouts', async () => {
    const wrapper = mountTable([
      {
        id: 'audio_cpp:linux-cpu-local',
        version: 'linux-cpu-local',
        type: 'local',
        install_type: 'local',
        source_ref: 'main',
        is_active: true,
      },
    ])

    const buttons = wrapper.findAll('button')
    expect(buttons.length).toBe(3)
    await buttons[0].trigger('click')
    expect(wrapper.emitted('sync')?.[0]).toEqual(['audio_cpp:linux-cpu-local'])
  })

  it('does not offer sync for local installs without a branch', () => {
    const wrapper = mountTable([
      {
        id: 'audio_cpp:linux-cpu-local',
        version: 'linux-cpu-local',
        type: 'local',
        install_type: 'local',
        source_ref: 'a76ec04f620da829e4a53032247369083ba1ad45',
        source_ref_type: 'commit',
        is_active: true,
      },
    ])
    expect(wrapper.findAll('button')).toHaveLength(2)
  })

  it('does not offer sync for release installs', () => {
    const wrapper = mountTable([
      {
        id: '1cat_vllm:v1',
        version: 'v1',
        type: 'release',
        install_type: 'release',
        is_active: false,
      },
    ])
    const labels = wrapper.findAll('button').map((b) => b.text())
    expect(labels).toContain('Activate')
    expect(labels.filter((label) => label === 'Activate')).toHaveLength(1)
    // Activate + Delete only
    expect(wrapper.findAll('button')).toHaveLength(2)
  })

  it('shows failed status and emits retry instead of activate', async () => {
    const wrapper = mountTable([
      {
        id: 'llama_cpp:source-main-fail',
        version: 'source-main-fail',
        type: 'source',
        install_type: 'source',
        source_ref: 'main',
        source_branch: 'main',
        build_status: 'failed',
        build_error: 'cmake failed',
        retryable: true,
        is_active: false,
      },
    ])
    const tags = wrapper.findAll('.tag').map((node) => node.text())
    expect(tags).toContain('Failed')
    const labels = wrapper.findAll('button').map((b) => b.text())
    expect(labels).not.toContain('Activate')
    await wrapper.findAll('button')[0].trigger('click')
    expect(wrapper.emitted('retry')?.[0]).toEqual(['llama_cpp:source-main-fail'])
  })

  it('emits edit-config next to sync for cmake builds', async () => {
    const wrapper = mountTable([
      {
        id: 'llama_cpp:source-main',
        version: 'source-main',
        type: 'source',
        install_type: 'source',
        source_branch: 'main',
        cmake_editable: true,
        is_active: true,
      },
    ])
    const buttons = wrapper.findAll('button')
    expect(buttons.length).toBe(3)
    await buttons[1].trigger('click')
    expect(wrapper.emitted('edit-config')?.[0]).toEqual(['llama_cpp:source-main'])
  })

  it('offers cmake edit for failed retryable builds after retry', async () => {
    const wrapper = mountTable([
      {
        id: 'llama_cpp:source-main-fail',
        version: 'source-main-fail',
        type: 'source',
        install_type: 'source',
        source_branch: 'main',
        build_status: 'failed',
        retryable: true,
        cmake_editable: true,
        is_active: false,
      },
    ])
    const buttons = wrapper.findAll('button')
    expect(buttons.length).toBe(3)
    await buttons[0].trigger('click')
    expect(wrapper.emitted('retry')?.[0]).toEqual(['llama_cpp:source-main-fail'])
    await buttons[1].trigger('click')
    expect(wrapper.emitted('edit-config')?.[0]).toEqual(['llama_cpp:source-main-fail'])
  })

  it('offers cmake edit for commit-only llama builds without sync', async () => {
    const wrapper = mountTable([
      {
        id: 'ik_llama:source-deadbee',
        version: 'source-deadbee',
        type: 'source',
        install_type: 'source',
        source_ref: 'deadbeefdeadbeef',
        source_ref_type: 'commit',
        cmake_editable: true,
        is_active: true,
      },
    ])
    const buttons = wrapper.findAll('button')
    expect(buttons.length).toBe(2)
    await buttons[0].trigger('click')
    expect(wrapper.emitted('edit-config')?.[0]).toEqual(['ik_llama:source-deadbee'])
    expect(wrapper.emitted('sync')).toBeFalsy()
  })

  it('infers cmake edit from audio.cpp id when cmake_editable is omitted', async () => {
    const wrapper = mountTable([
      {
        id: 'audio_cpp:linux-cpu',
        version: 'linux-cpu',
        type: 'source',
        install_type: 'source',
        source_ref: 'a76ec04f620da829e4a53032247369083ba1ad45',
        source_ref_type: 'commit',
        is_active: true,
      },
    ])
    await wrapper.findAll('button')[0].trigger('click')
    expect(wrapper.emitted('edit-config')?.[0]).toEqual(['audio_cpp:linux-cpu'])
  })

  it('shows a CUDA badge from frozen ui or enable_cuda config', () => {
    const fromUi = mountTable([
      {
        id: 'llama_cpp:cuda-ui',
        version: 'cuda-ui',
        type: 'source',
        build_config: { cuda: true },
        is_active: true,
      },
    ])
    expect(fromUi.get('.cuda-badge').text()).toBe('CUDA')

    const fromStored = mountTable([
      {
        id: 'llama_cpp:cuda-stored',
        version: 'cuda-stored',
        type: 'source',
        build_config: { enable_cuda: true },
        is_active: true,
      },
    ])
    expect(fromStored.get('.cuda-badge').text()).toBe('CUDA')

    const fromBackend = mountTable([
      {
        id: 'audio_cpp:hip',
        version: 'hip',
        type: 'source',
        build_config: { backend: 'hip' },
        is_active: true,
      },
    ])
    expect(fromBackend.get('.cuda-badge').text()).toBe('HIP')
  })

  it('hides cmake edit for python engines and orphans', () => {
    const orphan = mountTable([
      {
        id: 'llama_cpp:disk-only',
        version: 'disk-only',
        type: 'broken',
        build_status: 'broken',
        retryable: false,
        orphan: true,
        is_active: false,
      },
    ])
    expect(orphan.findAll('.tag').map((node) => node.text())).toContain('Broken')
    expect(orphan.findAll('button')).toHaveLength(1)
    expect(orphan.emitted('edit-config')).toBeFalsy()

    const python = mountTable([
      {
        id: 'lmdeploy:v1',
        version: 'v1',
        type: 'pip',
        install_type: 'pip',
        cmake_editable: false,
        is_active: true,
      },
    ])
    expect(python.findAll('button')).toHaveLength(1)
  })

  it('hides cmake edit for 1Cat-vLLM even when a branch is present', () => {
    const wrapper = mountTable([
      {
        id: '1cat_vllm:src',
        version: 'src',
        type: 'source',
        install_type: 'source',
        source_branch: 'main',
        cmake_editable: false,
        is_active: true,
      },
    ])
    expect(wrapper.emitted('edit-config')).toBeFalsy()
    expect(wrapper.findAll('button')).toHaveLength(2)
  })
})
