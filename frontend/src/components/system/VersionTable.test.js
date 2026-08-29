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

    // Active fork: Sync + Delete (no Activate). First button is sync.
    const buttons = wrapper.findAll('button')
    expect(buttons.length).toBe(2)
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
    expect(buttons.length).toBe(2)
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
    expect(wrapper.findAll('button')).toHaveLength(1)
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
})
