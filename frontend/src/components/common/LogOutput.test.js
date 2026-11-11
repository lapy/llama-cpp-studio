import { flushPromises, mount } from '@vue/test-utils'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import LogOutput from './LogOutput.vue'

describe('LogOutput', () => {
  const writeText = vi.fn().mockResolvedValue(undefined)

  beforeEach(() => {
    writeText.mockClear()
    vi.stubGlobal('navigator', { clipboard: { writeText } })
  })

  function mountLog(props = {}) {
    return mount(LogOutput, {
      props: {
        lines: ['\u001b[1;34m[install_v100]\u001b[0m Building lean SM70-only sglang-kernel'],
        ...props,
      },
    })
  }

  it('renders ANSI-coloured lines in the fixed-height container with a compact toolbar', async () => {
    const wrapper = mountLog()
    await flushPromises()
    expect(wrapper.find('.log-toolbar').exists()).toBe(true)
    expect(wrapper.find('.log-body').text()).toContain(
      '[install_v100] Building lean SM70-only sglang-kernel',
    )
    expect(wrapper.findAll('.ansi-bold').length).toBeGreaterThan(0)
    expect(wrapper.find('.log-search input').exists()).toBe(true)
    expect(wrapper.find('button[aria-label="Follow log output"]').exists()).toBe(true)
    expect(wrapper.find('button[aria-label="Copy"]').exists()).toBe(true)
    wrapper.unmount()
  })

  it('hides stdout/stderr prefixes and styles stderr differently', async () => {
    const wrapper = mountLog({
      lines: ['stdout: server listening', 'stderr: out of memory'],
      streamAware: true,
    })
    await flushPromises()
    const text = wrapper.find('.log-body').text()
    expect(text).toContain('server listening')
    expect(text).not.toContain('stdout:')
    expect(text).not.toContain('stderr:')
    const stderr = wrapper.get('[data-stream="stderr"]')
    expect(stderr.classes()).toContain('log-line--stderr')
    wrapper.unmount()
  })

  it('keeps only the last carriage-return redraw of a progress line', async () => {
    const wrapper = mountLog({ lines: ['loading 10%\rloading 40%\rloading 100%'] })
    await flushPromises()
    expect(wrapper.find('.log-body').text()).toBe('loading 100%')
    wrapper.unmount()
  })

  it('filters and highlights search matches without another request', async () => {
    const wrapper = mountLog({
      lines: ['stdout: listening', 'stderr: error mapping layer', 'stdout: WARNING slow'],
      streamAware: true,
      levels: [
        { value: '', label: 'All' },
        { value: 'warn', label: 'Warn' },
        { value: 'error', label: 'Error' },
      ],
    })
    await flushPromises()

    await wrapper.find('.log-search input').setValue('layer')
    await flushPromises()
    expect(wrapper.findAll('.log-line')).toHaveLength(1)
    expect(wrapper.findAll('.log-match').length).toBeGreaterThan(0)
    expect(wrapper.find('.log-count').text()).toBe('1 / 3')

    await wrapper.find('.log-search input').setValue('')
    await wrapper.findAll('.log-level')[2].trigger('click')
    await flushPromises()
    expect(wrapper.findAll('.log-line')).toHaveLength(1)

    await wrapper.findAll('.log-level')[1].trigger('click')
    await flushPromises()
    expect(wrapper.findAll('.log-line')).toHaveLength(2)
    wrapper.unmount()
  })

  it('copies and downloads only the visible, plain-text lines', async () => {
    const wrapper = mountLog({
      lines: ['stdout: \u001b[32mone\u001b[0m', 'stdout: two'],
      streamAware: true,
      downloadable: true,
    })
    await flushPromises()
    await wrapper.find('button[aria-label="Copy"]').trigger('click')
    await flushPromises()
    expect(writeText).toHaveBeenCalledWith('one\ntwo')

    await wrapper.find('button[aria-label="Download log"]').trigger('click')
    expect(wrapper.emitted('download')?.[0]?.[0]).toEqual(['one', 'two'])
    wrapper.unmount()
  })

  it('shows a loading or empty message when there is no output yet', async () => {
    const empty = mountLog({ lines: [] })
    await flushPromises()
    expect(empty.find('.log-empty').text()).toBe('Waiting for output…')
    empty.unmount()

    const loading = mountLog({ lines: [], loading: true })
    await flushPromises()
    expect(loading.find('.log-empty').text()).toBe('Loading saved log…')
    loading.unmount()
  })
})
