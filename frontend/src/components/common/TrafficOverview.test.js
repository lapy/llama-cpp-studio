import { mount } from '@vue/test-utils'
import { describe, it, expect } from 'vitest'
import TrafficOverview from './TrafficOverview.vue'

describe('TrafficOverview', () => {
  it('counts only retained requests in the window and separates pending from successful requests', () => {
    const wrapper = mount(TrafficOverview, {
      props: {
        now: 10000,
        requests: [
          { started_at: 9999, status: 200 },
          { started_at: 9999, status: 404 },
          { started_at: 9999, status: 503 },
          { started_at: 9999, state: 'streaming' },
          { started_at: 8000, status: 200 },
          { started_at: 10001, status: 200 },
          { started_at: 'bad', status: 200 },
        ],
      },
    })
    expect(wrapper.get('svg title').text()).toContain('4 retained HTTP requests')
    expect(wrapper.get('svg title').text()).toContain(
      '1 successful, 1 HTTP 4xx, 1 HTTP 5xx, 1 other or in progress',
    )
    expect(wrapper.text()).toContain('newest 100 records')
    expect(
      wrapper.findAll('rect').filter((rect) => Number(rect.attributes('height')) > 0),
    ).toHaveLength(4)
  })

  it('shows an honest empty state and never invents a trend', () => {
    const wrapper = mount(TrafficOverview, { props: { now: 10000, requests: [] } })
    expect(wrapper.text()).toContain('No retained traffic in this window')
    expect(wrapper.findAll('rect').every((rect) => Number(rect.attributes('height')) === 0)).toBe(
      true,
    )
  })

  it('bounds the visual sample to 100 records and updates when observations change', async () => {
    const wrapper = mount(TrafficOverview, {
      props: {
        now: 10000,
        requests: Array.from({ length: 101 }, () => ({ started_at: 9999, status: 200 })),
      },
    })
    expect(wrapper.get('svg title').text()).toContain('100 retained HTTP requests')
    await wrapper.setProps({ now: 12000 })
    expect(wrapper.get('svg title').text()).toContain('0 retained HTTP requests')
  })
})
