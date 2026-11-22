import { describe, expect, it } from 'vitest'
import { mount } from '@vue/test-utils'
import NetworkAccessPanel from './NetworkAccessPanel.vue'

const network = {
  listen_host: '127.0.0.1',
  listen_source: 'loopback default',
  port: 2000,
  public_url: 'http://studio.local:2000',
  tls_note: 'Terminate TLS at a reverse proxy.',
}

describe('NetworkAccessPanel', () => {
  it('renders the read-only binding details', () => {
    const wrapper = mount(NetworkAccessPanel, { props: { network } })
    expect(wrapper.get('#proxy-network-title').text()).toBe('Network')
    expect(wrapper.text()).toContain('127.0.0.1')
    expect(wrapper.text()).toContain('loopback default')
    expect(wrapper.text()).toContain('2000')
    expect(wrapper.text()).toContain('http://studio.local:2000')
    expect(wrapper.text()).toContain('Terminate TLS at a reverse proxy.')
    expect(wrapper.find('input, select, textarea').exists()).toBe(false)
  })

  it('shows placeholders instead of inventing a binding', () => {
    const wrapper = mount(NetworkAccessPanel, { props: { network: { tls_note: 'plain HTTP' } } })
    const values = wrapper.findAll('dd').map((cell) => cell.text())
    expect(values[0]).toContain('—')
    expect(values[1]).toBe('—')
    expect(values[2]).toBe('Same origin (default)')
  })
})
