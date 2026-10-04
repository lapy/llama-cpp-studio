import { describe, expect, it } from 'vitest'
import client from './client'

async function effectiveTimeout(method, options = {}) {
  let timeout
  await client.request({
    url: '/api/status',
    method,
    ...options,
    adapter: async (config) => {
      timeout = config.timeout
      return { data: {}, status: 200, statusText: 'OK', headers: {}, config }
    },
  })
  return timeout
}

describe('control-plane request deadlines', () => {
  it.each(['get', 'head'])('bounds default %s requests after Axios merges defaults', async (method) => {
    expect(await effectiveTimeout(method)).toBe(20_000)
  })

  it('preserves an explicit read deadline', async () => {
    expect(await effectiveTimeout('get', { timeout: 1234 })).toBe(1234)
  })

  it('does not impose a deadline on mutations', async () => {
    expect(await effectiveTimeout('post')).toBe(0)
    expect(await effectiveTimeout('post', { timeout: 1234 })).toBe(1234)
  })
})
