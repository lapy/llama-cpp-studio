import { describe, expect, it } from 'vitest'

import { formatAxiosDetail } from './apiErrors'

describe('formatAxiosDetail', () => {
  it('prefers string, list and object API details', () => {
    expect(formatAxiosDetail({ response: { data: { detail: 'Model is busy' } } })).toBe(
      'Model is busy',
    )
    expect(
      formatAxiosDetail({
        response: {
          data: {
            detail: [
              { loc: ['body', 'name'], msg: 'is required' },
              { loc: [], msg: 'is invalid' },
            ],
          },
        },
      }),
    ).toBe('body.name: is required; : is invalid')
    expect(formatAxiosDetail({ response: { data: { detail: { msg: 'Nested message' } } } })).toBe(
      'Nested message',
    )
    expect(formatAxiosDetail({ response: { data: { message: 'Plain message' } } })).toBe(
      'Plain message',
    )
  })

  it('falls back to the request error message', () => {
    expect(formatAxiosDetail(new Error('Network down'))).toBe('Network down')
    expect(formatAxiosDetail({})).toBe('Request failed')
  })
})
