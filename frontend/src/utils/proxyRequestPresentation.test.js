import { describe, expect, it } from 'vitest'
import {
  formatAllowance,
  metricNote,
  number,
  percent,
  rate,
  tokenDetail,
  ttftDetail,
  ttftLabel,
} from './proxyRequestPresentation'

describe('proxy request presentation', () => {
  it('renders unknown metrics as an em dash rather than a zero', () => {
    expect(number(null)).toBe('—')
    expect(number(1200)).toBe('1,200')
    expect(rate(null)).toBe('—')
    expect(rate(12.345)).toBe('12.3')
    expect(percent(null)).toBe('—')
    expect(percent(0.8125)).toBe('81.3%')
    expect(formatAllowance(null)).toBe('unlimited')
    expect(formatAllowance(10000)).toBe('10,000')
  })

  it('labels a buffered response where TTFT is not observable', () => {
    expect(ttftLabel({ ttft_ms: 42.4 })).toBe('42.4 ms')
    expect(ttftLabel({ ttft_ms: null, response_transport: 'buffered' })).toBe('non-stream')
    expect(ttftLabel({})).toBe('—')
    expect(ttftDetail({ response_transport: 'buffered' })).toContain(
      'the first output event is the complete body',
    )
    expect(ttftDetail({ ttft_ms: null })).toContain('did not emit recognizable')
  })

  it('uses the proxy-reported metric semantics and a fixed cache definition', () => {
    const semantics = {
      ttft_ms: { label: 'TTFT', definition: 'Time to first token.', origin: 'engine' },
    }
    expect(metricNote('ttft_ms', semantics)).toBe('TTFT: Time to first token. Origin: engine')
    expect(metricNote('missing', semantics)).toBe('')
    expect(metricNote('input_cache')).toContain('Cache hit ratio')
  })

  it('summarises token accounting provenance without inventing values', () => {
    expect(tokenDetail({})).toBe('Usage was not reported for this request.')
    expect(
      tokenDetail({
        input_tokens_total: 10,
        input_tokens_cached: 4,
        input_tokens_processed: 6,
        usage_status: 'estimated',
      }),
    ).toBe('logical input 10 · cached 4 · engine processed 6 · usage estimated')
  })
})
