import { describe, expect, it } from 'vitest'
import { ansi256ToCss, renderAnsi, stripAnsi } from './ansi'

describe('ANSI rendering', () => {
  it('renders the installer colour example as styled text', () => {
    const segments = renderAnsi(
      '\u001b[1;34m[install_v100]\u001b[0m Building lean SM70-only sglang-kernel',
    )
    expect(segments.map((segment) => segment.text).join('')).toBe(
      '[install_v100] Building lean SM70-only sglang-kernel',
    )
    expect(segments[0].className).toContain('ansi-bold')
    expect(segments[0].style?.color).toBe('var(--ansi-fg-blue)')
  })

  it('strips cursor, erase, and OSC sequences without losing text', () => {
    const raw = '\u001b[2K\u001b[1Gloading\u001b]0;title\u0007 40%\u001b[?25l\u001b(B'
    expect(stripAnsi(raw)).toBe('loading 40%')
  })

  it('supports extended 256 and true-colour values', () => {
    const segments = renderAnsi('\u001b[38;5;196mred\u001b[0m \u001b[38;2;1;2;3mrgb\u001b[0m')
    expect(segments.map((segment) => segment.text).join('')).toBe('red rgb')
    expect(segments[0].style?.color).toBe('rgb(255 0 0)')
    expect(segments[2].style?.color).toBe('rgb(1 2 3)')
    expect(ansi256ToCss(244)).toBe('rgb(128 128 128)')
  })

  it('resets attributes when a later SGR says so', () => {
    const segments = renderAnsi('\u001b[1;31mbold red\u001b[22;39m plain')
    expect(segments[0].className).toContain('ansi-bold')
    expect(segments[0].style?.color).toBe('var(--ansi-fg-red)')
    expect(segments[1].className).toBeUndefined()
    expect(segments[1].style).toBeUndefined()
  })

  it('marks search matches without breaking the source text', () => {
    const segments = renderAnsi('Loading layer 7 of 32', 'layer')
    expect(segments.map((segment) => segment.text).join('')).toBe('Loading layer 7 of 32')
    expect(segments.some((segment) => segment.className === 'log-match')).toBe(true)
  })
})
