import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { copyText } from './clipboard'
let descriptor
beforeEach(() => {
  descriptor = Object.getOwnPropertyDescriptor(document, 'execCommand')
})
afterEach(() => {
  vi.unstubAllGlobals()
  if (descriptor) Object.defineProperty(document, 'execCommand', descriptor)
  else delete document.execCommand
  document.body.innerHTML = ''
})
function command(result = true) {
  const fn = vi.fn(() => result)
  Object.defineProperty(document, 'execCommand', { configurable: true, value: fn })
  return fn
}
describe('copyText', () => {
  it('uses the Clipboard API when it succeeds without reading the clipboard', async () => {
    const writeText = vi.fn().mockResolvedValue(undefined)
    vi.stubGlobal('navigator', { clipboard: { writeText } })
    const fallback = command()
    await copyText('one-time-key')
    expect(writeText).toHaveBeenCalledWith('one-time-key')
    expect(fallback).not.toHaveBeenCalled()
  })
  it('copies exact text on HTTP inside the active dialog and restores focus and selection', async () => {
    vi.stubGlobal('navigator', {})
    const dialog = document.createElement('div')
    dialog.setAttribute('role', 'dialog')
    const input = document.createElement('input')
    input.value = 'original'
    dialog.append(input)
    document.body.append(dialog)
    input.focus()
    input.setSelectionRange(2, 5)
    const fallback = command()
    fallback.mockImplementation(() => {
      expect(dialog.contains(document.activeElement)).toBe(true)
      expect(document.activeElement.value).toBe('first\nsecond')
      return true
    })
    await copyText('first\nsecond')
    expect(fallback).toHaveBeenCalledWith('copy')
    expect(document.activeElement).toBe(input)
    expect(input.selectionStart).toBe(2)
    expect(input.selectionEnd).toBe(5)
    expect(dialog.querySelector('textarea')).toBeNull()
  })
  it('falls back after an API permission rejection', async () => {
    vi.stubGlobal('navigator', {
      clipboard: { writeText: vi.fn().mockRejectedValue(new Error('Denied')) },
    })
    const fallback = command()
    await copyText('example')
    expect(fallback).toHaveBeenCalledWith('copy')
  })
  it.each([false, 'throws'])(
    'does not report success when the fallback returns %s',
    async (result) => {
      vi.stubGlobal('navigator', {})
      const fallback = command(result)
      if (result === 'throws')
        fallback.mockImplementation(() => {
          throw new Error('Copy blocked')
        })
      await expect(copyText('sensitive-test-value')).rejects.toThrow()
      expect(document.body.querySelector('textarea')).toBeNull()
      expect(document.body.textContent).not.toContain('sensitive-test-value')
    },
  )
  it('rejects empty values without touching the clipboard', async () => {
    const fallback = command()
    await expect(copyText('')).rejects.toThrow('no text')
    expect(fallback).not.toHaveBeenCalled()
  })
})
