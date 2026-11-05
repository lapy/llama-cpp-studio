import { describe, it, expect, beforeEach } from 'vitest'
import { useTheme } from './useTheme.js'

describe('useTheme', () => {
  beforeEach(() => {
    localStorage.clear()
    document.documentElement.removeAttribute('data-theme')
  })

  it('setTheme persists and updates document', () => {
    const { setTheme } = useTheme()
    setTheme('light')
    expect(localStorage.getItem('theme')).toBe('light')
    expect(document.documentElement.getAttribute('data-theme')).toBe('light')
  })

  it('toggleTheme switches between dark and light', () => {
    const { setTheme, toggleTheme, theme } = useTheme()
    setTheme('dark')
    toggleTheme()
    expect(theme.value).toBe('light')
    toggleTheme()
    expect(theme.value).toBe('dark')
  })

  it('ignores invalid theme names', () => {
    const { setTheme, theme } = useTheme()
    setTheme('dark')
    setTheme('invalid')
    expect(theme.value).toBe('dark')
  })
})
