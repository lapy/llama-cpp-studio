import { describe, it, expect } from 'vitest'
import router from './index.js'

describe('Vue router', () => {
  it('registers core routes', () => {
    const paths = new Set(router.getRoutes().map((r) => r.path))
    expect(paths.has('/models')).toBe(true)
    expect(paths.has('/search')).toBe(true)
    expect(paths.has('/engines')).toBe(true)
    expect(paths.has('/models/:id/config')).toBe(true)
  })

  it('redirects root to models', () => {
    const root = router.getRoutes().find((r) => r.path === '/')
    expect(root?.redirect).toBe('/models')
  })
})
