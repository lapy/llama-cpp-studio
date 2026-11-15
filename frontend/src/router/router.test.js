import { describe, it, expect } from 'vitest'
import router from './index.js'

describe('Vue router', () => {
  it('registers core routes', () => {
    const paths = new Set(router.getRoutes().map((r) => r.path))
    expect(paths.has('/models')).toBe(true)
    expect(paths.has('/search')).toBe(true)
    expect(paths.has('/engines')).toBe(true)
    expect(paths.has('/models/:id/config')).toBe(true)
    const restore = router.getRoutes().find((r) => r.path === '/restore')
    const system = router.getRoutes().find((r) => r.path === '/system')
    expect(restore?.redirect).toEqual({ path: '/system', hash: '#config-backup' })
    expect(system?.name).toBe('system')
    expect(system?.redirect).toBeUndefined()
  })

  it('redirects root to models', () => {
    const root = router.getRoutes().find((r) => r.path === '/')
    expect(root?.redirect).toBe('/models')
  })

  it('leaves model-config section scrolling to the view', () => {
    const scrollBehavior = router.options.scrollBehavior
    const from = router.resolve('/models/model-1/config#runtime')
    const to = router.resolve('/models/model-1/config#launch')
    expect(scrollBehavior(to, from, null)).toBe(false)
    expect(scrollBehavior(to, router.resolve('/models'), null)).toEqual({ top: 0 })
  })
})
