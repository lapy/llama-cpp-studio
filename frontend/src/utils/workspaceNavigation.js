import { nextTick } from 'vue'

export function workspaceTitle(route) {
  if (route.path === '/system' && route.hash === '#config-backup') return 'Backup & restore'
  if (route.path === '/system' && route.hash === '#history') return 'Change history'
  return route.meta?.title || 'Workspace'
}

/** Initial loads keep the skip link first; subsequent page changes focus main. */
export function installWorkspaceNavigation(router) {
  return router.afterEach(async (to, from, failure) => {
    if (failure || typeof document === 'undefined') return
    document.title = `${workspaceTitle(to)} — llama.cpp Studio`
    if (!from.matched.length || (to.path === from.path && to.hash === from.hash)) return
    await nextTick()
    document.getElementById('main-content')?.focus({ preventScroll: true })
  })
}
