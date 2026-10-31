import { getCurrentInstance, watch } from 'vue'

/** Persist only explicitly supplied, non-sensitive view controls. */
export function useQueryState(bindings, onRestore = () => {}) {
  const router = getCurrentInstance()?.appContext.config.globalProperties.$router
  if (!router) return
  let restoring = false
  function restore(query) {
    let changed = false
    restoring = true
    for (const [key, binding] of Object.entries(bindings)) {
      const raw = typeof query[key] === 'string' ? query[key] : ''
      const next = binding.parse(raw)
      if (binding.get() !== next) {
        changed = true
        binding.set(next)
      }
    }
    restoring = false
    if (changed) onRestore()
  }
  restore(router.currentRoute.value.query)
  watch(() => router.currentRoute.value.query, restore)
  watch(
    () => Object.values(bindings).map((binding) => binding.get()),
    () => {
      if (restoring) return
      const query = { ...router.currentRoute.value.query }
      let push = false
      for (const [key, binding] of Object.entries(bindings)) {
        const value = binding.get()
        if (
          binding.history === 'push' &&
          binding.parse(router.currentRoute.value.query[key] || '') !== value
        )
          push = true
        if (value === '' || value === binding.default) delete query[key]
        else query[key] = String(value)
      }
      if (JSON.stringify(query) !== JSON.stringify(router.currentRoute.value.query)) {
        void router[push ? 'push' : 'replace']({ query }).catch(() => {})
      }
    },
    { flush: 'sync' },
  )
}
