import { shallowRef } from 'vue'
export const routeFailure = shallowRef(null)
export function installRouteRecovery(router) {
  router.onError((_error, to) => {
    routeFailure.value = { path: to.fullPath }
  })
  router.afterEach((_to, _from, failure) => {
    if (!failure) routeFailure.value = null
  })
}
