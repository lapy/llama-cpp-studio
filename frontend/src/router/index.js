import { createRouter, createWebHistory } from 'vue-router'
import { installRouteRecovery } from '@/utils/routeRecovery'
import { installWorkspaceNavigation } from '@/utils/workspaceNavigation'

const routes = [
  {
    path: '/:pathMatch(.*)*',
    name: 'not-found',
    meta: { title: 'Page not found' },
    component: () => import('@/views/NotFoundView.vue'),
  },
  {
    path: '/',
    redirect: '/models',
  },
  {
    path: '/models',
    name: 'models',
    meta: { title: 'Models' },
    component: () => import('@/views/ModelLibrary.vue'),
  },
  {
    path: '/audio',
    name: 'audio',
    meta: { title: 'Audio' },
    component: () => import('@/views/AudioWorkspace.vue'),
  },
  {
    path: '/search',
    name: 'search',
    meta: { title: 'Search models' },
    component: () => import('@/views/ModelSearch.vue'),
  },
  {
    path: '/models/:id/config',
    name: 'model-config',
    meta: { title: 'Model configuration' },
    component: () => import('@/views/ModelConfig.vue'),
    props: true,
  },
  {
    path: '/system',
    name: 'system',
    meta: { title: 'System' },
    component: () => import('@/views/SystemView.vue'),
  },
  {
    path: '/proxy',
    name: 'proxy',
    meta: { title: 'Proxy' },
    component: () => import('@/views/ProxyView.vue'),
  },
  {
    path: '/engines',
    name: 'engines',
    meta: { title: 'Engines' },
    component: () => import('@/views/EnginesView.vue'),
  },
  {
    path: '/restore',
    redirect: { path: '/system', hash: '#config-backup' },
  },
]

const router = createRouter({
  history: createWebHistory(),
  routes,
  scrollBehavior(to, from, saved) {
    return saved || (to.path !== from.path || to.hash !== from.hash ? { top: 0 } : false)
  },
})

router.beforeEach((to) => {
  if (to.hash === '#config-backup' && to.path !== '/system')
    return { path: '/system', hash: '#config-backup' }
})

installWorkspaceNavigation(router)
installRouteRecovery(router)

export default router
