import { createRouter, createWebHistory } from 'vue-router'

const routes = [
  {
    path: '/',
    redirect: '/models'
  },
  {
    path: '/models',
    name: 'models',
    component: () => import('@/views/ModelLibrary.vue')
  },
  {
    path: '/audio',
    name: 'audio',
    component: () => import('@/views/AudioWorkspace.vue')
  },
  {
    path: '/search',
    name: 'search',
    component: () => import('@/views/ModelSearch.vue')
  },
  {
    path: '/models/:id/config',
    name: 'model-config',
    component: () => import('@/views/ModelConfig.vue'),
    props: true
  },
  {
    path: '/system',
    redirect: { path: '/engines', hash: '#config-backup' }
  },
  {
    path: '/engines',
    name: 'engines',
    component: () => import('@/views/EnginesView.vue')
  },
  {
    path: '/restore',
    redirect: { path: '/engines', hash: '#config-backup' }
  }
]

const router = createRouter({
  history: createWebHistory(),
  routes
})

export default router
