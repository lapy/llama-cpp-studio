import { createRouter, createWebHistory } from 'vue-router'
import ModelLibrary from '@/views/ModelLibrary.vue'
import ModelSearch from '@/views/ModelSearch.vue'
import ModelConfig from '@/views/ModelConfig.vue'
import System from '@/views/System.vue'

const routes = [
  {
    path: '/',
    redirect: '/models'
  },
  {
    path: '/models',
    name: 'models',
    component: ModelLibrary
  },
  {
    path: '/search',
    name: 'search',
    component: ModelSearch
  },
  {
    path: '/models/:id/config',
    name: 'model-config',
    component: ModelConfig,
    props: true
  },
  {
    path: '/system',
    name: 'system',
    component: System
  }
]

const router = createRouter({
  history: createWebHistory(),
  routes
})

export default router
