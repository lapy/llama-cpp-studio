import { createRouter, createWebHistory } from 'vue-router'
import ModelLibrary from '@/views/ModelLibrary.vue'
import ModelSearch from '@/views/ModelSearch.vue'
import ModelConfig from '@/views/ModelConfig.vue'
import LlamaCppManager from '@/views/LlamaCppManager.vue'
import SystemStatus from '@/views/SystemStatus.vue'
import LMDeploy from '@/views/LMDeploy.vue'

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
    path: '/llama-versions',
    name: 'llama-versions',
    component: LlamaCppManager
  },
  {
    path: '/system',
    name: 'system',
    component: SystemStatus
  },
  {
    path: '/lmdeploy',
    name: 'lmdeploy',
    component: LMDeploy
  }
]

const router = createRouter({
  history: createWebHistory(),
  routes
})

export default router
