<template>
  <div id="app" class="animate-fade-in">
    <ConfirmDialog />
    <Toast />
    <div class="layout-wrapper">
      <!-- Header -->
      <AppHeader 
        :llama-swap-status="systemStore.systemStatus?.proxy_status || null"
      />

      <!-- Navigation -->
      <AppNavigation />

      <!-- Main Content -->
      <main class="layout-main">
        <router-view />
      </main>

      <!-- Footer -->
      <AppFooter />
    </div>
  </div>
</template>

<script setup>
// Vue
import { ref, onMounted, onUnmounted } from 'vue'
import { useRouter } from 'vue-router'

// PrimeVue
import ConfirmDialog from 'primevue/confirmdialog'
import Toast from 'primevue/toast'
import { useToast } from 'primevue/usetoast'

// Stores
import { useEnginesStore } from '@/stores/engines'
import { useProgressStore } from '@/stores/progress'

// Composables
import { useTheme } from '@/composables/useTheme'

// Components
import AppHeader from '@/components/layout/AppHeader.vue'
import AppNavigation from '@/components/layout/AppNavigation.vue'
import AppFooter from '@/components/layout/AppFooter.vue'

const toast = useToast()
const systemStore = useEnginesStore()
const progressStore = useProgressStore()
const { initTheme } = useTheme()

const statusLoading = ref(false)
const router = useRouter()

let unsubscribeNotifications = null
let unsubscribeTaskUpdated = null
let removeRouterAfterEach = null
function onVisibilityRefresh() {
  if (document.visibilityState === 'visible') {
    systemStore.fetchSwapConfigStale()
  }
}

function mapNotificationSeverity(t) {
  const x = String(t || '').toLowerCase()
  if (x === 'success') return 'success'
  if (x === 'error' || x === 'danger') return 'error'
  if (x === 'warn' || x === 'warning') return 'warn'
  return 'info'
}

onMounted(() => {
  initTheme()
  progressStore.connect()
  refreshStatus()
  systemStore.fetchSwapConfigStale()
  document.addEventListener('visibilitychange', onVisibilityRefresh)
  removeRouterAfterEach = router.afterEach(() => {
    systemStore.fetchSwapConfigStale()
  })
  unsubscribeTaskUpdated = progressStore.subscribe('task_updated', (task) => {
    if (task?.status === 'completed' || task?.status === 'failed') {
      systemStore.fetchSwapConfigStale()
    }
  })
  unsubscribeNotifications = progressStore.subscribe('notification', (payload) => {
    if (!payload || typeof payload !== 'object') return
    const summary = payload.title || payload.summary || 'Notice'
    const detail = payload.message || payload.detail || ''
    const severity = mapNotificationSeverity(payload.type || payload.notification_type)
    toast.add({
      severity,
      summary,
      detail: detail || undefined,
      life: severity === 'error' ? 6000 : 4000,
    })
  })
})

onUnmounted(() => {
  if (unsubscribeNotifications) unsubscribeNotifications()
  if (unsubscribeTaskUpdated) unsubscribeTaskUpdated()
  if (removeRouterAfterEach) removeRouterAfterEach()
  document.removeEventListener('visibilitychange', onVisibilityRefresh)
  progressStore.disconnect()
})

const refreshStatus = async () => {
  statusLoading.value = true
  try {
    await systemStore.fetchSystemStatus()
  } catch (error) {
    toast.add({ severity: 'error', summary: 'Failed to refresh system status', detail: error?.message, life: 4000 })
  } finally {
    statusLoading.value = false
  }
}

</script>

<style scoped>
/* Layout styles are in global _base.css */
</style>
