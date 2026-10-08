<template>
  <div id="app">
    <RemoteAccessGate
      v-if="!accessResolved || remoteLoginRequired"
      :pending="!accessResolved"
      :error="remoteLoginError"
      :submitting="remoteLoginSubmitting"
      @submit="submitRemoteToken"
    />
    <template v-else>
    <a class="skip-link" href="#main-content">Skip to content</a>
    <ConfirmDialog />
    <Toast />
    <TaskNotifications />
    <div class="layout-wrapper">
      <!-- Header -->
      <AppHeader 
        :llama-swap-status="headerProxyStatus"
      />

      <!-- Navigation -->
      <AppNavigation />

      <!-- Main Content -->
      <main id="main-content" class="layout-main" tabindex="-1">
        <router-view />
      </main>

      <!-- Footer -->
      <AppFooter />
    </div>
    </template>
  </div>
</template>

<script setup>
// Vue
import { ref, computed, onMounted, onUnmounted } from 'vue'

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
import TaskNotifications from '@/components/common/TaskNotifications.vue'
import RemoteAccessGate from '@/components/common/RemoteAccessGate.vue'
import { setSessionExpiredHandler } from '@/api/client.js'

const toast = useToast()
const systemStore = useEnginesStore()
const progressStore = useProgressStore()
const { initTheme } = useTheme()

const statusLoading = ref(false)
const accessResolved = ref(false)
const remoteLoginRequired = ref(false)
const remoteLoginError = ref('')
const remoteLoginSubmitting = ref(false)
const STATUS_FRESH_MS = 45_000
const statusObservedAt = ref(0)
const statusCurrent = ref(false)
const statusClock = ref(Date.now())
let appStarted = false
let statusClockTimer = null
let statusRefreshTimer = null

const headerProxyStatus = computed(() => {
  const status = systemStore.systemStatus?.proxy_status || null
  if (!status) return null
  const fresh = statusCurrent.value
    && statusObservedAt.value
    && statusClock.value - statusObservedAt.value <= STATUS_FRESH_MS
  if (!fresh) return { ...status, healthy: null }
  return status
})

setSessionExpiredHandler(() => {
  remoteLoginError.value = 'Studio restarted or the management session expired. Enter the password again.'
  remoteLoginRequired.value = true
})

async function refreshAccessMode() {
  try {
    const response = await fetch('/api/access')
    if (!response.ok) return
    const body = await response.json()
    remoteLoginRequired.value = body.mode === 'remote' && !body.authenticated
  } catch {
    remoteLoginRequired.value = false
  }
}

async function submitRemoteToken(token) {
  remoteLoginError.value = ''
  remoteLoginSubmitting.value = true
  try {
    const response = await fetch('/api/session', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      credentials: 'include',
      body: JSON.stringify({ token }),
    })
    if (!response.ok) {
      remoteLoginError.value = 'That password was not accepted. Check STUDIO_API_TOKEN, or open studio_access.token in the data config directory, and try again.'
      return
    }
    remoteLoginRequired.value = false
    if (appStarted) {
      progressStore.connect()
      refreshStatus({ announce: false })
      systemStore.fetchSwapConfigStale()
    } else {
      startApp()
    }
  } catch {
    remoteLoginError.value = 'Studio could not reach the server to check the password. Try again in a moment.'
  } finally {
    remoteLoginSubmitting.value = false
  }
}

let unsubscribeNotifications = null
let unsubscribeTaskUpdated = null
let lastStaleVisibilityFetch = 0
const STALE_VISIBILITY_THROTTLE_MS = 30_000

function onVisibilityRefresh() {
  if (document.visibilityState !== 'visible') return
  const now = Date.now()
  if (now - lastStaleVisibilityFetch < STALE_VISIBILITY_THROTTLE_MS) return
  lastStaleVisibilityFetch = now
  refreshStatus({ announce: false })
  systemStore.fetchSwapConfigStale()
}

function mapNotificationSeverity(t) {
  const x = String(t || '').toLowerCase()
  if (x === 'success') return 'success'
  if (x === 'error' || x === 'danger') return 'error'
  if (x === 'warn' || x === 'warning') return 'warn'
  return 'info'
}

function startApp() {
  if (appStarted) return
  appStarted = true
  progressStore.connect()
  refreshStatus({ announce: true })
  systemStore.fetchSwapConfigStale()
  document.addEventListener('visibilitychange', onVisibilityRefresh)
  statusClockTimer = setInterval(() => {
    statusClock.value = Date.now()
  }, 15_000)
  statusRefreshTimer = setInterval(() => {
    if (document.visibilityState === 'hidden') return
    refreshStatus({ announce: false })
    systemStore.fetchSwapConfigStale()
  }, 30_000)
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
}

onMounted(async () => {
  initTheme()
  await refreshAccessMode()
  accessResolved.value = true
  if (!remoteLoginRequired.value) startApp()
})

onUnmounted(() => {
  if (unsubscribeNotifications) unsubscribeNotifications()
  if (unsubscribeTaskUpdated) unsubscribeTaskUpdated()
  document.removeEventListener('visibilitychange', onVisibilityRefresh)
  if (statusClockTimer) clearInterval(statusClockTimer)
  if (statusRefreshTimer) clearInterval(statusRefreshTimer)
  setSessionExpiredHandler(null)
  progressStore.disconnect()
})

const refreshStatus = async ({ announce = false } = {}) => {
  statusLoading.value = true
  try {
    await systemStore.fetchSystemStatus()
    statusObservedAt.value = Date.now()
    statusClock.value = statusObservedAt.value
    statusCurrent.value = true
  } catch (error) {
    statusCurrent.value = false
    if (announce) {
      toast.add({ severity: 'error', summary: 'Failed to refresh system status', detail: error?.message, life: 4000 })
    }
  } finally {
    statusLoading.value = false
  }
}

</script>

<style scoped>
/* Layout styles are in global _base.css */
</style>
