<template>
  <div id="app" class="animate-fade-in">
    <ConfirmDialog />
    
    <div class="layout-wrapper">
      <!-- Header -->
      <AppHeader 
        :status-loading="statusLoading"
        @refresh-status="refreshStatus"
        @show-system-info="showSystemInfo"
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
import { useConfirm } from 'primevue/useconfirm'

// Third-party
import { toast } from 'vue3-toastify'

// Stores
import { useSystemStore } from '@/stores/system'
import { useWebSocketStore } from '@/stores/websocket'

// Composables
import { useTheme } from '@/composables/useTheme'

// Components
import AppHeader from '@/components/layout/AppHeader.vue'
import AppNavigation from '@/components/layout/AppNavigation.vue'
import AppFooter from '@/components/layout/AppFooter.vue'

const router = useRouter()
const confirm = useConfirm()
const systemStore = useSystemStore()
const wsStore = useWebSocketStore()
const { initTheme } = useTheme()

const statusLoading = ref(false)

onMounted(() => {
  initTheme()
  wsStore.connect()
  refreshStatus()
})

onUnmounted(() => {
  wsStore.disconnect()
})

const refreshStatus = async () => {
  statusLoading.value = true
  try {
    await systemStore.fetchSystemStatus()
  } catch (error) {
    toast.error('Failed to refresh system status')
  } finally {
    statusLoading.value = false
  }
}

const showSystemInfo = () => {
  confirm.require({
    message: `GPU Count: ${systemStore.gpuInfo.device_count || 0}\nTotal VRAM: ${((systemStore.gpuInfo.total_vram || 0) / 1024**3).toFixed(1)} GB\nCUDA Version: ${systemStore.gpuInfo.cuda_version || 'Unknown'}`,
    header: 'System Information',
    icon: 'pi pi-info-circle',
    rejectLabel: 'Close',
    acceptLabel: '',
    accept: () => {},
    reject: () => {}
  })
}
</script>

<style scoped>
/* Layout styles are in global _base.css */
</style>
