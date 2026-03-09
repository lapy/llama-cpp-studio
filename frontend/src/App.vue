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

onMounted(() => {
  initTheme()
  progressStore.connect()
  refreshStatus()
})

onUnmounted(() => {
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
