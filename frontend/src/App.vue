<template>
  <div id="app" class="animate-fade-in">
    <ConfirmDialog />
    
    <div class="layout-wrapper">
      <!-- Header -->
      <header class="layout-header animate-slide-in-up">
        <div class="layout-header-content">
          <div class="logo">
            <span style="font-size: 1.5rem; margin-right: 0.5rem;">🎨</span>
            <span>llama.cpp Studio</span>
          </div>
          <div class="header-actions">
            <ThemeToggle />
            <Button 
              icon="pi pi-refresh" 
              @click="refreshStatus"
              :loading="statusLoading"
              severity="secondary"
              size="small"
              v-tooltip.top="'Refresh System Status'"
              class="animate-pulse"
            />
            <Button 
              icon="pi pi-info-circle" 
              @click="showSystemInfo"
              severity="secondary"
              size="small"
              v-tooltip.top="'Show System Information'"
            />
          </div>
        </div>
      </header>

      <!-- Navigation -->
      <nav class="layout-nav animate-slide-in-up">
        <div class="nav-content">
          <Button 
            label="Models" 
            icon="pi pi-database"
            :class="{ 'p-button-outlined': $route.name !== 'models' }"
            @click="$router.push('/models')"
            class="nav-button"
          />
          <Button 
            label="Search" 
            icon="pi pi-search"
            :class="{ 'p-button-outlined': $route.name !== 'search' }"
            @click="$router.push('/search')"
            class="nav-button"
          />
          <Button 
            label="llama.cpp" 
            icon="pi pi-code"
            :class="{ 'p-button-outlined': $route.name !== 'llama-versions' }"
            @click="$router.push('/llama-versions')"
            class="nav-button"
          />
          <Button 
            label="System" 
            icon="pi pi-desktop"
            :class="{ 'p-button-outlined': $route.name !== 'system' }"
            @click="$router.push('/system')"
            class="nav-button"
          />
        </div>
      </nav>

      <!-- Main Content -->
      <main class="layout-main">
        <router-view />
      </main>

      <!-- Footer -->
      <footer class="layout-footer">
        <div class="footer-content">
          <span>llama.cpp Studio v1.0.0</span>
          <div class="connection-status">
            <i :class="wsStore.connectionStatus === 'connected' ? 'pi pi-check-circle text-green-500' : 
                      wsStore.connectionStatus === 'reconnecting' ? 'pi pi-spin pi-spinner text-amber-500' : 
                      'pi pi-times-circle text-red-500'"></i>
            <span>{{ wsStore.connectionStatus === 'connected' ? 'Connected' : 
                     wsStore.connectionStatus === 'reconnecting' ? 'Reconnecting...' : 
                     'Disconnected' }}</span>
          </div>
        </div>
      </footer>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, onUnmounted } from 'vue'
import { useRouter } from 'vue-router'
import { toast } from 'vue3-toastify'
import { useConfirm } from 'primevue/useconfirm'
import { useSystemStore } from '@/stores/system'
import { useWebSocketStore } from '@/stores/websocket'
import { useTheme } from '@/composables/useTheme'
import Button from 'primevue/button'
import ConfirmDialog from 'primevue/confirmdialog'
import ThemeToggle from '@/components/ThemeToggle.vue'

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
.layout-wrapper {
  min-height: 100vh;
  display: flex;
  flex-direction: column;
}

.layout-header {
  background: var(--bg-card);
  border-bottom: 1px solid var(--border-primary);
  padding: var(--spacing-md) 0;
  box-shadow: var(--shadow-sm);
}

.layout-header-content {
  max-width: 1400px;
  margin: 0 auto;
  padding: 0 var(--spacing-md);
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.logo {
  display: flex;
  align-items: center;
  gap: var(--spacing-sm);
  font-size: 1.25rem;
  font-weight: 700;
  color: var(--accent-cyan);
}

.logo i {
  font-size: 1.5rem;
  color: var(--accent-cyan);
  margin-right: var(--spacing-sm);
}

.header-actions {
  display: flex;
  gap: var(--spacing-xs);
  align-items: center;
}

.layout-nav {
  background: var(--bg-secondary);
  border-bottom: 1px solid var(--border-primary);
  padding: var(--spacing-sm) 0;
  box-shadow: var(--shadow-sm);
}

.nav-content {
  max-width: 1400px;
  margin: 0 auto;
  padding: 0 var(--spacing-md);
  display: flex;
  gap: var(--spacing-xs);
}

/* Enhanced navigation button styling */
.nav-content .p-button {
  display: flex;
  align-items: center;
  gap: var(--spacing-sm);
  position: relative;
  overflow: hidden;
  transition: all var(--transition-normal);
}

.nav-content .p-button .p-button-icon {
  margin-right: var(--spacing-sm);
  transition: transform var(--transition-normal);
}

.nav-content .p-button:hover .p-button-icon {
  transform: scale(1.1) rotate(5deg);
}

.nav-content .p-button:not(.p-button-outlined) {
  background: var(--gradient-primary) !important;
  color: white !important;
  border: none !important;
  box-shadow: var(--shadow-md), var(--glow-primary) !important;
}

.nav-content .p-button:not(.p-button-outlined):hover {
  transform: translateY(-2px) scale(1.05) !important;
  box-shadow: var(--shadow-lg), var(--glow-primary) !important;
}

.nav-content .p-button.p-button-outlined:hover {
  background: var(--gradient-primary) !important;
  color: white !important;
  border-color: var(--accent-cyan) !important;
  transform: translateY(-2px) scale(1.05) !important;
}

.layout-main {
  flex: 1;
  max-width: 1400px;
  margin: 0 auto;
  padding: var(--spacing-md);
  width: 100%;
}

.layout-footer {
  background: var(--bg-card);
  border-top: 1px solid var(--border-primary);
  padding: var(--spacing-sm) 0;
  margin-top: auto;
  box-shadow: var(--shadow-sm);
}

.footer-content {
  max-width: 1400px;
  margin: 0 auto;
  padding: 0 var(--spacing-md);
  display: flex;
  justify-content: space-between;
  align-items: center;
  color: var(--text-secondary);
  font-size: 0.75rem;
}

.connection-status {
  display: flex;
  align-items: center;
  gap: var(--spacing-xs);
}
</style>
