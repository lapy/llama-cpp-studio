<template>
  <footer class="layout-footer">
    <div class="footer-content">
      <span>llama.cpp Studio v1.0.0</span>
      <div class="connection-status">
        <i :class="connectionStatus.icon" :style="{ color: connectionStatus.color }"></i>
        <span>{{ connectionStatus.label }}</span>
      </div>
    </div>
  </footer>
</template>

<script setup>
import { computed } from 'vue'
import { useWebSocketStore } from '@/stores/websocket'

const wsStore = useWebSocketStore()

const connectionStatus = computed(() => {
  if (wsStore.connectionStatus === 'connected') {
    return {
      icon: 'pi pi-check-circle',
      color: 'var(--status-success)',
      label: 'Connected'
    }
  }

  if (wsStore.connectionStatus === 'reconnecting') {
    return {
      icon: 'pi pi-spin pi-spinner',
      color: 'var(--status-warning)',
      label: 'Reconnecting...'
    }
  }

  return {
    icon: 'pi pi-times-circle',
    color: 'var(--status-error)',
    label: 'Disconnected'
  }
})
</script>

<style scoped>
/* Footer styles are in global _base.css */
</style>

