<template>
  <footer class="layout-footer">
    <div class="footer-content">
      <span>llama.cpp Studio v{{ appVersion }}</span>
      <div class="live-status" :title="progressStore.isConnected ? 'Live updates (SSE)' : 'Reconnecting…'">
        <i
          v-if="progressStore.isConnected"
          class="pi pi-check-circle"
          style="color: var(--status-success)"
          aria-hidden="true"
        />
        <i
          v-else
          class="pi pi-clock"
          style="color: var(--status-warning)"
          aria-hidden="true"
        />
        <span>{{ progressStore.isConnected ? 'Live' : 'Reconnecting…' }}</span>
      </div>
    </div>
  </footer>
</template>

<script setup>
import { useProgressStore } from '@/stores/progress'

const progressStore = useProgressStore()
const appVersion = typeof __APP_VERSION__ !== 'undefined' ? __APP_VERSION__ : '1.0.0'
</script>

<style scoped>
/* Footer styles are in global _base.css */
</style>

