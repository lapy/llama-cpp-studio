<template>
  <header class="layout-header animate-slide-in-up">
    <div class="layout-header-content">
      <div class="logo">
        <span style="font-size: 1.5rem; margin-right: 0.5rem;">🎨</span>
        <span>llama.cpp Studio</span>
      </div>
      <div class="header-actions">
        <slot name="actions">
          <ThemeToggle />
          <Button 
            icon="pi pi-refresh" 
            @click="$emit('refresh-status')"
            :loading="statusLoading"
            severity="secondary"
            size="small"
            v-tooltip.top="'Refresh System Status'"
          />
          <Button 
            icon="pi pi-info-circle" 
            @click="$emit('show-system-info')"
            severity="secondary"
            size="small"
            v-tooltip.top="'Show System Information'"
          />
        </slot>
      </div>
    </div>
  </header>
</template>

<script setup>
import Button from 'primevue/button'
import ThemeToggle from '@/components/ThemeToggle.vue'

defineProps({
  statusLoading: {
    type: Boolean,
    default: false
  }
})

defineEmits(['refresh-status', 'show-system-info'])
</script>

<style scoped>
/* Header styles are in global _base.css */
</style>

