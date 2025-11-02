<template>
  <div v-if="visible" class="empty-state">
    <div class="empty-state-content">
      <div class="empty-icon" aria-hidden="true">{{ icon }}</div>
      <h3 class="empty-title">{{ title }}</h3>
      <p class="empty-description">{{ description }}</p>
      
      <div class="empty-actions">
        <slot name="actions">
          <Button 
            v-if="showSmartAuto"
            label="Use Smart Auto" 
            icon="pi pi-bolt"
            @click="$emit('smart-auto')"
            severity="info"
            size="large"
            class="empty-action-primary"
            aria-label="Use Smart Auto to automatically configure settings"
          />
          <Button 
            v-if="showPresets"
            label="Choose Preset" 
            icon="pi pi-sliders-h"
            @click="$emit('presets')"
            severity="secondary"
            outlined
            size="large"
            aria-label="Choose a preset configuration"
          />
          <Button 
            label="Manual Setup" 
            icon="pi pi-cog"
            @click="$emit('manual')"
            text
            size="large"
            aria-label="Configure settings manually"
          />
        </slot>
      </div>
    </div>
  </div>
</template>

<script setup>
import Button from 'primevue/button'

const props = defineProps({
  visible: {
    type: Boolean,
    default: false
  },
  icon: {
    type: String,
    default: '🎯'
  },
  title: {
    type: String,
    default: 'Configure Your First Model'
  },
  description: {
    type: String,
    default: 'Start with a preset or let Smart Auto optimize for you'
  },
  showSmartAuto: {
    type: Boolean,
    default: true
  },
  showPresets: {
    type: Boolean,
    default: true
  }
})

defineEmits(['smart-auto', 'presets', 'manual'])
</script>

<style scoped>
.empty-state {
  display: flex;
  align-items: center;
  justify-content: center;
  min-height: 400px;
  padding: var(--spacing-xl);
  margin: var(--spacing-xl) 0;
}

.empty-state-content {
  text-align: center;
  max-width: 500px;
  width: 100%;
}

.empty-icon {
  font-size: 4rem;
  line-height: 1;
  margin-bottom: var(--spacing-lg);
  animation: float 3s ease-in-out infinite;
}

@keyframes float {
  0%, 100% {
    transform: translateY(0);
  }
  50% {
    transform: translateY(-10px);
  }
}

.empty-title {
  margin: 0 0 var(--spacing-md) 0;
  font-size: 1.75rem;
  font-weight: 700;
  color: var(--text-primary);
  background: linear-gradient(135deg, #22d3ee, #3b82f6);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}

.empty-description {
  margin: 0 0 var(--spacing-xl) 0;
  font-size: 1rem;
  color: var(--text-secondary);
  line-height: 1.6;
}

.empty-actions {
  display: flex;
  flex-direction: column;
  gap: var(--spacing-md);
  align-items: center;
}

.empty-action-primary {
  min-width: 200px;
}

@media (min-width: 600px) {
  .empty-actions {
    flex-direction: row;
    justify-content: center;
  }
}
</style>

