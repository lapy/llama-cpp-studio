<template>
  <div v-if="isModelRunning" class="performance-card" role="region" aria-label="Live model performance metrics">
    <div class="performance-header">
      <div class="performance-title">
        <i class="pi pi-tachometer-alt" aria-hidden="true"></i>
        <h4>Live Performance</h4>
        <span class="live-indicator" aria-label="Live metrics updating in real-time">
          <i class="pi pi-circle-fill" aria-hidden="true"></i>
          Live
        </span>
      </div>
      <Button 
        v-if="modelId"
        label="Stop" 
        icon="pi pi-stop" 
        size="small" 
        severity="danger"
        outlined
        @click="$emit('stop')"
      />
    </div>
    
    <div class="performance-metrics">
      <div class="metric-item">
        <div class="metric-header">
          <span class="metric-label">Speed</span>
          <span v-if="trend" class="metric-trend" :class="trendClass">
            <i :class="trendIcon"></i>
            {{ trend }}
          </span>
        </div>
        <div class="metric-value">{{ speed || 'N/A' }}</div>
        <div class="metric-description">Tokens per second</div>
      </div>
      
      <div class="metric-item">
        <div class="metric-header">
          <span class="metric-label">Context Used</span>
        </div>
        <div class="metric-value">{{ contextUsed || 'N/A' }}</div>
        <div class="metric-description">{{ contextPercent || '0' }}% of max</div>
      </div>
      
      <div class="metric-item">
        <div class="metric-header">
          <span class="metric-label">VRAM</span>
        </div>
        <div class="metric-value">{{ vramUsage || 'N/A' }}</div>
        <div class="metric-description">{{ vramPercent || '0' }}% used</div>
      </div>
    </div>
    
    <div v-if="!isModelRunning" class="performance-empty">
      <i class="pi pi-info-circle"></i>
      <p>Start the model to see live performance metrics</p>
    </div>
  </div>
</template>

<script setup>
import { computed } from 'vue'
import Button from 'primevue/button'

const props = defineProps({
  isModelRunning: {
    type: Boolean,
    default: false
  },
  speed: {
    type: String,
    default: null
  },
  trend: {
    type: String,
    default: null
  },
  contextUsed: {
    type: String,
    default: null
  },
  contextPercent: {
    type: String,
    default: null
  },
  vramUsage: {
    type: String,
    default: null
  },
  vramPercent: {
    type: String,
    default: null
  },
  modelId: {
    type: [Number, String],
    default: null
  }
})

const emit = defineEmits(['stop'])

const trendClass = computed(() => {
  if (!props.trend) return ''
  if (props.trend.startsWith('+')) return 'trend-up'
  if (props.trend.startsWith('-')) return 'trend-down'
  return 'trend-neutral'
})

const trendIcon = computed(() => {
  if (props.trend?.startsWith('+')) return 'pi pi-arrow-up'
  if (props.trend?.startsWith('-')) return 'pi pi-arrow-down'
  return 'pi pi-minus'
})
</script>

<style scoped>
.performance-card {
  background: var(--gradient-card);
  border: 2px solid var(--accent-cyan);
  border-radius: var(--radius-xl);
  padding: var(--spacing-xl);
  box-shadow: 0 0 20px rgba(34, 211, 238, 0.2);
  animation: pulse-border 2s ease-in-out infinite;
}

@keyframes pulse-border {
  0%, 100% {
    box-shadow: 0 0 20px rgba(34, 211, 238, 0.2);
  }
  50% {
    box-shadow: 0 0 30px rgba(34, 211, 238, 0.4);
  }
}

.performance-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: var(--spacing-xl);
  padding-bottom: var(--spacing-lg);
  border-bottom: 1px solid var(--border-primary);
}

.performance-title {
  display: flex;
  align-items: center;
  gap: var(--spacing-md);
}

.performance-title i {
  font-size: 1.5rem;
  color: var(--accent-cyan);
}

.performance-title h4 {
  margin: 0;
  font-size: 1.25rem;
  font-weight: 600;
  color: var(--text-primary);
}

.live-indicator {
  display: flex;
  align-items: center;
  gap: var(--spacing-xs);
  padding: var(--spacing-xs) var(--spacing-sm);
  background: rgba(34, 197, 94, 0.15);
  color: #22c55e;
  border-radius: var(--radius-sm);
  font-size: 0.75rem;
  font-weight: 600;
  letter-spacing: 0.5px;
}

.live-indicator i {
  font-size: 0.5rem;
  animation: pulse 2s ease-in-out infinite;
}

@keyframes pulse {
  0%, 100% {
    opacity: 1;
  }
  50% {
    opacity: 0.5;
  }
}

.performance-metrics {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: var(--spacing-lg);
}

.metric-item {
  display: flex;
  flex-direction: column;
  gap: var(--spacing-xs);
  padding: var(--spacing-md);
  background: var(--bg-surface);
  border-radius: var(--radius-md);
  border: 1px solid var(--border-primary);
}

.metric-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.metric-label {
  font-size: 0.875rem;
  font-weight: 600;
  color: var(--text-secondary);
  text-transform: uppercase;
  letter-spacing: 0.5px;
}

.metric-trend {
  display: flex;
  align-items: center;
  gap: var(--spacing-xs);
  font-size: 0.75rem;
  font-weight: 600;
  padding: 2px 6px;
  border-radius: var(--radius-sm);
}

.metric-trend.trend-up {
  background: rgba(34, 197, 94, 0.15);
  color: #22c55e;
}

.metric-trend.trend-down {
  background: rgba(239, 68, 68, 0.15);
  color: #ef4444;
}

.metric-trend.trend-neutral {
  background: rgba(156, 163, 175, 0.15);
  color: #9ca3af;
}

.metric-value {
  font-size: 1.5rem;
  font-weight: 700;
  color: var(--accent-cyan);
  line-height: 1.2;
}

.metric-description {
  font-size: 0.75rem;
  color: var(--text-secondary);
}

.performance-empty {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: var(--spacing-md);
  padding: var(--spacing-xl);
  color: var(--text-secondary);
  text-align: center;
}

.performance-empty i {
  font-size: 2rem;
  color: var(--accent-cyan);
  opacity: 0.5;
}

.performance-empty p {
  margin: 0;
  font-size: 0.9rem;
}
</style>

