<template>
  <div class="memory-status-card" :class="statusClass" role="region" :aria-label="`${title} memory status`">
    <div class="memory-card-header">
      <div class="memory-status-icon">
        <i v-if="status === 'good'" class="pi pi-check-circle"></i>
        <i v-else-if="status === 'warning'" class="pi pi-exclamation-triangle"></i>
        <i v-else class="pi pi-times-circle"></i>
      </div>
      <div class="memory-card-title">
        <h4>{{ title }}</h4>
        <span class="memory-status-badge">{{ statusText }}</span>
      </div>
    </div>
    
    <div class="memory-status-content" v-if="!loading">
      <div class="memory-usage-display">
        <div class="usage-item" v-if="currentValue !== null">
          <span class="usage-label">Current:</span>
          <span class="usage-value">{{ formatFileSize(currentValue) }}</span>
        </div>
        <div class="usage-item" v-if="estimatedValue !== null">
          <span class="usage-label">+ Model:</span>
          <span class="usage-value">{{ formatFileSize(estimatedValue) }}</span>
        </div>
        <div class="usage-item total">
          <span class="usage-label">= Total:</span>
          <span class="usage-value">{{ formatFileSize(totalValue) }}</span>
          <span class="usage-fraction">/ {{ formatFileSize(totalCapacity) }}</span>
        </div>
      </div>
      
      <div class="memory-progress-bar">
        <div class="stacked-bar" :class="statusClass">
          <div class="bar-current" :style="{ width: currentPercent + '%' }"></div>
          <div class="bar-additional"
            :style="{ width: additionalPercent + '%', left: currentPercent + '%' }"></div>
        </div>
        <div class="progress-label">{{ progressText }}</div>
      </div>
      
      <div class="memory-message" :class="statusClass">
        {{ statusMessage }}
      </div>
    </div>
    
    <div v-else class="memory-loading">
      <i class="pi pi-spin pi-spinner"></i>
      <span>Loading data...</span>
    </div>
  </div>
</template>

<script setup>
import { computed } from 'vue'

const props = defineProps({
  title: {
    type: String,
    required: true
  },
  currentValue: {
    type: Number,
    default: null
  },
  estimatedValue: {
    type: Number,
    default: null
  },
  totalCapacity: {
    type: Number,
    required: true
  },
  loading: {
    type: Boolean,
    default: false
  }
})

const formatFileSize = (bytes) => {
  if (typeof bytes !== 'number' || Number.isNaN(bytes)) return '0 B'
  if (bytes <= 0) return '0 B'
  const k = 1024
  const sizes = ['B', 'KB', 'MB', 'GB', 'TB', 'PB']
  const i = Math.min(sizes.length - 1, Math.floor(Math.log(bytes) / Math.log(k)))
  const value = bytes / Math.pow(k, i)
  return `${value.toFixed(value >= 10 || i < 2 ? 0 : 2)} ${sizes[i]}`
}

const totalValue = computed(() => {
  const current = props.currentValue || 0
  const estimated = props.estimatedValue || 0
  return current + estimated
})

const status = computed(() => {
  if (props.loading) return 'unknown'
  const usagePercent = props.totalCapacity > 0 
    ? (totalValue.value / props.totalCapacity) * 100
    : 0
  
  if (usagePercent < 70) return 'good'
  if (usagePercent < 90) return 'warning'
  return 'critical'
})

const statusClass = computed(() => {
  const s = status.value
  if (s === 'good') return 'status-good'
  if (s === 'warning') return 'status-warning'
  if (s === 'critical') return 'status-critical'
  return 'status-unknown'
})

const statusText = computed(() => {
  const s = status.value
  if (s === 'good') return 'Fits Comfortably'
  if (s === 'warning') return 'Tight Fit'
  if (s === 'critical') return 'Won\'t Fit'
  return 'Unknown'
})

const currentPercent = computed(() => {
  const total = props.totalCapacity || 1
  const current = props.currentValue || 0
  return Math.min(100, Math.max(0, Math.round((current / total) * 100)))
})

const additionalPercent = computed(() => {
  const total = props.totalCapacity || 1
  const additional = props.estimatedValue || 0
  const pct = Math.round((additional / total) * 100)
  return Math.max(0, Math.min(100 - currentPercent.value, pct))
})

const progressText = computed(() => {
  return `${currentPercent.value}% used + ${additionalPercent.value}% est • ${formatFileSize(totalValue.value)} total`
})

const statusMessage = computed(() => {
  const s = status.value
  if (s === 'good') {
    const available = props.totalCapacity - totalValue.value
    return `✅ Fits Comfortably - ${formatFileSize(available)} buffer remaining`
  }
  if (s === 'warning') {
    return '⚠️ Usage is high - consider optimizing configuration'
  }
  if (s === 'critical') {
    return '❌ Usage exceeds capacity - configuration will not work'
  }
  return 'Loading information...'
})
</script>

<style scoped>
.memory-status-card {
  background: var(--gradient-card);
  border: 2px solid var(--border-primary);
  border-radius: var(--radius-xl);
  padding: var(--spacing-xl);
  transition: all var(--transition-normal);
  position: relative;
  overflow: hidden;
}

.memory-status-card::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  height: 4px;
  background: var(--border-primary);
  transition: all var(--transition-normal);
}

.memory-status-card.status-good::before {
  background: linear-gradient(90deg, #22c55e, #16a34a);
}

.memory-status-card.status-warning::before {
  background: linear-gradient(90deg, #f59e0b, #d97706);
}

.memory-status-card.status-critical::before {
  background: linear-gradient(90deg, #ef4444, #dc2626);
}

.memory-card-header {
  display: flex;
  align-items: center;
  gap: var(--spacing-md);
  margin-bottom: var(--spacing-lg);
}

.memory-status-icon {
  width: 48px;
  height: 48px;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: var(--radius-lg);
  font-size: 1.5rem;
  flex-shrink: 0;
}

.memory-status-card.status-good .memory-status-icon {
  background: rgba(34, 197, 94, 0.15);
  color: #22c55e;
}

.memory-status-card.status-warning .memory-status-icon {
  background: rgba(245, 158, 11, 0.15);
  color: #f59e0b;
}

.memory-status-card.status-critical .memory-status-icon {
  background: rgba(239, 68, 68, 0.15);
  color: #ef4444;
}

.memory-card-title {
  flex: 1;
}

.memory-card-title h4 {
  margin: 0 0 var(--spacing-xs) 0;
  font-size: 1.25rem;
  font-weight: 600;
  color: var(--text-primary);
}

.memory-status-badge {
  display: inline-block;
  padding: var(--spacing-xs) var(--spacing-sm);
  border-radius: var(--radius-sm);
  font-size: 0.875rem;
  font-weight: 600;
  letter-spacing: 0.5px;
}

.memory-status-card.status-good .memory-status-badge {
  background: rgba(34, 197, 94, 0.15);
  color: #22c55e;
}

.memory-status-card.status-warning .memory-status-badge {
  background: rgba(245, 158, 11, 0.15);
  color: #f59e0b;
}

.memory-status-card.status-critical .memory-status-badge {
  background: rgba(239, 68, 68, 0.15);
  color: #ef4444;
}

.memory-status-content {
  display: flex;
  flex-direction: column;
  gap: var(--spacing-md);
}

.memory-usage-display {
  display: flex;
  flex-direction: column;
  gap: var(--spacing-sm);
  padding: var(--spacing-md);
  background: var(--bg-surface);
  border-radius: var(--radius-md);
  min-width: 0;
  box-sizing: border-box;
}

.usage-item {
  display: flex;
  align-items: center;
  gap: var(--spacing-sm);
  font-size: 0.9rem;
  min-width: 0;
  flex-wrap: wrap;
}

.usage-value {
  flex-shrink: 0;
  white-space: nowrap;
}

.usage-fraction {
  flex-shrink: 0;
  white-space: nowrap;
}

.usage-item.total {
  font-weight: 600;
  padding-top: var(--spacing-sm);
  border-top: 1px solid var(--border-primary);
  font-size: 1rem;
}

.usage-label {
  color: var(--text-secondary);
  min-width: 80px;
}

.usage-value {
  color: var(--text-primary);
  font-weight: 500;
  flex: 1;
}

.usage-item.total .usage-value {
  font-weight: 700;
  color: var(--accent-cyan);
}

.usage-fraction {
  color: var(--text-secondary);
  font-size: 0.875rem;
}

.memory-progress-bar {
  display: flex;
  flex-direction: column;
  gap: var(--spacing-sm);
}

.progress-label {
  font-size: 0.875rem;
  color: var(--text-secondary);
  text-align: center;
}

.memory-message {
  padding: var(--spacing-md);
  border-radius: var(--radius-md);
  font-size: 0.9rem;
  line-height: 1.5;
  font-weight: 500;
}

.memory-message.status-good {
  background: rgba(34, 197, 94, 0.1);
  color: #22c55e;
  border: 1px solid rgba(34, 197, 94, 0.2);
}

.memory-message.status-warning {
  background: rgba(245, 158, 11, 0.1);
  color: #f59e0b;
  border: 1px solid rgba(245, 158, 11, 0.2);
}

.memory-message.status-critical {
  background: rgba(239, 68, 68, 0.1);
  color: #ef4444;
  border: 1px solid rgba(239, 68, 68, 0.2);
}

.memory-loading {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: var(--spacing-md);
  padding: var(--spacing-xl);
  color: var(--text-secondary);
}

.memory-loading i {
  font-size: 2rem;
  color: var(--accent-primary);
  animation: spin 1s linear infinite;
}

.stacked-bar {
  position: relative;
  width: 100%;
  height: 10px;
  background: var(--bg-secondary);
  border-radius: 5px;
  overflow: hidden;
}

.stacked-bar .bar-current {
  position: absolute;
  left: 0;
  top: 0;
  bottom: 0;
  background: var(--accent-red);
}

.stacked-bar .bar-additional {
  position: absolute;
  top: 0;
  bottom: 0;
  background: linear-gradient(90deg, var(--accent-blue), var(--accent-cyan));
  opacity: 0.8;
  transform-origin: left;
}

.stacked-bar.success {
  box-shadow: inset 0 0 0 1px rgba(16, 185, 129, 0.2);
}

.stacked-bar.warning {
  box-shadow: inset 0 0 0 1px rgba(234, 179, 8, 0.3);
}

@keyframes spin {
  from {
    transform: rotate(0deg);
  }
  to {
    transform: rotate(360deg);
  }
}
</style>

