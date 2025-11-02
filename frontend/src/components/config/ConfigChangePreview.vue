<template>
  <Dialog 
    :visible="visible" 
    @update:visible="$emit('update:visible', $event)"
    :modal="true"
    :style="{ width: '600px' }"
    :header="`Preview ${type} Changes`"
    :dismissableMask="true"
    class="config-preview-dialog"
    aria-label="Configuration change preview"
    @touchstart="handleTouchStart"
    @touchmove="handleTouchMove"
    @touchend="handleTouchEnd"
  >
    <div class="preview-content">
      <div class="preview-header">
        <i class="pi pi-info-circle" aria-hidden="true"></i>
        <p>{{ type === 'preset' ? `${presetName} preset will change:` : 'Smart Auto will change:' }}</p>
      </div>

      <div class="changes-list">
        <div 
          v-for="change in changes" 
          :key="change.field"
          class="change-item"
          :class="{ 'has-impact': change.impact }"
        >
          <div class="change-field">
            <strong>{{ change.field }}</strong>
          </div>
          <div class="change-values">
            <span class="value-before">{{ formatValue(change.before) }}</span>
            <i class="pi pi-arrow-right" aria-hidden="true"></i>
            <span class="value-after">{{ formatValue(change.after) }}</span>
          </div>
          <div v-if="change.description" class="change-description">
            {{ change.description }}
          </div>
        </div>
      </div>

      <div v-if="impact" class="impact-preview">
        <h4>Expected Impact:</h4>
        <div class="impact-items">
          <div v-if="impact.performance" class="impact-item performance">
            <i class="pi pi-chart-line" aria-hidden="true"></i>
            <span>{{ impact.performance }}</span>
          </div>
          <div v-if="impact.vram" class="impact-item vram">
            <i class="pi pi-memory" aria-hidden="true"></i>
            <span>{{ impact.vram }}</span>
          </div>
          <div v-if="impact.ram" class="impact-item ram">
            <i class="pi pi-server" aria-hidden="true"></i>
            <span>{{ impact.ram }}</span>
          </div>
        </div>
      </div>

      <div class="preview-warning" v-if="hasWarnings">
        <i class="pi pi-exclamation-triangle" aria-hidden="true"></i>
        <p>Some changes may affect memory usage. Review the impact above.</p>
      </div>
    </div>

    <template #footer>
      <Button 
        label="Cancel" 
        icon="pi pi-times" 
        @click="$emit('cancel')"
        severity="secondary"
        outlined
        aria-label="Cancel configuration changes"
      />
      <Button 
        label="Apply Changes" 
        icon="pi pi-check" 
        @click="$emit('apply')"
        :loading="applying"
        aria-label="Apply configuration changes"
      />
    </template>
  </Dialog>
</template>

<script setup>
import { computed, ref } from 'vue'
import Dialog from 'primevue/dialog'
import Button from 'primevue/button'

const props = defineProps({
  visible: {
    type: Boolean,
    default: false
  },
  type: {
    type: String,
    default: 'smart-auto' // 'smart-auto' or 'preset'
  },
  presetName: {
    type: String,
    default: ''
  },
  changes: {
    type: Array,
    default: () => []
  },
  impact: {
    type: Object,
    default: null
  },
  applying: {
    type: Boolean,
    default: false
  }
})

const emit = defineEmits(['update:visible', 'apply', 'cancel'])

// Touch gesture handling for swipe to dismiss
const touchStartX = ref(0)
const touchStartY = ref(0)
const touchThreshold = 50

const handleTouchStart = (e) => {
  if (e.touches && e.touches.length > 0) {
    touchStartX.value = e.touches[0].clientX
    touchStartY.value = e.touches[0].clientY
  }
}

const handleTouchMove = (e) => {
  if (e.touches && e.touches.length > 0) {
    const deltaX = e.touches[0].clientX - touchStartX.value
    const deltaY = e.touches[0].clientY - touchStartY.value
    
    if (deltaY > touchThreshold && Math.abs(deltaX) < Math.abs(deltaY)) {
      e.preventDefault()
    }
  }
}

const handleTouchEnd = (e) => {
  if (e.changedTouches && e.changedTouches.length > 0) {
    const deltaX = e.changedTouches[0].clientX - touchStartX.value
    const deltaY = e.changedTouches[0].clientY - touchStartY.value
    
    if (deltaY > touchThreshold && Math.abs(deltaX) < Math.abs(deltaY)) {
      emit('cancel')
    }
  }
  
  touchStartX.value = 0
  touchStartY.value = 0
}

const formatValue = (value) => {
  if (value === null || value === undefined) return 'Not set'
  if (typeof value === 'boolean') return value ? 'Enabled' : 'Disabled'
  if (typeof value === 'number') {
    if (value >= 1000 && value < 1000000) return `${(value / 1000).toFixed(1)}K`
    if (value >= 1000000) return `${(value / 1000000).toFixed(1)}M`
    return value.toString()
  }
  return value.toString()
}

const hasWarnings = computed(() => {
  return props.impact && (props.impact.vram || props.impact.ram)
})
</script>

<style scoped>
.preview-content {
  display: flex;
  flex-direction: column;
  gap: var(--spacing-lg);
}

.preview-header {
  display: flex;
  align-items: center;
  gap: var(--spacing-md);
  padding: var(--spacing-md);
  background: rgba(34, 211, 238, 0.1);
  border: 1px solid rgba(34, 211, 238, 0.2);
  border-radius: var(--radius-md);
}

.preview-header i {
  font-size: 1.5rem;
  color: var(--accent-cyan);
}

.preview-header p {
  margin: 0;
  font-size: 1rem;
  color: var(--text-primary);
  font-weight: 500;
}

.changes-list {
  display: flex;
  flex-direction: column;
  gap: var(--spacing-md);
  max-height: 400px;
  overflow-y: auto;
}

.change-item {
  display: flex;
  flex-direction: column;
  gap: var(--spacing-xs);
  padding: var(--spacing-md);
  background: var(--bg-surface);
  border: 1px solid var(--border-primary);
  border-radius: var(--radius-md);
  transition: all var(--transition-normal);
}

.change-item.has-impact {
  border-left: 3px solid var(--accent-cyan);
}

.change-field {
  font-weight: 600;
  color: var(--text-primary);
  font-size: 0.95rem;
}

.change-values {
  display: flex;
  align-items: center;
  gap: var(--spacing-sm);
  font-size: 0.9rem;
}

.value-before {
  color: var(--text-secondary);
  text-decoration: line-through;
}

.value-after {
  color: var(--accent-cyan);
  font-weight: 600;
}

.change-values i {
  color: var(--text-secondary);
  font-size: 0.8rem;
}

.change-description {
  font-size: 0.85rem;
  color: var(--text-secondary);
  font-style: italic;
  margin-top: var(--spacing-xs);
}

.impact-preview {
  padding: var(--spacing-md);
  background: var(--bg-surface);
  border: 1px solid var(--border-primary);
  border-radius: var(--radius-md);
}

.impact-preview h4 {
  margin: 0 0 var(--spacing-md) 0;
  font-size: 1rem;
  font-weight: 600;
  color: var(--text-primary);
}

.impact-items {
  display: flex;
  flex-direction: column;
  gap: var(--spacing-sm);
}

.impact-item {
  display: flex;
  align-items: center;
  gap: var(--spacing-sm);
  padding: var(--spacing-sm);
  border-radius: var(--radius-sm);
}

.impact-item.performance {
  background: rgba(34, 197, 94, 0.1);
  color: #22c55e;
}

.impact-item.vram {
  background: rgba(245, 158, 11, 0.1);
  color: #f59e0b;
}

.impact-item.ram {
  background: rgba(59, 130, 246, 0.1);
  color: var(--accent-blue);
}

.impact-item i {
  font-size: 1.1rem;
}

.preview-warning {
  display: flex;
  align-items: center;
  gap: var(--spacing-sm);
  padding: var(--spacing-md);
  background: rgba(245, 158, 11, 0.1);
  border: 1px solid rgba(245, 158, 11, 0.3);
  border-radius: var(--radius-md);
}

.preview-warning i {
  color: #f59e0b;
  font-size: 1.2rem;
  flex-shrink: 0;
}

.preview-warning p {
  margin: 0;
  color: var(--text-primary);
  font-size: 0.9rem;
}
</style>

