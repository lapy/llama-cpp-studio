<template>
  <div class="logs-feed">
    <div class="logs-header">
      <div class="logs-controls">
        <Button 
          :icon="autoScroll ? 'pi pi-pause' : 'pi pi-play'"
          @click="autoScroll = !autoScroll"
          text
          size="small"
          :label="autoScroll ? 'Auto-scroll ON' : 'Auto-scroll OFF'"
        />
        <Button 
          icon="pi pi-trash" 
          @click="$emit('clear')" 
          text 
          size="small"
          label="Clear"
        />
        <span class="logs-count">{{ logs.length }} logs</span>
      </div>
    </div>
    
    <div class="logs-container" ref="logsContainer">
      <div 
        v-for="(log, index) in logs" 
        :key="index"
        class="log-entry"
        :class="`log-${log.log_type || 'combined'}`"
      >
        <span class="log-time">{{ formatTime(log.timestamp) }}</span>
        <span class="log-type-badge">{{ log.log_type || 'combined' }}</span>
        <span class="log-data">{{ log.data }}</span>
      </div>
      
      <div v-if="logs.length === 0" class="no-logs">
        <i class="pi pi-info-circle"></i>
        <span>No logs available</span>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, watch, nextTick } from 'vue'

const props = defineProps({
  logs: {
    type: Array,
    default: () => []
  }
})

const emit = defineEmits(['clear'])

const autoScroll = ref(true)
const logsContainer = ref(null)

watch(() => props.logs, async () => {
  if (autoScroll.value) {
    await nextTick()
    if (logsContainer.value) {
      logsContainer.value.scrollTop = logsContainer.value.scrollHeight
    }
  }
}, { deep: true })

const formatTime = (timestamp) => {
  if (!timestamp) return ''
  return new Date(timestamp).toLocaleTimeString()
}
</script>

<style scoped>
.logs-feed {
  background: var(--bg-card);
  border: 1px solid var(--border-primary);
  border-radius: 12px;
  overflow: hidden;
}

.logs-header {
  background: var(--bg-secondary);
  border-bottom: 1px solid var(--border-primary);
  padding: var(--spacing-sm) var(--spacing-md);
}

.logs-controls {
  display: flex;
  align-items: center;
  gap: var(--spacing-sm);
}

.logs-count {
  margin-left: auto;
  font-size: 0.875rem;
  color: var(--text-secondary);
}

.logs-container {
  max-height: 400px;
  overflow-y: auto;
  font-family: 'Courier New', monospace;
  font-size: 0.875rem;
  background: linear-gradient(135deg, #1a1a1a 0%, #2a2a2a 100%);
  color: #ffffff;
  border-radius: var(--radius-md);
  box-shadow: inset 0 2px 8px rgba(0, 0, 0, 0.3);
}

.log-entry {
  display: flex;
  align-items: center;
  gap: var(--spacing-sm);
  padding: var(--spacing-xs) var(--spacing-sm);
  border-bottom: 1px solid #333;
  transition: all 0.3s ease;
  position: relative;
  animation: slideInUp 0.3s ease-out;
}

.log-entry:hover {
  background: linear-gradient(90deg, rgba(34, 211, 238, 0.1) 0%, rgba(59, 130, 246, 0.1) 100%);
  transform: translateX(5px);
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
}

.log-entry::before {
  content: '';
  position: absolute;
  left: 0;
  top: 0;
  bottom: 0;
  width: 3px;
  background: var(--gradient-primary);
  opacity: 0;
  transition: opacity 0.3s ease;
}

.log-entry:hover::before {
  opacity: 1;
}

.log-time {
  color: #888;
  font-size: 0.75rem;
  min-width: 80px;
  flex-shrink: 0;
}

.log-type-badge {
  background: var(--accent-cyan);
  color: white;
  padding: 2px 6px;
  border-radius: 4px;
  font-size: 0.75rem;
  min-width: 60px;
  text-align: center;
  flex-shrink: 0;
}

.log-data {
  flex: 1;
  word-break: break-word;
  line-height: 1.4;
}

.no-logs {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: var(--spacing-sm);
  padding: var(--spacing-xl);
  color: var(--text-secondary);
  font-style: italic;
}

/* Dark theme specific styles */
.logs-container {
  background: #1a1a1a;
}

.log-entry {
  border-bottom-color: #333;
}

.log-entry:hover {
  background-color: #2a2a2a;
}

.log-time {
  color: #888;
}

/* Light theme adjustments */
:root[data-theme="light"] .logs-container {
  background: #f8f9fa;
  color: #333;
}

:root[data-theme="light"] .log-entry {
  border-bottom-color: #dee2e6;
}

:root[data-theme="light"] .log-entry:hover {
  background-color: #e9ecef;
}

:root[data-theme="light"] .log-time {
  color: #6c757d;
}
</style>
