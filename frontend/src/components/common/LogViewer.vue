<template>
  <div class="log-viewer" :class="{ 'compact': compact, 'no-header': !showHeader }">
    <div v-if="showHeader" class="log-viewer-header">
      <div class="log-viewer-controls">
        <Button 
          v-if="showAutoScroll"
          :icon="autoScroll ? 'pi pi-pause' : 'pi pi-play'"
          @click="autoScroll = !autoScroll"
          text
          size="small"
          :label="autoScroll ? 'Auto-scroll ON' : 'Auto-scroll OFF'"
        />
        <Button 
          v-if="showClear"
          icon="pi pi-trash" 
          @click="handleClear"
          text 
          size="small"
          label="Clear"
        />
        <span v-if="showCount" class="log-count">{{ logCount }} {{ logCount === 1 ? 'line' : 'lines' }}</span>
      </div>
    </div>
    
    <div class="log-viewer-content" ref="logContainer">
      <!-- Structured logs mode (array of objects) -->
      <template v-if="displayMode === 'structured' && structuredLogs.length > 0">
        <div 
          v-for="(log, index) in structuredLogs" 
          :key="log.id || index"
          class="log-entry"
          :class="`log-${log.log_type || 'combined'}`"
        >
          <span v-if="log.timestamp" class="log-time">{{ formatTime(log.timestamp) }}</span>
          <span v-if="log.log_type" class="log-type-badge">{{ log.log_type }}</span>
          <span class="log-data">{{ log.data || log.line || log }}</span>
        </div>
      </template>
      
      <!-- Raw logs mode (array of strings or string) -->
      <template v-else-if="displayMode === 'raw' && rawLogs.length > 0">
        <pre 
          v-for="(line, index) in rawLogs" 
          :key="index"
          :class="getLogLineClass(line)"
        >{{ line }}</pre>
      </template>
      
      <!-- Empty state -->
      <div v-if="isEmpty" class="no-logs">
        <i class="pi pi-info-circle"></i>
        <span>{{ emptyMessage || 'No logs available' }}</span>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, watch, nextTick, onMounted } from 'vue'
import Button from 'primevue/button'

const props = defineProps({
  // Accept multiple input formats
  logs: {
    type: [Array, String],
    default: () => []
  },
  // Display mode: 'auto' (detect), 'structured' (array of objects), 'raw' (strings)
  mode: {
    type: String,
    default: 'auto'
  },
  // Display options
  showHeader: {
    type: Boolean,
    default: true
  },
  showAutoScroll: {
    type: Boolean,
    default: true
  },
  showClear: {
    type: Boolean,
    default: true
  },
  showCount: {
    type: Boolean,
    default: true
  },
  compact: {
    type: Boolean,
    default: false
  },
  maxHeight: {
    type: String,
    default: '400px'
  },
  emptyMessage: {
    type: String,
    default: null
  }
})

const emit = defineEmits(['clear'])

const autoScroll = ref(true)
const logContainer = ref(null)

// Detect display mode
const displayMode = computed(() => {
  if (props.mode !== 'auto') {
    return props.mode
  }
  
  if (!props.logs || (Array.isArray(props.logs) && props.logs.length === 0)) {
    return 'raw'
  }
  
  // If it's a string, it's raw
  if (typeof props.logs === 'string') {
    return 'raw'
  }
  
  // If it's an array, check first element
  if (Array.isArray(props.logs) && props.logs.length > 0) {
    const first = props.logs[0]
    // If first element is an object with 'data' or 'line' property, it's structured
    if (typeof first === 'object' && (first.data !== undefined || first.line !== undefined)) {
      return 'structured'
    }
    // Otherwise it's raw strings
    return 'raw'
  }
  
  return 'raw'
})

// Process logs based on mode
const structuredLogs = computed(() => {
  if (displayMode.value !== 'structured') return []
  if (!Array.isArray(props.logs)) return []
  return props.logs.filter(Boolean)
})

const rawLogs = computed(() => {
  if (displayMode.value !== 'raw') return []
  
  if (typeof props.logs === 'string') {
    return props.logs.split('\n').filter(Boolean)
  }
  
  if (Array.isArray(props.logs)) {
    return props.logs.filter(Boolean)
  }
  
  return []
})

const isEmpty = computed(() => {
  if (displayMode.value === 'structured') {
    return structuredLogs.value.length === 0
  }
  return rawLogs.value.length === 0
})

const logCount = computed(() => {
  if (displayMode.value === 'structured') {
    return structuredLogs.value.length
  }
  return rawLogs.value.length
})

// Auto-scroll functionality
const scrollToBottom = () => {
  if (logContainer.value && autoScroll.value) {
    logContainer.value.scrollTop = logContainer.value.scrollHeight
  }
}

// Watch for log changes
watch(() => props.logs, async () => {
  if (autoScroll.value && logContainer.value) {
    await nextTick()
    scrollToBottom()
  }
}, { deep: true })

watch(() => logCount.value, async () => {
  if (autoScroll.value && logContainer.value) {
    await nextTick()
    scrollToBottom()
  }
})

// Scroll on mount
onMounted(() => {
  if (autoScroll.value && !isEmpty.value) {
    nextTick(() => scrollToBottom())
  }
})

const formatTime = (timestamp) => {
  if (!timestamp) return ''
  try {
    return new Date(timestamp).toLocaleTimeString()
  } catch {
    return timestamp
  }
}

const getLogLineClass = (line) => {
  if (!line || typeof line !== 'string') return 'log-normal'
  
  const lowerLine = line.toLowerCase()
  if (lowerLine.includes('error') || lowerLine.includes('failed') || lowerLine.includes('exception')) {
    return 'log-error'
  }
  if (lowerLine.includes('warning') || lowerLine.includes('warn')) {
    return 'log-warning'
  }
  if (lowerLine.includes('info') || lowerLine.includes('success')) {
    return 'log-info'
  }
  return 'log-normal'
}

const handleClear = () => {
  emit('clear')
}
</script>

<style scoped>
.log-viewer {
  background: var(--bg-card);
  border: 1px solid var(--border-primary);
  border-radius: var(--radius-xl);
  overflow: hidden;
  display: flex;
  flex-direction: column;
}

.log-viewer.compact {
  border-radius: var(--radius-md);
}

.log-viewer.no-header {
  border: none;
  background: transparent;
}

.log-viewer-header {
  background: var(--bg-secondary);
  border-bottom: 1px solid var(--border-primary);
  padding: var(--spacing-sm) var(--spacing-md);
}

.log-viewer-controls {
  display: flex;
  align-items: center;
  gap: var(--spacing-sm);
}

.log-count {
  margin-left: auto;
  font-size: 0.875rem;
  color: var(--text-secondary);
}

.log-viewer-content {
  overflow-y: auto;
  font-family: 'Courier New', monospace;
  font-size: 0.875rem;
  background: var(--bg-tertiary);
  color: var(--text-primary);
  border-radius: var(--radius-md);
  box-shadow: inset 0 2px 8px rgba(0, 0, 0, 0.3);
}

.log-viewer:not(.compact) .log-viewer-content {
  max-height: v-bind(maxHeight);
}

.log-viewer.compact .log-viewer-content {
  max-height: 200px;
  padding: var(--spacing-xs);
}

/* Structured log entry styles */
.log-entry {
  display: flex;
  align-items: center;
  gap: var(--spacing-sm);
  padding: var(--spacing-xs) var(--spacing-sm);
  border-bottom: 1px solid var(--border-primary);
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
  padding: var(--spacing-xs) var(--spacing-xs);
  border-radius: var(--radius-sm);
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

/* Raw log pre styles */
.log-viewer-content pre {
  margin: 0;
  padding: var(--spacing-xs) var(--spacing-sm);
  font-family: 'Courier New', monospace;
  font-size: 0.875rem;
  line-height: 1.4;
  white-space: pre-wrap;
  word-break: break-word;
  border-bottom: 1px solid var(--border-primary);
}

.log-viewer.compact .log-viewer-content pre {
  padding: 0.125rem var(--spacing-xs);
  font-size: 0.75rem;
}

/* Log level classes */
.log-error {
  color: var(--status-error);
  background: var(--status-error-soft);
  padding: var(--spacing-xs);
  border-radius: var(--radius-sm);
  margin: 0.125rem 0;
}

.log-warning {
  color: var(--status-warning);
  background: var(--status-warning-soft);
  padding: var(--spacing-xs);
  border-radius: var(--radius-sm);
  margin: 0.125rem 0;
}

.log-info {
  color: var(--status-info);
  background: var(--status-info-soft);
  padding: var(--spacing-xs);
  border-radius: var(--radius-sm);
  margin: 0.125rem 0;
}

.log-normal {
  color: var(--text-secondary);
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
.log-viewer-content {
  background: #1a1a1a;
}

.log-entry {
  border-bottom-color: #333;
}

.log-entry:hover {
  background-color: #2a2a2a;
}

.log-viewer-content pre {
  border-bottom-color: #333;
}

/* Light theme adjustments */
:root[data-theme="light"] .log-viewer-content {
  background: #f8f9fa;
  color: #333;
}

:root[data-theme="light"] .log-entry {
  border-bottom-color: #dee2e6;
}

:root[data-theme="light"] .log-entry:hover {
  background-color: #e9ecef;
}

:root[data-theme="light"] .log-viewer-content pre {
  border-bottom-color: #dee2e6;
}

:root[data-theme="light"] .log-time {
  color: #6c757d;
}

@keyframes slideInUp {
  from {
    opacity: 0;
    transform: translateY(10px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}
</style>
