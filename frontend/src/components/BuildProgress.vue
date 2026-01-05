<template>
  <div class="build-progress" v-if="builds.length > 0">
    <div class="build-header">
      <h3>Active Builds</h3>
      <Button 
        icon="pi pi-times" 
        @click="clearCompleted"
        severity="secondary"
        text
        size="small"
        :disabled="!hasCompletedBuilds"
      />
    </div>
    
    <div class="build-list">
      <div 
        v-for="build in builds" 
        :key="build.task_id"
        class="build-item"
        :class="{ 'completed': build.progress === 100, 'error': build.error }"
      >
        <div class="build-info">
          <div class="build-title">{{ build.title || 'llama.cpp Build' }}</div>
          <div class="build-stage">{{ build.stage }}</div>
          <div v-if="build.message" class="build-message">{{ build.message }}</div>
        </div>
        
        <div class="build-progress-bar">
          <ProgressBar 
            :value="build.progress" 
            :showValue="false"
            class="progress-bar"
          />
          <span class="progress-text">{{ build.progress }}%</span>
        </div>
        
        <div class="build-status">
          <i 
            v-if="build.progress === 100" 
            class="pi pi-check-circle status-icon success"
          ></i>
          <i 
            v-else-if="build.error" 
            class="pi pi-times-circle status-icon error"
          ></i>
          <i 
            v-else 
            class="pi pi-spin pi-spinner status-icon in-progress"
          ></i>
        </div>
        
        <div class="build-actions">
          <Button 
            icon="pi pi-eye" 
            @click="toggleLogs(build)"
            severity="secondary"
            text
            size="small"
            :label="build.showLogs ? 'Hide Logs' : 'Show Logs'"
          />
          <Button 
            v-if="build.error"
            icon="pi pi-refresh" 
            @click="retryBuild(build)"
            severity="secondary"
            text
            size="small"
          />
        </div>
        
        <!-- Build Logs -->
        <div v-if="build.showLogs && build.log_lines.length > 0" class="build-logs">
          <div class="logs-header">
            <span>Build Logs</span>
            <Button 
              icon="pi pi-times" 
              @click="build.showLogs = false"
              severity="secondary"
              text
              size="small"
            />
          </div>
          <div class="logs-content">
            <pre v-for="(line, index) in build.log_lines" :key="index" 
                 :class="getLogLineClass(line)">{{ line }}</pre>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, onUnmounted } from 'vue'
import { useWebSocketStore } from '@/stores/websocket'
import { useSystemStore } from '@/stores/system'
import { toast } from 'vue3-toastify'
import Button from 'primevue/button'
import ProgressBar from 'primevue/progressbar'

const wsStore = useWebSocketStore()
const systemStore = useSystemStore()

const builds = ref([])
const unsubscribe = ref(null)
const unsubscribeNotifications = ref(null)

const hasCompletedBuilds = computed(() => 
  builds.value.some(b => b.progress === 100 || b.error)
)

onMounted(() => {
  // Subscribe to build progress updates
  unsubscribe.value = wsStore.subscribeToBuildProgress(handleBuildProgress)
  
  // Also subscribe to notifications for completion events
  unsubscribeNotifications.value = wsStore.subscribeToNotifications(handleNotification)
})

onUnmounted(() => {
  if (unsubscribe.value) {
    unsubscribe.value()
  }
  if (unsubscribeNotifications.value) {
    unsubscribeNotifications.value()
  }
})

const handleBuildProgress = (data) => {
  const existingIndex = builds.value.findIndex(b => b.task_id === data.task_id)
  
  if (existingIndex >= 0) {
    // Update existing build
    const existing = builds.value[existingIndex]
    builds.value[existingIndex] = {
      ...existing,
      ...data,
      log_lines: [...(existing.log_lines || []), ...(data.log_lines || [])],
      showLogs: existing.showLogs || false
    }
  } else {
    // Add new build
    builds.value.push({
      ...data,
      log_lines: data.log_lines || [],
      showLogs: false,
      error: false,
      title: extractBuildTitle(data.message) || 'llama.cpp Build'
    })
  }
  
  // Show completion notification and refresh versions list
  if (data.progress === 100) {
    toast.success(`${extractBuildTitle(data.message) || 'Build'} completed successfully`)
    // Refresh the versions list to show the newly built version
    systemStore.fetchLlamaVersions()
  }
  
  // Also refresh versions list on error (in case of partial installations)
  if (data.stage === 'error') {
    toast.error(`${extractBuildTitle(data.message) || 'Build'} failed`)
    // Refresh versions list even on error to ensure UI is up to date
    systemStore.fetchLlamaVersions()
  }
}

const handleNotification = (data) => {
  // Handle completion notifications from backend
  if (data.notification_type === 'success' && 
      (data.title.includes('Build Complete') || data.title.includes('Installation Complete'))) {
    // Refresh versions list when build/installation completes
    systemStore.fetchLlamaVersions()
  }
  
  // Also refresh on error notifications
  if (data.notification_type === 'error' && 
      (data.title.includes('Build Failed') || data.title.includes('Installation Failed'))) {
    // Refresh versions list even on error to ensure UI is up to date
    systemStore.fetchLlamaVersions()
  }
}

const extractBuildTitle = (message) => {
  if (!message) return null
  
  // Try to extract build title from message
  if (message.includes('llama.cpp')) return 'llama.cpp Build'
  if (message.includes('install')) return 'Installation'
  if (message.includes('compile')) return 'Compilation'
  return null
}

const getLogLineClass = (line) => {
  if (line.includes('error') || line.includes('Error') || line.includes('ERROR')) {
    return 'log-error'
  }
  if (line.includes('warning') || line.includes('Warning') || line.includes('WARNING')) {
    return 'log-warning'
  }
  if (line.includes('info') || line.includes('Info') || line.includes('INFO')) {
    return 'log-info'
  }
  return 'log-normal'
}

const toggleLogs = (build) => {
  build.showLogs = !build.showLogs
}

const clearCompleted = () => {
  builds.value = builds.value.filter(b => b.progress < 100 && !b.error)
}

const retryBuild = (build) => {
  // Remove from list and let the user retry manually
  const index = builds.value.findIndex(b => b.task_id === build.task_id)
  if (index >= 0) {
    builds.value.splice(index, 1)
  }
  
  toast.info('Please try building again from the llama.cpp manager')
}
</script>

<style scoped>
.build-progress {
  background: var(--gradient-card);
  border: 1px solid var(--border-primary);
  border-radius: var(--radius-xl);
  padding: 1rem;
  margin-bottom: 1rem;
  box-shadow: var(--shadow-md);
}

.build-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 1rem;
}

.build-header h3 {
  margin: 0;
  color: var(--text-primary);
  font-size: 1rem;
  font-weight: 600;
}

.build-list {
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
}

.build-item {
  display: grid;
  grid-template-columns: 1fr auto auto auto;
  gap: 1rem;
  align-items: center;
  padding: 0.75rem;
  background: var(--bg-card);
  border: 1px solid var(--border-primary);
  border-radius: var(--radius-lg);
  transition: all 0.2s ease;
}

.build-item.completed {
  background: var(--bg-card);
  border-color: var(--status-success);
  opacity: 0.9;
}

.build-item.error {
  background: var(--bg-card);
  border-color: var(--status-error);
  opacity: 0.9;
}

.build-info {
  min-width: 0;
}

.build-title {
  font-weight: 600;
  color: var(--text-primary);
  margin-bottom: 0.25rem;
}

.build-stage {
  font-size: 0.875rem;
  color: var(--text-secondary);
  margin-bottom: 0.25rem;
}

.build-message {
  font-size: 0.75rem;
  color: var(--text-secondary);
  font-style: italic;
}

.build-progress-bar {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  min-width: 120px;
}

.progress-bar {
  flex: 1;
}

.progress-text {
  font-size: 0.875rem;
  font-weight: 600;
  color: var(--text-primary);
  min-width: 35px;
  text-align: right;
}

.build-status {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 24px;
}

.build-actions {
  display: flex;
  gap: 0.25rem;
}

.build-logs {
  grid-column: 1 / -1;
  margin-top: 0.75rem;
  background: var(--bg-tertiary);
  border: 1px solid var(--border-primary);
  border-radius: var(--radius-lg);
  overflow: hidden;
}

.logs-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0.5rem 0.75rem;
  background: var(--bg-surface);
  border-bottom: 1px solid var(--border-primary);
  font-size: 0.875rem;
  font-weight: 600;
  color: var(--text-primary);
}

.logs-content {
  max-height: 200px;
  overflow-y: auto;
  padding: 0.5rem;
}

.logs-content pre {
  margin: 0;
  padding: 0.25rem 0;
  font-family: 'Courier New', monospace;
  font-size: 0.75rem;
  line-height: 1.4;
  white-space: pre-wrap;
  word-break: break-word;
}

.log-error {
  color: var(--status-error);
  background: var(--status-error-soft);
  padding: 0.25rem;
  border-radius: var(--radius-sm);
  margin: 0.125rem 0;
}

.log-warning {
  color: var(--status-warning);
  background: var(--status-warning-soft);
  padding: 0.25rem;
  border-radius: var(--radius-sm);
  margin: 0.125rem 0;
}

.log-info {
  color: var(--status-info);
  background: var(--status-info-soft);
  padding: 0.25rem;
  border-radius: var(--radius-sm);
  margin: 0.125rem 0;
}

.log-normal {
  color: var(--text-secondary);
}

.status-icon {
  font-size: 1.25rem;
}

.status-icon.success {
  color: var(--status-success);
}

.status-icon.error {
  color: var(--status-error);
}

.status-icon.in-progress {
  color: var(--status-info);
}

@media (max-width: 768px) {
  .build-item {
    grid-template-columns: 1fr;
    gap: 0.5rem;
  }
  
  .build-progress-bar {
    min-width: auto;
  }
  
  .build-actions {
    justify-content: flex-start;
  }
}
</style>
