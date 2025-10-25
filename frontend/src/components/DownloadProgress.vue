<template>
  <div v-if="activeDownloads.length > 0" class="download-progress-container">
    <h3>Download Progress</h3>
    <div class="download-list">
      <div 
        v-for="download in activeDownloads" 
        :key="download.task_id"
        class="download-item"
      >
        <div class="download-header">
          <div class="download-info">
            <div class="download-filename">{{ download.filename }}</div>
            <div class="download-status">{{ download.message }}</div>
          </div>
          <div class="download-actions">
            <Button 
              icon="pi pi-times" 
              size="small" 
              severity="secondary" 
              text
              @click="removeDownload(download.task_id)"
              :disabled="download.progress < 100"
            />
          </div>
        </div>
        
        <div class="progress-container">
          <ProgressBar 
            :value="download.progress" 
            :showValue="false"
            class="progress-bar"
          />
          <div class="progress-text">
            <span class="progress-percentage">{{ download.progress }}%</span>
            <span class="progress-size">{{ formatBytes(download.bytes_downloaded) }} / {{ formatBytes(download.total_bytes) }}</span>
          </div>
        </div>
        
        <div v-if="download.speed_mbps > 0 || download.eta_seconds > 0" class="download-stats">
          <span v-if="download.speed_mbps > 0" class="speed">
            <i class="pi pi-download"></i>
            {{ (download.speed_mbps || 0).toFixed(1) }} MB/s
          </span>
          <span v-if="download.eta_seconds > 0" class="eta">
            <i class="pi pi-clock"></i>
            {{ formatTime(download.eta_seconds) }}
          </span>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, onUnmounted } from 'vue'
import { useWebSocketStore } from '@/stores/websocket'
import Button from 'primevue/button'
import ProgressBar from 'primevue/progressbar'

const wsStore = useWebSocketStore()
const activeDownloads = ref([])
const unsubscribe = ref(null)

onMounted(() => {
  // Subscribe to download progress updates
  unsubscribe.value = wsStore.subscribeToDownloadProgress(handleDownloadProgress)
})

onUnmounted(() => {
  if (unsubscribe.value) {
    unsubscribe.value()
  }
})

const handleDownloadProgress = (data) => {
  console.log('DownloadProgress received:', data) // Debug log
  const existingIndex = activeDownloads.value.findIndex(d => d.task_id === data.task_id)
  
  if (existingIndex >= 0) {
    // Update existing download
    activeDownloads.value[existingIndex] = {
      ...activeDownloads.value[existingIndex],
      ...data
    }
    
    // Remove completed downloads after a delay
    if (data.progress >= 100) {
      setTimeout(() => {
        removeDownload(data.task_id)
      }, 3000)
    }
  } else {
    // Add new download
    activeDownloads.value.push(data)
  }
}

const removeDownload = (taskId) => {
  const index = activeDownloads.value.findIndex(d => d.task_id === taskId)
  if (index >= 0) {
    activeDownloads.value.splice(index, 1)
  }
}

const formatBytes = (bytes) => {
  if (bytes === 0) return '0 B'
  const k = 1024
  const sizes = ['B', 'KB', 'MB', 'GB', 'TB']
  const i = Math.floor(Math.log(bytes) / Math.log(k))
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i]
}

const formatTime = (seconds) => {
  if (seconds < 60) {
    return `${seconds}s`
  } else if (seconds < 3600) {
    const minutes = Math.floor(seconds / 60)
    const remainingSeconds = seconds % 60
    return `${minutes}m ${remainingSeconds}s`
  } else {
    const hours = Math.floor(seconds / 3600)
    const minutes = Math.floor((seconds % 3600) / 60)
    return `${hours}h ${minutes}m`
  }
}
</script>

<style scoped>
.download-progress-container {
  margin-bottom: 2rem;
  padding: 1.5rem;
  background: rgba(255, 255, 255, 0.9);
  backdrop-filter: blur(10px);
  border-radius: 16px;
  border: 1px solid rgba(255, 255, 255, 0.3);
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
}

.download-progress-container h3 {
  margin: 0 0 1rem 0;
  color: var(--text-color);
  font-size: 1.1rem;
  font-weight: 600;
}

.download-list {
  display: flex;
  flex-direction: column;
  gap: 1rem;
}

.download-item {
  padding: 1rem;
  background: var(--surface-50);
  border-radius: var(--border-radius);
  border: 1px solid var(--surface-200);
}

.download-header {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  margin-bottom: 0.75rem;
}

.download-info {
  flex: 1;
}

.download-filename {
  font-weight: 600;
  color: var(--text-color);
  margin-bottom: 0.25rem;
}

.download-status {
  font-size: 0.875rem;
  color: var(--text-color-secondary);
}

.download-actions {
  margin-left: 1rem;
}

.progress-container {
  margin-bottom: 0.5rem;
}

.progress-bar {
  margin-bottom: 0.5rem;
}

.progress-text {
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-size: 0.875rem;
}

.progress-percentage {
  font-weight: 600;
  color: var(--primary-color);
}

.progress-size {
  color: var(--text-color-secondary);
}

.download-stats {
  display: flex;
  gap: 1rem;
  font-size: 0.75rem;
  color: var(--text-color-secondary);
}

.speed, .eta {
  display: flex;
  align-items: center;
  gap: 0.25rem;
}

.speed i, .eta i {
  font-size: 0.75rem;
}
</style>