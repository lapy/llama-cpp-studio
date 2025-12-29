<template>
  <div class="cuda-installer">
    <div class="card-header">
      <h3>CUDA Toolkit Manager</h3>
      <Button 
        icon="pi pi-refresh" 
        @click="refreshStatus"
        :loading="loading"
        severity="secondary"
        text
        v-tooltip="'Refresh CUDA status'"
      />
    </div>

    <!-- Current Installation Status -->
    <div v-if="cudaStatus" class="status-section">
      <div class="status-card" :class="{ 'installed': cudaStatus.installed }">
        <div class="status-header">
          <i :class="cudaStatus.installed ? 'pi pi-check-circle' : 'pi pi-times-circle'" 
             :style="{ color: cudaStatus.installed ? 'var(--green-500)' : 'var(--red-500)' }"></i>
          <h4>{{ cudaStatus.installed ? 'CUDA Installed' : 'CUDA Not Installed' }}</h4>
        </div>
        <div v-if="cudaStatus.installed" class="status-details">
          <div class="detail-row">
            <span class="detail-label">Current Version:</span>
            <span class="detail-value">{{ cudaStatus.version }}</span>
          </div>
          <div v-if="cudaStatus.cuda_path" class="detail-row">
            <span class="detail-label">Installation Path:</span>
            <span class="detail-value">{{ cudaStatus.cuda_path }}</span>
          </div>
          <div v-if="cudaStatus.installed_at" class="detail-row">
            <span class="detail-label">Installed:</span>
            <span class="detail-value">{{ formatDate(cudaStatus.installed_at) }}</span>
          </div>
        </div>
        <div v-else class="status-details">
          <p>CUDA Toolkit is required for CUDA-enabled builds. Install a version below to enable CUDA support.</p>
        </div>
      </div>
    </div>

    <!-- Installed Versions List -->
    <div v-if="cudaStatus?.installed_versions && cudaStatus.installed_versions.length > 0" class="installed-versions">
      <h4>Installed Versions</h4>
      <div class="version-list">
        <div 
          v-for="installed in cudaStatus.installed_versions" 
          :key="installed.version"
          class="version-card"
          :class="{ 'current': installed.is_current }"
        >
          <div class="version-header">
            <div class="version-info">
              <span class="version-label">CUDA {{ installed.version }}</span>
              <Tag v-if="installed.is_current" value="Current" severity="success" />
            </div>
            <Button
              icon="pi pi-trash"
              label="Uninstall"
              severity="danger"
              outlined
              size="small"
              @click="confirmUninstall(installed)"
              :disabled="installing || cudaStatus?.operation"
            />
          </div>
          <div class="version-details">
            <div class="detail-row">
              <span class="detail-label">Path:</span>
              <span class="detail-value">{{ installed.path }}</span>
            </div>
            <div v-if="installed.installed_at" class="detail-row">
              <span class="detail-label">Installed:</span>
              <span class="detail-value">{{ formatDate(installed.installed_at) }}</span>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Installation Section -->
    <div class="install-section">
      <h4>Install CUDA Toolkit</h4>
      <div class="install-form">
        <div class="form-group">
          <label>CUDA Version</label>
          <Select
            v-model="selectedCudaVersion"
            :options="availableVersions"
            placeholder="Select CUDA version"
            class="version-select"
          />
        </div>
        <div class="form-info">
          <p><strong>Platform:</strong> {{ cudaStatus?.platform?.[0] || 'Unknown' }} / {{ cudaStatus?.platform?.[1] || 'Unknown' }}</p>
          <p v-if="cudaStatus?.platform?.[0] === 'linux'" class="info-text">
            Linux installation may require appropriate permissions. The installer will run in silent mode.
          </p>
        </div>
      </div>

      <!-- Installation Progress -->
      <div v-if="cudaInstallProgress" class="install-progress">
        <ProgressBar :value="cudaInstallProgress.progress || 0" />
        <p class="progress-message">{{ cudaInstallProgress.message || 'Preparing installation...' }}</p>
      </div>

      <!-- Installation Logs -->
      <div v-if="cudaInstallLogs.length > 0" class="install-logs">
        <div class="logs-header">
          <h5>Installation Logs</h5>
          <Button
            icon="pi pi-times"
            @click="cudaInstallLogs = []"
            severity="secondary"
            text
            size="small"
          />
        </div>
        <div class="logs-content">
          <pre>{{ cudaInstallLogs.join('\n') }}</pre>
        </div>
      </div>

      <!-- Action Buttons -->
      <div class="action-buttons">
        <Button
          v-if="!installing && !cudaStatus?.operation"
          label="Install CUDA" 
          icon="pi pi-download"
          @click="handleInstall"
          :disabled="!selectedCudaVersion"
          severity="success"
        />
        <Button
          v-if="installing || cudaStatus?.operation"
          label="Installing..."
          icon="pi pi-spin pi-spinner"
          disabled
        />
      </div>
    </div>

    <!-- Operation Status -->
    <div v-if="cudaStatus?.operation" class="operation-status">
      <Message severity="info" :closable="false">
        <template #messageicon>
          <i class="pi pi-info-circle"></i>
        </template>
        Operation in progress: {{ cudaStatus.operation }}
        <span v-if="cudaStatus.operation_started_at">
          (Started: {{ formatDate(cudaStatus.operation_started_at) }})
        </span>
      </Message>
    </div>

    <!-- Error Message -->
    <div v-if="cudaStatus?.last_error" class="error-message">
      <Message severity="error" :closable="true" @close="cudaStatus.last_error = null">
        <template #messageicon>
          <i class="pi pi-exclamation-triangle"></i>
        </template>
        {{ cudaStatus.last_error }}
      </Message>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, onUnmounted } from 'vue'
import { useSystemStore } from '@/stores/system'
import { useWebSocketStore } from '@/stores/websocket'
import { useConfirm } from 'primevue/useconfirm'
import { toast } from 'vue3-toastify'
import Button from 'primevue/button'
import Select from 'primevue/select'
import ProgressBar from 'primevue/progressbar'
import Tag from 'primevue/tag'
import Message from 'primevue/message'

const systemStore = useSystemStore()
const wsStore = useWebSocketStore()
const confirm = useConfirm()

const cudaStatus = ref(null)
const selectedCudaVersion = ref('12.9')
const installing = ref(false)
const loading = ref(false)
const cudaInstallProgress = ref(null)
const cudaInstallLogs = ref([])

const availableVersions = computed(() => {
  return cudaStatus.value?.available_versions || ['13.0', '12.9', '12.8', '12.7', '12.6', '12.5', '12.4', '12.3', '12.2', '12.1', '12.0', '11.9', '11.8']
})

const formatDate = (dateString) => {
  if (!dateString) return 'Unknown'
  try {
    return new Date(dateString).toLocaleString()
  } catch {
    return dateString
  }
}

const refreshStatus = async () => {
  loading.value = true
  try {
    cudaStatus.value = await systemStore.getCudaStatus()
  } catch (error) {
    toast.error('Failed to fetch CUDA status')
  } finally {
    loading.value = false
  }
}

const handleInstall = async () => {
  if (!selectedCudaVersion.value) {
    toast.error('Please select a CUDA version')
    return
  }

  installing.value = true
  cudaInstallProgress.value = null
  cudaInstallLogs.value = []

  try {
    await systemStore.installCuda(selectedCudaVersion.value)
    toast.success(`CUDA ${selectedCudaVersion.value} installation started`)
  } catch (error) {
    const errorMsg = error.response?.data?.detail || error.message || 'Failed to start CUDA installation'
    toast.error(errorMsg)
    installing.value = false
  }
}

const confirmUninstall = (installed) => {
  confirm.require({
    message: `Are you sure you want to uninstall CUDA ${installed.version}? This will remove the installation directory and cannot be undone.`,
    header: 'Uninstall CUDA Toolkit',
    icon: 'pi pi-exclamation-triangle',
    rejectLabel: 'Cancel',
    acceptLabel: 'Uninstall',
    accept: async () => {
      try {
        await systemStore.uninstallCuda(installed.version)
        toast.success(`CUDA ${installed.version} uninstallation started`)
        await refreshStatus()
      } catch (error) {
        const errorMsg = error.response?.data?.detail || error.message || 'Failed to start CUDA uninstallation'
        toast.error(errorMsg)
      }
    }
  })
}

// WebSocket handlers
const handleCudaInstallStatus = (data) => {
  if (data.status === 'completed') {
    installing.value = false
    cudaInstallProgress.value = { progress: 100, message: 'Installation completed!' }
    toast.success(data.message || 'CUDA installation completed')
    refreshStatus()
  } else if (data.status === 'failed') {
    installing.value = false
    toast.error(data.message || 'CUDA installation failed')
  }
}

const handleCudaInstallProgress = (data) => {
  cudaInstallProgress.value = {
    progress: data.progress || 0,
    message: data.message || '',
    stage: data.stage || 'unknown'
  }
}

const handleCudaInstallLog = (data) => {
  cudaInstallLogs.value.push(data.line)
  if (cudaInstallLogs.value.length > 100) {
    cudaInstallLogs.value = cudaInstallLogs.value.slice(-100)
  }
}

let unsubscribeStatus = null
let unsubscribeProgress = null
let unsubscribeLog = null

onMounted(async () => {
  await refreshStatus()
  
  // Subscribe to CUDA installation events
  unsubscribeStatus = wsStore.subscribe('cuda_install_status', handleCudaInstallStatus)
  unsubscribeProgress = wsStore.subscribe('cuda_install_progress', handleCudaInstallProgress)
  unsubscribeLog = wsStore.subscribe('cuda_install_log', handleCudaInstallLog)
})

onUnmounted(() => {
  if (unsubscribeStatus) unsubscribeStatus()
  if (unsubscribeProgress) unsubscribeProgress()
  if (unsubscribeLog) unsubscribeLog()
})
</script>

<style scoped>
.cuda-installer {
  background: var(--gradient-card);
  border: 1px solid var(--border-primary);
  border-radius: var(--radius-xl);
  padding: var(--spacing-xl);
  margin-bottom: 2rem;
}

.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: var(--spacing-lg);
}

.card-header h3 {
  margin: 0;
  color: var(--text-primary);
  font-weight: 700;
  font-size: 1.3rem;
}

.status-section {
  margin-bottom: var(--spacing-xl);
}

.status-card {
  background: var(--bg-surface);
  border: 1px solid var(--border-primary);
  border-radius: var(--radius-lg);
  padding: var(--spacing-lg);
}

.status-card.installed {
  border-color: var(--green-500);
  background: var(--green-50);
}

.status-header {
  display: flex;
  align-items: center;
  gap: var(--spacing-md);
  margin-bottom: var(--spacing-md);
}

.status-header i {
  font-size: 1.5rem;
}

.status-header h4 {
  margin: 0;
  color: var(--text-primary);
  font-weight: 700;
}

.status-details {
  display: flex;
  flex-direction: column;
  gap: var(--spacing-sm);
}

.detail-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.detail-label {
  font-weight: 600;
  color: var(--text-secondary);
  font-size: 0.9rem;
}

.detail-value {
  font-weight: 600;
  color: var(--text-primary);
  font-family: 'Courier New', monospace;
  background: var(--bg-card);
  padding: var(--spacing-xs) var(--spacing-sm);
  border-radius: var(--radius-sm);
  border: 1px solid var(--border-primary);
  font-size: 0.875rem;
}

.installed-versions {
  margin-bottom: var(--spacing-xl);
}

.installed-versions h4 {
  margin-bottom: var(--spacing-md);
  color: var(--text-primary);
  font-weight: 700;
}

.version-list {
  display: flex;
  flex-direction: column;
  gap: var(--spacing-md);
}

.version-card {
  background: var(--bg-surface);
  border: 1px solid var(--border-primary);
  border-radius: var(--radius-lg);
  padding: var(--spacing-lg);
}

.version-card.current {
  border-color: var(--green-500);
  background: var(--green-50);
}

.version-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: var(--spacing-md);
}

.version-info {
  display: flex;
  align-items: center;
  gap: var(--spacing-sm);
}

.version-label {
  font-weight: 700;
  color: var(--text-primary);
  font-size: 1.1rem;
}

.version-details {
  display: flex;
  flex-direction: column;
  gap: var(--spacing-xs);
}

.install-section h4 {
  margin-bottom: var(--spacing-md);
  color: var(--text-primary);
  font-weight: 700;
}

.install-form {
  margin-bottom: var(--spacing-lg);
}

.form-group {
  margin-bottom: var(--spacing-md);
}

.form-group label {
  display: block;
  margin-bottom: var(--spacing-xs);
  font-weight: 600;
  color: var(--text-primary);
}

.version-select {
  width: 100%;
}

.form-info {
  margin-top: var(--spacing-md);
  padding: var(--spacing-md);
  background: var(--bg-surface);
  border-radius: var(--radius-md);
  border: 1px solid var(--border-primary);
}

.form-info p {
  margin: var(--spacing-xs) 0;
  color: var(--text-secondary);
  font-size: 0.9rem;
}

.info-text {
  color: var(--text-secondary);
  font-size: 0.875rem;
  font-style: italic;
}

.install-progress {
  margin-bottom: var(--spacing-lg);
}

.progress-message {
  margin-top: var(--spacing-sm);
  color: var(--text-secondary);
  font-size: 0.9rem;
}

.install-logs {
  margin-bottom: var(--spacing-lg);
  background: var(--bg-surface);
  border: 1px solid var(--border-primary);
  border-radius: var(--radius-md);
  padding: var(--spacing-md);
}

.logs-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: var(--spacing-sm);
}

.logs-header h5 {
  margin: 0;
  color: var(--text-primary);
  font-weight: 700;
}

.logs-content {
  max-height: 300px;
  overflow-y: auto;
  background: var(--bg-card);
  padding: var(--spacing-sm);
  border-radius: var(--radius-sm);
}

.logs-content pre {
  margin: 0;
  color: var(--text-primary);
  font-size: 0.8rem;
  font-family: 'Courier New', monospace;
  white-space: pre-wrap;
  word-wrap: break-word;
}

.action-buttons {
  display: flex;
  gap: var(--spacing-md);
}

.operation-status,
.error-message {
  margin-top: var(--spacing-lg);
}
</style>

