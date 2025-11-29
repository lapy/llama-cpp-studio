<template>
  <BaseDialog
    :visible="visible"
    header="Install CUDA Toolkit"
    :modal="true"
    :dialog-style="{ width: '60vw', maxWidth: '700px' }"
    :draggable="false"
    :resizable="false"
    @update:visible="$emit('update:visible', $event)"
  >
    <div class="cuda-install-dialog">
      <div v-if="cudaStatus" class="cuda-status-section">
        <div class="status-card" :class="{ 'installed': cudaStatus.installed }">
          <div class="status-header">
            <i :class="cudaStatus.installed ? 'pi pi-check-circle' : 'pi pi-times-circle'" 
               :style="{ color: cudaStatus.installed ? 'var(--status-success)' : 'var(--status-error)' }"></i>
            <h4>{{ cudaStatus.installed ? 'CUDA Installed' : 'CUDA Not Installed' }}</h4>
          </div>
          <div v-if="cudaStatus.installed" class="status-details">
            <p><strong>Version:</strong> {{ cudaStatus.version }}</p>
            <p v-if="cudaStatus.cuda_path"><strong>Path:</strong> {{ cudaStatus.cuda_path }}</p>
            <p v-if="cudaStatus.installed_at"><strong>Installed:</strong> {{ formatDate(cudaStatus.installed_at) }}</p>
          </div>
          <div v-else class="status-details">
            <p>CUDA Toolkit is required for CUDA-enabled builds.</p>
          </div>
        </div>
      </div>

      <div v-if="!cudaStatus?.installed" class="install-section">
        <div class="form-field">
          <label>CUDA Version</label>
          <Dropdown 
            v-model="selectedCudaVersion"
            :options="cudaStatus?.available_versions || ['12.6', '12.5', '12.4', '12.3', '12.2', '12.1', '12.0', '11.9', '11.8']"
            placeholder="Select CUDA version"
          />
          <small>Recommended: 12.6 (latest stable)</small>
        </div>

        <div class="platform-info">
          <p><strong>Platform:</strong> {{ cudaStatus?.platform?.[0] || 'Unknown' }} / {{ cudaStatus?.platform?.[1] || 'Unknown' }}</p>
          <p class="warning-text">
            <i class="pi pi-info-circle"></i>
            <span v-if="cudaStatus?.platform?.[0] === 'linux'">
              Linux installation may require sudo privileges. If installation fails, you may need to install CUDA manually.
            </span>
            <span v-else>
              The installer will run in silent mode. Please ensure no other CUDA installations are in progress.
            </span>
          </p>
        </div>

        <div v-if="cudaInstallProgress" class="install-progress">
          <ProgressBar :value="cudaInstallProgress.progress || 0" />
          <p class="progress-message">{{ cudaInstallProgress.message || 'Preparing installation...' }}</p>
        </div>

        <div v-if="cudaInstallLogs.length > 0" class="install-logs">
          <h5>Installation Logs</h5>
          <div class="log-container">
            <pre>{{ cudaInstallLogs.join('\n') }}</pre>
          </div>
        </div>
      </div>
    </div>

    <template #footer>
      <Button 
        label="Close" 
        icon="pi pi-times" 
        @click="$emit('update:visible', false)"
        severity="secondary"
        text
      />
      <Button 
        v-if="!cudaStatus?.installed && !installing"
        label="Install CUDA" 
        icon="pi pi-download"
        @click="handleInstall"
        :disabled="!selectedCudaVersion || cudaStatus?.operation"
        :loading="installing"
      />
    </template>
  </BaseDialog>
</template>

<script setup>
import { ref, watch } from 'vue'
import { useSystemStore } from '@/stores/system'
import { toast } from 'vue3-toastify'
import { formatDate } from '@/utils/formatting'
import Button from 'primevue/button'
import Dropdown from 'primevue/dropdown'
import ProgressBar from 'primevue/progressbar'
import BaseDialog from '@/components/common/BaseDialog.vue'

const props = defineProps({
  visible: {
    type: Boolean,
    default: false
  },
  cudaStatus: {
    type: Object,
    default: null
  },
  cudaInstallProgress: {
    type: Object,
    default: null
  },
  cudaInstallLogs: {
    type: Array,
    default: () => []
  }
})

const emit = defineEmits(['update:visible', 'install', 'status-update', 'progress', 'log'])

const systemStore = useSystemStore()

const selectedCudaVersion = ref('12.6')
const installing = ref(false)

const handleInstall = async () => {
  if (!selectedCudaVersion.value) {
    toast.error('Please select a CUDA version')
    return
  }

  installing.value = true
  
  try {
    await systemStore.installCuda(selectedCudaVersion.value)
    toast.success(`CUDA ${selectedCudaVersion.value} installation started`)
    emit('install', selectedCudaVersion.value)
  } catch (error) {
    const errorMsg = error.response?.data?.detail || error.message || 'Failed to start CUDA installation'
    toast.error(errorMsg)
  } finally {
    installing.value = false
  }
}

watch(() => props.cudaStatus, (status) => {
  if (status?.installed) {
    installing.value = false
  }
}, { deep: true })
</script>

<style scoped>
.cuda-install-dialog {
  display: flex;
  flex-direction: column;
  gap: var(--spacing-lg);
}

.cuda-status-section {
  margin-bottom: var(--spacing-md);
}

.status-card {
  border: 1px solid var(--border-primary);
  border-radius: var(--radius-lg);
  padding: var(--spacing-lg);
  background: var(--bg-card);
}

.status-card.installed {
  border-color: var(--status-success);
  background: var(--status-success-soft);
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
  font-size: 1.125rem;
}

.status-details {
  display: flex;
  flex-direction: column;
  gap: var(--spacing-sm);
}

.status-details p {
  margin: 0;
  font-size: 0.875rem;
}

.install-section {
  display: flex;
  flex-direction: column;
  gap: var(--spacing-md);
}

.form-field {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}

.form-field label {
  font-weight: 600;
  color: var(--text-primary);
  font-size: 0.9rem;
}

.form-field small {
  color: var(--text-secondary);
  font-size: 0.8rem;
}

.platform-info {
  padding: var(--spacing-md);
  background: var(--status-info-soft);
  border-radius: var(--radius-md);
  border-left: 3px solid var(--accent-blue);
}

.platform-info p {
  margin: 0.25rem 0;
  font-size: 0.875rem;
}

.warning-text {
  display: flex;
  align-items: flex-start;
  gap: var(--spacing-sm);
  color: var(--text-secondary);
}

.warning-text i {
  color: var(--accent-blue);
  margin-top: 0.125rem;
}

.install-progress {
  margin: var(--spacing-md) 0;
}

.progress-message {
  margin-top: var(--spacing-sm);
  font-size: 0.875rem;
  color: var(--text-secondary);
}

.install-logs {
  margin-top: var(--spacing-md);
}

.install-logs h5 {
  margin: 0 0 var(--spacing-sm) 0;
  font-size: 0.875rem;
  font-weight: 600;
}

.log-container {
  max-height: 300px;
  overflow-y: auto;
  background: var(--bg-secondary);
  border: 1px solid var(--border-primary);
  border-radius: var(--radius-md);
  padding: var(--spacing-md);
}

.log-container pre {
  margin: 0;
  font-family: 'Courier New', monospace;
  font-size: 0.75rem;
  line-height: 1.5;
  white-space: pre-wrap;
  word-wrap: break-word;
}
</style>

