<template>
  <div class="lmdeploy-page">
    <section class="card">
      <header class="card-header">
        <div>
          <h2>LMDeploy Installer</h2>
          <p class="card-subtitle">Install or remove LMDeploy inside the running container without rebuilding the image.</p>
        </div>
        <Tag
          :value="installed ? 'Installed' : 'Not Installed'"
          :severity="installed ? 'success' : 'warning'"
        />
      </header>

      <div class="status-grid">
        <div>
          <label>Status</label>
          <div class="status-value">
            <i
              :class="[
                'pi',
                operationInProgress ? 'pi-spin pi-spinner text-warning' : installed ? 'pi-check-circle text-success' : 'pi-info-circle text-muted'
              ]"
            ></i>
            <span>
              {{ operationInProgress ? `Running ${status?.operation}…` : installed ? 'Ready' : 'Install required' }}
            </span>
          </div>
          <small v-if="status?.operation_started_at">
            Started at {{ formatDate(status.operation_started_at) }}
          </small>
        </div>
        <div>
          <label>Version</label>
          <div class="status-value monospace">
            {{ status?.version || 'Unknown' }}
          </div>
        </div>
        <div>
          <label>Binary Path</label>
          <div class="status-value monospace truncate">
            {{ status?.binary_path || 'Not found' }}
          </div>
        </div>
        <div>
          <label>Virtual Environment</label>
          <div class="status-value monospace truncate">
            {{ status?.venv_path || 'Not created' }}
          </div>
        </div>
        <div>
          <label>Last Error</label>
          <div class="status-value error-text">
            {{ status?.last_error || 'None' }}
          </div>
        </div>
      </div>

      <div class="card-actions">
        <Button
          label="Install LMDeploy"
          icon="pi pi-download"
          severity="success"
          :loading="installing"
          :disabled="operationInProgress || installed"
          @click="startInstall"
        />
        <Button
          label="Remove LMDeploy"
          icon="pi pi-trash"
          severity="danger"
          outlined
          :loading="removing"
          :disabled="operationInProgress || !installed"
          @click="startRemoval"
        />
        <Button
          label="Refresh"
          icon="pi pi-refresh"
          severity="secondary"
          text
          :loading="statusLoading"
          @click="refresh"
        />
      </div>

      <p class="helper-text">
        Need to run safetensors models? Install LMDeploy here, then start runtimes from the Safetensors panel.
      </p>
    </section>

    <section class="card">
      <header class="card-header">
        <div>
          <h3>Installer Logs</h3>
          <p class="card-subtitle">Newest lines first. Use this to monitor pip progress if an install is running.</p>
        </div>
        <Button
          icon="pi pi-refresh"
          severity="secondary"
          text
          :loading="logLoading"
          @click="refreshLogs"
        />
      </header>
      <LogsFeed v-if="logContent" :logs="parsedLogLines" />
      <div v-else class="empty-log">
        <i class="pi pi-info-circle"></i>
        <p>No LMDeploy installer logs yet.</p>
      </div>
    </section>
  </div>
</template>

<script setup>
import { computed, onMounted, onUnmounted } from 'vue'
import { toast } from 'vue3-toastify'
import Button from 'primevue/button'
import Tag from 'primevue/tag'

import LogsFeed from '@/components/LogsFeed.vue'
import { useLmdeployStore } from '@/stores/lmdeploy'
import { formatDate } from '@/utils/formatting'

const store = useLmdeployStore()

const status = computed(() => store.status)
const installed = computed(() => !!status.value?.installed)
const operationInProgress = computed(() => !!status.value?.operation)
const installing = computed(() => store.installing || status.value?.operation === 'install')
const removing = computed(() => store.removing || status.value?.operation === 'remove')
const statusLoading = computed(() => store.loading)
const logLoading = computed(() => store.logLoading)
const logContent = computed(() => store.logs || '')
const parsedLogLines = computed(() =>
  (store.logs || '')
    .split('\n')
    .filter(Boolean)
    .map((line, index) => {
      const match = line.match(/^\[(.*?)\]\s*(.*)$/)
      const timestamp = match ? match[1] : ''
      const data = match ? match[2] : line
      return {
        timestamp,
        log_type: 'install',
        data,
        id: `${timestamp || 'log'}-${index}`
      }
    })
)

const refresh = async () => {
  try {
    await Promise.all([store.fetchStatus(), store.fetchLogs()])
  } catch (error) {
    toast.error('Failed to refresh LMDeploy status')
  }
}

const refreshLogs = async () => {
  try {
    await store.fetchLogs(16384)
  } catch (error) {
    toast.error('Failed to refresh LMDeploy logs')
  }
}

const startInstall = async () => {
  try {
    await store.install()
    toast.success('LMDeploy installation started')
  } catch (error) {
    toast.error(error?.response?.data?.detail || 'Failed to start installation')
  }
}

const startRemoval = async () => {
  try {
    await store.remove()
    toast.success('LMDeploy removal started')
  } catch (error) {
    toast.error(error?.response?.data?.detail || 'Failed to start removal')
  }
}

// formatDate is now imported from @/utils/formatting

onMounted(() => {
  refresh()
  store.startPolling()
})

onUnmounted(() => {
  store.stopPolling()
})
</script>

<style scoped>
.lmdeploy-page {
  display: flex;
  flex-direction: column;
  gap: var(--spacing-xl);
}

.card {
  background: var(--bg-card);
  border-radius: var(--radius-xl);
  padding: var(--spacing-xl);
  box-shadow: var(--shadow-lg);
  border: 1px solid var(--border-primary);
  display: flex;
  flex-direction: column;
  gap: var(--spacing-lg);
}

.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: var(--spacing-lg);
}

.card-subtitle {
  margin: 0;
  color: var(--text-secondary);
  font-size: 0.95rem;
}

.status-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  gap: var(--spacing-lg);
}

.status-grid label {
  display: block;
  font-size: 0.75rem;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--text-secondary);
  margin-bottom: var(--spacing-xs);
}

.status-value {
  display: flex;
  align-items: center;
  gap: var(--spacing-sm);
}

.monospace {
  font-family: 'JetBrains Mono', 'Fira Code', monospace;
}

.truncate {
  max-width: 320px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.error-text {
  color: var(--status-error);
}

.card-actions {
  display: flex;
  flex-wrap: wrap;
  gap: var(--spacing-md);
}

.helper-text {
  margin: 0;
  font-size: 0.9rem;
  color: var(--text-secondary);
}

.empty-log {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: var(--spacing-sm);
  color: var(--text-secondary);
  padding: var(--spacing-xl);
}

.text-success {
  color: var(--status-success);
}

.text-warning {
  color: var(--status-warning);
}

.text-muted {
  color: var(--text-secondary);
}
</style>

