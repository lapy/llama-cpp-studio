<template>
  <div class="system-status">
    <div class="card">
      <div class="card-header">
        <h2 class="card-title">System Status</h2>
        <div class="header-actions">
          <div class="connection-info">
            <div class="live-indicator" v-if="wsStore.isConnected">
              <i class="pi pi-circle-fill" style="color: #22d3ee; font-size: 0.5rem;"></i>
              <span>Live</span>
            </div>
            <div class="connection-status" v-else>
              <i class="pi pi-circle" style="color: #ef4444; font-size: 0.5rem;"></i>
              <span>{{ wsStore.connectionStatus }}</span>
            </div>
          </div>
          <Button 
            icon="pi pi-refresh" 
            @click="refreshStatus"
            :loading="systemStore.loading"
            severity="secondary"
            text
          />
        </div>
      </div>

      <!-- System Overview -->
      <div class="system-overview">
        <div class="overview-grid">
          <div class="overview-card">
            <div class="overview-header">
              <span style="color: #22d3ee; font-size: 1.75rem; font-weight: bold;">🖥️</span>
              <h3>CPU</h3>
            </div>
            <div class="overview-content">
              <div class="metric">
                <span class="metric-label">Usage</span>
                <span class="metric-value">{{ (systemStore.systemStatus.system?.cpu_percent || 0).toFixed(1) }}%</span>
              </div>
              <ProgressBar :value="systemStore.systemStatus.system?.cpu_percent || 0" />
            </div>
          </div>

          <div class="overview-card">
            <div class="overview-header">
              <span style="color: #22d3ee; font-size: 1.75rem; font-weight: bold;">💾</span>
              <h3>Memory</h3>
            </div>
            <div class="overview-content">
              <div class="metric">
                <span class="metric-label">Usage</span>
                <span class="metric-value">{{ (systemStore.systemStatus.system?.memory?.percent || 0).toFixed(1) }}%</span>
              </div>
              <div class="metric">
                <span class="metric-label">Available</span>
                <span class="metric-value">{{ formatFileSize(systemStore.systemStatus.system?.memory?.available || 0) }}</span>
              </div>
              <ProgressBar :value="systemStore.systemStatus.system?.memory?.percent || 0" />
            </div>
          </div>

          <div class="overview-card">
            <div class="overview-header">
              <span style="color: #22d3ee; font-size: 1.75rem; font-weight: bold;">💿</span>
              <h3>Storage</h3>
            </div>
            <div class="overview-content">
              <div class="metric">
                <span class="metric-label">Usage</span>
                <span class="metric-value">{{ (systemStore.systemStatus.system?.disk?.percent || 0).toFixed(1) }}%</span>
              </div>
              <div class="metric">
                <span class="metric-label">Free</span>
                <span class="metric-value">{{ formatFileSize(systemStore.systemStatus.system?.disk?.free || 0) }}</span>
              </div>
              <ProgressBar :value="systemStore.systemStatus.system?.disk?.percent || 0" />
            </div>
          </div>

          <div class="overview-card">
            <div class="overview-header">
              <span style="color: #22d3ee; font-size: 1.75rem; font-weight: bold;">🎮</span>
              <h3>GPU</h3>
            </div>
            <div class="overview-content">
              <div class="metric">
                <span class="metric-label">Count</span>
                <span class="metric-value">{{ systemStore.gpuInfo.device_count || 0 }}</span>
              </div>
              <div class="metric">
                <span class="metric-label">Total VRAM</span>
                <span class="metric-value">{{ formatFileSize(systemStore.gpuInfo.total_vram || 0) }}</span>
              </div>
              <div class="metric">
                <span class="metric-label">Available</span>
                <span class="metric-value">{{ formatFileSize(systemStore.gpuInfo.available_vram || 0) }}</span>
              </div>
              <div v-if="systemStore.gpuInfo.nvlink_topology?.has_nvlink" class="metric">
                <span class="metric-label">NVLink</span>
                <span class="metric-value">{{ systemStore.gpuInfo.nvlink_topology.recommended_strategy }}</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- GPU Details -->
      <div v-if="systemStore.gpuInfo.gpus && systemStore.gpuInfo.gpus.length > 0" class="gpu-details">
        <h3>GPU Details</h3>
        <div class="gpu-list">
          <div 
            v-for="gpu in systemStore.gpuInfo.gpus" 
            :key="gpu.index"
            class="gpu-card"
          >
            <div class="gpu-header">
              <h4>GPU {{ gpu.index }}: {{ gpu.name }}</h4>
              <div class="gpu-status">
                <span 
                  :class="['status-indicator', gpu.utilization?.gpu ? 'status-running' : 'status-stopped']"
                >
                  <i :class="gpu.utilization?.gpu ? 'pi pi-play' : 'pi pi-pause'"></i>
                  {{ gpu.utilization?.gpu ? `${gpu.utilization.gpu}%` : 'Idle' }}
                </span>
              </div>
            </div>
            
            <div class="gpu-metrics">
              <div class="metric-row">
                <span class="metric-label">Memory Usage</span>
                <div class="metric-bar">
                  <ProgressBar 
                    :value="(gpu.memory.used / gpu.memory.total) * 100"
                    :showValue="false"
                  />
                  <span class="metric-text">
                    {{ formatFileSize(gpu.memory.used) }} / {{ formatFileSize(gpu.memory.total) }}
                  </span>
                </div>
              </div>
              
              <div class="metric-row">
                <span class="metric-label">Compute Capability</span>
                <span class="metric-value">{{ gpu.compute_capability }}</span>
              </div>
              
              <div v-if="gpu.nvlink && gpu.nvlink.connections.length > 0" class="metric-row">
                <span class="metric-label">NVLink</span>
                <div class="nvlink-info">
                  <span class="nvlink-version">v{{ gpu.nvlink.nvlink_version }}</span>
                  <span class="nvlink-bandwidth">{{ gpu.nvlink.total_bandwidth }} GB/s</span>
                  <span class="nvlink-connections">{{ gpu.nvlink.connections.length }} links</span>
                </div>
              </div>
              
              <div v-if="gpu.temperature" class="metric-row">
                <span class="metric-label">Temperature</span>
                <span class="metric-value">{{ gpu.temperature }}°C</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- Proxy Status -->
      <div v-if="systemStore.systemStatus.proxy_status" class="proxy-status">
        <h3>Multi-Model Proxy</h3>
        <div class="proxy-card">
          <div class="proxy-header">
            <i class="pi pi-share-alt"></i>
            <h4>llama-swap Proxy</h4>
            <span class="status-indicator status-running">
              <i class="pi pi-check"></i>
              Active
            </span>
          </div>
          <div class="proxy-details">
            <div class="detail-row">
              <span class="detail-label">Port:</span>
              <span class="detail-value">{{ systemStore.systemStatus.proxy_status.port }}</span>
            </div>
            <div class="detail-row">
              <span class="detail-label">API Endpoint:</span>
              <span class="detail-value">{{ systemStore.systemStatus.proxy_status.endpoint }}</span>
            </div>
            <div class="detail-row">
              <span class="detail-label">Models Available:</span>
              <span class="detail-value">{{ sortedRunningInstances.length }}</span>
            </div>
          </div>
        </div>
      </div>

      <!-- Running Instances -->
      <div v-if="sortedRunningInstances && sortedRunningInstances.length > 0" class="running-instances">
        <h3>Running Models</h3>
        <div class="instance-list">
          <div 
            v-for="instance in sortedRunningInstances" 
            :key="instance.id"
            class="instance-card"
          >
            <div class="instance-header">
              <h4>{{ instance.proxy_model_name || `Model #${instance.model_id}` }}</h4>
              <div class="instance-status">
                <span class="status-indicator status-running">
                  <i class="pi pi-play"></i>
                  Running
                </span>
              </div>
            </div>
            
            <div class="instance-metrics">
              <div class="metric-row">
                <span class="metric-label">Model ID</span>
                <span class="metric-value">{{ instance.model_id }}</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- Empty State -->
      <div v-if="!sortedRunningInstances || sortedRunningInstances.length === 0" class="empty-instances">
        <i class="pi pi-play-circle" style="font-size: 3rem; color: var(--text-secondary);"></i>
        <h4>No Running Instances</h4>
        <p>Start a model from the Model Library to see running instances here.</p>
      </div>
    </div>
  </div>
</template>

<script setup>
import { onMounted, onUnmounted, computed } from 'vue'
import { useSystemStore } from '@/stores/system'
import { useWebSocketStore } from '@/stores/websocket'
import { toast } from 'vue3-toastify'
import { formatFileSize, formatDate } from '@/utils/formatting'

const systemStore = useSystemStore()
const wsStore = useWebSocketStore()
let unsubscribeUnifiedMonitoring = null

// Computed property to sort running instances by model name (alphabetical)
const sortedRunningInstances = computed(() => {
  if (!systemStore.systemStatus.running_instances) return []
  
  return [...systemStore.systemStatus.running_instances].sort((a, b) => {
    const nameA = (a.proxy_model_name || a.model_id || '').toLowerCase()
    const nameB = (b.proxy_model_name || b.model_id || '').toLowerCase()
    return nameA.localeCompare(nameB)
  })
})

onMounted(() => {
  refreshStatus()
  
  // Subscribe to unified monitoring updates for real-time status
  unsubscribeUnifiedMonitoring = wsStore.subscribeToUnifiedMonitoring((data) => {
    
    // Update system status with real-time data
    if (data.system) {
      systemStore.updateSystemStatus({
        system: data.system,
        proxy_status: data.proxy_status
      })
    }
    
    // Update running instances from both database and llama-swap
    if (data.models) {
      const runningInstances = data.models.running_instances || []
      const llamaSwapModels = data.models.llama_swap_models || []
      
      
      // Combine both sources of running models
      const allRunningInstances = [...runningInstances]
      
      // Add llama-swap models as additional instances
      if (Array.isArray(llamaSwapModels)) {
        llamaSwapModels.forEach(model => {
          // Check if this model is already in running_instances
          const exists = allRunningInstances.some(instance => 
            instance.proxy_model_name === model.model || 
            instance.model_id === model.model ||
            instance.proxy_model_name === model.name
          )
          
          if (!exists) {
            allRunningInstances.push({
              id: `llama-swap-${model.model}`,
              model_id: model.model,
              proxy_model_name: model.model,
              port: 'N/A', // llama-swap doesn't provide port info in /running
              started_at: new Date().toISOString(),
              source: 'llama-swap',
              state: model.state
            })
          }
        })
      }
      
      
      systemStore.updateSystemStatus({
        running_instances: allRunningInstances
      })
    }
    
    // Update GPU info if available
    if (data.gpu) {
      systemStore.updateGpuInfo(data.gpu)
    }
  })
  
})

onUnmounted(() => {
  if (unsubscribeUnifiedMonitoring) {
    unsubscribeUnifiedMonitoring()
  }
})

const refreshStatus = async () => {
  try {
    await systemStore.fetchSystemStatus()
  } catch (error) {
    toast.error('Failed to refresh system status')
  }
}

// formatFileSize and formatDate are now imported from @/utils/formatting
</script>

<style scoped>
.system-status {
  max-width: 1400px;
  margin: 0 auto;
}

.system-overview {
  margin-bottom: 2rem;
}

.overview-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: 1rem;
}

.overview-card {
  background: var(--gradient-card);
  border: 1px solid var(--border-primary);
  border-radius: var(--radius-xl);
  padding: var(--spacing-xl);
  box-shadow: var(--shadow-md);
  transition: all var(--transition-normal);
  position: relative;
  overflow: hidden;
  backdrop-filter: blur(10px);
  animation: fadeIn 0.6s ease-out;
}

.overview-card::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  height: 3px;
  background: var(--gradient-primary);
  opacity: 0;
  transition: opacity var(--transition-normal);
}

.overview-card:hover {
  transform: translateY(-5px) scale(1.02);
  box-shadow: var(--shadow-lg), var(--glow-primary);
}

.overview-card:hover::before {
  opacity: 1;
}

.overview-header {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  margin-bottom: 1rem;
}

.overview-header i {
  font-size: 1.75rem !important;
  color: var(--accent-cyan) !important;
  display: inline-block !important;
  visibility: visible !important;
  opacity: 1 !important;
}

.overview-header h3 {
  margin: 0;
  color: var(--text-primary);
  font-weight: 700;
  font-size: 1.2rem;
}

.overview-content {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}

.metric {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.metric-label {
  font-size: 0.9rem;
  color: var(--text-secondary);
  font-weight: 500;
}

.metric-value {
  font-weight: 700;
  color: var(--text-primary);
  font-size: 1rem;
}

.gpu-details,
.proxy-status {
  margin-bottom: 2rem;
}

.proxy-card {
  background: var(--gradient-card);
  border: 1px solid var(--border-primary);
  border-radius: var(--radius-xl);
  padding: var(--spacing-xl);
  box-shadow: var(--shadow-md);
  transition: all var(--transition-normal);
  position: relative;
  overflow: hidden;
}

.proxy-card::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  height: 3px;
  background: var(--gradient-success);
  opacity: 0;
  transition: opacity var(--transition-normal);
}

.proxy-card:hover {
  transform: translateY(-3px);
  box-shadow: var(--shadow-lg);
}

.proxy-card:hover::before {
  opacity: 1;
}

.proxy-header {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  margin-bottom: 1rem;
}

.proxy-header i {
  font-size: 1.75rem;
  background: var(--gradient-success);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}

.proxy-header h4 {
  margin: 0;
  color: var(--text-primary);
  font-weight: 700;
  font-size: 1.2rem;
}

.proxy-details {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
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
  background: var(--bg-surface);
  padding: var(--spacing-xs) var(--spacing-sm);
  border-radius: var(--radius-sm);
  border: 1px solid var(--border-primary);
}

.running-instances {
  margin-top: 2rem;
}

.gpu-list,
.instance-list {
  display: flex;
  flex-direction: column;
  gap: 1rem;
  margin-top: 1rem;
}

.gpu-card,
.instance-card {
  background: var(--gradient-card);
  border: 1px solid var(--border-primary);
  border-radius: var(--radius-xl);
  padding: var(--spacing-xl);
  box-shadow: var(--shadow-md);
  transition: all var(--transition-normal);
  position: relative;
  overflow: hidden;
}

.gpu-card::before,
.instance-card::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  height: 3px;
  background: var(--gradient-primary);
  opacity: 0;
  transition: opacity var(--transition-normal);
}

.gpu-card:hover,
.instance-card:hover {
  transform: translateY(-3px);
  box-shadow: var(--shadow-lg);
}

.gpu-card:hover::before,
.instance-card:hover::before {
  opacity: 1;
}

.gpu-header,
.instance-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 1rem;
}

.gpu-header h4,
.instance-header h4 {
  margin: 0;
  color: var(--text-primary);
  font-weight: 700;
  font-size: 1.1rem;
}

.gpu-metrics,
.instance-metrics {
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
}

.metric-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.metric-bar {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  flex: 1;
  margin-left: 1rem;
}

.metric-text {
  font-size: 0.875rem;
  color: var(--text-secondary);
  min-width: 120px;
  text-align: right;
}

.empty-instances {
  text-align: center;
  padding: var(--spacing-3xl) var(--spacing-xl);
  color: var(--text-secondary);
  background: var(--gradient-surface);
  border-radius: var(--radius-xl);
  border: 2px dashed var(--border-secondary);
  margin: var(--spacing-xl) 0;
  position: relative;
  overflow: hidden;
}

.empty-instances::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  height: 2px;
  background: var(--gradient-primary);
  opacity: 0.3;
}

.empty-instances i {
  font-size: 3rem !important;
  background: var(--gradient-primary);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  margin-bottom: var(--spacing-lg);
}

.empty-instances h4 {
  margin: var(--spacing-lg) 0 var(--spacing-md);
  color: var(--text-primary);
  font-weight: 700;
  font-size: 1.3rem;
}

@media (max-width: 768px) {
  .overview-grid {
    grid-template-columns: 1fr;
  }
  
  .gpu-header,
  .instance-header {
    flex-direction: column;
    align-items: flex-start;
    gap: 0.5rem;
  }
  
  .metric-bar {
    flex-direction: column;
    align-items: flex-start;
    margin-left: 0;
    margin-top: 0.5rem;
  }
  
  .metric-text {
    text-align: left;
    min-width: auto;
  }
}

.nvlink-info {
  display: flex;
  flex-direction: column;
  gap: 0.25rem;
  font-size: 0.875rem;
}

.nvlink-version {
  font-weight: 700;
  color: var(--accent-cyan);
}

.nvlink-bandwidth {
  color: var(--text-secondary);
  font-weight: 500;
}

.nvlink-connections {
  color: var(--text-secondary);
  font-size: 0.8rem;
  font-weight: 500;
}

.connection-info {
  display: flex;
  align-items: center;
}

.live-indicator,
.connection-status {
  display: flex;
  align-items: center;
  gap: var(--spacing-xs);
  font-size: 0.875rem;
  color: var(--text-secondary);
  font-weight: 500;
}

.live-indicator i {
  animation: pulse 2s infinite;
}

.connection-status {
  color: #ef4444;
}

@keyframes pulse {
  0% { opacity: 1; }
  50% { opacity: 0.5; }
  100% { opacity: 1; }
}
</style>
