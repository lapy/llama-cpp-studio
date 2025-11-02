<template>
  <div class="model-library">
    <div class="card">
      <div class="card-header">
        <h2 class="card-title">Downloaded Models</h2>
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
            @click="refreshModels"
            :loading="modelStore.loading"
            severity="secondary"
            text
          />
        </div>
      </div>

      <!-- Download Progress -->
      <DownloadProgress />

      <!-- Downloaded Models -->
      <div 
        v-if="modelStore.modelGroups.length > 0" 
        class="downloaded-models"
        @touchstart="handlePullToRefreshStart"
        @touchmove="handlePullToRefreshMove"
        @touchend="handlePullToRefreshEnd"
      >
        <div v-if="pullToRefreshDistance > 0" class="pull-to-refresh-indicator" :style="{ transform: `translateY(${Math.min(pullToRefreshDistance, 60)}px)` }">
          <i v-if="!modelStore.loading" class="pi pi-arrow-down" :class="{ 'rotated': pullToRefreshDistance >= 60 }"></i>
          <i v-else class="pi pi-spin pi-spinner"></i>
          <span>{{ pullToRefreshDistance >= 60 ? 'Release to refresh' : 'Pull to refresh' }}</span>
        </div>
        <div class="model-grid">
          <div 
            v-for="modelGroup in modelStore.modelGroups" 
            :key="modelGroup.huggingface_id"
            class="model-card"
          >
            <div class="model-card-header">
              <div>
                <div class="model-name">{{ modelGroup.huggingface_id }}</div>
                <div class="model-tags">
                  <span class="model-tag tag-type">{{ modelGroup.model_type }}</span>
                  <span class="model-tag tag-count">{{ modelGroup.quantizations.length }} quantizations</span>
                </div>
              </div>
              <div class="model-status">
                <span 
                  :class="[
                    'status-indicator', 
                    hasRunningQuantization(modelGroup) ? 'status-running' : 'status-stopped',
                    { 'llama-swap-running': hasLlamaSwapQuantization(modelGroup) }
                  ]"
                >
                  <i :class="hasLlamaSwapQuantization(modelGroup) ? 'pi pi-share-alt' : (hasRunningQuantization(modelGroup) ? 'pi pi-play' : 'pi pi-pause')"></i>
                  {{ getModelStatusText(modelGroup) }}
                </span>
              </div>
            </div>
            
            <!-- Quantization List -->
            <div class="quantization-list">
              <div 
                v-for="quantization in modelGroup.quantizations" 
                :key="quantization.id"
                class="quantization-item"
                :class="{ 'selected': selectedQuantization[modelGroup.huggingface_id] === quantization.id }"
              >
                <div class="quantization-info">
                  <div class="quantization-name">
                    {{ quantization.quantization }}
                    <Button 
                      v-if="quantization.is_active && quantization.proxy_name" 
                      icon="pi pi-external-link"
                      @click="openUpstreamUrl(quantization.proxy_name)"
                      severity="info"
                      size="small"
                      text
                      class="upstream-link"
                      v-tooltip.top="getUpstreamUrl(quantization.proxy_name)"
                    />
                  </div>
                  <div class="quantization-details">
                    <span class="quantization-size">{{ formatFileSize(quantization.file_size) }}</span>
                    <span 
                      v-if="quantization.is_active" 
                      class="quantization-status running"
                      :class="{ 'llama-swap-running': quantization.llama_swap_status === 'running' }"
                    >
                      <i :class="quantization.llama_swap_status === 'running' ? 'pi pi-share-alt' : 'pi pi-play'"></i>
                      Running
                    </span>
                  </div>
                </div>
                <div class="quantization-actions">
                  <Button 
                    icon="pi pi-check"
                    @click="selectQuantization(modelGroup.huggingface_id, quantization.id)"
                    :class="{ 'p-button-outlined': selectedQuantization[modelGroup.huggingface_id] !== quantization.id }"
                    size="small"
                    severity="success"
                    text
                  />
                  <Button 
                    icon="pi pi-trash"
                    @click="confirmDeleteQuantization(quantization)"
                    severity="danger"
                    size="small"
                    text
                  />
                </div>
              </div>
            </div>
            
            <!-- Model Actions -->
            <div class="model-actions">
              <div class="action-group">
                <Button 
                  v-if="!hasRunningQuantization(modelGroup)"
                  label="Start" 
                  icon="pi pi-play"
                  @click="startSelectedQuantization(modelGroup)"
                  :loading="startingModels[selectedQuantization[modelGroup.huggingface_id]]"
                  :disabled="!selectedQuantization[modelGroup.huggingface_id]"
                  severity="success"
                  size="small"
                />
                <Button 
                  v-else
                  label="Stop" 
                  icon="pi pi-stop"
                  @click="stopRunningQuantization(modelGroup)"
                  :loading="stoppingModels[getRunningQuantizationId(modelGroup)]"
                  severity="danger"
                  size="small"
                />
                <Button 
                  label="Configure" 
                  icon="pi pi-cog"
                  @click="configureSelectedQuantization(modelGroup)"
                  :disabled="!selectedQuantization[modelGroup.huggingface_id]"
                  severity="secondary"
                  size="small"
                  outlined
                />
              </div>
              <div class="action-group">
                <Button 
                  label="Delete All" 
                  icon="pi pi-trash"
                  @click="confirmDeleteGroup(modelGroup)"
                  severity="danger"
                  size="small"
                  outlined
                />
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- Empty State -->
      <div v-else class="empty-state">
        <i class="pi pi-download"></i>
        <h3>No Models Downloaded</h3>
        <p>Download models from HuggingFace to get started.</p>
        <Button 
          label="Search Models" 
          icon="pi pi-search"
          @click="goToSearch"
          severity="info"
        />
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, onUnmounted } from 'vue'
import { useRouter } from 'vue-router'
import { useModelStore } from '@/stores/models'
import { useWebSocketStore } from '@/stores/websocket'
import { toast } from 'vue3-toastify'
import { useConfirm } from 'primevue/useconfirm'
import Button from 'primevue/button'
import DownloadProgress from '@/components/DownloadProgress.vue'

const router = useRouter()
const modelStore = useModelStore()
const wsStore = useWebSocketStore()
const confirm = useConfirm()

// Reactive state
const startingModels = ref({})
const stoppingModels = ref({})
const selectedQuantization = ref({}) // Track selected quantization per model group

// Pull-to-refresh state
const pullToRefreshStartY = ref(0)
const pullToRefreshDistance = ref(0)
const pullToRefreshThreshold = 60
const isPullToRefreshActive = ref(false)

onMounted(async () => {
  await modelStore.fetchModels()
  
  // Subscribe to model status updates
  wsStore.subscribeToModelStatus((data) => {
    if (data.model_id) {
      // Find the quantization in the grouped structure
      modelStore.modelGroups.forEach(group => {
        const quantization = group.quantizations.find(q => q.id === data.model_id)
        if (quantization) {
          quantization.is_active = data.is_active
          quantization.loading = false
        }
      })
    }
  })
  
  // Subscribe to unified monitoring for real-time model status updates
  wsStore.subscribeToUnifiedMonitoring((data) => {
    if (data.models) {
      const runningInstances = data.models.running_instances || []
      
      // Create a set of all running model proxy names
      const runningProxyNames = new Set()
      
      // Add proxy names from running instances
      runningInstances.forEach(instance => {
        if (instance.proxy_model_name) {
          runningProxyNames.add(instance.proxy_model_name)
        }
      })
      
      // Update all model quantizations based on running status
      modelStore.modelGroups.forEach(group => {
        group.quantizations.forEach(quantization => {
          const proxyName = quantization.proxy_name || ''
          const isRunning = runningProxyNames.has(proxyName)
          
          // Update model status based on whether it's running or not
          modelStore.updateModelStatus(quantization.id, {
            is_active: isRunning,
            llama_swap_status: isRunning ? 'running' : 'stopped',
            llama_swap_model_name: isRunning ? proxyName : null,
            llama_swap_state: isRunning ? 'ready' : null
          })
        })
      })
    }
  })
  
  // Auto-select first quantization for each group
  modelStore.modelGroups.forEach(group => {
    if (!selectedQuantization.value[group.huggingface_id] && group.quantizations.length > 0) {
      selectedQuantization.value[group.huggingface_id] = group.quantizations[0].id
    }
  })
})

onUnmounted(() => {
  // Cleanup subscriptions
})

// Helper functions for grouped models
const hasRunningQuantization = (modelGroup) => {
  return modelGroup.quantizations.some(q => q.is_active)
}

const hasLlamaSwapQuantization = (modelGroup) => {
  return modelGroup.quantizations.some(q => q.llama_swap_status === 'running')
}

const getModelStatusText = (modelGroup) => {
  if (hasLlamaSwapQuantization(modelGroup)) {
    return 'Running'
  } else if (hasRunningQuantization(modelGroup)) {
    return 'Running'
  } else {
    return 'Stopped'
  }
}

const getRunningQuantizationId = (modelGroup) => {
  const running = modelGroup.quantizations.find(q => q.is_active)
  return running ? running.id : null
}

const selectQuantization = (huggingfaceId, quantizationId) => {
  selectedQuantization.value[huggingfaceId] = quantizationId
}

const startSelectedQuantization = async (modelGroup) => {
  const quantizationId = selectedQuantization.value[modelGroup.huggingface_id]
  if (!quantizationId) return
  
  startingModels.value[quantizationId] = true
  try {
    await modelStore.startModel(quantizationId)
    toast.success('Model is starting up')
  } catch (error) {
    toast.error('Failed to start model')
  } finally {
    startingModels.value[quantizationId] = false
  }
}

const stopRunningQuantization = async (modelGroup) => {
  const runningId = getRunningQuantizationId(modelGroup)
  if (!runningId) return
  
  stoppingModels.value[runningId] = true
  try {
    await modelStore.stopModel(runningId)
    toast.success('Model has been stopped')
  } catch (error) {
    toast.error('Failed to stop model')
  } finally {
    stoppingModels.value[runningId] = false
  }
}

const configureSelectedQuantization = (modelGroup) => {
  const quantizationId = selectedQuantization.value[modelGroup.huggingface_id]
  if (!quantizationId) return
  
  router.push(`/models/${quantizationId}/config`)
}

const confirmDeleteQuantization = (quantization) => {
  confirm.require({
    message: `Are you sure you want to delete the "${quantization.quantization}" quantization? This will remove the model file and cannot be undone.`,
    header: 'Delete Quantization',
    icon: 'pi pi-exclamation-triangle',
    rejectLabel: 'Cancel',
    acceptLabel: 'Delete',
    accept: async () => {
      try {
        await modelStore.deleteModel(quantization.id)
        toast.success(`${quantization.quantization} quantization has been deleted`)
        
        // If this was the selected quantization, select another one
        const modelGroup = modelStore.modelGroups.find(g => 
          g.quantizations.some(q => q.id === quantization.id)
        )
        if (modelGroup && selectedQuantization.value[modelGroup.huggingface_id] === quantization.id) {
          const remaining = modelGroup.quantizations.filter(q => q.id !== quantization.id)
          if (remaining.length > 0) {
            selectedQuantization.value[modelGroup.huggingface_id] = remaining[0].id
          } else {
            delete selectedQuantization.value[modelGroup.huggingface_id]
          }
        }
      } catch (error) {
        toast.error('Failed to delete quantization')
      }
    }
  })
}

const confirmDeleteGroup = (modelGroup) => {
  confirm.require({
    message: `Are you sure you want to delete all quantizations of "${modelGroup.huggingface_id}"? This will remove all model files and cannot be undone.`,
    header: 'Delete All Quantizations',
    icon: 'pi pi-exclamation-triangle',
    rejectLabel: 'Cancel',
    acceptLabel: 'Delete All',
    accept: async () => {
      try {
        await modelStore.deleteModelGroup(modelGroup.huggingface_id)
        toast.success(`${modelGroup.huggingface_id} has been deleted`)
        
        // Remove from selected quantizations
        delete selectedQuantization.value[modelGroup.huggingface_id]
      } catch (error) {
        toast.error('Failed to delete model group')
      }
    }
  })
}

const refreshModels = async () => {
  try {
    await modelStore.fetchModels()
    toast.success('Models refreshed')
  } catch (error) {
    toast.error('Failed to refresh models')
  }
}

// Pull-to-refresh handlers
const handlePullToRefreshStart = (e) => {
  // Only trigger if user is at the top of the page
  if (window.scrollY === 0 && e.touches && e.touches.length > 0) {
    pullToRefreshStartY.value = e.touches[0].clientY
    isPullToRefreshActive.value = true
  }
}

const handlePullToRefreshMove = (e) => {
  if (!isPullToRefreshActive.value || !e.touches || e.touches.length === 0) return
  
  const currentY = e.touches[0].clientY
  const deltaY = currentY - pullToRefreshStartY.value
  
  // Only allow pull if scrolling from top
  if (window.scrollY === 0 && deltaY > 0) {
    pullToRefreshDistance.value = deltaY
    // Prevent default scrolling if pulling down significantly
    if (deltaY > 10) {
      e.preventDefault()
    }
  } else {
    // Reset if user scrolls up
    pullToRefreshDistance.value = 0
    isPullToRefreshActive.value = false
  }
}

const handlePullToRefreshEnd = (e) => {
  if (pullToRefreshDistance.value >= pullToRefreshThreshold && window.scrollY === 0) {
    // Trigger refresh
    refreshModels()
  }
  
  // Reset state
  pullToRefreshDistance.value = 0
  pullToRefreshStartY.value = 0
  isPullToRefreshActive.value = false
}

const goToSearch = () => {
  router.push('/search')
}

const formatFileSize = (bytes) => {
  if (bytes === 0) return '0 B'
  const k = 1024
  const sizes = ['B', 'KB', 'MB', 'GB', 'TB']
  const i = Math.floor(Math.log(bytes) / Math.log(k))
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i]
}

const getUpstreamUrl = (proxyName) => {
  const host = window.location.hostname
  const port = '2000' // llama-swap proxy port
  return `@http://${host}:${port}/upstream/${proxyName}/`
}

const openUpstreamUrl = (proxyName) => {
  const host = window.location.hostname
  const port = '2000' // llama-swap proxy port
  const url = `http://${host}:${port}/upstream/${proxyName}/`
  window.open(url, '_blank')
}
</script>

<style scoped>
.model-library {
  max-width: 1400px;
  margin: 0 auto;
}

.downloaded-models {
  position: relative;
  margin-top: var(--spacing-md);
}

.pull-to-refresh-indicator {
  position: absolute;
  top: -50px;
  left: 50%;
  transform: translateX(-50%);
  display: flex;
  align-items: center;
  gap: var(--spacing-sm);
  padding: var(--spacing-md);
  color: var(--accent-cyan);
  font-size: 0.9rem;
  font-weight: 500;
  z-index: 10;
  transition: transform 0.2s ease-out;
  pointer-events: none;
}

.pull-to-refresh-indicator i {
  transition: transform 0.3s ease-out;
}

.pull-to-refresh-indicator i.rotated {
  transform: rotate(180deg);
}

.model-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
  gap: var(--spacing-md);
}

.model-card {
  background: var(--gradient-card);
  border: 1px solid var(--border-primary);
  border-radius: var(--radius-xl);
  padding: var(--spacing-lg);
  transition: all var(--transition-normal);
  box-shadow: var(--shadow-md);
  position: relative;
  overflow: hidden;
  backdrop-filter: blur(10px);
  animation: fadeIn 0.6s ease-out;
}

.model-card::before {
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

.model-card:hover {
  box-shadow: var(--shadow-lg), var(--glow-primary);
  transform: translateY(-5px) scale(1.02);
  border-color: var(--accent-cyan);
}

.model-card:hover::before {
  opacity: 1;
}

.model-card-header {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  margin-bottom: var(--spacing-sm);
}

.model-name {
  font-weight: 700;
  color: var(--text-primary);
  margin-bottom: var(--spacing-sm);
  font-size: 1.1rem;
  line-height: 1.3;
}

.model-status {
  display: flex;
  align-items: center;
}

.status-indicator {
  display: flex;
  align-items: center;
  gap: var(--spacing-xs);
  padding: var(--spacing-xs) var(--spacing-sm);
  border-radius: var(--radius-sm);
  font-size: 0.75rem;
  font-weight: 500;
}

.status-running {
  background: rgba(16, 185, 129, 0.1);
  color: var(--accent-green);
  border: 1px solid rgba(16, 185, 129, 0.2);
}

.status-stopped {
  background: var(--bg-surface);
  color: var(--text-secondary);
  border: 1px solid var(--border-secondary);
}

.model-actions {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: var(--spacing-sm);
  margin-top: var(--spacing-md);
  padding-top: var(--spacing-sm);
  border-top: 1px solid var(--border-primary);
}

.action-group {
  display: flex;
  gap: var(--spacing-xs);
  flex-wrap: wrap;
}

.quantization-list {
  margin: var(--spacing-sm) 0;
  display: flex;
  flex-direction: column;
  gap: var(--spacing-xs);
}

.quantization-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: var(--spacing-sm);
  background: var(--bg-surface);
  border: 1px solid var(--border-primary);
  border-radius: var(--radius-sm);
  transition: all var(--transition-normal);
}

.quantization-item:hover {
  border-color: var(--accent-cyan);
  background: var(--bg-tertiary);
}

.quantization-item.selected {
  border-color: var(--accent-blue);
  background: rgba(59, 130, 246, 0.1);
}

.quantization-info {
  display: flex;
  flex-direction: column;
  gap: var(--spacing-xs);
  flex: 1;
}

.quantization-name {
  display: flex;
  align-items: center;
  gap: var(--spacing-xs);
  font-weight: 600;
  color: var(--text-primary);
  font-size: 0.875rem;
}

.quantization-details {
  display: flex;
  gap: var(--spacing-sm);
  align-items: center;
  font-size: 0.75rem;
}

.quantization-size {
  color: var(--text-secondary);
}

.quantization-status {
  display: flex;
  align-items: center;
  gap: var(--spacing-xs);
  padding: 2px 6px;
  border-radius: var(--radius-sm);
  font-size: 0.75rem;
}

.quantization-status.running {
  background: rgba(16, 185, 129, 0.1);
  color: var(--accent-green);
  border: 1px solid rgba(16, 185, 129, 0.2);
}

.quantization-status.running.llama-swap-running {
  background: rgba(59, 130, 246, 0.1);
  color: var(--accent-blue);
  border: 1px solid rgba(59, 130, 246, 0.2);
}

.status-indicator.llama-swap-running {
  background: rgba(59, 130, 246, 0.1);
  color: var(--accent-blue);
  border: 1px solid rgba(59, 130, 246, 0.2);
}

.upstream-link {
  font-size: 0.7rem !important;
  padding: 1px 3px !important;
  height: auto !important;
  background: rgba(34, 211, 238, 0.1) !important;
  color: var(--accent-cyan) !important;
  border: 1px solid rgba(34, 211, 238, 0.2) !important;
  border-radius: var(--radius-sm) !important;
  transition: all var(--transition-normal) !important;
  min-width: 20px !important;
  margin-left: var(--spacing-xs) !important;
}

.upstream-link:hover {
  background: rgba(34, 211, 238, 0.2) !important;
  border-color: var(--accent-cyan) !important;
  transform: translateY(-1px) !important;
  box-shadow: var(--shadow-sm) !important;
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

.quantization-actions {
  display: flex;
  gap: var(--spacing-xs);
  align-items: center;
}

.quantization-actions .p-button {
  padding: 2px 4px !important;
  min-width: 24px !important;
}

.model-tag.tag-count {
  background: rgba(34, 211, 238, 0.1);
  color: var(--accent-cyan);
  border: 1px solid rgba(34, 211, 238, 0.2);
}

.empty-state {
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

.empty-state::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  height: 2px;
  background: var(--gradient-primary);
  opacity: 0.3;
}

.empty-state i {
  font-size: 3rem !important;
  background: var(--gradient-primary);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  margin-bottom: var(--spacing-lg);
}

.empty-state h3 {
  margin: var(--spacing-lg) 0 var(--spacing-md);
  color: var(--text-primary);
  font-weight: 700;
  font-size: 1.3rem;
}

.empty-state p {
  font-size: 1rem;
  line-height: 1.6;
  max-width: 400px;
  margin: 0 auto var(--spacing-lg);
}

/* Responsive */
@media (max-width: 768px) {
  .model-grid {
    grid-template-columns: 1fr;
  }
  
  .model-actions {
    flex-direction: column;
  }
}
</style>