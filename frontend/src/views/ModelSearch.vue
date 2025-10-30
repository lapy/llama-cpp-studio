<template>
  <div class="model-search">
    <div class="card">
      <div class="card-header">
        <h2 class="card-title">Model Search</h2>
        <div class="header-actions">
          <Button 
            icon="pi pi-refresh" 
            @click="refreshSearch"
            :loading="modelStore.searchLoading"
            severity="secondary"
            text
          />
        </div>
      </div>

      <!-- HuggingFace Token Section -->
      <div class="token-section">
        <Accordion :multiple="false" :activeIndex="tokenAccordionIndex">
          <AccordionTab>
            <template #header>
              <div class="token-header">
                <i class="pi pi-key"></i>
                <span>HuggingFace API Token</span>
                <span v-if="modelStore.hasHuggingfaceToken" class="token-status-indicator">
                  <i class="pi pi-check-circle"></i>
                  <span v-if="modelStore.tokenFromEnvironment">Set via Environment</span>
                  <span v-else>Configured</span>
                </span>
              </div>
            </template>
            <div v-if="!modelStore.hasHuggingfaceToken" class="token-setup">
              <p class="token-info">
                <i class="pi pi-info-circle"></i>
                Set a HuggingFace API token to search for models. Without a token, model search will be disabled.
              </p>
              <div class="token-input">
                <InputText 
                  v-model="newToken"
                  placeholder="Enter your HuggingFace API token"
                  type="password"
                  class="token-field"
                />
                <Button 
                  label="Set Token" 
                  icon="pi pi-key"
                  @click="setToken"
                  :loading="settingToken"
                  :disabled="!newToken.trim()"
                />
              </div>
              <p class="token-help">
                Get your token from <a href="https://huggingface.co/settings/tokens" target="_blank">HuggingFace Settings</a>
              </p>
            </div>
            <div v-else class="token-status">
              <p class="token-success">
                <i class="pi pi-check-circle"></i>
                HuggingFace token is set ({{ modelStore.huggingfaceToken }})
                <span v-if="modelStore.tokenFromEnvironment" class="env-badge">
                  <i class="pi pi-cog"></i>
                  From Environment Variable
                </span>
              </p>
              <Button 
                v-if="!modelStore.tokenFromEnvironment"
                label="Clear Token" 
                icon="pi pi-trash"
                severity="danger"
                outlined
                @click="clearToken"
                :loading="settingToken"
              />
              <p v-else class="env-info">
                <i class="pi pi-info-circle"></i>
                Token is set via HUGGINGFACE_API_KEY environment variable and cannot be modified via UI
              </p>
            </div>
          </AccordionTab>
        </Accordion>
      </div>

      <!-- Search Section -->
      <div class="search-section">
        <div class="search-bar">
          <InputText 
            v-model="searchQuery"
            placeholder="Search HuggingFace for GGUF models..."
            @keyup.enter="performSearch"
            class="search-input"
          />
          <Button 
            label="Search" 
            icon="pi pi-search"
            @click="performSearch"
            :loading="modelStore.searchLoading"
            :disabled="!searchQuery.trim()"
          />
        </div>
      </div>

      <!-- Search Results -->
      <div v-if="Array.isArray(modelStore.searchResults) && modelStore.searchResults.length > 0" class="search-results">
        <h3>Search Results</h3>
        <div class="model-grid">
          <div 
            v-for="model in modelStore.searchResults" 
            :key="model.id"
            class="model-card"
          >
            <div class="model-card-header">
              <div>
                <div class="model-name">{{ model.name }}</div>
                <div v-if="model.author || (typeof model.id === 'string' && model.id.includes('/'))" class="model-author">
                  by {{ model.author || (typeof model.id === 'string' && model.id.includes('/') ? model.id.split('/')[0] : '') }}
                </div>
                <div class="model-meta" v-if="model.parameters || model.architecture || model.language?.length">
                  <div class="model-meta-item" v-if="model.parameters">
                    <span>Parameters:</span>
                    <span>{{ model.parameters }}</span>
                  </div>
                  <div class="model-meta-item" v-if="model.architecture">
                    <span>Architecture:</span>
                    <span>{{ model.architecture }}</span>
                  </div>
                  <div class="model-meta-item" v-if="Array.isArray(model.language) && model.language.length">
                    <span>Language:</span>
                    <span>{{ model.language.join(', ') }}</span>
                  </div>
                  <div class="model-meta-item" v-if="model.license">
                    <span>License:</span>
                    <span>{{ model.license }}</span>
                  </div>
                </div>
              </div>
              <div class="model-stats">
                <i class="pi pi-download"></i>
                <span>{{ formatNumber(model.downloads) }}</span>
                <i class="pi pi-heart" v-if="model.likes"></i>
                <span v-if="model.likes">{{ formatNumber(model.likes) }}</span>
              </div>
            </div>
            
            <div class="model-description" v-if="model.description">
              {{ truncateText(model.description, 100) }}
            </div>
            
            <div class="model-links" v-if="model.readme_url">
              <a :href="model.readme_url" target="_blank" class="readme-link">
                <i class="pi pi-external-link"></i>
                View README
              </a>
            </div>
            
            <div class="quantizations">
              <!-- Downloaded Quantizations Section -->
              <div v-if="getDownloadedQuantizationsForModel(model.id).length > 0" class="downloaded-quantizations">
                <h4>Downloaded Quantizations:</h4>
                <div class="downloaded-list">
                  <div 
                    v-for="downloaded in getDownloadedQuantizationsForModel(model.id)" 
                    :key="downloaded.quantization"
                    class="downloaded-item"
                  >
                    <div class="downloaded-info">
                      <span class="downloaded-name">{{ downloaded.quantization }}</span>
                      <span class="downloaded-badge">
                        <i class="pi pi-check"></i>
                        Downloaded
                      </span>
                    </div>
                    <div class="downloaded-details">
                      <span class="downloaded-size">{{ formatFileSize(downloaded.file_size) }}</span>
                      <span class="downloaded-date">{{ formatDate(downloaded.downloaded_at) }}</span>
                    </div>
                  </div>
                </div>
              </div>
              
              <!-- Available Quantizations Section -->
              <h4>Available Quantizations:</h4>
              <div class="quantization-selector">
                <Dropdown 
                  v-model="selectedQuantization[model.id]"
                  :options="getQuantizationOptions(model.quantizations, model.id)"
                  optionLabel="label"
                  optionValue="value"
                  :placeholder="loadingQuantizationSizes[model.id] ? 'Loading sizes...' : 'Select quantization'"
                  class="quantization-dropdown"
                  :loading="loadingQuantizationSizes[model.id]"
                  @change="onQuantizationChange(model.id, $event.value)"
                  @show="onDropdownOpen(model.id)"
                />
                <div v-if="loadingQuantizationSizes[model.id]" class="loading-indicator">
                  <i class="pi pi-spin pi-spinner"></i>
                  <span>Fetching file sizes...</span>
                </div>
              </div>
              <div v-if="selectedQuantization[model.id]" class="selected-quantization-info">
                <div class="quant-info">
                  <div class="quant-name-row">
                    <span class="quant-name">{{ selectedQuantization[model.id] }}</span>
                  </div>
                  <span v-if="getQuantizationSizeWithUnit(model.quantizations, selectedQuantization[model.id])" class="quant-size">{{ getQuantizationSizeWithUnit(model.quantizations, selectedQuantization[model.id]) }}</span>
                </div>
                <Button 
                  :label="isModelDownloaded(model.id, selectedQuantization[model.id]) ? 'Downloaded' : 'Download'"
                  :icon="isModelDownloaded(model.id, selectedQuantization[model.id]) ? 'pi pi-check' : 'pi pi-download'"
                  @click="downloadSelectedQuantization(model.id)"
                  :disabled="isModelDownloaded(model.id, selectedQuantization[model.id]) || (downloadingModels[model.id]?.size > 0) || !selectedQuantization[model.id]"
                  :loading="downloadingModels[model.id]?.size > 0"
                  class="download-button"
                  :severity="isModelDownloaded(model.id, selectedQuantization[model.id]) ? 'success' : 'success'"
                />
              </div>
              
              <!-- Download Progress - Multiple concurrent downloads -->
              <div v-if="getModelDownloadProgress(model.id).length > 0" class="downloads-container">
                <div 
                  v-for="progressData in getModelDownloadProgress(model.id)" 
                  :key="progressData.taskId"
                  class="download-progress"
                >
                  <div class="progress-header">
                    <span class="progress-filename">{{ progressData.quantization }} - {{ progressData.filename }}</span>
                    <span class="progress-percentage">{{ progressData.progress }}%</span>
                  </div>
                  <div class="progress-bar-container">
                    <div class="progress-bar" :style="{ width: progressData.progress + '%' }"></div>
                  </div>
                  <div class="progress-details">
                    <div class="progress-row-1">
                      <span class="progress-size">
                        {{ formatBytes(progressData.bytes_downloaded) }} / {{ formatBytes(progressData.total_bytes) }}
                      </span>
                      <span v-if="progressData.speed_mbps > 0" class="progress-speed">
                        {{ (progressData.speed_mbps || 0).toFixed(1) }} MB/s
                      </span>
                    </div>
                    <div v-if="progressData.eta_seconds > 0" class="progress-eta-row">
                      <span class="progress-eta">
                        {{ formatTime(progressData.eta_seconds) }} remaining
                      </span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- Empty State -->
      <div v-else-if="!modelStore.searchLoading && searchQuery" class="empty-state">
        <i class="pi pi-search"></i>
        <h3>No models found</h3>
        <p>Try adjusting your search terms.</p>
      </div>

      <!-- Initial State -->
      <div v-else class="empty-state">
        <i class="pi pi-search"></i>
        <h3>Search for Models</h3>
        <p>Enter a search term above to find GGUF models on HuggingFace.</p>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, watch, onUnmounted } from 'vue'
import { useModelStore } from '@/stores/models'
import { useWebSocketStore } from '@/stores/websocket'
import { toast } from 'vue3-toastify'
import { useConfirm } from 'primevue/useconfirm'
import Button from 'primevue/button'
import InputText from 'primevue/inputtext'
import Dropdown from 'primevue/dropdown'
import Accordion from 'primevue/accordion'
import AccordionTab from 'primevue/accordiontab'

const modelStore = useModelStore()
const wsStore = useWebSocketStore()
const confirm = useConfirm()

// Reactive state
const searchQuery = ref('')
const newToken = ref('')
const settingToken = ref(false)
const tokenAccordionIndex = ref(-1)
const selectedQuantization = ref({})
const downloadingModels = ref({}) // {[modelId]: Set of task_ids}
const downloadProgress = ref({}) // {[task_id]: {modelId, quantization, progress, ...}}
const loadingQuantizationSizes = ref({})
const activeDownloadPolling = ref(null) // Polling interval ID

onMounted(async () => {
  await modelStore.fetchModels()
  await modelStore.fetchHuggingfaceTokenStatus()
  
  // Subscribe to download progress updates
  wsStore.subscribeToDownloadProgress((data) => {
    const taskId = data.task_id
    if (!taskId) return
    
    // Find model from search results by filename
    const model = Array.isArray(modelStore.searchResults) ? 
      modelStore.searchResults.find(m => {
        const quantizationData = Object.values(m.quantizations || {}).find(q => q.filename === data.filename)
        return quantizationData !== undefined
      }) : null
    
    if (model) {
      // Extract quantization from filename using regex
      const quantMatch = data.filename.match(/Q\d+[K_]?[A-Z]*|IQ\d+_[A-Z]+/)
      const quantization = quantMatch ? quantMatch[0] : 'unknown'
      
      downloadProgress.value[taskId] = {
        modelId: model.id,
        quantization: quantization,
        progress: data.progress,
        message: data.message,
        bytes_downloaded: data.bytes_downloaded,
        total_bytes: data.total_bytes,
        speed_mbps: data.speed_mbps,
        eta_seconds: data.eta_seconds,
        filename: data.filename
      }
      
      // Remove progress when download completes
      if (data.progress >= 100) {
        setTimeout(() => {
          delete downloadProgress.value[taskId]
          // Remove task_id from downloading models
          if (downloadingModels.value[model.id]) {
            downloadingModels.value[model.id].delete(taskId)
            if (downloadingModels.value[model.id].size === 0) {
              delete downloadingModels.value[model.id]
            }
          }
        }, 3000)
      }
    }
  })

// Subscribe to download complete events
wsStore.subscribeToDownloadComplete(async (data) => {
  console.log('Download complete event received:', data)
  
  // Refresh models list to update downloaded status
  await modelStore.fetchModels()
  
  // Force reactivity update on search results to refresh dropdown states
  if (Array.isArray(modelStore.searchResults)) {
    modelStore.searchResults = [...modelStore.searchResults]
  }
  
  // Show success notification
  toast.success(`Download completed: ${data.filename}`)
})

  // Watch for active downloads and start/stop polling
  watch(() => Object.keys(downloadProgress.value).length, (activeCount) => {
    if (activeCount > 0 && !activeDownloadPolling.value) {
      // Start polling every 10 seconds while downloads are active
      activeDownloadPolling.value = setInterval(async () => {
        await modelStore.fetchModels()
      }, 10000)
    } else if (activeCount === 0 && activeDownloadPolling.value) {
      // Stop polling when no active downloads
      clearInterval(activeDownloadPolling.value)
      activeDownloadPolling.value = null
    }
  })

  // Cleanup on unmount
  onUnmounted(() => {
    if (activeDownloadPolling.value) {
      clearInterval(activeDownloadPolling.value)
    }
  })
})

const performSearch = async () => {
  if (!searchQuery.value.trim()) return
  
  try {
    await modelStore.searchModels(searchQuery.value)
  } catch (error) {
    toast.error('Failed to search for models')
  }
}

const refreshSearch = () => {
  if (searchQuery.value.trim()) {
    performSearch()
  }
}

const setToken = async () => {
  if (!newToken.value.trim()) return
  
  settingToken.value = true
  try {
    await modelStore.setHuggingfaceToken(newToken.value)
    newToken.value = ''
    tokenAccordionIndex.value = -1
    toast.success('HuggingFace API token has been configured')
  } catch (error) {
    toast.error('Failed to set HuggingFace token')
  } finally {
    settingToken.value = false
  }
}

const clearToken = async () => {
  settingToken.value = true
  try {
    await modelStore.clearHuggingfaceToken()
    toast.success('HuggingFace API token has been removed')
  } catch (error) {
    toast.error('Failed to clear HuggingFace token')
  } finally {
    settingToken.value = false
  }
}

const getDownloadedQuantizations = (huggingfaceId) => {
  return modelStore.downloadedModels.filter(model => 
    model.huggingface_id === huggingfaceId
  ).map(model => model.quantization)
}

const getDownloadedQuantizationsForModel = (huggingfaceId) => {
  return modelStore.downloadedModels.filter(model => 
    model.huggingface_id === huggingfaceId
  ).map(model => ({
    quantization: model.quantization,
    name: model.name,
    file_size: model.file_size,
    downloaded_at: model.downloaded_at
  }))
}

const getQuantizationOptions = (quantizations, huggingfaceId) => {
  if (!quantizations || typeof quantizations !== 'object') return []
  
  const downloadedQuantizations = getDownloadedQuantizations(huggingfaceId)
  
  // Convert object to array format - ONLY show sizes if they come from API call
  const options = Object.entries(quantizations).map(([name, data]) => {
    let sizeText = ''
    let statusText = ''
    
    // Only show size if we have actual data from API (size_mb field means it came from API)
    if (data.size_mb && data.size_mb > 0) {
      const sizeMB = data.size_mb
      if (sizeMB >= 1024) {
        // Convert to GB for large files
        sizeText = ` (${Math.round((sizeMB / 1024) * 100) / 100} GB)`
      } else {
        sizeText = ` (${Math.round(sizeMB * 100) / 100} MB)`
      }
    }
    
    // Add download status
    if (downloadedQuantizations.includes(name)) {
      statusText = ' ✓ Downloaded'
    }
    
    return {
      label: `${name}${sizeText}${statusText}`,
      value: name,
      disabled: downloadedQuantizations.includes(name),
      sizeMB: data.size_mb || 0 // Store size for sorting
    }
  })
  
  // Sort by file size (increasing/smallest first)
  return options.sort((a, b) => a.sizeMB - b.sizeMB)
}

const getQuantizationSizeWithUnit = (quantizations, quantizationName) => {
  if (!quantizations || typeof quantizations !== 'object' || !quantizationName) return ''
  
  // Get size from object structure
  const quant = quantizations[quantizationName]
  if (!quant) return ''
  
  // Only return size if we have actual data from API
  if (quant.size_mb && quant.size_mb > 0) {
    const sizeMB = quant.size_mb
    if (sizeMB >= 1024) {
      // Convert to GB for large files
      return `${Math.round((sizeMB / 1024) * 100) / 100} GB`
    } else {
      return `${Math.round(sizeMB * 100) / 100} MB`
    }
  } else if (quant.size && quant.size > 0) {
    // Convert bytes to appropriate unit
    const sizeMB = quant.size / (1024 * 1024)
    if (sizeMB >= 1024) {
      return `${Math.round((sizeMB / 1024) * 100) / 100} GB`
    } else {
      return `${Math.round(sizeMB * 100) / 100} MB`
    }
  }
  
  // Return empty string if no size data available
  return ''
}

const onDropdownOpen = async (modelId) => {
  // Fetch actual file sizes from HuggingFace API when dropdown opens
  const model = Array.isArray(modelStore.searchResults) ? 
    modelStore.searchResults.find(m => m.id === modelId) : null
  
  if (model && model.quantizations) {
    // Check if we already have size data from API to avoid unnecessary API calls
    const hasApiData = Object.values(model.quantizations).some(q => q.size_mb)
    if (hasApiData) {
      console.log(`📊 API sizes already available for ${model.id}, skipping API call`)
      return
    }
    
    try {
      loadingQuantizationSizes.value[modelId] = true
      console.log(`🔍 Fetching actual sizes for ${model.id} when dropdown opened...`)
      console.log(`📊 Current quantizations:`, model.quantizations)
      
      const actualQuantizations = await modelStore.getQuantizationSizes(model.id, model.quantizations)
      
      console.log(`📊 API Response:`, actualQuantizations)
      
      // Update the model's quantizations with actual sizes
      // Use Object.assign to ensure Vue reactivity
      Object.assign(model.quantizations, actualQuantizations)
      console.log(`✅ Updated quantizations with API sizes:`, actualQuantizations)
      
      // Force reactivity update
      modelStore.searchResults = [...modelStore.searchResults]
    } catch (error) {
      console.error('❌ Failed to fetch actual quantization sizes:', error)
      // Continue with original sizes if API call fails
    } finally {
      loadingQuantizationSizes.value[modelId] = false
    }
  }
}

const onQuantizationChange = async (modelId, quantization) => {
  selectedQuantization.value[modelId] = quantization
}

const downloadSelectedQuantization = async (modelId) => {
  const model = Array.isArray(modelStore.searchResults) ? 
    modelStore.searchResults.find(m => m.id === modelId) : null
  const quantization = selectedQuantization.value[modelId]
  
  if (!model || !quantization) return
  
  const quantizationData = model.quantizations?.[quantization]
  if (!quantizationData) {
    toast.error('Quantization data not found')
    return
  }
  
  try {
    // Initialize Set for this model if doesn't exist
    if (!downloadingModels.value[modelId]) {
      downloadingModels.value[modelId] = new Set()
    }
    
    // Calculate total bytes - only if we have actual size data
    let totalBytes = 0
    if (quantizationData.size_mb && quantizationData.size_mb > 0) {
      // size_mb is already in MB, convert to bytes
      totalBytes = Math.round(quantizationData.size_mb * 1024 * 1024)
    } else if (quantizationData.size && quantizationData.size > 0) {
      // size might be in bytes or MB, check if it's reasonable
      if (quantizationData.size > 1000000) {
        // Likely already in bytes
        totalBytes = quantizationData.size
      } else {
        // Likely in MB, convert to bytes
        totalBytes = Math.round(quantizationData.size * 1024 * 1024)
      }
    }
    
    console.log(`Downloading ${quantizationData.filename}: ${totalBytes} bytes`)
    
    const response = await modelStore.downloadModel(model.id, quantizationData.filename, totalBytes)
    
    // Store the task_id for tracking
    const taskId = response.task_id
    if (taskId) {
      downloadingModels.value[modelId].add(taskId)
    }
    
    toast.success(`Downloading ${model.name} (${quantization})`)
  } catch (error) {
    // Handle 409 Conflict - already downloading
    if (error.response?.status === 409) {
      toast.warning('This quantization is already being downloaded')
    } else {
      toast.error('Failed to start model download')
    }
    console.error('Download error:', error)
  }
}

const isModelDownloaded = (huggingfaceId, quantization) => {
  // Check if this specific quantization is already downloaded
  return modelStore.allQuantizations.some(model => 
    model.huggingface_id === huggingfaceId && 
    model.quantization === quantization
  )
}

const formatNumber = (num) => {
  if (num >= 1000000) {
    return (num / 1000000).toFixed(1) + 'M'
  } else if (num >= 1000) {
    return (num / 1000).toFixed(1) + 'K'
  }
  return num.toString()
}

const truncateText = (text, maxLength) => {
  if (text.length <= maxLength) return text
  return text.substring(0, maxLength) + '...'
}

const formatBytes = (bytes) => {
  if (!bytes || bytes === 0) return '0 Bytes'
  if (typeof bytes !== 'number') return 'Unknown'
  
  const k = 1024
  const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB']
  const i = Math.floor(Math.log(bytes) / Math.log(k))
  
  // Ensure we don't go out of bounds
  const sizeIndex = Math.min(i, sizes.length - 1)
  const value = bytes / Math.pow(k, sizeIndex)
  
  // Format with appropriate decimal places
  let formattedValue
  if (sizeIndex === 0) {
    formattedValue = Math.round(value) // Bytes - no decimals
  } else if (sizeIndex === 1) {
    formattedValue = Math.round(value * 10) / 10 // KB - 1 decimal
  } else {
    formattedValue = Math.round(value * 100) / 100 // MB+ - 2 decimals
  }
  
  return formattedValue + ' ' + sizes[sizeIndex]
}

const formatFileSize = (bytes) => {
  if (!bytes || bytes === 0) return '0 B'
  if (typeof bytes !== 'number') return 'Unknown'
  
  const k = 1024
  const sizes = ['B', 'KB', 'MB', 'GB', 'TB']
  const i = Math.floor(Math.log(bytes) / Math.log(k))
  
  // Ensure we don't go out of bounds
  const sizeIndex = Math.min(i, sizes.length - 1)
  const value = bytes / Math.pow(k, sizeIndex)
  
  // Format with appropriate decimal places
  let formattedValue
  if (sizeIndex === 0) {
    formattedValue = Math.round(value) // Bytes - no decimals
  } else if (sizeIndex === 1) {
    formattedValue = Math.round(value * 10) / 10 // KB - 1 decimal
  } else {
    formattedValue = Math.round(value * 100) / 100 // MB+ - 2 decimals
  }
  
  return formattedValue + ' ' + sizes[sizeIndex]
}

const formatDate = (dateString) => {
  if (!dateString) return ''
  const date = new Date(dateString)
  return date.toLocaleDateString() + ' ' + date.toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'})
}

const formatTime = (seconds) => {
  if (seconds === 0) return '0s'
  const h = Math.floor(seconds / 3600)
  const m = Math.floor((seconds % 3600) / 60)
  const s = Math.floor(seconds % 60)
  
  let result = ''
  if (h > 0) result += `${h}h `
  if (m > 0) result += `${m}m `
  result += `${s}s`
  return result.trim()
}

const getModelDownloadProgress = (modelId) => {
  // Get all active progress entries for this model (only show if progress < 100)
  return Object.entries(downloadProgress.value)
    .filter(([taskId, data]) => data.modelId === modelId && data.progress < 100)
    .map(([taskId, data]) => ({
      taskId,
      ...data
    }))
}
</script>

<style scoped>
.model-search {
  max-width: 1400px;
  margin: 0 auto;
}

.search-section {
  margin-bottom: var(--spacing-md);
  padding: var(--spacing-md);
  background: var(--bg-surface);
  border-radius: var(--radius-md);
  border: 1px solid var(--border-primary);
}

.search-bar {
  display: flex;
  gap: var(--spacing-sm);
  align-items: center;
}

.search-input {
  flex: 1;
}

.search-results {
  margin-top: var(--spacing-md);
}

.search-results h3 {
  margin-bottom: var(--spacing-md);
  color: var(--text-primary);
  font-weight: 600;
}

.model-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
  gap: var(--spacing-md);
}

.model-card {
  background: var(--bg-card);
  border: 1px solid var(--border-primary);
  border-radius: var(--radius-md);
  padding: var(--spacing-md);
  transition: all var(--transition-normal);
  box-shadow: var(--shadow-sm);
  position: relative;
  overflow: hidden;
}

.model-card:hover {
  box-shadow: var(--shadow-md);
  transform: translateY(-1px);
  border-color: var(--accent-cyan);
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
  margin-bottom: var(--spacing-xs);
}

.model-author {
  font-size: 0.875rem;
  color: var(--text-secondary);
  margin-bottom: var(--spacing-xs);
}

.model-meta {
  margin-top: var(--spacing-xs);
  display: flex;
  flex-direction: column;
  gap: var(--spacing-xs);
}

.model-meta-item {
  display: flex;
  justify-content: space-between;
  font-size: 0.75rem;
  color: var(--text-secondary);
}

.model-meta-item span:first-child {
  font-weight: 600;
}

.model-stats {
  display: flex;
  align-items: center;
  gap: var(--spacing-xs);
  font-size: 0.75rem;
  color: var(--text-secondary);
}

.model-description {
  font-size: 0.875rem;
  color: var(--text-secondary);
  line-height: 1.4;
  margin-bottom: var(--spacing-sm);
}

.model-links {
  margin: var(--spacing-sm) 0;
}

.readme-link {
  display: inline-flex;
  align-items: center;
  gap: var(--spacing-xs);
  color: var(--accent-cyan);
  text-decoration: none;
  font-size: 0.875rem;
}

.readme-link:hover {
  text-decoration: underline;
}

.quantizations {
  margin-top: var(--spacing-sm);
  padding-top: var(--spacing-sm);
  border-top: 1px solid var(--border-primary);
}

.quantizations h4 {
  margin-bottom: var(--spacing-sm);
  font-size: 0.875rem;
  color: var(--text-primary);
  font-weight: 600;
}

.downloaded-quantizations {
  margin-bottom: var(--spacing-md);
  padding-bottom: var(--spacing-sm);
  border-bottom: 1px solid var(--border-primary);
}

.downloaded-list {
  display: flex;
  flex-direction: column;
  gap: var(--spacing-xs);
}

.downloaded-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: var(--spacing-sm);
  background: var(--bg-surface);
  border: 1px solid var(--border-primary);
  border-radius: var(--radius-sm);
  transition: all var(--transition-normal);
}

.downloaded-item:hover {
  border-color: var(--accent-green);
  background: var(--bg-tertiary);
}

.downloaded-info {
  display: flex;
  align-items: center;
  gap: var(--spacing-sm);
}

.downloaded-name {
  font-weight: 600;
  color: var(--text-primary);
}

.downloaded-badge {
  display: flex;
  align-items: center;
  gap: var(--spacing-xs);
  padding: 2px 6px;
  background: var(--accent-green);
  color: white;
  border-radius: var(--radius-xs);
  font-size: 0.75rem;
  font-weight: 500;
}

.downloaded-details {
  display: flex;
  flex-direction: column;
  align-items: flex-end;
  gap: 2px;
}

.downloaded-size {
  font-size: 0.75rem;
  color: var(--text-secondary);
  font-weight: 500;
}

.downloaded-date {
  font-size: 0.7rem;
  color: var(--text-tertiary);
}

.quantization-selector {
  margin-bottom: var(--spacing-sm);
}

.quantization-dropdown {
  width: 100%;
}

.selected-quantization-info {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: var(--spacing-sm);
}

.quant-info {
  display: flex;
  flex-direction: column;
  gap: var(--spacing-xs);
}

.quant-name-row {
  display: flex;
  align-items: center;
  gap: var(--spacing-sm);
}

.quant-name {
  font-weight: 600;
  color: var(--text-primary);
  font-size: 0.875rem;
}

.downloaded-badge {
  display: flex;
  align-items: center;
  gap: var(--spacing-xs);
  padding: 2px 6px;
  background: rgba(16, 185, 129, 0.1);
  color: var(--accent-green);
  border: 1px solid rgba(16, 185, 129, 0.2);
  border-radius: var(--radius-sm);
  font-size: 0.75rem;
  font-weight: 500;
}

.quant-size {
  font-size: 0.75rem;
  color: var(--text-secondary);
}

.download-button {
  flex-shrink: 0;
}

/* Download Progress Styles */
.downloads-container {
  margin-top: var(--spacing-md);
  display: flex;
  flex-direction: column;
  gap: var(--spacing-md);
}

.downloads-container .download-progress {
  margin-top: 0;
}

.download-progress {
  margin-top: var(--spacing-lg);
  padding: var(--spacing-lg);
  background: var(--bg-surface);
  border-radius: var(--radius-md);
  border: 1px solid var(--border-secondary);
  position: relative;
  z-index: 1;
  overflow: hidden;
  box-shadow: var(--shadow-sm);
  width: 100%;
  display: block;
}

.progress-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: var(--spacing-xs);
}

.progress-filename {
  font-size: 0.875rem;
  font-weight: 500;
  color: var(--text-primary);
  flex: 1;
  margin-right: var(--spacing-sm);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.progress-percentage {
  font-size: 0.875rem;
  font-weight: 600;
  color: var(--accent-blue);
  min-width: 3rem;
  text-align: right;
}

.progress-bar-container {
  width: 100%;
  height: 6px;
  background: var(--bg-secondary);
  border-radius: 3px;
  overflow: hidden;
  margin-bottom: var(--spacing-xs);
  position: relative;
}

.progress-bar {
  height: 100%;
  background: linear-gradient(90deg, var(--accent-blue), var(--accent-cyan));
  border-radius: 3px;
  transition: width 0.3s ease;
  position: relative;
  z-index: 1;
}

.progress-details {
  display: flex;
  flex-direction: column;
  gap: var(--spacing-xs);
  font-size: 0.75rem;
  color: var(--text-secondary);
}

.progress-row-1 {
  display: flex;
  gap: var(--spacing-sm);
  align-items: center;
  white-space: nowrap;
}

.progress-size {
  font-weight: 500;
  min-width: 150px;
}

.progress-speed {
  color: var(--accent-green);
  min-width: 70px;
}

.progress-eta-row {
  display: flex;
}

.progress-eta {
  color: var(--accent-orange);
}

.token-section {
  margin-bottom: var(--spacing-md);
}

.token-header {
  display: flex;
  align-items: center;
  gap: var(--spacing-sm);
}

.token-status-indicator {
  display: flex;
  align-items: center;
  gap: var(--spacing-xs);
  margin-left: auto;
  font-size: 0.75rem;
  color: var(--accent-green);
}

.token-setup {
  display: flex;
  flex-direction: column;
  gap: var(--spacing-md);
}

.token-info {
  display: flex;
  align-items: center;
  gap: var(--spacing-sm);
  color: var(--text-secondary);
  font-size: 0.875rem;
}

.token-input {
  display: flex;
  gap: var(--spacing-sm);
}

.token-field {
  flex: 1;
}

.token-help {
  font-size: 0.75rem;
  color: var(--text-muted);
}

.token-help a {
  color: var(--accent-cyan);
  text-decoration: none;
}

.token-help a:hover {
  text-decoration: underline;
}

.token-status {
  display: flex;
  flex-direction: column;
  gap: var(--spacing-sm);
}

.token-success {
  display: flex;
  align-items: center;
  gap: var(--spacing-sm);
  color: var(--accent-green);
  font-size: 0.875rem;
}

.env-badge {
  display: flex;
  align-items: center;
  gap: var(--spacing-xs);
  background: var(--bg-surface);
  padding: var(--spacing-xs) var(--spacing-sm);
  border-radius: var(--radius-sm);
  font-size: 0.75rem;
  color: var(--text-secondary);
}

.env-info {
  display: flex;
  align-items: center;
  gap: var(--spacing-sm);
  color: var(--text-secondary);
  font-size: 0.75rem;
}

.empty-state {
  text-align: center;
  padding: var(--spacing-2xl) var(--spacing-md);
  color: var(--text-secondary);
  background: var(--bg-surface);
  border-radius: var(--radius-md);
  border: 2px dashed var(--border-secondary);
  margin: var(--spacing-md) 0;
}

.empty-state i {
  font-size: 3rem !important;
  color: var(--accent-cyan);
  margin-bottom: var(--spacing-md);
}

/* Loading indicator for quantization sizes */
.loading-indicator {
  display: flex;
  align-items: center;
  gap: var(--spacing-xs);
  margin-top: var(--spacing-xs);
  color: var(--text-secondary);
  font-size: 0.875rem;
}

.loading-indicator i {
  color: var(--accent-blue);
}

.empty-state h3 {
  margin: var(--spacing-md) 0 var(--spacing-sm);
  color: var(--text-primary);
  font-weight: 700;
  font-size: 1.1rem;
}

.empty-state p {
  font-size: 0.875rem;
  line-height: 1.6;
  max-width: 400px;
  margin: 0 auto;
}

/* Responsive */
@media (max-width: 768px) {
  .model-grid {
    grid-template-columns: 1fr;
  }
  
  .search-bar {
    flex-direction: column;
  }
  
  .token-input {
    flex-direction: column;
  }
  
  .selected-quantization-info {
    flex-direction: column;
    align-items: stretch;
  }
}
</style>
