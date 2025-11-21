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
          :placeholder="`Search HuggingFace for ${formatLabel} models...`"
            @keyup.enter="performSearch"
            class="search-input"
          />
        <Dropdown
          v-model="selectedFormat"
          :options="formatOptions"
          optionLabel="label"
          optionValue="value"
          class="format-dropdown"
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
              <div class="model-name-row">
                <div class="model-name">{{ model.name }}</div>
                <span class="model-format-badge">{{ (model.model_format || 'gguf').toUpperCase() }}</span>
              </div>
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
                <div class="model-pipeline" v-if="model.pipeline_tag">
                  <span class="pipeline-badge">{{ formatPipelineLabel(model.pipeline_tag) }}</span>
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
            
            <div v-if="model.model_format === 'gguf'" class="quantizations">
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
            </div>
            <div v-else class="safetensors-section">
              <div class="safetensors-header">
                <div>
                  <h4>Safetensors Files</h4>
                  <p v-if="Array.isArray(model.safetensors_files) && model.safetensors_files.length">
                    {{ model.safetensors_files.length }} files
                  </p>
                  <p v-else>No safetensors files found for this model.</p>
                </div>
                <template v-if="isSafetensorsDownloaded(model)">
                  <span class="downloaded-badge">
                    <i class="pi pi-check"></i>
                    Downloaded
                  </span>
                </template>
                <template v-else>
                  <Button 
                    label="Download"
                    icon="pi pi-download"
                    severity="success"
                    :disabled="!Array.isArray(model.safetensors_files) || model.safetensors_files.length === 0 || (downloadingModels[model.id]?.size > 0) || isSafetensorsDownloaded(model)"
                    :loading="downloadingModels[model.id]?.size > 0"
                    @click="downloadSafetensorsBundle(model)"
                  />
                </template>
              </div>
              <div 
                v-if="getDownloadedSafetensorsForModel(model.id).length > 0" 
                class="safetensors-files-box"
              >
                <h4>Downloaded Safetensors ({{ getDownloadedSafetensorsForModel(model.id).length }})</h4>
                <div class="safetensors-files-list">
                  <div 
                    v-for="file in getDownloadedSafetensorsForModel(model.id)" 
                    :key="file.filename"
                    class="safetensors-file-name"
                  >
                    {{ file.filename }}
                  </div>
                </div>
              </div>
              <Accordion 
                :multiple="false" 
                :activeIndex="safetensorsAccordionIndex[model.id] ?? null"
                @tab-open="onSafetensorsAccordionOpen(model.id)"
                @tab-close="onSafetensorsAccordionClose(model.id)"
                class="safetensors-accordion"
              >
                <AccordionTab header="Safetensors metadata">
                  <div v-if="modelStore.safetensorsMetadataLoading[model.id]" class="loading-indicator">
                    <i class="pi pi-spin pi-spinner"></i>
                    <span>Loading tensor metadata...</span>
                  </div>
                  <div v-else-if="modelStore.safetensorsMetadata[model.id]" class="safetensors-metadata">
                    <div v-if="modelStore.safetensorsMetadata[model.id].error" class="metadata-error">
                      <i class="pi pi-exclamation-triangle"></i>
                      <span>{{ modelStore.safetensorsMetadata[model.id].error }}</span>
                    </div>
                    <template v-else>
                      <div v-if="modelStore.safetensorsMetadata[model.id].total_files === 0" class="metadata-empty">
                        No safetensors files found in this repository
                      </div>
                      <template v-else>
                        <div class="dtype-summary">
                          <h5>Data types</h5>
                          <div 
                            v-for="(count, dtype) in modelStore.safetensorsMetadata[model.id].dtype_totals" 
                            :key="dtype"
                            class="dtype-row"
                          >
                            <span class="dtype-name">{{ dtype }}</span>
                            <span class="dtype-count">{{ formatNumber(count) }}</span>
                          </div>
                        </div>
                        <div class="metadata-files">
                          <h5>Files</h5>
                          <div 
                            v-for="fileMeta in modelStore.safetensorsMetadata[model.id].files" 
                            :key="fileMeta.filename"
                            class="metadata-file-row"
                          >
                            <div class="metadata-file-name">{{ fileMeta.filename }} ({{ fileMeta.tensor_count }} tensors)</div>
                            <div class="metadata-dtypes">
                              <span 
                                v-for="(count, dtype) in fileMeta.dtype_counts" 
                                :key="dtype"
                                class="dtype-chip"
                              >
                                {{ dtype }}: {{ formatNumber(count) }}
                              </span>
                            </div>
                          </div>
                        </div>
                      </template>
                    </template>
                  </div>
                  <div v-else class="metadata-empty">
                    Expand to load tensor metadata
                  </div>
                </AccordionTab>
              </Accordion>
            </div>
            
            <!-- Download Progress - Multiple concurrent downloads -->
            <div v-if="getModelDownloadProgress(model.id).length > 0" class="downloads-container">
              <div 
                v-for="progressData in getModelDownloadProgress(model.id)" 
                :key="progressData.taskId"
                class="download-progress"
              >
                <div class="progress-header">
                  <span class="progress-filename">
                    <template v-if="progressData.format === 'safetensors-bundle'">
                      Safetensors Bundle — {{ progressData.current_filename || progressData.filename }}
                    </template>
                    <template v-else>
                      {{ progressData.quantization }} - {{ progressData.filename }}
                    </template>
                  </span>
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
                  <div v-if="progressData.format === 'safetensors-bundle' || progressData.format === 'gguf-bundle'" class="progress-bundle-row">
                    <span>File {{ progressData.files_completed }} / {{ progressData.files_total }}</span>
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
        <p>Enter a search term above to find {{ formatLabel }} models on HuggingFace.</p>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, watch, onUnmounted, computed } from 'vue'
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
const formatOptions = [
  { label: 'GGUF', value: 'gguf' },
  { label: 'Safetensors', value: 'safetensors' }
]
const selectedFormat = ref('gguf')
const formatLabel = computed(() => selectedFormat.value === 'safetensors' ? 'Safetensors' : 'GGUF')
const selectedQuantization = ref({})
const downloadingModels = ref({}) // {[modelId]: Set of task_ids}
const downloadProgress = ref({}) // {[task_id]: {modelId, quantization, progress, ...}}
const loadingQuantizationSizes = ref({})
const activeDownloadPolling = ref(null) // Polling interval ID
const safetensorsAccordionIndex = ref({})

const findModelByFilename = (filename) => {
  if (!Array.isArray(modelStore.searchResults)) return null
  return modelStore.searchResults.find(m => {
    const quantizations = Object.values(m.quantizations || {})
    if (quantizations.some(q => Array.isArray(q.files) && q.files.some(f => f.filename === filename))) {
      return true
    }
    return Array.isArray(m.safetensors_files) && m.safetensors_files.some(file => file.filename === filename)
  }) || null
}

onMounted(async () => {
  await modelStore.fetchModels()
  await modelStore.fetchSafetensorsModels()
  await modelStore.fetchHuggingfaceTokenStatus()
  selectedFormat.value = modelStore.searchFormat || 'gguf'
  
  // Subscribe to download progress updates
  wsStore.subscribeToDownloadProgress((data) => {
    const taskId = data.task_id
    if (!taskId) return
    
    // First, try to find model by stored taskId in downloadingModels
    let modelId = null
    for (const [mid, taskSet] of Object.entries(downloadingModels.value)) {
      if (taskSet.has(taskId)) {
        modelId = mid
        break
      }
    }
    
    // If not found by taskId, try to find by huggingface_id (most reliable)
    let model = null
    if (!modelId && data.huggingface_id) {
      model = Array.isArray(modelStore.searchResults) 
        ? modelStore.searchResults.find(m => m.id === data.huggingface_id)
        : null
      if (model) {
        modelId = model.id
      }
    }
    
    // Fallback to filename matching if huggingface_id not available
    if (!model && !modelId) {
      model = findModelByFilename(data.filename)
      if (model) {
        modelId = model.id
      }
    }
    
    // If we have modelId but not model object, find it
    if (modelId && !model) {
      model = Array.isArray(modelStore.searchResults)
        ? modelStore.searchResults.find(m => m.id === modelId)
        : null
    }
    
    const formatFromMessage = data.model_format || model?.model_format || 'gguf'
    
    if (modelId) {
      // Ensure taskId is tracked for this model
      if (!downloadingModels.value[modelId]) {
        downloadingModels.value[modelId] = new Set()
      }
      downloadingModels.value[modelId].add(taskId)
      
      let quantization = data.filename
      if (formatFromMessage === 'gguf' || formatFromMessage === 'gguf-bundle') {
        const quantMatch = data.filename.match(/Q\d+[K_]?[A-Z]*|IQ\d+_[A-Z]+/)
        quantization = quantMatch ? quantMatch[0] : 'unknown'
      }
      
      const isBundle = formatFromMessage === 'safetensors-bundle' || formatFromMessage === 'gguf-bundle'
      const currentFilename = data.current_filename || data.filename
      downloadProgress.value[taskId] = {
        modelId: modelId,
        quantization: isBundle ? (formatFromMessage === 'gguf-bundle' ? 'GGUF Bundle' : 'Safetensors Bundle') : quantization,
        progress: data.progress,
        message: data.message,
        bytes_downloaded: data.bytes_downloaded,
        total_bytes: data.total_bytes,
        speed_mbps: data.speed_mbps,
        eta_seconds: data.eta_seconds,
        filename: currentFilename,
        current_filename: currentFilename,
        format: formatFromMessage,
        files_total: data.files_total || (isBundle ? (formatFromMessage === 'safetensors-bundle' ? model?.safetensors_files?.length || 1 : 1) : 1),
        files_completed: data.files_completed || (isBundle ? 0 : 0)
      }
      
      // Remove progress when download completes
      if (data.progress >= 100) {
        setTimeout(() => {
          delete downloadProgress.value[taskId]
          // Remove task_id from downloading models
          if (downloadingModels.value[modelId]) {
            downloadingModels.value[modelId].delete(taskId)
            if (downloadingModels.value[modelId].size === 0) {
              delete downloadingModels.value[modelId]
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
  const format = (data.model_format || '').toLowerCase()
  if (format === 'safetensors' || format === 'safetensors_bundle' || format === 'safetensors-bundle') {
    await modelStore.fetchSafetensorsModels()
  }
  
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

watch(() => modelStore.searchFormat, (format) => {
  if (format && format !== selectedFormat.value) {
    selectedFormat.value = format
  }
})

watch(selectedFormat, async (newFormat, oldFormat) => {
  if (!searchQuery.value.trim()) return
  if (newFormat === oldFormat) return
  try {
    await modelStore.searchModels(searchQuery.value, 20, newFormat)
  } catch (error) {
    toast.error(`Failed to search for ${formatLabel.value} models`)
  }
})

const performSearch = async () => {
  if (!searchQuery.value.trim()) return
  
  try {
    await modelStore.searchModels(searchQuery.value, 20, selectedFormat.value)
    await modelStore.fetchSafetensorsModels()
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
  return modelStore.downloadedModels
    .filter(model => model.huggingface_id === huggingfaceId)
    .map(model => model.quantization)
}

const getDownloadedQuantizationsForModel = (huggingfaceId) => {
  return modelStore.downloadedModels
    .filter(model => model.huggingface_id === huggingfaceId)
    .map(model => ({
      quantization: model.quantization,
      name: model.name,
      file_size: model.file_size,
      downloaded_at: model.downloaded_at
    }))
}

const getDownloadedSafetensorsForModel = (huggingfaceId) => {
  if (!huggingfaceId) return []
  const group = (modelStore.safetensorsModels || []).find(entry => entry.huggingface_id === huggingfaceId)
  return group?.files || []
}

const getQuantizationOptions = (quantizations, huggingfaceId) => {
  if (!quantizations || typeof quantizations !== 'object') return []
  
  const downloadedQuantizations = getDownloadedQuantizations(huggingfaceId)
  
  // Convert object to array format - ONLY show sizes if they come from API call
  const options = Object.entries(quantizations).map(([name, data]) => {
    let sizeText = ''
    let statusText = ''
    
    // Prefer aggregated total_size/size_mb (may represent multiple shards)
    const sizeMB = data.size_mb || (data.total_size ? data.total_size / (1024 * 1024) : 0)
    if (sizeMB && sizeMB > 0) {
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
  
  // Prefer aggregated total_size/size_mb which may represent multiple shards
  let sizeMB = quant.size_mb
  if (!sizeMB && quant.total_size && quant.total_size > 0) {
    sizeMB = quant.total_size / (1024 * 1024)
  }

  if (sizeMB && sizeMB > 0) {
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
  
  // Fallback text if no size data available
  return 'Unknown size'
}

const onDropdownOpen = async (modelId) => {
  // Fetch actual file sizes from HuggingFace API when dropdown opens
  const model = Array.isArray(modelStore.searchResults) ? 
    modelStore.searchResults.find(m => m.id === modelId) : null
  
  if (model && model.model_format === 'gguf' && model.quantizations) {
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
  
  if (!model || model.model_format !== 'gguf' || !quantization) return
  
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
    
    // If we have a bundle of files for this quantization, use GGUF bundle endpoint
    if (Array.isArray(quantizationData.files) && quantizationData.files.length > 0) {
      const filesPayload = quantizationData.files.map((file) => ({
        filename: file.filename,
        size: file.size || 0
      }))

      const response = await modelStore.downloadGgufBundle(
        model.id,
        quantization,
        filesPayload,
        model.pipeline_tag || null
      )

      const taskId = response.task_id
      if (taskId) {
        downloadingModels.value[modelId].add(taskId)
      }
      toast.success(`Downloading ${model.name} (${quantization})`)
      return
    }

    // Fallback: legacy single-file behavior
    let totalBytes = 0
    const sizeMB = quantizationData.size_mb || (quantizationData.total_size ? quantizationData.total_size / (1024 * 1024) : 0)
    if (sizeMB && sizeMB > 0) {
      totalBytes = Math.round(sizeMB * 1024 * 1024)
    } else if (quantizationData.size && quantizationData.size > 0) {
      if (quantizationData.size > 1000000) {
        totalBytes = quantizationData.size
      } else {
        totalBytes = Math.round(quantizationData.size * 1024 * 1024)
      }
    }
    
    console.log(`Downloading ${quantizationData.filename}: ${totalBytes} bytes`)
    
    const response = await modelStore.downloadModel(
      model.id,
      quantizationData.filename,
      totalBytes,
      model.model_format || 'gguf',
      model.pipeline_tag || null
    )
    
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

const downloadSafetensorsBundle = async (model) => {
  if (!model) {
    toast.warning('Model details unavailable')
    return
  }

  const repoFiles = Array.isArray(model.repo_files) && model.repo_files.length > 0
    ? model.repo_files
    : model.safetensors_files

  if (!Array.isArray(repoFiles) || repoFiles.length === 0) {
    toast.warning('No files available to download')
    return
  }

  const filesPayload = repoFiles.map((file) => ({
    filename: file.filename,
    size: file.size || 0
  }))

  try {
    if (!downloadingModels.value[model.id]) {
      downloadingModels.value[model.id] = new Set()
    }
    const response = await modelStore.downloadSafetensorsBundle(model.id, filesPayload)
    const taskId = response.task_id
    if (taskId) {
      downloadingModels.value[model.id].add(taskId)
    }
    toast.success('Downloading safetensors bundle')
  } catch (error) {
    if (error.response?.status === 409) {
      toast.warning('Safetensors bundle already downloading')
    } else {
      toast.error('Failed to start safetensors bundle download')
    }
    console.error('Safetensors bundle download error:', error)
  }
}

const ensureSafetensorsMetadata = async (modelId) => {
  if (!modelId) return
  if (modelStore.safetensorsMetadata?.[modelId]) return
  try {
    await modelStore.fetchSafetensorsMetadata(modelId)
  } catch (error) {
    console.error('Failed to load safetensors metadata:', error)
    toast.error('Unable to load safetensors metadata')
  }
}

const onSafetensorsAccordionOpen = async (modelId) => {
  safetensorsAccordionIndex.value[modelId] = 0
  await ensureSafetensorsMetadata(modelId)
}

const onSafetensorsAccordionClose = (modelId) => {
  safetensorsAccordionIndex.value[modelId] = null
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

const formatPipelineLabel = (tag) => {
  if (!tag) return ''
  const lower = tag.toLowerCase()
  if (lower.includes('embed')) return 'Embedding'
  if (lower.includes('feature')) return 'Feature Extraction'
  return tag
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
  // Get all active progress entries for this model
  // We no longer filter out progress >= 100 here; instead, entries are cleaned up
  // a few seconds after completion so the user can briefly see the 100% state.
  return Object.entries(downloadProgress.value)
    .filter(([taskId, data]) => data.modelId === modelId)
    .map(([taskId, data]) => ({
      taskId,
      ...data
    }))
}

const isSafetensorsDownloaded = (model) => {
  if (!model) return false
  const huggingfaceId = model.huggingface_id || model.id
  if (!huggingfaceId) return false
  return getDownloadedSafetensorsForModel(huggingfaceId).length > 0
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

.format-dropdown {
  width: 180px;
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

.model-name-row {
  display: flex;
  align-items: center;
  gap: var(--spacing-xs);
}

.model-format-badge {
  padding: 2px 8px;
  border-radius: var(--radius-sm);
  background: var(--accent-cyan);
  color: #fff;
  font-size: 0.7rem;
  font-weight: 600;
  letter-spacing: 0.05em;
}

.model-pipeline {
  margin-top: var(--spacing-xs);
}

.pipeline-badge {
  display: inline-flex;
  align-items: center;
  gap: var(--spacing-xxs);
  padding: 2px 8px;
  border-radius: var(--radius-sm);
  background: rgba(14, 165, 233, 0.12);
  color: var(--accent-cyan);
  font-size: 0.7rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  border: 1px solid rgba(14, 165, 233, 0.25);
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

.safetensors-section {
  margin-top: var(--spacing-sm);
  padding-top: var(--spacing-sm);
  border-top: 1px solid var(--border-primary);
  display: flex;
  flex-direction: column;
  gap: var(--spacing-sm);
}

.safetensors-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: var(--spacing-md);
}

.safetensors-header h4 {
  margin: 0;
}

.safetensors-header p {
  margin: 4px 0 0;
  color: var(--text-secondary);
  font-size: 0.85rem;
}

.safetensors-files-box {
  margin-top: var(--spacing-sm);
  padding: var(--spacing-md);
  background: var(--bg-surface);
  border: 1px solid var(--border-primary);
  border-radius: var(--radius-sm);
}

.safetensors-files-box h4 {
  margin: 0 0 var(--spacing-sm);
  font-size: 0.9rem;
  font-weight: 600;
  color: var(--text-primary);
}

.safetensors-files-list {
  display: flex;
  flex-direction: column;
  gap: 2px;
  font-family: monospace;
  font-size: 0.85rem;
  color: var(--text-secondary);
  line-height: 1.6;
}

.safetensors-file-name {
  padding: 2px 0;
  word-break: break-all;
}

.safetensors-file {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: var(--spacing-sm);
  border: 1px solid var(--border-primary);
  border-radius: var(--radius-sm);
  background: var(--bg-surface);
}

.safetensors-file .file-info {
  display: flex;
  flex-direction: column;
  gap: 2px;
}

.file-name {
  font-weight: 600;
  color: var(--text-primary);
}

.safetensors-accordion {
  border: 1px solid var(--border-primary);
  border-radius: var(--radius-sm);
}

.safetensors-metadata {
  display: flex;
  flex-direction: column;
  gap: var(--spacing-sm);
}

.dtype-summary,
.metadata-files {
  background: var(--bg-surface);
  padding: var(--spacing-sm);
  border-radius: var(--radius-sm);
  border: 1px solid var(--border-primary);
}

.dtype-row,
.metadata-file-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-top: 4px;
  font-size: 0.85rem;
}

.dtype-chip {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  background: var(--bg-tertiary);
  padding: 2px 6px;
  border-radius: var(--radius-sm);
  font-size: 0.75rem;
  margin-right: var(--spacing-xs);
}

.metadata-empty,
.empty-safetensors {
  font-size: 0.85rem;
  color: var(--text-secondary);
  text-align: center;
  padding: var(--spacing-md);
}

.metadata-error {
  display: flex;
  align-items: center;
  gap: var(--spacing-sm);
  padding: var(--spacing-sm);
  background: var(--bg-warning);
  border: 1px solid var(--border-warning);
  border-radius: var(--radius-sm);
  color: var(--text-warning);
  font-size: 0.9rem;
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

.progress-bundle-row {
  font-size: 0.85rem;
  color: var(--text-secondary);
  margin-top: 4px;
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
