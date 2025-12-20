import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import axios from 'axios'

export const useModelStore = defineStore('models', () => {
  const models = ref([]) // This will now contain grouped models
  const loading = ref(false)
  const searchResults = ref([])
  const searchLoading = ref(false)
  const searchFormat = ref('gguf')
  const huggingfaceToken = ref(null)
  const hasHuggingfaceToken = ref(false)
  const tokenFromEnvironment = ref(false)
  const safetensorsMetadata = ref({})
  const safetensorsMetadataLoading = ref({})
  const safetensorsModels = ref([])
  const safetensorsLoading = ref(false)
  const safetensorsRuntime = ref({})
  const safetensorsRuntimeLoading = ref({})
  const safetensorsMetadataRefreshing = ref({})
  const lmdeployStatus = ref(null)
  const lmdeployStatusLoading = ref(false)
  const lmdeployStarting = ref({})
  const lmdeployStopping = ref({})
  const hfMetadata = ref({})
  const hfMetadataLoading = ref({})
  
  // Model loading state tracking (models currently being loaded by llama-swap)
  const loadingModels = ref({})  // { proxyName: { started_at, elapsed_seconds } }

  // Flatten all quantizations for backward compatibility
  const allQuantizations = computed(() => {
    const quantizations = []
    models.value.forEach(group => {
      group.quantizations.forEach(quant => {
        quantizations.push({
          ...quant,
          base_model_name: group.base_model_name,
          huggingface_id: group.huggingface_id,
          model_type: group.model_type,
          pipeline_tag: quant.pipeline_tag || group.pipeline_tag,
          is_embedding_model: quant.is_embedding_model ?? group.is_embedding_model ?? false
        })
      })
    })
    return quantizations
  })

  const downloadedModels = computed(() => 
    allQuantizations.value.filter(model => model.file_path)
  )

  const runningModels = computed(() => 
    allQuantizations.value.filter(model => model.is_active)
  )

  // Get all model groups (for grouped display)
  const modelGroups = computed(() => models.value)
  
  // Check if a model is currently loading (by model ID or proxy name)
  const isModelLoading = (modelIdOrProxyName) => {
    // Check by proxy name first
    if (loadingModels.value[modelIdOrProxyName]) {
      return true
    }
    // Check by model ID - find the model and check its proxy name
    const model = allQuantizations.value.find(m => m.id === modelIdOrProxyName)
    if (model?.proxy_name && loadingModels.value[model.proxy_name]) {
      return true
    }
    return false
  }
  
  // Get loading progress for a model (elapsed seconds)
  const getModelLoadingProgress = (modelIdOrProxyName) => {
    // Check by proxy name first
    if (loadingModels.value[modelIdOrProxyName]) {
      return loadingModels.value[modelIdOrProxyName]
    }
    // Check by model ID
    const model = allQuantizations.value.find(m => m.id === modelIdOrProxyName)
    if (model?.proxy_name && loadingModels.value[model.proxy_name]) {
      return loadingModels.value[model.proxy_name]
    }
    return null
  }
  
  // Update loading models from unified monitoring data
  const updateLoadingModels = (loadingData) => {
    loadingModels.value = loadingData || {}
  }
  
  // Check if any models are currently loading
  const hasLoadingModels = computed(() => Object.keys(loadingModels.value).length > 0)

  const fetchModels = async () => {
    loading.value = true
    try {
      const response = await axios.get('/api/models')
      models.value = response.data
    } catch (error) {
      console.error('Failed to fetch models:', error)
      throw error
    } finally {
      loading.value = false
    }
  }

  const fetchSafetensorsModels = async () => {
    safetensorsLoading.value = true
    try {
      const response = await axios.get('/api/models/safetensors')
      safetensorsModels.value = Array.isArray(response.data) ? response.data : []
    } catch (error) {
      console.error('Failed to fetch safetensors models:', error)
      throw error
    } finally {
      safetensorsLoading.value = false
    }
  }

  const searchModels = async (query, limit = 20, modelFormat = searchFormat.value) => {
    searchLoading.value = true
    try {
      const response = await axios.post('/api/models/search', { query, limit, model_format: modelFormat })
      // Ensure searchResults is always an array
      searchResults.value = Array.isArray(response.data) ? response.data : []
      searchFormat.value = modelFormat
      return searchResults.value
    } catch (error) {
      console.error('Failed to search models:', error)
      // Ensure searchResults is reset to empty array on error
      searchResults.value = []
      throw error
    } finally {
      searchLoading.value = false
    }
  }

  const downloadModel = async (huggingfaceId, filename, totalBytes = 0, modelFormat = 'gguf', pipelineTag = null) => {
  try {
    const response = await axios.post('/api/models/download', {
      huggingface_id: huggingfaceId,
      filename,
      total_bytes: totalBytes,
      model_format: modelFormat,
      pipeline_tag: pipelineTag
    })
    // Refresh models list after download starts
    await fetchModels()
    if (modelFormat === 'safetensors') {
      await fetchSafetensorsModels()
    }
    return response.data
  } catch (error) {
    console.error('Failed to download model:', error)
    throw error
  }
}

  const deleteModel = async (modelId) => {
    try {
      await axios.delete(`/api/models/${modelId}`)
      await fetchModels()
    } catch (error) {
      console.error('Failed to delete model:', error)
      throw error
    }
  }

  const downloadSafetensorsBundle = async (huggingfaceId, files) => {
    try {
      const response = await axios.post('/api/models/safetensors/download-bundle', {
        huggingface_id: huggingfaceId,
        files
      })
      return response.data
    } catch (error) {
      console.error('Failed to start safetensors bundle download:', error)
      throw error
    }
  }

  const downloadGgufBundle = async (huggingfaceId, quantization, files, pipelineTag = null) => {
    try {
      const response = await axios.post('/api/models/gguf/download-bundle', {
        huggingface_id: huggingfaceId,
        quantization,
        files,
        pipeline_tag: pipelineTag
      })
      return response.data
    } catch (error) {
      console.error('Failed to start GGUF bundle download:', error)
      throw error
    }
  }

  const deleteModelGroup = async (huggingfaceId) => {
    try {
      await axios.post('/api/models/delete-group', { huggingface_id: huggingfaceId })
      await fetchModels()
    } catch (error) {
      console.error('Failed to delete model group:', error)
      throw error
    }
  }

  const deleteSafetensorsModel = async (huggingfaceId) => {
    try {
      await axios.delete('/api/models/safetensors', { data: { huggingface_id: huggingfaceId } })
      await fetchSafetensorsModels()
    } catch (error) {
      console.error('Failed to delete safetensors model:', error)
      throw error
    }
  }

  const fetchHfMetadata = async (modelId) => {
    if (!modelId) return null
    if (hfMetadata.value[modelId]) {
      return hfMetadata.value[modelId]
    }
    hfMetadataLoading.value[modelId] = true
    try {
      const response = await axios.get(`/api/models/${modelId}/hf-metadata`)
      hfMetadata.value[modelId] = response.data || {}
      return hfMetadata.value[modelId]
    } catch (error) {
      console.error('Failed to fetch HF metadata:', error)
      throw error
    } finally {
      hfMetadataLoading.value[modelId] = false
    }
  }

  const fetchLmdeployStatus = async () => {
    lmdeployStatusLoading.value = true
    try {
      const response = await axios.get('/api/models/safetensors/lmdeploy/status')
      lmdeployStatus.value = response.data || null
      return response.data
    } catch (error) {
      console.error('Failed to fetch LMDeploy status:', error)
      throw error
    } finally {
      lmdeployStatusLoading.value = false
    }
  }

  const fetchSafetensorsRuntimeConfig = async (modelId) => {
    if (!modelId) return null
    safetensorsRuntimeLoading.value[modelId] = true
    try {
      const response = await axios.get(`/api/models/safetensors/${modelId}/lmdeploy/config`)
      safetensorsRuntime.value[modelId] = response.data
      if (response.data?.manager) {
        lmdeployStatus.value = {
          ...(lmdeployStatus.value || {}),
          manager: response.data.manager
        }
      }
      return response.data
    } catch (error) {
      console.error('Failed to fetch LMDeploy config:', error)
      throw error
    } finally {
      safetensorsRuntimeLoading.value[modelId] = false
    }
  }

  const regenerateSafetensorsMetadata = async (modelId) => {
    if (!modelId) return
    safetensorsMetadataRefreshing.value[modelId] = true
    try {
      await axios.post(`/api/models/safetensors/${modelId}/metadata/regenerate`)
      await fetchSafetensorsRuntimeConfig(modelId)
      await fetchLmdeployStatus()
      await fetchSafetensorsModels()
    } catch (error) {
      console.error('Failed to regenerate safetensors metadata:', error)
      throw error
    } finally {
      safetensorsMetadataRefreshing.value[modelId] = false
    }
  }

  const updateSafetensorsRuntimeConfig = async (modelId, config) => {
    if (!modelId) return
    try {
      await axios.put(`/api/models/safetensors/${modelId}/lmdeploy/config`, config)
      await fetchSafetensorsRuntimeConfig(modelId)
    } catch (error) {
      console.error('Failed to update LMDeploy config:', error)
      throw error
    }
  }

  const startSafetensorsRuntime = async (modelId, configOverride = null) => {
    if (!modelId) return
    lmdeployStarting.value[modelId] = true
    try {
      const payload = configOverride ? { config: configOverride } : {}
      await axios.post(`/api/models/safetensors/${modelId}/lmdeploy/start`, payload)
      await fetchSafetensorsRuntimeConfig(modelId)
      await fetchLmdeployStatus()
    } catch (error) {
      console.error('Failed to start LMDeploy runtime:', error)
      throw error
    } finally {
      lmdeployStarting.value[modelId] = false
    }
  }

  const stopSafetensorsRuntime = async (modelId) => {
    if (!modelId) return
    lmdeployStopping.value[modelId] = true
    try {
      await axios.post(`/api/models/safetensors/${modelId}/lmdeploy/stop`)
      await fetchLmdeployStatus()
      await fetchSafetensorsRuntimeConfig(modelId)
    } catch (error) {
      console.error('Failed to stop LMDeploy runtime:', error)
      throw error
    } finally {
      lmdeployStopping.value[modelId] = false
    }
  }

  const fetchHuggingfaceTokenStatus = async () => {
    try {
      const response = await axios.get('/api/models/huggingface-token')
      hasHuggingfaceToken.value = response.data.has_token
      huggingfaceToken.value = response.data.token_preview
      tokenFromEnvironment.value = response.data.from_environment
    } catch (error) {
      console.error('Failed to fetch HuggingFace token status:', error)
      throw error
    }
  }

  const setHuggingfaceToken = async (token) => {
    try {
      const response = await axios.post('/api/models/huggingface-token', { token })
      await fetchHuggingfaceTokenStatus()
      return response.data
    } catch (error) {
      console.error('Failed to set HuggingFace token:', error)
      throw error
    }
  }

  const clearHuggingfaceToken = async () => {
    try {
      const response = await axios.post('/api/models/huggingface-token', { token: '' })
      await fetchHuggingfaceTokenStatus()
      return response.data
    } catch (error) {
      console.error('Failed to clear HuggingFace token:', error)
      throw error
    }
  }

  const startModel = async (modelId) => {
    try {
      const response = await axios.post(`/api/models/${modelId}/start`)
      await fetchModels()
      return response.data
    } catch (error) {
      console.error('Failed to start model:', error)
      throw error
    }
  }

  const stopModel = async (modelId) => {
    try {
      await axios.post(`/api/models/${modelId}/stop`)
      await fetchModels()
    } catch (error) {
      console.error('Failed to stop model:', error)
      throw error
    }
  }

  const getModelConfig = async (modelId) => {
    try {
      const response = await axios.get(`/api/models/${modelId}/config`)
      return response.data
    } catch (error) {
      console.error('Failed to get model config:', error)
      throw error
    }
  }

  const updateModelConfig = async (modelId, config) => {
    try {
      await axios.put(`/api/models/${modelId}/config`, config)
    } catch (error) {
      console.error('Failed to update model config:', error)
      throw error
    }
  }

  const generateAutoConfig = async (modelId) => {
    try {
      const response = await axios.post(`/api/models/${modelId}/auto-config`)
      return response.data
    } catch (error) {
      console.error('Failed to generate auto config:', error)
      throw error
    }
  }

  const getModelDetails = async (modelId) => {
    try {
      const response = await axios.get(`/api/models/${modelId}/details`)
      return response.data
    } catch (error) {
      console.error('Failed to get model details:', error)
      throw error
    }
  }

  const getQuantizationSizes = async (huggingfaceId, quantizations) => {
    try {
      const response = await axios.post('/api/models/quantization-sizes', {
        huggingface_id: huggingfaceId,
        quantizations: quantizations
      })
      return response.data.quantizations
    } catch (error) {
      console.error('Failed to get quantization sizes:', error)
      throw error
    }
  }

  const fetchSafetensorsMetadata = async (modelId) => {
    if (!modelId) return null
    if (safetensorsMetadata.value[modelId]) {
      return safetensorsMetadata.value[modelId]
    }
    try {
      safetensorsMetadataLoading.value[modelId] = true
      const encodedId = encodeURIComponent(modelId)
      const response = await axios.get(`/api/models/safetensors/${encodedId}/metadata`)
      safetensorsMetadata.value[modelId] = response.data
      return response.data
    } catch (error) {
      console.error('Failed to fetch safetensors metadata:', error)
      throw error
    } finally {
      safetensorsMetadataLoading.value[modelId] = false
    }
  }

  const updateModelStatus = (modelId, status) => {
    // Find and update the model in the grouped structure
    models.value.forEach(group => {
      const quantization = group.quantizations.find(q => q.id === modelId)
      if (quantization) {
        // Create a new object to trigger reactivity
        Object.assign(quantization, status)
      }
    })
  }

  const updateModelStatusByFilename = (filename, status) => {
    // Find and update the model by filename (for llama-swap matching)
    models.value.forEach(group => {
      group.quantizations.forEach(quantization => {
        if (quantization.filename === filename || quantization.name === filename) {
          Object.assign(quantization, status)
        }
      })
    })
  }

  return {
    models,
    loading,
    searchResults,
    searchLoading,
    searchFormat,
    huggingfaceToken,
    hasHuggingfaceToken,
    tokenFromEnvironment,
    downloadedModels,
    runningModels,
    modelGroups,
    allQuantizations,
    fetchModels,
    searchModels,
    downloadModel,
    deleteModel,
    deleteModelGroup,
    deleteSafetensorsModel,
    fetchHuggingfaceTokenStatus,
    setHuggingfaceToken,
    clearHuggingfaceToken,
    startModel,
    stopModel,
    getModelConfig,
    updateModelConfig,
    generateAutoConfig,
    getModelDetails,
    getQuantizationSizes,
    downloadSafetensorsBundle,
    downloadGgufBundle,
    safetensorsModels,
    safetensorsLoading,
    fetchSafetensorsModels,
    safetensorsMetadata,
    safetensorsMetadataLoading,
    fetchSafetensorsMetadata,
    safetensorsRuntime,
    safetensorsRuntimeLoading,
    safetensorsMetadataRefreshing,
    lmdeployStatus,
    lmdeployStatusLoading,
    lmdeployStarting,
    lmdeployStopping,
    fetchLmdeployStatus,
    fetchSafetensorsRuntimeConfig,
    regenerateSafetensorsMetadata,
    updateSafetensorsRuntimeConfig,
    startSafetensorsRuntime,
    stopSafetensorsRuntime,
    updateModelStatus,
    updateModelStatusByFilename,
    hfMetadata,
    hfMetadataLoading,
    fetchHfMetadata,
    // Loading state tracking
    loadingModels,
    isModelLoading,
    getModelLoadingProgress,
    updateLoadingModels,
    hasLoadingModels
  }
})
