import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import axios from 'axios'

export const useModelStore = defineStore('models', () => {
  const models = ref([]) // This will now contain grouped models
  const loading = ref(false)
  const searchResults = ref([])
  const searchLoading = ref(false)
  const huggingfaceToken = ref(null)
  const hasHuggingfaceToken = ref(false)
  const tokenFromEnvironment = ref(false)

  // Flatten all quantizations for backward compatibility
  const allQuantizations = computed(() => {
    const quantizations = []
    models.value.forEach(group => {
      group.quantizations.forEach(quant => {
        quantizations.push({
          ...quant,
          base_model_name: group.base_model_name,
          huggingface_id: group.huggingface_id,
          model_type: group.model_type
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

  const searchModels = async (query, limit = 20) => {
    searchLoading.value = true
    try {
      const response = await axios.post('/api/models/search', { query, limit })
      // Ensure searchResults is always an array
      searchResults.value = Array.isArray(response.data) ? response.data : []
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

const downloadModel = async (huggingfaceId, filename, totalBytes = 0) => {
  try {
    const response = await axios.post('/api/models/download', {
      huggingface_id: huggingfaceId,
      filename,
      total_bytes: totalBytes
    })
    // Refresh models list after download starts
    await fetchModels()
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

  const deleteModelGroup = async (huggingfaceId) => {
    try {
      await axios.post('/api/models/delete-group', { huggingface_id: huggingfaceId })
      await fetchModels()
    } catch (error) {
      console.error('Failed to delete model group:', error)
      throw error
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
    updateModelStatus,
    updateModelStatusByFilename
  }
})
