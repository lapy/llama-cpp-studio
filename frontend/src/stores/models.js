import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import axios from 'axios'

export const useModelStore = defineStore('models', () => {
  const models = ref([])        // array of groups: { huggingface_id, base_model_name, quantizations[] }
  const loading = ref(false)
  const searchResults = ref([])
  const searchLoading = ref(false)
  const searchFormat = ref('gguf')
  const huggingfaceToken = ref(null)
  const hasHuggingfaceToken = ref(false)
  const tokenFromEnvironment = ref(false)
  const safetensorsModels = ref([])
  const safetensorsLoading = ref(false)
  const safetensorsMetadata = ref({})
  const safetensorsMetadataLoading = ref({})
  const hfMetadata = ref({})
  const hfMetadataLoading = ref({})

  // ── Computed ──────────────────────────────────────────────

  const allQuantizations = computed(() => {
    const result = []
    models.value.forEach(group => {
      ;(group.quantizations || []).forEach(q => {
        result.push({
          ...q,
          base_model_name: group.base_model_name,
          huggingface_id: group.huggingface_id,
          model_type: group.model_type,
          pipeline_tag: q.pipeline_tag || group.pipeline_tag,
          is_embedding_model: q.is_embedding_model ?? group.is_embedding_model ?? false,
        })
      })
    })
    return result
  })

  const downloadedModels = computed(() => allQuantizations.value.filter(m => m.downloaded_at))
  const runningModels    = computed(() => allQuantizations.value.filter(m => m.is_active))
  const modelGroups      = computed(() => models.value)

  // ── Models CRUD ───────────────────────────────────────────

  async function fetchModels() {
    loading.value = true
    try {
      const { data } = await axios.get('/api/models')
      models.value = data
    } catch (e) {
      console.error('Failed to fetch models:', e)
      throw e
    } finally {
      loading.value = false
    }
  }

  async function fetchSafetensorsModels() {
    safetensorsLoading.value = true
    try {
      const { data } = await axios.get('/api/models/safetensors')
      safetensorsModels.value = Array.isArray(data) ? data : []
    } catch (e) {
      console.error('Failed to fetch safetensors models:', e)
      throw e
    } finally {
      safetensorsLoading.value = false
    }
  }

  async function deleteModel(modelId) {
    await axios.delete(`/api/models/${modelId}`)
    await fetchModels()
  }

  async function deleteModelGroup(huggingfaceId) {
    await axios.post('/api/models/delete-group', { huggingface_id: huggingfaceId })
    await fetchModels()
  }

  async function deleteSafetensorsModel(huggingfaceId) {
    await axios.delete('/api/models/safetensors', { data: { huggingface_id: huggingfaceId } })
    await fetchSafetensorsModels()
  }

  // ── Search ────────────────────────────────────────────────

  async function searchModels(query, limit = 20, modelFormat = searchFormat.value) {
    searchLoading.value = true
    try {
      const { data } = await axios.post('/api/models/search', { query, limit, model_format: modelFormat })
      searchResults.value = Array.isArray(data) ? data : []
      searchFormat.value = modelFormat
      return searchResults.value
    } catch (e) {
      console.error('Failed to search models:', e)
      searchResults.value = []
      throw e
    } finally {
      searchLoading.value = false
    }
  }

  // ── Download ──────────────────────────────────────────────

  async function downloadModel(huggingfaceId, filename, totalBytes = 0, modelFormat = 'gguf', pipelineTag = null) {
    const { data } = await axios.post('/api/models/download', {
      huggingface_id: huggingfaceId,
      filename,
      total_bytes: totalBytes,
      model_format: modelFormat,
      pipeline_tag: pipelineTag,
    })
    await fetchModels()
    return data
  }

  async function downloadSafetensorsBundle(huggingfaceId, files) {
    const { data } = await axios.post('/api/models/safetensors/download-bundle', {
      huggingface_id: huggingfaceId,
      files,
    })
    return data
  }

  async function downloadGgufBundle(huggingfaceId, quantization, files, pipelineTag = null) {
    const { data } = await axios.post('/api/models/gguf/download-bundle', {
      huggingface_id: huggingfaceId,
      quantization,
      files,
      pipeline_tag: pipelineTag,
    })
    return data
  }

  // ── Start / Stop ──────────────────────────────────────────

  async function startModel(modelId) {
    const { data } = await axios.post(`/api/models/${modelId}/start`)
    await fetchModels()
    return data
  }

  async function stopModel(modelId) {
    await axios.post(`/api/models/${modelId}/stop`)
    await fetchModels()
  }

  // ── Config ────────────────────────────────────────────────

  async function getModelConfig(modelId) {
    const { data } = await axios.get(`/api/models/${modelId}/config`)
    return data
  }

  async function updateModelConfig(modelId, config) {
    await axios.put(`/api/models/${modelId}/config`, config)
  }

  async function getModelDetails(modelId) {
    const { data } = await axios.get(`/api/models/${modelId}/details`)
    return data
  }

  // ── HuggingFace Token ─────────────────────────────────────

  async function fetchHuggingfaceTokenStatus() {
    const { data } = await axios.get('/api/models/huggingface-token')
    hasHuggingfaceToken.value = data.has_token
    huggingfaceToken.value = data.token_preview
    tokenFromEnvironment.value = data.from_environment
  }

  async function setHuggingfaceToken(token) {
    const { data } = await axios.post('/api/models/huggingface-token', { token })
    await fetchHuggingfaceTokenStatus()
    return data
  }

  async function clearHuggingfaceToken() {
    const { data } = await axios.post('/api/models/huggingface-token', { token: '' })
    await fetchHuggingfaceTokenStatus()
    return data
  }

  // ── Metadata ──────────────────────────────────────────────

  async function fetchHfMetadata(modelId) {
    if (!modelId) return null
    if (hfMetadata.value[modelId]) return hfMetadata.value[modelId]
    hfMetadataLoading.value[modelId] = true
    try {
      const { data } = await axios.get(`/api/models/${modelId}/hf-metadata`)
      hfMetadata.value[modelId] = data || {}
      return hfMetadata.value[modelId]
    } finally {
      hfMetadataLoading.value[modelId] = false
    }
  }

  async function fetchSafetensorsMetadata(modelId) {
    if (!modelId) return null
    if (safetensorsMetadata.value[modelId]) return safetensorsMetadata.value[modelId]
    safetensorsMetadataLoading.value[modelId] = true
    try {
      const encoded = encodeURIComponent(modelId)
      const { data } = await axios.get(`/api/models/safetensors/${encoded}/metadata`)
      safetensorsMetadata.value[modelId] = data
      return data
    } finally {
      safetensorsMetadataLoading.value[modelId] = false
    }
  }

  async function getQuantizationSizes(huggingfaceId, quantizations) {
    const { data } = await axios.post('/api/models/quantization-sizes', {
      huggingface_id: huggingfaceId,
      quantizations,
    })
    return data.quantizations
  }

  // ── Return ────────────────────────────────────────────────

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
    safetensorsModels,
    safetensorsLoading,
    safetensorsMetadata,
    safetensorsMetadataLoading,
    hfMetadata,
    hfMetadataLoading,

    fetchModels,
    fetchSafetensorsModels,
    deleteModel,
    deleteModelGroup,
    deleteSafetensorsModel,
    searchModels,
    downloadModel,
    downloadSafetensorsBundle,
    downloadGgufBundle,
    startModel,
    stopModel,
    getModelConfig,
    updateModelConfig,
    getModelDetails,
    fetchHuggingfaceTokenStatus,
    setHuggingfaceToken,
    clearHuggingfaceToken,
    fetchHfMetadata,
    fetchSafetensorsMetadata,
    getQuantizationSizes,
  }
})
