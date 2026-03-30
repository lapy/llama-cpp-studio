import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import axios from 'axios'
import { useEnginesStore } from '@/stores/engines'

function notifySwapConfigStale() {
  try {
    const store = useEnginesStore()
    store.markSwapConfigStaleLocal()
    void store.fetchSwapConfigStale()
  } catch {
    /* Pinia may not be ready in edge cases */
  }
}

/** Encode model id for `/api/models/{id}/…` path segments (handles `/`, `%`, etc.). */
function apiModelSegment(modelId) {
  return encodeURIComponent(String(modelId))
}

export const useModelStore = defineStore('models', () => {
  const models = ref([])        // array of groups: { huggingface_id, base_model_name, quantizations[] }
  const loading = ref(false)
  const searchQuery = ref('')
  const searchLastQuery = ref('')
  const searchHasSearched = ref(false)
  const searchResults = ref([])
  const searchLoading = ref(false)
  const searchFormat = ref('gguf')
  const huggingfaceToken = ref(null)
  const hasHuggingfaceToken = ref(false)
  const tokenFromEnvironment = ref(false)
  const safetensorsModels = ref([])
  const safetensorsLoading = ref(false)

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
    await axios.delete(`/api/models/${apiModelSegment(modelId)}`)
    await fetchModels()
    notifySwapConfigStale()
  }

  async function deleteModelGroup(huggingfaceId) {
    await axios.post('/api/models/delete-group', { huggingface_id: huggingfaceId })
    await fetchModels()
    notifySwapConfigStale()
  }

  async function deleteSafetensorsModel(huggingfaceId) {
    await axios.delete('/api/models/safetensors', { data: { huggingface_id: huggingfaceId } })
    await fetchSafetensorsModels()
    notifySwapConfigStale()
  }

  // ── Search ────────────────────────────────────────────────

  async function searchModels(query, limit = 20, modelFormat = searchFormat.value) {
    searchLoading.value = true
    try {
      const { data } = await axios.post('/api/models/search', { query, limit, model_format: modelFormat })
      searchQuery.value = query
      searchLastQuery.value = query
      searchHasSearched.value = true
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

  function clearSearchState() {
    searchQuery.value = ''
    searchLastQuery.value = ''
    searchHasSearched.value = false
    searchResults.value = []
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

  async function downloadGgufBundle(huggingfaceId, quantization, files, pipelineTag = null, mmprojFilename = null, mmprojSize = 0) {
    const { data } = await axios.post('/api/models/gguf/download-bundle', {
      huggingface_id: huggingfaceId,
      quantization,
      files,
      pipeline_tag: pipelineTag,
      mmproj_filename: mmprojFilename,
      mmproj_size: mmprojSize,
    })
    return data
  }

  // ── Start / Stop ──────────────────────────────────────────

  async function startModel(modelId) {
    const { data } = await axios.post(`/api/models/${apiModelSegment(modelId)}/start`)
    await fetchModels()
    return data
  }

  async function stopModel(modelId) {
    await axios.post(`/api/models/${apiModelSegment(modelId)}/stop`)
    await fetchModels()
  }

  // ── Config ────────────────────────────────────────────────

  async function getModelConfig(modelId) {
    const { data } = await axios.get(`/api/models/${apiModelSegment(modelId)}/config`)
    return data
  }

  async function updateModelConfig(modelId, config) {
    await axios.put(`/api/models/${apiModelSegment(modelId)}/config`, config)
    notifySwapConfigStale()
  }

  async function updateModelProjector(modelId, mmprojFilename = null, totalBytes = 0) {
    const { data } = await axios.post(`/api/models/${apiModelSegment(modelId)}/projector`, {
      mmproj_filename: mmprojFilename,
      total_bytes: totalBytes,
    })
    notifySwapConfigStale()
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
    searchQuery,
    searchLastQuery,
    searchHasSearched,
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

    fetchModels,
    fetchSafetensorsModels,
    deleteModel,
    deleteModelGroup,
    deleteSafetensorsModel,
    searchModels,
    clearSearchState,
    downloadModel,
    downloadSafetensorsBundle,
    downloadGgufBundle,
    startModel,
    stopModel,
    getModelConfig,
    updateModelConfig,
    updateModelProjector,
    fetchHuggingfaceTokenStatus,
    setHuggingfaceToken,
    clearHuggingfaceToken,
    getQuantizationSizes,
  }
})
