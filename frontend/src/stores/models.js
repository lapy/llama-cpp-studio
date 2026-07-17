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
  const catalogResults = ref([])
  const catalogFacets = ref({})
  const catalogProviderStatus = ref({})
  const catalogTotal = ref(0)
  const catalogPage = ref(1)
  const catalogHasMore = ref(false)
  const catalogFilters = ref({})
  const huggingfaceToken = ref(null)
  const hasHuggingfaceToken = ref(false)
  const tokenFromEnvironment = ref(false)
  const safetensorsModels = ref([])
  const safetensorsLoading = ref(false)

  /** Monotonic counter so stale catalog search responses cannot overwrite newer ones. */
  let searchCatalogSeq = 0

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

  async function searchCatalog(query, options = {}) {
    const seq = ++searchCatalogSeq
    searchLoading.value = true
    try {
      const rawPage = Number(options.page)
      const page = Number.isFinite(rawPage) && rawPage >= 1 ? Math.floor(rawPage) : 1
      const rawPageSize = Number(options.page_size ?? options.limit)
      const pageSize = Number.isFinite(rawPageSize) && rawPageSize >= 1
        ? Math.min(100, Math.floor(rawPageSize))
        : 20
      const request = {
        query,
        page,
        page_size: pageSize,
        ...(options.filters || {}),
      }
      if (options.force_refresh) request.force_refresh = true
      const { data } = await axios.post('/api/model-catalog/search', request)
      if (seq !== searchCatalogSeq) return null
      catalogResults.value = Array.isArray(data?.items) ? data.items : []
      catalogFacets.value = data?.facets || {}
      catalogProviderStatus.value = data?.provider_status || {}
      catalogTotal.value = Number(data?.total || 0)
      catalogPage.value = Number(data?.page || 1)
      catalogHasMore.value = Boolean(data?.has_more)
      catalogFilters.value = options.filters || {}
      searchQuery.value = query
      searchLastQuery.value = query
      searchHasSearched.value = true
      searchResults.value = catalogResults.value
      return data
    } catch (e) {
      if (seq === searchCatalogSeq) {
        console.error('Failed to search normalized model catalog:', e)
        catalogResults.value = []
        searchResults.value = []
      }
      throw e
    } finally {
      if (seq === searchCatalogSeq) {
        searchLoading.value = false
      }
    }
  }

  async function installCatalogModel(result, variant, options = {}) {
    const { data } = await axios.post('/api/model-catalog/install', {
      catalog_id: result.id,
      provider: result.provider,
      provider_item_id: result.provider_item_id,
      variant_id: variant.id,
      install_method: variant.method,
      source: result.source,
      variant,
      ...options,
    })
    return data
  }

  async function importAudioBundle(sourcePath, options = {}) {
    const { data } = await axios.post('/api/model-catalog/import', {
      source_path: sourcePath,
      ...options,
    })
    return data
  }

  function clearSearchState() {
    searchQuery.value = ''
    searchLastQuery.value = ''
    searchHasSearched.value = false
    searchResults.value = []
    catalogResults.value = []
    catalogFacets.value = {}
    catalogProviderStatus.value = {}
    catalogTotal.value = 0
    catalogPage.value = 1
    catalogHasMore.value = false
    catalogFilters.value = {}
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

  async function downloadGgufBundle(
    huggingfaceId,
    quantization,
    files,
    pipelineTag = null,
    mmprojFilename = null,
    mmprojSize = 0,
    mtpFilename = null,
    mtpSize = 0,
  ) {
    const { data } = await axios.post('/api/models/gguf/download-bundle', {
      huggingface_id: huggingfaceId,
      quantization,
      files,
      pipeline_tag: pipelineTag,
      mmproj_filename: mmprojFilename,
      mmproj_size: mmprojSize,
      mtp_filename: mtpFilename,
      mtp_size: mtpSize,
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

  async function listReferenceAudio(modelId) {
    const { data } = await axios.get(`/api/models/${apiModelSegment(modelId)}/reference-audio`)
    return Array.isArray(data?.items) ? data.items : []
  }

  async function uploadReferenceAudio(modelId, file) {
    const form = new FormData()
    form.append('file', file)
    const { data } = await axios.post(
      `/api/models/${apiModelSegment(modelId)}/reference-audio`,
      form,
      { headers: { 'Content-Type': 'multipart/form-data' } },
    )
    notifySwapConfigStale()
    return data
  }

  async function deleteReferenceAudio(modelId, filename) {
    await axios.delete(
      `/api/models/${apiModelSegment(modelId)}/reference-audio/${encodeURIComponent(filename)}`,
    )
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

  async function updateModelMtp(modelId, mtpFilename = null, totalBytes = 0) {
    const { data } = await axios.post(`/api/models/${apiModelSegment(modelId)}/mtp`, {
      mtp_filename: mtpFilename,
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
    catalogResults,
    catalogFacets,
    catalogProviderStatus,
    catalogTotal,
    catalogPage,
    catalogHasMore,
    catalogFilters,
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
    searchCatalog,
    installCatalogModel,
    importAudioBundle,
    clearSearchState,
    downloadModel,
    downloadSafetensorsBundle,
    downloadGgufBundle,
    startModel,
    stopModel,
    getModelConfig,
    updateModelConfig,
    listReferenceAudio,
    uploadReferenceAudio,
    deleteReferenceAudio,
    updateModelProjector,
    updateModelMtp,
    fetchHuggingfaceTokenStatus,
    setHuggingfaceToken,
    clearHuggingfaceToken,
    getQuantizationSizes,
  }
})
