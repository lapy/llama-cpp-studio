<template>
  <div class="model-search">

    <!-- Search bar -->
    <div class="search-bar">
      <div class="search-input-wrap">
        <i class="pi pi-search search-icon" />
        <InputText
          v-model="query"
          placeholder="Search HuggingFace models…"
          class="search-input"
          @keyup.enter="search"
        />
        <Button
          v-if="query"
          icon="pi pi-times"
          text
          severity="secondary"
          class="clear-btn"
          @click="clearSearchResults"
        />
      </div>

      <Dropdown
        v-model="searchFormat"
        :options="formatOptions"
        optionLabel="label"
        optionValue="value"
        class="format-select"
      />

      <Button
        label="Search"
        icon="pi pi-search"
        severity="success"
        :loading="searching"
        :disabled="!query.trim()"
        @click="search"
      />
    </div>

    <!-- Download progress -->
    <ProgressTracker type="download" :show-completed="true" />

    <!-- Results loading -->
    <div v-if="searching" class="loading-row">
      <ProgressSpinner style="width:40px;height:40px" strokeWidth="4" />
      <span>Searching…</span>
    </div>

    <!-- Empty search state -->
    <div v-else-if="!searchResults.length && hasSearched" class="empty-state">
      <i class="pi pi-search" style="font-size:3rem;color:var(--text-secondary)" />
      <h3>No results for "{{ lastQuery }}"</h3>
      <p>Try different keywords or change the format filter.</p>
    </div>

    <!-- Initial prompt -->
    <div v-else-if="!searchResults.length && !hasSearched" class="empty-state">
      <i class="pi pi-search" style="font-size:3rem;color:var(--text-secondary)" />
      <h3>Search for models</h3>
      <p>Enter a model name or keyword above to find models on HuggingFace.</p>
    </div>

    <!-- Results table -->
    <div v-else class="results-table">
      <div class="results-header">
        <span class="results-count">{{ searchResults.length }} result{{ searchResults.length !== 1 ? 's' : '' }}</span>
      </div>

      <div
        v-for="result in searchResults"
        :key="result.modelId || result.id"
        class="result-row"
      >
        <!-- Main row -->
        <div class="result-main" @click="toggleExpand(result.modelId || result.id)">
          <div class="result-expand-icon">
            <i :class="['pi', expanded.has(result.modelId || result.id) ? 'pi-chevron-down' : 'pi-chevron-right']" />
          </div>

          <div class="result-info">
            <div class="result-name">
              <a
                :href="`https://huggingface.co/${result.modelId || result.id}`"
                target="_blank"
                class="model-link"
                @click.stop
              >
                {{ result.modelId || result.id }}
              </a>
              <Tag v-if="result.pipeline_tag" :value="result.pipeline_tag" severity="secondary" class="pipeline-tag" />
            </div>
            <div class="result-meta">
              <span v-if="result.author" class="meta-item">
                <i class="pi pi-user" /> {{ result.author }}
              </span>
              <span v-if="result.downloads != null" class="meta-item">
                <i class="pi pi-download" /> {{ formatNumber(result.downloads) }}
              </span>
              <span v-if="result.likes != null" class="meta-item">
                <i class="pi pi-heart" /> {{ formatNumber(result.likes) }}
              </span>
              <span v-if="getResultArtifactCount(result)" class="meta-item">
                <i class="pi pi-database" /> {{ getResultArtifactCount(result) }}
              </span>
              <span v-if="getResultSizeSummary(result)" class="meta-item">
                <i class="pi pi-box" /> {{ getResultSizeSummary(result) }}
              </span>
              <span v-if="result.license" class="meta-item license">
                {{ result.license }}
              </span>
            </div>
          </div>

          <div class="result-tags">
            <Tag
              v-for="tag in (result.tags || []).filter(t => interestingTag(t)).slice(0, 3)"
              :key="tag"
              :value="tag"
              severity="info"
              class="model-tag"
            />
          </div>
        </div>

        <!-- Expanded: quantization files -->
        <Transition name="row-expand">
          <div v-if="expanded.has(result.modelId || result.id)" class="result-files">
            <div v-if="loadingFiles.has(result.modelId || result.id)" class="files-loading">
              <ProgressSpinner style="width:24px;height:24px" strokeWidth="4" />
              <span>Loading files…</span>
            </div>

            <div v-else-if="getFiles(result.modelId || result.id).length === 0" class="files-empty">
              <span>No downloadable files found for the selected format.</span>
            </div>

            <table v-else class="files-table">
              <thead>
                <tr>
                  <th>{{ searchFormat === 'gguf' ? 'Item' : 'Model' }}</th>
                  <th>Size</th>
                  <th v-if="searchFormat === 'gguf'">Shards</th>
                  <th v-if="searchFormat === 'gguf'">Projector</th>
                  <th>Status</th>
                  <th></th>
                </tr>
              </thead>
              <tbody>
                <tr v-for="file in getFiles(result.modelId || result.id)" :key="file.key || file.filename">
                  <td class="file-name">
                    <code>{{ formatResultItemLabel(file, result) }}</code>
                    <span v-if="searchFormat === 'gguf' && file.kind === 'quant' && file.variantPrefix" class="file-subtext">
                      {{ file.variantPrefix }} variant
                    </span>
                    <span v-else-if="file.subtext" class="file-subtext">
                      {{ file.subtext }}
                    </span>
                  </td>
                  <td class="file-size">{{ formatBytes(file.size) }}</td>
                  <td v-if="searchFormat === 'gguf'" class="file-count">
                    {{ file.kind === 'quant' ? (file.files?.length || 0) : 1 }}
                  </td>
                  <td v-if="searchFormat === 'gguf'" class="projector-cell">
                    <Dropdown
                      v-if="file.kind === 'quant'"
                      :model-value="getSelectedProjector(result.modelId || result.id, file)"
                      :options="file.projectorOptions || [{ label: 'None', value: '' }]"
                      optionLabel="label"
                      optionValue="value"
                      class="projector-select"
                      :disabled="downloadingFiles.has(getDownloadKey(result.modelId || result.id, file))"
                      @update:model-value="setSelectedProjector(result.modelId || result.id, file, $event)"
                    />
                  </td>
                  <td class="file-status">
                    <Tag
                      v-if="isFileDownloading(result.modelId || result.id, file)"
                      value="Downloading"
                      severity="warning"
                    />
                    <Tag v-else-if="file.downloaded" value="Downloaded" severity="success" />
                    <Tag v-else value="Available" severity="warning" />
                  </td>
                  <td class="file-action">
                    <div class="file-actions">
                      <Button
                        v-if="file.downloaded"
                        label="Configure"
                        icon="pi pi-cog"
                        size="small"
                        severity="secondary"
                        text
                        @click="configureDownloaded(result.modelId || result.id, file)"
                      />
                      <Button
                        v-if="file.downloaded && searchFormat === 'gguf' && file.kind === 'quant' && hasProjectorSelectionChanged(result.modelId || result.id, file)"
                        label="Apply projector"
                        icon="pi pi-save"
                        size="small"
                        severity="success"
                        outlined
                        :loading="downloadingFiles.has(getDownloadKey(result.modelId || result.id, file))"
                        @click="updateProjector(result, file)"
                      />
                      <Button
                        v-if="!file.downloaded"
                        label="Download"
                        icon="pi pi-download"
                        size="small"
                        severity="success"
                        outlined
                        :loading="downloadingFiles.has(getDownloadKey(result.modelId || result.id, file))"
                        @click="downloadFile(result, file)"
                      />
                    </div>
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
        </Transition>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, onUnmounted } from 'vue'
import { storeToRefs } from 'pinia'
import { useRouter } from 'vue-router'
import { useToast } from 'primevue/usetoast'
import Button from 'primevue/button'
import Tag from 'primevue/tag'
import InputText from 'primevue/inputtext'
import Dropdown from 'primevue/dropdown'
import ProgressSpinner from 'primevue/progressspinner'
import ProgressTracker from '@/components/common/ProgressTracker.vue'
import { useModelStore } from '@/stores/models'
import { useProgressStore } from '@/stores/progress'
import axios from 'axios'

const router = useRouter()
const toast = useToast()
const modelStore = useModelStore()
const progressStore = useProgressStore()
const {
  searchQuery: query,
  searchLastQuery: lastQuery,
  searchHasSearched: hasSearched,
  searchResults,
  searchLoading: searching,
  searchFormat,
} = storeToRefs(modelStore)

// ── State ──────────────────────────────────────────────────
const expanded = ref(new Set())
const loadingFiles = ref(new Set())
const downloadingFiles = ref(new Set())
const filesCache = ref({})   // modelId -> files[]
const projectorSelections = ref({})

const formatOptions = [
  { label: 'GGUF', value: 'gguf' },
  { label: 'Safetensors', value: 'safetensors' },
]

// ── Search ─────────────────────────────────────────────────
async function search() {
  if (!query.value.trim()) return
  expanded.value = new Set()
  filesCache.value = {}
  projectorSelections.value = {}
  try {
    searchResults.value = await modelStore.searchModels(query.value.trim(), 20, searchFormat.value)
  } catch (e) {
    toast.add({ severity: 'error', summary: 'Search failed', detail: e.message, life: 4000 })
    searchResults.value = []
  }
}

function clearSearchResults() {
  modelStore.clearSearchState()
  expanded.value = new Set()
  filesCache.value = {}
  projectorSelections.value = {}
}

// ── Expand row & load files ────────────────────────────────
async function toggleExpand(modelId) {
  if (expanded.value.has(modelId)) {
    expanded.value.delete(modelId)
    expanded.value = new Set(expanded.value)
    return
  }
  expanded.value.add(modelId)
  expanded.value = new Set(expanded.value)
  if (!filesCache.value[modelId]) {
    await loadFiles(modelId)
  }
}

async function loadFiles(modelId) {
  loadingFiles.value.add(modelId)
  loadingFiles.value = new Set(loadingFiles.value)
  try {
    const result = searchResults.value.find(r => (r.modelId || r.id) === modelId)
    if (!result) return

    let files = []
    if (searchFormat.value === 'gguf') {
      const projectorOptions = getProjectorOptions(result.mmproj_files || [])
      const quantEntries = Object.entries(result.quantizations || {}).map(([key, entry]) => ({
        key,
        kind: 'quant',
        quantizationKey: key,
        quantization: entry.quantization || '',
        variantPrefix: entry.variant_prefix || '',
        size: entry.total_size || 0,
        projectorOptions,
        files: (entry.files || []).map(f => ({
          filename: f.filename,
          size: f.size || 0,
        })),
      }))

      const allFiles = quantEntries.flatMap(entry => entry.files)
      if (allFiles.length) {
        try {
          const filenames = allFiles.map(f => f.filename).join(',')
          const { data } = await axios.get(`/api/models/search/${encodeURIComponent(modelId)}/file-sizes`, {
            params: { filenames },
          })
          const sizes = data.sizes || {}
          files = quantEntries.map(entry => {
            const resolvedFiles = entry.files.map(f => ({
              ...f,
              size: sizes[f.filename] ?? f.size,
            }))
            const downloaded = findDownloadedQuantization(modelId, entry, resolvedFiles)
            return {
              ...entry,
              files: resolvedFiles,
              size: resolvedFiles.reduce((sum, f) => sum + (f.size || 0), 0),
              downloaded,
              modelId: downloaded?.id,
            }
          }).sort((a, b) => (a.size || 0) - (b.size || 0))
        } catch {
          files = quantEntries.map(entry => {
            const downloaded = findDownloadedQuantization(modelId, entry, entry.files)
            return { ...entry, downloaded, modelId: downloaded?.id }
          }).sort((a, b) => (a.size || 0) - (b.size || 0))
        }
      }
      files.forEach((entry) => {
        if (entry.kind !== 'quant') return
        ensureProjectorSelection(modelId, entry, entry.downloaded?.mmproj_filename || '')
      })
    } else {
      const stFiles = result.safetensors_files || []
      let resolvedFiles = stFiles.map(file => ({ filename: file.filename, size: file.size || 0 }))
      if (resolvedFiles.length) {
        try {
          const filenames = resolvedFiles.map(file => file.filename).join(',')
          const { data } = await axios.get(`/api/models/search/${encodeURIComponent(modelId)}/file-sizes`, {
            params: { filenames },
          })
          const sizes = data.sizes || {}
          resolvedFiles = resolvedFiles.map(file => ({
            ...file,
            size: sizes[file.filename] ?? file.size,
          }))
        } catch {
          // Keep the size hints returned by the search API.
        }
      }

      const downloadedBundle = findDownloadedSafetensorsBundle(result.modelId || result.id)
      const totalSize = resolvedFiles.reduce((sum, file) => sum + (file.size || 0), 0)
      files = resolvedFiles.length
        ? [{
            key: 'safetensors-bundle',
            kind: 'safetensors-bundle',
            filename: result.modelId || result.id,
            size: totalSize,
            files: resolvedFiles,
            downloaded: downloadedBundle,
            modelId: downloadedBundle?.model_id,
            subtext: `${resolvedFiles.length} file${resolvedFiles.length === 1 ? '' : 's'}`,
          }]
        : []
    }

    filesCache.value[modelId] = files
  } catch (e) {
    filesCache.value[modelId] = []
    console.error('Failed to load files:', e)
  } finally {
    loadingFiles.value.delete(modelId)
    loadingFiles.value = new Set(loadingFiles.value)
  }
}

function getFiles(modelId) {
  return filesCache.value[modelId] || []
}

// ── Download ───────────────────────────────────────────────
async function downloadFile(result, file) {
  const modelId = result.modelId || result.id
  const key = getDownloadKey(modelId, file)
  downloadingFiles.value.add(key)
  downloadingFiles.value = new Set(downloadingFiles.value)
  try {
    if (searchFormat.value === 'gguf' && file.kind === 'quant') {
      const selectedProjector = getSelectedProjector(modelId, file)
      const selectedProjectorOption = getSelectedProjectorOption(file, selectedProjector)
      await modelStore.downloadGgufBundle(
        modelId,
        file.quantizationKey || file.quantization,
        file.files || [],
        result.pipeline_tag || null,
        selectedProjector || null,
        selectedProjectorOption?.size || 0,
      )
    } else if (searchFormat.value === 'safetensors') {
      await modelStore.downloadSafetensorsBundle(
        modelId,
        file.files || []
      )
    } else {
      await modelStore.downloadModel(
        modelId,
        file.filename,
        file.size || 0,
        searchFormat.value,
        result.pipeline_tag || null
      )
    }
    toast.add({ severity: 'success', summary: 'Download started', detail: 'Track progress above', life: 3000 })
    // Refresh files to update downloaded status
    delete filesCache.value[modelId]
    await loadFiles(modelId)
    await modelStore.fetchModels()
    if (searchFormat.value === 'safetensors') {
      await modelStore.fetchSafetensorsModels()
    }
  } catch (e) {
    toast.add({ severity: 'error', summary: 'Download failed', detail: e.message, life: 4000 })
  } finally {
    downloadingFiles.value.delete(key)
    downloadingFiles.value = new Set(downloadingFiles.value)
  }
}

async function updateProjector(result, file) {
  const repoId = result.modelId || result.id
  const downloadKey = getDownloadKey(repoId, file)
  const model = file.modelId
    ? modelStore.allQuantizations.find(m => m.id === file.modelId)
    : findDownloadedQuantization(repoId, file, file.files || [])
  if (!model?.id) return

  const selectedProjector = getSelectedProjector(repoId, file) || null
  const selectedProjectorOption = getSelectedProjectorOption(file, selectedProjector)

  downloadingFiles.value.add(downloadKey)
  downloadingFiles.value = new Set(downloadingFiles.value)
  try {
    const response = await modelStore.updateModelProjector(
      model.id,
      selectedProjector,
      selectedProjectorOption?.size || 0,
    )
    if (response?.applied) {
      await refreshModelSearchState()
      toast.add({ severity: 'success', summary: 'Projector updated', detail: response.message, life: 3000 })
    } else {
      toast.add({ severity: 'success', summary: 'Projector update started', detail: response?.message || 'Track progress above', life: 3000 })
    }
  } catch (e) {
    toast.add({ severity: 'error', summary: 'Projector update failed', detail: e.message, life: 4000 })
  } finally {
    downloadingFiles.value.delete(downloadKey)
    downloadingFiles.value = new Set(downloadingFiles.value)
  }
}


function configureDownloaded(modelId, file) {
  const model = searchFormat.value === 'safetensors'
    ? findDownloadedSafetensorsBundle(modelId)
    : file.modelId
      ? modelStore.allQuantizations.find(m => m.id === file.modelId)
      : file.kind === 'quant'
        ? findDownloadedQuantization(modelId, file, file.files || [])
        : findDownloadedModel(modelId, file.filename)
  if (model) {
    router.push(`/models/${encodeURIComponent(model.id || model.model_id)}/config`)
  }
}

// ── Helpers ────────────────────────────────────────────────
function getDownloadKey(modelId, file) {
  return `${modelId}:${file.quantizationKey || file.filename}`
}

function isFileDownloading(modelId, file) {
  return downloadingFiles.value.has(getDownloadKey(modelId, file))
}

function getProjectorSelectionKey(modelId, file) {
  return `${modelId}:${file.quantizationKey || file.filename}:projector`
}

function parseProjectorPrecision(filename) {
  const upper = (filename || '').toUpperCase()
  if (upper.includes('BF16')) return 'BF16'
  if (upper.includes('F16')) return 'F16'
  if (upper.includes('F32')) return 'F32'
  return null
}

function getProjectorOptions(mmprojFiles = []) {
  const byPrecision = new Map()
  mmprojFiles.forEach((file) => {
    const precision = parseProjectorPrecision(file.filename)
    if (!precision || byPrecision.has(precision)) return
    byPrecision.set(precision, {
      label: precision,
      value: file.filename,
      size: file.size || 0,
    })
  })

  return [
    { label: 'None', value: '', size: 0 },
    ...Array.from(byPrecision.values()).sort((a, b) => a.label.localeCompare(b.label)),
  ]
}

function ensureProjectorSelection(modelId, file, value = '') {
  const key = getProjectorSelectionKey(modelId, file)
  if (Object.prototype.hasOwnProperty.call(projectorSelections.value, key)) return
  const defaultValue = value || getDefaultProjectorValue(file)
  projectorSelections.value = {
    ...projectorSelections.value,
    [key]: defaultValue,
  }
}

function setSelectedProjector(modelId, file, value) {
  projectorSelections.value = {
    ...projectorSelections.value,
    [getProjectorSelectionKey(modelId, file)]: value || '',
  }
}

function getSelectedProjector(modelId, file) {
  const key = getProjectorSelectionKey(modelId, file)
  if (Object.prototype.hasOwnProperty.call(projectorSelections.value, key)) {
    return projectorSelections.value[key]
  }
  return file.downloaded?.mmproj_filename || ''
}

function getSelectedProjectorOption(file, value) {
  return (file.projectorOptions || []).find(option => option.value === (value || '')) || null
}

function getDefaultProjectorValue(file) {
  const f16 = (file.projectorOptions || []).find(option => option.label === 'F16')
  return f16?.value || ''
}

function hasProjectorSelectionChanged(modelId, file) {
  return (getSelectedProjector(modelId, file) || '') !== (file.downloaded?.mmproj_filename || '')
}

function isDownloaded(hfId, filename) {
  return modelStore.allQuantizations.find(
    m => m.huggingface_id === hfId &&
    (m.filename === filename || (m.quantization && filename.includes(m.quantization)))
  )
}

function findDownloadedQuantization(hfId, entry, files = []) {
  return modelStore.allQuantizations.find(m =>
    m.huggingface_id === hfId &&
    (
      (entry.quantization && m.quantization === entry.quantization) ||
      files.some(file => m.filename === file.filename)
    )
  )
}

function findDownloadedSafetensorsBundle(hfId) {
  return modelStore.safetensorsModels.find(model => model.huggingface_id === hfId)
}

function findDownloadedModel(hfId, filename) {
  return modelStore.allQuantizations.find(
    m => m.huggingface_id === hfId &&
    (m.filename === filename || (m.quantization && filename.includes(m.quantization)))
  )
}

function formatResultItemLabel(entry, result) {
  if (!entry) return ''
  if (entry.kind === 'quant') {
    return entry.quantizationKey || entry.quantization || 'Unknown'
  }
  return result.modelId || result.id
}

const INTERESTING_TAGS = new Set([
  'text-generation', 'chat', 'instruct', 'code', 'embedding', 'vision',
  'multimodal', 'image-text-to-text', 'fill-mask', 'question-answering',
])

function interestingTag(tag) {
  return INTERESTING_TAGS.has(tag) || tag.startsWith('language:') || /^\d+[bBmM]$/.test(tag)
}

// Use decimal (1000) so MB/GB match Hugging Face's display
function formatBytes(bytes) {
  if (!bytes) return '—'
  const units = ['B', 'KB', 'MB', 'GB', 'TB']
  let i = 0; let val = bytes
  while (val >= 1000 && i < units.length - 1) { val /= 1000; i++ }
  return `${val.toFixed(1)} ${units[i]}`
}

function formatNumber(n) {
  if (n == null) return ''
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}K`
  return String(n)
}

function getResultArtifactCount(result) {
  if ((result.format || searchFormat.value) === 'gguf') {
    const quantCount = Object.keys(result.quantizations || {}).length
    return quantCount ? `${quantCount} quant${quantCount === 1 ? '' : 's'}` : ''
  }
  const fileCount = (result.safetensors_files || []).length
  return fileCount ? `${fileCount} file${fileCount === 1 ? '' : 's'}` : ''
}

function getResultSizeSummary(result) {
  if ((result.format || searchFormat.value) === 'gguf') {
    const sizes = Object.values(result.quantizations || {})
      .map(entry => entry?.total_size || 0)
      .filter(size => size > 0)
    if (!sizes.length) return ''
    return `from ${formatBytes(Math.min(...sizes))}`
  }

  const totalSize = (result.safetensors_files || [])
    .reduce((sum, file) => sum + (file.size || 0), 0)
  return totalSize > 0 ? formatBytes(totalSize) : ''
}

// ── Lifecycle ──────────────────────────────────────────────
function markDownloadedFromEvent(payload) {
  const hfId = payload?.huggingface_id
  const quantization = payload?.quantization
  if (!hfId || !quantization) return

  const cachedRows = filesCache.value[hfId]
  if (!Array.isArray(cachedRows) || cachedRows.length === 0) return

  const nextRows = cachedRows.map((row) => {
    if (row.kind !== 'quant') return row

    const matchesQuantization = row.quantizationKey === quantization || row.quantization === quantization
    const matchesFilename = Array.isArray(payload?.filenames)
      && payload.filenames.some(filename => (row.files || []).some(file => file.filename === filename))

    if (!matchesQuantization && !matchesFilename) return row

    const downloaded = {
      ...(row.downloaded || {}),
      id: payload?.model_id || row.modelId || row.downloaded?.id,
      mmproj_filename: payload?.mmproj_filename || row.downloaded?.mmproj_filename || '',
    }

    const updatedRow = {
      ...row,
      downloaded,
      modelId: downloaded.id,
    }
    // Keep the projector selector in sync with backend state for this quant.
    const key = getProjectorSelectionKey(hfId, updatedRow)
    projectorSelections.value = {
      ...projectorSelections.value,
      [key]: downloaded.mmproj_filename || '',
    }
    return updatedRow
  })

  filesCache.value = {
    ...filesCache.value,
    [hfId]: nextRows,
  }
}

async function refreshModelSearchState() {
  await modelStore.fetchModels()
  await modelStore.fetchSafetensorsModels()
  const expandedIds = Array.from(expanded.value)
  filesCache.value = {}
  await Promise.all(expandedIds.map(id => loadFiles(id)))
}

let unsubscribeDownloadComplete = null

onMounted(async () => {
  if (!modelStore.models.length) await modelStore.fetchModels()
  if (!modelStore.safetensorsModels.length) await modelStore.fetchSafetensorsModels()
  unsubscribeDownloadComplete = progressStore.subscribeToDownloadComplete(async (payload) => {
    const hfId = payload?.huggingface_id
    if (!hfId) return
    if (!searchResults.value.some(result => (result.modelId || result.id) === hfId)) return
    if (payload?.model_format === 'gguf-bundle') {
      markDownloadedFromEvent(payload)
    }
    await refreshModelSearchState()
  })
})

onUnmounted(() => {
  if (typeof unsubscribeDownloadComplete === 'function') unsubscribeDownloadComplete()
})
</script>

<style scoped>
.model-search {
  max-width: 960px;
  margin: 0 auto;
  padding: var(--spacing-lg, 1.5rem);
  display: flex;
  flex-direction: column;
  gap: var(--spacing-md, 0.75rem);
}

/* ── Search bar ───────────────────────────────────────── */
.search-bar {
  display: flex;
  gap: 0.5rem;
  align-items: center;
  flex-wrap: wrap;
}

.search-input-wrap {
  flex: 1;
  min-width: 200px;
  position: relative;
}

.search-icon {
  position: absolute;
  left: 0.75rem;
  top: 50%;
  transform: translateY(-50%);
  color: var(--text-secondary, #9ca3af);
  pointer-events: none;
}

.search-input {
  width: 100%;
  padding-left: 2.25rem !important;
}

.clear-btn {
  position: absolute;
  right: 0.25rem;
  top: 50%;
  transform: translateY(-50%);
}

.format-select { width: 140px; }

/* ── Loading ──────────────────────────────────────────── */
.loading-row {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  padding: 1.5rem 0;
  color: var(--text-secondary, #9ca3af);
}

/* ── Empty state ──────────────────────────────────────── */
.empty-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 1rem;
  padding: 4rem 0;
  text-align: center;
  color: var(--text-secondary, #9ca3af);
}

.empty-state h3 { margin: 0; font-size: 1.1rem; color: var(--text-primary, #f1f5f9); }
.empty-state p  { margin: 0; font-size: 0.875rem; }

/* ── Results table ────────────────────────────────────── */
.results-table {
  display: flex;
  flex-direction: column;
  gap: 0.375rem;
}

.results-header {
  padding: 0.25rem 0;
}

.results-count {
  font-size: 0.8rem;
  color: var(--text-secondary, #9ca3af);
}

.result-row {
  background: var(--bg-card, #161b2e);
  border: 1px solid var(--border-primary, #2a2f45);
  border-radius: var(--radius-lg, 0.75rem);
  overflow: hidden;
}

.result-main {
  display: flex;
  align-items: center;
  padding: 0.75rem 1rem;
  cursor: pointer;
  gap: 0.75rem;
  transition: background 0.15s;
}

.result-main:hover { background: var(--bg-card-hover, #1e2235); }

.result-expand-icon {
  color: var(--text-secondary, #9ca3af);
  width: 16px;
  flex-shrink: 0;
  font-size: 0.75rem;
}

.result-info { flex: 1; min-width: 0; }

.result-name {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  flex-wrap: wrap;
  margin-bottom: 0.3rem;
}

.model-link {
  font-weight: 600;
  font-size: 0.9rem;
  color: var(--text-primary, #f1f5f9);
  text-decoration: none;
  font-family: monospace;
}

.model-link:hover { color: var(--accent-cyan, #22d3ee); text-decoration: underline; }

.pipeline-tag { flex-shrink: 0; }

.result-meta {
  display: flex;
  gap: 0.875rem;
  flex-wrap: wrap;
}

.meta-item {
  font-size: 0.75rem;
  color: var(--text-secondary, #9ca3af);
  display: flex;
  align-items: center;
  gap: 0.25rem;
}

.meta-item .pi { font-size: 0.7rem; }
.license { font-style: italic; }

.result-tags {
  display: flex;
  gap: 0.25rem;
  flex-wrap: wrap;
  flex-shrink: 0;
}

.model-tag { font-size: 0.7rem; }

/* ── Expanded files ───────────────────────────────────── */
.row-expand-enter-active,
.row-expand-leave-active { transition: all 0.2s ease; overflow: hidden; }
.row-expand-enter-from,
.row-expand-leave-to    { max-height: 0; opacity: 0; }
.row-expand-enter-to,
.row-expand-leave-from  { max-height: 600px; opacity: 1; }

.result-files {
  border-top: 1px solid var(--border-primary, #2a2f45);
  padding: 0.75rem 1rem;
  background: var(--bg-surface, #1e2235);
}

.files-loading,
.files-empty {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  font-size: 0.875rem;
  color: var(--text-secondary, #9ca3af);
  padding: 0.5rem 0;
}

.files-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.875rem;
}

.files-table th {
  text-align: left;
  font-size: 0.7rem;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: var(--text-secondary, #9ca3af);
  padding: 0.375rem 0.5rem;
  border-bottom: 1px solid var(--border-primary, #2a2f45);
}

.files-table td {
  padding: 0.4rem 0.5rem;
  border-bottom: 1px solid rgba(255, 255, 255, 0.04);
  vertical-align: middle;
}

.files-table tr:last-child td { border-bottom: none; }

.file-subtext { color: var(--text-secondary, #9ca3af); font-size: 0.75rem; }
.file-size { color: var(--text-secondary, #9ca3af); white-space: nowrap; }
.file-count { color: var(--text-secondary, #9ca3af); white-space: nowrap; }
.projector-cell { min-width: 9rem; }
.projector-select { min-width: 8rem; }
.file-actions { display: flex; align-items: center; gap: 0.35rem; justify-content: flex-end; flex-wrap: wrap; }
.not-downloaded { color: var(--text-secondary, #9ca3af); }

.safetensors-download {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  margin-top: 0.75rem;
  padding-top: 0.75rem;
  border-top: 1px solid var(--border-primary, #2a2f45);
}

.safetensors-download small {
  font-size: 0.75rem;
  color: var(--text-secondary, #9ca3af);
}
</style>
