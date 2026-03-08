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
          @click="query = ''; searchResults = []"
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
                  <th>File</th>
                  <th>Size</th>
                  <th>Status</th>
                  <th></th>
                </tr>
              </thead>
              <tbody>
                <tr v-for="file in getFiles(result.modelId || result.id)" :key="file.filename">
                  <td class="file-name">
                    <code>{{ file.filename }}</code>
                    <Tag v-if="file.quantization" :value="file.quantization" severity="info" />
                  </td>
                  <td class="file-size">{{ formatBytes(file.size) }}</td>
                  <td class="file-status">
                    <Tag v-if="file.downloaded" value="Downloaded" severity="success" />
                    <span v-else class="not-downloaded">—</span>
                  </td>
                  <td class="file-action">
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
                      v-else
                      label="Download"
                      icon="pi pi-download"
                      size="small"
                      severity="success"
                      outlined
                      :loading="downloadingFiles.has(`${result.modelId || result.id}:${file.filename}`)"
                      @click="downloadFile(result, file)"
                    />
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
import { ref, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { useToast } from 'primevue/usetoast'
import Button from 'primevue/button'
import Tag from 'primevue/tag'
import InputText from 'primevue/inputtext'
import Dropdown from 'primevue/dropdown'
import ProgressSpinner from 'primevue/progressspinner'
import ProgressTracker from '@/components/common/ProgressTracker.vue'
import { useModelStore } from '@/stores/models'
import axios from 'axios'

const router = useRouter()
const toast = useToast()
const modelStore = useModelStore()

// ── State ──────────────────────────────────────────────────
const query = ref('')
const lastQuery = ref('')
const searchFormat = ref('gguf')
const searching = ref(false)
const hasSearched = ref(false)
const searchResults = ref([])
const expanded = ref(new Set())
const loadingFiles = ref(new Set())
const downloadingFiles = ref(new Set())
const filesCache = ref({})   // modelId -> files[]

const formatOptions = [
  { label: 'GGUF', value: 'gguf' },
  { label: 'Safetensors', value: 'safetensors' },
]

// ── Search ─────────────────────────────────────────────────
async function search() {
  if (!query.value.trim()) return
  searching.value = true
  hasSearched.value = true
  lastQuery.value = query.value
  expanded.value = new Set()
  filesCache.value = {}
  try {
    searchResults.value = await modelStore.searchModels(query.value.trim(), 20, searchFormat.value)
  } catch (e) {
    toast.add({ severity: 'error', summary: 'Search failed', detail: e.message, life: 4000 })
    searchResults.value = []
  } finally {
    searching.value = false
  }
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
      // Backend returns quantizations as a dict: { "Q4_K_M": { quantization, files: [{filename, size}], total_size, size_mb } }
      const quantEntries = Object.values(result.quantizations || {})
      // Flatten to individual files, keeping quant label
      const allFiles = quantEntries.flatMap(entry =>
        (entry.files || []).map(f => ({
          filename: f.filename,
          size: f.size || entry.total_size || 0,
          quantization: entry.quantization || '',
          variantPrefix: entry.variant_prefix || '',
        }))
      )

      if (allFiles.length) {
        // Try to get accurate sizes from the API
        try {
          const filenames = allFiles.map(f => f.filename).join(',')
          const { data } = await axios.get(`/api/models/search/${encodeURIComponent(modelId)}/file-sizes`, {
            params: { filenames },
          })
          const sizes = data.sizes || {}
          files = allFiles.map(f => {
            const downloaded = isDownloaded(modelId, f.filename)
            return {
              ...f,
              size: sizes[f.filename] ?? f.size,
              downloaded,
              modelId: downloaded?.id,
            }
          })
        } catch {
          files = allFiles.map(f => {
            const downloaded = isDownloaded(modelId, f.filename)
            return { ...f, downloaded, modelId: downloaded?.id }
          })
        }
      }
    } else {
      // Safetensors: backend returns safetensors_files: [{ filename }]
      const stFiles = result.safetensors_files || []
      files = stFiles.map(f => ({
        filename: f.filename,
        size: f.size || 0,
        downloaded: false,
      }))
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
  const key = `${modelId}:${file.filename}`
  downloadingFiles.value.add(key)
  downloadingFiles.value = new Set(downloadingFiles.value)
  try {
    await modelStore.downloadModel(
      modelId,
      file.filename,
      file.size || 0,
      searchFormat.value,
      result.pipeline_tag || null
    )
    toast.add({ severity: 'success', summary: 'Download started', detail: 'Track progress above', life: 3000 })
    // Refresh files to update downloaded status
    delete filesCache.value[modelId]
    await loadFiles(modelId)
    await modelStore.fetchModels()
  } catch (e) {
    toast.add({ severity: 'error', summary: 'Download failed', detail: e.message, life: 4000 })
  } finally {
    downloadingFiles.value.delete(key)
    downloadingFiles.value = new Set(downloadingFiles.value)
  }
}


function configureDownloaded(modelId, file) {
  const model = file.modelId
    ? modelStore.allQuantizations.find(m => m.id === file.modelId)
    : findDownloadedModel(modelId, file.filename)
  if (model) {
    router.push(`/models/${encodeURIComponent(model.id)}/config`)
  }
}

// ── Helpers ────────────────────────────────────────────────
function isDownloaded(hfId, filename) {
  return modelStore.allQuantizations.find(
    m => m.huggingface_id === hfId &&
    (m.filename === filename || (m.quantization && filename.includes(m.quantization)))
  )
}

function findDownloadedModel(hfId, filename) {
  return modelStore.allQuantizations.find(
    m => m.huggingface_id === hfId &&
    (m.filename === filename || (m.quantization && filename.includes(m.quantization)))
  )
}

function extractQuantization(filename) {
  if (!filename) return null
  const match = filename.match(/[_-](Q\d[_A-Z0-9]*(?:_M|_S|_XS|_XL|_XXS)?|IQ\d_[A-Z]+|BF16|F16|F32)/i)
  return match?.[1]?.toUpperCase() ?? null
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

// ── Lifecycle ──────────────────────────────────────────────
onMounted(async () => {
  if (!modelStore.models.length) await modelStore.fetchModels()
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

.file-name { display: flex; align-items: center; gap: 0.4rem; }
.file-name code { font-size: 0.8rem; }
.file-size { color: var(--text-secondary, #9ca3af); white-space: nowrap; }
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
