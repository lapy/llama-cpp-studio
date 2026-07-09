<template>
  <div class="model-search page-shell">

    <!-- Search bar -->
    <div class="search-bar">
      <div class="search-input-wrap">
        <i class="pi pi-search search-icon" />
        <InputText
          v-model="query"
          placeholder="Search models and audio packages…"
          class="search-input"
          @keyup.enter="runSearch"
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
        @click="runSearch"
      />
      <Button
        label="Import audio bundle"
        icon="pi pi-folder-open"
        severity="secondary"
        outlined
        @click="showAudioImportDialog = true"
      />
    </div>

    <div class="catalog-filters">
      <Dropdown
        v-model="engineFilter"
        :options="engineFilterOptions"
        optionLabel="label"
        optionValue="value"
        placeholder="Any engine"
        showClear
        class="catalog-filter"
        @change="runSearch"
      />
      <Dropdown
        v-model="taskFilter"
        :options="taskFilterOptions"
        placeholder="Any task"
        showClear
        class="catalog-filter"
        @change="runSearch"
      />
      <Dropdown
        v-model="inputModalityFilter"
        :options="inputModalityOptions"
        placeholder="Any input"
        showClear
        class="catalog-filter"
        @change="runSearch"
      />
      <Dropdown
        v-model="outputModalityFilter"
        :options="outputModalityOptions"
        placeholder="Any output"
        showClear
        class="catalog-filter"
        @change="runSearch"
      />
      <Dropdown
        v-model="providerFilter"
        :options="providerOptions"
        optionLabel="label"
        optionValue="value"
        placeholder="Any source"
        showClear
        class="catalog-filter"
        @change="runSearch"
      />
    </div>

    <ProgressTracker
      :type="['audio_model_install', 'audio_model_import']"
      show-completed
      section-title="Audio package activity"
    />

    <LoadingState v-if="searching" message="Searching…" inline />

    <EmptyState
      v-else-if="!searchResults.length && hasSearched"
      icon="pi pi-search"
      :title="`No results for “${lastQuery}”`"
      description="Try different keywords or broaden the engine, task, or modality filters."
    />

    <EmptyState
      v-else-if="!searchResults.length && !hasSearched"
      icon="pi pi-search"
      title="Discover compatible models"
      description="Search Hugging Face or browse version-pinned audio.cpp packages by engine and task."
    />

    <div v-else-if="catalogMode" class="catalog-results">
      <div class="results-header">
        <span class="results-count">{{ catalogTotal }} verified result{{ catalogTotal !== 1 ? 's' : '' }}</span>
        <div class="provider-statuses">
          <Tag
            v-for="(status, provider) in catalogProviderStatus"
            :key="provider"
            :value="status.available ? `${provider} ready` : `${provider} unavailable`"
            :severity="status.available ? 'success' : 'warning'"
            v-tooltip.bottom="status.reason || status.manager_warning || ''"
          />
        </div>
      </div>

      <article
        v-for="result in sortedSearchResults"
        :key="result.id"
        class="catalog-card"
      >
        <div class="catalog-card__head">
          <div>
            <a
              v-if="catalogSourceUrl(result)"
              :href="catalogSourceUrl(result)"
              target="_blank"
              class="model-link"
            >
              {{ result.display_name }}
            </a>
            <span v-else class="model-link">{{ result.display_name }}</span>
            <div class="catalog-card__id">{{ result.provider_item_id }}</div>
          </div>
          <div class="result-name__tags">
            <Tag :value="result.provider === 'audio_cpp' ? 'audio.cpp catalog' : 'Hugging Face'" severity="info" />
            <Tag v-if="result.gated" value="Gated" severity="warning" />
            <Tag :value="result.package_kind" severity="secondary" />
          </div>
        </div>

        <p v-if="result.description" class="catalog-card__description">{{ result.description }}</p>

        <div class="catalog-badges">
          <Tag v-if="result.family" :value="result.family" severity="secondary" />
          <Tag v-for="task in result.tasks || []" :key="`task-${task}`" :value="task" :severity="pipelineTagSeverity(task)" />
          <Tag v-for="modality in result.input_modalities || []" :key="`in-${modality}`" :value="`${modality} in`" severity="info" />
          <Tag v-for="modality in result.output_modalities || []" :key="`out-${modality}`" :value="`${modality} out`" severity="success" />
          <Tag v-if="(result.features || []).includes('streaming')" value="Streaming" severity="success" />
        </div>

        <div v-if="result.unavailable_reason" class="catalog-unavailable">
          <i class="pi pi-exclamation-triangle" />
          {{ result.unavailable_reason }}
        </div>
        <div v-else class="compatibility-evidence">
          <i class="pi pi-verified" />
          Compatible with {{ (result.compatible_engines || []).join(', ') }}
          <span v-if="compatibilityEvidence(result)">— {{ compatibilityEvidence(result) }}</span>
        </div>

        <div class="install-variants">
          <div
            v-for="variant in result.install_variants || []"
            :key="variant.id"
            class="install-variant"
          >
            <div>
              <strong>{{ variant.label || variant.id }}</strong>
              <span class="install-variant__meta">
                {{ variant.method }}
                <template v-if="variant.size_bytes"> · {{ formatBytes(variant.size_bytes) }}</template>
                <template v-if="variant.files?.length"> · {{ variant.files.length }} file{{ variant.files.length === 1 ? '' : 's' }}</template>
              </span>
              <small v-if="variant.external_inputs_required">Additional local source input may be required.</small>
            </div>
            <Button
              label="Install"
              icon="pi pi-download"
              size="small"
              severity="success"
              outlined
              :disabled="!variant.installable"
              :loading="catalogInstallKey === `${result.id}:${variant.id}`"
              @click="installCatalogVariant(result, variant)"
            />
          </div>
        </div>
      </article>

      <div v-if="catalogTotal > catalogPageSize" class="catalog-pagination">
        <Button label="Previous" icon="pi pi-chevron-left" severity="secondary" outlined
          :disabled="catalogPage <= 1 || searching" @click="changeCatalogPage(catalogPage - 1)" />
        <span>Page {{ catalogPage }}</span>
        <Button label="Next" icon="pi pi-chevron-right" iconPos="right" severity="secondary" outlined
          :disabled="!catalogHasMore || searching" @click="changeCatalogPage(catalogPage + 1)" />
      </div>
    </div>

    <!-- Legacy result renderer remains as a compatibility adapter. -->
    <div v-else class="results-table">
      <div class="results-header">
        <span class="results-count">{{ searchResults.length }} result{{ searchResults.length !== 1 ? 's' : '' }}</span>
        <div class="results-sort">
          <label class="results-sort__label" for="model-search-sort">Sort</label>
          <Dropdown
            id="model-search-sort"
            v-model="sortBy"
            :options="sortOptions"
            optionLabel="label"
            optionValue="value"
            class="results-sort__dropdown"
          />
        </div>
      </div>

      <div
        v-for="result in sortedSearchResults"
        :key="result.modelId || result.id"
        class="result-row"
      >
        <!-- Main row -->
        <div
          class="result-main interactive-row"
          tabindex="0"
          role="button"
          :aria-expanded="expanded.has(result.modelId || result.id)"
          :aria-label="`Expand ${result.modelId || result.id}`"
          @click="toggleExpand(result.modelId || result.id)"
          @keydown.enter.prevent="toggleExpand(result.modelId || result.id)"
          @keydown.space.prevent="toggleExpand(result.modelId || result.id)"
        >
          <div class="result-expand-icon">
            <i :class="['pi', expanded.has(result.modelId || result.id) ? 'pi-chevron-down' : 'pi-chevron-right']" />
          </div>

          <div class="result-info">
            <div class="result-name">
              <a
                :href="`https://huggingface.co/${result.modelId || result.id}`"
                target="_blank"
                :class="[
                  'model-link',
                  expanded.has(result.modelId || result.id)
                    ? 'model-link--expanded'
                    : 'model-link--collapsed',
                ]"
                :title="result.modelId || result.id"
                @click.stop
              >
                {{ result.modelId || result.id }}
              </a>
              <div class="result-name__tags">
                <Tag
                  v-if="result.pipeline_tag"
                  :value="result.pipeline_tag"
                  :severity="pipelineTagSeverity(result.pipeline_tag)"
                  class="pipeline-tag"
                />
                <Tag
                  v-if="result.gated"
                  value="Gated"
                  severity="warning"
                  class="pipeline-tag"
                  v-tooltip.bottom="'Accept the model license on Hugging Face before download'"
                />
              </div>
            </div>
            <div class="result-meta">
              <span v-if="result.author" class="meta-item">
                <i class="pi pi-user" /> {{ result.author }}
              </span>
              <span class="meta-item">
                <i class="pi pi-download" /> {{ formatNumber(result.downloads ?? 0) }}
              </span>
              <span class="meta-item">
                <i class="pi pi-heart" /> {{ formatNumber(result.likes ?? 0) }}
              </span>
              <span v-if="getResultArtifactCount(result)" class="meta-item">
                <i class="pi pi-database" /> {{ getResultArtifactCount(result) }}
              </span>
              <span v-if="getResultSizeSummary(result)" class="meta-item">
                <i class="pi pi-box" /> {{ getResultSizeSummary(result) }}
              </span>
              <span v-if="result.library_name" class="meta-item" :title="result.library_name">
                <i class="pi pi-book" /> {{ result.library_name }}
              </span>
              <span v-if="result.parameters" class="meta-item" :title="result.architecture || ''">
                <i class="pi pi-sliders-h" /> {{ result.parameters }}
              </span>
              <span v-else-if="result.architecture" class="meta-item">
                <i class="pi pi-sitemap" /> {{ result.architecture }}
              </span>
              <span v-if="formatContextLength(result.context_length)" class="meta-item">
                <i class="pi pi-window-maximize" /> {{ formatContextLength(result.context_length) }}
              </span>
              <span v-if="formatLanguageHint(result.language)" class="meta-item">
                <i class="pi pi-globe" /> {{ formatLanguageHint(result.language) }}
              </span>
              <span v-if="formatSearchUpdatedAt(result.updated_at)" class="meta-item">
                <i class="pi pi-calendar" /> {{ formatSearchUpdatedAt(result.updated_at) }}
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
            <LoadingState
              v-if="loadingFiles.has(result.modelId || result.id)"
              message="Loading files…"
              inline
              small
            />

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
                      :disabled="isFileDownloading(result.modelId || result.id, file)"
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
                        :loading="isFileDownloading(result.modelId || result.id, file)"
                        @click="updateProjector(result, file)"
                      />
                      <Button
                        v-if="!file.downloaded"
                        label="Download"
                        icon="pi pi-download"
                        size="small"
                        severity="success"
                        outlined
                        :loading="isFileDownloading(result.modelId || result.id, file)"
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

    <Dialog
      v-model:visible="showInstallOptionsDialog"
      modal
      header="Prepare audio.cpp package"
      :style="{ width: 'min(34rem, 94vw)' }"
    >
      <p class="dialog-help">
        This package runs a pinned audio.cpp converter. Provide the local input requested by the package.
        Repository code is never executed.
      </p>
      <div class="audio-install-form">
        <label>
          <span>Source file</span>
          <InputText
            v-model="installOptions.source_file"
            placeholder="/data/input/model.pt"
            fluid
          />
        </label>
        <div class="field-divider">or</div>
        <label>
          <span>Source directory</span>
          <InputText
            v-model="installOptions.source_dir"
            placeholder="/data/input/prepared-source"
            fluid
          />
        </label>
        <label>
          <span>Variant (optional)</span>
          <InputText
            v-model="installOptions.variant"
            placeholder="Package-specific variant"
            fluid
          />
        </label>
        <label>
          <span>Family override (optional)</span>
          <InputText
            v-model="installOptions.family"
            placeholder="Detected automatically"
            fluid
          />
        </label>
      </div>
      <template #footer>
        <Button label="Cancel" severity="secondary" text @click="closeInstallOptions" />
        <Button
          label="Start install"
          icon="pi pi-download"
          severity="success"
          :disabled="!installOptions.source_file && !installOptions.source_dir"
          @click="confirmCatalogInstall"
        />
      </template>
    </Dialog>

    <Dialog
      v-model:visible="showAudioImportDialog"
      modal
      header="Import prepared audio.cpp bundle"
      :style="{ width: 'min(34rem, 94vw)' }"
    >
      <p class="dialog-help">
        The directory is copied into managed storage and accepted only when the active
        <code>audiocpp_cli</code> identifies a valid family and task.
      </p>
      <div class="audio-install-form">
        <label>
          <span>Local directory</span>
          <InputText
            v-model="audioImport.source_path"
            placeholder="/data/models/my-audio-model"
            fluid
          />
        </label>
        <label>
          <span>Package ID (optional)</span>
          <InputText
            v-model="audioImport.package_id"
            placeholder="Defaults to the directory name"
            fluid
          />
        </label>
        <label>
          <span>Family override (optional)</span>
          <InputText
            v-model="audioImport.family"
            placeholder="Detected automatically"
            fluid
          />
        </label>
      </div>
      <template #footer>
        <Button label="Cancel" severity="secondary" text @click="showAudioImportDialog = false" />
        <Button
          label="Validate and import"
          icon="pi pi-folder-open"
          severity="success"
          :loading="audioImportSubmitting"
          :disabled="!audioImport.source_path"
          @click="submitAudioImport"
        />
      </template>
    </Dialog>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, onUnmounted } from 'vue'
import { storeToRefs } from 'pinia'
import { useRouter } from 'vue-router'
import { useToast } from 'primevue/usetoast'
import Button from 'primevue/button'
import Tag from 'primevue/tag'
import InputText from 'primevue/inputtext'
import Dropdown from 'primevue/dropdown'
import Dialog from 'primevue/dialog'
import LoadingState from '@/components/common/LoadingState.vue'
import EmptyState from '@/components/common/EmptyState.vue'
import ProgressTracker from '@/components/common/ProgressTracker.vue'
import { useModelStore } from '@/stores/models'
import { useEnginesStore } from '@/stores/engines'
import { useProgressStore } from '@/stores/progress'
import axios from 'axios'

const router = useRouter()
const toast = useToast()
const modelStore = useModelStore()
const enginesStore = useEnginesStore()
const progressStore = useProgressStore()
const { tasks: progressTasks } = storeToRefs(progressStore)
const {
  searchQuery: query,
  searchLastQuery: lastQuery,
  searchHasSearched: hasSearched,
  searchResults,
  searchLoading: searching,
  searchFormat,
  catalogFacets,
  catalogProviderStatus,
  catalogTotal,
  catalogPage,
  catalogHasMore,
} = storeToRefs(modelStore)

// ── State ──────────────────────────────────────────────────
const expanded = ref(new Set())
const loadingFiles = ref(new Set())
const downloadingFiles = ref(new Set())
const filesCache = ref({})   // modelId -> files[]
const projectorSelections = ref({})
const catalogMode = ref(import.meta.env.MODE !== 'test')
const catalogInstallKey = ref(null)
const showInstallOptionsDialog = ref(false)
const pendingCatalogInstall = ref(null)
const installOptions = ref({
  source_file: '',
  source_dir: '',
  variant: '',
  family: '',
})
const showAudioImportDialog = ref(false)
const audioImportSubmitting = ref(false)
const audioImport = ref({
  source_path: '',
  package_id: '',
  family: '',
})
const catalogPageSize = 20
const engineFilter = ref(null)
const taskFilter = ref(null)
const inputModalityFilter = ref(null)
const outputModalityFilter = ref(null)
const providerFilter = ref(null)

const formatOptions = [
  { label: 'All packages', value: 'all' },
  { label: 'GGUF', value: 'gguf' },
  { label: 'Safetensors', value: 'safetensors' },
  { label: 'Prepared audio', value: 'mixed' },
]

const engineFilterOptions = computed(() => {
  const descriptors = enginesStore.engineDescriptors || []
  if (descriptors.length) {
    return descriptors.map(engine => ({ value: engine.id, label: engine.label }))
  }
  return [
    { value: 'llama_cpp', label: 'llama.cpp' },
    { value: 'ik_llama', label: 'ik_llama.cpp' },
    { value: 'lmdeploy', label: 'LMDeploy' },
    { value: '1cat_vllm', label: '1Cat-vLLM' },
    { value: 'audio_cpp', label: 'audio.cpp' },
  ]
})
const taskFilterOptions = computed(() => {
  const values = catalogFacets.value?.tasks || []
  return values.length ? values : ['tts', 'asr', 'vad', 'diar', 'sep', 'gen', 'vc', 's2s', 'align', 'text-generation', 'embeddings']
})
const inputModalityOptions = computed(() => catalogFacets.value?.input_modalities || ['text', 'audio', 'image'])
const outputModalityOptions = computed(() => catalogFacets.value?.output_modalities || ['text', 'audio', 'segments', 'events', 'embedding'])
const providerOptions = [
  { label: 'Hugging Face', value: 'huggingface' },
  { label: 'audio.cpp packages', value: 'audio_cpp' },
]

/** Default: downloads high → low */
const sortBy = ref('downloads_desc')

const sortOptions = [
  { label: 'Downloads (high → low)', value: 'downloads_desc' },
  { label: 'Downloads (low → high)', value: 'downloads_asc' },
  { label: 'Likes (high → low)', value: 'likes_desc' },
  { label: 'Likes (low → high)', value: 'likes_asc' },
  { label: 'Name A–Z', value: 'name_asc' },
  { label: 'Name Z–A', value: 'name_desc' },
]

/** Stable color per pipeline_tag (PrimeVue Tag severities). Unknown tags get a deterministic hash color. */
const PIPELINE_TAG_SEVERITY = Object.freeze({
  'text-generation': 'success',
  'text2text-generation': 'success',
  'fill-mask': 'info',
  'question-answering': 'info',
  summarization: 'warning',
  translation: 'warning',
  /* Multimodal / vision — avoid `contrast` (often near-white on light tags in dark UI) */
  'image-text-to-text': 'info',
  'image-to-text': 'info',
  'visual-question-answering': 'warning',
  'automatic-speech-recognition': 'info',
  'feature-extraction': 'secondary',
  'text-classification': 'info',
  'token-classification': 'info',
  'sentence-similarity': 'info',
  'reinforcement-learning': 'secondary',
  robotics: 'secondary',
  'depth-estimation': 'warning',
  'tabular-classification': 'info',
  'tabular-regression': 'info',
  'table-question-answering': 'info',
  'object-detection': 'warning',
  'image-classification': 'warning',
  'audio-classification': 'info',
  'audio-to-audio': 'info',
  'text-to-speech': 'info',
  'text-to-audio': 'info',
  'voice-activity-detection': 'secondary',
  'other': 'secondary',
})

const PIPELINE_SEVERITY_POOL = ['success', 'info', 'warning', 'secondary']

function pipelineTagSeverity(pipelineTag) {
  if (pipelineTag == null || pipelineTag === '') return 'secondary'
  const k = String(pipelineTag).toLowerCase().trim()
  if (PIPELINE_TAG_SEVERITY[k]) return PIPELINE_TAG_SEVERITY[k]
  let h = 0
  for (let i = 0; i < k.length; i++) h = (h * 31 + k.charCodeAt(i)) >>> 0
  return PIPELINE_SEVERITY_POOL[h % PIPELINE_SEVERITY_POOL.length]
}

function modelIdKey(r) {
  return String(r.display_name || r.modelId || r.id || '').toLowerCase()
}

function numOrMissing(v, missingSentinel) {
  const n = typeof v === 'number' ? v : Number(v)
  return Number.isFinite(n) ? n : missingSentinel
}

const sortedSearchResults = computed(() => {
  const list = [...searchResults.value]
  const key = sortBy.value

  list.sort((a, b) => {
    const idA = modelIdKey(a)
    const idB = modelIdKey(b)

    switch (key) {
      case 'downloads_desc': {
        const da = numOrMissing(a.downloads ?? a.metadata?.downloads, -1)
        const db = numOrMissing(b.downloads ?? b.metadata?.downloads, -1)
        if (db !== da) return db - da
        break
      }
      case 'downloads_asc': {
        const da = numOrMissing(a.downloads ?? a.metadata?.downloads, Number.MAX_SAFE_INTEGER)
        const db = numOrMissing(b.downloads ?? b.metadata?.downloads, Number.MAX_SAFE_INTEGER)
        if (da !== db) return da - db
        break
      }
      case 'likes_desc': {
        const la = numOrMissing(a.likes ?? a.metadata?.likes, -1)
        const lb = numOrMissing(b.likes ?? b.metadata?.likes, -1)
        if (lb !== la) return lb - la
        break
      }
      case 'likes_asc': {
        const la = numOrMissing(a.likes ?? a.metadata?.likes, Number.MAX_SAFE_INTEGER)
        const lb = numOrMissing(b.likes ?? b.metadata?.likes, Number.MAX_SAFE_INTEGER)
        if (la !== lb) return la - lb
        break
      }
      case 'name_asc':
        return idA.localeCompare(idB)
      case 'name_desc':
        return idB.localeCompare(idA)
      default:
        return 0
    }
    return idA.localeCompare(idB)
  })

  return list
})

// ── Search ─────────────────────────────────────────────────
function catalogFiltersPayload() {
  return {
    ...(engineFilter.value ? { engine: engineFilter.value } : {}),
    ...(taskFilter.value ? { task: taskFilter.value } : {}),
    ...(inputModalityFilter.value ? { input_modality: inputModalityFilter.value } : {}),
    ...(outputModalityFilter.value ? { output_modality: outputModalityFilter.value } : {}),
    ...(providerFilter.value ? { provider: providerFilter.value } : {}),
    ...(!engineFilter.value && searchFormat.value && searchFormat.value !== 'all'
      ? { artifact_format: searchFormat.value }
      : {}),
  }
}

async function search(page = 1) {
  const pageNum = Number(page)
  const safePage = Number.isFinite(pageNum) && pageNum >= 1 ? Math.floor(pageNum) : 1
  expanded.value = new Set()
  filesCache.value = {}
  projectorSelections.value = {}
  try {
    catalogMode.value = true
    await modelStore.searchCatalog(query.value.trim(), {
      page: safePage,
      page_size: catalogPageSize,
      filters: catalogFiltersPayload(),
    })
  } catch (e) {
    toast.add({ severity: 'error', summary: 'Search failed', detail: e?.response?.data?.detail || e.message, life: 4000 })
    searchResults.value = []
  }
}

function runSearch() {
  return search(1)
}

async function changeCatalogPage(page) {
  await search(page)
}

function catalogSourceUrl(result) {
  if (result?.provider === 'huggingface' && result?.source?.id) {
    return `https://huggingface.co/${result.source.id}`
  }
  if (result?.provider === 'audio_cpp') {
    return 'https://github.com/0xShug0/audio.cpp'
  }
  return null
}

function compatibilityEvidence(result) {
  const engine = engineFilter.value || result?.compatible_engines?.[0]
  const evidence = result?.compatibility?.[engine]?.evidence
  return Array.isArray(evidence) ? evidence[0] : ''
}

async function installCatalogVariant(result, variant) {
  if (variant?.external_inputs_required) {
    pendingCatalogInstall.value = { result, variant }
    installOptions.value = {
      source_file: '',
      source_dir: '',
      variant: '',
      family: result?.family || '',
    }
    showInstallOptionsDialog.value = true
    return
  }
  await startCatalogInstall(result, variant)
}

async function startCatalogInstall(result, variant, options = {}) {
  const key = `${result.id}:${variant.id}`
  catalogInstallKey.value = key
  try {
    const response = await modelStore.installCatalogModel(result, variant, options)
    toast.add({
      severity: 'success',
      summary: 'Installation started',
      detail: response?.message || 'Track progress in notifications.',
      life: 3500,
    })
  } catch (e) {
    toast.add({
      severity: 'error',
      summary: 'Install failed',
      detail: e?.response?.data?.detail || e.message,
      life: 5000,
    })
  } finally {
    catalogInstallKey.value = null
  }
}

function closeInstallOptions() {
  showInstallOptionsDialog.value = false
  pendingCatalogInstall.value = null
}

async function confirmCatalogInstall() {
  const pending = pendingCatalogInstall.value
  if (!pending) return
  const options = Object.fromEntries(
    Object.entries(installOptions.value).filter(([, value]) => String(value || '').trim()),
  )
  showInstallOptionsDialog.value = false
  pendingCatalogInstall.value = null
  await startCatalogInstall(pending.result, pending.variant, options)
}

async function submitAudioImport() {
  if (!audioImport.value.source_path) return
  audioImportSubmitting.value = true
  try {
    const response = await modelStore.importAudioBundle(
      audioImport.value.source_path,
      {
        ...(audioImport.value.package_id ? { package_id: audioImport.value.package_id } : {}),
        ...(audioImport.value.family ? { family: audioImport.value.family } : {}),
      },
    )
    toast.add({
      severity: 'success',
      summary: 'Import started',
      detail: response?.message || 'The bundle will be inspected before it is added.',
      life: 3500,
    })
    showAudioImportDialog.value = false
    audioImport.value = { source_path: '', package_id: '', family: '' }
  } catch (e) {
    toast.add({
      severity: 'error',
      summary: 'Import failed',
      detail: e?.response?.data?.detail || e.message,
      life: 5000,
    })
  } finally {
    audioImportSubmitting.value = false
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
    toast.add({ severity: 'success', summary: 'Download started', detail: 'Track progress in notifications', life: 3000 })
    // Refresh files to update downloaded status
    delete filesCache.value[modelId]
    await loadFiles(modelId)
    await modelStore.fetchModels()
    if (searchFormat.value === 'safetensors') {
      await modelStore.fetchSafetensorsModels()
    }
  } catch (e) {
    toast.add({ severity: 'error', summary: 'Download failed', detail: e.message, life: 4000 })
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
      toast.add({ severity: 'success', summary: 'Projector update started', detail: response?.message || 'Track progress in notifications', life: 3000 })
    }
  } catch (e) {
    toast.add({ severity: 'error', summary: 'Projector update failed', detail: e.message, life: 4000 })
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

function downloadTaskIdentityMatches(task, modelId, file) {
  if (task?.type !== 'download') return false

  const meta = task.metadata || {}
  if (meta.huggingface_id && meta.huggingface_id !== modelId) return false

  if (searchFormat.value === 'gguf' && file.kind === 'quant') {
    if (meta.model_id && file.modelId && meta.model_id === file.modelId) {
      return true
    }
    if (!meta.quantization) return false
    const quantKey = file.quantizationKey || file.quantization
    return meta.quantization === quantKey || meta.quantization === file.quantization
  }

  if (searchFormat.value === 'safetensors') {
    return meta.huggingface_id === modelId && !meta.quantization && !meta.filename
  }

  if (meta.filename) {
    return meta.filename === file.filename
  }

  return false
}

function isActiveDownloadTask(task, modelId, file) {
  return task?.status === 'running' && downloadTaskIdentityMatches(task, modelId, file)
}

function isFileDownloading(modelId, file) {
  const key = getDownloadKey(modelId, file)
  if (downloadingFiles.value.has(key)) return true
  return Object.values(progressTasks.value).some(task => isActiveDownloadTask(task, modelId, file))
}

function clearPendingDownload(modelId, file) {
  const key = getDownloadKey(modelId, file)
  if (!downloadingFiles.value.has(key)) return
  downloadingFiles.value.delete(key)
  downloadingFiles.value = new Set(downloadingFiles.value)
}

function reconcilePendingDownload(modelId, file) {
  const key = getDownloadKey(modelId, file)
  if (!downloadingFiles.value.has(key)) return

  const tasks = Object.values(progressTasks.value)
  if (tasks.some(task => isActiveDownloadTask(task, modelId, file))) {
    clearPendingDownload(modelId, file)
    return
  }
  if (tasks.some(task => downloadTaskIdentityMatches(task, modelId, file) && task.status !== 'running')) {
    clearPendingDownload(modelId, file)
  }
}

function reconcilePendingDownloadsForHfId(hfId) {
  for (const file of filesCache.value[hfId] || []) {
    reconcilePendingDownload(hfId, file)
  }
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

/** Card model-index context length → short label for the result row */
function formatContextLength(ctx) {
  if (ctx == null || ctx === '') return ''
  const n = Number(ctx)
  if (!Number.isFinite(n)) return String(ctx)
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M ctx`
  if (n >= 10_000) return `${Math.round(n / 1000)}k ctx`
  if (n >= 1000) return `${(n / 1000).toFixed(1)}k ctx`
  return `${n} ctx`
}

function formatLanguageHint(langs) {
  if (langs == null) return ''
  const arr = Array.isArray(langs) ? langs : [langs]
  const flat = arr.map(l => String(l).trim()).filter(Boolean)
  if (!flat.length) return ''
  return flat.slice(0, 4).join(', ')
}

function formatSearchUpdatedAt(iso) {
  if (!iso) return ''
  const d = new Date(iso)
  if (Number.isNaN(d.getTime())) return ''
  const diff = Date.now() - d.getTime()
  const day = 86400000
  if (diff < 0) return 'updated'
  if (diff < day) return 'updated today'
  if (diff < 7 * day) return `updated ${Math.floor(diff / day)}d ago`
  if (diff < 60 * day) return `updated ${Math.floor(diff / (7 * day))}w ago`
  return `updated ${d.toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: 'numeric' })}`
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
let unsubscribeDownloadTaskCreated = null
let unsubscribeDownloadTaskUpdated = null

function handleDownloadTaskEvent(task) {
  if (task?.type !== 'download') return
  const hfId = task.metadata?.huggingface_id
  if (hfId) reconcilePendingDownloadsForHfId(hfId)
}

onMounted(async () => {
  if (!enginesStore.engineDescriptors.length) {
    await enginesStore.fetchEngineDescriptors().catch(() => {})
  }
  if (!modelStore.models.length) await modelStore.fetchModels()
  if (!modelStore.safetensorsModels.length) await modelStore.fetchSafetensorsModels()
  unsubscribeDownloadTaskCreated = progressStore.subscribe('task_created', handleDownloadTaskEvent)
  unsubscribeDownloadTaskUpdated = progressStore.subscribe('task_updated', handleDownloadTaskEvent)
  unsubscribeDownloadComplete = progressStore.subscribeToDownloadComplete(async (payload) => {
    const hfId = payload?.huggingface_id
    if (!hfId) return
    if (!searchResults.value.some(result => (result.modelId || result.id) === hfId)) return
    reconcilePendingDownloadsForHfId(hfId)
    if (payload?.model_format === 'gguf-bundle') {
      markDownloadedFromEvent(payload)
    }
    await refreshModelSearchState()
  })
})

onUnmounted(() => {
  if (typeof unsubscribeDownloadTaskCreated === 'function') unsubscribeDownloadTaskCreated()
  if (typeof unsubscribeDownloadTaskUpdated === 'function') unsubscribeDownloadTaskUpdated()
  if (typeof unsubscribeDownloadComplete === 'function') unsubscribeDownloadComplete()
})
</script>

<style scoped>
/* layout: .page-shell */

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

.catalog-filters {
  display: flex;
  gap: 0.5rem;
  align-items: center;
  flex-wrap: wrap;
}

.catalog-filter {
  min-width: 10rem;
  flex: 1 1 10rem;
  max-width: 15rem;
}

.catalog-results {
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
}

.provider-statuses,
.catalog-badges {
  display: flex;
  gap: 0.35rem;
  flex-wrap: wrap;
}

.catalog-card {
  padding: 1rem;
  background: var(--bg-card, #161b2e);
  border: 1px solid var(--border-primary, #2a2f45);
  border-radius: var(--radius-lg, 0.75rem);
}

.catalog-card__head {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 1rem;
}

.catalog-card__id {
  margin-top: 0.2rem;
  color: var(--text-secondary);
  font-size: 0.75rem;
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
}

.catalog-card__description {
  margin: 0.75rem 0;
  color: var(--text-secondary);
  font-size: 0.875rem;
  line-height: 1.5;
}

.catalog-unavailable,
.compatibility-evidence {
  display: flex;
  gap: 0.45rem;
  align-items: center;
  margin-top: 0.75rem;
  font-size: 0.8rem;
}

.catalog-unavailable {
  color: var(--accent-yellow, #f59e0b);
}

.compatibility-evidence {
  color: var(--accent-green, #10b981);
}

.install-variants {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
  margin-top: 0.85rem;
}

.install-variant {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 1rem;
  padding: 0.65rem 0.75rem;
  border: 1px solid var(--border-primary);
  border-radius: var(--radius-md);
  background: var(--bg-surface);
}

.install-variant > div {
  display: flex;
  flex-direction: column;
  gap: 0.2rem;
  min-width: 0;
}

.install-variant__meta,
.install-variant small {
  color: var(--text-secondary);
  font-size: 0.75rem;
}

.catalog-pagination {
  display: flex;
  justify-content: center;
  align-items: center;
  gap: 1rem;
  margin-top: 0.5rem;
}

/* ── Results table ────────────────────────────────────── */
.results-table {
  display: flex;
  flex-direction: column;
  gap: 0.375rem;
}

.results-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 0.75rem;
  flex-wrap: wrap;
  padding: 0.25rem 0;
}

.results-count {
  font-size: 0.8rem;
  color: var(--text-secondary, #9ca3af);
}

.results-sort {
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.results-sort__label {
  font-size: 0.75rem;
  color: var(--text-secondary, #9ca3af);
  white-space: nowrap;
}

.results-sort__dropdown {
  min-width: 12.5rem;
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
  align-items: flex-start;
  gap: 0.5rem;
  width: 100%;
  margin-bottom: 0.3rem;
}

.result-name__tags {
  display: inline-flex;
  align-items: center;
  gap: 0.5rem;
  flex-shrink: 0;
  margin-left: auto;
}

.model-link {
  font-weight: 600;
  font-size: 0.9rem;
  color: var(--text-primary, #f1f5f9);
  text-decoration: none;
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
  min-width: 0;
  align-self: flex-start;
}

/* Collapsed row: keep the bar compact; full id in title tooltip */
.model-link--collapsed {
  flex: 0 1 auto;
  min-width: 0;
  max-width: min(100%, 40ch);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

/* Expanded: show full repo id (may wrap) */
.model-link--expanded {
  white-space: normal;
  word-break: break-word;
  overflow: visible;
  max-width: none;
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

.result-files :deep(.page-loading--inline) {
  padding: 0.5rem 0;
}

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

.dialog-help {
  margin: 0 0 1rem;
  color: var(--text-secondary, #9ca3af);
  line-height: 1.45;
}

.audio-install-form {
  display: grid;
  gap: 0.9rem;
}

.audio-install-form label {
  display: grid;
  gap: 0.4rem;
  color: var(--text-primary, #f1f5f9);
  font-size: 0.84rem;
  font-weight: 600;
}

.field-divider {
  margin: -0.5rem 0;
  color: var(--text-secondary, #9ca3af);
  font-size: 0.72rem;
  text-align: center;
  text-transform: uppercase;
}
</style>
