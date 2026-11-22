<template>
  <div class="model-search page-shell page-shell--wide">
    <PageHeader
      title="Search models"
      description="Find compatible models and choose a download for your setup."
    >
      <template #meta>
        <Tag v-if="resultCountLabel" :value="resultCountLabel" severity="info" />
      </template>
      <template #actions>
        <Button
          icon="pi pi-refresh"
          text
          severity="secondary"
          aria-label="Refresh search"
          :loading="searching"
          v-tooltip.top="'Refresh'"
          @click="refreshSearch"
        />
        <Button
          label="Import audio bundle"
          icon="pi pi-folder-open"
          severity="secondary"
          outlined
          @click="showAudioImportDialog = true"
        />
      </template>
    </PageHeader>

    <ModelSearchControls
      v-model:query="query"
      v-model:task-filter="taskFilter"
      v-model:filters-open="filtersOpen"
      v-model:search-format="searchFormat"
      v-model:engine-filter="engineFilter"
      v-model:input-modality-filter="inputModalityFilter"
      v-model:output-modality-filter="outputModalityFilter"
      v-model:provider-filter="providerFilter"
      v-model:install-method-filter="installMethodFilter"
      :searching="searching"
      :secondary-filter-count="secondaryFilterCount"
      :active-filter-chips="activeFilterChips"
      :show-format-select="showFormatSelect"
      :task-filter-options="taskFilterOptions"
      :format-options="formatOptions"
      :engine-filter-options="engineFilterOptions"
      :input-modality-options="inputModalityOptions"
      :output-modality-options="outputModalityOptions"
      :provider-options="providerOptions"
      :install-method-options="installMethodOptions"
      @search="runSearch"
      @clear="clearSearchResults"
      @reset="resetSecondaryFilters"
      @format-change="onFormatChange"
      @engine-change="onEngineFilterChange"
    />

    <HuggingFaceTokenNotice
      v-if="!modelStore.hasHuggingfaceToken && !tokenNoticeDismissed"
      dismissible
      @open="showTokenDialog = true"
      @dismiss="dismissTokenNotice"
    />

    <StateBanner
      v-if="unavailableProviders.length"
      severity="warning"
      :message="unavailableProviderMessage"
    >
      <template #actions>
        <Button label="Retry" size="small" @click="runSearch" />
      </template>
    </StateBanner>

    <StateBanner
      v-if="searchError && searchResults.length"
      severity="warning"
      message="Could not refresh search. Showing the previous results."
    >
      <template #actions>
        <Button label="Retry" size="small" @click="runSearch" />
      </template>
    </StateBanner>

    <EmptyState
      v-if="searchError && !searchResults.length && !searching"
      icon="pi pi-exclamation-circle"
      role="alert"
      title="Could not search"
      :description="String(searchError)"
    >
      <Button label="Retry" icon="pi pi-refresh" @click="runSearch" />
      <Button
        label="Import audio bundle"
        severity="secondary"
        outlined
        @click="showAudioImportDialog = true"
      />
    </EmptyState>

    <EmptyState
      v-else-if="!searchResults.length && hasSearched && !searching"
      icon="pi pi-search"
      :title="`No results for “${lastQuery}”`"
      description="Try different keywords or broaden the engine, task, or modality filters."
    >
      <Button
        v-if="secondaryFilterCount || taskFilter"
        label="Clear filters"
        severity="secondary"
        outlined
        @click="clearAllSearchFilters"
      />
      <Button
        label="Import audio bundle"
        severity="secondary"
        text
        @click="showAudioImportDialog = true"
      />
    </EmptyState>

    <EmptyState
      v-else-if="!searchResults.length && !hasSearched && !searching"
      icon="pi pi-search"
      title="Search the model catalog"
      description="Search by model name or Hugging Face repository. Use Task to find audio models, or start from an example."
    >
      <Button
        v-for="example in searchExamples"
        :key="example.label"
        :label="`Try ${example.label}`"
        size="small"
        severity="secondary"
        outlined
        @click="searchExample(example.query)"
      />
      <Button
        label="Import audio bundle"
        size="small"
        severity="secondary"
        text
        @click="showAudioImportDialog = true"
      />
    </EmptyState>

    <LoadingState v-else-if="searching && !searchResults.length" message="Searching…" />

    <PageSection
      v-else-if="catalogMode && (searchResults.length || searching)"
      class="catalog-results"
      :class="{ 'catalog-results--loading': searching }"
      :aria-busy="searching ? 'true' : 'false'"
      variant="plain"
      title="Search results"
    >
      <template #meta>
        <Tag v-if="catalogPageRangeLabel" :value="catalogPageRangeLabel" severity="info" />
      </template>
      <template #actions>
        <div class="results-sort">
          <label class="results-sort__label" for="catalog-search-sort">Sort</label>
          <Select
            id="catalog-search-sort"
            v-model="sortBy"
            :options="sortOptions"
            optionLabel="label"
            optionValue="value"
            class="results-sort__dropdown"
            @change="onSortChange"
          />
        </div>
      </template>

      <div v-if="searching" class="catalog-results__status" aria-live="polite">
        Updating results…
      </div>

      <div class="catalog-card-grid page-card-grid">
        <ModelSearchResult
          v-for="result in sortedSearchResults"
          :key="result.id"
          :result="result"
          :ready="hasReadyCatalogEngine(result)"
          :has-token="modelStore.hasHuggingfaceToken"
          :engine-label="catalogEngineLabel"
          :variant-state="catalogVariantState(result)"
          @install="handleCatalogVariantAction(result, $event)"
          @configure="configureCatalogVariant(result, $event)"
          @refresh="refreshCatalogVariant(result, $event)"
          @browse="openCatalogVariantPicker(result)"
          @set-token="goToTokenSettings"
          @open-external="openExternal"
          @apply-projector="updateCatalogProjector(result, $event)"
          @apply-mtp="updateCatalogMtp(result, $event)"
          @apply-dflash="updateCatalogDflash(result, $event)"
          @apply-dspark="updateCatalogDspark(result, $event)"
          @set-projector="setCatalogProjector(result, $event.variant, $event.value)"
          @set-mtp="setCatalogMtp(result, $event.variant, $event.value)"
          @set-dflash="setCatalogDflash(result, $event.variant, $event.value)"
          @set-dspark="setCatalogDspark(result, $event.variant, $event.value)"
        />
      </div>

      <div v-if="catalogHasMore || catalogPage > 1" class="catalog-pagination">
        <Button
          label="Previous"
          icon="pi pi-chevron-left"
          severity="secondary"
          outlined
          :disabled="catalogPage <= 1 || searching"
          @click="changeCatalogPage(catalogPage - 1)"
        />
        <span>Page {{ catalogPage }} of {{ catalogTotalPages }}</span>
        <Button
          label="Next"
          icon="pi pi-chevron-right"
          iconPos="right"
          severity="secondary"
          outlined
          :disabled="!catalogHasMore || searching"
          @click="changeCatalogPage(catalogPage + 1)"
        />
      </div>
    </PageSection>

    <!-- Legacy result renderer remains as a compatibility adapter. -->
    <PageSection v-else class="results-table" variant="plain" title="Search results">
      <template #meta>
        <Tag
          :value="`${searchResults.length} result${searchResults.length !== 1 ? 's' : ''}`"
          severity="info"
        />
      </template>
      <template #actions>
        <div class="results-sort">
          <label class="results-sort__label" for="model-search-sort">Sort</label>
          <Select
            id="model-search-sort"
            v-model="sortBy"
            :options="sortOptions"
            optionLabel="label"
            optionValue="value"
            class="results-sort__dropdown"
          />
        </div>
      </template>

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
            <i
              :class="[
                'pi',
                expanded.has(result.modelId || result.id) ? 'pi-chevron-down' : 'pi-chevron-right',
              ]"
            />
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
              v-for="tag in (result.tags || []).filter((t) => interestingTag(t)).slice(0, 3)"
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

            <div v-else class="table-shell">
              <div class="table-shell__scroll">
                <table class="files-table">
                  <thead>
                    <tr>
                      <th>{{ searchFormat === 'gguf' ? 'Item' : 'Model' }}</th>
                      <th>Size</th>
                      <th v-if="searchFormat === 'gguf'">Shards</th>
                      <th v-if="searchFormat === 'gguf'">Projector</th>
                      <th v-if="searchFormat === 'gguf'">MTP</th>
                      <th v-if="searchFormat === 'gguf'">DFlash</th>
                      <th v-if="searchFormat === 'gguf'">DSpark</th>
                      <th>Status</th>
                      <th></th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr
                      v-for="file in getFiles(result.modelId || result.id)"
                      :key="file.key || file.filename"
                    >
                      <td class="file-name">
                        <code>{{ formatResultItemLabel(file, result) }}</code>
                        <span
                          v-if="
                            searchFormat === 'gguf' && file.kind === 'quant' && file.variantPrefix
                          "
                          class="file-subtext"
                        >
                          {{ file.variantPrefix }} variant
                        </span>
                        <span v-else-if="file.subtext" class="file-subtext">
                          {{ file.subtext }}
                        </span>
                      </td>
                      <td class="file-size">{{ formatBytes(file.size) }}</td>
                      <td v-if="searchFormat === 'gguf'" class="file-count">
                        {{ file.kind === 'quant' ? file.files?.length || 0 : 1 }}
                      </td>
                      <td v-if="searchFormat === 'gguf'" class="projector-cell">
                        <Select
                          v-if="file.kind === 'quant'"
                          :model-value="getSelectedProjector(result.modelId || result.id, file)"
                          :options="file.projectorOptions || [{ label: 'None', value: '' }]"
                          optionLabel="label"
                          optionValue="value"
                          class="projector-select"
                          :disabled="isFileDownloading(result.modelId || result.id, file)"
                          @update:model-value="
                            setSelectedProjector(result.modelId || result.id, file, $event)
                          "
                        />
                      </td>
                      <td v-if="searchFormat === 'gguf'" class="projector-cell">
                        <Select
                          v-if="
                            file.kind === 'quant' &&
                            (file.mtpOptions || []).some((opt) => opt.value)
                          "
                          :model-value="getSelectedMtp(result.modelId || result.id, file)"
                          :options="file.mtpOptions || [{ label: 'None', value: '' }]"
                          optionLabel="label"
                          optionValue="value"
                          class="projector-select"
                          :disabled="isFileDownloading(result.modelId || result.id, file)"
                          @update:model-value="
                            setSelectedMtp(result.modelId || result.id, file, $event)
                          "
                        />
                      </td>
                      <td v-if="searchFormat === 'gguf'" class="projector-cell">
                        <Select
                          v-if="
                            file.kind === 'quant' &&
                            (file.dflashOptions || []).some((opt) => opt.value)
                          "
                          :model-value="getSelectedDflash(result.modelId || result.id, file)"
                          :options="file.dflashOptions || [{ label: 'None', value: '' }]"
                          optionLabel="label"
                          optionValue="value"
                          class="projector-select"
                          :disabled="isFileDownloading(result.modelId || result.id, file)"
                          @update:model-value="
                            setSelectedDflash(result.modelId || result.id, file, $event)
                          "
                        />
                      </td>
                      <td v-if="searchFormat === 'gguf'" class="projector-cell">
                        <Select
                          v-if="
                            file.kind === 'quant' &&
                            (file.dsparkOptions || []).some((opt) => opt.value)
                          "
                          :model-value="getSelectedDspark(result.modelId || result.id, file)"
                          :options="file.dsparkOptions || [{ label: 'None', value: '' }]"
                          optionLabel="label"
                          optionValue="value"
                          class="projector-select"
                          :disabled="isFileDownloading(result.modelId || result.id, file)"
                          @update:model-value="
                            setSelectedDspark(result.modelId || result.id, file, $event)
                          "
                        />
                      </td>
                      <td class="file-status">
                        <Tag
                          v-if="isFileDownloading(result.modelId || result.id, file)"
                          value="Downloading"
                          severity="warning"
                        />
                        <Tag v-else-if="file.downloaded" value="Downloaded" severity="success" />
                        <Tag v-else value="Available" severity="warn" />
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
                            v-if="file.downloaded"
                            label="Refresh"
                            icon="pi pi-refresh"
                            size="small"
                            severity="secondary"
                            text
                            :loading="isFileDownloading(result.modelId || result.id, file)"
                            :disabled="isFileDownloading(result.modelId || result.id, file)"
                            @click="refreshDownloaded(result, file)"
                          />
                          <Button
                            v-if="
                              file.downloaded &&
                              searchFormat === 'gguf' &&
                              file.kind === 'quant' &&
                              hasProjectorSelectionChanged(result.modelId || result.id, file)
                            "
                            label="Apply projector"
                            icon="pi pi-save"
                            size="small"
                            severity="success"
                            outlined
                            :loading="isFileDownloading(result.modelId || result.id, file)"
                            @click="updateProjector(result, file)"
                          />
                          <Button
                            v-if="
                              file.downloaded &&
                              searchFormat === 'gguf' &&
                              file.kind === 'quant' &&
                              hasMtpSelectionChanged(result.modelId || result.id, file)
                            "
                            label="Apply MTP"
                            icon="pi pi-save"
                            size="small"
                            severity="success"
                            outlined
                            :loading="isFileDownloading(result.modelId || result.id, file)"
                            @click="updateMtp(result, file)"
                          />
                          <Button
                            v-if="
                              file.downloaded &&
                              searchFormat === 'gguf' &&
                              file.kind === 'quant' &&
                              hasDflashSelectionChanged(result.modelId || result.id, file)
                            "
                            label="Apply DFlash"
                            icon="pi pi-save"
                            size="small"
                            severity="success"
                            outlined
                            :loading="isFileDownloading(result.modelId || result.id, file)"
                            @click="updateDflash(result, file)"
                          />
                          <Button
                            v-if="
                              file.downloaded &&
                              searchFormat === 'gguf' &&
                              file.kind === 'quant' &&
                              hasDsparkSelectionChanged(result.modelId || result.id, file)
                            "
                            label="Apply DSpark"
                            icon="pi pi-save"
                            size="small"
                            severity="success"
                            outlined
                            :loading="isFileDownloading(result.modelId || result.id, file)"
                            @click="updateDspark(result, file)"
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
            </div>
          </div>
        </Transition>
      </div>
    </PageSection>

    <CatalogVariantPickerDialog
      v-model:visible="showCatalogVariantPicker"
      :result="variantPickerResult"
      :variant-state="pickerVariantState"
      @install="handleCatalogVariantAction(variantPickerResult, $event)"
      @configure="configureCatalogVariant(variantPickerResult, $event)"
      @refresh="refreshCatalogVariant(variantPickerResult, $event)"
      @apply-projector="updateCatalogProjector(variantPickerResult, $event)"
      @apply-mtp="updateCatalogMtp(variantPickerResult, $event)"
      @apply-dflash="updateCatalogDflash(variantPickerResult, $event)"
      @set-projector="setCatalogProjector(variantPickerResult, $event.variant, $event.value)"
      @set-mtp="setCatalogMtp(variantPickerResult, $event.variant, $event.value)"
      @set-dflash="setCatalogDflash(variantPickerResult, $event.variant, $event.value)"
    />

    <CatalogInstallOptionsDialog
      v-model:visible="showInstallOptionsDialog"
      :result="pendingCatalogInstall?.result"
      :variant="pendingCatalogInstall?.variant"
      @confirm="confirmCatalogInstall"
      @cancel="closeInstallOptions"
    />

    <HuggingFaceTokenDialog v-model:visible="showTokenDialog" />

    <AudioBundleImportDialog
      v-model:visible="showAudioImportDialog"
      :loading="audioImportSubmitting"
      @import="submitAudioImport"
    />
  </div>
</template>

<script setup>
import { ref, computed, onMounted, onUnmounted, watch, nextTick } from 'vue'
import { storeToRefs } from 'pinia'
import { useRoute, useRouter } from 'vue-router'
import { useToast } from 'primevue/usetoast'
import Button from 'primevue/button'
import Tag from 'primevue/tag'
import Select from 'primevue/select'
import { watchDialogFocus } from '@/composables/useFocusReturn'
import LoadingState from '@/components/common/LoadingState.vue'
import EmptyState from '@/components/common/EmptyState.vue'
import PageHeader from '@/components/common/PageHeader.vue'
import PageSection from '@/components/common/PageSection.vue'
import StateBanner from '@/components/common/StateBanner.vue'
import ModelSearchControls from '@/components/models/ModelSearchControls.vue'
import ModelSearchResult from '@/components/models/ModelSearchResult.vue'
import CatalogVariantPickerDialog from '@/components/models/CatalogVariantPickerDialog.vue'
import CatalogInstallOptionsDialog from '@/components/models/CatalogInstallOptionsDialog.vue'
import AudioBundleImportDialog from '@/components/models/AudioBundleImportDialog.vue'
import HuggingFaceTokenNotice from '@/components/models/HuggingFaceTokenNotice.vue'
import HuggingFaceTokenDialog from '@/components/models/HuggingFaceTokenDialog.vue'
import {
  catalogDflashOptions,
  catalogDsparkOptions,
  catalogMtpOptions,
  catalogProjectorOptions,
  formatBytes,
  formatContextLength,
  formatLanguageHint,
  formatNumber,
  formatSearchUpdatedAt,
  getDflashOptions,
  getDsparkOptions,
  getMtpOptions,
  getProjectorOptions,
  pipelineTagSeverity,
} from '@/utils/modelSearchPresentation'
import { useModelStore } from '@/stores/models'
import { ENGINE_DESCRIPTORS, engineLabel } from '@/engines/registry'
import { useEnginesStore } from '@/stores/engines'
import { useProgressStore } from '@/stores/progress'
import axios from 'axios'

const route = useRoute()
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
const filesCache = ref({}) // modelId -> files[]
const projectorSelections = ref({})
const mtpSelections = ref({})
const dflashSelections = ref({})
const dsparkSelections = ref({})
const catalogMode = ref(import.meta.env.MODE !== 'test')
const catalogActionKey = ref(null)
const catalogDownloadingKeys = ref(new Set())
const catalogInstallingKeys = ref(new Set())
const catalogInstallTaskIds = ref(new Map())
const showInstallOptionsDialog = ref(false)
const pendingCatalogInstall = ref(null)
const showAudioImportDialog = ref(false)
const variantPickerResult = ref(null)
const syncingToRoute = ref(false)
const showCatalogVariantPicker = computed({
  get: () => variantPickerResult.value != null,
  set: (visible) => {
    if (!visible) variantPickerResult.value = null
  },
})
const audioImportSubmitting = ref(false)
const catalogPageSize = 20
const filtersOpen = ref(false)
const showTokenDialog = ref(false)
watchDialogFocus(showTokenDialog)
watchDialogFocus(showInstallOptionsDialog)
watchDialogFocus(showAudioImportDialog)
watchDialogFocus(showCatalogVariantPicker)
const tokenNoticeDismissed = ref(
  typeof sessionStorage !== 'undefined' &&
    sessionStorage.getItem('llama-studio.token-notice') === '1',
)
const searchError = ref('')
const engineFilter = ref(null)
const taskFilter = ref(null)
const inputModalityFilter = ref(null)
const outputModalityFilter = ref(null)
const providerFilter = ref(null)
const installMethodFilter = ref(null)

const formatOptions = [
  { label: 'All packages', value: 'all' },
  { label: 'GGUF', value: 'gguf' },
  { label: 'Safetensors', value: 'safetensors' },
  { label: 'Prepared audio', value: 'mixed' },
]

function catalogEngineLabel(id) {
  return engineFilterOptions.value.find((engine) => engine.value === id)?.label || id
}
function hasReadyCatalogEngine(result) {
  return (enginesStore.engineDescriptors || []).some(
    (engine) =>
      engine.runnable &&
      engine.enabled !== false &&
      (result.compatible_engines || []).includes(engine.id),
  )
}
const engineFilterOptions = computed(() => {
  const descriptors = enginesStore.engineDescriptors || []
  if (descriptors.length) {
    return descriptors.map((engine) => ({ value: engine.id, label: engine.label }))
  }
  return Object.entries(ENGINE_DESCRIPTORS).map(([id, descriptor]) => ({
    value: id,
    label: descriptor.label || id,
  }))
})
const TASK_LABELS = {
  tts: 'Speech',
  asr: 'Transcribe',
  vad: 'Voice activity',
  diar: 'Diarization',
  sep: 'Separation',
  gen: 'Music',
  vc: 'Voice conversion',
  s2s: 'Speech to speech',
  align: 'Alignment',
  clon: 'Voice clone',
  vdes: 'Voice design',
  spk: 'Speaker',
  svc: 'Singing voice',
  'text-generation': 'Text',
  'text2text-generation': 'Text to text',
  embeddings: 'Embeddings',
}
const taskFilterOptions = computed(() => {
  const values = catalogFacets.value?.tasks?.length
    ? catalogFacets.value.tasks
    : Object.keys(TASK_LABELS)
  return values.map((value) => ({ value, label: TASK_LABELS[value] || value }))
})
const inputModalityOptions = computed(
  () => catalogFacets.value?.input_modalities || ['text', 'audio', 'image'],
)
const outputModalityOptions = computed(
  () =>
    catalogFacets.value?.output_modalities || ['text', 'audio', 'segments', 'events', 'embedding'],
)
const providerOptions = [
  { label: 'Hugging Face', value: 'huggingface' },
  { label: `${engineLabel('audio_cpp')} packages`, value: 'audio_cpp' },
]
const INSTALL_METHOD_LABELS = {
  direct: 'Direct HF',
  composite: 'Assemble',
  converter: 'Convert',
  bundled: 'Bundled',
  unavailable: 'Unavailable',
}
const installMethodOptions = computed(() => {
  const values = catalogFacets.value?.install_methods || []
  return values.map((value) => ({
    value,
    label: INSTALL_METHOD_LABELS[value] || value,
  }))
})

const showFormatSelect = computed(() => !engineFilter.value)

const unavailableProviders = computed(() =>
  Object.entries(catalogProviderStatus.value || {})
    .filter(([, status]) => status?.available === false)
    .map(([provider, status]) => ({ provider, status })),
)

const unavailableProviderMessage = computed(() =>
  unavailableProviders.value
    .map(
      (item) =>
        `${item.provider} unavailable${item.status.reason ? `: ${item.status.reason}` : ''}`,
    )
    .join(' · '),
)

const resultCountLabel = computed(() => {
  if (!hasSearched.value) return ''
  const total = catalogTotal.value || searchResults.value.length
  if (!total) return ''
  return `${total} result${total === 1 ? '' : 's'}`
})

const searchExamples = [
  { label: 'Qwen3', query: 'Qwen3' },
  { label: 'Llama 3', query: 'Llama-3' },
  { label: 'Whisper', query: 'Whisper' },
]

function searchExample(term) {
  query.value = term
  runSearch()
}

function refreshSearch() {
  if (hasSearched.value) runSearch()
}

const catalogTotalPages = computed(() =>
  Math.max(1, Math.ceil((catalogTotal.value || 0) / catalogPageSize)),
)

const catalogPageRangeLabel = computed(() => {
  const shown = searchResults.value.length
  if (!shown) return ''
  const total = catalogTotal.value || shown
  if (!catalogHasMore.value && catalogPage.value <= 1) {
    return `${total} result${total === 1 ? '' : 's'}`
  }
  const start = (catalogPage.value - 1) * catalogPageSize + 1
  const end = start + shown - 1
  if (catalogHasMore.value) return `${start}–${end}, more available`
  return `${start}–${end} of ${total}`
})

/** Default: downloads high → low */
const sortBy = ref('relevance')

const sortOptions = [
  { label: 'Best match', value: 'relevance' },
  { label: 'Downloads (high → low)', value: 'downloads_desc' },
  { label: 'Downloads (low → high)', value: 'downloads_asc' },
  { label: 'Likes (high → low)', value: 'likes_desc' },
  { label: 'Likes (low → high)', value: 'likes_asc' },
  { label: 'Name A–Z', value: 'name_asc' },
  { label: 'Name Z–A', value: 'name_desc' },
]

function modelIdKey(r) {
  return String(r.display_name || r.modelId || r.id || '').toLowerCase()
}

function numOrMissing(v, missingSentinel) {
  const n = typeof v === 'number' ? v : Number(v)
  return Number.isFinite(n) ? n : missingSentinel
}

const sortedSearchResults = computed(() => {
  if (catalogMode.value) return searchResults.value
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
    ...(installMethodFilter.value ? { install_method: installMethodFilter.value } : {}),
    ...(!engineFilter.value && searchFormat.value && searchFormat.value !== 'all'
      ? { artifact_format: searchFormat.value }
      : {}),
  }
}

function buildSearchRouteQuery() {
  const next = {}
  const trimmed = query.value.trim()
  if (trimmed) next.q = trimmed
  if (!engineFilter.value && searchFormat.value && searchFormat.value !== 'all') {
    next.format = searchFormat.value
  }
  if (engineFilter.value) next.engine = engineFilter.value
  if (taskFilter.value) next.task = taskFilter.value
  if (inputModalityFilter.value) next.input = inputModalityFilter.value
  if (outputModalityFilter.value) next.output = outputModalityFilter.value
  if (providerFilter.value) next.provider = providerFilter.value
  if (installMethodFilter.value) next.install_method = installMethodFilter.value
  if (sortBy.value && sortBy.value !== 'relevance') next.sort = sortBy.value
  if (catalogPage.value > 1) next.page = String(catalogPage.value)
  return next
}

function routeQueryEqual(a, b) {
  const keys = new Set([...Object.keys(a || {}), ...Object.keys(b || {})])
  for (const key of keys) {
    if (String(a?.[key] ?? '') !== String(b?.[key] ?? '')) return false
  }
  return true
}

async function syncSearchToRoute() {
  const next = buildSearchRouteQuery()
  if (routeQueryEqual(next, route.query)) return
  syncingToRoute.value = true
  try {
    await router.replace({ name: 'search', query: next })
  } finally {
    await nextTick()
    syncingToRoute.value = false
  }
}

function applySearchFromRoute(queryObj = route.query) {
  const nextQuery = typeof queryObj.q === 'string' ? queryObj.q : ''
  query.value = nextQuery
  searchFormat.value =
    typeof queryObj.format === 'string' && queryObj.format ? queryObj.format : 'all'
  engineFilter.value = typeof queryObj.engine === 'string' ? queryObj.engine : null
  taskFilter.value = typeof queryObj.task === 'string' ? queryObj.task : null
  inputModalityFilter.value = typeof queryObj.input === 'string' ? queryObj.input : null
  outputModalityFilter.value = typeof queryObj.output === 'string' ? queryObj.output : null
  providerFilter.value = typeof queryObj.provider === 'string' ? queryObj.provider : null
  installMethodFilter.value =
    typeof queryObj.install_method === 'string' ? queryObj.install_method : null
  sortBy.value = typeof queryObj.sort === 'string' && queryObj.sort ? queryObj.sort : 'relevance'
}

function onFormatChange() {
  if (hasSearched.value) runSearch()
}

function onEngineFilterChange() {
  if (hasSearched.value) runSearch()
}

function onSortChange() {
  runSearch()
}

const secondaryFilterCount = computed(() => {
  let count = 0
  if (engineFilter.value) count += 1
  if (inputModalityFilter.value) count += 1
  if (outputModalityFilter.value) count += 1
  if (providerFilter.value) count += 1
  if (installMethodFilter.value) count += 1
  if (searchFormat.value && searchFormat.value !== 'all') count += 1
  return count
})

async function search(page = 1, { syncRoute = true } = {}) {
  searchError.value = ''
  const pageNum = Number(page)
  const safePage = Number.isFinite(pageNum) && pageNum >= 1 ? Math.floor(pageNum) : 1
  expanded.value = new Set()
  filesCache.value = {}
  projectorSelections.value = {}
  mtpSelections.value = {}
  dflashSelections.value = {}
  variantPickerResult.value = null
  try {
    catalogMode.value = true
    const data = await modelStore.searchCatalog(query.value.trim(), {
      page: safePage,
      page_size: catalogPageSize,
      filters: catalogFiltersPayload(),
      sort: sortBy.value,
    })
    if (data == null) return
    notifyUnavailableProviders(data?.provider_status)
    enrichVisibleCatalogSizes()
    if (syncRoute) await syncSearchToRoute()
  } catch (e) {
    searchError.value = e?.response?.data?.detail || e.message || 'Search failed'
    toast.add({
      severity: 'error',
      summary: 'Search failed',
      detail: searchError.value,
      life: 4000,
    })
  }
}

function notifyUnavailableProviders(statuses = {}) {
  if (!searchResults.value.length) {
    for (const [provider, status] of Object.entries(statuses)) {
      if (status?.available === false && status?.reason) {
        toast.add({
          severity: 'warn',
          summary: `${provider} search unavailable`,
          detail: status.reason,
          life: 5000,
        })
      }
    }
  }
}

function runSearch() {
  return search(1)
}

async function changeCatalogPage(page) {
  await search(page)
}

function goToTokenSettings() {
  showTokenDialog.value = true
}

function dismissTokenNotice() {
  tokenNoticeDismissed.value = true
  try {
    sessionStorage.setItem('llama-studio.token-notice', '1')
  } catch {
    /* ignore */
  }
}

function resetSecondaryFilters() {
  taskFilter.value = null
  engineFilter.value = null
  inputModalityFilter.value = null
  outputModalityFilter.value = null
  providerFilter.value = null
  installMethodFilter.value = null
  searchFormat.value = 'all'
  runSearch()
}

function clearAllSearchFilters() {
  resetSecondaryFilters()
}

function openExternal(url) {
  if (!url) return
  window.open(url, '_blank', 'noopener,noreferrer')
}

function catalogHfId(result) {
  if (result?.provider === 'huggingface') {
    return result.source?.id || String(result.provider_item_id || '').split(':')[0] || ''
  }
  return result?.modelId || result?.id || ''
}

function resultMatchesHfId(result, hfId) {
  if (!hfId) return false
  if ((result.modelId || result.id) === hfId) return true
  if (result.provider === 'huggingface') {
    return (
      result.source?.id === hfId || String(result.provider_item_id || '').startsWith(`${hfId}:`)
    )
  }
  return false
}

function findCatalogDownloadedModel(result, variant) {
  if (!result || !variant) return null
  if (result.provider === 'huggingface') {
    const hfId = catalogHfId(result)
    if (!hfId) return null
    if (result.artifact_format === 'safetensors') {
      return findDownloadedSafetensorsBundle(hfId) || null
    }
    return (
      modelStore.allQuantizations.find(
        (model) =>
          model.huggingface_id === hfId &&
          (model.quantization === variant.id ||
            model.quantization === variant.label ||
            (Array.isArray(variant.files) &&
              variant.files.some((file) => {
                const name = typeof file === 'string' ? file : file?.filename
                return name && model.filename === name
              }))),
      ) || null
    )
  }
  if (result.provider === 'audio_cpp') {
    const packageId = catalogAudioPackageId(result, variant)
    const safePackageId =
      packageId
        .replace(/[^a-zA-Z0-9._-]+/g, '-')
        .replace(/-{2,}/g, '-')
        .replace(/^[-._]+|[-._]+$/g, '')
        .slice(0, 96) || 'audio-model'
    const recordIds = new Set([packageId, `audio-cpp--${packageId}`, `audio-cpp--${safePackageId}`])
    return (
      modelStore.allQuantizations.find(
        (model) =>
          recordIds.has(model.id) ||
          recordIds.has(model.model_id) ||
          model.huggingface_id === packageId ||
          model.package_id === packageId ||
          model?.source?.id === packageId ||
          model?.source?.package_id === packageId ||
          model?.artifact?.package_id === packageId ||
          model?.manifest?.package_id === packageId,
      ) || null
    )
  }
  return null
}

function catalogNeedsInstallOptions(variant, result) {
  // Always confirm audio.cpp installs so family override is available.
  // Converter packages also collect optional/required local inputs here.
  if (result?.provider === 'audio_cpp') return true
  return Boolean(variant?.external_inputs_required || variant?.external_inputs_optional)
}

function configureCatalogVariant(result, variant) {
  const model = findCatalogDownloadedModel(result, variant)
  if (!model) return
  router.push(`/models/${encodeURIComponent(model.id || model.model_id)}/config`)
}

async function runModelRefresh(modelId, downloadKey, { catalog = false } = {}) {
  if (!modelId) return
  const busySet = catalog ? catalogDownloadingKeys : downloadingFiles
  busySet.value.add(downloadKey)
  busySet.value = new Set(busySet.value)
  try {
    const response = await modelStore.refreshModel(modelId)
    if (!response?.updated) {
      busySet.value.delete(downloadKey)
      busySet.value = new Set(busySet.value)
      toast.add({
        severity: 'info',
        summary: 'Up to date',
        detail: response?.message || 'Already up to date',
        life: 3000,
      })
      return
    }
    toast.add({
      severity: 'success',
      summary: 'Update started',
      detail: response?.message || 'Track progress in notifications',
      life: 3000,
    })
  } catch (e) {
    busySet.value.delete(downloadKey)
    busySet.value = new Set(busySet.value)
    toast.add({
      severity: 'error',
      summary: 'Refresh failed',
      detail: e?.response?.data?.detail || e.message,
      life: 4000,
    })
  }
}

async function refreshCatalogVariant(result, variant) {
  const model = findCatalogDownloadedModel(result, variant)
  if (!model?.id && !model?.model_id) return
  const hfId = catalogHfId(result)
  const downloadKey = `${hfId}:${variant.id}`
  await runModelRefresh(model.id || model.model_id, downloadKey, { catalog: true })
}

async function refreshDownloaded(result, file) {
  const repoId = result.modelId || result.id
  const downloaded =
    searchFormat.value === 'safetensors'
      ? findDownloadedSafetensorsBundle(repoId)
      : findDownloadedQuantization(repoId, file, file.files || [])
  const modelId =
    downloaded?.id ||
    downloaded?.model_id ||
    (searchFormat.value === 'safetensors' && repoId ? String(repoId).replaceAll('/', '--') : null)
  if (!modelId) {
    toast.add({
      severity: 'warn',
      summary: 'Refresh unavailable',
      detail: 'Could not resolve the library model for this file',
      life: 3000,
    })
    return
  }
  await runModelRefresh(modelId, getDownloadKey(repoId, file), { catalog: false })
}

function isCatalogVariantBusy(result, variant) {
  if (result?.provider === 'huggingface') {
    return isCatalogVariantDownloading(result, variant)
  }
  const key = `${result.id}:${variant.id}`
  return (
    catalogInstallingKeys.value.has(key) || catalogAudioVariantHasActiveInstallTask(result, variant)
  )
}

function catalogAudioPackageId(result, variant) {
  return String(
    result?.source?.package_id ||
      result?.source?.id ||
      result?.provider_item_id ||
      variant?.id ||
      result?.id ||
      '',
  ).trim()
}

function catalogAudioVariantHasActiveInstallTask(result, variant) {
  const packageId = catalogAudioPackageId(result, variant)
  if (!packageId) return false
  return Object.values(progressTasks.value).some((task) => {
    if (task?.type !== 'audio_model_install') return false
    if (!['queued', 'running', 'cancelling'].includes(task.status)) return false
    return String(task.metadata?.package_id || '') === packageId
  })
}

function rememberCatalogInstall(key, taskId) {
  const pending = new Set(catalogInstallingKeys.value)
  pending.add(key)
  catalogInstallingKeys.value = pending
  if (!taskId) return
  const taskIds = new Map(catalogInstallTaskIds.value)
  taskIds.set(key, String(taskId))
  catalogInstallTaskIds.value = taskIds
}

function forgetCatalogInstall(key) {
  const pending = new Set(catalogInstallingKeys.value)
  pending.delete(key)
  catalogInstallingKeys.value = pending
  const taskIds = new Map(catalogInstallTaskIds.value)
  taskIds.delete(key)
  catalogInstallTaskIds.value = taskIds
}

function catalogVariantFilenames(variant) {
  return normalizeVariantFiles(variant?.files || [])
    .map((file) => file.filename)
    .filter(Boolean)
}

function catalogVariantsNeedSizes(result) {
  const variants = result?.install_variants || []
  if (!variants.length) return false
  return variants.some(
    (variant) => !variant.size_bytes && catalogVariantFilenames(variant).length > 0,
  )
}

async function enrichCatalogVariantSizes(result) {
  if (!result || result.provider !== 'huggingface') return
  if (!catalogVariantsNeedSizes(result)) return
  const hfId = catalogHfId(result)
  if (!hfId) return

  const variants = result.install_variants || []
  const filenames = [...new Set(variants.flatMap(catalogVariantFilenames))]
  if (!filenames.length) return

  try {
    const { data } = await axios.get(`/api/models/search/${encodeURIComponent(hfId)}/file-sizes`, {
      params: { filenames: filenames.join(',') },
    })
    const sizes = data.sizes || {}
    result.install_variants = variants.map((variant) => {
      const names = catalogVariantFilenames(variant)
      const sizeBytes = names.reduce((sum, name) => sum + (Number(sizes[name]) || 0), 0)
      return {
        ...variant,
        size_bytes: sizeBytes || variant.size_bytes || null,
      }
    })
    const total = (result.install_variants || []).reduce(
      (sum, variant) => sum + (Number(variant.size_bytes) || 0),
      0,
    )
    if (total > 0) result.size_bytes = total
    // Keep Pinia searchResults / catalogResults in sync when this object is shared.
    searchResults.value = [...searchResults.value]
  } catch {
    // Keep the empty size hints returned by the search API.
  }
}

function enrichVisibleCatalogSizes() {
  for (const result of searchResults.value) {
    const variants = result?.install_variants || []
    // Inline single-variant cards need sizes without opening the picker.
    if (variants.length <= 1) {
      void enrichCatalogVariantSizes(result)
    }
  }
}

async function openCatalogVariantPicker(result) {
  variantPickerResult.value = result
  await enrichCatalogVariantSizes(result)
}

/**
 * Resolve the per-variant state the result card and picker rows render from.
 * Store-backed data and selection maps stay here so the components only render.
 */
function catalogVariantState(result) {
  const state = {}
  for (const variant of result?.install_variants || []) {
    state[variant.id] = {
      downloaded: findCatalogDownloadedModel(result, variant),
      busy: isCatalogVariantBusy(result, variant),
      projector: getCatalogProjector(result, variant),
      mtp: getCatalogMtp(result, variant),
      dflash: getCatalogDflash(result, variant),
      dspark: getCatalogDspark(result, variant),
    }
  }
  return state
}

const pickerVariantState = computed(() => catalogVariantState(variantPickerResult.value))

function handleCatalogVariantAction(result, variant) {
  if (result?.provider === 'huggingface') {
    return downloadHfCatalogVariant(result, variant)
  }
  return installCatalogVariant(result, variant)
}

function normalizeVariantFiles(files = []) {
  return files.map((file) => (typeof file === 'string' ? { filename: file, size: 0 } : file))
}

function catalogProjectorFile(result, variant) {
  return {
    quantizationKey: variant.id,
    projectorOptions: catalogProjectorOptions(result),
  }
}

function catalogMtpFile(result, variant) {
  return {
    quantizationKey: variant.id,
    mtpOptions: catalogMtpOptions(result),
  }
}

function catalogDflashFile(result, variant) {
  return {
    quantizationKey: variant.id,
    dflashOptions: catalogDflashOptions(result),
  }
}

function catalogDsparkFile(result, variant) {
  return {
    quantizationKey: variant.id,
    dsparkOptions: catalogDsparkOptions(result),
  }
}

function getCatalogProjector(result, variant) {
  const hfId = catalogHfId(result)
  const file = catalogProjectorFile(result, variant)
  const key = getProjectorSelectionKey(hfId, file)
  if (Object.prototype.hasOwnProperty.call(projectorSelections.value, key)) {
    return projectorSelections.value[key]
  }
  const downloaded = findCatalogDownloadedModel(result, variant)
  if (downloaded) {
    return downloaded.mmproj_filename || ''
  }
  return getDefaultProjectorValue(file)
}

function setCatalogProjector(result, variant, value) {
  setSelectedProjector(catalogHfId(result), catalogProjectorFile(result, variant), value)
}

function getCatalogMtp(result, variant) {
  const hfId = catalogHfId(result)
  const file = catalogMtpFile(result, variant)
  const key = getMtpSelectionKey(hfId, file)
  if (Object.prototype.hasOwnProperty.call(mtpSelections.value, key)) {
    return mtpSelections.value[key]
  }
  const downloaded = findCatalogDownloadedModel(result, variant)
  if (downloaded) {
    return downloaded.mtp_filename || ''
  }
  return getDefaultMtpValue(file)
}

function setCatalogMtp(result, variant, value) {
  setSelectedMtp(catalogHfId(result), catalogMtpFile(result, variant), value)
  if (value) {
    setSelectedDflash(catalogHfId(result), catalogDflashFile(result, variant), '')
  }
}

function getCatalogDflash(result, variant) {
  const hfId = catalogHfId(result)
  const file = catalogDflashFile(result, variant)
  const key = getDflashSelectionKey(hfId, file)
  if (Object.prototype.hasOwnProperty.call(dflashSelections.value, key)) {
    return dflashSelections.value[key]
  }
  const downloaded = findCatalogDownloadedModel(result, variant)
  if (downloaded) {
    return downloaded.dflash_filename || ''
  }
  return getDefaultDflashValue(file)
}

function getCatalogDspark(result, variant) {
  const hfId = catalogHfId(result)
  const file = catalogDsparkFile(result, variant)
  const key = getDsparkSelectionKey(hfId, file)
  if (Object.prototype.hasOwnProperty.call(dsparkSelections.value, key)) {
    return dsparkSelections.value[key]
  }
  const downloaded = findCatalogDownloadedModel(result, variant)
  if (downloaded) {
    return downloaded.dspark_filename || ''
  }
  return getDefaultDsparkValue(file)
}

function setCatalogDspark(result, variant, value) {
  setSelectedDspark(catalogHfId(result), catalogDsparkFile(result, variant), value)
}

function setCatalogDflash(result, variant, value) {
  setSelectedDflash(catalogHfId(result), catalogDflashFile(result, variant), value)
  if (value) {
    setSelectedMtp(catalogHfId(result), catalogMtpFile(result, variant), '')
  }
}

function catalogVariantHasActiveDownloadTask(result, variant) {
  const hfId = catalogHfId(result)
  if (!hfId || result.provider !== 'huggingface') return false
  const downloaded = findCatalogDownloadedModel(result, variant)
  return Object.values(progressTasks.value).some((task) => {
    if (task?.type !== 'download' || task.status !== 'running') return false
    const meta = task.metadata || {}
    if (meta.huggingface_id !== hfId) return false
    if (result.artifact_format === 'safetensors') {
      return !meta.quantization && !meta.filename
    }
    if (downloaded && meta.model_id && meta.model_id === (downloaded.id || downloaded.model_id)) {
      return true
    }
    const quantKey = variant.id
    return (
      meta.quantization === quantKey ||
      meta.quantization === result.metadata?.raw?.quantizations?.[quantKey]?.quantization
    )
  })
}

function isCatalogVariantDownloading(result, variant) {
  const hfId = catalogHfId(result)
  if (!hfId || result.provider !== 'huggingface') return false
  const key = `${hfId}:${variant.id}`
  if (catalogDownloadingKeys.value.has(key)) return true
  return catalogVariantHasActiveDownloadTask(result, variant)
}

async function downloadHfCatalogVariant(result, variant) {
  const hfId = catalogHfId(result)
  if (!hfId) return
  const actionKey = `${result.id}:${variant.id}`
  const downloadKey = `${hfId}:${variant.id}`
  catalogActionKey.value = actionKey
  catalogDownloadingKeys.value.add(downloadKey)
  catalogDownloadingKeys.value = new Set(catalogDownloadingKeys.value)
  try {
    const raw = result.metadata?.raw || {}
    const pipelineTag = result.metadata?.pipeline_tag || result.tasks?.[0] || null
    const artifactFormat = result.artifact_format || searchFormat.value

    if (artifactFormat === 'safetensors') {
      const files = normalizeVariantFiles(variant.files || raw.safetensors_files || [])
      const response = await modelStore.downloadSafetensorsBundle(hfId, files)
      rememberDownloadTask(response?.task_id)
    } else {
      const quantMeta = raw.quantizations?.[variant.id] || {}
      const files = quantMeta.files?.length
        ? quantMeta.files
        : normalizeVariantFiles(variant.files || [])
      const projectorFile = catalogProjectorFile(result, variant)
      const selectedProjector = getCatalogProjector(result, variant) || ''
      const projectorOption = getSelectedProjectorOption(projectorFile, selectedProjector)
      const mtpFile = catalogMtpFile(result, variant)
      const selectedMtp = getCatalogMtp(result, variant) || ''
      const mtpOption = getSelectedMtpOption(mtpFile, selectedMtp)
      const dflashFile = catalogDflashFile(result, variant)
      const selectedDflash = getCatalogDflash(result, variant) || ''
      const dflashOption = getSelectedDflashOption(dflashFile, selectedDflash)
      const dsparkFile = catalogDsparkFile(result, variant)
      const selectedDspark = getCatalogDspark(result, variant) || ''
      const dsparkOption = getSelectedDsparkOption(dsparkFile, selectedDspark)
      await modelStore.downloadGgufBundle(
        hfId,
        variant.id,
        files,
        pipelineTag,
        selectedProjector || null,
        projectorOption?.size || 0,
        selectedMtp || null,
        mtpOption?.size || 0,
        selectedDflash || null,
        dflashOption?.size || 0,
        selectedDspark || null,
        dsparkOption?.size || 0,
      )
    }
    toast.add({
      severity: 'success',
      summary: 'Download started',
      detail: 'Track progress in notifications',
      life: 3000,
    })
    await modelStore.fetchModels()
    if (artifactFormat === 'safetensors') {
      await modelStore.fetchSafetensorsModels()
    }
  } catch (e) {
    catalogDownloadingKeys.value.delete(downloadKey)
    catalogDownloadingKeys.value = new Set(catalogDownloadingKeys.value)
    toast.add({
      severity: 'error',
      summary: 'Download failed',
      detail: e?.response?.data?.detail || e.message,
      life: 4000,
    })
  } finally {
    catalogActionKey.value = null
  }
}

async function installCatalogVariant(result, variant) {
  if (catalogNeedsInstallOptions(variant, result)) {
    pendingCatalogInstall.value = { result, variant }
    showInstallOptionsDialog.value = true
    return
  }
  await startCatalogInstall(result, variant)
}

async function startCatalogInstall(result, variant, options = {}) {
  const key = `${result.id}:${variant.id}`
  rememberCatalogInstall(key)
  try {
    const response = await modelStore.installCatalogModel(result, variant, options)
    if (response?.task_id) {
      const task = progressTasks.value[String(response.task_id)]
      const status = String(task?.status || '')
      if (['completed', 'failed', 'cancelled', 'canceled'].includes(status)) {
        forgetCatalogInstall(key)
        if (status === 'completed') await refreshModelSearchState()
      } else {
        rememberCatalogInstall(key, response.task_id)
      }
    } else {
      // The backend normally supplies a task ID. Do not leave an unavailable task
      // showing as active if an older backend response omits it.
      forgetCatalogInstall(key)
    }
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
    forgetCatalogInstall(key)
  }
}

function closeInstallOptions() {
  pendingCatalogInstall.value = null
}

async function confirmCatalogInstall(options = {}) {
  const pending = pendingCatalogInstall.value
  if (!pending) return
  showInstallOptionsDialog.value = false
  pendingCatalogInstall.value = null
  await startCatalogInstall(pending.result, pending.variant, options)
}

async function submitAudioImport(payload = {}) {
  if (!payload.source_path) return
  audioImportSubmitting.value = true
  try {
    const response = await modelStore.importAudioBundle(payload.source_path, {
      ...(payload.package_id ? { package_id: payload.package_id } : {}),
      ...(payload.family ? { family: payload.family } : {}),
    })
    toast.add({
      severity: 'success',
      summary: 'Import started',
      detail: response?.message || 'The bundle will be inspected before it is added.',
      life: 3500,
    })
    showAudioImportDialog.value = false
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

const activeFilterChips = computed(() => {
  const chips = []
  if (engineFilter.value) {
    const match = engineFilterOptions.value.find((option) => option.value === engineFilter.value)
    chips.push({
      key: 'engine',
      label: match?.label || engineFilter.value,
      clear: () => {
        engineFilter.value = null
        runSearch()
      },
    })
  }
  if (taskFilter.value) {
    const match = taskFilterOptions.value.find((option) => option.value === taskFilter.value)
    chips.push({
      key: 'task',
      label: match?.label || taskFilter.value,
      clear: () => {
        taskFilter.value = null
        runSearch()
      },
    })
  }
  if (searchFormat.value && searchFormat.value !== 'all') {
    const match = formatOptions.find((option) => option.value === searchFormat.value)
    chips.push({
      key: 'format',
      label: match?.label || searchFormat.value,
      clear: () => {
        searchFormat.value = 'all'
        runSearch()
      },
    })
  }
  if (inputModalityFilter.value) {
    chips.push({
      key: 'input',
      label: `${inputModalityFilter.value} in`,
      clear: () => {
        inputModalityFilter.value = null
        runSearch()
      },
    })
  }
  if (outputModalityFilter.value) {
    chips.push({
      key: 'output',
      label: `${outputModalityFilter.value} out`,
      clear: () => {
        outputModalityFilter.value = null
        runSearch()
      },
    })
  }
  if (providerFilter.value) {
    const match = providerOptions.find((option) => option.value === providerFilter.value)
    chips.push({
      key: 'provider',
      label: match?.label || providerFilter.value,
      clear: () => {
        providerFilter.value = null
        runSearch()
      },
    })
  }
  return chips
})

function clearSearchResults() {
  modelStore.clearSearchState()
  expanded.value = new Set()
  filesCache.value = {}
  projectorSelections.value = {}
  mtpSelections.value = {}
  dflashSelections.value = {}
  variantPickerResult.value = null
  syncingToRoute.value = true
  router.replace({ name: 'search', query: {} }).finally(async () => {
    await nextTick()
    syncingToRoute.value = false
  })
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
    const result = searchResults.value.find((r) => (r.modelId || r.id) === modelId)
    if (!result) return

    let files = []
    if (searchFormat.value === 'gguf') {
      const projectorOptions = getProjectorOptions(result.mmproj_files || [])
      const mtpOptions = getMtpOptions(result.mtp_files || [])
      const dflashOptions = getDflashOptions(result.dflash_files || [])
      const dsparkOptions = getDsparkOptions(result.dspark_files || [])
      const quantEntries = Object.entries(result.quantizations || {}).map(([key, entry]) => ({
        key,
        kind: 'quant',
        quantizationKey: key,
        quantization: entry.quantization || '',
        variantPrefix: entry.variant_prefix || '',
        size: entry.total_size || 0,
        projectorOptions,
        mtpOptions,
        dflashOptions,
        dsparkOptions,
        files: (entry.files || []).map((f) => ({
          filename: f.filename,
          size: f.size || 0,
        })),
      }))

      const allFiles = quantEntries.flatMap((entry) => entry.files)
      if (allFiles.length) {
        try {
          const filenames = allFiles.map((f) => f.filename).join(',')
          const { data } = await axios.get(
            `/api/models/search/${encodeURIComponent(modelId)}/file-sizes`,
            {
              params: { filenames },
            },
          )
          const sizes = data.sizes || {}
          files = quantEntries
            .map((entry) => {
              const resolvedFiles = entry.files.map((f) => ({
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
            })
            .sort((a, b) => (a.size || 0) - (b.size || 0))
        } catch {
          files = quantEntries
            .map((entry) => {
              const downloaded = findDownloadedQuantization(modelId, entry, entry.files)
              return { ...entry, downloaded, modelId: downloaded?.id }
            })
            .sort((a, b) => (a.size || 0) - (b.size || 0))
        }
      }
      files.forEach((entry) => {
        if (entry.kind !== 'quant') return
        ensureProjectorSelection(modelId, entry, entry.downloaded?.mmproj_filename || '')
        ensureMtpSelection(modelId, entry, entry.downloaded?.mtp_filename || '')
        ensureDflashSelection(modelId, entry, entry.downloaded?.dflash_filename || '')
        ensureDsparkSelection(modelId, entry, entry.downloaded?.dspark_filename || '')
      })
    } else {
      const stFiles = result.safetensors_files || []
      let resolvedFiles = stFiles.map((file) => ({ filename: file.filename, size: file.size || 0 }))
      if (resolvedFiles.length) {
        try {
          const filenames = resolvedFiles.map((file) => file.filename).join(',')
          const { data } = await axios.get(
            `/api/models/search/${encodeURIComponent(modelId)}/file-sizes`,
            {
              params: { filenames },
            },
          )
          const sizes = data.sizes || {}
          resolvedFiles = resolvedFiles.map((file) => ({
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
        ? [
            {
              key: 'safetensors-bundle',
              kind: 'safetensors-bundle',
              filename: result.modelId || result.id,
              size: totalSize,
              files: resolvedFiles,
              downloaded: downloadedBundle,
              modelId: downloadedBundle?.model_id,
              subtext: `${resolvedFiles.length} file${resolvedFiles.length === 1 ? '' : 's'}`,
            },
          ]
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
      const selectedMtp = getSelectedMtp(modelId, file)
      const selectedMtpOption = getSelectedMtpOption(file, selectedMtp)
      const selectedDflash = getSelectedDflash(modelId, file)
      const selectedDflashOption = getSelectedDflashOption(file, selectedDflash)
      await modelStore.downloadGgufBundle(
        modelId,
        file.quantizationKey || file.quantization,
        file.files || [],
        result.pipeline_tag || null,
        selectedProjector || null,
        selectedProjectorOption?.size || 0,
        selectedMtp || null,
        selectedMtpOption?.size || 0,
        selectedDflash || null,
        selectedDflashOption?.size || 0,
      )
    } else if (searchFormat.value === 'safetensors') {
      await modelStore.downloadSafetensorsBundle(modelId, file.files || [])
    } else {
      await modelStore.downloadModel(
        modelId,
        file.filename,
        file.size || 0,
        searchFormat.value,
        result.pipeline_tag || null,
      )
    }
    toast.add({
      severity: 'success',
      summary: 'Download started',
      detail: 'Track progress in notifications',
      life: 3000,
    })
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
    ? modelStore.allQuantizations.find((m) => m.id === file.modelId)
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
      downloadingFiles.value.delete(downloadKey)
      downloadingFiles.value = new Set(downloadingFiles.value)
      await refreshModelSearchState()
      toast.add({
        severity: 'success',
        summary: 'Projector updated',
        detail: response.message,
        life: 3000,
      })
    } else {
      toast.add({
        severity: 'success',
        summary: 'Projector update started',
        detail: response?.message || 'Track progress in notifications',
        life: 3000,
      })
    }
  } catch (e) {
    toast.add({
      severity: 'error',
      summary: 'Projector update failed',
      detail: e.message,
      life: 4000,
    })
    downloadingFiles.value.delete(downloadKey)
    downloadingFiles.value = new Set(downloadingFiles.value)
  }
}

async function updateMtp(result, file) {
  const repoId = result.modelId || result.id
  const downloadKey = getDownloadKey(repoId, file)
  const model = file.modelId
    ? modelStore.allQuantizations.find((m) => m.id === file.modelId)
    : findDownloadedQuantization(repoId, file, file.files || [])
  if (!model?.id) return

  const selectedMtp = getSelectedMtp(repoId, file) || null
  const selectedMtpOption = getSelectedMtpOption(file, selectedMtp)

  downloadingFiles.value.add(downloadKey)
  downloadingFiles.value = new Set(downloadingFiles.value)
  try {
    const response = await modelStore.updateModelMtp(
      model.id,
      selectedMtp,
      selectedMtpOption?.size || 0,
    )
    if (response?.applied) {
      downloadingFiles.value.delete(downloadKey)
      downloadingFiles.value = new Set(downloadingFiles.value)
      await refreshModelSearchState()
      toast.add({
        severity: 'success',
        summary: 'MTP draft updated',
        detail: response.message,
        life: 3000,
      })
    } else {
      toast.add({
        severity: 'success',
        summary: 'MTP draft update started',
        detail: response?.message || 'Track progress in notifications',
        life: 3000,
      })
    }
  } catch (e) {
    toast.add({
      severity: 'error',
      summary: 'MTP draft update failed',
      detail: e.message,
      life: 4000,
    })
    downloadingFiles.value.delete(downloadKey)
    downloadingFiles.value = new Set(downloadingFiles.value)
  }
}

async function updateDflash(result, file) {
  const repoId = result.modelId || result.id
  const downloadKey = getDownloadKey(repoId, file)
  const model = file.modelId
    ? modelStore.allQuantizations.find((m) => m.id === file.modelId)
    : findDownloadedQuantization(repoId, file, file.files || [])
  if (!model?.id) return

  const selectedDflash = getSelectedDflash(repoId, file) || null
  const selectedDflashOption = getSelectedDflashOption(file, selectedDflash)

  downloadingFiles.value.add(downloadKey)
  downloadingFiles.value = new Set(downloadingFiles.value)
  try {
    const response = await modelStore.updateModelDflash(
      model.id,
      selectedDflash,
      selectedDflashOption?.size || 0,
    )
    if (response?.applied) {
      downloadingFiles.value.delete(downloadKey)
      downloadingFiles.value = new Set(downloadingFiles.value)
      await refreshModelSearchState()
      toast.add({
        severity: 'success',
        summary: 'DFlash draft updated',
        detail: response.message,
        life: 3000,
      })
    } else {
      toast.add({
        severity: 'success',
        summary: 'DFlash draft update started',
        detail: response?.message || 'Track progress in notifications',
        life: 3000,
      })
    }
  } catch (e) {
    toast.add({
      severity: 'error',
      summary: 'DFlash draft update failed',
      detail: e.message,
      life: 4000,
    })
    downloadingFiles.value.delete(downloadKey)
    downloadingFiles.value = new Set(downloadingFiles.value)
  }
}

async function updateDspark(result, file) {
  const repoId = result.modelId || result.id
  const downloadKey = getDownloadKey(repoId, file)
  const model = file.modelId
    ? modelStore.allQuantizations.find((m) => m.id === file.modelId)
    : findDownloadedQuantization(repoId, file, file.files || [])
  if (!model?.id) return

  const selectedDspark = getSelectedDspark(repoId, file) || null
  const selectedDsparkOption = getSelectedDsparkOption(file, selectedDspark)

  downloadingFiles.value.add(downloadKey)
  downloadingFiles.value = new Set(downloadingFiles.value)
  try {
    const response = await modelStore.updateModelDspark(
      model.id,
      selectedDspark,
      selectedDsparkOption?.size || 0,
    )
    if (response?.applied) {
      downloadingFiles.value.delete(downloadKey)
      downloadingFiles.value = new Set(downloadingFiles.value)
      await refreshModelSearchState()
      toast.add({
        severity: 'success',
        summary: 'DSpark draft updated',
        detail: response.message,
        life: 3000,
      })
    } else {
      toast.add({
        severity: 'success',
        summary: 'DSpark draft update started',
        detail: response?.message || 'Track progress in notifications',
        life: 3000,
      })
    }
  } catch (e) {
    toast.add({
      severity: 'error',
      summary: 'DSpark draft update failed',
      detail: e.message,
      life: 4000,
    })
    downloadingFiles.value.delete(downloadKey)
    downloadingFiles.value = new Set(downloadingFiles.value)
  }
}

async function updateCatalogProjector(result, variant) {
  const model = findCatalogDownloadedModel(result, variant)
  if (!model?.id) return
  const hfId = catalogHfId(result)
  const downloadKey = `${hfId}:${variant.id}`
  const projectorFile = catalogProjectorFile(result, variant)
  const selectedProjector = getCatalogProjector(result, variant) || ''
  const projectorOption = getSelectedProjectorOption(projectorFile, selectedProjector)

  catalogDownloadingKeys.value.add(downloadKey)
  catalogDownloadingKeys.value = new Set(catalogDownloadingKeys.value)
  try {
    const response = await modelStore.updateModelProjector(
      model.id,
      selectedProjector || null,
      projectorOption?.size || 0,
    )
    if (response?.applied) {
      catalogDownloadingKeys.value.delete(downloadKey)
      catalogDownloadingKeys.value = new Set(catalogDownloadingKeys.value)
      await refreshModelSearchState()
      toast.add({
        severity: 'success',
        summary: 'Projector updated',
        detail: response.message,
        life: 3000,
      })
    } else {
      toast.add({
        severity: 'success',
        summary: 'Projector update started',
        detail: response?.message || 'Track progress in notifications',
        life: 3000,
      })
    }
  } catch (e) {
    toast.add({
      severity: 'error',
      summary: 'Projector update failed',
      detail: e?.response?.data?.detail || e.message,
      life: 4000,
    })
    catalogDownloadingKeys.value.delete(downloadKey)
    catalogDownloadingKeys.value = new Set(catalogDownloadingKeys.value)
  }
}

async function updateCatalogMtp(result, variant) {
  const model = findCatalogDownloadedModel(result, variant)
  if (!model?.id) return
  const hfId = catalogHfId(result)
  const downloadKey = `${hfId}:${variant.id}`
  const mtpFile = catalogMtpFile(result, variant)
  const selectedMtp = getCatalogMtp(result, variant) || ''
  const mtpOption = getSelectedMtpOption(mtpFile, selectedMtp)

  catalogDownloadingKeys.value.add(downloadKey)
  catalogDownloadingKeys.value = new Set(catalogDownloadingKeys.value)
  try {
    const response = await modelStore.updateModelMtp(
      model.id,
      selectedMtp || null,
      mtpOption?.size || 0,
    )
    if (response?.applied) {
      catalogDownloadingKeys.value.delete(downloadKey)
      catalogDownloadingKeys.value = new Set(catalogDownloadingKeys.value)
      await refreshModelSearchState()
      toast.add({
        severity: 'success',
        summary: 'MTP draft updated',
        detail: response.message,
        life: 3000,
      })
    } else {
      toast.add({
        severity: 'success',
        summary: 'MTP draft update started',
        detail: response?.message || 'Track progress in notifications',
        life: 3000,
      })
    }
  } catch (e) {
    toast.add({
      severity: 'error',
      summary: 'MTP draft update failed',
      detail: e?.response?.data?.detail || e.message,
      life: 4000,
    })
    catalogDownloadingKeys.value.delete(downloadKey)
    catalogDownloadingKeys.value = new Set(catalogDownloadingKeys.value)
  }
}

async function updateCatalogDflash(result, variant) {
  const model = findCatalogDownloadedModel(result, variant)
  if (!model?.id) return
  const hfId = catalogHfId(result)
  const downloadKey = `${hfId}:${variant.id}`
  const dflashFile = catalogDflashFile(result, variant)
  const selectedDflash = getCatalogDflash(result, variant) || ''
  const dflashOption = getSelectedDflashOption(dflashFile, selectedDflash)

  catalogDownloadingKeys.value.add(downloadKey)
  catalogDownloadingKeys.value = new Set(catalogDownloadingKeys.value)
  try {
    const response = await modelStore.updateModelDflash(
      model.id,
      selectedDflash || null,
      dflashOption?.size || 0,
    )
    if (response?.applied) {
      catalogDownloadingKeys.value.delete(downloadKey)
      catalogDownloadingKeys.value = new Set(catalogDownloadingKeys.value)
      await refreshModelSearchState()
      toast.add({
        severity: 'success',
        summary: 'DFlash draft updated',
        detail: response.message,
        life: 3000,
      })
    } else {
      toast.add({
        severity: 'success',
        summary: 'DFlash draft update started',
        detail: response?.message || 'Track progress in notifications',
        life: 3000,
      })
    }
  } catch (e) {
    toast.add({
      severity: 'error',
      summary: 'DFlash draft update failed',
      detail: e?.response?.data?.detail || e.message,
      life: 4000,
    })
    catalogDownloadingKeys.value.delete(downloadKey)
    catalogDownloadingKeys.value = new Set(catalogDownloadingKeys.value)
  }
}

async function updateCatalogDspark(result, variant) {
  const model = findCatalogDownloadedModel(result, variant)
  if (!model?.id) return
  const hfId = catalogHfId(result)
  const downloadKey = `${hfId}:${variant.id}`
  const dsparkFile = catalogDsparkFile(result, variant)
  const selectedDspark = getCatalogDspark(result, variant) || ''
  const dsparkOption = getSelectedDsparkOption(dsparkFile, selectedDspark)

  catalogDownloadingKeys.value.add(downloadKey)
  catalogDownloadingKeys.value = new Set(catalogDownloadingKeys.value)
  try {
    const response = await modelStore.updateModelDspark(
      model.id,
      selectedDspark || null,
      dsparkOption?.size || 0,
    )
    if (response?.applied) {
      catalogDownloadingKeys.value.delete(downloadKey)
      catalogDownloadingKeys.value = new Set(catalogDownloadingKeys.value)
      await refreshModelSearchState()
      toast.add({
        severity: 'success',
        summary: 'DSpark draft updated',
        detail: response.message,
        life: 3000,
      })
    } else {
      toast.add({
        severity: 'success',
        summary: 'DSpark draft update started',
        detail: response?.message || 'Track progress in notifications',
        life: 3000,
      })
    }
  } catch (e) {
    toast.add({
      severity: 'error',
      summary: 'DSpark draft update failed',
      detail: e?.response?.data?.detail || e.message,
      life: 4000,
    })
    catalogDownloadingKeys.value.delete(downloadKey)
    catalogDownloadingKeys.value = new Set(catalogDownloadingKeys.value)
  }
}

function configureDownloaded(modelId, file) {
  const model =
    searchFormat.value === 'safetensors'
      ? findDownloadedSafetensorsBundle(modelId)
      : file.modelId
        ? modelStore.allQuantizations.find((m) => m.id === file.modelId)
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
  return Object.values(progressTasks.value).some((task) =>
    isActiveDownloadTask(task, modelId, file),
  )
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
  if (tasks.some((task) => isActiveDownloadTask(task, modelId, file))) {
    clearPendingDownload(modelId, file)
    return
  }
  if (
    tasks.some(
      (task) => downloadTaskIdentityMatches(task, modelId, file) && task.status !== 'running',
    )
  ) {
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

function getMtpSelectionKey(modelId, file) {
  return `${modelId}:${file.quantizationKey || file.filename}:mtp`
}

function getDflashSelectionKey(modelId, file) {
  return `${modelId}:${file.quantizationKey || file.filename}:dflash`
}

function getDsparkSelectionKey(modelId, file) {
  return `${modelId}:${file.quantizationKey || file.filename}:dspark`
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

function ensureMtpSelection(modelId, file, value = '') {
  const key = getMtpSelectionKey(modelId, file)
  if (Object.prototype.hasOwnProperty.call(mtpSelections.value, key)) return
  const defaultValue = value || getDefaultMtpValue(file)
  mtpSelections.value = {
    ...mtpSelections.value,
    [key]: defaultValue,
  }
}

function ensureDflashSelection(modelId, file, value = '') {
  const key = getDflashSelectionKey(modelId, file)
  if (Object.prototype.hasOwnProperty.call(dflashSelections.value, key)) return
  const defaultValue = value || getDefaultDflashValue(file)
  dflashSelections.value = {
    ...dflashSelections.value,
    [key]: defaultValue,
  }
}

function ensureDsparkSelection(modelId, file, value = '') {
  const key = getDsparkSelectionKey(modelId, file)
  if (Object.prototype.hasOwnProperty.call(dsparkSelections.value, key)) return
  const defaultValue = value || getDefaultDsparkValue(file)
  dsparkSelections.value = {
    ...dsparkSelections.value,
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
  return (file.projectorOptions || []).find((option) => option.value === (value || '')) || null
}

function getDefaultProjectorValue(file) {
  const f16 = (file.projectorOptions || []).find((option) => option.label === 'F16')
  return f16?.value || ''
}

function setSelectedMtp(modelId, file, value) {
  mtpSelections.value = {
    ...mtpSelections.value,
    [getMtpSelectionKey(modelId, file)]: value || '',
  }
  if (value) {
    dflashSelections.value = {
      ...dflashSelections.value,
      [getDflashSelectionKey(modelId, file)]: '',
    }
  }
}

function getSelectedMtp(modelId, file) {
  const key = getMtpSelectionKey(modelId, file)
  if (Object.prototype.hasOwnProperty.call(mtpSelections.value, key)) {
    return mtpSelections.value[key]
  }
  return file.downloaded?.mtp_filename || ''
}

function getSelectedMtpOption(file, value) {
  return (file.mtpOptions || []).find((option) => option.value === (value || '')) || null
}

function getDefaultMtpValue(file) {
  const options = file.mtpOptions || []
  const preferred = ['Q8_0', 'Default', 'Q4_0']
  for (const label of preferred) {
    const match = options.find((option) => option.label === label && option.value)
    if (match) return match.value
  }
  const first = options.find((option) => option.value)
  return first?.value || ''
}

function setSelectedDflash(modelId, file, value) {
  dflashSelections.value = {
    ...dflashSelections.value,
    [getDflashSelectionKey(modelId, file)]: value || '',
  }
  if (value) {
    mtpSelections.value = {
      ...mtpSelections.value,
      [getMtpSelectionKey(modelId, file)]: '',
    }
  }
}

function getSelectedDflash(modelId, file) {
  const key = getDflashSelectionKey(modelId, file)
  if (Object.prototype.hasOwnProperty.call(dflashSelections.value, key)) {
    return dflashSelections.value[key]
  }
  return file.downloaded?.dflash_filename || ''
}

function getSelectedDflashOption(file, value) {
  return (file.dflashOptions || []).find((option) => option.value === (value || '')) || null
}

function setSelectedDspark(modelId, file, value) {
  dsparkSelections.value = {
    ...dsparkSelections.value,
    [getDsparkSelectionKey(modelId, file)]: value || '',
  }
  if (value) {
    mtpSelections.value = {
      ...mtpSelections.value,
      [getMtpSelectionKey(modelId, file)]: '',
    }
    dflashSelections.value = {
      ...dflashSelections.value,
      [getDflashSelectionKey(modelId, file)]: '',
    }
  }
}

function getSelectedDspark(modelId, file) {
  const key = getDsparkSelectionKey(modelId, file)
  if (Object.prototype.hasOwnProperty.call(dsparkSelections.value, key)) {
    return dsparkSelections.value[key]
  }
  return file.downloaded?.dspark_filename || ''
}

function getSelectedDsparkOption(file, value) {
  return (file.dsparkOptions || []).find((option) => option.value === (value || '')) || null
}

function getDefaultDsparkValue(file) {
  const options = file.dsparkOptions || []
  const preferred = ['Q8_0', 'BF16', 'F16', 'Default']
  for (const label of preferred) {
    const match = options.find((option) => option.label === label && option.value)
    if (match) return match.value
  }
  const first = options.find((option) => option.value)
  return first?.value || ''
}

function getDefaultDflashValue(file) {
  const options = file.dflashOptions || []
  const preferred = ['BF16', 'F16', 'Default']
  for (const label of preferred) {
    const match = options.find((option) => option.label === label && option.value)
    if (match) return match.value
  }
  const first = options.find((option) => option.value)
  return first?.value || ''
}

function hasProjectorSelectionChanged(modelId, file) {
  return (getSelectedProjector(modelId, file) || '') !== (file.downloaded?.mmproj_filename || '')
}

function hasMtpSelectionChanged(modelId, file) {
  return (getSelectedMtp(modelId, file) || '') !== (file.downloaded?.mtp_filename || '')
}

function hasDflashSelectionChanged(modelId, file) {
  return (getSelectedDflash(modelId, file) || '') !== (file.downloaded?.dflash_filename || '')
}

function hasDsparkSelectionChanged(modelId, file) {
  return (getSelectedDspark(modelId, file) || '') !== (file.downloaded?.dspark_filename || '')
}

function findDownloadedQuantization(hfId, entry, files = []) {
  return modelStore.allQuantizations.find(
    (m) =>
      m.huggingface_id === hfId &&
      ((entry.quantization && m.quantization === entry.quantization) ||
        files.some((file) => m.filename === file.filename)),
  )
}

function findDownloadedSafetensorsBundle(hfId) {
  return modelStore.safetensorsModels.find((model) => model.huggingface_id === hfId)
}

function findDownloadedModel(hfId, filename) {
  return modelStore.allQuantizations.find(
    (m) =>
      m.huggingface_id === hfId &&
      (m.filename === filename || (m.quantization && filename.includes(m.quantization))),
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
  'text-generation',
  'chat',
  'instruct',
  'code',
  'embedding',
  'vision',
  'multimodal',
  'image-text-to-text',
  'fill-mask',
  'question-answering',
])

function interestingTag(tag) {
  return INTERESTING_TAGS.has(tag) || tag.startsWith('language:') || /^\d+[bBmM]$/.test(tag)
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
      .map((entry) => entry?.total_size || 0)
      .filter((size) => size > 0)
    if (!sizes.length) return ''
    return `from ${formatBytes(Math.min(...sizes))}`
  }

  const totalSize = (result.safetensors_files || []).reduce(
    (sum, file) => sum + (file.size || 0),
    0,
  )
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

    const matchesQuantization =
      row.quantizationKey === quantization || row.quantization === quantization
    const matchesFilename =
      Array.isArray(payload?.filenames) &&
      payload.filenames.some((filename) =>
        (row.files || []).some((file) => file.filename === filename),
      )

    if (!matchesQuantization && !matchesFilename) return row

    const downloaded = {
      ...(row.downloaded || {}),
      id: payload?.model_id || row.modelId || row.downloaded?.id,
      mmproj_filename: payload?.mmproj_filename || row.downloaded?.mmproj_filename || '',
      mtp_filename: payload?.mtp_filename || row.downloaded?.mtp_filename || '',
      dflash_filename: payload?.dflash_filename || row.downloaded?.dflash_filename || '',
      dspark_filename: payload?.dspark_filename || row.downloaded?.dspark_filename || '',
    }

    const updatedRow = {
      ...row,
      downloaded,
      modelId: downloaded.id,
    }
    // Keep companion selectors in sync with backend state for this quant.
    const projectorKey = getProjectorSelectionKey(hfId, updatedRow)
    projectorSelections.value = {
      ...projectorSelections.value,
      [projectorKey]: downloaded.mmproj_filename || '',
    }
    const mtpKey = getMtpSelectionKey(hfId, updatedRow)
    mtpSelections.value = {
      ...mtpSelections.value,
      [mtpKey]: downloaded.mtp_filename || '',
    }
    const dflashKey = getDflashSelectionKey(hfId, updatedRow)
    dflashSelections.value = {
      ...dflashSelections.value,
      [dflashKey]: downloaded.dflash_filename || '',
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
  await Promise.all(expandedIds.map((id) => loadFiles(id)))
}

let unsubscribeDownloadComplete = null
let unsubscribeDownloadTaskCreated = null
let unsubscribeDownloadTaskUpdated = null

function reconcileCatalogDownloadsForHfId(hfId) {
  if (!hfId) return
  const next = new Set(catalogDownloadingKeys.value)
  let changed = false
  for (const key of [...next]) {
    if (!key.startsWith(`${hfId}:`)) continue
    const variantId = key.slice(hfId.length + 1)
    const result = searchResults.value.find((item) => resultMatchesHfId(item, hfId))
    const variant = result?.install_variants?.find((entry) => entry.id === variantId)
    if (!result || !variant || !catalogVariantHasActiveDownloadTask(result, variant)) {
      next.delete(key)
      changed = true
    }
  }
  if (changed) catalogDownloadingKeys.value = next
}

const trackedDownloadTasks = ref(new Set())

function rememberDownloadTask(taskId) {
  if (!taskId) return
  trackedDownloadTasks.value.add(String(taskId))
}

function handleDownloadTaskEvent(task) {
  if (task?.type !== 'download') return
  const taskId = task.task_id ? String(task.task_id) : ''
  if (trackedDownloadTasks.value.size && !trackedDownloadTasks.value.has(taskId)) return
  const hfId = task.metadata?.huggingface_id
  if (hfId) {
    reconcilePendingDownloadsForHfId(hfId)
    reconcileCatalogDownloadsForHfId(hfId)
  }
  if (!taskId || !trackedDownloadTasks.value.has(taskId)) return
  const status = String(task.status || '')
  if (status !== 'failed' && status !== 'cancelled' && status !== 'canceled') return
  trackedDownloadTasks.value.delete(taskId)
  if (status === 'failed') {
    toast.add({
      severity: 'error',
      summary: 'Download failed',
      detail: task.message || 'The download failed.',
      life: 4000,
    })
    return
  }
  toast.add({
    severity: 'warn',
    summary: 'Download cancelled',
    detail: task.message || 'The download was cancelled.',
    life: 4000,
  })
}

function catalogInstallKeysForTask(task) {
  if (task?.type !== 'audio_model_install') return []
  const taskId = task.task_id ? String(task.task_id) : ''
  const keys = new Set()
  for (const [key, rememberedTaskId] of catalogInstallTaskIds.value) {
    if (rememberedTaskId === taskId) keys.add(key)
  }
  const packageId = String(task.metadata?.package_id || '')
  if (!packageId) return [...keys]
  for (const result of searchResults.value) {
    if (result?.provider !== 'audio_cpp') continue
    for (const variant of result.install_variants || []) {
      if (catalogAudioPackageId(result, variant) === packageId) {
        keys.add(`${result.id}:${variant.id}`)
      }
    }
  }
  return [...keys]
}

async function handleCatalogInstallTaskEvent(task) {
  if (task?.type !== 'audio_model_install') return
  const keys = catalogInstallKeysForTask(task)
  if (!keys.length) return
  const status = String(task.status || '')
  if (['queued', 'running', 'cancelling'].includes(status)) {
    for (const key of keys) rememberCatalogInstall(key, task.task_id)
    return
  }
  if (!['completed', 'failed', 'cancelled', 'canceled'].includes(status)) return
  for (const key of keys) forgetCatalogInstall(key)
  if (status === 'completed') await refreshModelSearchState()
}

function handleTaskEvent(task) {
  handleDownloadTaskEvent(task)
  return handleCatalogInstallTaskEvent(task)
}

onMounted(async () => {
  try {
    await modelStore.fetchHuggingfaceTokenStatus?.()
  } catch {
    // Token status is optional for search.
  }
  try {
    if (!enginesStore.engineDescriptors.length) {
      await enginesStore.fetchEngineDescriptors?.()
    }
  } catch {
    // Engine descriptors are optional for search.
  }
  if (!modelStore.models.length) await modelStore.fetchModels()
  if (!modelStore.safetensorsModels.length) await modelStore.fetchSafetensorsModels()

  applySearchFromRoute()
  const hasRouteSearch = Boolean(
    route.query.q ||
    route.query.engine ||
    route.query.task ||
    route.query.input ||
    route.query.output ||
    route.query.provider ||
    route.query.format,
  )
  if (hasRouteSearch) {
    const page = Number(route.query.page)
    await search(Number.isFinite(page) && page >= 1 ? page : 1, { syncRoute: false })
    await syncSearchToRoute()
  }

  unsubscribeDownloadTaskCreated = progressStore.subscribe('task_created', handleTaskEvent)
  unsubscribeDownloadTaskUpdated = progressStore.subscribe('task_updated', handleTaskEvent)
  unsubscribeDownloadComplete = progressStore.subscribeToDownloadComplete(async (payload) => {
    const hfId = payload?.huggingface_id
    if (!hfId) return
    if (!searchResults.value.some((result) => resultMatchesHfId(result, hfId))) return
    reconcilePendingDownloadsForHfId(hfId)
    reconcileCatalogDownloadsForHfId(hfId)
    if (payload?.model_format === 'gguf-bundle') {
      markDownloadedFromEvent(payload)
    }
    await refreshModelSearchState()
  })
})

watch(
  () => route.query,
  async () => {
    if (syncingToRoute.value) return
    applySearchFromRoute()
    const page = Number(route.query.page)
    const safePage = Number.isFinite(page) && page >= 1 ? page : 1
    const hasRouteSearch = Boolean(
      route.query.q ||
      route.query.engine ||
      route.query.task ||
      route.query.input ||
      route.query.output ||
      route.query.provider ||
      route.query.format,
    )
    if (hasRouteSearch) {
      await search(safePage, { syncRoute: false })
    }
  },
)

onUnmounted(() => {
  if (typeof unsubscribeDownloadTaskCreated === 'function') unsubscribeDownloadTaskCreated()
  if (typeof unsubscribeDownloadTaskUpdated === 'function') unsubscribeDownloadTaskUpdated()
  if (typeof unsubscribeDownloadComplete === 'function') unsubscribeDownloadComplete()
})
</script>

<style scoped>
/* layout: .page-shell */

.catalog-results {
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
  position: relative;
}

.catalog-results--loading {
  opacity: 0.92;
}

.catalog-results__status {
  font-size: 0.8rem;
  color: var(--text-secondary);
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

.results-sort {
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.results-sort__label {
  font-size: var(--text-xs);
  color: var(--text-secondary);
  white-space: nowrap;
}

.results-sort__dropdown {
  min-width: 12.5rem;
}

.result-row {
  background: var(--bg-card);
  border: 1px solid var(--border-primary);
  border-radius: var(--radius-lg);
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

.result-main:hover {
  background: var(--bg-card-hover);
}

.result-expand-icon {
  color: var(--text-secondary);
  width: 16px;
  flex-shrink: 0;
  font-size: var(--text-xs);
}

.result-info {
  flex: 1;
  min-width: 0;
}

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
  color: var(--text-primary);
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

.model-link:hover {
  color: var(--accent-primary);
  text-decoration: underline;
}

.pipeline-tag {
  flex-shrink: 0;
}

.result-meta {
  display: flex;
  gap: 0.875rem;
  flex-wrap: wrap;
}

.meta-item {
  font-size: var(--text-xs);
  color: var(--text-secondary);
  display: flex;
  align-items: center;
  gap: 0.25rem;
}

.meta-item .pi {
  font-size: var(--text-xs);
}

.result-tags {
  display: flex;
  gap: 0.25rem;
  flex-wrap: wrap;
  flex-shrink: 0;
}

.model-tag {
  font-size: var(--text-xs);
}

/* ── Expanded files ───────────────────────────────────── */
.row-expand-enter-active,
.row-expand-leave-active {
  transition: all 0.2s ease;
  overflow: hidden;
}
.row-expand-enter-from,
.row-expand-leave-to {
  max-height: 0;
  opacity: 0;
}
.row-expand-enter-to,
.row-expand-leave-from {
  max-height: 600px;
  opacity: 1;
}

.result-files {
  border-top: 1px solid var(--border-primary);
  padding: 0.75rem 1rem;
  background: var(--bg-surface);
}

.result-files :deep(.page-loading--inline) {
  padding: 0.5rem 0;
}

.files-empty {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  font-size: 0.875rem;
  color: var(--text-secondary);
  padding: 0.5rem 0;
}

.files-table {
  font-size: var(--text-sm);
}

.file-subtext {
  color: var(--text-secondary);
  font-size: var(--text-xs);
}
.file-size {
  color: var(--text-secondary);
  white-space: nowrap;
}
.file-count {
  color: var(--text-secondary);
  white-space: nowrap;
}
.projector-cell {
  min-width: 9rem;
}
.projector-select {
  min-width: 8rem;
}
.file-actions {
  display: flex;
  align-items: center;
  gap: 0.35rem;
  justify-content: flex-end;
  flex-wrap: wrap;
}
.not-downloaded {
  color: var(--text-secondary);
}

.safetensors-download {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  margin-top: 0.75rem;
  padding-top: 0.75rem;
  border-top: 1px solid var(--border-primary);
}

.safetensors-download small {
  font-size: var(--text-xs);
  color: var(--text-secondary);
}

@media (max-width: 768px) {
  .result-name {
    flex-wrap: wrap;
  }

  .result-name__tags {
    flex-wrap: wrap;
    margin-left: 0;
    justify-content: flex-start;
  }

  .model-link--collapsed {
    max-width: 100%;
  }

  .results-sort {
    width: 100%;
    flex-wrap: wrap;
  }

  .results-sort__dropdown {
    min-width: 0;
    flex: 1 1 100%;
  }
}
</style>
