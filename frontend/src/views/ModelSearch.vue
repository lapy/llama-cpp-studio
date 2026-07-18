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
        v-if="showFormatSelect"
        v-model="searchFormat"
        :options="formatOptions"
        optionLabel="label"
        optionValue="value"
        class="format-select"
        @change="onFormatChange"
      />

      <Button
        label="Search"
        icon="pi pi-search"
        severity="success"
        :loading="searching"
        @click="runSearch"
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
        @change="onEngineFilterChange"
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
      <Dropdown
        v-if="installMethodOptions.length"
        v-model="installMethodFilter"
        :options="installMethodOptions"
        optionLabel="label"
        optionValue="value"
        placeholder="Any install method"
        showClear
        class="catalog-filter"
        @change="runSearch"
      />
      <Button
        label="Import audio bundle"
        icon="pi pi-folder-open"
        severity="secondary"
        text
        class="catalog-filters__import"
        @click="showAudioImportDialog = true"
      />
    </div>

    <ProgressTracker
      :type="['audio_model_install', 'audio_model_import']"
      show-completed
      section-title="Audio package activity"
    />

    <div v-if="!modelStore.hasHuggingfaceToken" class="token-warning">
      <i class="pi pi-key" />
      <span>No HuggingFace token set. Gated models won't be accessible.</span>
      <Button
        label="Set token"
        icon="pi pi-arrow-right"
        size="small"
        text
        class="token-warning__action"
        @click="goToTokenSettings"
      />
    </div>

    <EmptyState
      v-if="!searchResults.length && hasSearched && !searching"
      icon="pi pi-search"
      :title="`No results for “${lastQuery}”`"
      description="Try different keywords or broaden the engine, task, or modality filters."
    />

    <EmptyState
      v-else-if="!searchResults.length && !hasSearched && !searching"
      icon="pi pi-search"
      title="Discover compatible models"
      description="Search Hugging Face or browse version-pinned audio.cpp packages by engine and task."
    />

    <LoadingState
      v-else-if="searching && !searchResults.length"
      message="Searching…"
      inline
    />

    <div
      v-else-if="catalogMode && (searchResults.length || searching)"
      class="catalog-results"
      :class="{ 'catalog-results--loading': searching }"
      :aria-busy="searching ? 'true' : 'false'"
    >
      <div v-if="searching" class="catalog-results__status" aria-live="polite">
        Updating results…
      </div>

      <div class="results-header">
        <span class="results-count">
          {{ catalogTotal }} result{{ catalogTotal !== 1 ? 's' : '' }}
          <template v-if="catalogTotal > 0"> · {{ catalogPageRangeLabel }}</template>
        </span>
        <div class="results-header__actions">
          <div class="results-sort">
            <label class="results-sort__label" for="catalog-search-sort">Sort</label>
            <Dropdown
              id="catalog-search-sort"
              v-model="sortBy"
              :options="sortOptions"
              optionLabel="label"
              optionValue="value"
              class="results-sort__dropdown"
              @change="onSortChange"
            />
          </div>
          <div v-if="unavailableProviders.length" class="provider-statuses">
            <Tag
              v-for="item in unavailableProviders"
              :key="item.provider"
              :value="`${item.provider} unavailable`"
              severity="warning"
              v-tooltip.bottom="item.status.reason || item.status.manager_warning || ''"
            />
          </div>
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
            <div class="catalog-card__id">{{ catalogCardSubtitle(result) }}</div>
            <div v-if="catalogCardMeta(result).length" class="catalog-card__meta">
              <span
                v-for="item in catalogCardMeta(result)"
                :key="item.key"
                class="meta-item"
              >
                <i :class="item.icon" /> {{ item.label }}
              </span>
            </div>
          </div>
          <div class="result-name__tags">
            <Tag
              :value="result.provider === 'audio_cpp' ? 'audio.cpp' : 'Hugging Face'"
              severity="info"
            />
            <Tag v-if="result.gated" value="Gated" severity="warning" />
            <Tag
              v-if="(result.features || []).includes('multimodal')"
              value="Vision"
              severity="success"
            />
            <Tag
              v-if="(result.features || []).includes('mtp') || (result.metadata?.raw?.mtp_files || []).length"
              value="MTP"
              severity="info"
            />
          </div>
        </div>

        <p v-if="result.description" class="catalog-card__description">{{ result.description }}</p>

        <div v-if="catalogPrimaryBadges(result).length" class="catalog-badges">
          <Tag
            v-for="badge in catalogPrimaryBadges(result)"
            :key="badge.key"
            :value="badge.label"
            :severity="badge.severity"
            v-tooltip.bottom="badge.tooltip || ''"
          />
        </div>

        <div v-if="result.unavailable_reason" class="catalog-unavailable">
          <i class="pi pi-exclamation-triangle" />
          {{ result.unavailable_reason }}
        </div>
        <div v-else-if="(result.compatible_engines || []).length" class="compatibility-evidence">
          <i class="pi pi-verified" />
          Compatible with {{ (result.compatible_engines || []).join(' · ') }}
        </div>

        <div
          v-if="result.gated && !modelStore.hasHuggingfaceToken"
          class="catalog-gated-cta"
        >
          <span>This model is gated and needs a Hugging Face token.</span>
          <div class="catalog-gated-cta__actions">
            <Button
              label="Set token"
              icon="pi pi-key"
              size="small"
              severity="warning"
              outlined
              @click="goToTokenSettings"
            />
            <Button
              v-if="catalogSourceUrl(result)"
              label="Open on HF"
              icon="pi pi-external-link"
              size="small"
              severity="secondary"
              text
              @click="openExternal(catalogSourceUrl(result))"
            />
          </div>
        </div>

        <div class="install-variants">
          <template v-if="(result.install_variants || []).length <= 1">
            <div
              v-for="variant in result.install_variants || []"
              :key="variant.id"
              class="install-variant"
              :class="{
                'install-variant--recommended': isRecommendedCatalogVariant(variant),
                'install-variant--downloaded': !!findCatalogDownloadedModel(result, variant),
              }"
            >
              <div>
                <div class="install-variant__title">
                  <strong>{{ variant.label || variant.id }}</strong>
                  <Tag
                    v-if="isRecommendedCatalogVariant(variant)"
                    value="Recommended"
                    severity="success"
                  />
                  <Tag
                    v-if="findCatalogDownloadedModel(result, variant)"
                    value="Downloaded"
                    severity="success"
                  />
                  <Tag
                    v-if="catalogInstallMethodLabel(result, variant)"
                    :value="catalogInstallMethodLabel(result, variant)"
                    :severity="catalogInstallMethodSeverity(variant)"
                  />
                  <Tag
                    v-if="result.gated || variant.gated"
                    value="Gated HF"
                    severity="warn"
                  />
                </div>
                <span class="install-variant__meta">
                  <template v-if="variant.size_bytes">{{ formatBytes(variant.size_bytes) }}</template>
                  <template v-if="variant.files?.length">
                    <template v-if="variant.size_bytes"> · </template>
                    {{ variant.files.length }} file{{ variant.files.length === 1 ? '' : 's' }}
                  </template>
                </span>
                <small v-if="catalogInstallMethodHint(result, variant)">{{ catalogInstallMethodHint(result, variant) }}</small>
                <small v-else-if="variant.external_inputs_required">Additional local source input may be required.</small>
                <div
                  v-if="catalogHasProjector(result) || catalogHasMtp(result)"
                  class="install-variant__companions"
                >
                  <div
                    v-if="catalogHasProjector(result)"
                    class="install-variant__projector"
                  >
                    <label :for="`catalog-projector-${result.id}-${variant.id}`">Projector</label>
                    <Dropdown
                      :id="`catalog-projector-${result.id}-${variant.id}`"
                      :model-value="getCatalogProjector(result, variant)"
                      :options="catalogProjectorOptions(result)"
                      optionLabel="label"
                      optionValue="value"
                      class="projector-select"
                      :disabled="isCatalogVariantBusy(result, variant)"
                      @update:model-value="setCatalogProjector(result, variant, $event)"
                    />
                  </div>
                  <div
                    v-if="catalogHasMtp(result)"
                    class="install-variant__projector"
                  >
                    <label :for="`catalog-mtp-${result.id}-${variant.id}`">MTP draft</label>
                    <Dropdown
                      :id="`catalog-mtp-${result.id}-${variant.id}`"
                      :model-value="getCatalogMtp(result, variant)"
                      :options="catalogMtpOptions(result)"
                      optionLabel="label"
                      optionValue="value"
                      class="projector-select"
                      :disabled="isCatalogVariantBusy(result, variant)"
                      @update:model-value="setCatalogMtp(result, variant, $event)"
                    />
                  </div>
                </div>
              </div>
              <div class="install-variant__actions">
                <template v-if="findCatalogDownloadedModel(result, variant)">
                  <Button
                    label="Configure"
                    icon="pi pi-cog"
                    size="small"
                    severity="secondary"
                    outlined
                    @click="configureCatalogVariant(result, variant)"
                  />
                  <Button
                    v-if="hasCatalogProjectorSelectionChanged(result, variant)"
                    label="Apply projector"
                    icon="pi pi-save"
                    size="small"
                    severity="success"
                    outlined
                    :loading="isCatalogVariantBusy(result, variant)"
                    :disabled="isCatalogVariantBusy(result, variant)"
                    @click="updateCatalogProjector(result, variant)"
                  />
                  <Button
                    v-if="hasCatalogMtpSelectionChanged(result, variant)"
                    label="Apply MTP"
                    icon="pi pi-save"
                    size="small"
                    severity="success"
                    outlined
                    :loading="isCatalogVariantBusy(result, variant)"
                    :disabled="isCatalogVariantBusy(result, variant)"
                    @click="updateCatalogMtp(result, variant)"
                  />
                </template>
                <Button
                  v-else
                  :label="catalogVariantActionLabel(result)"
                  icon="pi pi-download"
                  size="small"
                  severity="success"
                  outlined
                  :disabled="!variant.installable || isCatalogVariantBusy(result, variant)"
                  :loading="isCatalogVariantBusy(result, variant)"
                  @click="handleCatalogVariantAction(result, variant)"
                />
              </div>
            </div>
          </template>
          <div v-else class="install-variant install-variant--summary">
            <div>
              <strong>{{ catalogVariantsSummaryTitle(result) }}</strong>
              <span class="install-variant__meta">{{ catalogVariantsSummaryMeta(result) }}</span>
              <span
                v-if="catalogDownloadedVariantCount(result)"
                class="install-variant__owned"
              >
                {{ catalogDownloadedVariantCount(result) }} already in library
              </span>
            </div>
            <Button
              :label="catalogVariantsBrowseLabel(result)"
              icon="pi pi-list"
              size="small"
              severity="success"
              @click="openCatalogVariantPicker(result)"
            />
          </div>
        </div>
      </article>

      <div v-if="catalogTotal > catalogPageSize" class="catalog-pagination">
        <Button label="Previous" icon="pi pi-chevron-left" severity="secondary" outlined
          :disabled="catalogPage <= 1 || searching" @click="changeCatalogPage(catalogPage - 1)" />
        <span>Page {{ catalogPage }} of {{ catalogTotalPages }}</span>
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
                  <th v-if="searchFormat === 'gguf'">MTP</th>
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
                  <td v-if="searchFormat === 'gguf'" class="projector-cell">
                    <Dropdown
                      v-if="file.kind === 'quant' && (file.mtpOptions || []).some(opt => opt.value)"
                      :model-value="getSelectedMtp(result.modelId || result.id, file)"
                      :options="file.mtpOptions || [{ label: 'None', value: '' }]"
                      optionLabel="label"
                      optionValue="value"
                      class="projector-select"
                      :disabled="isFileDownloading(result.modelId || result.id, file)"
                      @update:model-value="setSelectedMtp(result.modelId || result.id, file, $event)"
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
                        v-if="file.downloaded && searchFormat === 'gguf' && file.kind === 'quant' && hasMtpSelectionChanged(result.modelId || result.id, file)"
                        label="Apply MTP"
                        icon="pi pi-save"
                        size="small"
                        severity="success"
                        outlined
                        :loading="isFileDownloading(result.modelId || result.id, file)"
                        @click="updateMtp(result, file)"
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
      v-model:visible="showCatalogVariantPicker"
      modal
      :header="catalogVariantPickerHeader"
      class="dialog-width-md catalog-variant-picker"
      :style="{ width: 'min(40rem, 94vw)' }"
    >
      <div class="variant-picker-toolbar">
        <div
          v-if="(variantPickerResult?.install_variants || []).length > 8"
          class="variant-picker-filter"
        >
          <i class="pi pi-filter search-icon" />
          <InputText
            v-model="variantPickerFilter"
            placeholder="Filter quantizations…"
            class="variant-picker-filter__input"
          />
        </div>
        <label
          v-if="variantPickerResult?.artifact_format === 'gguf' && (variantPickerResult?.install_variants || []).length > 8"
          class="variant-picker-popular"
        >
          <input v-model="variantPickerPopularOnly" type="checkbox" />
          Popular only
        </label>
      </div>

      <div class="install-variants install-variants--picker">
        <div
          v-for="variant in filteredPickerVariants"
          :key="variant.id"
          class="install-variant"
          :class="{
            'install-variant--recommended': isRecommendedCatalogVariant(variant),
            'install-variant--downloaded': !!findCatalogDownloadedModel(variantPickerResult, variant),
          }"
        >
          <div>
            <div class="install-variant__title">
              <strong>{{ variant.label || variant.id }}</strong>
              <Tag
                v-if="isRecommendedCatalogVariant(variant)"
                value="Recommended"
                severity="success"
              />
              <Tag
                v-if="findCatalogDownloadedModel(variantPickerResult, variant)"
                value="Downloaded"
                severity="success"
              />
              <Tag
                v-if="catalogInstallMethodLabel(variantPickerResult, variant)"
                :value="catalogInstallMethodLabel(variantPickerResult, variant)"
                :severity="catalogInstallMethodSeverity(variant)"
              />
              <Tag
                v-if="variantPickerResult.gated || variant.gated"
                value="Gated HF"
                severity="warn"
              />
            </div>
            <span class="install-variant__meta">
              <template v-if="variant.size_bytes">{{ formatBytes(variant.size_bytes) }}</template>
              <template v-if="variant.files?.length">
                <template v-if="variant.size_bytes"> · </template>
                {{ variant.files.length }} file{{ variant.files.length === 1 ? '' : 's' }}
              </template>
            </span>
            <small v-if="catalogInstallMethodHint(variantPickerResult, variant)">{{ catalogInstallMethodHint(variantPickerResult, variant) }}</small>
            <small v-else-if="variant.external_inputs_required">Additional local source input may be required.</small>
            <div
              v-if="catalogHasProjector(variantPickerResult) || catalogHasMtp(variantPickerResult)"
              class="install-variant__companions"
            >
              <div
                v-if="catalogHasProjector(variantPickerResult)"
                class="install-variant__projector"
              >
                <label :for="`catalog-projector-modal-${variantPickerResult.id}-${variant.id}`">Projector</label>
                <Dropdown
                  :id="`catalog-projector-modal-${variantPickerResult.id}-${variant.id}`"
                  :model-value="getCatalogProjector(variantPickerResult, variant)"
                  :options="catalogProjectorOptions(variantPickerResult)"
                  optionLabel="label"
                  optionValue="value"
                  class="projector-select"
                  :disabled="isCatalogVariantBusy(variantPickerResult, variant)"
                  @update:model-value="setCatalogProjector(variantPickerResult, variant, $event)"
                />
              </div>
              <div
                v-if="catalogHasMtp(variantPickerResult)"
                class="install-variant__projector"
              >
                <label :for="`catalog-mtp-modal-${variantPickerResult.id}-${variant.id}`">MTP draft</label>
                <Dropdown
                  :id="`catalog-mtp-modal-${variantPickerResult.id}-${variant.id}`"
                  :model-value="getCatalogMtp(variantPickerResult, variant)"
                  :options="catalogMtpOptions(variantPickerResult)"
                  optionLabel="label"
                  optionValue="value"
                  class="projector-select"
                  :disabled="isCatalogVariantBusy(variantPickerResult, variant)"
                  @update:model-value="setCatalogMtp(variantPickerResult, variant, $event)"
                />
              </div>
            </div>
          </div>
          <div class="install-variant__actions">
            <template v-if="findCatalogDownloadedModel(variantPickerResult, variant)">
              <Button
                label="Configure"
                icon="pi pi-cog"
                size="small"
                severity="secondary"
                outlined
                @click="configureCatalogVariant(variantPickerResult, variant)"
              />
              <Button
                v-if="hasCatalogProjectorSelectionChanged(variantPickerResult, variant)"
                label="Apply projector"
                icon="pi pi-save"
                size="small"
                severity="success"
                outlined
                :loading="isCatalogVariantBusy(variantPickerResult, variant)"
                :disabled="isCatalogVariantBusy(variantPickerResult, variant)"
                @click="updateCatalogProjector(variantPickerResult, variant)"
              />
              <Button
                v-if="hasCatalogMtpSelectionChanged(variantPickerResult, variant)"
                label="Apply MTP"
                icon="pi pi-save"
                size="small"
                severity="success"
                outlined
                :loading="isCatalogVariantBusy(variantPickerResult, variant)"
                :disabled="isCatalogVariantBusy(variantPickerResult, variant)"
                @click="updateCatalogMtp(variantPickerResult, variant)"
              />
            </template>
            <Button
              v-else
              :label="catalogVariantActionLabel(variantPickerResult)"
              icon="pi pi-download"
              size="small"
              severity="success"
              outlined
              :disabled="!variant.installable || isCatalogVariantBusy(variantPickerResult, variant)"
              :loading="isCatalogVariantBusy(variantPickerResult, variant)"
              @click="handleCatalogVariantAction(variantPickerResult, variant)"
            />
          </div>
        </div>
        <p v-if="!filteredPickerVariants.length" class="variant-picker-empty">
          No quantizations match this filter.
        </p>
      </div>
      <template #footer>
        <Button label="Close" severity="secondary" text @click="closeCatalogVariantPicker" />
      </template>
    </Dialog>

    <Dialog
      v-model:visible="showInstallOptionsDialog"
      modal
      :header="installOptionsDialogHeader"
      :style="{ width: 'min(34rem, 94vw)' }"
    >
      <p class="dialog-help">
        {{ installOptionsDialogHelp }}
      </p>
      <div class="audio-install-form">
        <template v-if="catalogNeedsExternalInputs">
          <label>
            <span>Source file {{ installOptionsInputsRequired ? '' : '(optional)' }}</span>
            <InputText
              v-model="installOptions.source_file"
              placeholder="/data/input/model.pt"
              fluid
            />
          </label>
          <div class="field-divider">or</div>
          <label>
            <span>Source directory {{ installOptionsInputsRequired ? '' : '(optional)' }}</span>
            <InputText
              v-model="installOptions.source_dir"
              placeholder="/data/input/prepared-source"
              fluid
            />
          </label>
          <label>
            <span>Output file (optional)</span>
            <InputText
              v-model="installOptions.output_file"
              placeholder="/data/output/model.safetensors"
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
        </template>
        <label>
          <span>Family override (optional)</span>
          <InputText
            v-model="installOptions.family"
            placeholder="Detected automatically from package / config.json"
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
          :disabled="installOptionsInputsRequired && !installOptions.source_file && !installOptions.source_dir"
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
import { ref, computed, onMounted, onUnmounted, watch, nextTick } from 'vue'
import { storeToRefs } from 'pinia'
import { useRoute, useRouter } from 'vue-router'
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
const filesCache = ref({})   // modelId -> files[]
const projectorSelections = ref({})
const mtpSelections = ref({})
const catalogMode = ref(import.meta.env.MODE !== 'test')
const catalogActionKey = ref(null)
const catalogDownloadingKeys = ref(new Set())
const showInstallOptionsDialog = ref(false)
const pendingCatalogInstall = ref(null)
const installOptions = ref({
  source_file: '',
  source_dir: '',
  output_file: '',
  variant: '',
  family: '',
})
const installOptionsInputsRequired = computed(
  () => Boolean(pendingCatalogInstall.value?.variant?.external_inputs_required),
)
const catalogNeedsExternalInputs = computed(() => {
  const variant = pendingCatalogInstall.value?.variant
  return Boolean(variant?.external_inputs_required || variant?.external_inputs_optional)
})
const installOptionsDialogHeader = computed(() =>
  catalogNeedsExternalInputs.value
    ? 'Prepare audio.cpp package'
    : 'Install audio.cpp package',
)
const installOptionsDialogHelp = computed(() => {
  const variant = pendingCatalogInstall.value?.variant
  if (!variant) {
    return 'This package uses audio.cpp model_manager.py. Repository code is never executed.'
  }
  if (variant.external_inputs_required) {
    return (
      variant.operation_description
      || 'This converter requires a local source file or directory. Repository code is never executed.'
    )
  }
  if (variant.external_inputs_optional) {
    return (
      variant.operation_description
      || 'Optional local source override for this converter. Leave blank to download the default upstream assets via model_manager.py.'
    )
  }
  return (
    variant.method_hint
    || 'Confirm install options. Family is usually auto-detected; override only if inspect fails.'
  )
})
const showAudioImportDialog = ref(false)
const variantPickerResult = ref(null)
const variantPickerFilter = ref('')
const variantPickerPopularOnly = ref(false)
const syncingToRoute = ref(false)
const showCatalogVariantPicker = computed({
  get: () => variantPickerResult.value != null,
  set: (visible) => {
    if (!visible) {
      variantPickerResult.value = null
      variantPickerFilter.value = ''
      variantPickerPopularOnly.value = false
    }
  },
})
const catalogVariantPickerHeader = computed(() => {
  const result = variantPickerResult.value
  if (!result) return 'Select variant'
  const name = result.display_name || result.provider_item_id || 'model'
  return result.artifact_format === 'gguf'
    ? `Quantizations — ${name}`
    : `Install options — ${name}`
})
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
const installMethodFilter = ref(null)

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
const INSTALL_METHOD_LABELS = {
  direct: 'Direct HF',
  composite: 'Assemble',
  converter: 'Convert',
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

const catalogTotalPages = computed(() =>
  Math.max(1, Math.ceil((catalogTotal.value || 0) / catalogPageSize)),
)

const catalogPageRangeLabel = computed(() => {
  const total = catalogTotal.value || 0
  if (!total) return '0 of 0'
  const start = (catalogPage.value - 1) * catalogPageSize + 1
  const end = Math.min(catalogPage.value * catalogPageSize, total)
  return `${start}–${end} of ${total}`
})

const RECOMMENDED_QUANT_IDS = Object.freeze([
  'Q4_K_M',
  'Q5_K_M',
  'Q4_K_S',
  'Q6_K',
  'Q8_0',
  'Q3_K_M',
])

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
  if (sortBy.value && sortBy.value !== 'downloads_desc') next.sort = sortBy.value
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
  query.value = typeof queryObj.q === 'string' ? queryObj.q : ''
  if (typeof queryObj.format === 'string' && queryObj.format) {
    searchFormat.value = queryObj.format
  }
  engineFilter.value = typeof queryObj.engine === 'string' ? queryObj.engine : null
  taskFilter.value = typeof queryObj.task === 'string' ? queryObj.task : null
  inputModalityFilter.value = typeof queryObj.input === 'string' ? queryObj.input : null
  outputModalityFilter.value = typeof queryObj.output === 'string' ? queryObj.output : null
  providerFilter.value = typeof queryObj.provider === 'string' ? queryObj.provider : null
  installMethodFilter.value = typeof queryObj.install_method === 'string'
    ? queryObj.install_method
    : null
  sortBy.value = typeof queryObj.sort === 'string' && queryObj.sort
    ? queryObj.sort
    : 'downloads_desc'
}

function onFormatChange() {
  if (hasSearched.value) runSearch()
}

function onEngineFilterChange() {
  if (hasSearched.value) runSearch()
}

function onSortChange() {
  syncSearchToRoute()
}

async function search(page = 1, { syncRoute = true } = {}) {
  const pageNum = Number(page)
  const safePage = Number.isFinite(pageNum) && pageNum >= 1 ? Math.floor(pageNum) : 1
  expanded.value = new Set()
  filesCache.value = {}
  projectorSelections.value = {}
  mtpSelections.value = {}
  variantPickerResult.value = null
  variantPickerFilter.value = ''
  variantPickerPopularOnly.value = false
  try {
    catalogMode.value = true
    const data = await modelStore.searchCatalog(query.value.trim(), {
      page: safePage,
      page_size: catalogPageSize,
      filters: catalogFiltersPayload(),
    })
    if (data == null) return
    notifyUnavailableProviders(data?.provider_status)
    if (syncRoute) await syncSearchToRoute()
  } catch (e) {
    toast.add({ severity: 'error', summary: 'Search failed', detail: e?.response?.data?.detail || e.message, life: 4000 })
    searchResults.value = []
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
  router.push({ name: 'models' })
}

function openExternal(url) {
  if (!url) return
  window.open(url, '_blank', 'noopener,noreferrer')
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

function catalogCardSubtitle(result) {
  if (result?.provider === 'huggingface') {
    return result.source?.id || String(result.provider_item_id || '').split(':')[0] || ''
  }
  return result?.provider_item_id || result?.id || ''
}

function catalogCardMeta(result) {
  const items = []
  const author = result?.metadata?.raw?.author || result?.author
  const downloads = result?.downloads ?? result?.metadata?.downloads
  const likes = result?.likes ?? result?.metadata?.likes
  if (author) items.push({ key: 'author', icon: 'pi pi-user', label: author })
  if (downloads != null && downloads !== '') {
    items.push({ key: 'downloads', icon: 'pi pi-download', label: formatNumber(downloads) })
  }
  if (likes != null && likes !== '') {
    items.push({ key: 'likes', icon: 'pi pi-heart', label: formatNumber(likes) })
  }
  return items
}

function catalogPrimaryBadges(result) {
  const badges = []
  const tasks = result?.tasks || []
  if (result?.family) {
    badges.push({ key: 'family', label: result.family, severity: 'secondary' })
  }
  for (const task of tasks.slice(0, 2)) {
    badges.push({
      key: `task-${task}`,
      label: task,
      severity: pipelineTagSeverity(task),
    })
  }
  const modalities = [
    ...(result?.input_modalities || []).map((m) => `${m} in`),
    ...(result?.output_modalities || []).map((m) => `${m} out`),
  ]
  if (modalities.length) {
    badges.push({
      key: 'modalities',
      label: modalities.slice(0, 2).join(' · '),
      severity: 'secondary',
      tooltip: modalities.join(' · '),
    })
  }
  return badges
}

function catalogHfId(result) {
  if (result?.provider === 'huggingface') {
    return result.source?.id
      || String(result.provider_item_id || '').split(':')[0]
      || ''
  }
  return result?.modelId || result?.id || ''
}

function resultMatchesHfId(result, hfId) {
  if (!hfId) return false
  if ((result.modelId || result.id) === hfId) return true
  if (result.provider === 'huggingface') {
    return result.source?.id === hfId
      || String(result.provider_item_id || '').startsWith(`${hfId}:`)
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
    return modelStore.allQuantizations.find((model) =>
      model.huggingface_id === hfId
      && (
        model.quantization === variant.id
        || model.quantization === variant.label
        || (Array.isArray(variant.files) && variant.files.some((file) => {
          const name = typeof file === 'string' ? file : file?.filename
          return name && model.filename === name
        }))
      ),
    ) || null
  }
  if (result.provider === 'audio_cpp') {
    const packageId = String(
      result.source?.package_id
      || result.source?.id
      || result.provider_item_id
      || variant.id
      || result.id
      || '',
    ).trim()
    const recordId = packageId.startsWith('audio-cpp--')
      ? packageId
      : `audio-cpp--${packageId}`
    return modelStore.allQuantizations.find((model) =>
      model.id === recordId
      || model.id === packageId
      || model.model_id === recordId
      || model.model_id === packageId
      || model.huggingface_id === packageId
      || model.package_id === packageId
      || model?.artifact?.package_id === packageId
      || model?.manifest?.package_id === packageId,
    ) || null
  }
  return null
}

function catalogInstallMethodLabel(result, variant) {
  if (result?.provider !== 'audio_cpp') return ''
  return variant?.method_label
    || ({
      direct: 'Direct HF',
      composite: 'Assemble (model manager)',
      converter: 'Convert (model manager)',
    })[variant?.method]
    || ''
}

function catalogInstallMethodSeverity(variant) {
  if (variant?.method === 'direct') return 'secondary'
  if (variant?.method === 'composite') return 'info'
  if (variant?.method === 'converter') return 'warn'
  return 'secondary'
}

function catalogInstallMethodHint(result, variant) {
  if (result?.provider !== 'audio_cpp') return ''
  if (variant?.method_hint) return variant.method_hint
  if (variant?.uses_model_manager || ['composite', 'converter'].includes(variant?.method)) {
    return 'Installed via audio.cpp model_manager.py (assemble/convert), not a plain HF snapshot.'
  }
  if (result?.gated || variant?.gated) {
    return 'Gated Hugging Face repo — a Hugging Face token with access is required.'
  }
  return ''
}

function catalogNeedsInstallOptions(variant, result) {
  // Always confirm audio.cpp installs so family override is available.
  // Converter packages also collect optional/required local inputs here.
  if (result?.provider === 'audio_cpp') return true
  return Boolean(variant?.external_inputs_required || variant?.external_inputs_optional)
}

function catalogDownloadedVariantCount(result) {
  return (result?.install_variants || []).filter((variant) =>
    !!findCatalogDownloadedModel(result, variant),
  ).length
}

function configureCatalogVariant(result, variant) {
  const model = findCatalogDownloadedModel(result, variant)
  if (!model) return
  router.push(`/models/${encodeURIComponent(model.id || model.model_id)}/config`)
}

function isCatalogVariantBusy(result, variant) {
  if (result?.provider === 'huggingface') {
    return isCatalogVariantDownloading(result, variant)
  }
  return catalogActionKey.value === `${result.id}:${variant.id}`
}

function catalogVariantActionLabel(result) {
  return result?.provider === 'huggingface' ? 'Download' : 'Install'
}

function catalogVariantKey(variant) {
  return String(variant?.id || variant?.label || '').toUpperCase()
}

function isRecommendedCatalogVariant(variant) {
  const key = catalogVariantKey(variant)
  return RECOMMENDED_QUANT_IDS.some((id) =>
    key === id || key.endsWith(`-${id}`) || key.includes(`_${id}`) || key.includes(`-${id}`),
  )
}

function recommendedCatalogVariantScore(variant) {
  const key = catalogVariantKey(variant)
  const idx = RECOMMENDED_QUANT_IDS.findIndex((id) =>
    key === id || key.endsWith(`-${id}`),
  )
  return idx === -1 ? 1000 : idx
}

function catalogVariantsSummaryTitle(result) {
  const count = (result?.install_variants || []).length
  if (result?.artifact_format === 'gguf') {
    return `${count} quantization${count === 1 ? '' : 's'}`
  }
  return `${count} install option${count === 1 ? '' : 's'}`
}

function catalogVariantsSummaryMeta(result) {
  const labels = (result?.install_variants || [])
    .map((variant) => variant.label || variant.id)
    .filter(Boolean)
  if (!labels.length) return 'Open to choose a variant'
  const preview = labels.slice(0, 4).join(' · ')
  const remaining = labels.length - 4
  return remaining > 0 ? `${preview} · +${remaining} more` : preview
}

function catalogVariantsBrowseLabel(result) {
  return result?.artifact_format === 'gguf' ? 'Browse quants' : 'Browse options'
}

function openCatalogVariantPicker(result) {
  variantPickerFilter.value = ''
  variantPickerPopularOnly.value = false
  variantPickerResult.value = result
}

function closeCatalogVariantPicker() {
  variantPickerResult.value = null
  variantPickerFilter.value = ''
  variantPickerPopularOnly.value = false
}

const filteredPickerVariants = computed(() => {
  const result = variantPickerResult.value
  if (!result) return []
  let list = [...(result.install_variants || [])]
  if (variantPickerPopularOnly.value) {
    const popular = list.filter(isRecommendedCatalogVariant)
    if (popular.length) list = popular
  }
  const needle = variantPickerFilter.value.trim().toLowerCase()
  if (needle) {
    list = list.filter((variant) =>
      String(variant.label || variant.id || '').toLowerCase().includes(needle),
    )
  }
  list.sort((a, b) => {
    const scoreDelta = recommendedCatalogVariantScore(a) - recommendedCatalogVariantScore(b)
    if (scoreDelta !== 0) return scoreDelta
    return (a.size_bytes || 0) - (b.size_bytes || 0)
  })
  return list
})

function handleCatalogVariantAction(result, variant) {
  if (result?.provider === 'huggingface') {
    return downloadHfCatalogVariant(result, variant)
  }
  return installCatalogVariant(result, variant)
}

function normalizeVariantFiles(files = []) {
  return files.map((file) => (
    typeof file === 'string' ? { filename: file, size: 0 } : file
  ))
}

function catalogHasProjector(result) {
  return result?.artifact_format === 'gguf'
    && (result.metadata?.raw?.mmproj_files || []).length > 0
}

function catalogHasMtp(result) {
  return result?.artifact_format === 'gguf'
    && (result.metadata?.raw?.mtp_files || []).length > 0
}

function catalogProjectorOptions(result) {
  return getProjectorOptions(result?.metadata?.raw?.mmproj_files || [])
}

function catalogMtpOptions(result) {
  return getMtpOptions(result?.metadata?.raw?.mtp_files || [])
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
}

function hasCatalogProjectorSelectionChanged(result, variant) {
  const downloaded = findCatalogDownloadedModel(result, variant)
  if (!downloaded || !catalogHasProjector(result)) return false
  return (getCatalogProjector(result, variant) || '') !== (downloaded.mmproj_filename || '')
}

function hasCatalogMtpSelectionChanged(result, variant) {
  const downloaded = findCatalogDownloadedModel(result, variant)
  if (!downloaded || !catalogHasMtp(result)) return false
  return (getCatalogMtp(result, variant) || '') !== (downloaded.mtp_filename || '')
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
    return meta.quantization === quantKey
      || meta.quantization === (result.metadata?.raw?.quantizations?.[quantKey]?.quantization)
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
      await modelStore.downloadSafetensorsBundle(hfId, files)
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
      await modelStore.downloadGgufBundle(
        hfId,
        variant.id,
        files,
        pipelineTag,
        selectedProjector || null,
        projectorOption?.size || 0,
        selectedMtp || null,
        mtpOption?.size || 0,
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
    installOptions.value = {
      source_file: '',
      source_dir: '',
      output_file: '',
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
  catalogActionKey.value = key
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
    catalogActionKey.value = null
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
  mtpSelections.value = {}
  variantPickerResult.value = null
  variantPickerFilter.value = ''
  variantPickerPopularOnly.value = false
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
    const result = searchResults.value.find(r => (r.modelId || r.id) === modelId)
    if (!result) return

    let files = []
    if (searchFormat.value === 'gguf') {
      const projectorOptions = getProjectorOptions(result.mmproj_files || [])
      const mtpOptions = getMtpOptions(result.mtp_files || [])
      const quantEntries = Object.entries(result.quantizations || {}).map(([key, entry]) => ({
        key,
        kind: 'quant',
        quantizationKey: key,
        quantization: entry.quantization || '',
        variantPrefix: entry.variant_prefix || '',
        size: entry.total_size || 0,
        projectorOptions,
        mtpOptions,
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
        ensureMtpSelection(modelId, entry, entry.downloaded?.mtp_filename || '')
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
      const selectedMtp = getSelectedMtp(modelId, file)
      const selectedMtpOption = getSelectedMtpOption(file, selectedMtp)
      await modelStore.downloadGgufBundle(
        modelId,
        file.quantizationKey || file.quantization,
        file.files || [],
        result.pipeline_tag || null,
        selectedProjector || null,
        selectedProjectorOption?.size || 0,
        selectedMtp || null,
        selectedMtpOption?.size || 0,
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
      downloadingFiles.value.delete(downloadKey)
      downloadingFiles.value = new Set(downloadingFiles.value)
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

async function updateMtp(result, file) {
  const repoId = result.modelId || result.id
  const downloadKey = getDownloadKey(repoId, file)
  const model = file.modelId
    ? modelStore.allQuantizations.find(m => m.id === file.modelId)
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
      toast.add({ severity: 'success', summary: 'MTP draft updated', detail: response.message, life: 3000 })
    } else {
      toast.add({ severity: 'success', summary: 'MTP draft update started', detail: response?.message || 'Track progress in notifications', life: 3000 })
    }
  } catch (e) {
    toast.add({ severity: 'error', summary: 'MTP draft update failed', detail: e.message, life: 4000 })
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
      toast.add({ severity: 'success', summary: 'Projector updated', detail: response.message, life: 3000 })
    } else {
      toast.add({ severity: 'success', summary: 'Projector update started', detail: response?.message || 'Track progress in notifications', life: 3000 })
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
      toast.add({ severity: 'success', summary: 'MTP draft updated', detail: response.message, life: 3000 })
    } else {
      toast.add({ severity: 'success', summary: 'MTP draft update started', detail: response?.message || 'Track progress in notifications', life: 3000 })
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

function getMtpSelectionKey(modelId, file) {
  return `${modelId}:${file.quantizationKey || file.filename}:mtp`
}

function parseProjectorPrecision(filename) {
  const upper = (filename || '').toUpperCase()
  if (upper.includes('BF16')) return 'BF16'
  if (upper.includes('F16')) return 'F16'
  if (upper.includes('F32')) return 'F32'
  return null
}

function parseMtpLabel(filename) {
  const upper = (filename || '').toUpperCase()
  const match = upper.match(/\b(Q[0-9]+(?:_[A-Z0-9]+)?|IQ[0-9]_[A-Z0-9]+|UD-[A-Z0-9_]+|BF16|F16|F32)\b/)
  if (match) return match[1]
  const base = String(filename || '').split('/').pop() || ''
  if (/^mtp[-_]/i.test(base) && !/\bQ\d/i.test(base)) return 'Default'
  return 'Default'
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

function getMtpOptions(mtpFiles = []) {
  const byLabel = new Map()
  mtpFiles.forEach((file) => {
    const label = file.label || parseMtpLabel(file.filename)
    if (byLabel.has(label)) return
    byLabel.set(label, {
      label,
      value: file.filename,
      size: file.size || 0,
    })
  })
  const preferred = ['Q8_0', 'Q4_0', 'Default', 'BF16', 'F16']
  return [
    { label: 'None', value: '', size: 0 },
    ...Array.from(byLabel.values()).sort((a, b) => {
      const ia = preferred.indexOf(a.label)
      const ib = preferred.indexOf(b.label)
      if (ia !== -1 || ib !== -1) return (ia === -1 ? 99 : ia) - (ib === -1 ? 99 : ib)
      return a.label.localeCompare(b.label)
    }),
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

function ensureMtpSelection(modelId, file, value = '') {
  const key = getMtpSelectionKey(modelId, file)
  if (Object.prototype.hasOwnProperty.call(mtpSelections.value, key)) return
  const defaultValue = value || getDefaultMtpValue(file)
  mtpSelections.value = {
    ...mtpSelections.value,
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

function setSelectedMtp(modelId, file, value) {
  mtpSelections.value = {
    ...mtpSelections.value,
    [getMtpSelectionKey(modelId, file)]: value || '',
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
  return (file.mtpOptions || []).find(option => option.value === (value || '')) || null
}

function getDefaultMtpValue(file) {
  const options = file.mtpOptions || []
  const preferred = ['Q8_0', 'Default', 'Q4_0']
  for (const label of preferred) {
    const match = options.find(option => option.label === label && option.value)
    if (match) return match.value
  }
  const first = options.find(option => option.value)
  return first?.value || ''
}

function hasProjectorSelectionChanged(modelId, file) {
  return (getSelectedProjector(modelId, file) || '') !== (file.downloaded?.mmproj_filename || '')
}

function hasMtpSelectionChanged(modelId, file) {
  return (getSelectedMtp(modelId, file) || '') !== (file.downloaded?.mtp_filename || '')
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
      mtp_filename: payload?.mtp_filename || row.downloaded?.mtp_filename || '',
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

function handleDownloadTaskEvent(task) {
  if (task?.type !== 'download') return
  const hfId = task.metadata?.huggingface_id
  if (hfId) {
    reconcilePendingDownloadsForHfId(hfId)
    reconcileCatalogDownloadsForHfId(hfId)
  }
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
    route.query.q
    || route.query.engine
    || route.query.task
    || route.query.input
    || route.query.output
    || route.query.provider
    || route.query.format
  )
  if (hasRouteSearch) {
    const page = Number(route.query.page)
    await search(Number.isFinite(page) && page >= 1 ? page : 1, { syncRoute: false })
    await syncSearchToRoute()
  }

  unsubscribeDownloadTaskCreated = progressStore.subscribe('task_created', handleDownloadTaskEvent)
  unsubscribeDownloadTaskUpdated = progressStore.subscribe('task_updated', handleDownloadTaskEvent)
  unsubscribeDownloadComplete = progressStore.subscribeToDownloadComplete(async (payload) => {
    const hfId = payload?.huggingface_id
    if (!hfId) return
    if (!searchResults.value.some(result => resultMatchesHfId(result, hfId))) return
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
      route.query.q
      || route.query.engine
      || route.query.task
      || route.query.input
      || route.query.output
      || route.query.provider
      || route.query.format
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

.token-warning {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.65rem 0.85rem;
  border-radius: var(--radius-md, 0.5rem);
  border: 1px solid rgba(245, 158, 11, 0.35);
  background: rgba(245, 158, 11, 0.08);
  color: var(--text-secondary, #9ca3af);
  font-size: 0.85rem;
}

.token-warning__action {
  margin-left: auto;
}

.results-header__actions {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  flex-wrap: wrap;
}

.catalog-filters {
  display: flex;
  gap: 0.5rem;
  align-items: center;
  flex-wrap: wrap;
}

.catalog-filters__import {
  margin-left: auto;
  white-space: nowrap;
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
  position: relative;
}

.catalog-results--loading {
  opacity: 0.92;
}

.catalog-results__status {
  font-size: 0.8rem;
  color: var(--text-secondary);
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

.catalog-card__meta {
  display: flex;
  flex-wrap: wrap;
  gap: 0.75rem;
  margin-top: 0.4rem;
  color: var(--text-secondary);
  font-size: 0.78rem;
}

.catalog-card__meta .meta-item {
  display: inline-flex;
  align-items: center;
  gap: 0.3rem;
}

.catalog-card__description {
  margin: 0.75rem 0;
  color: var(--text-secondary);
  font-size: 0.875rem;
  line-height: 1.5;
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
}

.catalog-gated-cta {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  justify-content: space-between;
  gap: 0.75rem;
  margin-top: 0.75rem;
  padding: 0.65rem 0.75rem;
  border-radius: var(--radius-md);
  border: 1px solid rgba(245, 158, 11, 0.3);
  background: rgba(245, 158, 11, 0.06);
  color: var(--text-secondary);
  font-size: 0.8rem;
}

.catalog-gated-cta__actions {
  display: flex;
  gap: 0.35rem;
  flex-wrap: wrap;
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

.install-variant--summary {
  background: var(--bg-elevated, var(--bg-surface));
}

.install-variant--recommended {
  border-color: rgba(16, 185, 129, 0.45);
}

.install-variant--downloaded {
  border-color: rgba(59, 130, 246, 0.35);
}

.install-variant__title {
  display: flex;
  align-items: center;
  gap: 0.4rem;
  flex-wrap: wrap;
}

.install-variant__owned {
  display: block;
  margin-top: 0.15rem;
  color: var(--accent-green, #10b981);
  font-size: 0.75rem;
}

.install-variants--picker {
  max-height: min(60vh, 28rem);
  overflow-y: auto;
  margin-top: 0.65rem;
  padding-right: 0.15rem;
}

.variant-picker-toolbar {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  flex-wrap: wrap;
}

.variant-picker-filter {
  position: relative;
  flex: 1 1 14rem;
  min-width: 12rem;
}

.variant-picker-filter .search-icon {
  position: absolute;
  left: 0.75rem;
  top: 50%;
  transform: translateY(-50%);
  color: var(--text-secondary);
  pointer-events: none;
}

.variant-picker-filter__input {
  width: 100%;
  padding-left: 2.25rem !important;
}

.variant-picker-popular {
  display: inline-flex;
  align-items: center;
  gap: 0.4rem;
  font-size: 0.8rem;
  color: var(--text-secondary);
  white-space: nowrap;
}

.variant-picker-empty {
  margin: 0.5rem 0 0;
  color: var(--text-secondary);
  font-size: 0.85rem;
}

.install-variant__meta,
.install-variant small {
  color: var(--text-secondary);
  font-size: 0.75rem;
}

.install-variant__companions {
  display: flex;
  flex-direction: column;
  gap: 0.35rem;
  margin-top: 0.35rem;
}

.install-variant__actions {
  display: flex;
  flex-direction: column;
  align-items: stretch;
  gap: 0.35rem;
  flex-shrink: 0;
}

.install-variant__projector {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  margin-top: 0;
}

.install-variant__projector label {
  font-size: 0.75rem;
  color: var(--text-secondary);
  white-space: nowrap;
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
