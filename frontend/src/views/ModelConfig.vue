<template>
  <div class="model-config-view page-shell page-shell--relaxed">

    <LoadingState v-if="loading" message="Loading configuration…" />

    <EmptyState
      v-else-if="!model"
      icon="pi pi-exclamation-circle"
      title="Model not found"
    >
      <Button label="Back to Models" icon="pi pi-arrow-left" @click="$router.push('/models')" />
    </EmptyState>

    <template v-else>
      <PageHeader>
        <template #start>
          <Button icon="pi pi-arrow-left" text severity="secondary" @click="$router.push('/models')" />
        </template>
        <template #title>
          <div class="config-page-title">
            <h1 class="page-title">{{ model.display_name || model.base_model_name }}</h1>
            <div class="header-meta">
              <Tag :value="model.format || 'gguf'" severity="info" />
              <Tag v-if="model.quantization" :value="model.quantization" severity="secondary" />
              <a :href="`https://huggingface.co/${model.huggingface_id}`" target="_blank" class="hf-link">
                <i class="pi pi-external-link" /> {{ model.huggingface_id }}
              </a>
            </div>
          </div>
        </template>
        <template v-if="!loading && model && hasUnsavedChanges" #actions>
          <Tag value="Unsaved changes" severity="warning" class="unsaved-tag" />
        </template>
      </PageHeader>

      <!-- Engine Selector -->
      <div class="config-card">
        <div class="section-label">Engine</div>
        <div class="engine-selector">
          <div
            v-for="eng in engineOptions"
            :key="eng.value"
            class="engine-option"
            :class="{ selected: config.engine === eng.value }"
            @click="changeEngine(eng.value)"
          >
            <div class="engine-option-label">
              <span
                v-if="eng.value === 'llama_cpp'"
                class="engine-mark engine-mark--llama"
                aria-hidden="true"
              >L</span>
              <span
                v-else-if="eng.value === 'ik_llama'"
                class="engine-mark engine-mark--ik"
                aria-hidden="true"
              >IK</span>
              <i
                v-else-if="eng.value === 'lmdeploy'"
                class="pi pi-server engine-icon-lmdeploy"
                aria-hidden="true"
              />
              <i
                v-else-if="eng.value === '1cat_vllm'"
                class="pi pi-bolt engine-icon-onecat-vllm"
                aria-hidden="true"
              />
              <span class="engine-name">{{ eng.label }}</span>
            </div>
          </div>
        </div>
      </div>

      <div class="config-card">
        <div class="section-label">
          llama-swap model ID
          <small class="section-hint">Fixed YAML key for this quantization; used for running-state tracking.</small>
        </div>
        <InputText
          :model-value="llamaSwapStableId"
          readonly
          class="w-full"
          aria-label="Stable llama-swap model ID"
        />
      </div>

      <div class="config-card">
        <div class="section-label">
          Primary routing alias
          <small class="section-hint">
            Optional id your application sends in API <code>model</code> requests (llama-swap <code>alias</code>).
            Must be unique across all models. Running state uses the stable llama-swap id
            (<code>{{ llamaSwapStableId || '…' }}</code>), not this alias.
          </small>
        </div>
        <InputText
          v-model="config.model_alias"
          placeholder="e.g. my-app-model"
          class="w-full"
        />
      </div>

      <div class="config-card">
        <div class="section-label">
          Sub-ID variants (setParamsByID)
          <small class="section-hint">
            Different request-body parameters per model sub-id (e.g.
            <code>my-model:high</code>). llama-swap creates aliases for each non-empty
            sub-id. Configure
            <code>chat_template_kwargs</code> per variant (see
            <a
              href="https://github.com/mostlygeek/llama-swap/blob/main/config.example.yaml"
              target="_blank"
              rel="noopener noreferrer"
            >llama-swap config.example.yaml</a>).
          </small>
        </div>
        <div
          v-for="(variant, vIdx) in setParamsByIdVariants"
          :key="variant._key"
          class="set-params-variant"
        >
          <div class="set-params-variant__header">
            <InputText
              :model-value="variant.sub_id"
              placeholder="Sub-ID suffix (empty = base model)"
              class="set-params-sub-id"
              aria-label="Sub-ID suffix"
              @update:model-value="(v) => { variant.sub_id = v; syncSetParamsByIdFromVariants() }"
            />
            <Button
              icon="pi pi-trash"
              severity="danger"
              text
              rounded
              type="button"
              aria-label="Remove variant"
              @click="removeSetParamsByIdVariant(vIdx)"
            />
          </div>
          <div class="section-label set-params-kwargs-label">chat_template_kwargs</div>
          <div
            v-for="(row, kIdx) in variant.kwargsRows"
            :key="`${variant._key}-kw-${kIdx}`"
            class="swap-env-row"
          >
            <InputText
              :model-value="row.key"
              placeholder="key"
              class="swap-env-key"
              aria-label="chat_template_kwargs key"
              @update:model-value="(v) => { row.key = v; syncSetParamsByIdFromVariants() }"
            />
            <InputText
              :model-value="row.value"
              placeholder="value (e.g. true, false, 42, text)"
              class="flex-1"
              aria-label="chat_template_kwargs value"
              @update:model-value="(v) => { row.value = v; syncSetParamsByIdFromVariants() }"
            />
            <Button
              icon="pi pi-trash"
              severity="danger"
              text
              rounded
              type="button"
              aria-label="Remove kwarg"
              @click="removeSetParamsKwargRow(vIdx, kIdx)"
            />
          </div>
          <Button
            label="Add kwarg"
            icon="pi pi-plus"
            severity="secondary"
            outlined
            type="button"
            class="mt-1"
            @click="addSetParamsKwargRow(vIdx)"
          />
        </div>
        <Button
          label="Add variant"
          icon="pi pi-plus"
          severity="secondary"
          outlined
          type="button"
          class="mt-2"
          @click="addSetParamsByIdVariant"
        />
      </div>

      <Message
        v-if="paramRegistry.scan_error"
        severity="warn"
        :closable="false"
        class="config-scan-message"
      >
        Could not read engine CLI help: {{ paramRegistry.scan_error }}. Open Engines and use
        <strong>Rescan CLI parameters</strong> for this engine.
      </Message>
      <Message
        v-else-if="paramRegistry.scan_pending"
        severity="info"
        :closable="false"
        class="config-scan-message"
      >
        CLI parameters are not loaded for this engine yet. Activate the engine on the Engines page
        (or use <strong>Rescan CLI parameters</strong> there), then reopen this page.
      </Message>
      <Message
        v-if="unrecognizedSavedKeys.length"
        severity="warn"
        :closable="false"
        class="config-scan-message"
      >
        Deprecated or unrecognized saved keys for this engine will be dropped on the next save:
        <code>{{ unrecognizedSavedKeys.join(', ') }}</code>
      </Message>

      <!-- Catalog-backed: search → tags → single params pane -->
      <template v-if="catalogSections.length">
        <div class="config-card config-toolbar">
          <div class="config-toolbar__row">
            <span class="p-input-icon-left config-search-wrap">
              <i class="pi pi-search" aria-hidden="true" />
              <InputText
                v-model="paramSearchQuery"
                type="search"
                placeholder="Search name, flag, or description…"
                class="config-search-input"
                aria-label="Search parameters to add"
              />
            </span>
            <Button
              v-if="paramSearchQuery"
              icon="pi pi-times"
              text
              rounded
              severity="secondary"
              v-tooltip.top="'Clear search'"
              aria-label="Clear search"
              @click="paramSearchQuery = ''"
            />
          </div>
          <div class="config-toolbar__row config-toolbar__toggles">
            <div class="toggle-field">
              <InputSwitch v-model="hideUnsupportedParams" input-id="toggle-hide-unsupported" />
              <label for="toggle-hide-unsupported">Hide unsupported in this build</label>
            </div>
          </div>
        </div>

        <div v-if="!paramSearchQuery.trim()" class="config-card config-search-hint-card">
          <p class="config-muted-hint">
            Use the search box to find parameters (including by description). Results appear as tags — click a tag to add it
            to the parameters pane below.
          </p>
        </div>

        <div v-else class="config-card config-search-tags-card">
          <div class="section-label">Add parameter</div>
          <p class="config-tag-lead">
            Click a tag to add it to the pane. Search matches labels, keys, CLI flags, and descriptions (all words must match).
          </p>
          <div v-if="searchTagResults.length" class="param-tag-cloud" role="list">
            <button
              v-for="p in searchTagResults"
              :key="p.key"
              type="button"
              class="param-search-tag"
              role="listitem"
              @click="addParamKey(p.key)"
            >
              <span class="param-search-tag__label">{{ p.label }}</span>
              <code class="param-search-tag__key">{{ p.key }}</code>
            </button>
          </div>
          <Message v-else severity="secondary" :closable="false" class="config-scan-message">
            No parameters match. Try other words, turn off “hide unsupported”, or clear the search.
          </Message>
        </div>

        <div class="config-card config-params-pane">
          <div class="section-label">
            Parameters
            <small class="section-hint">Added parameters stay here until removed (reset to default)</small>
          </div>
          <Message
            v-if="!paneParams.length"
            severity="secondary"
            :closable="false"
            class="config-scan-message"
          >
            No parameters in the pane. Non-default values from your saved config are shown automatically; search above to add more.
          </Message>
          <div v-else class="params-grid section-params">
            <div
              v-for="param in paneParams"
              :key="`${param.sectionId}-${param.key}`"
              class="param-field"
              :class="{ 'param-field--unsupported': param.supported === false }"
            >
              <div class="param-field__head">
                <label :for="`p-${param.sectionId}-${param.key}`" class="param-field__label">
                  {{ param.label }}
                  <code class="param-key-hint">{{ param.key }}</code>
                  <Tag
                    v-if="param.supported === false"
                    value="Not in this build"
                    severity="secondary"
                    class="param-supported-tag"
                  />
                  <i class="pi pi-info-circle param-info" v-tooltip.top="paramDescriptionTooltip(param)" />
                </label>
                <Button
                  type="button"
                  icon="pi pi-times"
                  text
                  rounded
                  severity="secondary"
                  class="param-remove-btn"
                  aria-label="Remove parameter (reset to default)"
                  v-tooltip.top="'Remove from pane (reset to default)'"
                  @click="removeParamKey(param.key)"
                />
              </div>
              <template v-if="param.type === 'int' && (param.key === 'ctx_size' || param.key === 'session_len')">
                <div class="param-slider-row">
                  <Slider
                    v-model="config[param.key]"
                    :min="512"
                    :max="maxContextSuggestion || 131072"
                    :step="256"
                    class="param-slider"
                    :disabled="param.supported === false"
                  />
                  <span v-if="maxContextSuggestion" class="param-hint">
                    Suggested max: {{ maxContextSuggestion.toLocaleString() }} tokens
                  </span>
                </div>
                <InputNumber
                  :id="`p-${param.sectionId}-${param.key}`"
                  v-model="config[param.key]"
                  :placeholder="String(param.default ?? '')"
                  class="param-input"
                  :disabled="param.supported === false"
                />
              </template>
              <template v-else-if="param.type === 'int' && param.key === 'n_gpu_layers'">
                <div class="param-slider-row">
                  <Slider
                    v-model="config[param.key]"
                    :min="0"
                    :max="layerCountSuggestion || 128"
                    :step="1"
                    class="param-slider"
                    :disabled="param.supported === false"
                  />
                  <span v-if="layerCountSuggestion" class="param-hint">
                    Detected layers: {{ layerCountSuggestion }}
                  </span>
                </div>
                <InputNumber
                  :id="`p-${param.sectionId}-${param.key}`"
                  v-model="config[param.key]"
                  :placeholder="String(param.default ?? '')"
                  class="param-input"
                  :disabled="param.supported === false"
                />
              </template>
              <Dropdown
                v-else-if="param.value_kind === 'flag' && param.negative_flag"
                :id="`p-${param.sectionId}-${param.key}`"
                v-model="config[param.key]"
                :options="triStateOptions"
                optionLabel="label"
                optionValue="value"
                placeholder="Default"
                class="param-input"
                :disabled="param.supported === false"
              />
              <Chips
                v-else-if="param.value_kind === 'repeatable'"
                :id="`p-${param.sectionId}-${param.key}`"
                v-model="config[param.key]"
                separator=","
                class="param-input"
                :disabled="param.supported === false"
              />
              <MultiSelect
                v-else-if="isDelimitedEnumParam(param)"
                :id="`p-${param.sectionId}-${param.key}`"
                v-model="config[param.key]"
                :options="param.options || []"
                optionLabel="label"
                optionValue="value"
                display="chip"
                :placeholder="csvEnumPlaceholder(param)"
                class="param-input w-full"
                :disabled="param.supported === false"
              />
              <Dropdown
                v-else-if="param.options && param.options.length"
                :id="`p-${param.sectionId}-${param.key}`"
                v-model="config[param.key]"
                :options="param.options"
                optionLabel="label"
                optionValue="value"
                :placeholder="param.default != null ? String(param.default) : ''"
                class="param-input"
                :disabled="param.supported === false"
              />
              <InputNumber
                v-else-if="param.type === 'int'"
                :id="`p-${param.sectionId}-${param.key}`"
                v-model="config[param.key]"
                :placeholder="String(param.default ?? '')"
                class="param-input"
                :disabled="param.supported === false"
              />
              <InputNumber
                v-else-if="param.type === 'float'"
                :id="`p-${param.sectionId}-${param.key}`"
                v-model="config[param.key]"
                :minFractionDigits="1"
                :maxFractionDigits="4"
                :placeholder="String(param.default ?? '')"
                class="param-input"
                :disabled="param.supported === false"
              />
              <InputSwitch
                v-else-if="param.type === 'bool'"
                :id="`p-${param.sectionId}-${param.key}`"
                v-model="config[param.key]"
                :disabled="param.supported === false"
              />
              <Textarea
                v-else-if="param.type === 'json'"
                :id="`p-${param.sectionId}-${param.key}`"
                :model-value="jsonParamDisplay(config[param.key])"
                rows="4"
                class="w-full textarea-cli param-input"
                :placeholder="jsonParamPlaceholder(param)"
                :disabled="param.supported === false"
                autoResize
                @update:model-value="(v) => updateJsonParam(param.key, v)"
              />
              <InputText
                v-else
                :id="`p-${param.sectionId}-${param.key}`"
                v-model="config[param.key]"
                :placeholder="param.default != null ? String(param.default) : ''"
                class="param-input"
                :disabled="param.supported === false"
              />
            </div>
          </div>
        </div>
      </template>

      <!-- Custom CLI Arguments -->
      <div class="config-card">
        <div class="section-label">
          Custom Arguments
          <small class="section-hint">Raw CLI flags appended to the server command</small>
        </div>
        <Textarea
          v-model="config.custom_args"
          rows="2"
          placeholder="e.g. --some-flag value --another-flag"
          class="w-full textarea-cli"
          autoResize
        />
      </div>

      <div v-if="showNvidiaGpuBind" class="config-card">
        <div class="section-label">
          Bind the model to run on specific GPUs
          <small class="section-hint">
            Sets <code>CUDA_VISIBLE_DEVICES</code> for this model’s llama-swap process. Leave unset to use all
            {{ nvidiaGpuSelectOptions.length }} detected NVIDIA GPU(s). Selecting every GPU is the same as unset (no restriction).
          </small>
        </div>
        <MultiSelect
          v-model="cudaVisibleDeviceSelection"
          :options="nvidiaGpuSelectOptions"
          option-label="label"
          option-value="value"
          placeholder="All GPUs (default)"
          display="chip"
          class="w-full cuda-gpu-multiselect"
          :max-selected-labels="3"
          selected-items-label="{0} GPUs selected"
          filter
          aria-label="Select NVIDIA GPUs for CUDA_VISIBLE_DEVICES"
        />
      </div>

      <div class="config-card">
        <div class="section-label">
          llama-swap environment
          <small class="section-hint">
            Variables passed to the upstream process as YAML
            <code>env</code> (see
            <a
              href="https://github.com/mostlygeek/llama-swap/blob/main/config.example.yaml"
              target="_blank"
              rel="noopener noreferrer"
            >llama-swap config.example.yaml</a>). For GGUF, model paths and the engine binary use
            llama-swap <code>macros</code> (shown in the preview). Real environment variables
            (<code>CUDA_VISIBLE_DEVICES</code>, merged <code>LD_LIBRARY_PATH</code>, etc.) live only in
            YAML <code>env</code>. Do not set <code>LLAMA_STUDIO_*</code> keys in swap env—they are
            reserved and ignored.
            <template v-if="showNvidiaGpuBind">
              <code>CUDA_VISIBLE_DEVICES</code> is configured above when NVIDIA GPUs are detected.
            </template>
          </small>
        </div>
        <div
          v-for="item in swapEnvRowsDisplayed"
          :key="item.originalIndex"
          class="swap-env-row"
        >
          <InputText
            :model-value="item.row.key"
            placeholder="VAR_NAME"
            class="swap-env-key"
            aria-label="Environment variable name"
            @update:model-value="(v) => { item.row.key = v; syncSwapEnvFromRows() }"
          />
          <InputText
            :model-value="item.row.value"
            placeholder="value"
            class="flex-1"
            aria-label="Environment variable value"
            @update:model-value="(v) => { item.row.value = v; syncSwapEnvFromRows() }"
          />
          <Button
            icon="pi pi-trash"
            severity="danger"
            text
            rounded
            type="button"
            aria-label="Remove variable"
            @click="removeSwapEnvRow(item.originalIndex)"
          />
        </div>
        <Button
          label="Add variable"
          icon="pi pi-plus"
          severity="secondary"
          outlined
          type="button"
          class="mt-2"
          @click="addSwapEnvRow"
        />
      </div>

      <div class="config-card config-cmd-actions-card">
        <div class="section-label">
          llama-swap command
          <small class="section-hint">
            Inspect the generated <code>cmd</code>, env, macros, filters, and aliases in a dialog.
          </small>
        </div>
        <div class="config-cmd-actions">
          <Button
            label="Live preview"
            icon="pi pi-eye"
            severity="secondary"
            outlined
            :loading="unsavedCmdPreviewLoading && cmdPreviewDialogVisible && cmdPreviewDialogMode === 'unsaved'"
            @click="openCmdPreviewDialog('unsaved')"
          />
          <Button
            label="Saved command"
            icon="pi pi-terminal"
            severity="secondary"
            outlined
            :loading="cmdPreviewLoading && cmdPreviewDialogVisible && cmdPreviewDialogMode === 'saved'"
            @click="openCmdPreviewDialog('saved')"
          />
        </div>
      </div>

      <!-- Actions -->
      <div class="config-actions">
        <Button
          label="Save Configuration"
          icon="pi pi-save"
          severity="success"
          :loading="saving"
          @click="saveConfig"
        />
        <Button
          v-if="showApplyLlamaSwap"
          label="Apply"
          icon="pi pi-bolt"
          severity="warning"
          :loading="applyingLlamaSwap"
          :disabled="saving || applyingLlamaSwap"
          v-tooltip.top="'Regenerate llama-swap-config.yaml and reload the proxy (stops all loaded models). Saves pending edits first if needed.'"
          @click="applyLlamaSwapFromModelConfig"
        />
        <Button
          label="Templates"
          icon="pi pi-bookmark"
          severity="secondary"
          outlined
          @click="openTemplatesDialog"
        />
        <Button
          label="Reset to Saved"
          icon="pi pi-refresh"
          severity="secondary"
          outlined
          @click="resetConfig"
        />
      </div>

      <Dialog
        v-model:visible="cmdPreviewDialogVisible"
        :header="cmdPreviewDialogHeader"
        modal
        class="dialog-width-lg cmd-preview-dialog"
      >
        <p class="cmd-preview-dialog-hint">{{ cmdPreviewDialogHint }}</p>
        <div v-if="activeCmdPreview.loading" class="cmd-preview-loading">
          <i class="pi pi-spin pi-spinner" aria-hidden="true" />
          <span>{{ activeCmdPreview.loadingText }}</span>
        </div>
        <Message
          v-else-if="activeCmdPreview.error"
          severity="warn"
          :closable="false"
          class="cmd-preview-message"
        >
          {{ activeCmdPreview.error }}
        </Message>
        <Textarea
          v-else-if="activeCmdPreview.cmd"
          :model-value="activeCmdPreview.cmd"
          readonly
          rows="10"
          class="w-full textarea-cli cmd-preview-textarea"
          autoResize
        />
        <Message v-else severity="secondary" :closable="false" class="cmd-preview-message">
          {{ activeCmdPreview.emptyMessage }}
        </Message>
        <template v-if="activeCmdPreview.env">
          <div class="section-label cmd-preview-env-label">llama-swap env ({{ activeCmdPreview.suffix }})</div>
          <Textarea
            :model-value="activeCmdPreview.env"
            readonly
            rows="4"
            class="w-full textarea-cli cmd-preview-textarea"
            autoResize
          />
        </template>
        <template v-if="activeCmdPreview.macros">
          <div class="section-label cmd-preview-env-label">llama-swap macros ({{ activeCmdPreview.suffix }})</div>
          <Textarea
            :model-value="activeCmdPreview.macros"
            readonly
            rows="4"
            class="w-full textarea-cli cmd-preview-textarea"
            autoResize
          />
        </template>
        <template v-if="activeCmdPreview.filters">
          <div class="section-label cmd-preview-env-label">llama-swap filters ({{ activeCmdPreview.suffix }})</div>
          <Textarea
            :model-value="activeCmdPreview.filters"
            readonly
            rows="6"
            class="w-full textarea-cli cmd-preview-textarea"
            autoResize
          />
        </template>
        <template v-if="activeCmdPreview.aliases">
          <div class="section-label cmd-preview-env-label">llama-swap aliases ({{ activeCmdPreview.suffix }})</div>
          <Textarea
            :model-value="activeCmdPreview.aliases"
            readonly
            rows="3"
            class="w-full textarea-cli cmd-preview-textarea"
            autoResize
          />
        </template>

        <template #footer>
          <Button
            v-if="cmdPreviewDialogMode === 'unsaved'"
            label="Refresh"
            icon="pi pi-refresh"
            severity="secondary"
            outlined
            :loading="unsavedCmdPreviewLoading"
            @click="fetchUnsavedCmdPreview"
          />
          <Button
            v-else
            label="Refresh"
            icon="pi pi-refresh"
            severity="secondary"
            outlined
            :loading="cmdPreviewLoading"
            @click="fetchSavedCmdPreview"
          />
          <Button label="Close" severity="secondary" outlined @click="cmdPreviewDialogVisible = false" />
        </template>
      </Dialog>

      <Dialog
        v-model:visible="templatesDialogVisible"
        header="Configuration templates"
        modal
        class="dialog-width-md config-templates-dialog"
        @show="fetchConfigTemplates"
      >
        <p class="config-templates-lead">
          Save a snapshot of engine settings to reuse on other models, or restore a previous layout.
          Routing aliases are omitted by default so each model keeps its own llama-swap id.
        </p>

        <div class="config-templates-section">
          <div class="section-label">Save snapshot</div>
          <div class="config-templates-field">
            <label for="tpl-name">Name</label>
            <InputText
              id="tpl-name"
              v-model="templateSaveForm.name"
              placeholder="e.g. Qwen reasoning defaults"
              class="w-full"
            />
          </div>
          <div class="config-templates-field">
            <label for="tpl-desc">Description (optional)</label>
            <Textarea
              id="tpl-desc"
              v-model="templateSaveForm.description"
              rows="2"
              class="w-full"
              autoResize
            />
          </div>
          <div class="config-templates-field">
            <label for="tpl-source">Snapshot from</label>
            <Dropdown
              id="tpl-source"
              v-model="templateSaveForm.snapshot_source"
              :options="templateSnapshotSourceOptions"
              option-label="label"
              option-value="value"
              class="w-full"
            />
          </div>
          <div class="config-templates-field">
            <label for="tpl-scope">Engines to include</label>
            <Dropdown
              id="tpl-scope"
              v-model="templateSaveForm.engines_scope"
              :options="templateEnginesScopeOptions"
              option-label="label"
              option-value="value"
              class="w-full"
            />
          </div>
          <div class="config-templates-check">
            <InputSwitch v-model="templateSaveForm.include_routing" input-id="tpl-routing" />
            <label for="tpl-routing">Include routing aliases in template</label>
          </div>
          <Button
            label="Save template"
            icon="pi pi-save"
            :loading="templateSaveLoading"
            :disabled="!templateSaveForm.name.trim()"
            @click="saveConfigTemplate"
          />
        </div>

        <div class="config-templates-section config-templates-section--apply">
          <div class="section-label">Apply template</div>
          <div v-if="configTemplatesLoading" class="cmd-preview-loading">
            <i class="pi pi-spin pi-spinner" aria-hidden="true" />
            <span>Loading templates…</span>
          </div>
          <Message v-else-if="!configTemplates.length" severity="secondary" :closable="false">
            No templates saved yet.
          </Message>
          <template v-else>
            <div class="config-templates-field">
              <label for="tpl-pick">Template</label>
              <Dropdown
                id="tpl-pick"
                v-model="templateApplyForm.template_id"
                :options="configTemplates"
                option-label="name"
                option-value="id"
                placeholder="Select a template"
                class="w-full"
              />
            </div>
            <div class="config-templates-field">
              <label for="tpl-apply-mode">Apply mode</label>
              <Dropdown
                id="tpl-apply-mode"
                v-model="templateApplyForm.apply_engines"
                :options="templateApplyModeOptions"
                option-label="label"
                option-value="value"
                class="w-full"
              />
            </div>
            <div class="config-templates-check">
              <InputSwitch
                v-model="templateApplyForm.include_routing"
                input-id="tpl-apply-routing"
              />
              <label for="tpl-apply-routing">Apply routing aliases from template</label>
            </div>
            <div class="config-templates-apply-actions">
              <Button
                label="Apply to form"
                icon="pi pi-arrow-down"
                :loading="templateApplyLoading"
                :disabled="!templateApplyForm.template_id"
                @click="applyConfigTemplate(false)"
              />
              <Button
                label="Apply & save"
                icon="pi pi-check"
                severity="success"
                :loading="templateApplyLoading"
                :disabled="!templateApplyForm.template_id"
                @click="applyConfigTemplate(true)"
              />
            </div>
          </template>
        </div>

        <div v-if="configTemplates.length" class="config-templates-section">
          <div class="section-label">Saved templates</div>
          <ul class="config-templates-list">
            <li v-for="tpl in configTemplates" :key="tpl.id" class="config-templates-list-item">
              <div class="config-templates-list-main">
                <strong>{{ tpl.name }}</strong>
                <span v-if="tpl.description" class="config-templates-list-desc">{{ tpl.description }}</span>
                <small class="config-templates-list-meta">
                  {{ (tpl.engine_ids || []).join(', ') || tpl.engine || '—' }}
                  <span v-if="tpl.include_routing"> · includes routing</span>
                </small>
              </div>
              <Button
                icon="pi pi-trash"
                severity="danger"
                text
                rounded
                type="button"
                aria-label="Delete template"
                :loading="templateDeleteId === tpl.id"
                @click="deleteConfigTemplate(tpl.id)"
              />
            </li>
          </ul>
        </div>

        <template #footer>
          <Button label="Close" severity="secondary" outlined @click="templatesDialogVisible = false" />
        </template>
      </Dialog>
    </template>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, onBeforeUnmount, watch, nextTick } from 'vue'
import { useRoute } from 'vue-router'
import { useToast } from 'primevue/usetoast'
import axios from 'axios'
import Button from 'primevue/button'
import Dialog from 'primevue/dialog'
import Tag from 'primevue/tag'
import InputText from 'primevue/inputtext'
import InputNumber from 'primevue/inputnumber'
import InputSwitch from 'primevue/inputswitch'
import Dropdown from 'primevue/dropdown'
import Chips from 'primevue/chips'
import Message from 'primevue/message'
import Textarea from 'primevue/textarea'
import Slider from 'primevue/slider'
import MultiSelect from 'primevue/multiselect'
import LoadingState from '@/components/common/LoadingState.vue'
import EmptyState from '@/components/common/EmptyState.vue'
import PageHeader from '@/components/common/PageHeader.vue'
import { useModelStore } from '@/stores/models'
import { useEnginesStore } from '@/stores/engines'

const route = useRoute()
const toast = useToast()
const modelStore = useModelStore()
const enginesStore = useEnginesStore()

// ── State ──────────────────────────────────────────────────
const loading = ref(true)
const saving = ref(false)
const model = ref(null)
const config = ref({})
const savedConfig = ref({})          // for reset
const paramRegistry = ref({
  sections: [],
  scan_error: null,
  scan_pending: false,
})
const paramSearchQuery = ref('')
const hideUnsupportedParams = ref(false)
/** Catalog keys currently shown in the params pane (order = add / derive order). */
const activeParamKeys = ref([])
const modelLimits = ref(null)        // engine-agnostic: { max_context_length?, layer_count? } from config runtime_limits

const cmdPreviewText = ref('')
const cmdPreviewEnvText = ref('')
const cmdPreviewMacrosText = ref('')
const cmdPreviewFiltersText = ref('')
const cmdPreviewAliasesText = ref('')
const cmdPreviewError = ref(null)
const cmdPreviewLoading = ref(false)
const unsavedCmdPreviewText = ref('')
const unsavedCmdPreviewEnvText = ref('')
const unsavedCmdPreviewMacrosText = ref('')
const unsavedCmdPreviewFiltersText = ref('')
const unsavedCmdPreviewAliasesText = ref('')
const unsavedCmdPreviewError = ref(null)
const unsavedCmdPreviewLoading = ref(false)
/** Rows for llama-swap YAML `env` (synced into config.swap_env). */
const swapEnvRows = ref([{ key: '', value: '' }])
/** Sub-ID variants for llama-swap ``filters.setParamsByID`` (synced into config.set_params_by_id). */
const setParamsByIdVariants = ref([])
let setParamsByIdVariantKeySeq = 0
/** From GET /api/gpu-info (used for NVIDIA GPU binding UI). */
const gpuInfo = ref({
  vendor: null,
  gpus: [],
  device_count: 0,
  cpu_only_mode: true,
})
/** Indices (strings) for MultiSelect; mirrors `CUDA_VISIBLE_DEVICES` when NVIDIA GPUs exist. */
const cudaVisibleDeviceSelection = ref([])
let suppressCudaVisibleWatch = false
const applyingLlamaSwap = ref(false)
const cmdPreviewDialogVisible = ref(false)
const cmdPreviewDialogMode = ref('unsaved')
const templatesDialogVisible = ref(false)
const configTemplates = ref([])
const configTemplatesLoading = ref(false)
const templateSaveLoading = ref(false)
const templateApplyLoading = ref(false)
const templateDeleteId = ref(null)
const templateSaveForm = ref({
  name: '',
  description: '',
  snapshot_source: 'form',
  engines_scope: 'all',
  include_routing: false,
})
const templateApplyForm = ref({
  template_id: null,
  apply_engines: 'active',
  include_routing: false,
})
const templateSnapshotSourceOptions = [
  { label: 'Current form (unsaved)', value: 'form' },
  { label: 'Last saved configuration', value: 'saved' },
]
const templateEnginesScopeOptions = [
  { label: 'All configured engines', value: 'all' },
  { label: 'Active engine only', value: 'active' },
]
const templateApplyModeOptions = [
  { label: 'Merge into current engine', value: 'active' },
  { label: 'Merge all engine sections', value: 'all' },
  { label: 'Switch engine + merge template engine', value: 'set_engine' },
]
const triStateOptions = [
  { label: 'Default', value: null },
  { label: 'Enabled', value: true },
  { label: 'Disabled', value: false },
]
let unsavedPreviewTimer = null
let gpuInfoRetryTimer = null
let unsavedPreviewRequestId = 0
/** @type {AbortController | null} */
let unsavedPreviewAbort = null

const allEngineOptions = [
  { value: 'llama_cpp', label: 'llama.cpp', icon: 'pi-microchip' },
  { value: 'ik_llama',  label: 'ik_llama.cpp', icon: 'pi-microchip' },
  { value: 'lmdeploy',  label: 'LMDeploy', icon: 'pi-server' },
  { value: '1cat_vllm', label: '1Cat-vLLM', icon: 'pi-server' },
]

// GGUF is not compatible with LMDeploy / 1Cat-vLLM; show them only for safetensors
const SAFETENSORS_ONLY_ENGINES = ['lmdeploy', '1cat_vllm']
const engineOptions = computed(() => {
  const fmt = model.value?.format
  if (fmt === 'safetensors') return allEngineOptions
  return allEngineOptions.filter(eng => !SAFETENSORS_ONLY_ENGINES.includes(eng.value))
})

const showNvidiaGpuBind = computed(() => {
  const g = gpuInfo.value || {}
  return (
    g.vendor === 'nvidia' &&
    Array.isArray(g.gpus) &&
    g.gpus.length > 0 &&
    !g.cpu_only_mode
  )
})

const nvidiaGpuSelectOptions = computed(() => {
  if (!showNvidiaGpuBind.value) return []
  return gpuInfo.value.gpus.map((gpu) => {
    const idx = gpu.index != null ? gpu.index : 0
    const name = typeof gpu.name === 'string' ? gpu.name : 'GPU'
    const short =
      name.length > 56 ? `${name.slice(0, 54)}…` : name
    return {
      value: String(idx),
      label: `GPU ${idx} · ${short}`,
    }
  })
})

/** Hide CUDA_VISIBLE_DEVICES from the generic env table when the NVIDIA binding control is shown. */
const swapEnvRowsDisplayed = computed(() =>
  swapEnvRows.value
    .map((row, originalIndex) => ({ row, originalIndex }))
    .filter(({ row }) => {
      if (!showNvidiaGpuBind.value) return true
      return String(row.key || '').trim().toUpperCase() !== 'CUDA_VISIBLE_DEVICES'
    }),
)

const catalogSections = computed(() =>
  Array.isArray(paramRegistry.value.sections) ? paramRegistry.value.sections : [],
)

/** Flat list of all catalog params (sections order). */
const catalogParamList = computed(() => {
  const out = []
  for (const s of catalogSections.value) {
    for (const p of s.params || []) {
      if (p?.reserved) continue
      out.push(p)
    }
  }
  return out
})

const catalogParamByKey = computed(() => {
  const m = new Map()
  for (const s of catalogSections.value) {
    for (const p of s.params || []) {
      m.set(p.key, { ...p, sectionId: s.id })
    }
  }
  return m
})

const paneParams = computed(() => {
  const m = catalogParamByKey.value
  const out = []
  for (const key of activeParamKeys.value) {
    const p = m.get(key)
    if (p) out.push(p)
  }
  return out
})

const llamaSwapStableId = computed(() => {
  const m = model.value
  if (!m) return ''
  if (m.llama_swap_id) return m.llama_swap_id
  if (m.proxy_name) return m.proxy_name
  return ''
})

const currentEngineSection = computed(() => (
  (config.value.engines && config.value.engines[config.value.engine]) || {}
))

const unrecognizedSavedKeys = computed(() => {
  const known = new Set(['custom_args', 'model_alias', 'set_params_by_id', 'swap_env'])
  for (const key of catalogParamByKey.value.keys()) known.add(key)
  return Object.keys(currentEngineSection.value || {}).filter((key) => !known.has(key))
})

const hasUnsavedChanges = computed(() => {
  try {
    return JSON.stringify(config.value) !== JSON.stringify(savedConfig.value)
  } catch {
    return false
  }
})

/**
 * Saved model config is out of sync with llama-swap-config.yaml (server stale flag).
 * Shown only when there are no unsaved edits — save first, then Apply.
 */
const showApplyLlamaSwap = computed(
  () =>
    Boolean(
      enginesStore.swapConfigStale?.applicable &&
        enginesStore.swapConfigStale?.stale &&
        !hasUnsavedChanges.value
    )
)

/** Multi-word AND search on label, key, flags, description. */
function paramMatchesSearch(param, queryRaw) {
  if (hideUnsupportedParams.value && param.supported === false) return false
  const raw = queryRaw.trim().toLowerCase()
  if (!raw) return false
  const hay = [
    param.label || '',
    param.key || '',
    param.description || '',
    ...(param.flags || []),
  ]
    .join(' ')
    .toLowerCase()
  const tokens = raw.split(/\s+/).filter(Boolean)
  return tokens.every(t => hay.includes(t))
}

const searchTagResults = computed(() => {
  const q = paramSearchQuery.value
  if (!q.trim()) return []
  const active = new Set(activeParamKeys.value)
  const out = []
  for (const p of catalogParamList.value) {
    if (active.has(p.key)) continue
    if (paramMatchesSearch(p, q)) out.push(p)
    if (out.length >= 100) break
  }
  return out
})

function paramDescriptionTooltip(param) {
  const parts = [param.description].filter(Boolean)
  if (param.primary_flag) {
    parts.push(`Primary flag: ${param.primary_flag}`)
  }
  if (param.negative_flag) {
    parts.push(`Negative flag: ${param.negative_flag}`)
  } else if (param.flags?.length) {
    parts.push(`CLI: ${param.flags.join(', ')}`)
  }
  return parts.join('\n\n') || param.label || param.key
}

function hasExplicitValue(sec, key) {
  return Object.prototype.hasOwnProperty.call(sec || {}, key)
}

function paramIsActiveInSection(sec, param) {
  if (!hasExplicitValue(sec, param.key)) return false
  const value = sec[param.key]
  if (param.value_kind === 'flag') {
    return value === true || (param.negative_flag && value === false)
  }
  if (param.value_kind === 'repeatable') {
    return Array.isArray(value) ? value.length > 0 : Boolean(value)
  }
  if (isDelimitedEnumParam(param)) {
    return normalizeCsvEnumValue(value, param).length > 0
  }
  return value !== undefined && value !== null && value !== ''
}

function setActiveKeysFromSection(sec, params) {
  if (!params.length) {
    activeParamKeys.value = []
    return
  }
  const keys = []
  for (const p of params) {
    if (paramIsActiveInSection(sec, p)) keys.push(p.key)
  }
  activeParamKeys.value = keys
}

function csvEnumPlaceholder(param) {
  if (Array.isArray(param.default) && param.default.length) {
    return param.default.join(', ')
  }
  if (param.default != null && param.default !== '') {
    return String(param.default)
  }
  return 'Select one or more'
}

function isDelimitedEnumParam(param) {
  return ['csv_enum', 'semicolon_enum'].includes(param.value_kind) || param.type === 'multiselect'
}

function delimitedEnumSeparator(param) {
  return param.value_kind === 'semicolon_enum' ? ';' : ','
}

function normalizeCsvEnumValue(value, param) {
  if (Array.isArray(value)) {
    return value.filter((v) => v != null && v !== '')
  }
  if (value == null || value === '') return []
  if (typeof value === 'string') {
    return value.split(delimitedEnumSeparator(param)).map((s) => s.trim()).filter(Boolean)
  }
  return [value]
}

function defaultValueForParam(param) {
  if (param.value_kind === 'repeatable') return Array.isArray(param.default) ? [...param.default] : []
  if (isDelimitedEnumParam(param)) {
    return normalizeCsvEnumValue(param.default, param)
  }
  if (param.value_kind === 'flag') return param.negative_flag ? null : true
  return param.default ?? null
}

function addParamKey(key) {
  if (activeParamKeys.value.includes(key)) return
  const p = catalogParamByKey.value.get(key)
  if (!p) return
  const engine = config.value.engine
  const sec = (config.value.engines && config.value.engines[engine]) || {}
  activeParamKeys.value = [...activeParamKeys.value, key]
  const v = sec[key]
  if (Array.isArray(v)) {
    config.value[key] = [...v]
  } else {
    config.value[key] = v !== undefined && v !== null && v !== '' ? v : defaultValueForParam(p)
  }
}

function removeParamKey(key) {
  activeParamKeys.value = activeParamKeys.value.filter(k => k !== key)
  if (Object.prototype.hasOwnProperty.call(config.value, key)) {
    delete config.value[key]
  }
}

const maxContextSuggestion = computed(() => {
  if (!model.value) return null
  const limits = modelLimits.value
  const cfg = config.value || {}
  if (limits?.max_context_length != null && Number(limits.max_context_length) > 0) {
    return Number(limits.max_context_length)
  }
  if (cfg.session_len != null && Number(cfg.session_len) > 0) return Number(cfg.session_len)
  if (cfg.ctx_size != null && Number(cfg.ctx_size) > 0) return Number(cfg.ctx_size)
  return null
})

const layerCountSuggestion = computed(() => {
  const limits = modelLimits.value
  if (limits?.layer_count != null && Number(limits.layer_count) > 0) {
    return Number(limits.layer_count)
  }
  return null
})

function formatEnvPreviewLines(env) {
  if (!env || !Array.isArray(env) || !env.length) return ''
  return env.join('\n')
}

function formatMacrosPreview(macros) {
  if (!macros || typeof macros !== 'object') return ''
  const lines = Object.entries(macros).map(([k, v]) => `${k}: ${v}`)
  return lines.length ? lines.join('\n') : ''
}

function formatFiltersPreview(filters) {
  if (!filters || typeof filters !== 'object') return ''
  try {
    return JSON.stringify(filters, null, 2)
  } catch {
    return ''
  }
}

function formatAliasesPreview(aliases) {
  if (!aliases || !Array.isArray(aliases) || !aliases.length) return ''
  return aliases.join('\n')
}

const cmdPreviewDialogHeader = computed(() =>
  cmdPreviewDialogMode.value === 'saved'
    ? 'Saved llama-swap command'
    : 'Unsaved llama-swap preview',
)

const cmdPreviewDialogHint = computed(() =>
  cmdPreviewDialogMode.value === 'saved'
    ? 'Full cmd from the last saved DB config (not unsaved edits). Updates after Save or Apply.'
    : 'Live preview for the current form state. Refreshes while this dialog is open.',
)

const activeCmdPreview = computed(() => {
  if (cmdPreviewDialogMode.value === 'saved') {
    return {
      loading: cmdPreviewLoading.value,
      error: cmdPreviewError.value,
      cmd: cmdPreviewText.value,
      env: cmdPreviewEnvText.value,
      macros: cmdPreviewMacrosText.value,
      filters: cmdPreviewFiltersText.value,
      aliases: cmdPreviewAliasesText.value,
      emptyMessage: 'No saved command yet. Save configuration to generate one.',
      suffix: 'saved',
      loadingText: 'Loading saved command…',
    }
  }
  return {
    loading: unsavedCmdPreviewLoading.value,
    error: unsavedCmdPreviewError.value,
    cmd: unsavedCmdPreviewText.value,
    env: unsavedCmdPreviewEnvText.value,
    macros: unsavedCmdPreviewMacrosText.value,
    filters: unsavedCmdPreviewFiltersText.value,
    aliases: unsavedCmdPreviewAliasesText.value,
    emptyMessage: 'Preview will appear once the current form can be rendered into a command.',
    suffix: 'generated',
    loadingText: 'Refreshing preview…',
  }
})

function openCmdPreviewDialog(mode) {
  cmdPreviewDialogMode.value = mode
  cmdPreviewDialogVisible.value = true
  onCmdPreviewDialogShow()
}

function onCmdPreviewDialogShow() {
  if (cmdPreviewDialogMode.value === 'saved') {
    void fetchSavedCmdPreview()
  } else {
    void fetchUnsavedCmdPreview()
  }
}

function refreshSavedCmdPreviewIfVisible() {
  if (cmdPreviewDialogVisible.value && cmdPreviewDialogMode.value === 'saved') {
    void fetchSavedCmdPreview()
  }
}

function jsonParamDisplay(value) {
  if (value == null || value === '') return ''
  if (typeof value === 'string') {
    const trimmed = value.trim()
    if (!trimmed) return ''
    try {
      return JSON.stringify(JSON.parse(trimmed), null, 2)
    } catch {
      return value
    }
  }
  if (typeof value === 'object') {
    try {
      return JSON.stringify(value, null, 2)
    } catch {
      return ''
    }
  }
  return String(value)
}

function jsonParamPlaceholder(param) {
  if (param?.default != null && typeof param.default === 'object') {
    try {
      return JSON.stringify(param.default, null, 2)
    } catch {
      return String(param.default)
    }
  }
  if (param?.default != null) return String(param.default)
  return '{"key": "value"}'
}

function updateJsonParam(key, text) {
  const trimmed = (text ?? '').trim()
  if (!trimmed) {
    delete config.value[key]
    return
  }
  try {
    config.value[key] = JSON.parse(trimmed)
  } catch {
    config.value[key] = text
  }
}

function _nextSetParamsVariantKey() {
  setParamsByIdVariantKeySeq += 1
  return `spid-${setParamsByIdVariantKeySeq}`
}

function _setParamsByIdEqual(a, b) {
  try {
    return JSON.stringify(a ?? []) === JSON.stringify(b ?? [])
  } catch {
    return false
  }
}

function parseKwargScalar(text) {
  const trimmed = (text ?? '').trim()
  if (!trimmed) return ''
  try {
    const parsed = JSON.parse(trimmed)
    if (parsed !== null && typeof parsed === 'object') return trimmed
    return parsed
  } catch {
    return trimmed
  }
}

function formatKwargValueForInput(value) {
  if (value === true) return 'true'
  if (value === false) return 'false'
  if (value == null) return ''
  if (typeof value === 'object') {
    try {
      return JSON.stringify(value)
    } catch {
      return String(value)
    }
  }
  return String(value)
}

function syncSetParamsByIdFromVariants() {
  const out = []
  for (const variant of setParamsByIdVariants.value) {
    const kwargs = {}
    for (const row of variant.kwargsRows || []) {
      const k = (row.key || '').trim()
      if (!k) continue
      const v = row.value != null ? String(row.value) : ''
      if (v.trim() === '') continue
      kwargs[k] = parseKwargScalar(v)
    }
    if (!Object.keys(kwargs).length) continue
    out.push({
      sub_id: (variant.sub_id || '').trim(),
      params: { chat_template_kwargs: kwargs },
    })
  }
  const next = out.length ? out : []
  const current = Array.isArray(config.value.set_params_by_id) ? config.value.set_params_by_id : []
  if (_setParamsByIdEqual(next, current)) return
  config.value.set_params_by_id = next
}

function initSetParamsByIdFromConfig() {
  const raw = config.value.set_params_by_id
  if (!Array.isArray(raw) || !raw.length) {
    setParamsByIdVariants.value = []
    return
  }
  setParamsByIdVariants.value = raw.map((item) => {
    const kwargs = item?.params?.chat_template_kwargs
    const kwargsRows =
      kwargs && typeof kwargs === 'object' && !Array.isArray(kwargs)
        ? Object.entries(kwargs).map(([key, value]) => ({
            key,
            value: formatKwargValueForInput(value),
          }))
        : [{ key: '', value: '' }]
    return {
      _key: _nextSetParamsVariantKey(),
      sub_id: typeof item?.sub_id === 'string' ? item.sub_id : '',
      kwargsRows: kwargsRows.length ? kwargsRows : [{ key: '', value: '' }],
    }
  })
}

function addSetParamsByIdVariant() {
  setParamsByIdVariants.value = [
    ...setParamsByIdVariants.value,
    {
      _key: _nextSetParamsVariantKey(),
      sub_id: '',
      kwargsRows: [{ key: '', value: '' }],
    },
  ]
  syncSetParamsByIdFromVariants()
}

function removeSetParamsByIdVariant(idx) {
  setParamsByIdVariants.value = setParamsByIdVariants.value.filter((_, i) => i !== idx)
  syncSetParamsByIdFromVariants()
}

function addSetParamsKwargRow(variantIdx) {
  const variant = setParamsByIdVariants.value[variantIdx]
  if (!variant) return
  variant.kwargsRows = [...(variant.kwargsRows || []), { key: '', value: '' }]
  syncSetParamsByIdFromVariants()
}

function removeSetParamsKwargRow(variantIdx, kwIdx) {
  const variant = setParamsByIdVariants.value[variantIdx]
  if (!variant) return
  const next = (variant.kwargsRows || []).filter((_, i) => i !== kwIdx)
  variant.kwargsRows = next.length ? next : [{ key: '', value: '' }]
  syncSetParamsByIdFromVariants()
}

function _swapEnvShallowEqual(
  a,
  b,
) {
  const x = a && typeof a === 'object' && !Array.isArray(a) ? a : {}
  const y = b && typeof b === 'object' && !Array.isArray(b) ? b : {}
  const xk = Object.keys(x)
  const yk = Object.keys(y)
  if (xk.length !== yk.length) return false
  return xk.every((k) => Object.prototype.hasOwnProperty.call(y, k) && String(x[k]) === String(y[k]))
}

function syncSwapEnvFromRows() {
  const out = {}
  for (const row of swapEnvRows.value) {
    const k = (row.key || '').trim()
    if (!k) continue
    const v = row.value != null ? String(row.value) : ''
    if (v.trim() === '') continue
    out[k] = typeof row.value === 'string' ? row.value : v
  }
  if (_swapEnvShallowEqual(config.value.swap_env, out)) return
  config.value.swap_env = Object.keys(out).length ? { ...out } : {}
}

function initSwapEnvRowsFromConfig() {
  const o = config.value.swap_env
  if (o && typeof o === 'object' && !Array.isArray(o) && Object.keys(o).length) {
    swapEnvRows.value = Object.entries(o).map(([key, value]) => ({
      key,
      value: value != null ? String(value) : '',
    }))
  } else {
    swapEnvRows.value = [{ key: '', value: '' }]
  }
}

function addSwapEnvRow() {
  swapEnvRows.value = [...swapEnvRows.value, { key: '', value: '' }]
  syncSwapEnvFromRows()
}

function removeSwapEnvRow(idx) {
  const next = swapEnvRows.value.filter((_, i) => i !== idx)
  swapEnvRows.value = next.length ? next : [{ key: '', value: '' }]
  syncSwapEnvFromRows()
  syncCudaSelectionFromEnv()
}

function getRawCudaVisibleFromSwapEnv() {
  const se = config.value.swap_env
  if (!se || typeof se !== 'object' || Array.isArray(se)) return ''
  const entry = Object.keys(se).find((k) => k.toUpperCase() === 'CUDA_VISIBLE_DEVICES')
  return entry ? String(se[entry] ?? '') : ''
}

function parseCudaDeviceList(raw) {
  if (raw == null || raw === '') return []
  return String(raw)
    .split(/[\s,]+/)
    .map((t) => t.trim())
    .filter((t) => /^\d+$/.test(t))
}

function upsertSwapEnvRowCanonical(canonicalKey, value) {
  const upper = canonicalKey.toUpperCase()
  const rows = [...swapEnvRows.value]
  let idx = rows.findIndex((r) => String(r.key || '').trim().toUpperCase() === upper)
  if (idx < 0) {
    const onlyBlank =
      rows.length === 1 &&
      !String(rows[0].key || '').trim() &&
      !String(rows[0].value || '').trim()
    if (onlyBlank) {
      swapEnvRows.value = [{ key: canonicalKey, value }]
    } else {
      swapEnvRows.value = [...rows, { key: canonicalKey, value }]
    }
  } else {
    rows[idx] = { ...rows[idx], key: canonicalKey, value }
    swapEnvRows.value = rows
  }
  syncSwapEnvFromRows()
}

function removeSwapEnvRowByCanonicalKey(canonicalKey) {
  const upper = canonicalKey.toUpperCase()
  const next = swapEnvRows.value.filter(
    (r) => String(r.key || '').trim().toUpperCase() !== upper,
  )
  swapEnvRows.value = next.length ? next : [{ key: '', value: '' }]
  syncSwapEnvFromRows()
}

function applyNvidiaCudaSelection(selected) {
  if (!showNvidiaGpuBind.value) return
  const allVals = nvidiaGpuSelectOptions.value.map((o) => o.value)
  const sel = [...new Set(selected)]
    .filter((v) => allVals.includes(v))
    .sort((a, b) => Number(a) - Number(b))
  const allSelected = allVals.length > 0 && sel.length === allVals.length
  if (sel.length === 0 || allSelected) {
    removeSwapEnvRowByCanonicalKey('CUDA_VISIBLE_DEVICES')
  } else {
    upsertSwapEnvRowCanonical('CUDA_VISIBLE_DEVICES', sel.join(','))
  }
}

function syncCudaSelectionFromEnv() {
  if (!showNvidiaGpuBind.value) {
    cudaVisibleDeviceSelection.value = []
    return
  }
  const raw = getRawCudaVisibleFromSwapEnv()
  let parsed = parseCudaDeviceList(raw)
  const valid = new Set(nvidiaGpuSelectOptions.value.map((o) => o.value))
  parsed = parsed.filter((p) => valid.has(p))
  suppressCudaVisibleWatch = true
  cudaVisibleDeviceSelection.value = parsed
  nextTick(() => {
    suppressCudaVisibleWatch = false
  })
}

watch(
  cudaVisibleDeviceSelection,
  (nv) => {
    if (suppressCudaVisibleWatch || !showNvidiaGpuBind.value) return
    applyNvidiaCudaSelection(Array.isArray(nv) ? [...nv] : [])
  },
  { deep: true },
)

watch(showNvidiaGpuBind, (on) => {
  if (on) nextTick(() => syncCudaSelectionFromEnv())
  else cudaVisibleDeviceSelection.value = []
})

// ── Helpers ────────────────────────────────────────────────
function modelApiUrl(suffix) {
  const id = encodeURIComponent(String(route.params.id))
  return `/api/models/${id}${suffix}`
}

function formatAxiosDetail(e) {
  const d = e?.response?.data?.detail
  if (typeof d === 'string') return d
  if (Array.isArray(d)) {
    return d
      .map((x) =>
        typeof x === 'object' && x?.msg
          ? `${Array.isArray(x.loc) ? x.loc.join('.') : ''}: ${x.msg}`.replace(/^\.\s*/, '')
          : String(x)
      )
      .filter(Boolean)
      .join('; ')
  }
  if (d && typeof d === 'object' && typeof d.msg === 'string') return d.msg
  return e?.message || 'Request failed'
}

function findModelById(id) {
  const sid = String(id)
  for (const group of modelStore.models) {
    for (const q of group.quantizations || []) {
      if (String(q.id) === sid) return { ...q, base_model_name: group.base_model_name, huggingface_id: group.huggingface_id }
    }
  }
  // Fallback: search allQuantizations
  return modelStore.allQuantizations.find(m => String(m.id) === sid) ?? null
}

function hasGpuInfoPayload(data) {
  return data && typeof data === 'object' && (
    Object.prototype.hasOwnProperty.call(data, 'gpus') ||
    Object.prototype.hasOwnProperty.call(data, 'device_count') ||
    Object.prototype.hasOwnProperty.call(data, 'cpu_only_mode')
  )
}

async function fetchGpuInfo(attempt = 0) {
  try {
    const cached = enginesStore.gpuInfo
    if (attempt === 0 && hasGpuInfoPayload(cached)) {
      gpuInfo.value = cached
      if (!cached.detecting) return
    }
    const data = await enginesStore.fetchGpuInfo()
    gpuInfo.value =
      data && typeof data === 'object'
        ? data
        : { vendor: null, gpus: [], device_count: 0, cpu_only_mode: true }
    if (data?.detecting && attempt < 3) {
      if (gpuInfoRetryTimer) clearTimeout(gpuInfoRetryTimer)
      gpuInfoRetryTimer = window.setTimeout(() => {
        void fetchGpuInfo(attempt + 1)
      }, 750 * (attempt + 1))
    }
  } catch (e) {
    console.error('Failed to fetch GPU info:', e)
    gpuInfo.value = { vendor: null, gpus: [], device_count: 0, cpu_only_mode: true }
  }
}

async function fetchParamRegistry(engine) {
  try {
    const { data } = await axios.get('/api/models/param-registry', {
      params: { engine },
    })
    paramRegistry.value = {
      sections: data.sections || [],
      scan_error: data.scan_error ?? null,
      scan_pending: Boolean(data.scan_pending),
    }
  } catch (e) {
    console.error('Failed to fetch param registry:', e)
    paramRegistry.value = {
      sections: [],
      scan_error: null,
      scan_pending: false,
    }
  }
}

function buildWorkingConfigFromApi(cfg) {
  const engines =
    cfg.engines && typeof cfg.engines === 'object'
      ? JSON.parse(JSON.stringify(cfg.engines))
      : {}
  const engine = cfg.engine ?? 'llama_cpp'
  const sec = engines[engine] || {}
  return {
    engine,
    engines,
    ...sec,
  }
}

function buildEngineStashFromForm(sourceConfig = config.value) {
  syncSetParamsByIdFromVariants()
  const stash = {}
  if (typeof sourceConfig.model_alias === 'string' && sourceConfig.model_alias.trim()) {
    stash.model_alias = sourceConfig.model_alias.trim()
  }
  if (typeof sourceConfig.custom_args === 'string' && sourceConfig.custom_args.trim()) {
    stash.custom_args = sourceConfig.custom_args
  }
  const se = sourceConfig.swap_env
  if (se && typeof se === 'object' && !Array.isArray(se)) {
    const cleaned = {}
    for (const [k, v] of Object.entries(se)) {
      const name = String(k).trim()
      if (!name) continue
      if (v == null || v === '') continue
      if (typeof v === 'number' && Number.isNaN(v)) continue
      cleaned[name] = v
    }
    stash.swap_env = cleaned
  } else {
    stash.swap_env = {}
  }
  const spid = sourceConfig.set_params_by_id
  if (Array.isArray(spid) && spid.length) {
    stash.set_params_by_id = JSON.parse(JSON.stringify(spid))
  }
  for (const key of activeParamKeys.value) {
    if (!Object.prototype.hasOwnProperty.call(sourceConfig, key)) continue
    const value = sourceConfig[key]
    if (value == null || value === '' || (typeof value === 'number' && Number.isNaN(value))) continue
    if (Array.isArray(value)) {
      if (!value.length) continue
      stash[key] = [...value]
      continue
    }
    stash[key] = value
  }
  return stash
}

function buildPersistedPayload(sourceConfig = config.value) {
  syncSwapEnvFromRows()
  syncSetParamsByIdFromVariants()
  const engines =
    sourceConfig.engines && typeof sourceConfig.engines === 'object'
      ? JSON.parse(JSON.stringify(sourceConfig.engines))
      : {}
  engines[sourceConfig.engine] = buildEngineStashFromForm(sourceConfig)
  return {
    engine: sourceConfig.engine,
    engines,
  }
}

function stashCurrentEngineIntoEngines(engineKey) {
  if (!engineKey) return
  syncSwapEnvFromRows()
  syncSetParamsByIdFromVariants()
  if (!config.value.engines) config.value.engines = {}
  config.value.engines[engineKey] = buildEngineStashFromForm(config.value)
}

function applyEngineSectionToForm(engine) {
  const sec = (config.value.engines && config.value.engines[engine]) || {}
  const params = catalogParamList.value
  if (!params.length) {
    const eng = config.value.engine
    const engMap = config.value.engines
    for (const k of Object.keys(config.value)) {
      if (k !== 'engine' && k !== 'engines') delete config.value[k]
    }
    Object.assign(config.value, sec)
    config.value.engine = eng
    config.value.engines = engMap
    initSwapEnvRowsFromConfig()
    initSetParamsByIdFromConfig()
    syncCudaSelectionFromEnv()
    return
  }
  const allowed = new Set([
    'engine',
    'engines',
    'custom_args',
    'model_alias',
    'set_params_by_id',
    'swap_env',
    ...activeParamKeys.value,
  ])
  for (const k of Object.keys(config.value)) {
    if (!allowed.has(k)) delete config.value[k]
  }
  config.value.model_alias = typeof sec.model_alias === 'string' ? sec.model_alias : ''
  config.value.custom_args = typeof sec.custom_args === 'string' ? sec.custom_args : ''
  config.value.swap_env =
    sec.swap_env && typeof sec.swap_env === 'object' && !Array.isArray(sec.swap_env)
      ? { ...sec.swap_env }
      : {}
  config.value.set_params_by_id = Array.isArray(sec.set_params_by_id)
    ? JSON.parse(JSON.stringify(sec.set_params_by_id))
    : []
  initSetParamsByIdFromConfig()
  for (const p of params) {
    if (!activeParamKeys.value.includes(p.key)) continue
    const v = sec[p.key]
    if (isDelimitedEnumParam(p)) {
      const normalized = normalizeCsvEnumValue(
        v !== undefined && v !== null && v !== '' ? v : p.default,
        p,
      )
      config.value[p.key] = normalized.length ? normalized : defaultValueForParam(p)
      continue
    }
    config.value[p.key] =
      Array.isArray(v) ? [...v] : (v !== undefined && v !== null && v !== '' ? v : defaultValueForParam(p))
  }
  initSwapEnvRowsFromConfig()
  initSetParamsByIdFromConfig()
  syncCudaSelectionFromEnv()
}

// ── Engine change ──────────────────────────────────────────
async function changeEngine(engine) {
  if (engine === config.value.engine) return
  stashCurrentEngineIntoEngines(config.value.engine)
  config.value.engine = engine
  paramSearchQuery.value = ''
  hideUnsupportedParams.value = false
  await fetchParamRegistry(engine)
  const sec = (config.value.engines && config.value.engines[engine]) || {}
  setActiveKeysFromSection(sec, catalogParamList.value)
  applyEngineSectionToForm(engine)
}

// ── Load ───────────────────────────────────────────────────
async function loadAll() {
  loading.value = true
  try {
    if (!modelStore.models.length) await modelStore.fetchModels()
    const found = findModelById(route.params.id)
    if (!found) { loading.value = false; return }
    model.value = found

    const cfgResp = await axios.get(modelApiUrl('/config'))
    const cfg = cfgResp.data
    let engine = cfg.engine ?? found.engine ?? 'llama_cpp'
    if (found.format !== 'safetensors' && ['lmdeploy', '1cat_vllm'].includes(engine)) {
      engine = 'llama_cpp'
    }

    await fetchParamRegistry(engine)

    const merged = buildWorkingConfigFromApi({ ...cfg, engine })
    config.value = merged
    const sec = (merged.engines && merged.engines[engine]) || {}
    setActiveKeysFromSection(sec, catalogParamList.value)
    applyEngineSectionToForm(engine)
    savedConfig.value = JSON.parse(JSON.stringify(config.value))
    modelLimits.value = cfg.runtime_limits ?? null
  } catch (e) {
    toast.add({ severity: 'error', summary: 'Failed to load config', detail: formatAxiosDetail(e), life: 4000 })
  } finally {
    loading.value = false
  }
  if (model.value) {
    void fetchGpuInfo()
  }
}

// ── Config templates ───────────────────────────────────────
function openTemplatesDialog() {
  templatesDialogVisible.value = true
}

async function fetchConfigTemplates() {
  configTemplatesLoading.value = true
  try {
    const { data } = await axios.get('/api/model-config-templates')
    configTemplates.value = Array.isArray(data) ? data : []
    if (
      templateApplyForm.value.template_id &&
      !configTemplates.value.some((t) => t.id === templateApplyForm.value.template_id)
    ) {
      templateApplyForm.value.template_id = null
    }
  } catch (e) {
    configTemplates.value = []
    toast.add({
      severity: 'error',
      summary: 'Templates',
      detail: formatAxiosDetail(e) || 'Could not load templates.',
      life: 4000,
    })
  } finally {
    configTemplatesLoading.value = false
  }
}

async function saveConfigTemplate() {
  templateSaveLoading.value = true
  try {
    const body = {
      name: templateSaveForm.value.name.trim(),
      description: templateSaveForm.value.description,
      include_routing: templateSaveForm.value.include_routing,
      engines_scope: templateSaveForm.value.engines_scope,
      use_saved: templateSaveForm.value.snapshot_source === 'saved',
    }
    if (templateSaveForm.value.snapshot_source === 'form') {
      body.config = buildPersistedPayload(config.value)
    }
    await axios.post(modelApiUrl('/config/save-template'), body)
    toast.add({
      severity: 'success',
      summary: 'Template saved',
      detail: `"${body.name}" is available for other models.`,
      life: 3000,
    })
    templateSaveForm.value.name = ''
    templateSaveForm.value.description = ''
    await fetchConfigTemplates()
  } catch (e) {
    toast.add({
      severity: 'error',
      summary: 'Save template failed',
      detail: formatAxiosDetail(e) || 'Could not save template.',
      life: 4000,
    })
  } finally {
    templateSaveLoading.value = false
  }
}

async function applyConfigTemplate(persist) {
  if (!templateApplyForm.value.template_id) return
  templateApplyLoading.value = true
  try {
    const { data } = await axios.post(modelApiUrl('/config/apply-template'), {
      template_id: templateApplyForm.value.template_id,
      apply_engines: templateApplyForm.value.apply_engines,
      include_routing: templateApplyForm.value.include_routing,
      persist,
    })
    const merged = buildWorkingConfigFromApi(data.config)
    config.value = merged
    const eng = config.value.engine
    const sec = (merged.engines && merged.engines[eng]) || {}
    setActiveKeysFromSection(sec, catalogParamList.value)
    applyEngineSectionToForm(eng)
    if (persist) {
      savedConfig.value = JSON.parse(JSON.stringify(config.value))
      enginesStore.markSwapConfigStaleLocal()
      void enginesStore.fetchSwapConfigStale()
      refreshSavedCmdPreviewIfVisible()
    }
    templatesDialogVisible.value = false
    toast.add({
      severity: 'success',
      summary: persist ? 'Template applied & saved' : 'Template applied',
      detail: data.template_name
        ? `Loaded settings from "${data.template_name}".`
        : 'Template settings loaded into the form.',
      life: 3000,
    })
  } catch (e) {
    toast.add({
      severity: 'error',
      summary: 'Apply template failed',
      detail: formatAxiosDetail(e) || 'Could not apply template.',
      life: 4000,
    })
  } finally {
    templateApplyLoading.value = false
  }
}

async function deleteConfigTemplate(templateId) {
  templateDeleteId.value = templateId
  try {
    await axios.delete(`/api/model-config-templates/${encodeURIComponent(templateId)}`)
    if (templateApplyForm.value.template_id === templateId) {
      templateApplyForm.value.template_id = null
    }
    await fetchConfigTemplates()
    toast.add({ severity: 'success', summary: 'Template deleted', life: 2000 })
  } catch (e) {
    toast.add({
      severity: 'error',
      summary: 'Delete failed',
      detail: formatAxiosDetail(e) || 'Could not delete template.',
      life: 4000,
    })
  } finally {
    templateDeleteId.value = null
  }
}

// ── Save ───────────────────────────────────────────────────
async function saveConfig() {
  saving.value = true
  try {
    const payload = buildPersistedPayload(config.value)
    const { data } = await axios.put(modelApiUrl('/config'), payload)
    const merged = buildWorkingConfigFromApi(data)
    config.value = merged
    const eng = config.value.engine
    const sec = (merged.engines && merged.engines[eng]) || {}
    setActiveKeysFromSection(sec, catalogParamList.value)
    applyEngineSectionToForm(eng)
    savedConfig.value = JSON.parse(JSON.stringify(config.value))
    enginesStore.markSwapConfigStaleLocal()
    void enginesStore.fetchSwapConfigStale()
    refreshSavedCmdPreviewIfVisible()
    toast.add({ severity: 'success', summary: 'Saved', detail: 'Configuration saved', life: 2000 })
    return true
  } catch (e) {
    const detail = formatAxiosDetail(e) || 'Save failed'
    toast.add({ severity: 'error', summary: 'Save failed', detail, life: 4000 })
    return false
  } finally {
    saving.value = false
  }
}

async function applyLlamaSwapFromModelConfig() {
  applyingLlamaSwap.value = true
  try {
    if (hasUnsavedChanges.value) {
      const ok = await saveConfig()
      if (!ok) return
    }
    await enginesStore.applySwapConfig()
    toast.add({
      severity: 'success',
      summary: 'llama-swap applied',
      detail: 'llama-swap-config.yaml was regenerated and the proxy reloaded.',
      life: 4000,
    })
    refreshSavedCmdPreviewIfVisible()
  } catch (e) {
    const detail = formatAxiosDetail(e) || 'Apply failed'
    toast.add({ severity: 'error', summary: 'Apply failed', detail, life: 5000 })
  } finally {
    applyingLlamaSwap.value = false
  }
}

// ── Reset ──────────────────────────────────────────────────
function resetConfig() {
  config.value = JSON.parse(JSON.stringify(savedConfig.value))
  const engine = config.value.engine
  const sec = (config.value.engines && config.value.engines[engine]) || {}
  setActiveKeysFromSection(sec, catalogParamList.value)
  applyEngineSectionToForm(engine)
  toast.add({ severity: 'info', summary: 'Reset', detail: 'Config reset to saved values', life: 2000 })
}

async function fetchSavedCmdPreview() {
  if (!model.value) return
  cmdPreviewLoading.value = true
  cmdPreviewError.value = null
  try {
    const { data } = await axios.get(modelApiUrl('/saved-llama-swap-cmd'))
    if (data?.ok && data.cmd) {
      cmdPreviewText.value = data.cmd
      cmdPreviewEnvText.value = formatEnvPreviewLines(data.env)
      cmdPreviewMacrosText.value = formatMacrosPreview(data.macros)
      cmdPreviewFiltersText.value = formatFiltersPreview(data.filters)
      cmdPreviewAliasesText.value = formatAliasesPreview(data.aliases)
      cmdPreviewError.value = null
    } else {
      cmdPreviewText.value = ''
      cmdPreviewEnvText.value = ''
      cmdPreviewMacrosText.value = ''
      cmdPreviewFiltersText.value = ''
      cmdPreviewAliasesText.value = ''
      cmdPreviewError.value = data?.error || 'Could not build saved command.'
    }
  } catch (e) {
    cmdPreviewText.value = ''
    cmdPreviewEnvText.value = ''
    cmdPreviewMacrosText.value = ''
    cmdPreviewFiltersText.value = ''
    cmdPreviewAliasesText.value = ''
    cmdPreviewError.value = formatAxiosDetail(e) || 'Could not load saved command.'
  } finally {
    cmdPreviewLoading.value = false
  }
}

async function fetchUnsavedCmdPreview() {
  if (!model.value || loading.value) return
  if (paramRegistry.value.scan_pending) return
  if (unsavedPreviewAbort) unsavedPreviewAbort.abort()
  unsavedPreviewAbort = new AbortController()
  const { signal } = unsavedPreviewAbort
  const requestId = ++unsavedPreviewRequestId
  unsavedCmdPreviewLoading.value = true
  unsavedCmdPreviewError.value = null
  try {
    const payload = buildPersistedPayload(config.value)
    const { data } = await axios.post(modelApiUrl('/preview-llama-swap-cmd'), payload, {
      signal,
    })
    if (requestId !== unsavedPreviewRequestId) return
    if (data?.ok && data.cmd) {
      unsavedCmdPreviewText.value = data.cmd
      unsavedCmdPreviewEnvText.value = formatEnvPreviewLines(data.env)
      unsavedCmdPreviewMacrosText.value = formatMacrosPreview(data.macros)
      unsavedCmdPreviewFiltersText.value = formatFiltersPreview(data.filters)
      unsavedCmdPreviewAliasesText.value = formatAliasesPreview(data.aliases)
      unsavedCmdPreviewError.value = null
    } else {
      unsavedCmdPreviewText.value = ''
      unsavedCmdPreviewEnvText.value = ''
      unsavedCmdPreviewMacrosText.value = ''
      unsavedCmdPreviewFiltersText.value = ''
      unsavedCmdPreviewAliasesText.value = ''
      unsavedCmdPreviewError.value = data?.error || 'Could not build preview command.'
    }
  } catch (e) {
    if (axios.isCancel?.(e) || e?.code === 'ERR_CANCELED' || e?.name === 'CanceledError') {
      return
    }
    if (requestId !== unsavedPreviewRequestId) return
    unsavedCmdPreviewText.value = ''
    unsavedCmdPreviewEnvText.value = ''
    unsavedCmdPreviewMacrosText.value = ''
    unsavedCmdPreviewFiltersText.value = ''
    unsavedCmdPreviewAliasesText.value = ''
    unsavedCmdPreviewError.value = formatAxiosDetail(e) || 'Could not build preview command.'
  } finally {
    if (requestId === unsavedPreviewRequestId) {
      unsavedCmdPreviewLoading.value = false
    }
  }
}

watch(
  [
    () => loading.value,
    () => model.value?.id,
    () => config.value.engine,
    () => activeParamKeys.value.slice(),
    () => JSON.stringify(buildEngineStashFromForm(config.value)),
  ],
  () => {
    if (unsavedPreviewTimer) clearTimeout(unsavedPreviewTimer)
    if (loading.value || !model.value || paramRegistry.value.scan_pending) {
      unsavedCmdPreviewLoading.value = false
      return
    }
    if (!cmdPreviewDialogVisible.value || cmdPreviewDialogMode.value !== 'unsaved') {
      return
    }
    unsavedPreviewTimer = window.setTimeout(() => {
      void fetchUnsavedCmdPreview()
    }, 700)
  },
  { deep: false },
)

// ── Lifecycle ──────────────────────────────────────────────
onMounted(loadAll)
onBeforeUnmount(() => {
  if (unsavedPreviewTimer) clearTimeout(unsavedPreviewTimer)
  if (gpuInfoRetryTimer) clearTimeout(gpuInfoRetryTimer)
  if (unsavedPreviewAbort) unsavedPreviewAbort.abort()
})
</script>

<style scoped>
/* layout: .page-shell.page-shell--relaxed */

.config-scan-message {
  margin-bottom: 1rem;
}

.config-page-title {
  display: flex;
  flex-direction: column;
  align-items: flex-start;
  gap: 0.4rem;
  min-width: 0;
}

.header-meta {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  flex-wrap: wrap;
}

.textarea-cli {
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
  font-size: 0.875rem;
}

.swap-env-row {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  margin-bottom: 0.5rem;
}

.swap-env-key {
  flex: 0 0 12rem;
  min-width: 0;
}

.set-params-variant {
  margin-bottom: 1rem;
  padding-bottom: 0.75rem;
  border-bottom: 1px solid var(--surface-border, #374151);
}

.set-params-variant:last-of-type {
  border-bottom: none;
}

.set-params-variant__header {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  margin-bottom: 0.5rem;
}

.set-params-sub-id {
  flex: 1;
  min-width: 0;
}

.set-params-kwargs-label {
  font-size: 0.85rem;
  margin-bottom: 0.35rem;
}

.cmd-preview-env-label {
  margin-top: 0.75rem;
  margin-bottom: 0.35rem;
  font-size: 0.85rem;
}

.config-cmd-actions-card {
  margin-bottom: 1rem;
}

.config-cmd-actions {
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
}

.cmd-preview-dialog-hint {
  margin: 0 0 1rem;
  font-size: 0.875rem;
  color: var(--text-secondary, #9ca3af);
  line-height: 1.45;
}

.cmd-preview-dialog .cmd-preview-textarea {
  max-height: 40vh;
  overflow-y: auto;
}

.cmd-preview-loading {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  font-size: 0.875rem;
  color: var(--text-secondary, #9ca3af);
}

.cmd-preview-loading .pi-spinner {
  font-size: 1.1rem;
  color: var(--accent-cyan, #22d3ee);
}

.cmd-preview-message {
  margin: 0;
}

.cmd-preview-textarea {
  min-height: 8rem;
  white-space: pre-wrap;
  word-break: break-word;
  line-height: 1.45;
}

.hf-link {
  font-size: 0.875rem;
  color: var(--accent-cyan, #22d3ee);
  text-decoration: none;
  display: flex;
  align-items: center;
  gap: 0.25rem;
}

.hf-link:hover { text-decoration: underline; }

.param-field--unsupported {
  opacity: 0.88;
}

.param-supported-tag {
  margin-left: 0.35rem;
  vertical-align: middle;
  font-size: 0.65rem !important;
}

/* ── Card ─────────────────────────────────────────────── */
.config-card {
  background: var(--bg-card, #161b2e);
  border: 1px solid var(--border-primary, #2a2f45);
  border-radius: var(--radius-lg, 0.75rem);
  padding: 1.25rem;
}

.section-label {
  font-size: 0.75rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--text-secondary, #9ca3af);
  margin-bottom: 0.875rem;
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.section-hint {
  font-weight: 400;
  text-transform: none;
  letter-spacing: normal;
  color: var(--text-secondary, #9ca3af);
  opacity: 0.7;
}

/* ── Engine selector ──────────────────────────────────── */
.engine-selector {
  display: flex;
  gap: 0.5rem;
  flex-wrap: wrap;
}

.engine-option {
  display: inline-flex;
  align-items: center;
  padding: 0.5rem 1rem;
  border-radius: var(--radius-md, 0.5rem);
  border: 1px solid var(--border-primary, #2a2f45);
  cursor: pointer;
  transition: all 0.15s;
  font-size: 0.875rem;
  user-select: none;
}

.engine-option:hover {
  border-color: var(--accent-cyan, #22d3ee);
  background: rgba(34, 211, 238, 0.05);
}

.engine-option.selected {
  border-color: var(--accent-cyan, #22d3ee);
  background: rgba(34, 211, 238, 0.1);
  color: var(--accent-cyan, #22d3ee);
  font-weight: 600;
}

.engine-option-label {
  display: inline-flex;
  align-items: center;
  gap: 0.5rem;
}

.engine-name {
  font-size: 0.875rem;
}

.engine-mark {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  min-width: 1.6rem;
  height: 1.6rem;
  padding: 0 0.4rem;
  border-radius: 999px;
  font-size: 0.7rem;
  font-weight: 700;
  line-height: 1;
  letter-spacing: 0.04em;
  color: #fff;
  box-shadow: inset 0 0 0 1px rgba(255, 255, 255, 0.1);
}

.engine-mark--llama {
  background: linear-gradient(135deg, #0ea5e9, #2563eb);
}

.engine-mark--ik {
  background: linear-gradient(135deg, #8b5cf6, #ec4899);
}

.engine-icon-lmdeploy {
  font-size: 1.1rem;
  color: var(--accent-cyan, #22d3ee);
}

.engine-icon-onecat-vllm {
  font-size: 1.1rem;
  color: var(--accent-amber, #f59e0b);
}

/* ── Params grid ──────────────────────────────────────── */
.params-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(240px, 1fr));
  gap: 0.875rem;
}

.param-field {
  display: flex;
  flex-direction: column;
  gap: 0.25rem;
}

.param-field__head .param-field__label {
  font-size: 0.8rem;
  font-weight: 500;
  color: var(--text-secondary, #9ca3af);
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 0.3rem;
  flex: 1;
  min-width: 0;
}

.param-input { width: 100%; }

.param-info {
  font-size: 0.7rem;
  cursor: help;
  opacity: 0.6;
}

/* ── Actions (sticky bar) ───────────────────────────────── */
.config-actions {
  display: flex;
  gap: 0.75rem;
  justify-content: flex-end;
  padding-bottom: var(--spacing-lg, 1.5rem);
  position: sticky;
  bottom: 0;
  z-index: 10;
  background: linear-gradient(
    to top,
    var(--bg-primary, #0f111a) 65%,
    transparent
  );
  padding-top: 0.75rem;
  margin-top: 0.5rem;
}

.param-slider-row {
  display: flex;
  flex-direction: column;
  gap: 0.25rem;
  margin-bottom: 0.25rem;
}

.param-slider {
  width: 100%;
  max-width: 15rem;
}

/* Align PrimeVue slider handle with the track bar */
.param-slider :deep(.p-slider) {
  height: 0.5rem;
}
.param-slider :deep(.p-slider-handle) {
  width: 1rem;
  height: 1rem;
  top: 50%;
  margin-top: -0.5rem;
}

.param-hint {
  font-size: 0.75rem;
  color: var(--text-secondary, #9ca3af);
}

/* ── Unsaved indicator ─────────────────────────────────── */
.unsaved-tag {
  font-size: 0.75rem;
}

/* ── Catalog toolbar (search, toggles, jump nav) ───────── */
.config-toolbar {
  margin-bottom: 1rem;
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
}

.config-toolbar__row {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 0.5rem;
}

.config-toolbar__toggles {
  gap: 1rem;
}

.config-search-wrap {
  position: relative;
  flex: 1;
  min-width: min(100%, 12rem);
}

.config-search-wrap .pi-search {
  position: absolute;
  left: 0.75rem;
  top: 50%;
  transform: translateY(-50%);
  color: var(--text-secondary, #9ca3af);
  pointer-events: none;
  z-index: 1;
  font-size: 0.875rem;
}

.config-search-wrap :deep(.p-inputtext),
.config-search-wrap :deep(input.config-search-input) {
  width: 100%;
  padding-left: 2.35rem;
}

.toggle-field {
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.toggle-field label {
  font-size: 0.875rem;
  color: var(--text-secondary, #9ca3af);
  cursor: pointer;
  user-select: none;
}

.config-search-hint-card,
.config-search-tags-card,
.config-params-pane {
  margin-bottom: 1rem;
}

.config-muted-hint,
.config-tag-lead {
  margin: 0;
  font-size: 0.875rem;
  line-height: 1.55;
  color: var(--text-secondary, #9ca3af);
}

.config-tag-lead {
  margin-bottom: 0.75rem;
}

.param-tag-cloud {
  display: flex;
  flex-wrap: wrap;
  gap: 0.45rem;
}

.param-search-tag {
  display: inline-flex;
  align-items: center;
  gap: 0.35rem;
  flex-wrap: wrap;
  max-width: 100%;
  padding: 0.35rem 0.65rem;
  border-radius: 999px;
  border: 1px solid var(--border-primary, #2a2f45);
  background: color-mix(in srgb, var(--accent-cyan, #22d3ee) 8%, var(--bg-card, #161b2e));
  color: var(--text-primary, #e5e7eb);
  font-size: 0.8125rem;
  cursor: pointer;
  transition:
    border-color 0.15s ease,
    background 0.15s ease,
    transform 0.12s ease;
}

.param-search-tag:hover {
  border-color: var(--accent-cyan, #22d3ee);
  background: color-mix(in srgb, var(--accent-cyan, #22d3ee) 14%, var(--bg-card, #161b2e));
  transform: translateY(-1px);
}

.param-search-tag:focus-visible {
  outline: 2px solid var(--accent-cyan, #22d3ee);
  outline-offset: 2px;
}

.param-search-tag__key {
  font-size: 0.68rem;
  padding: 0.1rem 0.35rem;
  border-radius: 0.25rem;
  background: rgba(0, 0, 0, 0.3);
  color: var(--text-secondary, #9ca3af);
}

.param-field__head {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 0.5rem;
  margin-bottom: 0.35rem;
}

.param-remove-btn {
  flex-shrink: 0;
  margin-top: -0.15rem;
}

.section-params {
  padding-top: 0.5rem;
  border-top: 1px solid var(--border-primary, #2a2f45);
}

.param-key-hint {
  margin-left: 0.35rem;
  padding: 0.1rem 0.35rem;
  font-size: 0.65rem;
  font-weight: 400;
  color: var(--text-secondary, #9ca3af);
  background: rgba(0, 0, 0, 0.25);
  border-radius: 0.25rem;
  vertical-align: middle;
}

.config-templates-lead {
  margin: 0 0 1rem;
  font-size: 0.875rem;
  color: var(--text-secondary, #9ca3af);
  line-height: 1.45;
}

.config-templates-section {
  margin-bottom: 1.25rem;
  padding-bottom: 1.25rem;
  border-bottom: 1px solid var(--border-primary, #2a2f45);
}

.config-templates-section--apply {
  border-bottom: none;
}

.config-templates-field {
  margin-bottom: 0.75rem;
}

.config-templates-field label {
  display: block;
  font-size: 0.8rem;
  margin-bottom: 0.35rem;
  color: var(--text-secondary, #9ca3af);
}

.config-templates-check {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  margin-bottom: 0.75rem;
  font-size: 0.875rem;
}

.config-templates-apply-actions {
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
}

.config-templates-list {
  list-style: none;
  margin: 0;
  padding: 0;
}

.config-templates-list-item {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 0.5rem;
  padding: 0.5rem 0;
  border-bottom: 1px solid var(--border-primary, #2a2f45);
}

.config-templates-list-item:last-child {
  border-bottom: none;
}

.config-templates-list-main {
  display: flex;
  flex-direction: column;
  gap: 0.15rem;
  min-width: 0;
}

.config-templates-list-desc {
  font-size: 0.85rem;
  color: var(--text-secondary, #9ca3af);
}

.config-templates-list-meta {
  font-size: 0.75rem;
  color: var(--text-secondary, #9ca3af);
  opacity: 0.85;
}
</style>
