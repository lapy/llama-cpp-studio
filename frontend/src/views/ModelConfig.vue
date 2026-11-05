<template>
  <div class="model-config-view page-shell page-shell--wide">
    <LoadingState v-if="loading && !model" message="Loading configuration…" />

    <EmptyState
      v-else-if="!model && loadError"
      icon="pi pi-exclamation-circle"
      title="Could not load configuration"
      :description="loadError"
    >
      <Button label="Retry" icon="pi pi-refresh" @click="loadAll" />
      <Button
        label="Back to Models"
        icon="pi pi-arrow-left"
        severity="secondary"
        outlined
        @click="$router.push('/models')"
      />
    </EmptyState>

    <EmptyState v-else-if="!model" icon="pi pi-exclamation-circle" title="Model not found">
      <Button label="Back to Models" icon="pi pi-arrow-left" @click="$router.push('/models')" />
    </EmptyState>

    <template v-else>
      <PageHeader>
        <template #start>
          <Button
            icon="pi pi-arrow-left"
            text
            severity="secondary"
            aria-label="Back to Models"
            @click="requestLeave('/models')"
          />
        </template>
        <template #title>
          <div class="config-page-title">
            <h1 class="page-title">{{ model.display_name || model.base_model_name }}</h1>
            <div class="header-meta">
              <Tag :value="formatLabel(model.format || 'gguf')" severity="info" />
              <Tag v-if="model.quantization" :value="model.quantization" severity="secondary" />
              <Tag v-if="model.family" :value="model.family" severity="secondary" />
              <Tag
                v-for="task in model.tasks || []"
                :key="task"
                :value="taskLabel(task)"
                severity="success"
              />
              <a
                v-if="model.huggingface_id"
                :href="`https://huggingface.co/${model.huggingface_id}`"
                target="_blank"
                class="hf-link"
              >
                <i class="pi pi-external-link" /> {{ model.huggingface_id }}
              </a>
            </div>
          </div>
        </template>
        <template v-if="!loading && model" #actions>
          <Button
            v-if="canCheckForUpdates"
            label="Check for updates"
            icon="pi pi-refresh"
            size="small"
            severity="secondary"
            outlined
            :loading="refreshingModel"
            :disabled="refreshingModel"
            @click="checkForModelUpdates"
          />
          <Button
            v-if="isAudioEngine"
            label="Open Audio"
            icon="pi pi-volume-up"
            size="small"
            severity="secondary"
            outlined
            @click="openAudioWorkspace"
          />
          <Tag
            v-if="hasUnsavedChanges"
            value="Unsaved changes"
            severity="warn"
            class="unsaved-tag"
          />
        </template>
      </PageHeader>

      <div class="config-status" aria-live="polite">
        <strong class="runtime-state">{{
          hasUnsavedChanges
            ? 'Unsaved changes'
            : pendingApply
              ? 'Saved · awaiting application'
              : publishedStepLabel
        }}</strong>
        <p class="runtime-state__detail" :title="runtimeStateDetail">{{ runtimeStateDetail }}</p>
      </div>

      <div v-if="modelLaunchPlan?.action === 'invalid'" class="state-banner" role="alert">
        {{ modelLaunchPlan.reasons.join(' ') }} Fix the configuration before applying settings.
      </div>
      <div v-if="loadError" class="state-banner" role="alert">
        <span>Could not refresh this configuration. {{ loadError }}</span>
        <Button label="Retry" size="small" severity="secondary" outlined @click="loadAll" />
      </div>

      <div v-if="draftOffer" class="state-banner" role="status">
        <span>A draft of your edits for this model and engine is saved in this session.</span>
        <Button label="Restore draft" size="small" @click="restoreDraft" />
        <Button
          label="Discard draft"
          size="small"
          severity="secondary"
          outlined
          @click="discardDraftOffer"
        />
      </div>

      <div class="config-card config-workbench">
        <div class="section-label section-label--inline">Engine</div>
        <div
          class="engine-selector"
          role="radiogroup"
          aria-label="Engine"
          @keydown="onEngineKeydown"
        >
          <button
            v-for="eng in visibleEngineOptions"
            :key="eng.value"
            type="button"
            class="engine-option"
            role="radio"
            :class="{
              selected: config.engine === eng.value,
              disabled: eng.disabled,
            }"
            :aria-checked="config.engine === eng.value ? 'true' : 'false'"
            :aria-disabled="eng.disabled ? 'true' : 'false'"
            :disabled="eng.disabled"
            :tabindex="engineTabIndex(eng)"
            v-tooltip.bottom="
              eng.disabledReason ||
              (eng.runnable || !eng.descriptor
                ? ''
                : 'No active version. Activate one from Engines.')
            "
            @click="changeEngine(eng.value)"
          >
            <div class="engine-option-label">
              <template v-if="engineMark(eng.value)">
                <span
                  v-if="engineMark(eng.value).text"
                  :class="['engine-mark', engineMark(eng.value).className]"
                  aria-hidden="true"
                  >{{ engineMark(eng.value).text }}</span
                >
                <i v-else :class="engineMark(eng.value).icon" aria-hidden="true" />
              </template>
              <span class="engine-name">{{ eng.label }}</span>
            </div>
            <small v-if="eng.disabledReason" class="engine-disabled-reason">
              {{ eng.disabledReason }}
            </small>
          </button>
        </div>
        <p v-if="!visibleEngineOptions.length" class="config-muted-hint">
          No installed engine is compatible with this model.
        </p>
        <p v-else-if="unavailableSavedEngine" class="config-muted-hint">
          {{ unavailableSavedEngine.label }} is saved for this model. It is hidden because it is not
          installed or not compatible.
        </p>
        <div v-if="isSglangEngine" class="param-field sglang-runtime-selector">
          <label for="sglang-runtime">SGLang runtime</label>
          <Select
            input-id="sglang-runtime"
            :model-value="sglangRuntime"
            :options="sglangRuntimeOptions"
            option-label="label"
            option-value="value"
            @update:model-value="changeSglangRuntime"
          />
          <small class="config-muted-hint"
            >Each runtime has its own CLI parameters. Saved values are kept when switching.</small
          >
        </div>
      </div>

      <div class="config-section-tabs" role="tablist" aria-label="Configuration sections">
        <button
          v-for="tab in pageTabs"
          :key="tab.id"
          type="button"
          :id="`config-tab-${tab.id}`"
          role="tab"
          class="config-section-tab"
          :class="{ selected: pageTab === tab.id }"
          :aria-selected="pageTab === tab.id"
          :aria-controls="`config-panel-${tab.id}`"
          :tabindex="pageTab === tab.id ? 0 : -1"
          @click="pageTab = tab.id"
          @keydown="onRovingTabKeydown"
        >
          <span class="engine-option-label">
            <i :class="tab.icon" aria-hidden="true" />
            <span>{{ tab.label }}</span>
          </span>
        </button>
      </div>

      <div
        v-if="!isAudioEngine"
        v-show="pageTab === 'runtime'"
        id="config-panel-runtime"
        role="tabpanel"
        aria-labelledby="config-tab-runtime"
        tabindex="0"
        class="config-tab-panel"
      >
        <div class="config-card">
          <div v-if="basicParams.length" class="workbench-block">
            <div class="section-label section-label--inline">
              Basics
              <i
                class="pi pi-info-circle param-info"
                tabindex="0"
                aria-label="About basics"
                v-tooltip.top="
                  'Common settings for this engine. Default is inherited. Override is saved with this model.'
                "
              />
            </div>
            <div class="params-grid">
              <div v-for="param in basicParams" :key="param.key" class="param-field">
                <label :for="`basic-${param.key}`" class="param-field__label">
                  <span class="param-field__name">{{ param.label }}</span>
                  <code class="param-key-hint">{{ param.key }}</code>
                  <Tag
                    class="param-field__state"
                    :value="isExplicitOverride(param) ? 'Override' : 'Default'"
                    :severity="isExplicitOverride(param) ? 'info' : 'secondary'"
                  />
                </label>
                <InputNumber
                  :input-id="`basic-${param.key}`"
                  v-model="config[param.key]"
                  :placeholder="param.default != null ? String(param.default) : 'Engine default'"
                  class="param-input"
                  :disabled="param.supported === false"
                  :min="param.min"
                  :max="param.max"
                  :pt="{ pcInputText: { root: { 'aria-describedby': `basic-help-${param.key}` } } }"
                />
                <small :id="`basic-help-${param.key}`" class="param-default-help">{{
                  parameterHelp(param)
                }}</small>
                <button
                  v-if="isExplicitOverride(param)"
                  class="param-reset"
                  type="button"
                  @click="useParameterDefault(param)"
                >
                  Use engine default
                </button>
              </div>
            </div>
          </div>

          <div v-if="showNvidiaGpuBind" class="workbench-block gpu-bind">
            <div class="section-label section-label--inline">
              GPUs
              <i
                class="pi pi-info-circle param-info"
                tabindex="0"
                aria-label="About GPU assignment"
                v-tooltip.top="
                  'All GPUs follows the deployment. Choose GPUs to pick devices in the order you click them. CPU only hides every CUDA device. The first card you pick becomes device 0.'
                "
              />
            </div>
            <div
              id="gpu-mode"
              class="gpu-mode"
              role="radiogroup"
              aria-label="GPU assignment mode"
              @keydown="onGpuModeKeydown"
            >
              <button
                v-for="mode in gpuModeOptions"
                :key="mode.value"
                type="button"
                class="gpu-mode__option"
                role="radio"
                :class="{ selected: (config.gpu_mode || 'inherit') === mode.value }"
                :aria-checked="(config.gpu_mode || 'inherit') === mode.value ? 'true' : 'false'"
                :tabindex="(config.gpu_mode || 'inherit') === mode.value ? 0 : -1"
                @click="setGpuMode(mode.value)"
              >
                {{ mode.label }}
              </button>
            </div>
            <div
              v-if="(config.gpu_mode || 'inherit') === 'selected'"
              class="gpu-card-grid"
              role="group"
              aria-label="Select NVIDIA GPUs for CUDA_VISIBLE_DEVICES"
            >
              <button
                v-for="gpu in nvidiaGpuCards"
                :key="gpu.value"
                type="button"
                class="gpu-card"
                :class="{ selected: gpu.order != null }"
                :aria-pressed="gpu.order != null ? 'true' : 'false'"
                @click="toggleGpuCard(gpu.value)"
              >
                <span class="gpu-card__index" aria-hidden="true">{{ gpu.index }}</span>
                <span class="gpu-card__body">
                  <span class="gpu-card__name">{{ gpu.name }}</span>
                  <span class="gpu-card__meta">{{
                    gpu.order != null ? `Device ${gpu.order}` : 'Not used'
                  }}</span>
                </span>
              </button>
            </div>
          </div>

          <template v-if="!isAudioEngine && catalogSections.length">
            <div class="workbench-block">
              <div class="config-toolbar__row">
                <span class="p-input-icon-left config-search-wrap">
                  <i class="pi pi-search" aria-hidden="true" />
                  <InputText
                    v-model="paramSearchQuery"
                    type="search"
                    placeholder="Add a parameter…"
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
                <div class="toggle-field">
                  <ToggleSwitch
                    v-model="hideUnsupportedParams"
                    input-id="toggle-hide-unsupported"
                  />
                  <label for="toggle-hide-unsupported">Hide unsupported</label>
                </div>
              </div>
              <div v-if="paramSearchQuery.trim()" class="param-tag-cloud-wrap">
                <div class="section-label section-label--inline">Add parameter</div>
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
                  No parameters match. Try other words, turn off “hide unsupported”, or clear the
                  search.
                </Message>
              </div>
              <p v-if="!paneParams.length" class="config-muted-hint">
                Search above to add parameters. Saved values that differ from the engine default
                appear here.
              </p>
              <div v-else class="params-grid">
                <div
                  v-for="param in paneParams"
                  :key="`${param.sectionId}-${param.key}`"
                  class="param-field"
                  :class="{ 'param-field--unsupported': param.supported === false }"
                >
                  <div class="param-field__head">
                    <label :for="`p-${param.sectionId}-${param.key}`" class="param-field__label">
                      <span class="param-field__name">
                        {{ param.label }}
                        <Tag
                          v-if="param.supported === false"
                          value="Not in this build"
                          severity="secondary"
                          class="param-supported-tag"
                        />
                        <i
                          class="pi pi-info-circle param-info"
                          tabindex="0"
                          :aria-label="`About ${param.label}`"
                          v-tooltip.top="paramDescriptionTooltip(param)"
                        />
                      </span>
                      <code class="param-key-hint">{{ param.key }}</code>
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
                  <template
                    v-if="
                      param.type === 'int' &&
                      (param.key === 'ctx_size' || param.key === 'session_len')
                    "
                  >
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
                      :input-id="`p-${param.sectionId}-${param.key}`"
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
                      :input-id="`p-${param.sectionId}-${param.key}`"
                      v-model="config[param.key]"
                      :placeholder="String(param.default ?? '')"
                      class="param-input"
                      :disabled="param.supported === false"
                    />
                  </template>
                  <Select
                    v-else-if="param.value_kind === 'flag' && param.negative_flag"
                    :input-id="`p-${param.sectionId}-${param.key}`"
                    v-model="config[param.key]"
                    :options="triStateOptions"
                    optionLabel="label"
                    optionValue="value"
                    placeholder="Default"
                    showClear
                    class="param-input"
                    :disabled="param.supported === false"
                  />
                  <InputTags
                    v-else-if="['repeatable', 'space_list'].includes(param.value_kind)"
                    :input-id="`p-${param.sectionId}-${param.key}`"
                    v-model="config[param.key]"
                    delimiter=","
                    class="param-input"
                    :disabled="param.supported === false"
                  />
                  <MultiSelect
                    v-else-if="isDelimitedEnumParam(param)"
                    :input-id="`p-${param.sectionId}-${param.key}`"
                    v-model="config[param.key]"
                    :options="param.options || []"
                    optionLabel="label"
                    optionValue="value"
                    display="chip"
                    :placeholder="csvEnumPlaceholder(param)"
                    class="param-input w-full"
                    :disabled="param.supported === false"
                  />
                  <Select
                    v-else-if="param.options && param.options.length"
                    :input-id="`p-${param.sectionId}-${param.key}`"
                    v-model="config[param.key]"
                    :options="param.options"
                    optionLabel="label"
                    optionValue="value"
                    :placeholder="param.default != null ? String(param.default) : ''"
                    :showClear="param.required !== true"
                    class="param-input"
                    :disabled="param.supported === false"
                  />
                  <InputNumber
                    v-else-if="param.type === 'int'"
                    :input-id="`p-${param.sectionId}-${param.key}`"
                    v-model="config[param.key]"
                    :placeholder="String(param.default ?? '')"
                    class="param-input"
                    :disabled="param.supported === false"
                  />
                  <InputNumber
                    v-else-if="param.type === 'float'"
                    :input-id="`p-${param.sectionId}-${param.key}`"
                    v-model="config[param.key]"
                    :minFractionDigits="1"
                    :maxFractionDigits="4"
                    :placeholder="String(param.default ?? '')"
                    class="param-input"
                    :disabled="param.supported === false"
                  />
                  <ToggleSwitch
                    v-else-if="param.type === 'bool'"
                    :input-id="`p-${param.sectionId}-${param.key}`"
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
                  <small class="param-default-help">{{ parameterHelp(param) }}</small>
                  <button
                    v-if="isExplicitOverride(param)"
                    class="param-reset"
                    type="button"
                    @click="useParameterDefault(param)"
                  >
                    Use engine default
                  </button>
                </div>
              </div>
            </div>
          </template>
        </div>
      </div>

      <Message
        v-if="!isAudioEngine && selectedParamRegistry.scan_error"
        severity="warn"
        :closable="false"
        class="config-scan-message"
      >
        Could not read
        {{ isSglangEngine && sglangRuntime === 'diffusion' ? 'SGLang diffusion' : 'engine' }} CLI
        help: {{ selectedParamRegistry.scan_error }}. Open Engines and use
        <strong>Rescan CLI parameters</strong> for this engine.
      </Message>
      <Message
        v-else-if="!isAudioEngine && selectedParamRegistry.scan_pending"
        severity="info"
        :closable="false"
        class="config-scan-message"
      >
        {{
          isSglangEngine && sglangRuntime === 'diffusion'
            ? 'SGLang diffusion CLI parameters are'
            : 'CLI parameters are'
        }}
        not loaded for this engine yet. Activate the engine on the Engines page (or use
        <strong>Rescan CLI parameters</strong> there), then reopen this page.
      </Message>
      <Message
        v-if="unrecognizedSavedKeys.length"
        severity="warn"
        :closable="false"
        class="config-scan-message"
      >
        {{
          isAudioEngine
            ? 'Unrecognized audio.cpp keys are preserved on save:'
            : isSglangEngine
              ? 'Unrecognized SGLang keys are preserved on save:'
              : 'Deprecated or unrecognized saved keys for this engine will be dropped on the next save:'
        }}
        <code>{{ unrecognizedSavedKeys.join(', ') }}</code>
      </Message>
      <Message
        v-for="warning in paramRegistry.compatibility_warnings || []"
        :key="warning"
        severity="warn"
        :closable="false"
        class="config-scan-message"
      >
        {{ warning }}
      </Message>

      <div
        v-if="isAudioEngine && pageTab === 'server' && showNvidiaGpuBind"
        class="config-card gpu-bind"
      >
        <div class="section-label section-label--inline">
          GPUs
          <i
            class="pi pi-info-circle param-info"
            tabindex="0"
            aria-label="About GPU assignment"
            v-tooltip.top="
              'All GPUs follows the deployment. Choose GPUs to pick devices in the order you click them. CPU only hides every CUDA device. The first card you pick becomes device 0.'
            "
          />
        </div>
        <div
          id="gpu-mode"
          class="gpu-mode"
          role="radiogroup"
          aria-label="GPU assignment mode"
          @keydown="onGpuModeKeydown"
        >
          <button
            v-for="mode in gpuModeOptions"
            :key="mode.value"
            type="button"
            class="gpu-mode__option"
            role="radio"
            :class="{ selected: (config.gpu_mode || 'inherit') === mode.value }"
            :aria-checked="(config.gpu_mode || 'inherit') === mode.value ? 'true' : 'false'"
            :tabindex="(config.gpu_mode || 'inherit') === mode.value ? 0 : -1"
            @click="setGpuMode(mode.value)"
          >
            {{ mode.label }}
          </button>
        </div>
        <div
          v-if="(config.gpu_mode || 'inherit') === 'selected'"
          class="gpu-card-grid"
          role="group"
          aria-label="Select NVIDIA GPUs for CUDA_VISIBLE_DEVICES"
        >
          <button
            v-for="gpu in nvidiaGpuCards"
            :key="gpu.value"
            type="button"
            class="gpu-card"
            :class="{ selected: gpu.order != null }"
            :aria-pressed="gpu.order != null ? 'true' : 'false'"
            @click="toggleGpuCard(gpu.value)"
          >
            <span class="gpu-card__index" aria-hidden="true">{{ gpu.index }}</span>
            <span class="gpu-card__body">
              <span class="gpu-card__name">{{ gpu.name }}</span>
              <span class="gpu-card__meta">{{
                gpu.order != null ? `Device ${gpu.order}` : 'Not used'
              }}</span>
            </span>
          </button>
        </div>
      </div>

      <AudioModelConfig
        v-if="isAudioEngine"
        v-show="pageTab !== 'launch'"
        hide-tabs
        :active-tab="pageTab"
        :config="config"
        :param-registry="paramRegistry"
        :stable-model-id="stableModelId"
        :model-id="model?.id || ''"
        @rescan-complete="fetchParamRegistry(audioEngineId, { rescan: true })"
      />

      <div
        v-show="pageTab === 'launch'"
        id="config-panel-launch"
        role="tabpanel"
        aria-labelledby="config-tab-launch"
        tabindex="0"
        class="config-tab-panel"
      >
        <div class="config-card config-launch">
          <div v-if="showCompanionsCard" class="advanced-block">
            <div class="section-label section-label--inline">
              Companions
              <i
                class="pi pi-info-circle param-info"
                tabindex="0"
                aria-label="About companions"
                v-tooltip.top="
                  'Attach mmproj, MTP, or DFlash weights. MTP and DFlash cannot be used together.'
                "
              />
            </div>
            <div v-if="companionsLoading" class="companions-loading">
              Loading companion options…
            </div>
            <div v-else class="companions-grid">
              <div class="companion-field">
                <label for="config-mmproj">Projector (mmproj)</label>
                <div class="companion-field__row">
                  <Select
                    id="config-mmproj"
                    v-model="selectedMmproj"
                    :options="mmprojOptions"
                    optionLabel="label"
                    optionValue="value"
                    class="w-full"
                    :disabled="companionBusy"
                    placeholder="None"
                  />
                  <Button
                    label="Apply"
                    icon="pi pi-save"
                    size="small"
                    severity="success"
                    outlined
                    :loading="companionBusyField === 'mmproj'"
                    :disabled="companionBusy || !mmprojSelectionChanged"
                    @click="applyCompanion('mmproj')"
                  />
                </div>
              </div>
              <div class="companion-field">
                <label for="config-mtp">MTP draft</label>
                <div class="companion-field__row">
                  <Select
                    id="config-mtp"
                    v-model="selectedMtp"
                    :options="mtpOptions"
                    optionLabel="label"
                    optionValue="value"
                    class="w-full"
                    :disabled="companionBusy"
                    placeholder="None"
                    @update:model-value="onMtpSelected"
                  />
                  <Button
                    label="Apply"
                    icon="pi pi-save"
                    size="small"
                    severity="success"
                    outlined
                    :loading="companionBusyField === 'mtp'"
                    :disabled="companionBusy || !mtpSelectionChanged"
                    @click="applyCompanion('mtp')"
                  />
                </div>
              </div>
              <div class="companion-field">
                <label for="config-dflash">DFlash draft</label>
                <div class="companion-field__row">
                  <Select
                    id="config-dflash"
                    v-model="selectedDflash"
                    :options="dflashOptions"
                    optionLabel="label"
                    optionValue="value"
                    class="w-full"
                    :disabled="companionBusy"
                    placeholder="None"
                    @update:model-value="onDflashSelected"
                  />
                  <Button
                    label="Apply"
                    icon="pi pi-save"
                    size="small"
                    severity="success"
                    outlined
                    :loading="companionBusyField === 'dflash'"
                    :disabled="companionBusy || !dflashSelectionChanged"
                    @click="applyCompanion('dflash')"
                  />
                </div>
              </div>
            </div>
          </div>

          <div class="config-split-grid split-grid">
            <div class="advanced-field">
              <label class="advanced-field__label">
                {{ stableIdLabel }}
              </label>
              <InputText
                :model-value="stableModelId"
                readonly
                class="w-full"
                :aria-label="stableIdAria"
              />
            </div>

            <div class="advanced-field">
              <label class="advanced-field__label">
                {{ aliasLabel }}
                <i
                  class="pi pi-info-circle param-info"
                  tabindex="0"
                  aria-label="About API alias"
                  v-tooltip.top="
                    aliasTooltip
                  "
                />
              </label>
              <InputText
                v-model="config.model_alias"
                placeholder="e.g. my-app-model"
                class="w-full"
              />
            </div>
          </div>

          <RequestAliasesEditor
            :key="config.engine"
            v-model="config.set_params_by_id"
            :base-name="requestAliasBase"
            @validity="requestAliasesValid = $event"
          />

          <div class="advanced-block">
            <div class="section-label section-label--inline">
              Custom arguments
              <i
                class="pi pi-info-circle param-info"
                tabindex="0"
                aria-label="About custom arguments"
                v-tooltip.top="'Raw CLI flags appended to the server command.'"
              />
            </div>
            <Textarea
              v-model="config.custom_args"
              rows="2"
              placeholder="e.g. --some-flag value --another-flag"
              class="w-full textarea-cli"
              autoResize
            />
            <Button
              v-if="canImportCommand"
              label="Parse into parameters"
              icon="pi pi-sparkles"
              size="small"
              severity="secondary"
              outlined
              class="mt-2"
              type="button"
              @click="openParseCommandDialog(config.custom_args)"
            />
          </div>
          <div class="advanced-block">
            <div class="section-label section-label--inline">
              Environment
              <i
                class="pi pi-info-circle param-info"
                tabindex="0"
                aria-label="About environment variables"
                v-tooltip.top="
                  'Environment variables passed to this model process. CUDA_VISIBLE_DEVICES is set from the GPU control when NVIDIA GPUs are detected. LLAMA_STUDIO_* keys are reserved and ignored.'
                "
              />
            </div>
            <div
              v-for="item in runtimeEnvRowsDisplayed"
              :key="item.originalIndex"
              class="runtime-env-row"
            >
              <InputText
                :model-value="item.row.key"
                placeholder="VAR_NAME"
                class="runtime-env-key"
                aria-label="Environment variable name"
                @update:model-value="
                  (v) => {
                    item.row.key = v
                    syncRuntimeEnvFromRows()
                  }
                "
              />
              <Select
                :model-value="item.row.mode || 'set'"
                :options="envModeOptions"
                option-label="label"
                option-value="value"
                class="runtime-env-mode"
                aria-label="Environment variable mode"
                @update:model-value="
                  (v) => {
                    item.row.mode = v
                    syncRuntimeEnvFromRows()
                  }
                "
              />
              <InputText
                v-if="(item.row.mode || 'set') !== 'unset'"
                :model-value="item.row.value"
                placeholder="value"
                class="runtime-env-value"
                aria-label="Environment variable value"
                @update:model-value="
                  (v) => {
                    item.row.value = v
                    syncRuntimeEnvFromRows()
                  }
                "
              />
              <Button
                icon="pi pi-trash"
                severity="danger"
                text
                rounded
                type="button"
                aria-label="Remove variable"
                @click="removeRuntimeEnvRow(item.originalIndex)"
              />
            </div>
            <Button
              label="Add variable"
              icon="pi pi-plus"
              severity="secondary"
              outlined
              type="button"
              class="mt-2"
              @click="addRuntimeEnvRow"
            />
          </div>
          <div class="advanced-block">
            <div class="section-label section-label--inline">Command</div>
            <div class="config-cmd-actions">
              <Button
                label="Live preview"
                icon="pi pi-eye"
                severity="secondary"
                outlined
                :loading="
                  unsavedCmdPreviewLoading &&
                  cmdPreviewDialogVisible &&
                  cmdPreviewDialogMode === 'unsaved'
                "
                @click="openCmdPreviewDialog('unsaved')"
              />
              <Button
                label="Saved command"
                icon="pi pi-terminal"
                severity="secondary"
                outlined
                :loading="
                  cmdPreviewLoading && cmdPreviewDialogVisible && cmdPreviewDialogMode === 'saved'
                "
                @click="openCmdPreviewDialog('saved')"
              />
            </div>
          </div>
        </div>
      </div>

      <p v-if="persistenceNotice" class="persistence-notice" role="status">
        <strong>{{ persistenceNotice.summary }}.</strong>
        {{ persistenceNotice.detail }}
        <button v-if="persistenceNotice.retry" type="button" @click="retryPersistence">
          Try again
        </button>
        <button v-if="persistenceNotice.refresh" type="button" @click="refreshPersistedConfig">
          Refresh
        </button>
      </p>
      <!-- Actions -->
      <div class="config-actions">
        <Button
          label="Save changes"
          icon="pi pi-save"
          severity="success"
          :loading="saving"
          :disabled="configSaveHeld"
          @click="saveConfig"
        />
        <Button
          v-if="showApplyRuntime"
          :label="applyRuntimeLabel"
          icon="pi pi-bolt"
          severity="warning"
          :loading="applyingRuntime"
          :disabled="saving || applyingRuntime || configSaveHeld"
          v-tooltip.top="applyRuntimeHint"
          @click="requestApply"
        />
        <details class="config-secondary-actions" @keydown.esc.stop.prevent="closeSecondaryActions">
          <summary>More actions</summary>
          <div class="config-secondary-actions__menu">
            <Button
              v-if="canImportCommand"
              label="Import command"
              icon="pi pi-sparkles"
              severity="secondary"
              outlined
              @click="openParseCommandDialog()"
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
        </details>
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
        <p v-if="activeCmdPreview.revision" class="cmd-preview-dialog-hint">
          {{ activeCmdPreview.revision }}
        </p>
        <template v-if="activeCmdPreview.launcher">
          <div class="section-label cmd-preview-env-label">Launcher command (diagnostic)</div>
          <Textarea
            :model-value="activeCmdPreview.launcher"
            readonly
            rows="3"
            class="w-full textarea-cli cmd-preview-textarea"
            autoResize
          />
        </template>
        <template v-if="activeCmdPreview.env">
          <div class="section-label cmd-preview-env-label">
            Process environment ({{ activeCmdPreview.suffix }})
          </div>
          <Textarea
            :model-value="activeCmdPreview.env"
            readonly
            rows="4"
            class="w-full textarea-cli cmd-preview-textarea"
            autoResize
          />
        </template>
        <template v-if="activeCmdPreview.filters">
          <div class="section-label cmd-preview-env-label">
            Request defaults ({{ activeCmdPreview.suffix }})
          </div>
          <Textarea
            :model-value="activeCmdPreview.filters"
            readonly
            rows="6"
            class="w-full textarea-cli cmd-preview-textarea"
            autoResize
          />
        </template>
        <template v-if="activeCmdPreview.aliases">
          <div class="section-label cmd-preview-env-label">
            Model aliases ({{ activeCmdPreview.suffix }})
          </div>
          <Textarea
            :model-value="activeCmdPreview.aliases"
            readonly
            rows="3"
            class="w-full textarea-cli cmd-preview-textarea"
            autoResize
          />
        </template>
        <template v-if="activeCmdPreview.sidecar">
          <div class="section-label cmd-preview-env-label">
            Runtime config file for this model
            <small class="section-hint">
              Written when you Apply. Companion model paths and voice presets appear here.
              <template v-if="activeCmdPreview.sidecarPath">
                ({{ activeCmdPreview.sidecarPath }})
              </template>
            </small>
          </div>
          <Textarea
            :model-value="activeCmdPreview.sidecar"
            readonly
            rows="12"
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
          <Button
            label="Close"
            severity="secondary"
            outlined
            @click="cmdPreviewDialogVisible = false"
          />
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
          Routing aliases are omitted by default so each model keeps its own Studio proxy id.
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
            <Select
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
            <Select
              id="tpl-scope"
              v-model="templateSaveForm.engines_scope"
              :options="templateEnginesScopeOptions"
              option-label="label"
              option-value="value"
              class="w-full"
            />
          </div>
          <div class="config-templates-check">
            <ToggleSwitch v-model="templateSaveForm.include_routing" input-id="tpl-routing" />
            <label for="tpl-routing">Include routing aliases in template</label>
          </div>
          <PersistenceAlert
            :notice="templateSaveNotice"
            @retry="saveConfigTemplate"
            @refresh="refreshTemplateSave"
          />
          <Button
            label="Save template"
            icon="pi pi-save"
            :loading="templateSaveLoading"
            :disabled="!templateSaveForm.name.trim() || templateSaveHeld"
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
              <Select
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
              <Select
                id="tpl-apply-mode"
                v-model="templateApplyForm.apply_engines"
                :options="templateApplyModeOptions"
                option-label="label"
                option-value="value"
                class="w-full"
              />
            </div>
            <div class="config-templates-check">
              <ToggleSwitch
                v-model="templateApplyForm.include_routing"
                input-id="tpl-apply-routing"
              />
              <label for="tpl-apply-routing">Apply routing aliases from template</label>
            </div>
            <PersistenceAlert
              :notice="templateMutationNotice"
              @retry="retryTemplateMutation"
              @refresh="refreshTemplateMutation"
            />
            <div class="config-templates-apply-actions">
              <Button
                label="Apply to form"
                icon="pi pi-arrow-down"
                :loading="templateApplyLoading"
                :disabled="!templateApplyForm.template_id || templateMutationHeld"
                @click="applyConfigTemplate(false)"
              />
              <Button
                label="Apply & save"
                icon="pi pi-check"
                severity="success"
                :loading="templateApplyLoading"
                :disabled="!templateApplyForm.template_id || templateMutationHeld"
                @click="applyConfigTemplate(true)"
              />
            </div>
          </template>
        </div>

        <div v-if="configTemplates.length" class="config-templates-section">
          <div class="section-label">Saved templates</div>
          <ul class="config-templates-list">
            <li
              v-for="tpl in configTemplates"
              :key="tpl.id"
              class="config-templates-list-item"
              :class="{ 'config-templates-list-item--edit': templateEditId === tpl.id }"
            >
              <div v-if="templateEditId === tpl.id" class="config-templates-edit">
                <label class="sr-only" :for="`tpl-edit-name-${tpl.id}`">Template name</label>
                <InputText
                  :id="`tpl-edit-name-${tpl.id}`"
                  v-model="templateEditForm.name"
                  class="w-full"
                />
                <label class="sr-only" :for="`tpl-edit-desc-${tpl.id}`">Template description</label>
                <InputText
                  :id="`tpl-edit-desc-${tpl.id}`"
                  v-model="templateEditForm.description"
                  placeholder="Description"
                  class="w-full"
                />
                <div class="config-templates-apply-actions">
                  <Button
                    label="Save"
                    size="small"
                    icon="pi pi-check"
                    :loading="templateEditSaving"
                    :disabled="!templateEditForm.name.trim() || templateMutationHeld"
                    @click="saveTemplateEdit"
                  />
                  <Button
                    label="Cancel"
                    size="small"
                    severity="secondary"
                    outlined
                    @click="cancelTemplateEdit"
                  />
                </div>
              </div>
              <template v-else>
                <div class="config-templates-list-main">
                  <strong>{{ tpl.name }}</strong>
                  <span v-if="tpl.description" class="config-templates-list-desc">{{
                    tpl.description
                  }}</span>
                  <small class="config-templates-list-meta">
                    {{ (tpl.engine_ids || []).join(', ') || tpl.engine || '—' }}
                    <span v-if="tpl.include_routing"> · includes routing</span>
                  </small>
                </div>
                <div class="config-templates-actions">
                  <Button
                    icon="pi pi-pencil"
                    text
                    rounded
                    type="button"
                    aria-label="Rename template"
                    :disabled="templateMutationHeld"
                    @click="startTemplateEdit(tpl)"
                  />
                  <Button
                    icon="pi pi-trash"
                    severity="danger"
                    text
                    rounded
                    type="button"
                    aria-label="Delete template"
                    :loading="templateDeleteId === tpl.id"
                    :disabled="templateMutationHeld"
                    @click="deleteConfigTemplate(tpl.id)"
                  />
                </div>
              </template>
            </li>
          </ul>
        </div>

        <template #footer>
          <Button
            label="Close"
            severity="secondary"
            outlined
            @click="templatesDialogVisible = false"
          />
        </template>
      </Dialog>

      <UnsavedChangesDialog
        :visible="leavePromptVisible"
        :saving="saving"
        @stay="stayOnPage"
        @discard="discardAndLeave"
        @save="saveAndLeave"
      />

      <Dialog
        v-model:visible="applyImpactVisible"
        header="Apply saved settings"
        modal
        class="dialog-width-sm"
      >
        <p>{{ applyImpactMessage }}</p>
        <ul v-if="applyDifferences.length" class="apply-diff">
          <li v-for="row in applyDifferences" :key="row.field">
            <span class="apply-diff__field">{{ row.field }}</span>
            saved {{ row.saved }}, published {{ row.published }}, running {{ row.running }}
          </li>
        </ul>
        <p v-if="applyImpactModels" class="config-muted-hint">{{ applyImpactModels }}</p>
        <p v-if="applyWithheld" role="alert">
          {{ applyWithheld.message }}
          Apply again confirms that earlier attempt and continues this one.
        </p>
        <template #footer>
          <Button
            :label="applyingRuntime ? 'Stop apply' : 'Cancel'"
            severity="secondary"
            outlined
            @click="applyingRuntime ? requestApplyStop() : cancelApplyImpact()"
          />
          <Button
            :label="applyWithheld ? 'Apply again' : applyRuntimeLabel"
            icon="pi pi-bolt"
            severity="warning"
            :loading="applyingRuntime"
            @click="confirmApplyImpact"
          />
        </template>
      </Dialog>

      <ParseCommandDialog
        v-if="canImportCommand"
        v-model:visible="parseCommandDialogVisible"
        :catalog-params="parseCatalogParams"
        :current-values="parseCurrentValues"
        :current-env="parseCurrentEnv"
        :custom-args="typeof config.custom_args === 'string' ? config.custom_args : ''"
        :seed-text="parseCommandSeed"
        @apply="applyParsedCommand"
      />
    </template>
  </div>
</template>

<script setup>
import { formatLabel, taskLabel } from '@/utils/modelPresentation'
import UnsavedChangesDialog from '@/components/common/UnsavedChangesDialog.vue'
import { ref, computed, onMounted, onBeforeUnmount, watch, nextTick } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { clearDraft, readDraft, useDraftGuard, writeDraft } from '@/composables/useDraftGuard'
import { onRovingTabKeydown } from '@/composables/useRovingTabs'
import { watchDialogFocus } from '@/composables/useFocusReturn'
import { withheldConfirmation } from '@/composables/actionConfirmation'
import {
  classifyPersistenceError,
  noteDocumentSaveFailure,
  saveHeldForRefresh,
} from '@/composables/persistenceOutcome'
import PersistenceAlert from '@/components/common/PersistenceAlert.vue'
import { useToast } from 'primevue/usetoast'
import axios from 'axios'
import Button from 'primevue/button'
import Dialog from 'primevue/dialog'
import Tag from 'primevue/tag'
import InputText from 'primevue/inputtext'
import InputNumber from 'primevue/inputnumber'
import ToggleSwitch from 'primevue/toggleswitch'
import Select from 'primevue/select'
import InputTags from 'primevue/inputtags'
import Message from 'primevue/message'
import Textarea from 'primevue/textarea'
import Slider from 'primevue/slider'
import MultiSelect from 'primevue/multiselect'
import LoadingState from '@/components/common/LoadingState.vue'
import EmptyState from '@/components/common/EmptyState.vue'
import PageHeader from '@/components/common/PageHeader.vue'
import AudioModelConfig from '@/components/audio/AudioModelConfig.vue'
import RequestAliasesEditor from '@/components/RequestAliasesEditor.vue'
import ParseCommandDialog from '@/components/ParseCommandDialog.vue'
import {
  useAudioModelConfig,
  AUDIO_NESTED_SCOPE_KEYS,
  AUDIO_STUDIO_KNOWN_KEYS,
  coerceAudioParamValue,
  defaultValueForAudioParam,
  pruneStaleAudioRequestDefaults,
} from '@/composables/useAudioModelConfig'
import { audioTabFromConfig } from '@/composables/useAudioInferenceClient'
import { audioTaskDefaultsKey } from '@/composables/useAudioContract'
import { useModelStore } from '@/stores/models'
import { useEnginesStore } from '@/stores/engines'
import { STUDIO_RESERVED_KEYS, studioEnvSkipReason } from '@/utils/parseCliCommand'
import {
  AUDIO_ENGINE_IDS,
  ENGINE_DESCRIPTORS,
  ENGINE_IDS_BY_FALLBACK_FORMAT,
  FALLBACK_ENGINE_OPTIONS,
  SGLANG_RUNTIME_OPTIONS,
  VENV_ENGINE_IDS,
  isAudioEngineId,
} from '@/engines/registry'

const audioEngineId = AUDIO_ENGINE_IDS[0] || ''

const route = useRoute()
const router = useRouter()
const toast = useToast()
function engineMark(engineId) {
  return ENGINE_DESCRIPTORS[engineId]?.card?.mark || null
}
const modelStore = useModelStore()
const enginesStore = useEnginesStore()

// ── State ──────────────────────────────────────────────────
const loading = ref(true)
const loadError = ref('')
const draftOffer = ref(null)
const draftsEnabled = ref(false)
const applyImpactVisible = ref(false)
const applyWithheld = ref(null)
const BASIC_PARAM_KEYS = ['ctx_size', 'n_gpu_layers', 'parallel', 'threads']
const saving = ref(false)
const persistenceNotice = ref(null)
const persistenceAction = ref('save')
const templateSaveNotice = ref(null)
const templateSaveHeld = computed(() => saveHeldForRefresh(templateSaveNotice.value))
const templateMutationNotice = ref(null)
const templateMutationHeld = computed(() => saveHeldForRefresh(templateMutationNotice.value))
const templateMutationKind = ref('apply')
const templateApplyPersist = ref(false)
const requestAliasBase = computed(
  () =>
    String(config.value.model_alias || '')
      .trim()
      .toLowerCase()
      .replace(/[/\\\s]+/g, '-')
      .replace(/[^a-z0-9._-]/g, '-')
      .replace(/-{2,}/g, '-')
      .replace(/^[._-]+|[._-]+$/g, '') || stableModelId.value,
)
const requestAliasesValid = ref(true)
const configSaveHeld = computed(
  () => !requestAliasesValid.value || saveHeldForRefresh(persistenceNotice.value),
)

async function refreshTemplateSave() {
  const ok = await fetchConfigTemplates()
  templateSaveNotice.value = ok ? null : classifyPersistenceError(new Error('reload failed'))
}

function retryTemplateMutation() {
  if (templateMutationKind.value === 'delete') {
    void deleteConfigTemplate(templateDeleteId.value)
    return
  }
  if (templateMutationKind.value === 'edit') {
    void saveTemplateEdit()
    return
  }
  void applyConfigTemplate(templateApplyPersist.value)
}

async function refreshTemplateMutation() {
  const templatesOk = await fetchConfigTemplates()
  const configOk = await refreshPersistedConfig()
  templateMutationNotice.value =
    templatesOk && configOk ? null : classifyPersistenceError(new Error('reload failed'))
}
const refreshingModel = ref(false)
const companionsLoading = ref(false)
const companionBusyField = ref(null)
const companionFiles = ref({ mmproj_files: [], mtp_files: [], dflash_files: [] })
const selectedMmproj = ref('')
const selectedMtp = ref('')
const selectedDflash = ref('')
const model = ref(null)
const config = ref({})
const savedConfig = ref({}) // for reset
const paramRegistry = ref({
  sections: [],
  runtime_profiles: {},
  scan_error: null,
  scan_pending: false,
  task_profile: null,
  request_field_groups: [],
  // Empty until the backend payload lands; the composable then falls back to
  // the shared contract tables, never to a hardcoded endpoint.
  request_defaults_key: '',
  api_endpoint: '',
  request_policy: null,
  api_example_hint: '',
  instructions_policy: '',
  supports_voice_presets: false,
})
let audioRegistryRefreshTimer = null
let suppressAudioRegistryWatch = false
const paramSearchQuery = ref('')
const hideUnsupportedParams = ref(false)
/** Catalog keys currently shown in the params pane (order = add / derive order). */
const activeParamKeys = ref([])
const modelLimits = ref(null) // engine-agnostic: { max_context_length?, layer_count? } from config runtime_limits

const cmdPreviewText = ref('')
const cmdPreviewEnvText = ref('')
const cmdPreviewFiltersText = ref('')
const cmdPreviewAliasesText = ref('')
const cmdPreviewSidecarText = ref('')
const cmdPreviewSidecarPath = ref('')
const cmdPreviewLauncherText = ref('')
const cmdPreviewRevisionText = ref('')
const cmdPreviewError = ref(null)
const cmdPreviewLoading = ref(false)
const unsavedCmdPreviewText = ref('')
const unsavedCmdPreviewEnvText = ref('')
const unsavedCmdPreviewFiltersText = ref('')
const unsavedCmdPreviewAliasesText = ref('')
const unsavedCmdPreviewSidecarText = ref('')
const unsavedCmdPreviewSidecarPath = ref('')
const unsavedCmdPreviewLauncherText = ref('')
const unsavedCmdPreviewRevisionText = ref('')
const unsavedCmdPreviewError = ref(null)
const unsavedCmdPreviewLoading = ref(false)
/** Environment overrides for the native model process. */
const runtimeEnvRows = ref([{ key: '', value: '' }])
/** From GET /api/gpu-list (used for NVIDIA GPU binding UI). */
const gpuInfo = ref({
  vendor: null,
  gpus: [],
  device_count: 0,
  cpu_only_mode: true,
})
/** Indices (strings) for MultiSelect; mirrors `CUDA_VISIBLE_DEVICES` when NVIDIA GPUs exist. */
const cudaVisibleDeviceSelection = ref([])
let suppressCudaVisibleWatch = false
const applyingRuntime = ref(false)
const applyStopRequested = ref(false)
const cmdPreviewDialogVisible = ref(false)
const cmdPreviewDialogMode = ref('unsaved')
const templatesDialogVisible = ref(false)
const parseCommandDialogVisible = ref(false)
const parseCommandSeed = ref('')
const configTemplates = ref([])
const configTemplatesLoading = ref(false)
const templateSaveLoading = ref(false)
const templateApplyLoading = ref(false)
const templateDeleteId = ref(null)
const templateEditId = ref(null)
const templateEditSaving = ref(false)
const templateEditForm = ref({ name: '', description: '' })
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
let unsavedPreviewRequestId = 0
/** @type {AbortController | null} */
let unsavedPreviewAbort = null

// Engine identity (labels, icons, fallback compatibility) lives in
// @/engines/registry; these are local aliases for the picker below.
const fallbackEngineOptions = FALLBACK_ENGINE_OPTIONS
const curatedPackageKinds = ['prepared_bundle', 'builtin']
const inferredEnginesByFormat = ENGINE_IDS_BY_FALLBACK_FORMAT

function inferredEnginesForModel(options, fmt, packageKind) {
  if (curatedPackageKinds.includes(packageKind)) return ['audio_cpp']
  const fromDescriptors = options
    .filter((option) => {
      const formats = option.descriptor?.artifact_formats
      return (
        Array.isArray(formats) && formats.length === 1 && String(formats[0]).toLowerCase() === fmt
      )
    })
    .map((option) => option.value)
  return [...new Set([...(inferredEnginesByFormat[fmt] || []), ...fromDescriptors])]
}

const engineOptions = computed(() => {
  const descriptors = Array.isArray(enginesStore.engineDescriptors)
    ? enginesStore.engineDescriptors
    : []
  const options = descriptors.length
    ? descriptors.map((descriptor) => ({
        value: descriptor.id,
        label: descriptor.label,
        runnable: Boolean(descriptor.runnable),
        descriptor,
      }))
    : fallbackEngineOptions
  const verified = Array.isArray(model.value?.compatible_engines)
    ? model.value.compatible_engines.filter(Boolean)
    : []
  const fmt = String(model.value?.format || '').toLowerCase()
  const packageKind = model.value?.artifact?.package_kind || model.value?.package_kind
  const inferred = inferredEnginesForModel(options, fmt, packageKind)
  const curated = curatedPackageKinds.includes(packageKind)
  const allowed = new Set(curated && verified.length ? verified : [...verified, ...inferred])
  return options.map((option) => {
    const compatible = allowed.has(option.value)
    return {
      ...option,
      disabled: !compatible || option.descriptor?.enabled === false,
      disabledReason:
        option.descriptor?.enabled === false
          ? 'Disabled by the AUDIO_CPP_ENABLED feature gate'
          : compatible
            ? ''
            : `Not compatible with this ${packageKind || fmt || 'model'} artifact`,
    }
  })
})

const isAudioEngine = computed(() => isAudioEngineId(config.value.engine))
const engineConfigCopy = computed(
  () => ENGINE_DESCRIPTORS[config.value.engine]?.configCopy || {},
)
const stableIdLabel = computed(() => engineConfigCopy.value.stableIdLabel || 'Stable ID')
const stableIdAria = computed(
  () => engineConfigCopy.value.stableIdAria || 'Stable Studio proxy model ID',
)
const aliasLabel = computed(() => engineConfigCopy.value.aliasLabel || 'API alias')
const aliasTooltip = computed(
  () =>
    engineConfigCopy.value.aliasTooltip ||
    'Optional id apps send as model. Must be unique. Running state uses the stable ID, not this alias.',
)
function runtimeKeyFor(engineId) {
  return ENGINE_DESCRIPTORS[engineId]?.runtimeKey || ''
}
const engineRuntimeKey = computed(() => runtimeKeyFor(config.value.engine))
const isSglangEngine = computed(() => Boolean(engineRuntimeKey.value))
const sglangRuntimeOptions = SGLANG_RUNTIME_OPTIONS
const sglangRuntime = computed(() =>
  normalizeSglangRuntime(config.value[engineRuntimeKey.value]),
)
function normalizeSglangRuntime(runtime) {
  return runtime === 'diffusion' ? 'diffusion' : 'llm'
}
const pageTab = ref('runtime')
const pageTabs = computed(() =>
  isAudioEngine.value
    ? [
        { id: 'server', label: 'Runtime', icon: 'pi pi-server' },
        { id: 'assets', label: 'Assets', icon: 'pi pi-folder-open' },
        { id: 'api', label: 'Defaults', icon: 'pi pi-sliders-h' },
        { id: 'launch', label: 'Launch', icon: 'pi pi-play' },
      ]
    : [
        { id: 'runtime', label: 'Runtime', icon: 'pi pi-server' },
        { id: 'launch', label: 'Launch', icon: 'pi pi-play' },
      ],
)
watch(isAudioEngine, (audio) => {
  const ids = new Set(pageTabs.value.map((tab) => tab.id))
  if (!ids.has(pageTab.value)) pageTab.value = audio ? 'server' : 'runtime'
})
const canImportCommand = computed(() => !isAudioEngine.value && catalogSections.value.length > 0)

const parseCatalogParams = computed(() => {
  const out = []
  for (const section of catalogSections.value) {
    for (const param of section.params || []) {
      if (param?.key) out.push({ ...param, sectionId: section.id })
    }
  }
  return out
})

const parseCurrentValues = computed(() => {
  const out = {}
  for (const key of activeParamKeys.value) {
    if (Object.prototype.hasOwnProperty.call(config.value, key)) {
      out[key] = config.value[key]
    }
  }
  return out
})

const parseCurrentEnv = computed(() => {
  const env = config.value.runtime_env
  return env && typeof env === 'object' && !Array.isArray(env) ? { ...env } : {}
})
const modelFormat = computed(() =>
  String(model.value?.format || model.value?.model_format || 'gguf').toLowerCase(),
)
const canCheckForUpdates = computed(() => {
  if (!model.value?.huggingface_id) return false
  if (isAudioEngine.value) return false
  return modelFormat.value === 'gguf' || modelFormat.value === 'safetensors'
})
const showCompanionsCard = computed(
  () => modelFormat.value === 'gguf' && !!model.value?.huggingface_id,
)
const companionBusy = computed(() => !!companionBusyField.value)
const mmprojOptions = computed(() =>
  companionDropdownOptions(companionFiles.value.mmproj_files, 'mmproj'),
)
const mtpOptions = computed(() => companionDropdownOptions(companionFiles.value.mtp_files, 'mtp'))
const dflashOptions = computed(() =>
  companionDropdownOptions(companionFiles.value.dflash_files, 'dflash'),
)
const mmprojSelectionChanged = computed(
  () => (selectedMmproj.value || '') !== (model.value?.mmproj_filename || ''),
)
const mtpSelectionChanged = computed(
  () => (selectedMtp.value || '') !== (model.value?.mtp_filename || ''),
)
const dflashSelectionChanged = computed(
  () => (selectedDflash.value || '') !== (model.value?.dflash_filename || ''),
)

function openAudioWorkspace() {
  const modelId = model.value?.id || route.params.id
  if (!modelId) return
  const tab = audioTabFromConfig(config.value || {})
  router.push({ name: 'audio', query: { model: modelId, tab } })
}

const showNvidiaGpuBind = computed(() => {
  const g = gpuInfo.value || {}
  return g.vendor === 'nvidia' && Array.isArray(g.gpus) && g.gpus.length > 0 && !g.cpu_only_mode
})

const nvidiaGpuSelectOptions = computed(() => {
  if (!showNvidiaGpuBind.value) return []
  return gpuInfo.value.gpus.map((gpu) => {
    const idx = gpu.index != null ? gpu.index : 0
    const name = typeof gpu.name === 'string' && gpu.name ? gpu.name : `GPU ${idx}`
    return {
      value: gpu.uuid ? String(gpu.uuid) : String(idx),
      label: name,
      name,
      index: idx,
    }
  })
})

const nvidiaGpuCards = computed(() => {
  const selected = cudaVisibleDeviceSelection.value || []
  return nvidiaGpuSelectOptions.value.map((gpu) => {
    const order = selected.indexOf(gpu.value)
    return { ...gpu, order: order >= 0 ? order : null }
  })
})

/** Hide CUDA_VISIBLE_DEVICES from the generic env table when the NVIDIA binding control is shown. */
const runtimeEnvRowsDisplayed = computed(() =>
  runtimeEnvRows.value
    .map((row, originalIndex) => ({ row, originalIndex }))
    .filter(({ row }) => {
      if (!showNvidiaGpuBind.value) return true
      return (
        String(row.key || '')
          .trim()
          .toUpperCase() !== 'CUDA_VISIBLE_DEVICES'
      )
    }),
)

const selectedParamRegistry = computed(() => {
  if (!isSglangEngine.value) return paramRegistry.value
  const profile =
    ENGINE_DESCRIPTORS[config.value.engine]?.runtimeProfiles?.[sglangRuntime.value] || 'default'
  if (profile === 'default') return paramRegistry.value
  return paramRegistry.value.runtime_profiles?.[profile] || { sections: [], scan_pending: true }
})
const catalogSections = computed(() =>
  Array.isArray(selectedParamRegistry.value.sections) ? selectedParamRegistry.value.sections : [],
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
    if (BASIC_PARAM_KEYS.includes(key)) continue
    const p = m.get(key)
    if (p) out.push(p)
  }
  return out
})

const stableModelId = computed(() => {
  const m = model.value
  if (!m) return ''
  if (m.proxy_model_id) return m.proxy_model_id
  if (m.proxy_name) return m.proxy_name
  return ''
})

const audioModelConfig = useAudioModelConfig(config, paramRegistry, enginesStore, stableModelId)

const {
  audioEditableParams,
  audioParamValue,
  ensureTtsConfigShape,
  seedSessionVoiceFromPackagedIds,
} = audioModelConfig

const currentEngineSection = computed(
  () => (config.value.engines && config.value.engines[config.value.engine]) || {},
)

const AUDIO_STUDIO_KNOWN_KEYS_REF = AUDIO_STUDIO_KNOWN_KEYS

const unrecognizedSavedKeys = computed(() => {
  const known = new Set([
    'custom_args',
    'model_alias',
    'set_params_by_id',
    'swap_aliases',
    'runtime_env',
    'runtime_env_unset',
    'gpu_mode',
    'gpu_devices',
    'load_options',
    'session_options',
    ...(isAudioEngine.value ? AUDIO_STUDIO_KNOWN_KEYS_REF : []),
    ...(engineRuntimeKey.value ? [engineRuntimeKey.value] : []),
  ])
  for (const key of catalogParamByKey.value.keys()) known.add(key)
  if (isSglangEngine.value) {
    for (const registry of [
      paramRegistry.value,
      ...Object.values(paramRegistry.value.runtime_profiles || {}),
    ]) {
      for (const section of registry?.sections || []) {
        for (const param of section.params || []) known.add(param.key)
      }
    }
  }
  return Object.keys(currentEngineSection.value || {}).filter((key) => !known.has(key))
})

const hasUnsavedChanges = computed(() => {
  try {
    return (
      !requestAliasesValid.value ||
      JSON.stringify(config.value) !== JSON.stringify(savedConfig.value)
    )
  } catch {
    return false
  }
})

const { leavePromptVisible, finishLeave } = useDraftGuard(
  () => hasUnsavedChanges.value && !loading.value,
)
watchDialogFocus(applyImpactVisible)
watchDialogFocus(leavePromptVisible)
watchDialogFocus(templatesDialogVisible)

/**
 * Saved model config differs from its published native runtime manifest.
 * Shown only when there are no unsaved edits — save first, then Apply.
 */
const showApplyRuntime = computed(
  () =>
    !hasUnsavedChanges.value &&
    ['restart_now', 'publish_next_start'].includes(modelLaunchPlan.value?.action),
)

const gpuModeOptions = [
  { label: 'All GPUs', value: 'inherit' },
  { label: 'Choose GPUs', value: 'selected' },
  { label: 'CPU only', value: 'cpu' },
]

function onGpuModeKeydown(event) {
  const keys = ['ArrowRight', 'ArrowDown', 'ArrowLeft', 'ArrowUp', 'Home', 'End']
  if (!keys.includes(event.key)) return
  event.preventDefault()
  const index = Math.max(
    0,
    gpuModeOptions.findIndex((mode) => mode.value === (config.value.gpu_mode || 'inherit')),
  )
  const next =
    event.key === 'Home'
      ? 0
      : event.key === 'End'
        ? gpuModeOptions.length - 1
        : (index +
            (['ArrowRight', 'ArrowDown'].includes(event.key) ? 1 : -1) +
            gpuModeOptions.length) %
          gpuModeOptions.length
  setGpuMode(gpuModeOptions[next].value)
  event.currentTarget.querySelectorAll('[role="radio"]')[next]?.focus()
}

const envModeOptions = [
  { label: 'Set value', value: 'set' },
  { label: 'Unset', value: 'unset' },
]

const modelLaunchPlan = computed(() => {
  const rows = enginesStore.runtimePlan?.models
  if (!Array.isArray(rows)) return null
  const catalogId = String(route.params.id || '')
  return rows.find((row) => row.catalog_id === catalogId || row.model_id === catalogId) || null
})

const selectiveModelApply = computed(() => {
  const pending = enginesStore.runtimePlan
  const entry = modelLaunchPlan.value
  return Boolean(
    pending?.launch_manifests &&
    entry &&
    !pending.requires_proxy_reload &&
    !entry.requires_proxy_reload &&
    (entry.action === 'restart_now' || entry.action === 'publish_next_start'),
  )
})

const applyRuntimeLabel = computed(() => {
  return modelLaunchPlan.value?.running ? 'Restart this model' : 'Use on next start'
})

const pendingApply = computed(() => Boolean(showApplyRuntime.value))

const modelRuntimeQuality = computed(() => String(model.value?.runtime_quality || '').toLowerCase())

const modelIsRunning = computed(() => {
  if (modelRuntimeQuality.value === 'unreachable') return false
  const plan = modelLaunchPlan.value
  if (plan && typeof plan.running === 'boolean') return plan.running
  return Boolean(model.value?.is_active)
})

const publishedStepLabel = computed(() => {
  if (
    enginesStore.runtimePlan?.launch_manifests === false ||
    modelRuntimeQuality.value === 'unreachable'
  )
    return 'Status unknown'
  if (modelRuntimeQuality.value === 'stale') {
    return model.value?.is_active ? 'Running · stale' : 'Status stale'
  }
  return modelIsRunning.value ? 'Settings active' : 'Saved · ready for next start'
})

const runtimeStateDetail = computed(() => {
  if (hasUnsavedChanges.value)
    return 'These edits are only in this form. Save them before they can be applied.'
  if (pendingApply.value) {
    const action = modelLaunchPlan.value?.action
    if (action === 'publish_next_start') {
      return 'Saved changes for this model are not published yet. They apply the next time it starts.'
    }
    if (action === 'restart_now') {
      return 'Saved changes for this model are not in use yet. Apply them to restart this model.'
    }
    return 'Pending changes for this model are not in use yet. Apply them with the action below. Saving does not restart the model.'
  }
  if (enginesStore.runtimePlan?.launch_manifests === false)
    return 'The native runtime plan is unavailable. Refresh to check whether saved settings are in use.'
  if (modelRuntimeQuality.value === 'unreachable') {
    return 'Saved settings are on this page, but the proxy could not be reached. This is not a verified stopped or running state.'
  }
  if (modelRuntimeQuality.value === 'stale') {
    return model.value?.is_active
      ? 'The proxy is unreachable. The last observation still had this model running.'
      : 'The proxy is unreachable. The last observation did not show this model running.'
  }
  if (modelIsRunning.value)
    return 'These saved settings are published and this model is running them.'
  return 'These settings are published. The model is stopped, so they are used the next time it starts.'
})

watch(
  () => enginesStore.runtimePlanState?.stale,
  (stale, previous) => {
    if (stale && stale !== previous) void enginesStore.fetchRuntimePlan?.()
  },
)

const applyImpactMessage = computed(() => {
  if (modelLaunchPlan.value?.running) {
    return 'Restart this model. Requests already running on it can be interrupted. Other models stay loaded.'
  }
  return 'Use these saved settings the next time this model starts. It stays stopped, and other models stay loaded.'
})

const applyDifferences = computed(() => {
  const rows = modelLaunchPlan.value?.differences
  return Array.isArray(rows) ? rows : []
})

const applyImpactModels = computed(() => {
  const name = model.value?.display_name || model.value?.base_model_name || 'This model'
  return `Affected: ${name}.`
})

const basicParams = computed(() =>
  BASIC_PARAM_KEYS.map((key) => catalogParamByKey.value.get(key)).filter(Boolean),
)

function engineIsInstalled(option) {
  const count = option?.descriptor?.installed_versions
  if (count == null) return true
  return Number(count) > 0
}

const visibleEngineOptions = computed(() =>
  engineOptions.value.filter((option) => !option.disabled && engineIsInstalled(option)),
)
const unavailableSavedEngine = computed(() => {
  const selected = engineOptions.value.find((option) => option.value === config.value.engine)
  if (!selected) return null
  if (visibleEngineOptions.value.some((option) => option.value === selected.value)) return null
  return selected
})

function closeSecondaryActions(event) {
  event.currentTarget.open = false
  event.currentTarget.querySelector('summary')?.focus()
}
function useParameterDefault(param) {
  delete config.value[param.key]
}
function parameterHelp(param) {
  const value =
    param.default == null || param.default === ''
      ? 'chosen by the engine'
      : JSON.stringify(param.default)
  return `Engine default: ${value}`
}

function isExplicitOverride(param) {
  const value = config.value?.[param.key]
  if (value == null || value === '') return false
  if (Array.isArray(value) && value.length === 0) return false
  return true
}

function engineTabIndex(eng) {
  if (eng.disabled) return -1
  return config.value.engine === eng.value ? 0 : -1
}

function onEngineKeydown(event) {
  const options = visibleEngineOptions.value.filter((option) => !option.disabled)
  if (!options.length) return
  const index = Math.max(
    0,
    options.findIndex((option) => option.value === config.value.engine),
  )
  let next
  if (event.key === 'ArrowRight' || event.key === 'ArrowDown') next = (index + 1) % options.length
  else if (event.key === 'ArrowLeft' || event.key === 'ArrowUp')
    next = (index - 1 + options.length) % options.length
  else if (event.key === 'Home') next = 0
  else if (event.key === 'End') next = options.length - 1
  else return
  event.preventDefault()
  changeEngine(options[next].value)
  const buttons = event.currentTarget?.querySelectorAll?.('[role="radio"]:not([disabled])')
  buttons?.[next]?.focus()
}

function requestLeave(path) {
  router.push(path)
}

const applyRuntimeHint = computed(() => {
  if (modelLaunchPlan.value?.running) {
    return 'Restart only this model. Its current requests can be interrupted. Other models stay loaded.'
  }
  return 'Publish this model’s saved settings for its next start. It stays stopped, and other models stay loaded.'
})

/** Multi-word AND search on label, key, flags, description. */
function paramMatchesSearch(param, queryRaw) {
  if (hideUnsupportedParams.value && param.supported === false) return false
  const raw = queryRaw.trim().toLowerCase()
  if (!raw) return false
  const hay = [param.label || '', param.key || '', param.description || '', ...(param.flags || [])]
    .join(' ')
    .toLowerCase()
  const tokens = raw.split(/\s+/).filter(Boolean)
  return tokens.every((t) => hay.includes(t))
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
  if (value === null) return true
  if (param.value_kind === 'flag') {
    return value === true || (param.negative_flag && value === false)
  }
  if (['repeatable', 'space_list'].includes(param.value_kind)) {
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
    return value
      .split(delimitedEnumSeparator(param))
      .map((s) => s.trim())
      .filter(Boolean)
  }
  return [value]
}

function defaultValueForParam(param) {
  if (['repeatable', 'space_list'].includes(param.value_kind))
    return Array.isArray(param.default) ? [...param.default] : []
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
  activeParamKeys.value = activeParamKeys.value.filter((k) => k !== key)
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

function previewEnvLines(data) {
  if (Array.isArray(data?.engine_env) && data.engine_env.length) {
    const unset =
      Array.isArray(data.engine_env_unset) && data.engine_env_unset.length
        ? `\n# unset\n${data.engine_env_unset.join('\n')}`
        : ''
    return formatEnvPreviewLines(data.engine_env) + unset
  }
  return formatEnvPreviewLines(data?.env)
}

function formatRevisionPreview(data) {
  if (!data?.launch_revision) return ''
  const published = data.published_revision || 'not published'
  return `Saved revision ${data.launch_revision}\nPublished revision ${published}`
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

function formatSidecarPreview(sidecar) {
  if (!sidecar || typeof sidecar !== 'object') return ''
  try {
    return JSON.stringify(sidecar, null, 2)
  } catch {
    return ''
  }
}

const cmdPreviewDialogHeader = computed(() =>
  cmdPreviewDialogMode.value === 'saved'
    ? 'Saved Studio proxy command'
    : 'Unsaved Studio proxy preview',
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
      filters: cmdPreviewFiltersText.value,
      aliases: cmdPreviewAliasesText.value,
      sidecar: cmdPreviewSidecarText.value,
      sidecarPath: cmdPreviewSidecarPath.value,
      launcher: cmdPreviewLauncherText.value,
      revision: cmdPreviewRevisionText.value,
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
    filters: unsavedCmdPreviewFiltersText.value,
    aliases: unsavedCmdPreviewAliasesText.value,
    sidecar: unsavedCmdPreviewSidecarText.value,
    sidecarPath: unsavedCmdPreviewSidecarPath.value,
    launcher: unsavedCmdPreviewLauncherText.value,
    revision: unsavedCmdPreviewRevisionText.value,
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
  const param = catalogParamByKey.value.get(key)
  const trimmed = (text ?? '').trim()
  if (!trimmed) {
    if (param && param.required !== true) config.value[key] = null
    else delete config.value[key]
    return
  }
  try {
    config.value[key] = JSON.parse(trimmed)
  } catch {
    config.value[key] = text
  }
}

function _runtimeEnvShallowEqual(a, b) {
  const x = a && typeof a === 'object' && !Array.isArray(a) ? a : {}
  const y = b && typeof b === 'object' && !Array.isArray(b) ? b : {}
  const xk = Object.keys(x)
  const yk = Object.keys(y)
  if (xk.length !== yk.length) return false
  return xk.every(
    (k) => Object.prototype.hasOwnProperty.call(y, k) && String(x[k]) === String(y[k]),
  )
}

function _runtimeEnvUnsetEqual(a, b) {
  const x = Array.isArray(a) ? a.map((name) => String(name)) : []
  const y = Array.isArray(b) ? b.map((name) => String(name)) : []
  if (x.length !== y.length) return false
  return x.every((name, index) => name === y[index])
}

function syncRuntimeEnvFromRows() {
  const out = {}
  const unset = []
  for (const row of runtimeEnvRows.value) {
    const k = (row.key || '').trim()
    if (!k) continue
    if (row.mode === 'unset') {
      unset.push(k)
      continue
    }
    out[k] = row.value != null ? String(row.value) : ''
  }
  if (
    _runtimeEnvShallowEqual(config.value.runtime_env, out) &&
    _runtimeEnvUnsetEqual(config.value.runtime_env_unset, unset)
  ) {
    return
  }
  config.value.runtime_env = out
  config.value.runtime_env_unset = unset
}

function initRuntimeEnvRowsFromConfig() {
  const rows = []
  const o = config.value.runtime_env
  if (o && typeof o === 'object' && !Array.isArray(o)) {
    for (const [key, value] of Object.entries(o)) {
      rows.push({
        key,
        value: value != null ? String(value) : '',
        mode: 'set',
      })
    }
  }
  const unset = Array.isArray(config.value.runtime_env_unset) ? config.value.runtime_env_unset : []
  for (const name of unset) {
    rows.push({ key: String(name), value: '', mode: 'unset' })
  }
  runtimeEnvRows.value = rows.length ? rows : [{ key: '', value: '', mode: 'set' }]
}

function addRuntimeEnvRow() {
  runtimeEnvRows.value = [...runtimeEnvRows.value, { key: '', value: '' }]
  syncRuntimeEnvFromRows()
}

function removeRuntimeEnvRow(idx) {
  const next = runtimeEnvRows.value.filter((_, i) => i !== idx)
  runtimeEnvRows.value = next.length ? next : [{ key: '', value: '' }]
  syncRuntimeEnvFromRows()
  syncCudaSelectionFromEnv()
}

function getRawCudaVisibleFromRuntimeEnv() {
  const se = config.value.runtime_env
  if (!se || typeof se !== 'object' || Array.isArray(se)) return ''
  const entry = Object.keys(se).find((k) => k.toUpperCase() === 'CUDA_VISIBLE_DEVICES')
  return entry ? String(se[entry] ?? '') : ''
}

function parseCudaDeviceList(raw) {
  if (raw == null || raw === '') return []
  return String(raw)
    .split(',')
    .map((t) => t.trim())
    .filter(Boolean)
}

function toggleGpuCard(value) {
  const current = Array.isArray(cudaVisibleDeviceSelection.value)
    ? [...cudaVisibleDeviceSelection.value]
    : []
  const index = current.indexOf(value)
  if (index >= 0) current.splice(index, 1)
  else current.push(value)
  cudaVisibleDeviceSelection.value = current
}

function setGpuMode(mode) {
  config.value.gpu_mode = mode
  if (mode === 'cpu') {
    config.value.gpu_devices = []
    removeRuntimeEnvRowByCanonicalKey('CUDA_VISIBLE_DEVICES')
    const unset = new Set(config.value.runtime_env_unset || [])
    unset.add('CUDA_VISIBLE_DEVICES')
    config.value.runtime_env_unset = [...unset]
    initRuntimeEnvRowsFromConfig()
    return
  }
  config.value.runtime_env_unset = (config.value.runtime_env_unset || []).filter(
    (name) => String(name).toUpperCase() !== 'CUDA_VISIBLE_DEVICES',
  )
  if (mode === 'inherit') {
    config.value.gpu_devices = []
    removeRuntimeEnvRowByCanonicalKey('CUDA_VISIBLE_DEVICES')
    return
  }
  applyNvidiaCudaSelection(cudaVisibleDeviceSelection.value)
}

function upsertRuntimeEnvRowCanonical(canonicalKey, value) {
  const upper = canonicalKey.toUpperCase()
  const rows = [...runtimeEnvRows.value]
  let idx = rows.findIndex(
    (r) =>
      String(r.key || '')
        .trim()
        .toUpperCase() === upper,
  )
  if (idx < 0) {
    const onlyBlank =
      rows.length === 1 && !String(rows[0].key || '').trim() && !String(rows[0].value || '').trim()
    if (onlyBlank) {
      runtimeEnvRows.value = [{ key: canonicalKey, value }]
    } else {
      runtimeEnvRows.value = [...rows, { key: canonicalKey, value }]
    }
  } else {
    rows[idx] = { ...rows[idx], key: canonicalKey, value }
    runtimeEnvRows.value = rows
  }
  syncRuntimeEnvFromRows()
}

function removeRuntimeEnvRowByCanonicalKey(canonicalKey) {
  const upper = canonicalKey.toUpperCase()
  const next = runtimeEnvRows.value.filter(
    (r) =>
      String(r.key || '')
        .trim()
        .toUpperCase() !== upper,
  )
  runtimeEnvRows.value = next.length ? next : [{ key: '', value: '' }]
  syncRuntimeEnvFromRows()
}

function applyNvidiaCudaSelection(selected) {
  if (!showNvidiaGpuBind.value) return
  if ((config.value.gpu_mode || 'inherit') !== 'selected') return
  config.value.runtime_env_unset = (config.value.runtime_env_unset || []).filter(
    (name) => String(name).toUpperCase() !== 'CUDA_VISIBLE_DEVICES',
  )
  const allVals = nvidiaGpuSelectOptions.value.map((o) => o.value)
  const sel = []
  for (const value of selected || []) {
    if (allVals.includes(value) && !sel.includes(value)) sel.push(value)
  }
  config.value.gpu_devices = [...sel]
  if (sel.length === 0) {
    removeRuntimeEnvRowByCanonicalKey('CUDA_VISIBLE_DEVICES')
  } else {
    upsertRuntimeEnvRowCanonical('CUDA_VISIBLE_DEVICES', sel.join(','))
  }
}

function syncCudaSelectionFromEnv() {
  if (!showNvidiaGpuBind.value) {
    cudaVisibleDeviceSelection.value = []
    return
  }
  const raw = getRawCudaVisibleFromRuntimeEnv()
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
  if (d && typeof d === 'object' && typeof d.message === 'string') return d.message
  if (Array.isArray(d)) {
    return d
      .map((x) =>
        typeof x === 'object' && x?.msg
          ? `${Array.isArray(x.loc) ? x.loc.join('.') : ''}: ${x.msg}`.replace(/^\.\s*/, '')
          : String(x),
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
      if (String(q.id) === sid)
        return {
          ...q,
          base_model_name: group.base_model_name,
          huggingface_id: group.huggingface_id,
        }
    }
  }
  // Fallback: search allQuantizations
  return modelStore.allQuantizations.find((m) => String(m.id) === sid) ?? null
}

async function fetchGpuListForBind() {
  try {
    const data = await enginesStore.fetchGpuList()
    gpuInfo.value =
      data && typeof data === 'object'
        ? data
        : { vendor: null, gpus: [], device_count: 0, cpu_only_mode: true }
  } catch (e) {
    console.error('Failed to fetch GPU list:', e)
    gpuInfo.value = { vendor: null, gpus: [], device_count: 0, cpu_only_mode: true }
  }
}

async function fetchParamRegistry(engine, { draftFamily, draftTask, rescan = false } = {}) {
  try {
    const family = draftFamily ?? (isAudioEngineId(engine) ? config.value?.family : undefined)
    const task = draftTask ?? (isAudioEngineId(engine) ? config.value?.task : undefined)
    const { data } = await axios.get('/api/models/param-registry', {
      params: {
        engine,
        ...(model.value?.id ? { model_id: model.value.id } : {}),
        ...(isAudioEngineId(engine) && family ? { family } : {}),
        ...(isAudioEngineId(engine) && task ? { task } : {}),
        ...(rescan ? { rescan: true } : {}),
      },
    })
    paramRegistry.value = {
      sections: data.sections || [],
      runtime_profiles: data.runtime_profiles || {},
      scan_error: data.scan_error ?? null,
      scan_pending: Boolean(data.scan_pending),
      profile_fingerprint: data.profile_fingerprint ?? null,
      inspection: data.inspection ?? null,
      compatibility_warnings: data.compatibility_warnings || [],
      task_profile: data.task_profile ?? null,
      request_field_groups: data.request_field_groups || [],
      request_defaults_key: data.request_defaults_key || '',
      api_endpoint: data.api_endpoint || '',
      request_policy: data.request_policy || null,
      api_example_hint: data.api_example_hint || '',
      instructions_policy: data.instructions_policy || '',
      instructions_policy_source: data.instructions_policy_source || '',
      instructions_vocabulary: data.instructions_vocabulary || null,
      supports_voice_presets: Boolean(data.supports_voice_presets),
      policy_family: data.policy_family || family || '',
      policy_task: data.policy_task || task || '',
      contract_fingerprint: data.contract_fingerprint || '',
      contract_changed: Boolean(data.contract_changed),
      contract_review_required: Boolean(data.contract_review_required),
      discovery_source: data.discovery_source || '',
      catalog_source: data.catalog_source || '',
      contract_drift_warnings: data.contract_drift_warnings || [],
      last_reviewed_fingerprint: data.last_reviewed_fingerprint || '',
      sidecar_session_fields: data.sidecar_session_fields || [],
      family_dependencies: data.family_dependencies || {},
      packaged_voices: data.packaged_voices || [],
    }
    if (isAudioEngineId(engine)) {
      pruneStaleAudioRequestDefaults(
        config.value,
        paramRegistry.value.request_defaults_key ||
          audioTaskDefaultsKey(paramRegistry.value.policy_task || config.value?.task),
      )
      seedSessionVoiceFromPackagedIds()
    }
  } catch (e) {
    console.error('Failed to fetch param registry:', e)
    paramRegistry.value = {
      sections: [],
      runtime_profiles: {},
      scan_error: null,
      scan_pending: false,
      instructions_policy: '',
      supports_voice_presets: false,
      packaged_voices: [],
    }
  }
}

function scheduleAudioRegistryRefresh() {
  if (audioRegistryRefreshTimer) clearTimeout(audioRegistryRefreshTimer)
  audioRegistryRefreshTimer = window.setTimeout(() => {
    audioRegistryRefreshTimer = null
    if (!isAudioEngineId(config.value?.engine) || loading.value) return
    void fetchParamRegistry(audioEngineId)
  }, 250)
}

function buildWorkingConfigFromApi(cfg) {
  const engines =
    cfg.engines && typeof cfg.engines === 'object' ? JSON.parse(JSON.stringify(cfg.engines)) : {}
  const engine = cfg.engine ?? 'llama_cpp'
  const sec = engines[engine] || {}
  return {
    engine,
    engines,
    ...sec,
    ...(runtimeKeyFor(engine)
      ? { [runtimeKeyFor(engine)]: normalizeSglangRuntime(sec[runtimeKeyFor(engine)]) }
      : {}),
  }
}

function buildEngineStashFromForm(sourceConfig = config.value) {
  const preserveUnknownAudioKeys = isAudioEngineId(sourceConfig.engine)
  const preserveSglangKeys = Boolean(runtimeKeyFor(sourceConfig.engine))
  const previous = sourceConfig.engines?.[sourceConfig.engine]
  const stash =
    (preserveUnknownAudioKeys || preserveSglangKeys) && previous && typeof previous === 'object'
      ? JSON.parse(JSON.stringify(previous))
      : {}
  if (preserveSglangKeys) {
    // Only replace the displayed runtime's editable parameters. Keep settings
    // belonging to the other runtime and keys missing from an older CLI scan.
    for (const param of catalogParamList.value) delete stash[param.key]
    stash[runtimeKeyFor(sourceConfig.engine)] = normalizeSglangRuntime(
    sourceConfig[runtimeKeyFor(sourceConfig.engine)],
  )
  }
  if (Array.isArray(sourceConfig.set_params_by_id)) {
    stash.set_params_by_id = JSON.parse(JSON.stringify(sourceConfig.set_params_by_id))
  } else delete stash.set_params_by_id
  if (Array.isArray(sourceConfig.swap_aliases)) stash.swap_aliases = [...sourceConfig.swap_aliases]
  delete stash.model_alias
  delete stash.custom_args
  delete stash.request_options
  if (typeof sourceConfig.model_alias === 'string' && sourceConfig.model_alias.trim()) {
    stash.model_alias = sourceConfig.model_alias.trim()
  }
  if (typeof sourceConfig.custom_args === 'string' && sourceConfig.custom_args.trim()) {
    stash.custom_args = sourceConfig.custom_args
  }
  const se = sourceConfig.runtime_env
  if (se && typeof se === 'object' && !Array.isArray(se)) {
    const cleaned = {}
    for (const [k, v] of Object.entries(se)) {
      const name = String(k).trim()
      if (!name) continue
      if (v == null) continue
      if (typeof v === 'number' && Number.isNaN(v)) continue
      cleaned[name] = typeof v === 'string' ? v : String(v)
    }
    stash.runtime_env = cleaned
  } else {
    stash.runtime_env = {}
  }
  if (Array.isArray(sourceConfig.runtime_env_unset) && sourceConfig.runtime_env_unset.length) {
    stash.runtime_env_unset = sourceConfig.runtime_env_unset
      .map((name) => String(name))
      .filter(Boolean)
  }
  if (sourceConfig.gpu_mode) stash.gpu_mode = sourceConfig.gpu_mode
  if (Array.isArray(sourceConfig.gpu_devices)) {
    stash.gpu_devices = sourceConfig.gpu_devices.map((item) => String(item))
  }
  if (preserveUnknownAudioKeys) {
    for (const param of audioEditableParams.value) {
      const nestedKey = AUDIO_NESTED_SCOPE_KEYS[param.scope]
      const value = audioParamValue(param, sourceConfig)
      const empty =
        value == null ||
        value === '' ||
        (typeof value === 'number' && Number.isNaN(value)) ||
        (Array.isArray(value) && value.length === 0)
      if (nestedKey) {
        if (!stash[nestedKey] || typeof stash[nestedKey] !== 'object') {
          stash[nestedKey] = {}
        }
        if (empty) {
          if (param.required !== true) stash[nestedKey][param.key] = null
          else delete stash[nestedKey][param.key]
        } else {
          stash[nestedKey][param.key] = Array.isArray(value)
            ? [...value]
            : coerceAudioParamValue(param, value)
        }
        continue
      }
      if (empty) {
        if (param.required !== true) stash[param.key] = null
        else delete stash[param.key]
      } else {
        stash[param.key] = Array.isArray(value) ? [...value] : coerceAudioParamValue(param, value)
      }
    }
    for (const nestedKey of ['load_options', 'session_options']) {
      if (
        stash[nestedKey] &&
        typeof stash[nestedKey] === 'object' &&
        !Object.keys(stash[nestedKey]).length
      ) {
        delete stash[nestedKey]
      }
    }
    const presets = sourceConfig.voice_presets
    if (
      presets &&
      typeof presets === 'object' &&
      !Array.isArray(presets) &&
      Object.keys(presets).length
    ) {
      stash.voice_presets = JSON.parse(JSON.stringify(presets))
    } else {
      delete stash.voice_presets
    }
    const defaultPreset = sourceConfig.default_voice_preset
    if (defaultPreset != null && defaultPreset !== '') {
      stash.default_voice_preset = JSON.parse(JSON.stringify(defaultPreset))
    } else {
      delete stash.default_voice_preset
    }
    const speechDefaults = sourceConfig.speech_defaults
    if (
      speechDefaults &&
      typeof speechDefaults === 'object' &&
      !Array.isArray(speechDefaults) &&
      Object.keys(speechDefaults).length
    ) {
      stash.speech_defaults = JSON.parse(JSON.stringify(speechDefaults))
    } else {
      delete stash.speech_defaults
    }
    const transcriptionDefaults = sourceConfig.transcription_defaults
    if (
      transcriptionDefaults &&
      typeof transcriptionDefaults === 'object' &&
      !Array.isArray(transcriptionDefaults) &&
      Object.keys(transcriptionDefaults).length
    ) {
      stash.transcription_defaults = JSON.parse(JSON.stringify(transcriptionDefaults))
    } else {
      delete stash.transcription_defaults
    }
    const taskDefaults = sourceConfig.task_defaults
    if (
      taskDefaults &&
      typeof taskDefaults === 'object' &&
      !Array.isArray(taskDefaults) &&
      Object.keys(taskDefaults).length
    ) {
      stash.task_defaults = JSON.parse(JSON.stringify(taskDefaults))
    } else {
      delete stash.task_defaults
    }
    const reviewedFp = String(sourceConfig.last_reviewed_fingerprint || '').trim()
    if (reviewedFp) {
      stash.last_reviewed_fingerprint = reviewedFp
    } else {
      delete stash.last_reviewed_fingerprint
    }
    return stash
  }
  for (const key of activeParamKeys.value) {
    if (!Object.prototype.hasOwnProperty.call(sourceConfig, key)) continue
    const value = sourceConfig[key]
    const param = catalogParamByKey.value.get(key)
    if (value == null) {
      if (param && param.required !== true) stash[key] = null
      continue
    }
    if (value === '' || (typeof value === 'number' && Number.isNaN(value))) continue
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
  syncRuntimeEnvFromRows()

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
  syncRuntimeEnvFromRows()

  if (!config.value.engines) config.value.engines = {}
  config.value.engines[engineKey] = buildEngineStashFromForm(config.value)
}

function applyEngineSectionToForm(engine) {
  const sec = (config.value.engines && config.value.engines[engine]) || {}
  if (runtimeKeyFor(engine))
    config.value[runtimeKeyFor(engine)] = normalizeSglangRuntime(sec[runtimeKeyFor(engine)])
  const params = catalogParamList.value
  if (isAudioEngineId(engine)) {
    const eng = config.value.engine
    const engMap = config.value.engines
    for (const key of Object.keys(config.value)) {
      if (key !== 'engine' && key !== 'engines') delete config.value[key]
    }
    Object.assign(config.value, JSON.parse(JSON.stringify(sec)))
    config.value.engine = eng
    config.value.engines = engMap
    if (!config.value.load_options || typeof config.value.load_options !== 'object') {
      config.value.load_options = {}
    }
    if (!config.value.session_options || typeof config.value.session_options !== 'object') {
      config.value.session_options = {}
    }
    ensureTtsConfigShape()
    for (const param of audioEditableParams.value) {
      if (!param.required) continue
      const nestedKey = AUDIO_NESTED_SCOPE_KEYS[param.scope]
      if (nestedKey) {
        if (config.value[nestedKey][param.key] == null) {
          config.value[nestedKey][param.key] = defaultValueForAudioParam(param)
        }
      } else if (config.value[param.key] == null) {
        config.value[param.key] = defaultValueForAudioParam(param)
      }
    }
    activeParamKeys.value = []
    initRuntimeEnvRowsFromConfig()

    syncCudaSelectionFromEnv()
    return
  }
  if (!params.length) {
    const eng = config.value.engine
    const engMap = config.value.engines
    for (const k of Object.keys(config.value)) {
      if (k !== 'engine' && k !== 'engines') delete config.value[k]
    }
    Object.assign(config.value, sec)
    config.value.engine = eng
    config.value.engines = engMap
    if (runtimeKeyFor(engine))
      config.value[runtimeKeyFor(engine)] = normalizeSglangRuntime(sec[runtimeKeyFor(engine)])
    initRuntimeEnvRowsFromConfig()

    syncCudaSelectionFromEnv()
    return
  }
  const allowed = new Set([
    'engine',
    'engines',
    'custom_args',
    'model_alias',
    'set_params_by_id',
    'swap_aliases',
    'runtime_env',
    'runtime_env_unset',
    'gpu_mode',
    'gpu_devices',
    ...(runtimeKeyFor(engine) ? [runtimeKeyFor(engine)] : []),
    ...activeParamKeys.value,
  ])
  for (const k of Object.keys(config.value)) {
    if (!allowed.has(k)) delete config.value[k]
  }
  if (Array.isArray(sec.set_params_by_id))
    config.value.set_params_by_id = JSON.parse(JSON.stringify(sec.set_params_by_id))
  else delete config.value.set_params_by_id
  if (Array.isArray(sec.swap_aliases)) config.value.swap_aliases = [...sec.swap_aliases]
  else delete config.value.swap_aliases
  config.value.model_alias = typeof sec.model_alias === 'string' ? sec.model_alias : ''
  config.value.custom_args = typeof sec.custom_args === 'string' ? sec.custom_args : ''
  config.value.runtime_env =
    sec.runtime_env && typeof sec.runtime_env === 'object' && !Array.isArray(sec.runtime_env)
      ? { ...sec.runtime_env }
      : {}
  config.value.runtime_env_unset = Array.isArray(sec.runtime_env_unset)
    ? sec.runtime_env_unset.map((name) => String(name)).filter(Boolean)
    : []
  if (typeof sec.gpu_mode === 'string' && sec.gpu_mode) {
    config.value.gpu_mode = sec.gpu_mode
  } else {
    delete config.value.gpu_mode
  }
  if (Array.isArray(sec.gpu_devices)) {
    config.value.gpu_devices = sec.gpu_devices.map((item) => String(item))
  } else {
    delete config.value.gpu_devices
  }

  for (const p of params) {
    if (!activeParamKeys.value.includes(p.key)) continue
    const v = sec[p.key]
    if (v === null) {
      config.value[p.key] = null
      continue
    }
    if (isDelimitedEnumParam(p)) {
      const normalized = normalizeCsvEnumValue(
        v !== undefined && v !== null && v !== '' ? v : p.default,
        p,
      )
      config.value[p.key] = normalized.length ? normalized : defaultValueForParam(p)
      continue
    }
    config.value[p.key] = Array.isArray(v)
      ? [...v]
      : v !== undefined && v !== null && v !== ''
        ? v
        : defaultValueForParam(p)
  }
  initRuntimeEnvRowsFromConfig()

  syncCudaSelectionFromEnv()
}

// ── Engine change ──────────────────────────────────────────
async function changeEngine(engine) {
  if (engine === config.value.engine) return
  stashCurrentEngineIntoEngines(config.value.engine)
  config.value.engine = engine
  const runtimeKey = runtimeKeyFor(engine)
  if (runtimeKey) {
    config.value[runtimeKey] = normalizeSglangRuntime(config.value.engines?.[engine]?.[runtimeKey])
  } else if (engineRuntimeKey.value) {
    delete config.value[engineRuntimeKey.value]
  }
  paramSearchQuery.value = ''
  hideUnsupportedParams.value = false
  await fetchParamRegistry(engine)
  const sec = (config.value.engines && config.value.engines[engine]) || {}
  setActiveKeysFromSection(sec, catalogParamList.value)
  applyEngineSectionToForm(engine)
}

function changeSglangRuntime(runtime) {
  if (!isSglangEngine.value || normalizeSglangRuntime(runtime) === sglangRuntime.value) return
  const engine = config.value.engine
  stashCurrentEngineIntoEngines(engine)
  const sec = config.value.engines[engine]
  const runtimeKey = runtimeKeyFor(engine)
  sec[runtimeKey] = normalizeSglangRuntime(runtime)
  config.value[runtimeKey] = sec[runtimeKey]
  paramSearchQuery.value = ''
  hideUnsupportedParams.value = false
  setActiveKeysFromSection(sec, catalogParamList.value)
  applyEngineSectionToForm(engine)
}

function companionDropdownOptions(files = [], kind = 'mmproj') {
  const none = { label: 'None', value: '', size: 0 }
  const options = (files || []).map((file) => {
    const filename = file.filename || file.value || ''
    let label = file.label || filename
    if (kind === 'mmproj') {
      const m = filename.match(/(F16|F32|BF16|Q\d[_A-Z0-9]*)/i)
      label = m ? m[1].toUpperCase() : filename
    }
    return { label, value: filename, size: file.size || 0 }
  })
  options.sort((a, b) => a.label.localeCompare(b.label))
  return [none, ...options]
}

function syncCompanionSelectionsFromModel() {
  selectedMmproj.value = model.value?.mmproj_filename || ''
  selectedMtp.value = model.value?.mtp_filename || ''
  selectedDflash.value = model.value?.dflash_filename || ''
}

async function loadCompanions() {
  if (!showCompanionsCard.value || !model.value?.id) {
    companionFiles.value = { mmproj_files: [], mtp_files: [], dflash_files: [] }
    return
  }
  companionsLoading.value = true
  try {
    const data = await modelStore.fetchModelCompanions(model.value.id)
    companionFiles.value = {
      mmproj_files: data?.mmproj_files || [],
      mtp_files: data?.mtp_files || [],
      dflash_files: data?.dflash_files || [],
    }
    syncCompanionSelectionsFromModel()
  } catch (e) {
    companionFiles.value = { mmproj_files: [], mtp_files: [], dflash_files: [] }
    toast.add({
      severity: 'warn',
      summary: 'Companions',
      detail: formatAxiosDetail(e) || 'Could not load companion files from Hugging Face.',
      life: 4000,
    })
  } finally {
    companionsLoading.value = false
  }
}

async function checkForModelUpdates() {
  if (!model.value?.id || refreshingModel.value) return
  refreshingModel.value = true
  try {
    const response = await modelStore.refreshModel(model.value.id)
    if (!response?.updated) {
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
      life: 3500,
    })
  } catch (e) {
    toast.add({
      severity: 'error',
      summary: 'Refresh failed',
      detail: formatAxiosDetail(e) || e.message,
      life: 4000,
    })
  } finally {
    refreshingModel.value = false
  }
}

function onMtpSelected(value) {
  selectedMtp.value = value || ''
  if (selectedMtp.value) selectedDflash.value = ''
}

function onDflashSelected(value) {
  selectedDflash.value = value || ''
  if (selectedDflash.value) selectedMtp.value = ''
}

async function applyCompanion(kind) {
  if (!model.value?.id || companionBusy.value) return
  companionBusyField.value = kind
  try {
    let response
    if (kind === 'mmproj') {
      const opt = mmprojOptions.value.find((o) => o.value === (selectedMmproj.value || ''))
      response = await modelStore.updateModelProjector(
        model.value.id,
        selectedMmproj.value || null,
        opt?.size || 0,
      )
    } else if (kind === 'mtp') {
      const opt = mtpOptions.value.find((o) => o.value === (selectedMtp.value || ''))
      response = await modelStore.updateModelMtp(
        model.value.id,
        selectedMtp.value || null,
        opt?.size || 0,
      )
      if (selectedMtp.value) selectedDflash.value = ''
    } else {
      const opt = dflashOptions.value.find((o) => o.value === (selectedDflash.value || ''))
      response = await modelStore.updateModelDflash(
        model.value.id,
        selectedDflash.value || null,
        opt?.size || 0,
      )
      if (selectedDflash.value) selectedMtp.value = ''
    }

    await modelStore.fetchModels()
    const found = findModelById(route.params.id)
    if (found) model.value = found
    syncCompanionSelectionsFromModel()

    if (response?.applied) {
      toast.add({
        severity: 'success',
        summary: 'Companion updated',
        detail: response.message || 'Applied',
        life: 3000,
      })
    } else {
      toast.add({
        severity: 'success',
        summary: 'Download started',
        detail: response?.message || 'Track progress in notifications',
        life: 3000,
      })
    }
  } catch (e) {
    toast.add({
      severity: 'error',
      summary: 'Companion update failed',
      detail: formatAxiosDetail(e) || e.message,
      life: 4000,
    })
  } finally {
    companionBusyField.value = null
  }
}

// ── Load ───────────────────────────────────────────────────
async function loadAll() {
  loading.value = true
  loadError.value = ''
  draftsEnabled.value = false
  const gpuListPromise = fetchGpuListForBind()
  const engineDescriptorsPromise = enginesStore.fetchEngineDescriptors().catch((error) => {
    console.error('Failed to fetch engine descriptors:', error)
    return []
  })
  try {
    if (!modelStore.models.length) await modelStore.fetchModels()
    const found = findModelById(route.params.id)
    if (!found) {
      loading.value = false
      return
    }
    model.value = found
    syncCompanionSelectionsFromModel()

    const cfgResp = await axios.get(modelApiUrl('/config'))
    const cfg = cfgResp.data
    let engine = cfg.engine ?? found.engine ?? 'llama_cpp'
    // R91: the venv-engine family comes from the descriptors, not a literal list
    // (audio.cpp is cmake, so it stays excluded exactly as before).
    if (found.format !== 'safetensors' && VENV_ENGINE_IDS.includes(engine)) {
      engine = 'llama_cpp'
    }

    const merged = buildWorkingConfigFromApi({ ...cfg, engine })
    suppressAudioRegistryWatch = true
    config.value = merged
    modelLimits.value = cfg.runtime_limits ?? null

    await Promise.all([
      fetchParamRegistry(engine),
      gpuListPromise,
      engineDescriptorsPromise,
      loadCompanions(),
      enginesStore.fetchRuntimePlan?.() ?? Promise.resolve(),
    ])

    const sec = (merged.engines && merged.engines[engine]) || {}
    setActiveKeysFromSection(sec, catalogParamList.value)
    applyEngineSectionToForm(engine)
    savedConfig.value = JSON.parse(JSON.stringify(config.value))
    lookForDraft()
  } catch (e) {
    loadError.value = formatAxiosDetail(e) || 'Could not load configuration'
    toast.add({
      severity: 'error',
      summary: 'Failed to load config',
      detail: loadError.value,
      life: 4000,
    })
  } finally {
    suppressAudioRegistryWatch = false
    loading.value = false
    draftsEnabled.value = Boolean(model.value)
  }
}

// ── Config templates ───────────────────────────────────────
function openParseCommandDialog(seed = '') {
  if (!canImportCommand.value) return
  parseCommandSeed.value = typeof seed === 'string' ? seed : ''
  parseCommandDialogVisible.value = true
}

function applyParsedCommand({ params = [], env = [], customArgs } = {}) {
  let applied = 0
  for (const item of params) {
    if (!item || !item.key) continue
    if (STUDIO_RESERVED_KEYS.has(item.key)) continue
    const catalogParam = catalogParamByKey.value.get(item.key)
    if (!catalogParam || catalogParam.reserved) continue
    if (!activeParamKeys.value.includes(item.key)) addParamKey(item.key)
    const param = catalogParamByKey.value.get(item.key)
    if (isDelimitedEnumParam(param) && !Array.isArray(item.value)) {
      config.value[item.key] = normalizeCsvEnumValue(item.value, param)
    } else if (['repeatable', 'space_list'].includes(param?.value_kind)) {
      config.value[item.key] = Array.isArray(item.value)
        ? [...item.value]
        : item.value == null
          ? []
          : [item.value]
    } else {
      config.value[item.key] = item.value
    }
    applied += 1
  }
  let envApplied = 0
  for (const item of env) {
    if (!item || !item.key) continue
    if (studioEnvSkipReason(item.key)) continue
    const value = item.value == null ? '' : String(item.value)
    if (!value.trim()) continue
    upsertRuntimeEnvRowCanonical(item.key, value)
    envApplied += 1
  }
  if (envApplied) syncCudaSelectionFromEnv()
  if (customArgs !== undefined) {
    config.value.custom_args = typeof customArgs === 'string' ? customArgs : ''
  }
  const parts = []
  if (applied) parts.push(`${applied} parameter${applied === 1 ? '' : 's'}`)
  if (envApplied) parts.push(`${envApplied} env var${envApplied === 1 ? '' : 's'}`)
  toast.add({
    severity: 'success',
    summary: 'Imported command',
    detail: parts.length
      ? `Applied ${parts.join(' and ')} to the form.`
      : 'Updated Custom Arguments from leftover tokens.',
    life: 3000,
  })
}

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
    return true
  } catch (e) {
    configTemplates.value = []
    toast.add({
      severity: 'error',
      summary: 'Templates',
      detail: formatAxiosDetail(e) || 'Could not load templates.',
      life: 4000,
    })
    return false
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
    templateSaveNotice.value = null
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
    const outcome = noteDocumentSaveFailure(toast, e)
    if (outcome) templateSaveNotice.value = outcome
    else {
      toast.add({
        severity: 'error',
        summary: 'Save template failed',
        detail: formatAxiosDetail(e) || 'Could not save template.',
        life: 4000,
      })
    }
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
      enginesStore.invalidateRuntimePlan()
      void enginesStore.fetchRuntimePlanState()
      void enginesStore.fetchRuntimePlan?.()
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
    templateApplyPersist.value = persist
    templateMutationKind.value = 'apply'
    const outcome = noteDocumentSaveFailure(toast, e)
    if (outcome) {
      templateMutationNotice.value = outcome
      if (outcome.committed === true && persist) {
        const reloaded = await refreshPersistedConfig()
        templateMutationNotice.value = reloaded
          ? { ...outcome, refresh: false, retry: false }
          : {
              ...outcome,
              detail:
                'The document was replaced, but it could not be reloaded. Your edits are still here. Refresh before trying again.',
              refresh: true,
              retry: false,
            }
      }
    } else {
      toast.add({
        severity: 'error',
        summary: 'Apply template failed',
        detail: formatAxiosDetail(e) || 'Could not apply template.',
        life: 4000,
      })
    }
  } finally {
    templateApplyLoading.value = false
  }
}

function startTemplateEdit(tpl) {
  templateEditId.value = tpl.id
  templateEditForm.value = {
    name: tpl.name || '',
    description: tpl.description || '',
  }
  templateMutationNotice.value = null
}

function cancelTemplateEdit() {
  templateEditId.value = null
}

async function saveTemplateEdit() {
  const templateId = templateEditId.value
  const name = templateEditForm.value.name.trim()
  if (!templateId || !name) return
  templateEditSaving.value = true
  templateMutationKind.value = 'edit'
  try {
    await axios.put(`/api/model-config-templates/${encodeURIComponent(templateId)}`, {
      name,
      description: templateEditForm.value.description.trim(),
    })
    templateMutationNotice.value = null
    templateEditId.value = null
    await fetchConfigTemplates()
    toast.add({
      severity: 'success',
      summary: 'Template updated',
      detail: `"${name}" saved.`,
      life: 2500,
    })
  } catch (e) {
    const outcome = noteDocumentSaveFailure(toast, e)
    if (outcome) templateMutationNotice.value = outcome
    else {
      toast.add({
        severity: 'error',
        summary: 'Update template failed',
        detail: formatAxiosDetail(e) || 'Could not update template.',
        life: 4000,
      })
    }
  } finally {
    templateEditSaving.value = false
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
    templateMutationKind.value = 'delete'
    const outcome = noteDocumentSaveFailure(toast, e)
    if (outcome) templateMutationNotice.value = outcome
    else {
      toast.add({
        severity: 'error',
        summary: 'Delete failed',
        detail: formatAxiosDetail(e) || 'Could not delete template.',
        life: 4000,
      })
    }
  } finally {
    templateDeleteId.value = null
  }
}

// ── Save ───────────────────────────────────────────────────
function applyLoadedConfig(data) {
  const merged = buildWorkingConfigFromApi(data)
  config.value = merged
  const eng = config.value.engine
  const sec = (merged.engines && merged.engines[eng]) || {}
  setActiveKeysFromSection(sec, catalogParamList.value)
  applyEngineSectionToForm(eng)
  savedConfig.value = JSON.parse(JSON.stringify(config.value))
  clearDraft(route.params.id, config.value.engine)
  draftOffer.value = null
}

async function refreshPersistedConfig() {
  const draft = JSON.parse(JSON.stringify(config.value))
  try {
    const { data } = await axios.get(modelApiUrl('/config'))
    applyLoadedConfig(data)
    persistenceNotice.value = null
    return true
  } catch {
    config.value = draft
    persistenceNotice.value = {
      outcome: 'unknown',
      committed: 'unknown',
      summary: 'Outcome unknown',
      detail:
        'The saved configuration could not be reloaded. Your edits are still here. Refresh before trying again.',
      refresh: true,
      retry: false,
      preserveEdits: true,
      success: false,
    }
    return false
  }
}

function retryPersistence() {
  if (persistenceAction.value === 'apply') {
    void requestApply()
    return
  }
  void saveConfig()
}

function reportPersistenceFailure(error, action) {
  const outcome = classifyPersistenceError(error)
  if (!outcome) return false
  persistenceAction.value = action
  persistenceNotice.value = {
    ...outcome,
    retry: action === 'apply' ? outcome.code === 'STORE_QUEUE_FULL' : outcome.retry,
  }
  if (action === 'apply' && outcome.committed === true) {
    persistenceNotice.value = {
      ...outcome,
      retry: false,
      refresh: true,
      detail:
        'The change was stored, but acknowledgement failed. Refresh before trying the action again.',
    }
  }
  toast.add({
    severity: outcome.committed === true ? 'warn' : 'error',
    summary: persistenceNotice.value.summary,
    detail: persistenceNotice.value.detail,
    life: 6000,
  })
  return true
}

async function saveConfig() {
  if (!requestAliasesValid.value) return
  saving.value = true
  const draft = JSON.parse(JSON.stringify(config.value))
  try {
    const payload = buildPersistedPayload(config.value)
    const { data } = await axios.put(modelApiUrl('/config'), payload)
    applyLoadedConfig(data)
    enginesStore.invalidateRuntimePlan()
    void enginesStore.fetchRuntimePlanState()
    void enginesStore.fetchRuntimePlan?.()
    refreshSavedCmdPreviewIfVisible()
    persistenceNotice.value = null
    toast.add({
      severity: 'success',
      summary: 'Saved',
      detail: 'Saved. Check the runtime plan below for any engine changes to apply.',
      life: 3000,
    })
    return true
  } catch (e) {
    config.value = draft
    if (reportPersistenceFailure(e, 'save')) {
      if (persistenceNotice.value?.committed === true) {
        const notice = persistenceNotice.value
        const reloaded = await refreshPersistedConfig()
        if (reloaded) {
          persistenceNotice.value = { ...notice, refresh: false, retry: false }
        } else {
          config.value = draft
          persistenceNotice.value = {
            ...notice,
            detail:
              'The document was replaced, but it could not be reloaded. Your edits are still here. Refresh before trying again.',
            refresh: true,
            retry: false,
          }
        }
      }
      return false
    }
    const detail = formatAxiosDetail(e) || 'Save failed'
    toast.add({ severity: 'error', summary: 'Save failed', detail, life: 4000 })
    return false
  } finally {
    saving.value = false
  }
}

async function requestApply() {
  if (hasUnsavedChanges.value) {
    const ok = await saveConfig()
    if (!ok) return
  }
  try {
    await enginesStore.fetchRuntimePlan()
  } catch {
    /* The impact copy still describes the global proxy reload. */
  }
  applyImpactVisible.value = true
}

function cancelApplyImpact() {
  applyWithheld.value = null
  applyImpactVisible.value = false
}

function requestApplyStop() {
  applyStopRequested.value = true
}

async function followRuntimeApply(initial) {
  let data = initial
  const deadline = Date.now() + 120000
  while (data?.status === 'running' && data.operation_id && Date.now() < deadline) {
    if (applyStopRequested.value) {
      applyStopRequested.value = false
      try {
        await axios.post(
          modelApiUrl(`/runtime/apply/${encodeURIComponent(data.operation_id)}/cancel`),
        )
      } catch {
        /* The next poll still reports whether the apply stopped. */
      }
    }
    await new Promise((resolve) => setTimeout(resolve, 400))
    const next = await axios.get(
      modelApiUrl(`/runtime/apply/${encodeURIComponent(data.operation_id)}`),
    )
    data = next.data
  }
  return data
}

async function confirmApplyImpact() {
  const finished = await applyModelSettings()
  if (finished !== false) {
    applyWithheld.value = null
    applyImpactVisible.value = false
  }
}

function stayOnPage() {
  finishLeave(false)
}

function discardAndLeave() {
  restoreSavedConfig()
  draftOffer.value = null
  finishLeave(true)
}

async function saveAndLeave() {
  const ok = await saveConfig()
  finishLeave(Boolean(ok))
}

function persistDraftNow() {
  const modelId = String(route.params.id || '')
  const engine = config.value?.engine
  if (!draftsEnabled.value || loading.value || !modelId) return
  if (!hasUnsavedChanges.value) {
    clearDraft(modelId, engine)
    return
  }
  writeDraft(modelId, engine, {
    engine,
    config: JSON.parse(JSON.stringify(config.value)),
    activeParamKeys: [...activeParamKeys.value],
  })
}

function lookForDraft() {
  const parsed = readDraft(route.params.id, config.value?.engine)
  if (!parsed) {
    draftOffer.value = null
    return
  }
  try {
    if (JSON.stringify(parsed.config) === JSON.stringify(config.value)) {
      clearDraft(route.params.id, config.value?.engine)
      draftOffer.value = null
      return
    }
  } catch {
    /* Offer the draft when it cannot be compared. */
  }
  draftOffer.value = parsed
}

function restoreDraft() {
  if (!draftOffer.value?.config) return
  config.value = JSON.parse(JSON.stringify(draftOffer.value.config))
  if (Array.isArray(draftOffer.value.activeParamKeys)) {
    activeParamKeys.value = [...draftOffer.value.activeParamKeys]
  }
  const engine = config.value.engine
  const sec = (config.value.engines && config.value.engines[engine]) || {}
  setActiveKeysFromSection(sec, catalogParamList.value)
  applyEngineSectionToForm(engine)
  draftOffer.value = null
}

function discardDraftOffer() {
  clearDraft(route.params.id, config.value?.engine)
  draftOffer.value = null
}

async function applyModelSettings() {
  applyingRuntime.value = true
  applyStopRequested.value = false
  try {
    await enginesStore.fetchRuntimePlan()
    if (selectiveModelApply.value) {
      const entry = modelLaunchPlan.value
      const mode = entry.running ? 'restart_now' : 'next_start'
      const confirmation = applyWithheld.value || {}
      const started = await axios.post(modelApiUrl('/runtime/apply'), {
        mode,
        expected_desired_revision: entry.desired_revision,
        expected_published_revision: entry.published_revision,
        idempotency_key: `${enginesStore.runtimePlan?.plan_id || 'plan'}:${entry.model_id}:${mode}`,
        confirm_operation_id: confirmation.confirm_operation_id,
        confirm_state: confirmation.confirm_state,
      })
      const data = await followRuntimeApply(started.data)
      toast.add({
        severity: data?.status === 'succeeded' ? 'success' : 'warn',
        summary: data?.status === 'succeeded' ? 'Model settings applied' : 'Apply did not finish',
        detail: data?.message || 'The model apply finished.',
        life: 5000,
      })
      await enginesStore.fetchRuntimePlan()
      await enginesStore.fetchRuntimePlanState()
      await modelStore.fetchModels()
      const refreshedModel = findModelById(route.params.id)
      if (refreshedModel) model.value = refreshedModel
      refreshSavedCmdPreviewIfVisible()
      return true
    }
    throw new Error('The runtime plan changed. Refresh this model before applying settings.')
  } catch (e) {
    const held = withheldConfirmation(e)
    if (held) {
      applyWithheld.value = held
      return false
    }
    if (!reportPersistenceFailure(e, 'apply')) {
      const detail = formatAxiosDetail(e) || 'Apply failed'
      toast.add({ severity: 'error', summary: 'Apply failed', detail, life: 5000 })
    }
    return true
  } finally {
    applyingRuntime.value = false
    applyStopRequested.value = false
  }
}

// ── Reset ──────────────────────────────────────────────────
function restoreSavedConfig() {
  config.value = JSON.parse(JSON.stringify(savedConfig.value))
  const engine = config.value.engine
  const sec = (config.value.engines && config.value.engines[engine]) || {}
  setActiveKeysFromSection(sec, catalogParamList.value)
  applyEngineSectionToForm(engine)
  clearDraft(route.params.id, engine)
}

function resetConfig() {
  restoreSavedConfig()
  draftOffer.value = null
  toast.add({
    severity: 'info',
    summary: 'Reset',
    detail: 'Config reset to saved values',
    life: 2000,
  })
}

async function fetchSavedCmdPreview() {
  if (!model.value) return
  cmdPreviewLoading.value = true
  cmdPreviewError.value = null
  try {
    const { data } = await axios.get(modelApiUrl('/saved-command'))
    if (data?.ok && (data.engine_command || data.cmd)) {
      cmdPreviewText.value = data.engine_command || data.cmd
      cmdPreviewLauncherText.value = data.launcher_command || ''
      cmdPreviewRevisionText.value = formatRevisionPreview(data)
      cmdPreviewEnvText.value = previewEnvLines(data)
      cmdPreviewFiltersText.value = formatFiltersPreview(data.filters)
      cmdPreviewAliasesText.value = formatAliasesPreview(data.aliases)
      cmdPreviewSidecarText.value = formatSidecarPreview(data.sidecar)
      cmdPreviewSidecarPath.value = data.sidecar_path || ''
      cmdPreviewError.value = null
    } else {
      cmdPreviewText.value = ''
      cmdPreviewEnvText.value = ''
      cmdPreviewFiltersText.value = ''
      cmdPreviewAliasesText.value = ''
      cmdPreviewSidecarText.value = ''
      cmdPreviewSidecarPath.value = ''
      cmdPreviewError.value = data?.error || 'Could not build saved command.'
    }
  } catch (e) {
    cmdPreviewText.value = ''
    cmdPreviewEnvText.value = ''
    cmdPreviewFiltersText.value = ''
    cmdPreviewAliasesText.value = ''
    cmdPreviewSidecarText.value = ''
    cmdPreviewSidecarPath.value = ''
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
    const { data } = await axios.post(modelApiUrl('/preview-command'), payload, {
      signal,
    })
    if (requestId !== unsavedPreviewRequestId) return
    if (data?.ok && (data.engine_command || data.cmd)) {
      unsavedCmdPreviewText.value = data.engine_command || data.cmd
      unsavedCmdPreviewLauncherText.value = data.launcher_command || ''
      unsavedCmdPreviewRevisionText.value = formatRevisionPreview(data)
      unsavedCmdPreviewEnvText.value = previewEnvLines(data)
      unsavedCmdPreviewFiltersText.value = formatFiltersPreview(data.filters)
      unsavedCmdPreviewAliasesText.value = formatAliasesPreview(data.aliases)
      unsavedCmdPreviewSidecarText.value = formatSidecarPreview(data.sidecar)
      unsavedCmdPreviewSidecarPath.value = data.sidecar_path || ''
      unsavedCmdPreviewError.value = null
    } else {
      unsavedCmdPreviewText.value = ''
      unsavedCmdPreviewEnvText.value = ''
      unsavedCmdPreviewFiltersText.value = ''
      unsavedCmdPreviewAliasesText.value = ''
      unsavedCmdPreviewSidecarText.value = ''
      unsavedCmdPreviewSidecarPath.value = ''
      unsavedCmdPreviewError.value = data?.error || 'Could not build preview command.'
    }
  } catch (e) {
    if (axios.isCancel?.(e) || e?.code === 'ERR_CANCELED' || e?.name === 'CanceledError') {
      return
    }
    if (requestId !== unsavedPreviewRequestId) return
    unsavedCmdPreviewText.value = ''
    unsavedCmdPreviewEnvText.value = ''
    unsavedCmdPreviewFiltersText.value = ''
    unsavedCmdPreviewAliasesText.value = ''
    unsavedCmdPreviewSidecarText.value = ''
    unsavedCmdPreviewSidecarPath.value = ''
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

watch(
  [() => config.value?.engine, () => config.value?.family, () => config.value?.task],
  ([engine], [prevEngine]) => {
    if (suppressAudioRegistryWatch || loading.value) return
    if (!isAudioEngineId(engine)) return
    if (prevEngine && !isAudioEngineId(prevEngine)) {
      void fetchParamRegistry(audioEngineId)
      return
    }
    scheduleAudioRegistryRefresh()
  },
)

watch(
  config,
  () => {
    persistDraftNow()
  },
  { deep: true },
)
watch(
  activeParamKeys,
  () => {
    persistDraftNow()
  },
  { deep: true },
)

// ── Lifecycle ──────────────────────────────────────────────
onMounted(loadAll)
onBeforeUnmount(() => {
  if (audioRegistryRefreshTimer) clearTimeout(audioRegistryRefreshTimer)
  if (unsavedPreviewTimer) clearTimeout(unsavedPreviewTimer)
  if (unsavedPreviewAbort) unsavedPreviewAbort.abort()
})
</script>

<style scoped>
.model-config-view.page-shell {
  gap: var(--spacing-md);
}

.config-scan-message {
  margin: 0;
}

.config-page-title {
  display: flex;
  flex-direction: row;
  flex-wrap: wrap;
  align-items: center;
  gap: 0.4rem 0.75rem;
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

.runtime-env-row {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 0.5rem;
  margin-bottom: 0.5rem;
}

.runtime-env-row :deep(.runtime-env-key) {
  flex: 0 0 11rem;
  width: 11rem;
  max-width: 100%;
  min-width: 0;
}

.runtime-env-row :deep(.runtime-env-mode) {
  flex: 0 0 10.75rem;
  width: 10.75rem;
  max-width: 100%;
  min-width: 0;
}

.runtime-env-row :deep(.runtime-env-value) {
  flex: 1 1 12rem;
  width: auto;
  min-width: 8rem;
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

.hf-link:hover {
  text-decoration: underline;
}

.param-field--unsupported {
  opacity: 0.88;
}

.param-supported-tag {
  margin-left: 0.35rem;
  vertical-align: middle;
  font-size: var(--text-xs) !important;
}

/* ── Card ─────────────────────────────────────────────── */
.config-card {
  background: var(--bg-card);
  border: 1px solid var(--border-primary);
  border-radius: var(--radius-lg);
  padding: var(--spacing-lg);
}

.workbench-block {
  margin-top: 0.75rem;
  padding-top: 0.75rem;
  border-top: 1px solid var(--border-primary, #2a2f45);
}

.config-card > .workbench-block:first-child {
  margin-top: 0;
  padding-top: 0;
  border-top: none;
}

.gpu-bind {
  display: flex;
  flex-direction: column;
  align-items: stretch;
  gap: 0.65rem;
}

.gpu-mode {
  display: flex;
  flex-wrap: wrap;
  gap: 0.4rem;
}

.gpu-mode__option {
  padding: 0.22rem 0.65rem;
  border-radius: var(--radius-md);
  border: 1px solid var(--border-primary);
  background: transparent;
  color: var(--text-secondary);
  font: inherit;
  font-size: 0.8125rem;
  font-weight: 600;
  cursor: pointer;
}

.gpu-mode__option.selected {
  border-color: var(--accent-cyan);
  background: rgba(34, 211, 238, 0.1);
  color: var(--accent-cyan);
}

.gpu-card-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(14rem, 1fr));
  gap: 0.5rem;
}

.gpu-card {
  display: flex;
  align-items: center;
  gap: 0.6rem;
  min-width: 0;
  padding: 0.45rem 0.65rem;
  border-radius: var(--radius-md);
  border: 1px solid var(--border-primary);
  background: transparent;
  color: inherit;
  font: inherit;
  text-align: left;
  cursor: pointer;
}

.gpu-card.selected {
  border-color: var(--accent-cyan);
  background: rgba(34, 211, 238, 0.1);
}

.gpu-card__index {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
  width: 1.5rem;
  height: 1.5rem;
  border-radius: var(--radius-sm);
  border: 1px solid var(--border-primary);
  background: var(--bg-tertiary);
  color: var(--text-secondary);
  font-size: var(--text-xs);
  font-weight: 700;
}

.gpu-card__body {
  display: flex;
  flex-direction: column;
  min-width: 0;
  gap: 0.1rem;
}

.gpu-card__name {
  font-size: 0.8125rem;
  font-weight: 600;
  color: var(--text-primary);
}

.gpu-card__meta {
  font-size: var(--text-xs);
  color: var(--text-secondary);
}

.advanced-block,
.advanced-field {
  min-width: 0;
}

.config-launch > * + .advanced-block,
.config-launch > * + .config-split-grid {
  margin-top: 0.75rem;
  padding-top: 0.75rem;
  border-top: 1px solid var(--border-primary, #2a2f45);
}

.advanced-field__label {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 0.35rem;
  margin-bottom: 0.35rem;
  font-size: 0.875rem;
  font-weight: 600;
  color: var(--text-primary);
}

.alias-link {
  font-weight: 500;
  letter-spacing: normal;
  text-transform: none;
  color: var(--accent-cyan, #22d3ee);
}

.param-tag-cloud-wrap {
  margin-top: 0.55rem;
}

.companions-loading {
  color: var(--text-secondary, #9ca3af);
  font-size: 0.875rem;
}

.companions-grid {
  display: grid;
  grid-template-columns: 1fr;
  gap: 0.875rem;
}

@media (min-width: 900px) {
  .companions-grid {
    grid-template-columns: repeat(3, minmax(0, 1fr));
  }
}

.companion-field label {
  display: block;
  font-size: 0.8rem;
  color: var(--text-secondary, #9ca3af);
  margin-bottom: 0.35rem;
}

.companion-field__row {
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
  align-items: center;
}

.companion-field__row .p-select {
  flex: 1;
  min-width: 0;
}

.section-label {
  font-size: 0.875rem;
  font-weight: 600;
  text-transform: none;
  letter-spacing: 0;
  color: var(--text-primary);
  margin-bottom: 0.5rem;
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 0.35rem 0.5rem;
}

.section-hint {
  font-weight: 400;
  text-transform: none;
  letter-spacing: normal;
  color: var(--text-secondary, #9ca3af);
  opacity: 0.7;
}

.section-hint-link {
  color: var(--primary-color, #3b82f6);
  text-decoration: underline;
  text-underline-offset: 2px;
}

/* ── Engine selector ──────────────────────────────────── */
.engine-selector {
  display: flex;
  flex-wrap: wrap;
  gap: 0.4rem;
}

.engine-option {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  justify-content: center;
  width: auto;
  padding: 0.2rem 0.6rem;
  border-radius: var(--radius-md);
  border: 1px solid var(--border-primary, #2a2f45);
  cursor: pointer;
  transition:
    border-color 0.15s,
    background 0.15s;
  font: inherit;
  font-size: 0.8125rem;
  user-select: none;
  box-sizing: border-box;
  background: transparent;
  color: inherit;
  text-align: center;
}

.engine-option:focus-visible {
  outline: 2px solid var(--accent-cyan);
  outline-offset: 2px;
}

.engine-option:disabled {
  cursor: not-allowed;
  opacity: 0.55;
}

.config-status {
  display: flex;
  flex-wrap: wrap;
  align-items: baseline;
  gap: 0.35rem 0.75rem;
  margin: 0;
  padding: 0.65rem 0.85rem;
  border: 1px solid var(--border-primary);
  border-radius: var(--radius-lg);
  background: var(--bg-card);
}

.runtime-state {
  display: flex;
  flex-wrap: wrap;
  gap: 0.35rem 0.5rem;
  align-items: center;
  margin: 0;
  font-size: 0.8125rem;
  font-weight: 600;
  color: var(--text-secondary);
}

.runtime-state .is-current {
  color: var(--text-primary);
}

.runtime-state .is-current::before {
  content: '';
  display: inline-block;
  width: 0.45rem;
  height: 0.45rem;
  margin-right: 0.35rem;
  border-radius: 50%;
  background: var(--accent-cyan);
  vertical-align: 0;
}

.runtime-state__detail {
  margin: 0;
  flex: 1 1 16rem;
  min-width: 0;
  font-size: 0.875rem;
  line-height: 1.45;
  color: var(--text-secondary);
}

.config-launch .config-split-grid {
  gap: 0.65rem;
}

.engine-option:hover {
  border-color: var(--accent-cyan, #22d3ee);
  background: rgba(34, 211, 238, 0.05);
}

.engine-option.selected {
  border-color: var(--accent-cyan);
  background: rgba(34, 211, 238, 0.1);
  color: var(--accent-cyan);
  font-weight: 600;
}

.engine-option.disabled {
  cursor: not-allowed;
  opacity: 0.5;
}

.engine-option.disabled:hover {
  border-color: var(--border-primary, #2a2f45);
  background: transparent;
}

.engine-disabled-reason {
  flex-basis: 100%;
  max-width: 100%;
  margin: 0.2rem 0 0;
  color: var(--text-secondary, #9ca3af);
  font-size: var(--text-xs);
  line-height: 1.3;
  text-align: center;
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
  min-width: 1.5rem;
  height: 1.5rem;
  padding: 0 0.45rem;
  border-radius: var(--radius-sm);
  font-size: var(--text-xs);
  font-weight: 700;
  line-height: 1;
  letter-spacing: 0.04em;
  color: var(--text-secondary);
  background: var(--bg-tertiary);
  border: 1px solid var(--border-primary);
  box-shadow: none;
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
  grid-template-columns: repeat(auto-fill, minmax(min(100%, 17rem), 1fr));
  gap: 0.75rem 1rem;
}

.param-field {
  display: flex;
  flex-direction: column;
  gap: 0.35rem;
  min-width: 0;
}

.param-field__label {
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto;
  align-items: center;
  column-gap: 0.5rem;
  row-gap: 0.15rem;
  flex: 1;
  min-width: 0;
  font-size: 0.875rem;
  font-weight: 500;
  color: var(--text-secondary);
}

.param-field__name {
  grid-column: 1;
  grid-row: 1;
  min-width: 0;
}

.param-field__label > .param-key-hint {
  grid-column: 1;
  grid-row: 2;
  margin-left: 0;
  justify-self: start;
}

.param-field__state {
  grid-column: 2;
  grid-row: 1;
  justify-self: end;
}

.param-input {
  width: 100%;
}

.audio-capability-tags {
  display: flex;
  flex-wrap: wrap;
  gap: 0.45rem;
}

.request-cap-toggle {
  display: flex;
  flex-direction: column;
  align-items: flex-start;
  gap: 0.15rem;
  width: 100%;
  padding: 0;
  border: 0;
  background: transparent;
  color: inherit;
  text-align: left;
  cursor: pointer;
}

.request-cap-toggle__title {
  display: inline-flex;
  align-items: center;
  gap: 0.45rem;
}

.request-cap-toggle__hint {
  margin: 0;
}

.request-cap-grid {
  display: grid;
  grid-template-columns: minmax(7.5rem, auto) minmax(0, 1fr) auto;
  gap: 0.2rem 0.65rem;
  margin-top: 0.55rem;
  padding-top: 0.55rem;
  border-top: 1px solid var(--border-primary, #2a2f45);
  font-size: var(--text-xs);
  line-height: 1.25;
}

.request-cap-item {
  display: contents;
}

.request-cap-item__key {
  color: var(--text-secondary, #9ca3af);
  font-size: var(--text-xs);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.request-cap-item__label {
  color: var(--text-primary, #e5e7eb);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.request-cap-item__info {
  justify-self: end;
}

.config-card--compact {
  padding: 0.75rem 0.9rem;
}

.section-label--inline {
  margin: 0;
}

.tts-profile-summary {
  margin-top: 0.35rem;
}

.tts-subsection {
  margin-top: 0.9rem;
  padding-top: 0.75rem;
  border-top: 1px solid var(--border-primary, #2a2f45);
}

.tts-subsection__head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 0.75rem;
  margin-bottom: 0.35rem;
}

.tts-subsection__title {
  font-size: 0.82rem;
  font-weight: 600;
  color: var(--text-primary, #e5e7eb);
}

.tts-speech-group + .tts-speech-group {
  margin-top: 0.75rem;
}

.tts-speech-group__label {
  font-size: var(--text-xs);
  font-weight: 600;
  margin-bottom: 0.2rem;
}

.voice-preset-card {
  border: 1px solid var(--border-primary, #2a2f45);
  border-radius: var(--radius-md, 0.5rem);
  padding: 0.65rem 0.75rem;
  margin-bottom: 0.65rem;
  background: var(--bg-surface, rgba(255, 255, 255, 0.02));
}

.voice-preset-card__head {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  margin-bottom: 0.55rem;
}

.voice-preset-card__name {
  flex: 1;
}

.voice-preset-card__grid {
  display: grid;
  gap: 0.65rem;
}

@media (min-width: 900px) {
  .voice-preset-card__grid {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
}

.param-info {
  font-size: var(--text-xs);
  cursor: help;
  opacity: 0.6;
}

/* ── Actions (sticky bar) ───────────────────────────────── */
.config-actions {
  display: flex;
  gap: 0.5rem;
  justify-content: flex-end;
  flex-wrap: wrap;
  position: sticky;
  bottom: calc(var(--bottom-nav-height) + 0.75rem);
  z-index: 10;
  margin-top: var(--spacing-sm);
  padding: 0.75rem 1rem;
  border: 1px solid var(--border-primary);
  border-radius: var(--radius-lg);
  background: var(--bg-card);
  box-shadow: var(--shadow-md);
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
  font-size: var(--text-xs);
  color: var(--text-secondary, #9ca3af);
}

/* ── Unsaved indicator ─────────────────────────────────── */
.unsaved-tag {
  font-size: var(--text-xs);
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

.config-toolbar__row .toggle-field {
  flex-shrink: 0;
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

.apply-diff {
  margin: 0.75rem 0 0;
  padding-left: 1.1rem;
  font-size: 0.875rem;
  overflow-wrap: anywhere;
}

.apply-diff__field {
  font-weight: 600;
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
  border-radius: var(--radius-md);
  border: 1px solid var(--border-primary);
  background: color-mix(in srgb, var(--accent-cyan) 8%, var(--bg-card));
  color: var(--text-primary);
  font-size: 0.8125rem;
  cursor: pointer;
  transition:
    border-color 0.15s ease,
    background 0.15s ease,
    transform 0.12s ease;
}

.param-search-tag:hover {
  border-color: var(--accent-cyan, #22d3ee);
  background: color-mix(in srgb, var(--accent-cyan) 12%, var(--bg-primary));
}

.param-search-tag:focus-visible {
  outline: 2px solid var(--accent-cyan, #22d3ee);
  outline-offset: 2px;
}

.param-search-tag__key {
  font-size: var(--text-xs);
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
  margin-left: 0;
  padding: 0.1rem 0.35rem;
  font-size: var(--text-xs);
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

.config-templates-list-item--edit {
  display: block;
}

.config-templates-edit {
  display: grid;
  gap: 0.4rem;
}

.config-templates-actions {
  display: flex;
  flex: 0 0 auto;
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
  font-size: var(--text-xs);
  color: var(--text-secondary, #9ca3af);
  opacity: 0.85;
}

@media (max-width: 768px) {
  .companion-field__row {
    flex-wrap: wrap;
  }

  .config-actions {
    flex-wrap: wrap;
    justify-content: stretch;
  }

  .config-actions .p-button {
    flex: 1 1 auto;
  }

  .config-templates-list-item {
    flex-direction: column;
    align-items: stretch;
  }

  .config-search-wrap {
    min-width: 0;
    width: 100%;
  }

  .toggle-field {
    flex: 1 1 100%;
  }
}

.config-secondary-actions {
  position: relative;
}
.config-secondary-actions summary {
  cursor: pointer;
  padding: 0.65rem 0.8rem;
  min-height: 2.75rem;
  border: 1px solid var(--border-secondary);
  border-radius: var(--radius-sm);
  color: var(--text-secondary);
}
.config-secondary-actions__menu {
  position: absolute;
  bottom: calc(100% + 0.5rem);
  right: 0;
  min-width: 14rem;
  max-width: calc(100vw - 4rem);
  padding: 0.5rem;
  display: grid;
  gap: 0.5rem;
  background: var(--bg-secondary);
  border: 1px solid var(--border-secondary);
  border-radius: var(--radius-md);
  box-shadow: var(--shadow-lg);
}
.param-default-help {
  font-size: var(--text-xs);
  line-height: 1.6;
  color: var(--text-secondary);
}
.param-reset {
  justify-self: start;
  width: fit-content;
  border: 0;
  background: transparent;
  color: var(--accent-primary);
  padding: 0.25rem 0;
  cursor: pointer;
  font-size: var(--text-xs);
}
</style>
