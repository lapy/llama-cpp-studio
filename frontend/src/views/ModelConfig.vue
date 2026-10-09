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
      <Button label="Back to Models" icon="pi pi-arrow-left" severity="secondary" outlined @click="$router.push('/models')" />
    </EmptyState>

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
          <Button icon="pi pi-arrow-left" text severity="secondary" aria-label="Back to Models" @click="requestLeave('/models')" />
        </template>
        <template #title>
          <div class="config-page-title">
            <h1 class="page-title">{{ model.display_name || model.base_model_name }}</h1>
            <div class="header-meta">
              <Tag :value="model.format || 'gguf'" severity="info" />
              <Tag v-if="model.quantization" :value="model.quantization" severity="secondary" />
              <Tag v-if="model.family" :value="model.family" severity="secondary" />
              <Tag v-for="task in model.tasks || []" :key="task" :value="task" severity="success" />
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
          <Tag v-if="hasUnsavedChanges" value="Unsaved changes" severity="warn" class="unsaved-tag" />
        </template>
      </PageHeader>

      <div class="config-status" aria-live="polite">
        <div class="runtime-state">
          <span :class="{ 'is-current': hasUnsavedChanges }">Unsaved</span>
          <span aria-hidden="true">→</span>
          <span :class="{ 'is-current': !hasUnsavedChanges && pendingApply }">Pending</span>
          <span aria-hidden="true">→</span>
          <span :class="{ 'is-current': !hasUnsavedChanges && !pendingApply }">{{ publishedStepLabel }}</span>
        </div>
        <p class="runtime-state__detail" :title="runtimeStateDetail">{{ runtimeStateDetail }}</p>
      </div>

      <div v-if="loadError" class="state-banner" role="alert">
        <span>Could not refresh this configuration. {{ loadError }}</span>
        <Button label="Retry" size="small" severity="secondary" outlined @click="loadAll" />
      </div>

      <div v-if="draftOffer" class="state-banner" role="status">
        <span>A draft of your edits for this model and engine is saved in this session.</span>
        <Button label="Restore draft" size="small" @click="restoreDraft" />
        <Button label="Discard draft" size="small" severity="secondary" outlined @click="discardDraftOffer" />
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
            v-tooltip.bottom="eng.disabledReason || (eng.runnable || !eng.descriptor ? '' : 'No active version. Activate one from Engines.')"
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
              <span
                v-else-if="eng.value === 'unsloth_llama'"
                class="engine-mark engine-mark--unsloth"
                aria-hidden="true"
              >US</span>
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
              <i
                v-else-if="['sglang', 'sglang_v100', 'vllm'].includes(eng.value)"
                class="pi pi-sparkles engine-icon-lmdeploy"
                aria-hidden="true"
              />
              <span
                v-else-if="eng.value === 'audio_cpp'"
                class="engine-mark engine-mark--audio"
                aria-hidden="true"
              >A</span>
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
          {{ unavailableSavedEngine.label }} is saved for this model. It is hidden because it is not installed or not compatible.
        </p>
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
            <i class="pi pi-info-circle param-info" tabindex="0" aria-label="About basics" v-tooltip.top="'Common settings for this engine. Default is inherited. Override is saved with this model.'" />
          </div>
          <div class="params-grid">
          <div v-for="param in basicParams" :key="param.key" class="param-field">
            <label :for="`basic-${param.key}`" class="param-field__label">
              <span class="param-field__name">{{ param.label }}</span>
              <code class="param-key-hint">{{ param.key }}</code>
              <Tag class="param-field__state" :value="isExplicitOverride(param) ? 'Override' : 'Default'" :severity="isExplicitOverride(param) ? 'info' : 'secondary'" />
            </label>
            <InputNumber
              :id="`basic-${param.key}`"
              v-model="config[param.key]"
              :placeholder="param.default != null ? String(param.default) : 'Engine default'"
              class="param-input"
              :disabled="param.supported === false"
            />
          </div>
          </div>
        </div>

        <div v-if="showNvidiaGpuBind" class="workbench-block gpu-bind">
          <div class="section-label section-label--inline">
            GPUs
            <i class="pi pi-info-circle param-info" tabindex="0" aria-label="About GPU assignment" v-tooltip.top="'All GPUs follows the deployment. Choose GPUs to pick devices in the order you click them. CPU only hides every CUDA device. The first card you pick becomes device 0.'" />
          </div>
          <div id="gpu-mode" class="gpu-mode" role="radiogroup" aria-label="GPU assignment mode">
            <button
              v-for="mode in gpuModeOptions"
              :key="mode.value"
              type="button"
              class="gpu-mode__option"
              role="radio"
              :class="{ selected: (config.gpu_mode || 'inherit') === mode.value }"
              :aria-checked="(config.gpu_mode || 'inherit') === mode.value ? 'true' : 'false'"
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
                <span class="gpu-card__meta">{{ gpu.order != null ? `Device ${gpu.order}` : 'Not used' }}</span>
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
                <ToggleSwitch v-model="hideUnsupportedParams" input-id="toggle-hide-unsupported" />
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
            No parameters match. Try other words, turn off “hide unsupported”, or clear the search.
          </Message>
            </div>
          <p v-if="!paneParams.length" class="config-muted-hint">
            Search above to add parameters. Saved values that differ from the engine default appear here.
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
                    <i class="pi pi-info-circle param-info" v-tooltip.top="paramDescriptionTooltip(param)" />
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
              <Select
                v-else-if="param.value_kind === 'flag' && param.negative_flag"
                :id="`p-${param.sectionId}-${param.key}`"
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
                v-else-if="param.value_kind === 'repeatable'"
                :id="`p-${param.sectionId}-${param.key}`"
                v-model="config[param.key]"
                delimiter=","
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
              <Select
                v-else-if="param.options && param.options.length"
                :id="`p-${param.sectionId}-${param.key}`"
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
              <ToggleSwitch
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
      </div>
      </div>

      <Message
        v-if="!isAudioEngine && paramRegistry.scan_error"
        severity="warn"
        :closable="false"
        class="config-scan-message"
      >
        Could not read engine CLI help: {{ paramRegistry.scan_error }}. Open Engines and use
        <strong>Rescan CLI parameters</strong> for this engine.
      </Message>
      <Message
        v-else-if="!isAudioEngine && paramRegistry.scan_pending"
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
        {{ isAudioEngine
          ? 'Unrecognized audio.cpp keys are preserved on save:'
          : 'Deprecated or unrecognized saved keys for this engine will be dropped on the next save:' }}
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
          <i class="pi pi-info-circle param-info" tabindex="0" aria-label="About GPU assignment" v-tooltip.top="'All GPUs follows the deployment. Choose GPUs to pick devices in the order you click them. CPU only hides every CUDA device. The first card you pick becomes device 0.'" />
        </div>
        <div id="gpu-mode" class="gpu-mode" role="radiogroup" aria-label="GPU assignment mode">
          <button
            v-for="mode in gpuModeOptions"
            :key="mode.value"
            type="button"
            class="gpu-mode__option"
            role="radio"
            :class="{ selected: (config.gpu_mode || 'inherit') === mode.value }"
            :aria-checked="(config.gpu_mode || 'inherit') === mode.value ? 'true' : 'false'"
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
              <span class="gpu-card__meta">{{ gpu.order != null ? `Device ${gpu.order}` : 'Not used' }}</span>
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
        :llama-swap-stable-id="llamaSwapStableId"
        :model-id="model?.id || ''"
        @rescan-complete="fetchParamRegistry('audio_cpp', { rescan: true })"
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
          <i class="pi pi-info-circle param-info" tabindex="0" aria-label="About companions" v-tooltip.top="'Attach mmproj, MTP, or DFlash weights. MTP and DFlash cannot be used together.'" />
        </div>
        <div v-if="companionsLoading" class="companions-loading">Loading companion options…</div>
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
          {{ config.engine === 'audio_cpp' ? 'Model ID while running' : 'Stable ID' }}
        </label>
        <InputText
          :model-value="llamaSwapStableId"
          readonly
          class="w-full"
          :aria-label="config.engine === 'audio_cpp' ? 'Stable model ID' : 'Stable llama-swap model ID'"
        />
      </div>

      <div class="advanced-field">
        <label class="advanced-field__label">
          {{ config.engine === 'audio_cpp' ? 'Friendly API name' : 'API alias' }}
          <i
            class="pi pi-info-circle param-info"
            tabindex="0"
            aria-label="About API alias"
            v-tooltip.top="config.engine === 'audio_cpp'
              ? 'Optional name apps send as model. Running state uses the stable ID.'
              : 'Optional id apps send as model. Must be unique. Running state uses the stable ID, not this alias.'"
          />
          <router-link
            v-if="config.engine !== 'audio_cpp'"
            class="alias-link"
            to="/engines#ev-section-routing"
          >Virtual models</router-link>
        </label>
        <InputText
          v-model="config.model_alias"
          placeholder="e.g. my-app-model"
          class="w-full"
        />
        </div>
      </div>

      <div v-if="!isAudioEngine" class="advanced-block">
        <div class="section-label section-label--inline">
          Sub-ID variants
          <i class="pi pi-info-circle param-info" tabindex="0" aria-label="About sub-ID variants" v-tooltip.top="'Request-body parameters per sub-id, such as my-model:high. Each non-empty sub-id becomes a llama-swap alias.'" />
        </div>
        <div v-if="setParamsByIdVariants.length" class="set-params-variant-grid">
          <div
            v-for="(variant, vIdx) in setParamsByIdVariants"
            :key="variant._key"
            class="set-params-variant"
          >
            <div class="set-params-variant__header">
              <div class="set-params-variant__summary">
                <strong>{{ setParamsVariantModelId(variant) }}</strong>
              </div>
              <Button
                label="Edit"
                icon="pi pi-pencil"
                severity="secondary"
                text
                size="small"
                type="button"
                :aria-expanded="isSetParamsVariantEditing(variant) ? 'true' : 'false'"
                @click="toggleSetParamsVariantEditor(variant)"
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
          </div>
        </div>
        <div
          v-for="edit in editingSetParamsVariants"
          :key="`editor-${edit.variant._key}`"
          class="set-params-variant-editor"
        >
          <div class="set-params-variant-editor__header">
            <strong>Editing {{ setParamsVariantModelId(edit.variant) }}</strong>
            <Button
              label="Done"
              icon="pi pi-check"
              severity="secondary"
              text
              size="small"
              type="button"
              @click="toggleSetParamsVariantEditor(edit.variant)"
            />
          </div>
          <div class="set-params-target" role="radiogroup" aria-label="Variant target">
                <button
                  type="button"
                  class="set-params-target__option"
                  role="radio"
                  :class="{ selected: edit.variant.is_base }"
                  :aria-checked="edit.variant.is_base ? 'true' : 'false'"
                  :disabled="isSetParamsVariantTargetTaken(edit.variant, true)"
                  @click="setSetParamsVariantBase(edit.variant, true)"
                >
                  Primary ID
                </button>
                <button
                  type="button"
                  class="set-params-target__option"
                  role="radio"
                  :class="{ selected: !edit.variant.is_base }"
                  :aria-checked="!edit.variant.is_base ? 'true' : 'false'"
                  @click="setSetParamsVariantBase(edit.variant, false)"
                >
                  Sub-ID
                </button>
          </div>
          <InputText
            v-if="!edit.variant.is_base"
            :model-value="edit.variant.sub_id"
            placeholder="Sub-ID suffix (e.g. high)"
            class="set-params-sub-id"
            :class="{ 'set-params-sub-id--invalid': hasSetParamsVariantIdConflict(edit.variant) }"
            aria-label="Sub-ID suffix"
            @update:model-value="(v) => { edit.variant.sub_id = v; syncSetParamsByIdFromVariants() }"
          />
          <small v-if="hasSetParamsVariantIdConflict(edit.variant)" class="set-params-variant__error">
            {{ setParamsVariantModelId(edit.variant) }} is already configured.
          </small>
          <div class="section-label set-params-kwargs-label">chat_template_kwargs</div>
          <div
            v-for="(row, kIdx) in edit.variant.kwargsRows"
            :key="`${edit.variant._key}-kw-${kIdx}`"
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
              class="swap-env-value"
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
              @click="removeSetParamsKwargRow(edit.index, kIdx)"
            />
          </div>
          <Button
            label="Add kwarg"
            icon="pi pi-plus"
            severity="secondary"
            outlined
            type="button"
            class="mt-1"
            @click="addSetParamsKwargRow(edit.index)"
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
        <div class="advanced-block">
          <div class="section-label section-label--inline">
            Custom arguments
            <i class="pi pi-info-circle param-info" tabindex="0" aria-label="About custom arguments" v-tooltip.top="'Raw CLI flags appended to the server command.'" />
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
            <i class="pi pi-info-circle param-info" tabindex="0" aria-label="About environment variables" v-tooltip.top="'Variables passed to the process as llama-swap env. CUDA_VISIBLE_DEVICES is set from the GPU control when NVIDIA GPUs are detected. LLAMA_STUDIO_* keys are reserved and ignored.'" />
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
          <Select
            :model-value="item.row.mode || 'set'"
            :options="envModeOptions"
            option-label="label"
            option-value="value"
            class="swap-env-mode"
            aria-label="Environment variable mode"
            @update:model-value="(v) => { item.row.mode = v; syncSwapEnvFromRows() }"
          />
          <InputText
            v-if="(item.row.mode || 'set') !== 'unset'"
            :model-value="item.row.value"
            placeholder="value"
            class="swap-env-value"
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
        <div class="advanced-block">
          <div class="section-label section-label--inline">Command</div>
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
      </div>
      </div>

      <p v-if="persistenceNotice" class="persistence-notice" role="status">
        <strong>{{ persistenceNotice.summary }}.</strong>
        {{ persistenceNotice.detail }}
        <button v-if="persistenceNotice.retry" type="button" @click="retryPersistence">Try again</button>
        <button v-if="persistenceNotice.refresh" type="button" @click="refreshPersistedConfig">Refresh</button>
      </p>
      <!-- Actions -->
      <div class="config-actions">
        <Button
          label="Save Configuration"
          icon="pi pi-save"
          severity="success"
          :loading="saving"
          :disabled="configSaveHeld"
          @click="saveConfig"
        />
        <Button
          v-if="showApplyLlamaSwap"
          :label="applyLlamaSwapLabel"
          icon="pi pi-bolt"
          severity="warning"
          :loading="applyingLlamaSwap"
          :disabled="saving || applyingLlamaSwap || configSaveHeld"
          v-tooltip.top="applyLlamaSwapHint"
          @click="requestApply"
        />
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
        <p v-if="activeCmdPreview.revision" class="cmd-preview-dialog-hint">{{ activeCmdPreview.revision }}</p>
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
          <PersistenceAlert :notice="templateSaveNotice" @retry="saveConfigTemplate" @refresh="refreshTemplateSave" />
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
            <PersistenceAlert :notice="templateMutationNotice" @retry="retryTemplateMutation" @refresh="refreshTemplateMutation" />
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
                  <span v-if="tpl.description" class="config-templates-list-desc">{{ tpl.description }}</span>
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
          <Button label="Close" severity="secondary" outlined @click="templatesDialogVisible = false" />
        </template>
      </Dialog>

      <Dialog
        v-model:visible="leavePromptVisible"
        header="Unsaved configuration"
        modal
        :closable="false"
        class="dialog-width-sm"
      >
        <p>This model has edits that are not saved. Leaving now discards them unless you save first.</p>
        <template #footer>
          <Button label="Stay" severity="secondary" outlined @click="stayOnPage" />
          <Button label="Discard" severity="danger" outlined @click="discardAndLeave" />
          <Button label="Save" icon="pi pi-save" severity="success" :loading="saving" @click="saveAndLeave" />
        </template>
      </Dialog>

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
            :label="applyingLlamaSwap ? 'Stop apply' : 'Cancel'"
            severity="secondary"
            outlined
            @click="applyingLlamaSwap ? requestApplyStop() : cancelApplyImpact()"
          />
          <Button :label="applyWithheld ? 'Apply again' : applyLlamaSwapLabel" icon="pi pi-bolt" severity="warning" :loading="applyingLlamaSwap" @click="confirmApplyImpact" />
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
import { ref, computed, onMounted, onBeforeUnmount, watch, nextTick } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { clearDraft, readDraft, useDraftGuard, writeDraft } from '@/composables/useDraftGuard'
import { onRovingTabKeydown } from '@/composables/useRovingTabs'
import { watchDialogFocus } from '@/composables/useFocusReturn'
import { withheldConfirmation } from '@/composables/actionConfirmation'
import { classifyPersistenceError, noteDocumentSaveFailure, saveHeldForRefresh } from '@/composables/persistenceOutcome'
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
import ParseCommandDialog from '@/components/ParseCommandDialog.vue'
import {
  useAudioModelConfig,
  AUDIO_NESTED_SCOPE_KEYS,
  AUDIO_STUDIO_KNOWN_KEYS,
  coerceAudioParamValue,
  defaultValueForAudioParam,
  isBogusAudioConfigKey,
  pruneStaleAudioRequestDefaults,
} from '@/composables/useAudioModelConfig'
import { audioTabFromConfig } from '@/composables/useAudioInferenceClient'
import { useModelStore } from '@/stores/models'
import { useEnginesStore } from '@/stores/engines'
import { STUDIO_RESERVED_KEYS, studioEnvSkipReason } from '@/utils/parseCliCommand'

const route = useRoute()
const router = useRouter()
const toast = useToast()
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
const configSaveHeld = computed(() => saveHeldForRefresh(persistenceNotice.value))

async function refreshTemplateSave() {
  const ok = await fetchConfigTemplates()
  templateSaveNotice.value = ok
    ? null
    : classifyPersistenceError(new Error('reload failed'))
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
  templateMutationNotice.value = templatesOk && configOk
    ? null
    : classifyPersistenceError(new Error('reload failed'))
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
const savedConfig = ref({})          // for reset
const paramRegistry = ref({
  sections: [],
  scan_error: null,
  scan_pending: false,
  task_profile: null,
  request_field_groups: [],
  request_defaults_key: 'task_defaults',
  api_endpoint: '/audioapi/v1/tasks/run',
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
const modelLimits = ref(null)        // engine-agnostic: { max_context_length?, layer_count? } from config runtime_limits

const cmdPreviewText = ref('')
const cmdPreviewEnvText = ref('')
const cmdPreviewMacrosText = ref('')
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
const unsavedCmdPreviewMacrosText = ref('')
const unsavedCmdPreviewFiltersText = ref('')
const unsavedCmdPreviewAliasesText = ref('')
const unsavedCmdPreviewSidecarText = ref('')
const unsavedCmdPreviewSidecarPath = ref('')
const unsavedCmdPreviewLauncherText = ref('')
const unsavedCmdPreviewRevisionText = ref('')
const unsavedCmdPreviewError = ref(null)
const unsavedCmdPreviewLoading = ref(false)
/** Rows for llama-swap YAML `env` (synced into config.swap_env). */
const swapEnvRows = ref([{ key: '', value: '' }])
/** Sub-ID variants for llama-swap ``filters.setParamsByID`` (synced into config.set_params_by_id). */
const setParamsByIdVariants = ref([])
const editingSetParamsVariantKey = ref(null)
const editingSetParamsVariants = computed(() => {
  const index = setParamsByIdVariants.value.findIndex(
    (variant) => variant._key === editingSetParamsVariantKey.value,
  )
  return index < 0 ? [] : [{ variant: setParamsByIdVariants.value[index], index }]
})
let setParamsByIdVariantKeySeq = 0
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
const applyingLlamaSwap = ref(false)
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

const fallbackEngineOptions = [
  { value: 'llama_cpp', label: 'llama.cpp', icon: 'pi-microchip' },
  { value: 'ik_llama',  label: 'ik_llama.cpp', icon: 'pi-microchip' },
  { value: 'unsloth_llama', label: 'Unsloth llama.cpp', icon: 'pi-microchip' },
  { value: 'lmdeploy',  label: 'LMDeploy', icon: 'pi-server' },
  { value: '1cat_vllm', label: '1Cat-vLLM', icon: 'pi-server' },
  { value: 'sglang', label: 'SGLang', icon: 'pi-sparkles' },
  { value: 'sglang_v100', label: 'SGLang V100', icon: 'pi-sparkles' },
  { value: 'vllm', label: 'vLLM', icon: 'pi-server' },
  { value: 'audio_cpp', label: 'audio.cpp', icon: 'pi-volume-up' },
]
const curatedPackageKinds = ['prepared_bundle', 'builtin']
const inferredEnginesByFormat = {
  gguf: ['llama_cpp', 'ik_llama', 'unsloth_llama'],
  safetensors: ['lmdeploy', '1cat_vllm', 'vllm', 'sglang', 'sglang_v100'],
}

function inferredEnginesForModel(options, fmt, packageKind) {
  if (curatedPackageKinds.includes(packageKind)) return ['audio_cpp']
  const fromDescriptors = options
    .filter((option) => {
      const formats = option.descriptor?.artifact_formats
      return Array.isArray(formats)
        && formats.length === 1
        && String(formats[0]).toLowerCase() === fmt
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
      disabledReason: option.descriptor?.enabled === false
        ? 'Disabled by the AUDIO_CPP_ENABLED feature gate'
        : compatible
          ? ''
          : `Not compatible with this ${packageKind || fmt || 'model'} artifact`,
    }
  })
})

const isAudioEngine = computed(() => config.value.engine === 'audio_cpp')
const pageTab = ref('runtime')
const pageTabs = computed(() => (
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
      ]
))
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
  const env = config.value.swap_env
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
const mmprojOptions = computed(() => companionDropdownOptions(companionFiles.value.mmproj_files, 'mmproj'))
const mtpOptions = computed(() => companionDropdownOptions(companionFiles.value.mtp_files, 'mtp'))
const dflashOptions = computed(() => companionDropdownOptions(companionFiles.value.dflash_files, 'dflash'))
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
    if (BASIC_PARAM_KEYS.includes(key)) continue
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

const setParamsVariantBaseId = computed(() => {
  const alias = String(config.value.model_alias || '').trim()
  return alias || llamaSwapStableId.value || model.value?.id || 'model'
})

const audioModelConfig = useAudioModelConfig(
  config,
  paramRegistry,
  enginesStore,
  llamaSwapStableId,
)

const {
  audioEditableParams,
  audioParamValue,
  ensureTtsConfigShape,
  seedSessionVoiceFromPackagedIds,
} = audioModelConfig

const currentEngineSection = computed(() => (
  (config.value.engines && config.value.engines[config.value.engine]) || {}
))

const AUDIO_STUDIO_KNOWN_KEYS_REF = AUDIO_STUDIO_KNOWN_KEYS

const unrecognizedSavedKeys = computed(() => {
  const known = new Set([
    'custom_args',
    'model_alias',
    'set_params_by_id',
    'swap_env',
    'swap_env_unset',
    'gpu_mode',
    'gpu_devices',
    'load_options',
    'session_options',
    ...(isAudioEngine.value ? AUDIO_STUDIO_KNOWN_KEYS_REF : []),
  ])
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

const { leavePromptVisible, finishLeave } = useDraftGuard(
  () => hasUnsavedChanges.value && !loading.value,
)
watchDialogFocus(applyImpactVisible)
watchDialogFocus(leavePromptVisible)
watchDialogFocus(templatesDialogVisible)

/**
 * Saved model config is out of sync with llama-swap-config.yaml (server stale flag).
 * Shown only when there are no unsaved edits — save first, then Apply.
 */
const showApplyLlamaSwap = computed(() => {
  if (hasUnsavedChanges.value) return false
  const pending = enginesStore.swapConfigPending
  if (pending?.launch_manifests && Array.isArray(pending.models)) {
    const action = modelLaunchPlan.value?.action
    return Boolean(action && action !== 'none')
  }
  return Boolean(
    enginesStore.swapConfigStale?.applicable &&
      enginesStore.swapConfigStale?.stale
  )
})

const gpuModeOptions = [
  { label: 'All GPUs', value: 'inherit' },
  { label: 'Choose GPUs', value: 'selected' },
  { label: 'CPU only', value: 'cpu' },
]

const envModeOptions = [
  { label: 'Set value', value: 'set' },
  { label: 'Unset', value: 'unset' },
]

const modelLaunchPlan = computed(() => {
  const rows = enginesStore.swapConfigPending?.models
  if (!Array.isArray(rows)) return null
  const catalogId = String(route.params.id || '')
  return rows.find((row) => row.catalog_id === catalogId || row.model_id === catalogId) || null
})

const selectiveModelApply = computed(() => {
  const pending = enginesStore.swapConfigPending
  const entry = modelLaunchPlan.value
  return Boolean(
    pending?.launch_manifests &&
      entry &&
      !pending.requires_proxy_reload &&
      !entry.requires_proxy_reload &&
      (entry.action === 'restart_now' || entry.action === 'publish_next_start'),
  )
})

const applyLlamaSwapLabel = computed(() => {
  if (!selectiveModelApply.value) return 'Reload proxy'
  return modelLaunchPlan.value?.running ? 'Restart this model' : 'Use on next start'
})

const pendingApply = computed(() => Boolean(showApplyLlamaSwap.value))

const modelRuntimeQuality = computed(() => String(model.value?.runtime_quality || '').toLowerCase())

const modelIsRunning = computed(() => {
  if (modelRuntimeQuality.value === 'unreachable') return false
  const plan = modelLaunchPlan.value
  if (plan && typeof plan.running === 'boolean') return plan.running
  return Boolean(model.value?.is_active)
})

const publishedStepLabel = computed(() => {
  if (modelRuntimeQuality.value === 'unreachable') return 'Status unknown'
  if (modelRuntimeQuality.value === 'stale') {
    return model.value?.is_active ? 'Running · stale' : 'Status stale'
  }
  return modelIsRunning.value ? 'In use' : 'Published'
})

const runtimeStateDetail = computed(() => {
  if (hasUnsavedChanges.value) return 'These edits are only in this form. Save them before they can be applied.'
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
  if (modelRuntimeQuality.value === 'unreachable') {
    return 'Saved settings are on this page, but the proxy could not be reached. This is not a verified stopped or running state.'
  }
  if (modelRuntimeQuality.value === 'stale') {
    return model.value?.is_active
      ? 'The proxy is unreachable. The last observation still had this model running.'
      : 'The proxy is unreachable. The last observation did not show this model running.'
  }
  if (modelIsRunning.value) return 'These saved settings are published and this model is running them.'
  return 'These settings are published. The model is stopped, so they are used the next time it starts.'
})

watch(
  () => enginesStore.swapConfigStale?.stale,
  (stale, previous) => {
    if (stale && stale !== previous) void enginesStore.fetchSwapConfigPending?.()
  },
)

const applyImpactMessage = computed(() => {
  if (!selectiveModelApply.value) {
    return 'Reload the proxy. Every loaded model stops, then pending saved settings are published.'
  }
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
  if (!selectiveModelApply.value) return 'Affected: all loaded models.'
  return `Affected: ${name}.`
})

const basicParams = computed(() => (
  BASIC_PARAM_KEYS
    .map((key) => catalogParamByKey.value.get(key))
    .filter(Boolean)
))

function engineIsInstalled(option) {
  const count = option?.descriptor?.installed_versions
  if (count == null) return true
  return Number(count) > 0
}

const visibleEngineOptions = computed(() => (
  engineOptions.value.filter((option) => !option.disabled && engineIsInstalled(option))
))
const unavailableSavedEngine = computed(() => {
  const selected = engineOptions.value.find((option) => option.value === config.value.engine)
  if (!selected) return null
  if (visibleEngineOptions.value.some((option) => option.value === selected.value)) return null
  return selected
})

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
  const index = Math.max(0, options.findIndex((option) => option.value === config.value.engine))
  let next
  if (event.key === 'ArrowRight' || event.key === 'ArrowDown') next = (index + 1) % options.length
  else if (event.key === 'ArrowLeft' || event.key === 'ArrowUp') next = (index - 1 + options.length) % options.length
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

const applyLlamaSwapHint = computed(() => {
  if (!selectiveModelApply.value) {
    return 'Regenerate llama-swap-config.yaml and reload the proxy. Reload proxy — affects all loaded models.'
  }
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
  if (value === null) return true
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

function previewEnvLines(data) {
  if (Array.isArray(data?.engine_env) && data.engine_env.length) {
    const unset = Array.isArray(data.engine_env_unset) && data.engine_env_unset.length
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
    macros: unsavedCmdPreviewMacrosText.value,
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
  const configuredIds = new Set()
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
    const subId = (variant.sub_id || '').trim()
    if (!variant.is_base && !subId) continue
    const modelId = setParamsVariantModelId(variant)
    if (configuredIds.has(modelId)) continue
    configuredIds.add(modelId)
    out.push({
      sub_id: variant.is_base ? '' : subId,
      params: { chat_template_kwargs: kwargs },
    })
  }
  const next = out.length ? out : []
  const current = Array.isArray(config.value.set_params_by_id) ? config.value.set_params_by_id : []
  if (_setParamsByIdEqual(next, current)) return
  config.value.set_params_by_id = next
}

function setParamsVariantModelId(variant) {
  const baseId = setParamsVariantBaseId.value
  if (variant?.is_base) return baseId
  const subId = String(variant?.sub_id || '').trim()
  return subId ? `${baseId}:${subId}` : `${baseId}:…`
}

function hasSetParamsVariantIdConflict(variant) {
  const id = setParamsVariantModelId(variant)
  if (id.endsWith(':…')) return false
  return setParamsByIdVariants.value.some(
    (candidate) => candidate !== variant && setParamsVariantModelId(candidate) === id,
  )
}

function isSetParamsVariantTargetTaken(variant, isBase) {
  if (!isBase) return false
  return setParamsByIdVariants.value.some(
    (candidate) => candidate !== variant && candidate.is_base,
  )
}

function initSetParamsByIdFromConfig() {
  const raw = config.value.set_params_by_id
  editingSetParamsVariantKey.value = null
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
      is_base: !String(item?.sub_id || '').trim(),
      kwargsRows: kwargsRows.length ? kwargsRows : [{ key: '', value: '' }],
    }
  })
}

function addSetParamsByIdVariant() {
  const variant = {
    _key: _nextSetParamsVariantKey(),
    sub_id: '',
    is_base: false,
    kwargsRows: [{ key: '', value: '' }],
  }
  setParamsByIdVariants.value = [
    ...setParamsByIdVariants.value,
    variant,
  ]
  editingSetParamsVariantKey.value = variant._key
  syncSetParamsByIdFromVariants()
}

function isSetParamsVariantEditing(variant) {
  return editingSetParamsVariantKey.value === variant._key
}

function toggleSetParamsVariantEditor(variant) {
  editingSetParamsVariantKey.value = isSetParamsVariantEditing(variant)
    ? null
    : variant._key
}

function setSetParamsVariantBase(variant, isBase) {
  variant.is_base = isBase
  syncSetParamsByIdFromVariants()
}

function removeSetParamsByIdVariant(idx) {
  const removed = setParamsByIdVariants.value[idx]
  setParamsByIdVariants.value = setParamsByIdVariants.value.filter((_, i) => i !== idx)
  if (removed?._key === editingSetParamsVariantKey.value) {
    editingSetParamsVariantKey.value = null
  }
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

function _swapEnvUnsetEqual(a, b) {
  const x = Array.isArray(a) ? a.map((name) => String(name)) : []
  const y = Array.isArray(b) ? b.map((name) => String(name)) : []
  if (x.length !== y.length) return false
  return x.every((name, index) => name === y[index])
}

function syncSwapEnvFromRows() {
  const out = {}
  const unset = []
  for (const row of swapEnvRows.value) {
    const k = (row.key || '').trim()
    if (!k) continue
    if (row.mode === 'unset') {
      unset.push(k)
      continue
    }
    out[k] = row.value != null ? String(row.value) : ''
  }
  if (
    _swapEnvShallowEqual(config.value.swap_env, out)
    && _swapEnvUnsetEqual(config.value.swap_env_unset, unset)
  ) {
    return
  }
  config.value.swap_env = out
  config.value.swap_env_unset = unset
}

function initSwapEnvRowsFromConfig() {
  const rows = []
  const o = config.value.swap_env
  if (o && typeof o === 'object' && !Array.isArray(o)) {
    for (const [key, value] of Object.entries(o)) {
      rows.push({
        key,
        value: value != null ? String(value) : '',
        mode: 'set',
      })
    }
  }
  const unset = Array.isArray(config.value.swap_env_unset) ? config.value.swap_env_unset : []
  for (const name of unset) {
    rows.push({ key: String(name), value: '', mode: 'unset' })
  }
  swapEnvRows.value = rows.length ? rows : [{ key: '', value: '', mode: 'set' }]
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
    removeSwapEnvRowByCanonicalKey('CUDA_VISIBLE_DEVICES')
    const unset = new Set(config.value.swap_env_unset || [])
    unset.add('CUDA_VISIBLE_DEVICES')
    config.value.swap_env_unset = [...unset]
    initSwapEnvRowsFromConfig()
    return
  }
  config.value.swap_env_unset = (config.value.swap_env_unset || []).filter(
    (name) => String(name).toUpperCase() !== 'CUDA_VISIBLE_DEVICES',
  )
  if (mode === 'inherit') {
    config.value.gpu_devices = []
    removeSwapEnvRowByCanonicalKey('CUDA_VISIBLE_DEVICES')
    return
  }
  applyNvidiaCudaSelection(cudaVisibleDeviceSelection.value)
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
  if ((config.value.gpu_mode || 'inherit') !== 'selected') return
  const allVals = nvidiaGpuSelectOptions.value.map((o) => o.value)
  const sel = []
  for (const value of selected || []) {
    if (allVals.includes(value) && !sel.includes(value)) sel.push(value)
  }
  config.value.gpu_devices = [...sel]
  if (sel.length === 0) {
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
  if (d && typeof d === 'object' && typeof d.message === 'string') return d.message
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
    const family = draftFamily ?? (engine === 'audio_cpp' ? config.value?.family : undefined)
    const task = draftTask ?? (engine === 'audio_cpp' ? config.value?.task : undefined)
    const { data } = await axios.get('/api/models/param-registry', {
      params: {
        engine,
        ...(model.value?.id ? { model_id: model.value.id } : {}),
        ...(engine === 'audio_cpp' && family ? { family } : {}),
        ...(engine === 'audio_cpp' && task ? { task } : {}),
        ...(rescan ? { rescan: true } : {}),
      },
    })
    paramRegistry.value = {
      sections: data.sections || [],
      scan_error: data.scan_error ?? null,
      scan_pending: Boolean(data.scan_pending),
      profile_fingerprint: data.profile_fingerprint ?? null,
      inspection: data.inspection ?? null,
      compatibility_warnings: data.compatibility_warnings || [],
      task_profile: data.task_profile ?? null,
      request_field_groups: data.request_field_groups || [],
      request_defaults_key: data.request_defaults_key || 'task_defaults',
      api_endpoint: data.api_endpoint || '/audioapi/v1/tasks/run',
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
      contract_grade: data.contract_grade || '',
      contract_warnings: data.contract_warnings || [],
      last_reviewed_fingerprint: data.last_reviewed_fingerprint || '',
      sidecar_session_fields: data.sidecar_session_fields || [],
      family_dependencies: data.family_dependencies || {},
      packaged_voices: data.packaged_voices || [],
    }
    if (engine === 'audio_cpp') {
      pruneStaleAudioRequestDefaults(
        config.value,
        paramRegistry.value.request_defaults_key,
      )
      seedSessionVoiceFromPackagedIds()
    }
  } catch (e) {
    console.error('Failed to fetch param registry:', e)
    paramRegistry.value = {
      sections: [],
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
    if (config.value?.engine !== 'audio_cpp' || loading.value) return
    void fetchParamRegistry('audio_cpp')
  }, 250)
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
  const preserveUnknownAudioKeys = sourceConfig.engine === 'audio_cpp'
  const previous = sourceConfig.engines?.[sourceConfig.engine]
  const stash = preserveUnknownAudioKeys && previous && typeof previous === 'object'
    ? JSON.parse(JSON.stringify(previous))
    : {}
  delete stash.model_alias
  delete stash.custom_args
  delete stash.set_params_by_id
  delete stash.request_options
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
      if (v == null) continue
      if (typeof v === 'number' && Number.isNaN(v)) continue
      cleaned[name] = typeof v === 'string' ? v : String(v)
    }
    stash.swap_env = cleaned
  } else {
    stash.swap_env = {}
  }
  if (Array.isArray(sourceConfig.swap_env_unset) && sourceConfig.swap_env_unset.length) {
    stash.swap_env_unset = sourceConfig.swap_env_unset.map((name) => String(name)).filter(Boolean)
  }
  if (sourceConfig.gpu_mode) stash.gpu_mode = sourceConfig.gpu_mode
  if (Array.isArray(sourceConfig.gpu_devices)) {
    stash.gpu_devices = sourceConfig.gpu_devices.map((item) => String(item))
  }
  const spid = sourceConfig.set_params_by_id
  if (Array.isArray(spid) && spid.length) {
    stash.set_params_by_id = JSON.parse(JSON.stringify(spid))
  }
  if (preserveUnknownAudioKeys) {
    for (const param of audioEditableParams.value) {
      const nestedKey = AUDIO_NESTED_SCOPE_KEYS[param.scope]
      const value = audioParamValue(param, sourceConfig)
      const empty = value == null
        || value === ''
        || (typeof value === 'number' && Number.isNaN(value))
        || (Array.isArray(value) && value.length === 0)
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
        stash[param.key] = Array.isArray(value)
          ? [...value]
          : coerceAudioParamValue(param, value)
      }
    }
    for (const nestedKey of ['load_options', 'session_options']) {
      if (
        stash[nestedKey]
        && typeof stash[nestedKey] === 'object'
        && !Object.keys(stash[nestedKey]).length
      ) {
        delete stash[nestedKey]
      }
    }
    const presets = sourceConfig.voice_presets
    if (presets && typeof presets === 'object' && !Array.isArray(presets) && Object.keys(presets).length) {
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
      speechDefaults
      && typeof speechDefaults === 'object'
      && !Array.isArray(speechDefaults)
      && Object.keys(speechDefaults).length
    ) {
      stash.speech_defaults = JSON.parse(JSON.stringify(speechDefaults))
    } else {
      delete stash.speech_defaults
    }
    const transcriptionDefaults = sourceConfig.transcription_defaults
    if (
      transcriptionDefaults
      && typeof transcriptionDefaults === 'object'
      && !Array.isArray(transcriptionDefaults)
      && Object.keys(transcriptionDefaults).length
    ) {
      stash.transcription_defaults = JSON.parse(JSON.stringify(transcriptionDefaults))
    } else {
      delete stash.transcription_defaults
    }
    const taskDefaults = sourceConfig.task_defaults
    if (
      taskDefaults
      && typeof taskDefaults === 'object'
      && !Array.isArray(taskDefaults)
      && Object.keys(taskDefaults).length
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
    for (const key of Object.keys(stash)) {
      if (isBogusAudioConfigKey(key)) delete stash[key]
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
  if (engine === 'audio_cpp') {
    const eng = config.value.engine
    const engMap = config.value.engines
    for (const key of Object.keys(config.value)) {
      if (key !== 'engine' && key !== 'engines') delete config.value[key]
    }
    Object.assign(config.value, JSON.parse(JSON.stringify(sec)))
    config.value.engine = eng
    config.value.engines = engMap
    for (const key of Object.keys(config.value)) {
      if (key !== 'engine' && key !== 'engines' && isBogusAudioConfigKey(key)) {
        delete config.value[key]
      }
    }
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
    initSwapEnvRowsFromConfig()
    initSetParamsByIdFromConfig()
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
    'swap_env_unset',
    'gpu_mode',
    'gpu_devices',
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
  config.value.swap_env_unset = Array.isArray(sec.swap_env_unset)
    ? sec.swap_env_unset.map((name) => String(name)).filter(Boolean)
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
  config.value.set_params_by_id = Array.isArray(sec.set_params_by_id)
    ? JSON.parse(JSON.stringify(sec.set_params_by_id))
    : []
  initSetParamsByIdFromConfig()
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
    if (!found) { loading.value = false; return }
    model.value = found
    syncCompanionSelectionsFromModel()

    const cfgResp = await axios.get(modelApiUrl('/config'))
    const cfg = cfgResp.data
    let engine = cfg.engine ?? found.engine ?? 'llama_cpp'
    if (found.format !== 'safetensors' && ['lmdeploy', '1cat_vllm', 'vllm', 'sglang', 'sglang_v100'].includes(engine)) {
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
      enginesStore.fetchSwapConfigPending?.() ?? Promise.resolve(),
    ])

    const sec = (merged.engines && merged.engines[engine]) || {}
    setActiveKeysFromSection(sec, catalogParamList.value)
    applyEngineSectionToForm(engine)
    savedConfig.value = JSON.parse(JSON.stringify(config.value))
    lookForDraft()
  } catch (e) {
    loadError.value = formatAxiosDetail(e) || 'Could not load configuration'
    toast.add({ severity: 'error', summary: 'Failed to load config', detail: loadError.value, life: 4000 })
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
    } else if (param?.value_kind === 'repeatable') {
      config.value[item.key] = Array.isArray(item.value) ? [...item.value] : item.value == null ? [] : [item.value]
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
    upsertSwapEnvRowCanonical(item.key, value)
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
      enginesStore.markSwapConfigStaleLocal()
      void enginesStore.fetchSwapConfigStale()
      void enginesStore.fetchSwapConfigPending?.()
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
              detail: 'The document was replaced, but it could not be reloaded. Your edits are still here. Refresh before trying again.',
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
      detail: 'The saved configuration could not be reloaded. Your edits are still here. Refresh before trying again.',
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
      detail: 'The change was stored, but acknowledgement failed. Refresh before trying the action again.',
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
  saving.value = true
  const draft = JSON.parse(JSON.stringify(config.value))
  try {
    const payload = buildPersistedPayload(config.value)
    const { data } = await axios.put(modelApiUrl('/config'), payload)
    applyLoadedConfig(data)
    enginesStore.markSwapConfigStaleLocal()
    void enginesStore.fetchSwapConfigStale()
    void enginesStore.fetchSwapConfigPending?.()
    refreshSavedCmdPreviewIfVisible()
    persistenceNotice.value = null
    toast.add({
      severity: 'success',
      summary: 'Saved',
      detail: 'Saved. These settings stay pending until you apply them.',
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
            detail: 'The document was replaced, but it could not be reloaded. Your edits are still here. Refresh before trying again.',
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
    await enginesStore.fetchSwapConfigPending()
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
  const finished = await applyLlamaSwapFromModelConfig()
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

async function applyLlamaSwapFromModelConfig() {
  applyingLlamaSwap.value = true
  applyStopRequested.value = false
  try {
    await enginesStore.fetchSwapConfigPending()
    if (selectiveModelApply.value) {
      const entry = modelLaunchPlan.value
      const mode = entry.running ? 'restart_now' : 'next_start'
      const confirmation = applyWithheld.value || {}
      const started = await axios.post(modelApiUrl('/runtime/apply'), {
        mode,
        expected_desired_revision: entry.desired_revision,
        expected_published_revision: entry.published_revision,
        idempotency_key: `${enginesStore.swapConfigPending?.plan_id || 'plan'}:${entry.model_id}:${mode}`,
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
      await enginesStore.fetchSwapConfigPending()
      await enginesStore.fetchSwapConfigStale()
      await modelStore.fetchModels()
      const refreshedModel = findModelById(route.params.id)
      if (refreshedModel) model.value = refreshedModel
      refreshSavedCmdPreviewIfVisible()
      return true
    }
    const confirmation = applyWithheld.value
      ? {
          confirm_operation_id: applyWithheld.value.confirm_operation_id,
          confirm_state: applyWithheld.value.confirm_state,
        }
      : null
    await enginesStore.applySwapConfig(confirmation)
    toast.add({
      severity: 'success',
      summary: 'llama-swap applied',
      detail: 'llama-swap-config.yaml was regenerated and the proxy reloaded.',
      life: 4000,
    })
    refreshSavedCmdPreviewIfVisible()
    return true
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
    applyingLlamaSwap.value = false
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
  toast.add({ severity: 'info', summary: 'Reset', detail: 'Config reset to saved values', life: 2000 })
}

async function fetchSavedCmdPreview() {
  if (!model.value) return
  cmdPreviewLoading.value = true
  cmdPreviewError.value = null
  try {
    const { data } = await axios.get(modelApiUrl('/saved-llama-swap-cmd'))
    if (data?.ok && (data.engine_command || data.cmd)) {
      cmdPreviewText.value = data.engine_command || data.cmd
      cmdPreviewLauncherText.value = data.launcher_command || ''
      cmdPreviewRevisionText.value = formatRevisionPreview(data)
      cmdPreviewEnvText.value = previewEnvLines(data)
      cmdPreviewMacrosText.value = formatMacrosPreview(data.macros)
      cmdPreviewFiltersText.value = formatFiltersPreview(data.filters)
      cmdPreviewAliasesText.value = formatAliasesPreview(data.aliases)
      cmdPreviewSidecarText.value = formatSidecarPreview(data.sidecar)
      cmdPreviewSidecarPath.value = data.sidecar_path || ''
      cmdPreviewError.value = null
    } else {
      cmdPreviewText.value = ''
      cmdPreviewEnvText.value = ''
      cmdPreviewMacrosText.value = ''
      cmdPreviewFiltersText.value = ''
      cmdPreviewAliasesText.value = ''
      cmdPreviewSidecarText.value = ''
      cmdPreviewSidecarPath.value = ''
      cmdPreviewError.value = data?.error || 'Could not build saved command.'
    }
  } catch (e) {
    cmdPreviewText.value = ''
    cmdPreviewEnvText.value = ''
    cmdPreviewMacrosText.value = ''
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
    const { data } = await axios.post(modelApiUrl('/preview-llama-swap-cmd'), payload, {
      signal,
    })
    if (requestId !== unsavedPreviewRequestId) return
    if (data?.ok && (data.engine_command || data.cmd)) {
      unsavedCmdPreviewText.value = data.engine_command || data.cmd
      unsavedCmdPreviewLauncherText.value = data.launcher_command || ''
      unsavedCmdPreviewRevisionText.value = formatRevisionPreview(data)
      unsavedCmdPreviewEnvText.value = previewEnvLines(data)
      unsavedCmdPreviewMacrosText.value = formatMacrosPreview(data.macros)
      unsavedCmdPreviewFiltersText.value = formatFiltersPreview(data.filters)
      unsavedCmdPreviewAliasesText.value = formatAliasesPreview(data.aliases)
      unsavedCmdPreviewSidecarText.value = formatSidecarPreview(data.sidecar)
      unsavedCmdPreviewSidecarPath.value = data.sidecar_path || ''
      unsavedCmdPreviewError.value = null
    } else {
      unsavedCmdPreviewText.value = ''
      unsavedCmdPreviewEnvText.value = ''
      unsavedCmdPreviewMacrosText.value = ''
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
    unsavedCmdPreviewMacrosText.value = ''
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
  [
    () => config.value?.engine,
    () => config.value?.family,
    () => config.value?.task,
  ],
  ([engine], [prevEngine]) => {
    if (suppressAudioRegistryWatch || loading.value) return
    if (engine !== 'audio_cpp') return
    if (prevEngine && prevEngine !== 'audio_cpp') {
      void fetchParamRegistry('audio_cpp')
      return
    }
    scheduleAudioRegistryRefresh()
  },
)

watch(config, () => { persistDraftNow() }, { deep: true })
watch(activeParamKeys, () => { persistDraftNow() }, { deep: true })

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
  flex-wrap: wrap;
  align-items: center;
  gap: 0.5rem;
  margin-bottom: 0.5rem;
}

.swap-env-row :deep(.swap-env-key) {
  flex: 0 0 11rem;
  width: 11rem;
  max-width: 100%;
  min-width: 0;
}

.swap-env-row :deep(.swap-env-mode) {
  flex: 0 0 10.75rem;
  width: 10.75rem;
  max-width: 100%;
  min-width: 0;
}

.swap-env-row :deep(.swap-env-value) {
  flex: 1 1 12rem;
  width: auto;
  min-width: 8rem;
}

.set-params-variant-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  align-items: start;
  gap: 0.65rem;
  margin-top: 0.65rem;
}

.set-params-variant {
  display: flex;
  flex-direction: column;
  min-width: 0;
  padding: 0.7rem;
  border: 1px solid var(--surface-border, #374151);
  border-radius: var(--radius-md);
  background: var(--bg-surface);
}

.set-params-variant__header {
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.set-params-variant__summary {
  display: flex;
  flex: 1;
  flex-direction: column;
  min-width: 0;
  gap: 0.1rem;
}

.set-params-variant__summary strong {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  font-size: 0.875rem;
}

.set-params-variant-editor {
  margin-top: 0.65rem;
  padding: 0.75rem;
  border: 1px solid rgba(34, 211, 238, 0.38);
  border-radius: var(--radius-md);
  background: rgba(34, 211, 238, 0.04);
}

.set-params-variant-editor__header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 0.5rem;
}

.set-params-sub-id {
  width: min(100%, 30rem);
  margin-top: 0.5rem;
  min-width: 0;
}

.set-params-target {
  display: flex;
  gap: 0.35rem;
  margin-top: 0.6rem;
}

.set-params-target__option {
  padding: 0.22rem 0.55rem;
  border: 1px solid var(--border-primary);
  border-radius: var(--radius-sm);
  background: transparent;
  color: var(--text-secondary);
  font: inherit;
  font-size: 0.75rem;
  font-weight: 600;
  cursor: pointer;
}

.set-params-target__option.selected {
  border-color: var(--accent-cyan);
  background: rgba(34, 211, 238, 0.1);
  color: var(--accent-cyan);
}

.set-params-target__option:disabled {
  cursor: not-allowed;
  opacity: 0.5;
}

.set-params-sub-id--invalid :deep(input) {
  border-color: var(--accent-red, #ef4444);
}

.set-params-variant__error {
  display: block;
  margin-top: 0.3rem;
  color: var(--accent-red, #ef4444);
  font-size: 0.75rem;
}

.set-params-kwargs-label {
  font-size: 0.85rem;
  margin-top: 0.6rem;
  margin-bottom: 0.35rem;
}

@media (max-width: 64rem) {
  .set-params-variant-grid {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
}

@media (max-width: 40rem) {
  .set-params-variant-grid {
    grid-template-columns: 1fr;
  }

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
  font-size: 0.75rem;
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
  font-size: 0.75rem;
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
  transition: border-color 0.15s, background 0.15s;
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
  font-size: 0.75rem;
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
  font-size: 0.72rem;
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

.param-input { width: 100%; }

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
  font-size: 0.78rem;
  line-height: 1.25;
}

.request-cap-item {
  display: contents;
}

.request-cap-item__key {
  color: var(--text-secondary, #9ca3af);
  font-size: 0.74rem;
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
  font-size: 0.78rem;
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
  font-size: 0.7rem;
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
  bottom: 0;
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
  font-size: 0.75rem;
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
  font-size: 0.75rem;
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
  font-size: 0.75rem;
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
</style>
