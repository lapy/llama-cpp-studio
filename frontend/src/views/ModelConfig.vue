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
            :class="{
              selected: config.engine === eng.value,
              disabled: eng.disabled,
            }"
            :aria-disabled="eng.disabled ? 'true' : 'false'"
            v-tooltip.bottom="eng.disabledReason || (eng.runnable ? '' : 'Engine is not installed or active')"
            @click="!eng.disabled && changeEngine(eng.value)"
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
        {{ isAudioEngine
          ? 'Unrecognized audio.cpp keys are preserved for compatibility:'
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

      <template v-if="isAudioEngine">
        <div v-if="audioInspectionSummary.length" class="config-card">
          <div class="section-label">
            Inspected package capabilities
            <small class="section-hint">Read from the installed bundle by the active audio.cpp CLI</small>
          </div>
          <div class="audio-capability-tags">
            <Tag
              v-for="item in audioInspectionSummary"
              :key="item"
              :value="item"
              severity="info"
            />
          </div>
        </div>

        <div v-if="isProfiledAudioModel" class="config-card">
          <div class="section-label">
            {{ requestDefaultsSectionTitle }}
            <small class="section-hint">
              Configure reusable request defaults for
              <code>{{ apiEndpoint }}</code>.
            </small>
          </div>

          <p v-if="taskProfile?.summary" class="config-muted-hint tts-profile-summary">
            {{ taskProfile.summary }}
          </p>
          <div v-if="taskWorkflowTags.length" class="audio-capability-tags">
            <Tag
              v-for="workflow in taskWorkflowTags"
              :key="workflow"
              :value="workflow"
              severity="secondary"
            />
          </div>
          <p v-if="taskProfile?.api_hint" class="config-muted-hint">{{ taskProfile.api_hint }}</p>

          <div v-if="supportsVoicePresets" class="tts-subsection">
            <div class="tts-subsection__head">
              <span class="tts-subsection__title">Voice presets</span>
              <Button
                label="Add preset"
                icon="pi pi-plus"
                size="small"
                severity="secondary"
                outlined
                type="button"
                @click="addVoicePreset"
              />
            </div>
            <p class="config-muted-hint">
              Named presets are written to the audio.cpp sidecar. Clients can pass
              <code>"voice": "preset-name"</code> or rely on the default preset.
            </p>
            <div v-if="!voicePresetRows.length" class="config-muted-hint">
              No voice presets yet.
            </div>
            <div v-for="row in voicePresetRows" :key="row.name" class="voice-preset-card">
              <div class="voice-preset-card__head">
                <InputText
                  :model-value="row.name"
                  class="voice-preset-card__name"
                  placeholder="preset-name"
                  @update:model-value="(value) => renameVoicePreset(row.name, value)"
                />
                <Button
                  icon="pi pi-trash"
                  severity="danger"
                  text
                  rounded
                  type="button"
                  aria-label="Remove preset"
                  @click="removeVoicePreset(row.name)"
                />
              </div>
              <div class="voice-preset-card__grid">
                <div
                  v-for="field in voicePresetFieldDefs"
                  :key="`${row.name}-${field.key}`"
                  class="param-field"
                >
                  <label class="param-field__label">{{ field.label }}</label>
                  <Textarea
                    v-if="field.type === 'textarea'"
                    :model-value="row.preset[field.key] || ''"
                    :placeholder="field.placeholder || ''"
                    rows="2"
                    class="w-full textarea-cli param-input"
                    @update:model-value="(value) => setVoicePresetField(row.name, field.key, value)"
                  />
                  <InputText
                    v-else
                    :model-value="row.preset[field.key] || ''"
                    :placeholder="field.placeholder || ''"
                    class="param-input"
                    @update:model-value="(value) => setVoicePresetField(row.name, field.key, value)"
                  />
                </div>
              </div>
            </div>
            <div class="param-field">
              <label class="param-field__label">Default voice preset</label>
              <Dropdown
                :model-value="defaultVoicePresetSelection"
                :options="defaultVoicePresetOptions"
                optionLabel="label"
                optionValue="value"
                placeholder="Inline default or choose a named preset"
                showClear
                class="param-input"
                @update:model-value="setDefaultVoicePresetSelection"
              />
            </div>
          </div>

          <div v-if="requestFieldGroups.length" class="tts-subsection">
            <div class="tts-subsection__title">Request defaults</div>
            <p class="config-muted-hint">
              Saved as Studio guidance and used to pre-fill the API example below. Override per request in
              <code>{{ apiEndpoint }}</code>.
            </p>
            <div
              v-for="group in requestFieldGroups"
              :key="group.id"
              class="tts-speech-group"
            >
              <div class="tts-speech-group__label">{{ group.label }}</div>
              <p v-if="group.description" class="config-muted-hint">{{ group.description }}</p>
              <div class="params-grid section-params">
                <div
                  v-for="field in group.fields"
                  :key="`${group.id}-${field.key}`"
                  class="param-field"
                >
                  <label class="param-field__label">{{ field.label }}</label>
                  <InputSwitch
                    v-if="field.type === 'bool'"
                    :model-value="Boolean(requestDefaultValue(field))"
                    @update:model-value="(value) => setRequestDefaultValue(field, value)"
                  />
                  <InputNumber
                    v-else-if="field.type === 'int' || field.type === 'float'"
                    :model-value="requestDefaultValue(field)"
                    :minFractionDigits="field.type === 'float' ? 1 : 0"
                    :maxFractionDigits="field.type === 'float' ? 6 : 0"
                    class="param-input"
                    @update:model-value="(value) => setRequestDefaultValue(field, value)"
                  />
                  <Textarea
                    v-else-if="field.type === 'textarea'"
                    :model-value="requestDefaultValue(field) || ''"
                    :placeholder="field.placeholder || ''"
                    rows="2"
                    class="w-full textarea-cli param-input"
                    @update:model-value="(value) => setRequestDefaultValue(field, value)"
                  />
                  <InputText
                    v-else
                    :model-value="requestDefaultValue(field) || ''"
                    :placeholder="field.placeholder || ''"
                    class="param-input"
                    @update:model-value="(value) => setRequestDefaultValue(field, value)"
                  />
                </div>
              </div>
            </div>
          </div>

          <div class="tts-subsection">
            <div class="tts-subsection__title">API example</div>
            <p v-if="apiExampleHint" class="config-muted-hint">{{ apiExampleHint }}</p>
            <Textarea
              :model-value="requestApiExample"
              readonly
              rows="12"
              class="w-full textarea-cli cmd-preview-textarea"
              autoResize
            />
          </div>
        </div>

        <div
          v-for="group in audioConfigGroups"
          :key="group.id"
          class="config-card"
        >
          <div class="section-label">
            {{ group.label }}
            <small class="section-hint">{{ group.description }}</small>
          </div>
          <div class="params-grid section-params">
            <div
              v-for="param in group.params"
              :key="`${param.scope}-${param.key}`"
              class="param-field"
              :class="{ 'param-field--unsupported': param.supported === false }"
            >
              <div class="param-field__head">
                <label :for="`audio-${param.scope}-${param.key}`" class="param-field__label">
                  {{ param.label }}
                  <code class="param-key-hint">{{ param.key }}</code>
                  <Tag v-if="param.required" value="Required" severity="danger" />
                  <Tag v-if="param.asset_selector" value="Bundle asset" severity="secondary" />
                  <i class="pi pi-info-circle param-info" v-tooltip.top="paramDescriptionTooltip(param)" />
                </label>
              </div>
              <Dropdown
                v-if="audioParamOptions(param).length"
                :id="`audio-${param.scope}-${param.key}`"
                :model-value="audioParamValue(param)"
                :options="audioParamOptions(param)"
                optionLabel="label"
                optionValue="value"
                :placeholder="param.default != null ? String(param.default) : 'Select…'"
                class="param-input"
                :disabled="param.supported === false"
                @update:model-value="(value) => setAudioParamValue(param, value)"
              />
              <InputNumber
                v-else-if="param.type === 'int' || param.type === 'float'"
                :id="`audio-${param.scope}-${param.key}`"
                :model-value="audioParamValue(param)"
                :min="param.minimum"
                :max="param.maximum"
                :minFractionDigits="param.type === 'float' ? 1 : 0"
                :maxFractionDigits="param.type === 'float' ? 6 : 0"
                class="param-input"
                :disabled="param.supported === false"
                @update:model-value="(value) => setAudioParamValue(param, value)"
              />
              <InputSwitch
                v-else-if="param.type === 'bool'"
                :id="`audio-${param.scope}-${param.key}`"
                :model-value="Boolean(audioParamValue(param))"
                :disabled="param.supported === false"
                @update:model-value="(value) => setAudioParamValue(param, value)"
              />
              <Chips
                v-else-if="param.type === 'list' || param.value_kind === 'repeatable'"
                :id="`audio-${param.scope}-${param.key}`"
                :model-value="audioParamValue(param) || []"
                separator=","
                class="param-input"
                :disabled="param.supported === false"
                @update:model-value="(value) => setAudioParamValue(param, value)"
              />
              <Textarea
                v-else-if="param.type === 'json'"
                :id="`audio-${param.scope}-${param.key}`"
                :model-value="jsonParamDisplay(audioParamValue(param))"
                rows="4"
                class="w-full textarea-cli param-input"
                :disabled="param.supported === false"
                @update:model-value="(value) => updateAudioJsonParam(param, value)"
              />
              <InputText
                v-else
                :id="`audio-${param.scope}-${param.key}`"
                :model-value="audioParamValue(param)"
                :placeholder="param.default != null ? String(param.default) : ''"
                class="param-input"
                :disabled="param.supported === false"
                @update:model-value="(value) => setAudioParamValue(param, value)"
              />
            </div>
          </div>
        </div>

        <div v-if="audioRequestCapabilities.length" class="config-card config-card--compact">
          <button
            type="button"
            class="request-cap-toggle"
            :aria-expanded="showRequestCapabilities"
            @click="showRequestCapabilities = !showRequestCapabilities"
          >
            <span class="request-cap-toggle__title">
              <span class="section-label section-label--inline">Request capabilities</span>
              <Tag :value="String(audioRequestCapabilities.length)" severity="secondary" />
              <i
                class="pi param-info"
                :class="showRequestCapabilities ? 'pi-chevron-up' : 'pi-chevron-down'"
                aria-hidden="true"
              />
            </span>
            <small class="section-hint request-cap-toggle__hint">
              API request options — not saved as server startup settings.
            </small>
          </button>
          <div v-show="showRequestCapabilities" class="request-cap-grid" role="list">
            <div
              v-for="param in audioRequestCapabilities"
              :key="`request-${param.key}`"
              class="request-cap-item"
              role="listitem"
            >
              <code class="request-cap-item__key">{{ param.key }}</code>
              <span class="request-cap-item__label">{{ param.label }}</span>
              <i
                class="pi pi-info-circle param-info request-cap-item__info"
                v-tooltip.top="paramDescriptionTooltip(param)"
              />
            </div>
          </div>
        </div>
      </template>

      <!-- Catalog-backed: search → tags → single params pane -->
      <template v-else-if="catalogSections.length">
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
        <template v-if="activeCmdPreview.sidecar">
          <div class="section-label cmd-preview-env-label">
            audio.cpp server sidecar ({{ activeCmdPreview.sidecarPath }})
          </div>
          <Textarea
            :model-value="activeCmdPreview.sidecar"
            readonly
            rows="12"
            class="w-full textarea-cli cmd-preview-textarea"
            autoResize
          />
        </template>
        <Message
          v-if="activeCmdPreview.genericTaskPath"
          severity="info"
          :closable="false"
          class="cmd-preview-message"
        >
          Generic audio tasks use the temporary llama-swap fallback:
          <code>{{ activeCmdPreview.genericTaskPath }}</code>
        </Message>

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
  tts_profile: null,
  speech_field_groups: [],
  asr_profile: null,
  transcription_field_groups: [],
  task_profile: null,
  request_field_groups: [],
  request_defaults_key: 'task_defaults',
  api_endpoint: '/v1/tasks/run',
  api_example_hint: '',
})
const paramSearchQuery = ref('')
const hideUnsupportedParams = ref(false)
const showRequestCapabilities = ref(false)
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
const cmdPreviewGenericTaskPath = ref('')
const cmdPreviewError = ref(null)
const cmdPreviewLoading = ref(false)
const unsavedCmdPreviewText = ref('')
const unsavedCmdPreviewEnvText = ref('')
const unsavedCmdPreviewMacrosText = ref('')
const unsavedCmdPreviewFiltersText = ref('')
const unsavedCmdPreviewAliasesText = ref('')
const unsavedCmdPreviewSidecarText = ref('')
const unsavedCmdPreviewSidecarPath = ref('')
const unsavedCmdPreviewGenericTaskPath = ref('')
const unsavedCmdPreviewError = ref(null)
const unsavedCmdPreviewLoading = ref(false)
/** Rows for llama-swap YAML `env` (synced into config.swap_env). */
const swapEnvRows = ref([{ key: '', value: '' }])
/** Sub-ID variants for llama-swap ``filters.setParamsByID`` (synced into config.set_params_by_id). */
const setParamsByIdVariants = ref([])
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
let unsavedPreviewRequestId = 0
/** @type {AbortController | null} */
let unsavedPreviewAbort = null

const fallbackEngineOptions = [
  { value: 'llama_cpp', label: 'llama.cpp', icon: 'pi-microchip' },
  { value: 'ik_llama',  label: 'ik_llama.cpp', icon: 'pi-microchip' },
  { value: 'lmdeploy',  label: 'LMDeploy', icon: 'pi-server' },
  { value: '1cat_vllm', label: '1Cat-vLLM', icon: 'pi-server' },
  { value: 'audio_cpp', label: 'audio.cpp', icon: 'pi-volume-up' },
]

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
    ? new Set(model.value.compatible_engines)
    : null
  const fmt = String(model.value?.format || '').toLowerCase()
  const packageKind = model.value?.artifact?.package_kind || model.value?.package_kind
  return options.map((option) => {
    let compatible = verified ? verified.has(option.value) : true
    if (!verified) {
      if (packageKind === 'prepared_bundle') compatible = option.value === 'audio_cpp'
      else if (fmt === 'gguf') compatible = ['llama_cpp', 'ik_llama'].includes(option.value)
      else if (fmt === 'safetensors') compatible = ['lmdeploy', '1cat_vllm'].includes(option.value)
    }
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

const audioEditableParams = computed(() => {
  if (!isAudioEngine.value) return []
  const seen = new Set()
  return catalogParamList.value.filter((param) => {
    const scope = param.scope || 'process'
    const identity = `${scope}:${param.key}`
    if (seen.has(identity) || param.read_only || scope === 'request_option') return false
    seen.add(identity)
    return true
  })
})

const audioConfigGroups = computed(() => {
  const params = audioEditableParams.value
  const definitions = [
    {
      id: 'model',
      label: 'Model identity',
      description: 'Inspected family, task, mode, and bundle asset selection.',
      scopes: ['model'],
    },
    {
      id: 'runtime',
      label: 'Runtime',
      description: 'Backend, device, threads, logging, and server process behavior.',
      scopes: ['process'],
    },
    {
      id: 'load',
      label: 'Model load options',
      description: 'Typed options applied once while loading this prepared bundle.',
      scopes: ['load_option'],
    },
    {
      id: 'session',
      label: 'Session options',
      description: 'Typed defaults used when audio.cpp opens a processing session.',
      scopes: ['session_option'],
    },
  ]
  return definitions
    .map((group) => ({
      ...group,
      params: params.filter((param) => group.scopes.includes(param.scope || 'process')),
    }))
    .filter((group) => group.params.length)
})

const audioRequestCapabilities = computed(() => {
  if (!isAudioEngine.value) return []
  const seen = new Set()
  return catalogParamList.value.filter((param) => {
    const requestOnly = param.read_only || param.scope === 'request_option'
    if (!requestOnly || seen.has(param.key)) return false
    seen.add(param.key)
    return true
  })
})

const OPENAI_SPEECH_TASKS = new Set(['tts', 'clon', 'vdes'])
const GENERIC_TASK_FAMILIES = new Set([
  'ace_step', 'stable_audio', 'heartmula', 'seed_vc', 'miocodec', 'vevo2',
  'htdemucs', 'mel_band_roformer', 'silero_vad', 'marblenet_vad', 'marblenet',
  'sortformer_diar', 'sortformer', 'qwen3_forced_aligner',
])

const isProfiledAudioModel = computed(() => {
  if (!isAudioEngine.value) return false
  return Boolean(paramRegistry.value.task_profile)
})

const taskProfile = computed(() => paramRegistry.value.task_profile || null)

const requestFieldGroups = computed(() => (
  Array.isArray(paramRegistry.value.request_field_groups)
    ? paramRegistry.value.request_field_groups
    : []
))

const requestDefaultsKey = computed(() => (
  paramRegistry.value.request_defaults_key || 'task_defaults'
))

const apiEndpoint = computed(() => paramRegistry.value.api_endpoint || '/v1/tasks/run')

const apiExampleHint = computed(() => paramRegistry.value.api_example_hint || '')

const requestDefaultsSectionTitle = computed(() => {
  const key = requestDefaultsKey.value
  if (key === 'speech_defaults') return 'Voice & speech defaults'
  if (key === 'transcription_defaults') return 'Transcription defaults'
  return 'Task request defaults'
})

const supportsVoicePresets = computed(() => {
  if (!isProfiledAudioModel.value) return false
  if (requestDefaultsKey.value !== 'speech_defaults') return false
  if (!OPENAI_SPEECH_TASKS.has(String(config.value.task || '').toLowerCase())) return false
  const family = String(config.value.family || '').toLowerCase()
  return !GENERIC_TASK_FAMILIES.has(family)
})

const taskWorkflowTags = computed(() => {
  const workflows = taskProfile.value?.workflows || []
  return workflows.map((item) => String(item).replace(/_/g, ' '))
})

const voicePresetFieldDefs = computed(() => {
  const fields = new Map()
  for (const group of requestFieldGroups.value) {
    for (const field of group.fields || []) {
      if (field.preset_field) {
        fields.set(field.key, field)
      }
    }
  }
  if (!fields.size) {
    return [
      { key: 'voice_id', label: 'Built-in voice id', type: 'string', placeholder: 'alba' },
      { key: 'voice_ref', label: 'Reference audio (WAV)', type: 'path', placeholder: 'samples/reference.wav' },
      { key: 'reference_text', label: 'Reference transcript', type: 'textarea', placeholder: 'Transcript for the reference clip…' },
    ]
  }
  return [...fields.values()]
})

const voicePresetRows = computed(() => {
  const presets = config.value.voice_presets
  if (!presets || typeof presets !== 'object' || Array.isArray(presets)) return []
  return Object.entries(presets).map(([name, preset]) => ({
    name,
    preset: preset && typeof preset === 'object' ? preset : {},
  }))
})

const defaultVoicePresetOptions = computed(() => {
  const options = voicePresetRows.value.map((row) => ({
    label: row.name,
    value: row.name,
  }))
  options.unshift({ label: 'Use inline default object', value: '__inline__' })
  return options
})

const defaultVoicePresetSelection = computed(() => {
  const value = config.value.default_voice_preset
  if (typeof value === 'string') return value
  if (value && typeof value === 'object') return '__inline__'
  return null
})

const requestApiExample = computed(() => {
  const modelId = config.value.model_alias || llamaSwapStableId.value || 'your-model-id'
  const endpoint = apiEndpoint.value
  const defaultsKey = requestDefaultsKey.value
  const defaults = config.value[defaultsKey]
  const body = { model: modelId }

  if (endpoint === '/v1/audio/speech') {
    body.input = 'Hello from audio.cpp.'
  } else if (endpoint === '/v1/audio/transcriptions') {
    body.audio = '/path/to/speech.wav'
  } else {
    body.task = config.value.task || 'gen'
    body.family = config.value.family || 'ace_step'
    if (config.value.mode) body.mode = config.value.mode
    body.audio = '/path/to/input.wav'
    body.text = 'Example prompt text.'
  }

  if (defaults && typeof defaults === 'object') {
    for (const [key, value] of Object.entries(defaults)) {
      if (key === 'options') continue
      if (key === 'prompt' && endpoint === '/v1/audio/transcriptions') continue
      if (value != null && value !== '') body[key] = value
    }
    if (defaults.prompt && endpoint === '/v1/audio/transcriptions') {
      body.options = { ...(body.options || {}), text: defaults.prompt }
    }
    if (defaults.options && typeof defaults.options === 'object' && Object.keys(defaults.options).length) {
      body.options = { ...(body.options || {}), ...defaults.options }
    }
  }

  if (endpoint === '/v1/audio/speech') {
    const defaultPreset = config.value.default_voice_preset
    if (typeof defaultPreset === 'string' && defaultPreset && !body.voice) {
      body.voice = defaultPreset
    }
  }

  const lines = [
    `curl http://localhost:2000${endpoint} \\`,
    "  -H 'Content-Type: application/json' \\",
  ]
  if (endpoint === '/v1/audio/speech') {
    lines.push('  -o speech.wav \\')
  }
  lines.push("  -d '" + JSON.stringify(body, null, 2).replace(/'/g, "'\\''") + "'")
  if (endpoint === '/v1/tasks/run') {
    lines.push('')
    lines.push(`# Direct upstream fallback:`)
    lines.push(`curl http://localhost:2000/upstream/${modelId}${endpoint} \\`)
    lines.push("  -H 'Content-Type: application/json' \\")
    lines.push("  -d '" + JSON.stringify(body, null, 2).replace(/'/g, "'\\''") + "'")
  }
  if (endpoint === '/v1/audio/transcriptions') {
    lines.push('')
    lines.push('# Multipart (OpenAI-compatible upload):')
    lines.push(`curl http://localhost:2000${endpoint} \\`)
    lines.push(`  -F "model=${modelId}" \\`)
    lines.push('  -F "file=@speech.wav"')
  }
  return lines.join('\n')
})

const audioInspectionSummary = computed(() => {
  if (!isAudioEngine.value) return []
  const inspection = paramRegistry.value.inspection || {}
  const summary = []
  if (inspection.family) summary.push(`Family: ${inspection.family}`)
  for (const task of inspection.tasks || []) {
    if (!task?.task) continue
    const modes = Array.isArray(task.modes) && task.modes.length
      ? ` (${task.modes.join(', ')})`
      : ''
    summary.push(`${task.task}${modes}`)
  }
  for (const capability of Object.keys(inspection.capabilities || {})) {
    if (inspection.capabilities[capability]) summary.push(capability)
  }
  return [...new Set(summary)]
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
  const known = new Set([
    'custom_args',
    'model_alias',
    'set_params_by_id',
    'swap_env',
    'load_options',
    'session_options',
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

const audioNestedScopeKeys = {
  load_option: 'load_options',
  session_option: 'session_options',
  request_option: 'request_options',
}

function audioParamValue(param, sourceConfig = config.value) {
  const nestedKey = audioNestedScopeKeys[param.scope]
  let value
  if (nestedKey) {
    const nested = sourceConfig[nestedKey]
    value = nested && typeof nested === 'object' && !Array.isArray(nested)
      ? nested[param.key]
      : undefined
  } else {
    value = sourceConfig[param.key]
  }
  return value !== undefined && value !== null && value !== ''
    ? value
    : defaultValueForParam(param)
}

function audioParamOptions(param) {
  if (param.key === 'mode') {
    const task = config.value.task
    const taskRow = (paramRegistry.value.inspection?.tasks || [])
      .find((item) => item?.task === task)
    if (taskRow?.modes?.length) {
      return taskRow.modes.map((mode) => ({ value: mode, label: mode }))
    }
  }
  if (param.key === 'backend') {
    const descriptor = (enginesStore.engineDescriptors || [])
      .find((item) => item.id === 'audio_cpp')
    const available = descriptor?.available_runtime_backends
    if (Array.isArray(available) && available.length) {
      return available.map((backend) => ({ value: backend, label: backend }))
    }
  }
  return Array.isArray(param.options) ? param.options : []
}

function setAudioParamValue(param, value) {
  const nestedKey = audioNestedScopeKeys[param.scope]
  if (nestedKey) {
    if (!config.value[nestedKey] || typeof config.value[nestedKey] !== 'object') {
      config.value[nestedKey] = {}
    }
    if (value === undefined || value === null || value === '') {
      delete config.value[nestedKey][param.key]
    } else {
      config.value[nestedKey][param.key] = value
    }
    return
  }
  if (value === undefined || value === null || value === '') delete config.value[param.key]
  else config.value[param.key] = value
  if (param.key === 'task') {
    const modeParam = audioEditableParams.value.find((item) => item.key === 'mode')
    if (!modeParam) return
    const options = audioParamOptions(modeParam)
    if (!options.some((option) => option.value === config.value.mode)) {
      const offline = options.find((option) => option.value === 'offline')
      config.value.mode = offline?.value || options[0]?.value || null
    }
  }
}

function updateAudioJsonParam(param, raw) {
  if (!raw || !String(raw).trim()) {
    setAudioParamValue(param, null)
    return
  }
  try {
    setAudioParamValue(param, JSON.parse(raw))
  } catch {
    setAudioParamValue(param, raw)
  }
}

function ensureRequestDefaultsShape() {
  const key = requestDefaultsKey.value
  if (!config.value[key] || typeof config.value[key] !== 'object' || Array.isArray(config.value[key])) {
    config.value[key] = {}
  }
}

function requestFieldKey(field) {
  return field.request_field || field.speech_field || field.transcription_field || field.key
}

function requestDefaultValue(field) {
  ensureRequestDefaultsShape()
  const defaults = config.value[requestDefaultsKey.value]
  if (field.nested || field.options_key) {
    const options = defaults.options
    if (!options || typeof options !== 'object') return field.type === 'bool' ? false : null
    if (field.key === 'prompt') return defaults.prompt ?? options.text
    return options[field.options_key || field.key]
  }
  return defaults[requestFieldKey(field)]
}

function setRequestDefaultValue(field, value) {
  ensureRequestDefaultsShape()
  const defaults = config.value[requestDefaultsKey.value]
  if (field.key === 'prompt') {
    const text = value == null ? '' : String(value).trim()
    if (!text) {
      delete defaults.prompt
      if (defaults.options?.text) delete defaults.options.text
      if (defaults.options && !Object.keys(defaults.options).length) delete defaults.options
    } else {
      defaults.prompt = text
    }
    return
  }
  if (field.nested || field.options_key) {
    if (!defaults.options || typeof defaults.options !== 'object') {
      defaults.options = {}
    }
    const key = field.options_key || field.key
    if (value === undefined || value === null || value === '') {
      delete defaults.options[key]
      if (!Object.keys(defaults.options).length) delete defaults.options
    } else if (field.type === 'bool') {
      defaults.options[key] = Boolean(value)
    } else if (field.type === 'int') {
      defaults.options[key] = parseInt(value, 10)
    } else if (field.type === 'float') {
      defaults.options[key] = parseFloat(value)
    } else {
      defaults.options[key] = String(value)
    }
    return
  }
  const key = requestFieldKey(field)
  if (value === undefined || value === null || value === '') {
    delete defaults[key]
  } else if (field.type === 'bool') {
    defaults[key] = Boolean(value)
  } else {
    defaults[key] = value
  }
}

function ensureTtsConfigShape() {
  if (!config.value.voice_presets || typeof config.value.voice_presets !== 'object' || Array.isArray(config.value.voice_presets)) {
    config.value.voice_presets = {}
  }
  ensureRequestDefaultsShape()
}

function addVoicePreset() {
  ensureTtsConfigShape()
  let index = 1
  let name = 'preset-1'
  while (config.value.voice_presets[name]) {
    index += 1
    name = `preset-${index}`
  }
  config.value.voice_presets[name] = {}
}

function removeVoicePreset(name) {
  ensureTtsConfigShape()
  if (!config.value.voice_presets[name]) return
  delete config.value.voice_presets[name]
  if (config.value.default_voice_preset === name) {
    config.value.default_voice_preset = null
  }
}

function renameVoicePreset(oldName, newName) {
  ensureTtsConfigShape()
  const trimmed = String(newName || '').trim()
  if (!trimmed || trimmed === oldName) return
  if (config.value.voice_presets[trimmed]) return
  config.value.voice_presets[trimmed] = config.value.voice_presets[oldName] || {}
  delete config.value.voice_presets[oldName]
  if (config.value.default_voice_preset === oldName) {
    config.value.default_voice_preset = trimmed
  }
}

function setVoicePresetField(name, key, value) {
  ensureTtsConfigShape()
  if (!config.value.voice_presets[name]) {
    config.value.voice_presets[name] = {}
  }
  const text = value == null ? '' : String(value).trim()
  if (!text) delete config.value.voice_presets[name][key]
  else config.value.voice_presets[name][key] = text
}

function setDefaultVoicePresetSelection(value) {
  if (!value) {
    config.value.default_voice_preset = null
    return
  }
  if (value === '__inline__') {
    if (typeof config.value.default_voice_preset !== 'object') {
      config.value.default_voice_preset = {}
    }
    return
  }
  config.value.default_voice_preset = value
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
      genericTaskPath: cmdPreviewGenericTaskPath.value,
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
    genericTaskPath: unsavedCmdPreviewGenericTaskPath.value,
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

async function fetchParamRegistry(engine) {
  try {
    const { data } = await axios.get('/api/models/param-registry', {
      params: {
        engine,
        ...(model.value?.id ? { model_id: model.value.id } : {}),
      },
    })
    paramRegistry.value = {
      sections: data.sections || [],
      scan_error: data.scan_error ?? null,
      scan_pending: Boolean(data.scan_pending),
      profile_fingerprint: data.profile_fingerprint ?? null,
      inspection: data.inspection ?? null,
      compatibility_warnings: data.compatibility_warnings || [],
      tts_profile: data.tts_profile ?? null,
      speech_field_groups: data.speech_field_groups || [],
      asr_profile: data.asr_profile ?? null,
      transcription_field_groups: data.transcription_field_groups || [],
      task_profile: data.task_profile ?? data.tts_profile ?? data.asr_profile ?? null,
      request_field_groups: data.request_field_groups
        || data.speech_field_groups
        || data.transcription_field_groups
        || [],
      request_defaults_key: data.request_defaults_key || 'task_defaults',
      api_endpoint: data.api_endpoint || '/v1/tasks/run',
      api_example_hint: data.api_example_hint || '',
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
  if (preserveUnknownAudioKeys) {
    for (const param of audioEditableParams.value) {
      const nestedKey = audioNestedScopeKeys[param.scope]
      const value = audioParamValue(param, sourceConfig)
      const empty = value == null
        || value === ''
        || (typeof value === 'number' && Number.isNaN(value))
        || (Array.isArray(value) && value.length === 0)
      if (nestedKey) {
        if (!stash[nestedKey] || typeof stash[nestedKey] !== 'object') {
          stash[nestedKey] = {}
        }
        if (empty) delete stash[nestedKey][param.key]
        else stash[nestedKey][param.key] = Array.isArray(value) ? [...value] : value
        continue
      }
      if (empty) delete stash[param.key]
      else stash[param.key] = Array.isArray(value) ? [...value] : value
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
    return stash
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
  if (engine === 'audio_cpp') {
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
      const nestedKey = audioNestedScopeKeys[param.scope]
      if (nestedKey) {
        if (config.value[nestedKey][param.key] == null) {
          config.value[nestedKey][param.key] = defaultValueForParam(param)
        }
      } else if (config.value[param.key] == null) {
        config.value[param.key] = defaultValueForParam(param)
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
  showRequestCapabilities.value = false
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

    const cfgResp = await axios.get(modelApiUrl('/config'))
    const cfg = cfgResp.data
    let engine = cfg.engine ?? found.engine ?? 'llama_cpp'
    if (found.format !== 'safetensors' && ['lmdeploy', '1cat_vllm'].includes(engine)) {
      engine = 'llama_cpp'
    }

    await Promise.all([
      fetchParamRegistry(engine),
      gpuListPromise,
      engineDescriptorsPromise,
    ])

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
      cmdPreviewSidecarText.value = formatSidecarPreview(data.sidecar)
      cmdPreviewSidecarPath.value = data.sidecar_path || ''
      cmdPreviewGenericTaskPath.value = data.generic_task_path || ''
      cmdPreviewError.value = null
    } else {
      cmdPreviewText.value = ''
      cmdPreviewEnvText.value = ''
      cmdPreviewMacrosText.value = ''
      cmdPreviewFiltersText.value = ''
      cmdPreviewAliasesText.value = ''
      cmdPreviewSidecarText.value = ''
      cmdPreviewSidecarPath.value = ''
      cmdPreviewGenericTaskPath.value = ''
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
    cmdPreviewGenericTaskPath.value = ''
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
      unsavedCmdPreviewSidecarText.value = formatSidecarPreview(data.sidecar)
      unsavedCmdPreviewSidecarPath.value = data.sidecar_path || ''
      unsavedCmdPreviewGenericTaskPath.value = data.generic_task_path || ''
      unsavedCmdPreviewError.value = null
    } else {
      unsavedCmdPreviewText.value = ''
      unsavedCmdPreviewEnvText.value = ''
      unsavedCmdPreviewMacrosText.value = ''
      unsavedCmdPreviewFiltersText.value = ''
      unsavedCmdPreviewAliasesText.value = ''
      unsavedCmdPreviewSidecarText.value = ''
      unsavedCmdPreviewSidecarPath.value = ''
      unsavedCmdPreviewGenericTaskPath.value = ''
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
    unsavedCmdPreviewGenericTaskPath.value = ''
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

.engine-option.disabled {
  cursor: not-allowed;
  opacity: 0.5;
}

.engine-option.disabled:hover {
  border-color: var(--border-primary, #2a2f45);
  background: transparent;
}

.engine-disabled-reason {
  max-width: 12rem;
  margin-left: 0.55rem;
  color: var(--text-secondary, #9ca3af);
  font-size: 0.68rem;
  line-height: 1.2;
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

.engine-mark--audio {
  background: linear-gradient(135deg, #10b981, #0891b2);
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
