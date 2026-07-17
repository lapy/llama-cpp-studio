<template>
  <div class="audio-model-config">
    <Message
      v-if="paramRegistry.scan_error"
      severity="warn"
      :closable="false"
      class="config-scan-message"
    >
      <div class="config-message__body">
        <strong>CLI parameters could not be loaded.</strong>
        {{ paramRegistry.scan_error }}
        <Button
          label="Rescan CLI parameters"
          icon="pi pi-refresh"
          size="small"
          severity="secondary"
          outlined
          :loading="rescanLoading"
          class="config-message__action"
          @click="rescanCliParams"
        />
      </div>
    </Message>
    <Message
      v-else-if="paramRegistry.scan_pending"
      severity="info"
      :closable="false"
      class="config-scan-message"
    >
      <div class="config-message__body">
        <strong>CLI parameters not indexed yet.</strong>
        Activate audio.cpp on the Engines page, then rescan.
        <Button
          label="Rescan now"
          icon="pi pi-refresh"
          size="small"
          severity="secondary"
          outlined
          :loading="rescanLoading"
          class="config-message__action"
          @click="rescanCliParams"
        />
      </div>
    </Message>

    <Message
      v-if="contractReviewRequired"
      severity="warn"
      :closable="false"
      class="config-scan-message"
    >
      <div class="config-message__body">
        <strong>audio.cpp contract changed — review required</strong>
        <ul class="config-checklist">
          <li
            v-for="item in contractReviewChecklist"
            :key="item.id"
            class="config-checklist__item"
            :class="{ 'config-checklist__item--done': item.done }"
          >
            <i
              class="pi"
              :class="item.done ? 'pi-check-circle' : 'pi-circle'"
              aria-hidden="true"
            />
            <div>
              <strong>{{ item.label }}</strong>
              <small>{{ item.detail }}</small>
            </div>
          </li>
        </ul>
        <div class="config-message__actions">
          <Button
            label="Rescan CLI parameters"
            icon="pi pi-refresh"
            size="small"
            severity="secondary"
            outlined
            :loading="rescanLoading"
            @click="rescanCliParams"
          />
          <Button
            label="Prune stale defaults &amp; mark reviewed"
            icon="pi pi-check"
            size="small"
            severity="warning"
            :disabled="!contractFingerprint"
            @click="markContractReviewed"
          />
        </div>
      </div>
    </Message>

    <div class="config-card config-card--compact">
      <div class="section-label section-label--inline">
        {{ taskKindMeta.label }} configuration
        <Tag
          :value="taskKindMeta.short"
          :severity="taskKindMeta.tagSeverity"
        />
      </div>
      <p class="config-muted-hint">
        Endpoint <code>{{ apiEndpoint }}</code>
        <template v-if="config.family"> · {{ config.family }}</template>
        <template v-if="config.task"> · {{ config.task }}</template>
      </p>
    </div>

    <div class="engine-selector" role="tablist" aria-label="Audio configuration sections">
      <button
        v-for="tab in tabs"
        :key="tab.id"
        type="button"
        role="tab"
        class="engine-option"
        :class="{ selected: activeTab === tab.id }"
        :aria-selected="activeTab === tab.id"
        @click="activeTab = tab.id"
      >
        <span class="engine-option-label">
          <i :class="tab.icon" aria-hidden="true" />
          <span class="engine-name">{{ tab.label }}</span>
        </span>
      </button>
    </div>

    <!-- Overview -->
    <div v-show="activeTab === 'overview'" class="config-tab-panel">
      <div v-if="taskProfile" class="config-card">
        <div class="config-profile-hero__head">
          <div>
            <div class="section-label section-label--inline">
              {{ taskProfile.label || 'Model profile' }}
              <i
                class="pi pi-info-circle param-info"
                v-tooltip.top="modelProfileTooltip"
                tabindex="0"
                aria-label="About the model profile"
              />
            </div>
          </div>
          <Tag :value="`${setupProgress}% ready`" :severity="setupProgress === 100 ? 'success' : 'info'" />
        </div>
        <div v-if="taskWorkflowTags.length" class="audio-capability-tags">
          <Tag
            v-for="workflow in taskWorkflowTags"
            :key="workflow"
            :value="workflow"
            severity="secondary"
          />
        </div>
      </div>

      <div v-if="audioInspectionSummary.length" class="config-card">
        <div class="section-label section-label--inline">
          Inspected bundle
          <i
            class="pi pi-info-circle param-info"
            v-tooltip.top="uiTooltips.inspectedBundle"
            tabindex="0"
            aria-label="About inspected bundle"
          />
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

      <div class="config-card">
        <div class="section-label section-label--inline">
          Setup checklist
          <i
            class="pi pi-info-circle param-info"
            v-tooltip.top="uiTooltips.setupChecklist"
            tabindex="0"
            aria-label="About setup checklist"
          />
        </div>
        <ul class="config-checklist">
          <li
            v-for="item in setupChecklist"
            :key="item.id"
            class="config-checklist__item"
            :class="{ 'config-checklist__item--done': item.done }"
          >
            <i
              class="pi"
              :class="item.done ? 'pi-check-circle' : 'pi-circle'"
              aria-hidden="true"
            />
            <div>
              <strong>{{ item.label }}</strong>
              <small>{{ item.detail }}</small>
            </div>
            <Button
              v-if="item.tab"
              :label="item.tab === 'api' ? 'Edit defaults' : 'Open'"
              size="small"
              text
              type="button"
              @click="activeTab = item.tab"
            />
          </li>
        </ul>
        <div class="config-checklist__actions">
          <Button
            label="Configure runtime"
            icon="pi pi-server"
            size="small"
            severity="secondary"
            outlined
            @click="activeTab = 'server'"
          />
          <Button
            label="Manage assets"
            icon="pi pi-folder-open"
            size="small"
            severity="secondary"
            outlined
            @click="activeTab = 'assets'"
          />
          <Button
            v-if="isProfiledAudioModel"
            label="Set defaults"
            icon="pi pi-sliders-h"
            size="small"
            severity="secondary"
            outlined
            @click="activeTab = 'api'"
          />
        </div>
      </div>
    </div>

    <!-- Server -->
    <div v-show="activeTab === 'server'" class="config-tab-panel">
      <div class="config-card">
        <div class="runtime-common-head">
          <div>
            <div class="section-label section-label--inline">
              Common settings
              <i
                class="pi pi-info-circle param-info"
                v-tooltip.top="uiTooltips.commonRuntime"
                tabindex="0"
                aria-label="About common settings"
              />
              <Tag :value="`${commonRuntimeParams.length} fields`" severity="secondary" />
            </div>
          </div>
          <div class="toggle-field runtime-advanced-toggle">
            <InputSwitch v-model="showAdvancedRuntime" input-id="audio-runtime-advanced" />
            <label for="audio-runtime-advanced">Advanced</label>
          </div>
        </div>

        <div v-if="commonRuntimeParams.length" class="params-grid section-params runtime-common-grid">
          <div
            v-for="param in commonRuntimeParams"
            :key="`common-${param.scope}-${param.key}`"
            class="param-field"
            :class="{ 'param-field--unsupported': param.supported === false }"
          >
            <div class="param-field__head">
              <label :for="`audio-common-${param.scope}-${param.key}`" class="param-field__label">
                {{ param.label }}
                <Tag v-if="param.required" value="Required" severity="danger" />
                <Tag v-if="param.asset_selector" value="Bundle asset" severity="secondary" />
                <i
                  class="pi pi-info-circle param-info"
                  v-tooltip.top="paramDescriptionTooltip(param)"
                />
              </label>
            </div>
            <AudioParamField
              :id="`audio-common-${param.scope}-${param.key}`"
              :param="param"
              :model-value="audioParamValue(param)"
              :options="audioParamOptions(param)"
              :disabled="param.supported === false"
              @update:model-value="(value) => setAudioParamValue(param, value)"
              @update:json="(value) => updateAudioJsonParam(param, value)"
            />
          </div>
        </div>
        <Message v-else severity="secondary" :closable="false" class="config-scan-message runtime-empty-message">
          No common runtime settings were found in the current audio.cpp parameter index.
        </Message>
      </div>

      <template v-if="showAdvancedRuntime">
        <div class="config-card config-toolbar">
          <div class="config-toolbar__row">
            <span class="p-input-icon-left config-search-wrap">
              <i class="pi pi-search" aria-hidden="true" />
              <InputText
                v-model="serverSearchQuery"
                type="search"
                placeholder="Filter advanced parameters…"
                class="config-search-input"
                aria-label="Filter advanced parameters"
              />
            </span>
            <Button
              v-if="serverSearchQuery"
              icon="pi pi-times"
              text
              rounded
              severity="secondary"
              aria-label="Clear search"
              @click="serverSearchQuery = ''"
            />
          </div>
          <div class="config-toolbar__row config-toolbar__toggles">
            <div class="toggle-field">
              <InputSwitch v-model="hideUnsupportedParams" input-id="audio-hide-unsupported" />
              <label for="audio-hide-unsupported">Hide unsupported in this build</label>
            </div>
          </div>
        </div>

        <div
          v-for="group in visibleServerGroups"
          :key="group.id"
          class="config-card"
        >
          <button
            type="button"
            class="request-cap-toggle"
            :aria-expanded="expandedGroups[group.id]"
            @click="toggleGroup(group.id)"
          >
            <span class="request-cap-toggle__title">
              <i
                class="pi"
                :class="expandedGroups[group.id] ? 'pi-chevron-down' : 'pi-chevron-right'"
                aria-hidden="true"
              />
              <span>
                <span class="section-label section-label--inline">
                  {{ group.label }}
                  <i
                    class="pi pi-info-circle param-info"
                    v-tooltip.top="runtimeGroupTooltip(group)"
                    tabindex="0"
                    aria-label="About runtime group"
                  />
                </span>
                <Tag :value="String(group.params.length)" severity="secondary" />
              </span>
            </span>
          </button>

          <div v-show="expandedGroups[group.id]" class="config-group-body">
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
                    <code class="param-key-hint">{{ paramStorageKey(param) }}</code>
                    <Tag v-if="param.required" value="Required" severity="danger" />
                    <Tag v-if="param.asset_selector" value="Bundle asset" severity="secondary" />
                    <Tag value="Sidecar" severity="success" class="param-supported-tag" />
                    <i
                      class="pi pi-info-circle param-info"
                      v-tooltip.top="paramDescriptionTooltip(param)"
                    />
                  </label>
                </div>
                <AudioParamField
                  :id="`audio-${param.scope}-${param.key}`"
                  :param="param"
                  :model-value="audioParamValue(param)"
                  :options="audioParamOptions(param)"
                  :disabled="param.supported === false"
                  @update:model-value="(value) => setAudioParamValue(param, value)"
                  @update:json="(value) => updateAudioJsonParam(param, value)"
                />
              </div>
            </div>
            <Message
              v-if="!group.params.length"
              severity="secondary"
              :closable="false"
              class="config-scan-message"
            >
              No parameters match your filter in this group.
            </Message>
          </div>
        </div>
      </template>
    </div>

    <!-- Assets -->
    <div v-show="activeTab === 'assets'" class="config-tab-panel">
      <div class="config-card">
        <div class="tts-subsection__head">
          <div>
            <div class="section-label section-label--inline">
              Reference audio
              <i
                class="pi pi-info-circle param-info"
                v-tooltip.top="uiTooltips.referenceAudio"
                tabindex="0"
                aria-label="About reference audio"
              />
              <Tag value="Max upload: 60 MB" severity="secondary" />
            </div>
          </div>
          <div class="reference-audio-actions">
            <input
              ref="referenceUploadInput"
              type="file"
              accept=".wav,audio/wav,audio/x-wav"
              class="reference-audio-upload-input"
              @change="onReferenceAudioSelected"
            />
            <Button
              label="Upload WAV"
              icon="pi pi-upload"
              size="small"
              severity="secondary"
              outlined
              type="button"
              :loading="referenceAudioUploading"
              @click="openReferenceAudioUpload"
            />
            <Button
              icon="pi pi-refresh"
              size="small"
              severity="secondary"
              text
              rounded
              type="button"
              aria-label="Refresh reference audio list"
              :loading="referenceAudioLoading"
              @click="loadReferenceAudio"
            />
          </div>
        </div>

        <div v-if="referenceAudioLoading && !referenceAudioItems.length" class="config-muted-hint">
          Loading reference audio…
        </div>
        <div v-else-if="!referenceAudioItems.length" class="config-muted-hint">
          No reference audio uploaded yet.
        </div>
        <div v-else class="reference-audio-list">
          <div
            v-for="item in referenceAudioItems"
            :key="item.path"
            class="reference-audio-row"
          >
            <div class="reference-audio-row__meta">
              <code class="reference-audio-row__path">
                {{ item.display_path || item.relative_path || item.path }}
              </code>
              <span class="reference-audio-row__size">{{ formatBytes(item.size_bytes) }}</span>
              <Tag
                v-for="usage in item.used_by || []"
                :key="`${item.path}-${usage}`"
                :value="usage"
                severity="info"
              />
            </div>
            <div class="reference-audio-row__actions">
              <Button
                v-if="supportsVoicePresets && voicePresetRows.length"
                label="Use in preset"
                icon="pi pi-link"
                size="small"
                text
                type="button"
                @click="openUseReferenceInPreset(item)"
              />
              <Button
                icon="pi pi-trash"
                severity="danger"
                text
                rounded
                type="button"
                aria-label="Delete reference audio"
                :loading="referenceAudioDeleting === item.filename"
                @click="deleteReferenceAudioItem(item)"
              />
            </div>
          </div>
        </div>
      </div>

      <div v-if="supportsVoicePresets" class="config-card">
        <div class="tts-subsection__head">
          <div>
            <div class="section-label section-label--inline">
              Voice presets
              <i
                class="pi pi-info-circle param-info"
                v-tooltip.top="uiTooltips.voicePresets"
                tabindex="0"
                aria-label="About voice presets"
              />
            </div>
          </div>
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

        <div v-if="!voicePresetRows.length" class="config-muted-hint">
          No voice presets yet.
        </div>
        <div v-for="row in voicePresetRows" :key="row.id" class="voice-preset-card">
          <div class="voice-preset-card__head">
            <InputText
              :model-value="voicePresetNameDraft(row.name)"
              class="voice-preset-card__name"
              placeholder="preset-name"
              @update:model-value="(value) => setVoicePresetNameDraft(row.name, value)"
              @blur="() => commitVoicePresetRename(row.name)"
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
              :key="`${row.id}-${field.key}`"
              class="param-field"
            >
              <label class="param-field__label">
                {{ field.label }}
                <i
                  class="pi pi-info-circle param-info"
                  v-tooltip.top="voicePresetFieldTooltip(field)"
                  tabindex="0"
                  aria-label="About voice preset field"
                />
              </label>
              <div v-if="field.type === 'path'" class="reference-path-field">
                <Dropdown
                  :model-value="row.preset[field.key] || ''"
                  :options="referenceAudioPathOptions"
                  optionLabel="label"
                  optionValue="value"
                  placeholder="Choose uploaded clip or type below"
                  showClear
                  editable
                  class="param-input reference-path-field__dropdown"
                  @update:model-value="(value) => setVoicePresetField(row.name, field.key, value)"
                />
              </div>
              <Textarea
                v-else-if="field.type === 'textarea'"
                :model-value="row.preset[field.key] || ''"
                :placeholder="field.placeholder || ''"
                rows="2"
                class="w-full textarea-cli param-input"
                @update:model-value="(value) => setVoicePresetField(row.name, field.key, value)"
              />
              <InputText
                v-else-if="field.type !== 'path'"
                :model-value="row.preset[field.key] || ''"
                :placeholder="field.placeholder || ''"
                class="param-input"
                @update:model-value="(value) => setVoicePresetField(row.name, field.key, value)"
              />
            </div>
          </div>
        </div>

        <div class="param-field">
          <label class="param-field__label">
            Default voice preset
            <i
              class="pi pi-info-circle param-info"
              v-tooltip.top="uiTooltips.defaultVoicePreset"
              tabindex="0"
              aria-label="About default voice preset"
            />
          </label>
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
    </div>

    <!-- API defaults -->
    <div v-show="activeTab === 'api'" class="config-tab-panel">
      <div class="config-card">
        <div class="section-label section-label--inline">
          {{ requestDefaultsSectionTitle }}
          <i
            class="pi pi-info-circle param-info"
            v-tooltip.top="uiTooltips.requestDefaults"
            tabindex="0"
            aria-label="About request defaults"
          />
          <Tag :value="apiEndpoint" severity="secondary" />
        </div>

        <p v-if="defaultsApplyHint" class="config-muted-hint">{{ defaultsApplyHint }}</p>
        <Message
          v-if="instructionsPolicyGuidance"
          severity="info"
          :closable="false"
          class="config-scan-message"
        >
          {{ instructionsPolicyGuidance }}
        </Message>

        <div v-if="swapSetParamsPreview" class="setparams-preview">
          <button
            type="button"
            class="setparams-preview__label setparams-preview__toggle"
            :aria-expanded="showSetParamsPreview"
            @click="showSetParamsPreview = !showSetParamsPreview"
          >
            <span>llama-swap <code>filters.setParams</code> preview</span>
            <i
              class="pi"
              :class="showSetParamsPreview ? 'pi-chevron-up' : 'pi-chevron-down'"
              aria-hidden="true"
            />
          </button>
          <pre v-if="showSetParamsPreview" class="setparams-preview__code">{{ JSON.stringify(swapSetParamsPreview, null, 2) }}</pre>
          <p v-if="showSetParamsPreview" class="config-muted-hint">
            Relative reference paths are resolved against the model and reference-audio roots when you apply config.
          </p>
        </div>

        <template v-if="isProfiledAudioModel">
          <div v-if="requestFieldGroups.length" class="tts-subsection">
            <div class="tts-subsection__title">Request default fields</div>
            <div
              v-for="group in requestFieldGroups"
              :key="group.id"
              class="tts-speech-group"
            >
              <div class="tts-speech-group__label">
                {{ group.label }}
                <i
                  class="pi pi-info-circle param-info"
                  v-tooltip.top="requestGroupTooltip(group)"
                  tabindex="0"
                  aria-label="About request default group"
                />
              </div>
              <div class="params-grid section-params">
                <div
                  v-for="field in group.fields"
                  :key="`${group.id}-${field.key}`"
                  class="param-field"
                >
                  <label class="param-field__label">
                    {{ field.label }}
                    <Tag value="Proxy" severity="info" class="param-supported-tag" />
                    <code v-if="field.nested || field.options_key" class="param-key-hint">
                      options.{{ field.options_key || field.key }}
                    </code>
                    <i
                      class="pi pi-info-circle param-info"
                      v-tooltip.top="requestDefaultFieldTooltip(field)"
                      tabindex="0"
                      aria-label="About request default field"
                    />
                  </label>
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
        </template>

        <Message v-else severity="secondary" :closable="false" class="config-scan-message">
          No request defaults profile is available for this audio model.
        </Message>
      </div>
    </div>

    <!-- Reference -->
    <div v-show="activeTab === 'reference'" class="config-tab-panel">
      <div class="config-card">
        <div class="tts-subsection__head">
          <div class="section-label section-label--inline">
            API example
            <i
              class="pi pi-info-circle param-info"
              v-tooltip.top="apiExampleTooltip"
              tabindex="0"
              aria-label="About API example"
            />
          </div>
          <Button
            label="Copy curl"
            icon="pi pi-copy"
            size="small"
            severity="secondary"
            outlined
            type="button"
            @click="copyApiExample"
          />
        </div>
        <p class="config-muted-hint">
          Endpoint: <code>{{ apiEndpoint }}</code> · Model id:
          <code>{{ config.model_alias || llamaSwapStableId || 'your-model-id' }}</code>
        </p>
        <Textarea
          :model-value="requestApiExample"
          readonly
          rows="14"
          class="w-full textarea-cli cmd-preview-textarea"
          autoResize
        />
      </div>

      <div v-if="audioRequestCapabilities.length" class="config-card">
        <div class="tts-subsection__head">
          <div class="section-label section-label--inline">
            Request-only parameters
            <i
              class="pi pi-info-circle param-info"
              v-tooltip.top="uiTooltips.requestOnlyParams"
              tabindex="0"
              aria-label="About request-only parameters"
            />
          </div>
          <Button
            v-if="isProfiledAudioModel"
            label="Edit Defaults"
            icon="pi pi-sliders-h"
            size="small"
            severity="secondary"
            outlined
            type="button"
            @click="activeTab = 'api'"
          />
        </div>
        <p class="config-muted-hint">
          These CLI options are not startup settings. Persist reusable values on the Defaults tab when a profile field exists.
        </p>
        <div class="config-toolbar__row">
          <span class="p-input-icon-left config-search-wrap">
            <i class="pi pi-search" aria-hidden="true" />
            <InputText
              v-model="referenceSearchQuery"
              type="search"
              placeholder="Filter request parameters…"
              class="config-search-input"
              aria-label="Filter request parameters"
            />
          </span>
        </div>
        <div class="request-cap-grid" role="list">
          <div
            v-for="param in filteredRequestCapabilities"
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
        <Message
          v-if="!filteredRequestCapabilities.length"
          severity="secondary"
          :closable="false"
          class="config-scan-message"
        >
          No request parameters match your filter.
        </Message>
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed, ref, reactive, watch, onMounted } from 'vue'
import { useToast } from 'primevue/usetoast'
import Button from 'primevue/button'
import Tag from 'primevue/tag'
import InputText from 'primevue/inputtext'
import InputNumber from 'primevue/inputnumber'
import InputSwitch from 'primevue/inputswitch'
import Dropdown from 'primevue/dropdown'
import Message from 'primevue/message'
import Textarea from 'primevue/textarea'
import AudioParamField from '@/components/audio/AudioParamField.vue'
import {
  useAudioModelConfig,
  paramDescriptionTooltip,
  AUDIO_NESTED_SCOPE_KEYS,
} from '@/composables/useAudioModelConfig'
import { useEnginesStore } from '@/stores/engines'
import { useModelStore } from '@/stores/models'

const props = defineProps({
  config: {
    type: Object,
    required: true,
  },
  paramRegistry: {
    type: Object,
    required: true,
  },
  llamaSwapStableId: {
    type: String,
    default: '',
  },
  modelId: {
    type: String,
    default: '',
  },
})

const emit = defineEmits(['rescan-complete'])

const toast = useToast()
const enginesStore = useEnginesStore()
const modelStore = useModelStore()

const activeTab = ref('overview')
const serverSearchQuery = ref('')
const referenceSearchQuery = ref('')
const hideUnsupportedParams = ref(false)
const showAdvancedRuntime = ref(false)
const showSetParamsPreview = ref(false)
const rescanLoading = ref(false)
const referenceAudioItems = ref([])
const referenceAudioLoading = ref(false)
const referenceAudioUploading = ref(false)
const referenceAudioDeleting = ref('')
const referenceUploadInput = ref(null)
const REFERENCE_AUDIO_MAX_BYTES = 60 * 1024 * 1024

const tabs = computed(() => [
  { id: 'overview', label: 'Overview', icon: 'pi pi-compass', hint: 'Setup' },
  { id: 'server', label: 'Runtime', icon: 'pi pi-server', hint: 'Sidecar' },
  { id: 'assets', label: 'Assets', icon: 'pi pi-folder-open', hint: 'Refs & voices' },
  { id: 'api', label: 'Defaults', icon: 'pi pi-sliders-h', hint: taskKindMeta.value.tabHint },
  { id: 'reference', label: 'API', icon: 'pi pi-code', hint: 'curl' },
])

const COMMON_RUNTIME_PARAM_ORDER = [
  'family',
  'task',
  'mode',
  'config',
  'weight',
  'backend',
  'device',
  'threads',
  'lazy_load',
]
const COMMON_RUNTIME_PARAM_KEYS = new Set(COMMON_RUNTIME_PARAM_ORDER)
const COMMON_RUNTIME_ORDER = new Map(
  COMMON_RUNTIME_PARAM_ORDER.map((key, index) => [key, index]),
)
const COMMON_RUNTIME_SCOPE_PRIORITY = new Map([
  ['model', 0],
  ['process', 1],
  ['load_option', 2],
  ['session_option', 3],
])

const uiTooltips = Object.freeze({
  modelProfile: 'Shows the detected audio task type and setup status for this model.',
  inspectedBundle: 'Capabilities detected from the installed audio bundle.',
  setupChecklist: 'Tracks the required runtime choices and optional saved defaults.',
  commonRuntime: 'The small set of runtime fields most often needed before applying configuration.',
  referenceAudio: 'Uploaded WAV files are reusable server-side references for audio requests.',
  voicePresets: 'Named voice settings are saved with the audio runtime configuration.',
  defaultVoicePreset: 'Used when a compatible speech request does not choose a voice.',
  requestDefaults: 'Saved values are applied by llama-swap before the request reaches audio.cpp.',
  apiExample: 'Example request using the current endpoint and model identifier.',
  requestOnlyParams: 'These fields can be sent in a request without becoming startup settings.',
})

const configRef = computed(() => props.config)
const registryRef = computed(() => props.paramRegistry)
const stableIdRef = computed(() => props.llamaSwapStableId)

const audio = useAudioModelConfig(configRef, registryRef, enginesStore, stableIdRef)

const {
  audioConfigGroups,
  audioRequestCapabilities,
  taskProfile,
  isProfiledAudioModel,
  requestFieldGroups,
  apiEndpoint,
  apiExampleHint,
  requestDefaultsSectionTitle,
  audioTaskKind,
  taskKindMeta,
  swapSetParamsPreview,
  configuredDefaultsCount,
  defaultsApplyHint,
  instructionsPolicyGuidance,
  contractReviewRequired,
  contractFingerprint,
  markContractReviewed,
  setupProgress,
  supportsVoicePresets,
  taskWorkflowTags,
  voicePresetFieldDefs,
  voicePresetRows,
  voicePresetNameDraft,
  setVoicePresetNameDraft,
  commitVoicePresetRename,
  defaultVoicePresetOptions,
  defaultVoicePresetSelection,
  audioInspectionSummary,
  setupChecklist,
  requestApiExample,
  audioParamValue,
  audioParamOptions,
  setAudioParamValue,
  updateAudioJsonParam,
  requestDefaultValue,
  setRequestDefaultValue,
  addVoicePreset,
  removeVoicePreset,
  setVoicePresetField,
  setDefaultVoicePresetSelection,
  filterGroupParams,
} = audio

const contractReviewChecklist = computed(() => {
  const defaultsKey = props.paramRegistry?.request_defaults_key || 'task_defaults'
  const hasActiveDefaults = Boolean(
    props.config?.[defaultsKey]
    && typeof props.config[defaultsKey] === 'object'
    && Object.keys(props.config[defaultsKey]).length,
  )
  return [
    {
      id: 'rescan',
      label: 'Capability scan current',
      detail: props.paramRegistry?.scan_pending
        ? 'Rescan CLI parameters after activating audio.cpp.'
        : 'Scan data is loaded for this model.',
      done: !props.paramRegistry?.scan_pending && !props.paramRegistry?.scan_error,
    },
    {
      id: 'defaults-key',
      label: `Defaults key is ${defaultsKey}`,
      detail: hasActiveDefaults
        ? `Active request defaults are stored under ${defaultsKey}.`
        : `No ${defaultsKey} values saved yet — confirm the endpoint still matches this model.`,
      done: Boolean(defaultsKey) && !props.paramRegistry?.scan_error,
    },
    {
      id: 'reviewed',
      label: 'Mark this model reviewed',
      detail: 'Prune stale defaults objects and record the current contract fingerprint.',
      done: !contractReviewRequired.value,
    },
  ]
})

const modelProfileTooltip = computed(() => [
  uiTooltips.modelProfile,
  taskProfile.value?.summary,
  taskProfile.value?.api_hint,
].filter(Boolean).join('\n\n'))

const apiExampleTooltip = computed(() => [
  uiTooltips.apiExample,
  apiExampleHint.value,
].filter(Boolean).join('\n\n'))

const referenceAudioPathOptions = computed(() =>
  referenceAudioItems.value.map((item) => ({
    label: item.display_path || item.relative_path || item.path,
    value: item.path,
  })),
)

const commonRuntimeParams = computed(() => {
  const chosen = new Map()
  for (const group of audioConfigGroups.value) {
    for (const param of group.params || []) {
      const scope = param.scope || 'process'
      const commonKey = COMMON_RUNTIME_PARAM_KEYS.has(param.key)
      const modelAsset = scope === 'model' && param.asset_selector
      const required = param.required === true
      if ((!commonKey && !modelAsset && !required) || param.supported === false) continue

      const existing = chosen.get(param.key)
      const existingScope = existing?.scope || 'process'
      const priority = COMMON_RUNTIME_SCOPE_PRIORITY.get(scope) ?? 99
      const existingPriority = COMMON_RUNTIME_SCOPE_PRIORITY.get(existingScope) ?? 99
      if (!existing || priority < existingPriority) {
        chosen.set(param.key, param)
      }
    }
  }
  return [...chosen.values()].sort((a, b) => {
    const aOrder = COMMON_RUNTIME_ORDER.get(a.key)
    const bOrder = COMMON_RUNTIME_ORDER.get(b.key)
    const aRank = aOrder ?? (a.required ? 50 : 99)
    const bRank = bOrder ?? (b.required ? 50 : 99)
    if (aRank !== bRank) return aRank - bRank
    return String(a.label || a.key).localeCompare(String(b.label || b.key))
  })
})

watch(
  () => props.modelId,
  (modelId) => {
    if (modelId) {
      void loadReferenceAudio()
    } else {
      referenceAudioItems.value = []
    }
  },
  { immediate: true },
)

watch(activeTab, (tab) => {
  if (tab === 'assets' && props.modelId) {
    void loadReferenceAudio()
  }
})

onMounted(() => {
  if (props.modelId) {
    void loadReferenceAudio()
  }
})

function formatBytes(bytes) {
  const value = Number(bytes) || 0
  if (value < 1024) return `${value} B`
  if (value < 1024 * 1024) return `${(value / 1024).toFixed(1)} KB`
  return `${(value / (1024 * 1024)).toFixed(1)} MB`
}

async function loadReferenceAudio() {
  if (!props.modelId) return
  referenceAudioLoading.value = true
  try {
    referenceAudioItems.value = await modelStore.listReferenceAudio(props.modelId)
  } catch (error) {
    toast.add({
      severity: 'error',
      summary: 'Failed to load reference audio',
      detail: error?.response?.data?.detail || error?.message || String(error),
      life: 5000,
    })
  } finally {
    referenceAudioLoading.value = false
  }
}

function openReferenceAudioUpload() {
  referenceUploadInput.value?.click()
}

async function onReferenceAudioSelected(event) {
  const file = event.target.files?.[0]
  event.target.value = ''
  if (!file || !props.modelId) return
  if (file.size > REFERENCE_AUDIO_MAX_BYTES) {
    toast.add({
      severity: 'warn',
      summary: 'Upload too large',
      detail: `Reference WAVs must be ${formatBytes(REFERENCE_AUDIO_MAX_BYTES)} or smaller.`,
      life: 5000,
    })
    return
  }
  referenceAudioUploading.value = true
  try {
    const saved = await modelStore.uploadReferenceAudio(props.modelId, file)
    await loadReferenceAudio()
    toast.add({
      severity: 'success',
      summary: 'Reference audio uploaded',
      detail: saved?.path
        ? `Saved as ${saved.display_path || saved.relative_path || saved.path}`
        : undefined,
      life: 3500,
    })
  } catch (error) {
    toast.add({
      severity: 'error',
      summary: 'Upload failed',
      detail: error?.response?.data?.detail || error?.message || String(error),
      life: 5000,
    })
  } finally {
    referenceAudioUploading.value = false
  }
}

async function deleteReferenceAudioItem(item) {
  if (!props.modelId || !item?.filename) return
  referenceAudioDeleting.value = item.filename
  try {
    await modelStore.deleteReferenceAudio(props.modelId, item.filename)
    await loadReferenceAudio()
    toast.add({
      severity: 'success',
      summary: 'Reference audio deleted',
      detail: item.display_path || item.relative_path || item.path,
      life: 3000,
    })
  } catch (error) {
    toast.add({
      severity: 'error',
      summary: 'Delete failed',
      detail: error?.response?.data?.detail || error?.message || String(error),
      life: 6000,
    })
  } finally {
    referenceAudioDeleting.value = ''
  }
}

function openUseReferenceInPreset(item) {
  const firstPreset = voicePresetRows.value[0]?.name
  if (!firstPreset) return
  setVoicePresetField(firstPreset, 'voice_ref', item.path)
  toast.add({
    severity: 'info',
    summary: 'Preset updated',
    detail: `Set voice_ref on "${firstPreset}" to ${item.display_path || item.path}. Save configuration to apply.`,
    life: 4500,
  })
}

const expandedGroups = reactive({})

watch(
  audioConfigGroups,
  (groups) => {
    for (const group of groups) {
      if (expandedGroups[group.id] === undefined) {
        expandedGroups[group.id] = group.defaultExpanded !== false
      }
    }
  },
  { immediate: true },
)

const visibleServerGroups = computed(() =>
  audioConfigGroups.value
    .map((group) => ({
      ...group,
      params: filterGroupParams(group.params, serverSearchQuery.value, hideUnsupportedParams.value),
    }))
    .filter((group) => group.params.length || !serverSearchQuery.value.trim()),
)

const filteredRequestCapabilities = computed(() => {
  const q = referenceSearchQuery.value.trim().toLowerCase()
  if (!q) return audioRequestCapabilities.value
  return audioRequestCapabilities.value.filter((param) => {
    const hay = [param.key, param.label, param.description].join(' ').toLowerCase()
    return hay.includes(q)
  })
})

function paramStorageKey(param) {
  const nestedKey = AUDIO_NESTED_SCOPE_KEYS[param.scope]
  if (nestedKey) return `${nestedKey}.${param.key}`
  return param.key
}

function runtimeGroupTooltip(group) {
  return group?.description || 'Settings in this group are written to the audio runtime configuration.'
}

function requestGroupTooltip(group) {
  return group?.description || 'Fields in this group are saved as request defaults.'
}

function requestDefaultFieldTooltip(field) {
  const modelHints = [field?.description, field?.hint].filter(Boolean)
  if (modelHints.length) return modelHints.join('\n\n')
  if (field?.nested || field?.options_key) {
    return 'Saved under options in the request body.'
  }
  if (field?.type === 'path') {
    return 'Saved as a server-side path for compatible audio requests.'
  }
  return 'Saved as a request default for this model.'
}

function voicePresetFieldTooltip(field) {
  const modelHints = [field?.description, field?.hint].filter(Boolean)
  if (modelHints.length) return modelHints.join('\n\n')
  if (field?.type === 'path') {
    return 'Reference paths are resolved on the server from the installed bundle.'
  }
  return 'Saved as part of the selected voice preset.'
}

function toggleGroup(groupId) {
  expandedGroups[groupId] = !expandedGroups[groupId]
}

async function rescanCliParams() {
  rescanLoading.value = true
  try {
    const data = await enginesStore.scanEngineParams('audio_cpp')
    if (data?.ok) {
      toast.add({
        severity: 'success',
        summary: 'CLI parameters scanned',
        detail: `Indexed ${data.param_count ?? 0} options for audio.cpp.`,
        life: 3500,
      })
      emit('rescan-complete')
    } else {
      toast.add({
        severity: 'warn',
        summary: 'Scan failed',
        detail: data?.scan_error || 'Unknown error',
        life: 6000,
      })
    }
  } catch (error) {
    toast.add({
      severity: 'error',
      summary: 'Scan failed',
      detail: error?.message || String(error),
      life: 5000,
    })
  } finally {
    rescanLoading.value = false
  }
}

async function copyApiExample() {
  try {
    await navigator.clipboard.writeText(requestApiExample.value)
    toast.add({
      severity: 'success',
      summary: 'Copied',
      detail: 'API example copied to clipboard.',
      life: 2500,
    })
  } catch {
    toast.add({
      severity: 'warn',
      summary: 'Copy failed',
      detail: 'Select the text manually.',
      life: 4000,
    })
  }
}
</script>

<style scoped>
.audio-model-config {
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
}

.config-message__actions {
  display: flex;
  flex-wrap: wrap;
  gap: 0.4rem;
  margin-top: 0.65rem;
}

.runtime-common-head {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 0.75rem;
}

.runtime-common-grid {
  margin-top: 0.85rem;
}

.runtime-advanced-toggle {
  flex-shrink: 0;
}

.runtime-empty-message {
  margin-top: 0.85rem;
}

.setparams-preview__toggle {
  width: 100%;
  border: 0;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 0.75rem;
  color: inherit;
  cursor: pointer;
  text-align: left;
}

.reference-audio-actions {
  display: flex;
  align-items: center;
  gap: 0.35rem;
}

.reference-audio-upload-input {
  display: none;
}

.reference-audio-list {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}

.reference-audio-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 0.75rem;
  padding: 0.55rem 0.65rem;
  border: 1px solid var(--border-primary, #2a2f45);
  border-radius: var(--radius-md, 0.5rem);
  background: var(--bg-surface, rgba(255, 255, 255, 0.02));
}

.reference-audio-row__meta {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 0.45rem;
  min-width: 0;
}

.reference-audio-row__path {
  font-size: 0.78rem;
}

.reference-audio-row__size {
  font-size: 0.75rem;
  color: var(--text-secondary, #9ca3af);
}

.reference-audio-row__actions {
  display: flex;
  align-items: center;
  gap: 0.25rem;
  flex-shrink: 0;
}

.reference-path-field {
  display: flex;
  flex-direction: column;
  gap: 0.35rem;
}

.reference-path-field__dropdown {
  width: 100%;
}
</style>
