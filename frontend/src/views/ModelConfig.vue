<template>
  <div class="model-config-view page-shell page-shell--wide">
    <template v-if="!model">
      <PageHeader
        title="Model configuration"
        description="Review the engine, runtime parameters, names, and launch settings for this model."
      />
      <LoadingState v-if="loading" message="Loading configuration…" />
      <EmptyState
        v-else-if="loadError"
        role="alert"
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
      <EmptyState
        v-else
        icon="pi pi-exclamation-circle"
        title="Model not found"
        description="This model is not in the catalog. It may have been removed; return to Models to choose another one."
      >
        <Button label="Back to Models" icon="pi pi-arrow-left" @click="$router.push('/models')" />
      </EmptyState>
    </template>

    <template v-else>
      <ModelConfigHeader
        :model="model"
        :loading="loading"
        :can-check-for-updates="canCheckForUpdates"
        :refreshing-model="refreshingModel"
        :is-audio-engine="isAudioEngine"
        @back="requestLeave('/models')"
        @check-for-updates="checkForModelUpdates"
        @open-audio="openAudioWorkspace"
      />

      <ModelConfigActionBar
        :model-label="modelIdentityLabel"
        :quantization="model.quantization || ''"
        :format="modelFormatLabel"
        :engine-label="currentEngineLabel"
        :status="configurationStatus"
        :dirty="hasUnsavedChanges"
        :can-save="canSaveConfig"
        :save-hint="saveConfigHint"
        :saving="saving"
        :show-apply="showApplyRuntime"
        :apply-label="applyRuntimeLabel"
        :applying="applyingRuntime"
        :apply-disabled="saving || applyingRuntime || configSaveHeld"
        :apply-hint="applyRuntimeHint"
        :running="modelIsRunning"
        :starting="isStartingModel"
        :can-start="canStartModel"
        :start-disabled-reason="startModelBlockedReason"
        :can-review="hasUnsavedChanges"
        :review-label="reviewChangesLabel"
        :show-import-command="canImportCommand"
        @save="saveConfig"
        @apply="requestApply"
        @start="startModelFromConfig"
        @open-library="openModelInLibrary"
        @review="changeReviewVisible = true"
        @templates="openTemplatesDialog"
        @import-command="openParseCommandDialog()"
        @reset="resetConfig"
      />

      <StateBanner v-if="modelLaunchPlan?.action === 'invalid'" severity="error">
        {{ modelLaunchPlan.reasons.join(' ') }} Fix the configuration before applying settings.
        <template #actions>
          <Button
            label="Review configuration"
            size="small"
            severity="secondary"
            outlined
            @click="goToConfigSection('overview')"
          />
        </template>
      </StateBanner>
      <StateBanner v-if="loadError" severity="error" role="status">
        Could not refresh this configuration. {{ loadError }}
        <template #actions>
          <Button label="Retry" size="small" severity="secondary" outlined @click="loadAll" />
        </template>
      </StateBanner>

      <StateBanner v-if="draftOffer" severity="info" role="status">
        A draft of your edits for this model and engine is saved in this session.
        <template #actions>
          <Button label="Restore draft" size="small" @click="restoreDraft" />
          <Button
            label="Discard draft"
            size="small"
            severity="secondary"
            outlined
            @click="discardDraftOffer"
          />
        </template>
      </StateBanner>
      <StateBanner v-if="draftRestored" severity="success" role="status">
        Restored unsaved changes from this session.
        <template #actions>
          <Button
            label="Dismiss"
            size="small"
            severity="secondary"
            outlined
            @click="draftRestored = false"
          />
        </template>
      </StateBanner>

      <StateBanner
        v-if="persistenceNotice"
        class="persistence-notice"
        severity="error"
        role="status"
      >
        <strong>{{ persistenceNotice.summary }}.</strong>
        {{ persistenceNotice.detail }}
        <template #actions>
          <Button
            v-if="persistenceNotice.retry"
            label="Try again"
            size="small"
            severity="secondary"
            outlined
            @click="retryPersistence"
          />
          <Button
            v-if="persistenceNotice.refresh"
            label="Refresh"
            size="small"
            severity="secondary"
            outlined
            @click="refreshPersistedConfig"
          />
        </template>
      </StateBanner>

      <StateBanner v-if="unrecognizedSavedKeys.length" severity="warning">
        {{
          isAudioEngine
            ? 'Unrecognized audio.cpp keys are preserved on save:'
            : isSglangEngine
              ? 'Unrecognized SGLang keys are preserved on save:'
              : 'Deprecated or unrecognized saved keys for this engine will be dropped on the next save:'
        }}
        <code>{{ unrecognizedSavedKeys.join(', ') }}</code>
        <template #actions>
          <Button
            v-if="hasUnsavedChanges"
            label="Review changes"
            size="small"
            severity="secondary"
            outlined
            @click="changeReviewVisible = true"
          />
        </template>
      </StateBanner>

      <SectionTabs
        :items="pageTabs"
        :model-value="pageTab"
        aria-label="Configuration sections"
        :id-prefix="tabIdPrefix"
        variant="pills"
        responsive-grid
        @update:model-value="selectPageTabById"
      />

      <div
        v-show="pageTab === 'overview'"
        :id="configSectionPanelId('overview')"
        role="tabpanel"
        :aria-labelledby="`${tabIdPrefix}-tab-overview`"
        tabindex="0"
        class="config-tab-panel"
      >
        <ModelConfigOverview
          :model="model"
          :config="config"
          :status="configurationStatus"
          :engine-options="visibleEngineOptions"
          :unavailable-saved-engine="unavailableSavedEngine"
          :sglang-runtime="sglangRuntime"
          :sglang-runtime-options="sglangRuntimeOptions"
          :is-sglang-engine="isSglangEngine"
          :is-audio-engine="isAudioEngine"
          :gpu-info="gpuInfo"
          :model-limits="modelLimits"
          :api-endpoint="paramRegistry.api_endpoint || ''"
          :proxy-port="proxyPort"
          :launch-plan="modelLaunchPlan"
          :plan-available="enginesStore.runtimePlan?.launch_manifests !== false"
          :has-unsaved-changes="hasUnsavedChanges"
          :pending-apply="pendingApply"
          :config-save-held="configSaveHeld"
          :model-runtime-quality="modelRuntimeQuality"
          :running="modelIsRunning"
          :starting="isStartingModel"
          :auto-fit="autoFitState"
          @select-engine="changeEngine"
          @update:sglang-runtime="changeSglangRuntime"
          @open-engines="requestLeave('/engines')"
          @go-to-section="goToConfigSection"
          @restore-auto-fit="restoreAutoFit"
          @review-auto-fit="goToConfigSection('runtime')"
        />
      </div>

      <div
        v-if="!isAudioEngine"
        v-show="pageTab === 'runtime'"
        :id="configSectionPanelId('runtime')"
        role="tabpanel"
        :aria-labelledby="`${tabIdPrefix}-tab-runtime`"
        tabindex="0"
        class="config-tab-panel"
      >
        <PageSection
          title="Runtime"
          description="Choose the parameters this engine uses when it starts, then review what is overridden."
        >
          <ModelConfigParamGroups
            v-if="catalogSections.length"
            :key="`${config.engine}:${sglangRuntime}`"
            :params="runtimeParams"
            :addable-params="addableParamList"
            :config="config"
            :search-query="paramSearchQuery"
            :hide-unsupported="hideUnsupportedParams"
            :is-modified="isExplicitOverride"
            :is-delimited-enum="isDelimitedEnumParam"
            :max-context-suggestion="maxContextSuggestion"
            :layer-count-suggestion="layerCountSuggestion"
            :pinned-keys="BASIC_PARAM_KEYS"
            :auto-fit-overrides="autoFitState.overrides"
            @update:search-query="paramSearchQuery = $event"
            @update:hide-unsupported="hideUnsupportedParams = $event"
            @update:param="setParamValue"
            @update:json="updateJsonParam"
            @reset:param="useParameterDefault"
            @remove:param="removeParamKey"
            @add:param="addParamKey"
            @restore-auto-fit="restoreAutoFit"
          />
          <ModelConfigRuntimeEmpty
            v-else
            :engine-label="selectedEngineLabel"
            :scan-error="selectedParamRegistry.scan_error || ''"
            :scan-pending="Boolean(selectedParamRegistry.scan_pending)"
            :saved-entries="savedRuntimeEntries"
            :rescanning="runtimeRescanning"
            @rescan="rescanRuntimeParams"
            @open-engines="openEnginesPage"
          />
          <StateBanner
            v-if="catalogSections.length && selectedParamRegistry.scan_error"
            severity="warning"
            role="status"
          >
            Could not read
            {{ isSglangEngine && sglangRuntime === 'diffusion' ? 'SGLang diffusion' : 'engine' }}
            CLI help: {{ selectedParamRegistry.scan_error }}. Rescan the engine's CLI parameters,
            then reopen this page.
            <template #actions>
              <Button
                label="Rescan runtime parameters"
                icon="pi pi-refresh"
                size="small"
                severity="secondary"
                outlined
                :loading="runtimeRescanning"
                @click="rescanRuntimeParams"
              />
              <Button
                label="Open Engines"
                icon="pi pi-server"
                size="small"
                severity="secondary"
                outlined
                @click="openEnginesPage"
              />
            </template>
          </StateBanner>
          <StateBanner
            v-else-if="catalogSections.length && selectedParamRegistry.scan_pending"
            severity="info"
            role="status"
          >
            {{
              isSglangEngine && sglangRuntime === 'diffusion'
                ? 'SGLang diffusion CLI parameters are'
                : 'CLI parameters are'
            }}
            not loaded for this engine yet. Activate the engine on the Engines page, then reopen
            this page.
            <template #actions>
              <Button
                label="Rescan runtime parameters"
                icon="pi pi-refresh"
                size="small"
                severity="secondary"
                outlined
                :loading="runtimeRescanning"
                @click="rescanRuntimeParams"
              />
              <Button
                label="Open Engines"
                icon="pi pi-server"
                size="small"
                severity="secondary"
                outlined
                @click="openEnginesPage"
              />
            </template>
          </StateBanner>
          <StateBanner
            v-for="warning in paramRegistry.compatibility_warnings || []"
            :key="warning"
            severity="warning"
          >
            {{ warning }}
          </StateBanner>
          <ModelConfigGpuCard
            v-if="showNvidiaGpuBind"
            v-model:mode="gpuMode"
            v-model:selected="cudaVisibleDeviceSelection"
            :gpus="nvidiaGpuSelectOptions"
            class="workbench-block"
          />
        </PageSection>
      </div>

      <ModelConfigGpuCard
        v-if="isAudioEngine && pageTab === 'server' && showNvidiaGpuBind"
        v-model:mode="gpuMode"
        v-model:selected="cudaVisibleDeviceSelection"
        :gpus="nvidiaGpuSelectOptions"
        class="config-card"
      />

      <AudioModelConfig
        v-if="isAudioEngine"
        v-show="pageTab !== 'launch' && pageTab !== 'overview' && pageTab !== 'names'"
        hide-tabs
        :active-tab="pageTab"
        :config="config"
        :param-registry="paramRegistry"
        :stable-model-id="stableModelId"
        :model-id="model?.id || ''"
        @rescan-complete="fetchParamRegistry(audioEngineId, { rescan: true })"
      />

      <div
        v-show="pageTab === 'names'"
        :id="configSectionPanelId('names')"
        role="tabpanel"
        :aria-labelledby="`${tabIdPrefix}-tab-names`"
        tabindex="0"
        class="config-tab-panel"
      >
        <ModelConfigNamesSummary
          :model="model"
          :config="config"
          :stable-model-id="stableModelId"
          @edit-section="goToConfigSection"
          @manage-templates="openTemplatesDialog"
        />
      </div>

      <div
        v-show="pageTab === 'launch'"
        :id="configSectionPanelId('launch')"
        role="tabpanel"
        :aria-labelledby="`${tabIdPrefix}-tab-launch`"
        tabindex="0"
        class="config-tab-panel"
      >
        <PageSection
          title="Launch"
          description="Control aliases, environment, and the exact command Studio uses to start this model."
        >
          <ModelConfigLaunchSettings
            v-model:model-alias="config.model_alias"
            v-model:set-params-by-id="config.set_params_by_id"
            v-model:custom-args="config.custom_args"
            :model="model"
            :engine="config.engine"
            :stable-model-id="stableModelId"
            :can-import-command="canImportCommand"
            :env-rows="runtimeEnvRowsDisplayed"
            :cmd-preview="cmdPreviewState"
            @refresh-model="refreshModelFromStore"
            @validity="requestAliasesValid = $event"
            @parse-command="openParseCommandDialog"
            @open-cmd-preview="openCmdPreviewDialog"
            @add-env-row="addRuntimeEnvRow"
            @remove-env-row="removeRuntimeEnvRow"
            @update-env-row="updateRuntimeEnvRow"
          />
        </PageSection>
      </div>

      <ConfigChangeReview
        :visible="changeReviewVisible"
        :groups="configChangeGroups"
        :saving="saving"
        :invalid="!requestAliasesValid"
        @update:visible="changeReviewVisible = $event"
        @save="saveFromReview"
        @discard="discardFromReview"
      />

      <Dialog
        v-model:visible="cmdPreviewDialogVisible"
        :header="cmdPreviewDialogHeader"
        modal
        class="dialog-width-lg cmd-preview-dialog"
      >
        <div class="dialog-stack">
          <p class="cmd-preview-dialog-hint">{{ cmdPreviewDialogHint }}</p>
          <LoadingState
            v-if="activeCmdPreview.loading"
            inline
            small
            :message="activeCmdPreview.loadingText"
          />
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
            <h3 class="page-subheading cmd-preview-env-label">Launcher command (diagnostic)</h3>
            <Textarea
              :model-value="activeCmdPreview.launcher"
              readonly
              rows="3"
              class="w-full textarea-cli cmd-preview-textarea"
              autoResize
            />
          </template>
          <template v-if="activeCmdPreview.env">
            <h3 class="page-subheading cmd-preview-env-label">
              Process environment ({{ activeCmdPreview.suffix }})
            </h3>
            <Textarea
              :model-value="activeCmdPreview.env"
              readonly
              rows="4"
              class="w-full textarea-cli cmd-preview-textarea"
              autoResize
            />
          </template>
          <template v-if="activeCmdPreview.filters">
            <h3 class="page-subheading cmd-preview-env-label">
              Request defaults ({{ activeCmdPreview.suffix }})
            </h3>
            <Textarea
              :model-value="activeCmdPreview.filters"
              readonly
              rows="6"
              class="w-full textarea-cli cmd-preview-textarea"
              autoResize
            />
          </template>
          <template v-if="activeCmdPreview.aliases">
            <h3 class="page-subheading cmd-preview-env-label">
              Model aliases ({{ activeCmdPreview.suffix }})
            </h3>
            <Textarea
              :model-value="activeCmdPreview.aliases"
              readonly
              rows="3"
              class="w-full textarea-cli cmd-preview-textarea"
              autoResize
            />
          </template>
          <template v-if="activeCmdPreview.sidecar">
            <h3 class="page-subheading cmd-preview-env-label">
              Runtime config file for this model
              <small class="section-hint">
                Written when you Apply. Companion model paths and voice presets appear here.
                <template v-if="activeCmdPreview.sidecarPath">
                  ({{ activeCmdPreview.sidecarPath }})
                </template>
              </small>
            </h3>
            <Textarea
              :model-value="activeCmdPreview.sidecar"
              readonly
              rows="12"
              class="w-full textarea-cli cmd-preview-textarea"
              autoResize
            />
          </template>
        </div>

        <template #footer>
          <div class="dialog-actions">
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
          </div>
        </template>
      </Dialog>

      <Dialog
        v-model:visible="templatesDialogVisible"
        header="Configuration templates"
        modal
        class="dialog-width-lg config-templates-dialog"
        @show="fetchConfigTemplates"
      >
        <div class="dialog-stack">
          <div class="config-templates-layout">
            <section class="config-templates-library">
              <div class="config-templates-library-head">
                <div>
                  <h3>Saved templates</h3>
                  <p>Apply settings, or only the routing and alias configuration, to this model.</p>
                </div>
                <label class="config-templates-search">
                  <i class="pi pi-search" aria-hidden="true" />
                  <InputText
                    v-model="templateSearch"
                    placeholder="Search templates"
                    aria-label="Search templates"
                  />
                </label>
              </div>

              <div class="config-templates-apply-mode">
                <label for="tpl-apply-mode">Apply settings as</label>
                <Select
                  id="tpl-apply-mode"
                  v-model="templateApplyForm.apply_engines"
                  :options="templateApplyModeOptions"
                  option-label="label"
                  option-value="value"
                  class="w-full"
                />
              </div>

              <PersistenceAlert
                :notice="templateMutationNotice"
                @retry="retryTemplateMutation"
                @refresh="refreshTemplateMutation"
              />
              <LoadingState v-if="templateApplyLoading" inline small message="Applying template…" />

              <LoadingState
                v-if="configTemplatesLoading"
                inline
                small
                message="Loading templates…"
              />
              <Message v-else-if="!configTemplates.length" severity="secondary" :closable="false">
                No templates saved yet. Save the current form on the right to create one.
              </Message>
              <Message
                v-else-if="!filteredConfigTemplates.length"
                severity="secondary"
                :closable="false"
              >
                No template matches “{{ templateSearch }}”.
              </Message>
              <ul v-else class="config-templates-cards">
                <li
                  v-for="tpl in filteredConfigTemplates"
                  :key="tpl.id"
                  class="config-template-card"
                >
                  <div v-if="templateEditId === tpl.id" class="config-template-card__edit">
                    <label class="sr-only" :for="`tpl-edit-name-${tpl.id}`">Template name</label>
                    <InputText
                      :id="`tpl-edit-name-${tpl.id}`"
                      v-model="templateEditForm.name"
                      class="w-full"
                    />
                    <label class="sr-only" :for="`tpl-edit-desc-${tpl.id}`"
                      >Template description</label
                    >
                    <InputText
                      :id="`tpl-edit-desc-${tpl.id}`"
                      v-model="templateEditForm.description"
                      placeholder="Description"
                      class="w-full"
                    />
                    <div class="config-template-card__actions">
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
                    <div class="config-template-card__body">
                      <div class="config-template-card__heading">
                        <strong>{{ tpl.name }}</strong>
                        <small v-if="tpl.updated_at">{{
                          new Date(tpl.updated_at).toLocaleDateString()
                        }}</small>
                      </div>
                      <p v-if="tpl.description" class="config-template-card__desc">
                        {{ tpl.description }}
                      </p>
                      <div class="config-template-card__tags">
                        <Tag
                          v-for="engineId in tpl.engine_ids || []"
                          :key="engineId"
                          :value="engineId"
                          severity="secondary"
                        />
                        <Tag
                          v-if="tpl.include_routing || tpl.has_routing"
                          value="routing"
                          severity="info"
                        />
                        <Tag
                          v-if="tpl.has_request_aliases"
                          value="request aliases"
                          severity="info"
                        />
                      </div>
                    </div>
                    <div class="config-template-card__actions">
                      <Button
                        label="Apply settings"
                        size="small"
                        severity="secondary"
                        outlined
                        :disabled="templateMutationHeld || templateApplyLoading"
                        @click="applyConfigTemplate(false, { templateId: tpl.id })"
                      />
                      <Button
                        label="Apply & save"
                        size="small"
                        severity="success"
                        outlined
                        :disabled="templateMutationHeld || templateApplyLoading"
                        @click="applyConfigTemplate(true, { templateId: tpl.id })"
                      />
                      <Button
                        v-if="tpl.include_routing || tpl.has_routing || tpl.has_request_aliases"
                        label="Routing only"
                        size="small"
                        severity="secondary"
                        :disabled="templateMutationHeld || templateApplyLoading"
                        @click="applyConfigTemplate(false, { templateId: tpl.id, mode: 'routing' })"
                      />
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
            </section>

            <aside class="config-templates-save">
              <h3>Save current form</h3>
              <p class="config-templates-save__hint">
                Capture engine parameters, request aliases, and optionally routing aliases for
                reuse.
              </p>
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
                <label for="tpl-routing">Include routing aliases</label>
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
            </aside>
          </div>
        </div>

        <template #footer>
          <div class="dialog-actions">
            <Button
              label="Close"
              severity="secondary"
              outlined
              @click="templatesDialogVisible = false"
            />
          </div>
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
        header="Review runtime changes"
        modal
        class="dialog-width-md"
      >
        <p class="apply-impact__lead">{{ applyImpactMessage }}</p>
        <ul v-if="applyDifferences.length" class="apply-diff">
          <li v-for="row in applyDifferences" :key="row.field" class="apply-diff__row">
            <span class="apply-diff__field">{{ applyDiffLabel(row.field) }}</span>
            <div class="apply-diff__values">
              <span>
                <small>Running</small>
                <span class="apply-diff__value">{{ row.running || 'not running' }}</span>
              </span>
              <i class="pi pi-arrow-right apply-diff__arrow" aria-hidden="true" />
              <span>
                <small>After apply</small>
                <strong class="apply-diff__value">{{ row.saved }}</strong>
              </span>
            </div>
            <small v-if="row.published !== row.saved" class="apply-diff__note">
              Published before apply: {{ row.published }}
            </small>
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
import { formatLabel } from '@/utils/modelPresentation'
import UnsavedChangesDialog from '@/components/common/UnsavedChangesDialog.vue'
import { ref, computed, onMounted, onBeforeUnmount, watch, nextTick } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { clearDraft, readDraft, useDraftGuard, writeDraft } from '@/composables/useDraftGuard'
import {
  MODEL_CONFIG_OVERVIEW_TAB,
  useModelConfigSections,
} from '@/composables/useModelConfigSections'
import { useModelConfigActionBar } from '@/composables/useModelConfigActionBar'
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
import ToggleSwitch from 'primevue/toggleswitch'
import Select from 'primevue/select'
import Message from 'primevue/message'
import Textarea from 'primevue/textarea'
import LoadingState from '@/components/common/LoadingState.vue'
import EmptyState from '@/components/common/EmptyState.vue'
import PageHeader from '@/components/common/PageHeader.vue'
import PageSection from '@/components/common/PageSection.vue'
import SectionTabs from '@/components/common/SectionTabs.vue'
import StateBanner from '@/components/common/StateBanner.vue'
import ModelConfigHeader from '@/components/models/ModelConfigHeader.vue'
import ModelConfigActionBar from '@/components/models/ModelConfigActionBar.vue'
import ModelConfigOverview from '@/components/models/ModelConfigOverview.vue'
import ModelConfigNamesSummary from '@/components/models/ModelConfigNamesSummary.vue'
import ModelConfigLaunchSettings from '@/components/models/ModelConfigLaunchSettings.vue'
import ModelConfigGpuCard from '@/components/models/ModelConfigGpuCard.vue'
import ModelConfigParamGroups from '@/components/models/ModelConfigParamGroups.vue'
import ModelConfigRuntimeEmpty from '@/components/models/ModelConfigRuntimeEmpty.vue'
import { useRuntimeEmptyState } from '@/composables/useRuntimeEmptyState'
import { useConfigChangeGroups } from '@/composables/useConfigChangeGroups'
import AudioModelConfig from '@/components/audio/AudioModelConfig.vue'
import ConfigChangeReview from '@/components/ConfigChangeReview.vue'
import ParseCommandDialog from '@/components/ParseCommandDialog.vue'
import { countConfigChanges } from '@/utils/configChanges'
import {
  defaultValueForParam,
  isDelimitedEnumParam,
  normalizeCsvEnumValue,
  paramIsActiveInSection,
} from '@/utils/modelParamValues'
import { useModelAutoFit } from '@/composables/useModelAutoFit'
import {
  useAudioModelConfig,
  AUDIO_NESTED_SCOPE_KEYS,
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
} from '@/engines/registry'

const audioEngineId = AUDIO_ENGINE_IDS[0] || ''

const route = useRoute()
const router = useRouter()
const toast = useToast()
const modelStore = useModelStore()
const enginesStore = useEnginesStore()

// ── State ──────────────────────────────────────────────────
const loading = ref(true)
const loadError = ref('')
const draftOffer = ref(null)
const draftRestored = ref(false)
const draftsEnabled = ref(false)
const applyImpactVisible = ref(false)
const changeReviewVisible = ref(false)
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
  const last = lastTemplateApply.value
  if (last) {
    void applyConfigTemplate(last.persist, {
      templateId: last.templateId,
      mode: last.mode,
    })
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
const cmdPreviewState = computed(() => ({
  dialogVisible: cmdPreviewDialogVisible.value,
  mode: cmdPreviewDialogMode.value,
  loading: cmdPreviewLoading.value,
  unsavedLoading: unsavedCmdPreviewLoading.value,
}))
const templatesDialogVisible = ref(false)
const parseCommandDialogVisible = ref(false)
const parseCommandSeed = ref('')
const configTemplates = ref([])
const configTemplatesLoading = ref(false)
const templateSearch = ref('')
const lastTemplateApply = ref(null)
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
const filteredConfigTemplates = computed(() => {
  const query = templateSearch.value.trim().toLowerCase()
  if (!query) return configTemplates.value
  return configTemplates.value.filter((template) =>
    [template.name, template.description, template.engine, ...(template.engine_ids || [])]
      .filter(Boolean)
      .join(' ')
      .toLowerCase()
      .includes(query),
  )
})
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

function isAudioPanel(engineId) {
  return ENGINE_DESCRIPTORS[engineId]?.configPanel === 'audio'
}
const isAudioEngine = computed(() => isAudioPanel(config.value.engine))
function runtimeKeyFor(engineId) {
  return ENGINE_DESCRIPTORS[engineId]?.runtimeKey || ''
}
const engineRuntimeKey = computed(() => runtimeKeyFor(config.value.engine))
const isSglangEngine = computed(() => Boolean(engineRuntimeKey.value))
const sglangRuntimeOptions = SGLANG_RUNTIME_OPTIONS
const sglangRuntime = computed(() => normalizeSglangRuntime(config.value[engineRuntimeKey.value]))
function normalizeSglangRuntime(runtime) {
  return runtime === 'diffusion' ? 'diffusion' : 'llm'
}
const DEFAULT_CONFIG_TABS = Object.freeze([
  Object.freeze({ id: 'runtime', label: 'Runtime', icon: 'pi pi-server' }),
  Object.freeze({ id: 'launch', label: 'Launch', icon: 'pi pi-play' }),
])
/**
 * A stable read-only section between the engine's parameter tabs and Launch.
 * Names are routing identity, not runtime parameters, so they get one place
 * that explains every published name.
 */
const NAMES_CONFIG_TAB = Object.freeze({ id: 'names', label: 'Names', icon: 'pi pi-tags' })
const pageTab = ref(MODEL_CONFIG_OVERVIEW_TAB.id)
const pageTabs = computed(() => {
  const engineTabs = ENGINE_DESCRIPTORS[config.value.engine]?.configTabs || DEFAULT_CONFIG_TABS
  const launchTabs = engineTabs.filter((tab) => tab.id === 'launch')
  const parameterTabs = engineTabs.filter((tab) => tab.id !== 'launch')
  return [MODEL_CONFIG_OVERVIEW_TAB, ...parameterTabs, NAMES_CONFIG_TAB, ...launchTabs]
})
watch(
  () => pageTabs.value,
  (tabs) => {
    const ids = new Set(tabs.map((tab) => tab.id))
    if (!ids.has(pageTab.value)) pageTab.value = tabs[0]?.id || MODEL_CONFIG_OVERVIEW_TAB.id
  },
)
const configSectionsReady = computed(
  () => !loading.value && Boolean(model.value) && Boolean(config.value?.engine),
)
/**
 * Audio panels live inside AudioModelConfig with `audio-config-*` ids, so the
 * page tab ids follow the same prefix; text engines keep `config-*`.
 */
const tabIdPrefix = computed(() => (isAudioEngine.value ? 'audio-config' : 'config'))
function configSectionPanelId(sectionId) {
  return `${tabIdPrefix.value}-panel-${sectionId}`
}
const { selectTab: selectPageTab } = useModelConfigSections({
  tabs: pageTabs,
  activeTab: pageTab,
  ready: configSectionsReady,
  panelId: configSectionPanelId,
})
function selectPageTabById(tabId) {
  const tab = pageTabs.value.find((item) => item.id === tabId)
  if (tab) selectPageTab(tab)
}
function goToConfigSection(sectionId) {
  const tab = pageTabs.value.find((item) => item.id === sectionId)
  if (tab) selectPageTab(tab)
}
function configCapability(name, fallback = true) {
  const value = ENGINE_DESCRIPTORS[config.value.engine]?.configCapabilities?.[name]
  return value === undefined ? fallback : value
}
const canImportCommand = computed(
  () => configCapability('commandImport') && catalogSections.value.length > 0,
)

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
  if (!configCapability('modelUpdates')) return false
  return modelFormat.value === 'gguf' || modelFormat.value === 'safetensors'
})

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

/** v-model bridge for the GPU card; selection side effects live in setGpuMode. */
const gpuMode = computed({
  get: () => config.value.gpu_mode || 'inherit',
  set: (mode) => setGpuMode(mode),
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

const { state: autoFitState, restore: restoreAutoFit } = useModelAutoFit({
  enabled: computed(() => Boolean(model.value) && !isAudioEngine.value),
  parameters: catalogParamList,
  config,
  activeParamKeys,
  catalogParamByKey,
  toast,
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
  applyAudioParamsToSection,
  ensureTtsConfigShape,
  seedSessionVoiceFromPackagedIds,
} = audioModelConfig

const currentEngineSection = computed(
  () => (config.value.engines && config.value.engines[config.value.engine]) || {},
)

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
    ...(ENGINE_DESCRIPTORS[config.value.engine]?.knownConfigKeys || []),
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

const configChangeGroups = useConfigChangeGroups({
  config,
  savedConfig,
  catalogParamByKey,
  engineOptions,
  hasUnsavedChanges,
  loading,
  buildEngineStashFromForm,
})

const configChangeCount = computed(() => countConfigChanges(configChangeGroups.value))
const reviewChangesLabel = computed(() =>
  configChangeCount.value ? `Review changes (${configChangeCount.value})` : 'Review changes',
)

const { leavePromptVisible, finishLeave } = useDraftGuard(
  () => hasUnsavedChanges.value && !loading.value,
)
watchDialogFocus(applyImpactVisible)
watchDialogFocus(changeReviewVisible)
watchDialogFocus(cmdPreviewDialogVisible)
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
    return 'These edits are only in this form. Save changes stores them; then apply them so the model uses them.'
  if (pendingApply.value) {
    const action = modelLaunchPlan.value?.action
    if (action === 'publish_next_start') {
      return 'Saved changes for this model are not published yet. They apply the next time it starts. Choose “Use on next start” in the action bar to publish them now.'
    }
    if (action === 'restart_now') {
      return 'Saved changes for this model are not in use yet. Choose “Restart this model” in the action bar to restart it with these settings.'
    }
    return 'Pending changes for this model are not in use yet. Choose the apply action in the action bar to apply them. Saving alone never restarts the model.'
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

// ── Action bar state ───────────────────────────────────────
const proxyPort = computed(() => enginesStore.systemStatus?.proxy_status?.port ?? null)
const hasUsableEngine = computed(() => visibleEngineOptions.value.length > 0)
const {
  modelIdentityLabel,
  modelFormatLabel,
  currentEngineLabel,
  isStartingModel,
  canSaveConfig,
  saveConfigHint,
  startModelBlockedReason,
  canStartModel,
  configurationStatus,
  startModelFromConfig,
  openModelInLibrary,
} = useModelConfigActionBar({
  router,
  toast,
  modelStore,
  enginesStore,
  model,
  config,
  saving,
  hasUnsavedChanges,
  modelIsRunning,
  modelRuntimeQuality,
  showApplyRuntime,
  publishedStepLabel,
  runtimeStateDetail,
  configSaveHeld,
  requestAliasesValid,
  persistenceNotice,
  persistenceAction,
  unrecognizedSavedKeys,
  modelLaunchPlan,
  engineOptions,
  hasUsableEngine,
  modelFormat,
  refreshModelFromStore,
  formatError: formatAxiosDetail,
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

function applyDiffLabel(field) {
  return catalogParamByKey.value.get(field)?.label || formatLabel(field)
}

const applyImpactModels = computed(() => {
  const name = model.value?.display_name || model.value?.base_model_name || 'This model'
  return `Affected: ${name}.`
})

const basicParams = computed(() =>
  BASIC_PARAM_KEYS.map((key) => catalogParamByKey.value.get(key)).filter(Boolean),
)

/**
 * Rows shown in the Runtime parameter groups: the always-visible basic params
 * plus every explicitly active catalog param (order preserved, keys deduped).
 */
const runtimeParams = computed(() => {
  const seen = new Set()
  const out = []
  for (const param of [...basicParams.value, ...paneParams.value]) {
    if (!param?.key || seen.has(param.key)) continue
    seen.add(param.key)
    out.push(param)
  }
  return out
})

/** Catalog params that search can still add to the pane (not already shown). */
const addableParamList = computed(() => {
  const shown = new Set(runtimeParams.value.map((param) => param.key))
  return catalogParamList.value.filter((param) => !shown.has(param.key))
})

const {
  selectedEngineLabel,
  savedRuntimeEntries,
  runtimeRescanning,
  rescanRuntimeParams,
  openEnginesPage,
} = useRuntimeEmptyState({ config, engineOptions, fetchParamRegistry, router })

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

function useParameterDefault(param) {
  delete config.value[param.key]
}

function setParamValue(key, value) {
  config.value[key] = value
}

function isExplicitOverride(param) {
  const value = config.value?.[param.key]
  if (value == null || value === '') return false
  if (Array.isArray(value) && value.length === 0) return false
  return true
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

function updateRuntimeEnvRow({ originalIndex, field, value }) {
  const row = runtimeEnvRows.value[originalIndex]
  if (!row) return
  row[field] = value
  syncRuntimeEnvFromRows()
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

function refreshModelFromStore() {
  const found = findModelById(route.params.id)
  if (found) model.value = found
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

function paramRegistryFlag(engineId, key) {
  return ENGINE_DESCRIPTORS[engineId]?.paramRegistry?.[key] === true
}

async function fetchParamRegistry(engine, { draftFamily, draftTask, rescan = false } = {}) {
  try {
    const family =
      draftFamily ?? (paramRegistryFlag(engine, 'family') ? config.value?.family : undefined)
    const task = draftTask ?? (paramRegistryFlag(engine, 'task') ? config.value?.task : undefined)
    const { data } = await axios.get('/api/models/param-registry', {
      params: {
        engine,
        ...(model.value?.id ? { model_id: model.value.id } : {}),
        ...(paramRegistryFlag(engine, 'family') && family ? { family } : {}),
        ...(paramRegistryFlag(engine, 'task') && task ? { task } : {}),
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
    if (paramRegistryFlag(engine, 'pruneRequestDefaults')) {
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
    if (!paramRegistryFlag(config.value?.engine, 'refreshOnChange') || loading.value) return
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
  const preserveUnknownAudioKeys = Boolean(
    ENGINE_DESCRIPTORS[sourceConfig.engine]?.configCapabilities?.preserveUnknownKeys,
  )
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
    applyAudioParamsToSection(stash, sourceConfig)
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
  if (isAudioPanel(engine)) {
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

// ── Load ───────────────────────────────────────────────────
async function loadAll() {
  loading.value = true
  loadError.value = ''
  draftsEnabled.value = false
  draftOffer.value = null
  draftRestored.value = false
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

async function applyConfigTemplate(persist, options = {}) {
  const templateId = options.templateId || templateApplyForm.value.template_id
  if (!templateId) return
  const mode = options.mode === 'routing' ? 'routing' : 'settings'
  templateApplyForm.value.template_id = templateId
  templateApplyPersist.value = persist
  lastTemplateApply.value = { templateId, persist, mode }
  templateApplyLoading.value = true
  try {
    const { data } = await axios.post(modelApiUrl('/config/apply-template'), {
      template_id: templateId,
      apply_engines: mode === 'routing' ? 'routing' : templateApplyForm.value.apply_engines,
      include_routing: mode === 'routing' ? true : templateApplyForm.value.include_routing,
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
    const routingOnly = mode === 'routing'
    toast.add({
      severity: 'success',
      summary: persist
        ? routingOnly
          ? 'Routing applied & saved'
          : 'Template applied & saved'
        : routingOnly
          ? 'Routing applied'
          : 'Template applied',
      detail: routingOnly
        ? 'Alias routing and request aliases were loaded from the template.'
        : data.template_name
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
  draftRestored.value = false
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

async function saveFromReview() {
  const ok = await saveConfig()
  if (ok) changeReviewVisible.value = false
}

function discardFromReview() {
  resetConfig()
  changeReviewVisible.value = false
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
      detail: 'Saved. Use Apply settings when you want the model to use these changes.',
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
  draftRestored.value = false
  finishLeave(true)
}

async function saveAndLeave() {
  const ok = await saveConfig()
  finishLeave(Boolean(ok))
}

function persistDraftNow() {
  const modelId = String(model.value?.id || route.params.id || '')
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
  draftRestored.value = true
}

function discardDraftOffer() {
  clearDraft(route.params.id, config.value?.engine)
  draftOffer.value = null
  draftRestored.value = false
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
  draftRestored.value = false
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
    if (!paramRegistryFlag(engine, 'refreshOnChange')) return
    if (prevEngine && !paramRegistryFlag(prevEngine, 'refreshOnChange')) {
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

/* The save bar is sticky under the app header; keep a scrolled-to section
   heading below both instead of behind them. */
.model-config-view .config-tab-panel {
  scroll-margin-top: calc(var(--app-header-height) + 5rem);
}

@media (max-width: 900px) {
  .model-config-view .config-tab-panel {
    scroll-margin-top: calc(var(--app-header-height) + 13rem);
  }
}

/* The shared section header switches to a column at 640 px; its
   `flex: 1 1 14rem` heading group would then reserve 14rem of height. Keep
   the heading at its content height on this page until the shared pattern
   carries the column-safe basis. */
@media (max-width: 640px) {
  .model-config-view :deep(.page-section__heading-group) {
    flex: 0 0 auto;
  }
}

.textarea-cli {
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
  font-size: 0.875rem;
}

.cmd-preview-env-label {
  margin-top: 0.75rem;
  margin-bottom: 0.35rem;
  font-size: 0.85rem;
}

.cmd-preview-dialog-hint {
  margin: 0 0 1rem;
  font-size: 0.875rem;
  color: var(--text-secondary);
  line-height: 1.45;
}

.cmd-preview-dialog .cmd-preview-textarea {
  max-height: 40vh;
  overflow-y: auto;
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

.workbench-block {
  margin-top: 0.75rem;
  padding-top: 0.75rem;
  border-top: 1px solid var(--border-primary);
}

.workbench-block:first-child {
  margin-top: 0;
  padding-top: 0;
  border-top: none;
}

.section-hint {
  font-weight: 400;
  text-transform: none;
  letter-spacing: normal;
  color: var(--text-secondary);
  opacity: 0.7;
}

.apply-impact__lead {
  margin: 0 0 0.85rem;
  line-height: 1.5;
}

.apply-diff {
  display: grid;
  gap: 0.45rem;
  margin: 0.75rem 0 0;
  padding: 0;
  list-style: none;
  font-size: 0.875rem;
}

.apply-diff__row {
  display: grid;
  gap: 0.35rem;
  padding: 0.6rem 0.7rem;
  border: 1px solid var(--border-primary);
  border-radius: var(--radius-md);
  overflow-wrap: anywhere;
}

.apply-diff__field {
  font-weight: 600;
}

.apply-diff__values {
  display: flex;
  align-items: baseline;
  flex-wrap: wrap;
  gap: 0.5rem;
}

.apply-diff__values > span {
  display: grid;
  gap: 0.1rem;
}

.apply-diff__values small,
.apply-diff__note {
  color: var(--text-secondary);
  font-size: 0.68rem;
}

.apply-diff__value {
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
  font-size: 0.78rem;
}

.apply-diff__arrow {
  color: var(--text-secondary);
  font-size: 0.7rem;
}

.config-muted-hint {
  margin: 0;
  font-size: 0.875rem;
  line-height: 1.55;
  color: var(--text-secondary);
}

.config-templates-layout {
  display: grid;
  grid-template-columns: minmax(0, 1.65fr) minmax(16rem, 0.9fr);
  gap: 1.1rem;
  align-items: start;
}

.config-templates-library,
.config-templates-save {
  min-width: 0;
  padding: 0.9rem;
  border: 1px solid var(--border-primary);
  border-radius: var(--radius-md);
  background: var(--bg-surface);
}

.config-templates-library-head {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 1rem;
  margin-bottom: 0.75rem;
}

.config-templates-library-head h3,
.config-templates-save h3 {
  margin: 0;
  font-size: 1rem;
}

.config-templates-library-head p,
.config-templates-save__hint {
  margin: 0.25rem 0 0;
  color: var(--text-secondary);
  font-size: var(--text-xs);
  line-height: 1.45;
}

.config-templates-search {
  display: flex;
  flex: 0 1 15rem;
  align-items: center;
  gap: 0.35rem;
  padding: 0 0.5rem;
  border: 1px solid var(--border-primary);
  border-radius: 999px;
}

.config-templates-search i {
  color: var(--text-secondary);
  font-size: 0.75rem;
}

.config-templates-search input {
  width: 100%;
  border: 0;
  background: transparent;
  color: var(--text-primary);
}

.config-templates-apply-mode {
  display: grid;
  grid-template-columns: auto minmax(10rem, 1fr);
  align-items: center;
  gap: 0.5rem;
  margin-bottom: 0.75rem;
  font-size: var(--text-xs);
}

.config-templates-apply-mode label {
  color: var(--text-secondary);
}

.config-templates-cards {
  display: grid;
  gap: 0.55rem;
  margin: 0;
  padding: 0;
  list-style: none;
}

.config-template-card {
  display: grid;
  gap: 0.55rem;
  padding: 0.7rem;
  border: 1px solid var(--border-primary);
  border-radius: var(--radius-md);
  background: var(--bg-card);
}

.config-template-card__body {
  display: grid;
  gap: 0.3rem;
  min-width: 0;
}

.config-template-card__heading {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 0.75rem;
}

.config-template-card__heading small {
  color: var(--text-secondary);
  font-size: 0.68rem;
  white-space: nowrap;
}

.config-template-card__desc {
  margin: 0;
  color: var(--text-secondary);
  font-size: var(--text-xs);
  line-height: 1.4;
}

.config-template-card__tags {
  display: flex;
  flex-wrap: wrap;
  gap: 0.3rem;
}

.config-template-card__actions {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 0.3rem;
}

.config-template-card__edit {
  display: grid;
  gap: 0.4rem;
}

.config-templates-save {
  display: grid;
  gap: 0.7rem;
}

.config-templates-field {
  margin-bottom: 0;
}

.config-templates-field label {
  display: block;
  font-size: 0.8rem;
  margin-bottom: 0.35rem;
  color: var(--text-secondary);
}

.config-templates-check {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  font-size: 0.875rem;
}

@media (max-width: 880px) {
  .config-templates-layout {
    grid-template-columns: 1fr;
  }
}

@media (max-width: 768px) {
  .config-templates-library-head {
    flex-direction: column;
  }

  .config-templates-search {
    width: 100%;
    flex-basis: auto;
  }
}
</style>
