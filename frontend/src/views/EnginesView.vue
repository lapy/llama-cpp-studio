<template>
  <div class="engines-view page-shell page-shell--relaxed page-shell--wide">
    <PageHeader
      title="Engines"
      description="Install a runtime, activate a version, and choose compatible models."
    />

    <!-- ── Engines Overview ───────────────────────────────── -->
    <section class="ev-section engine-catalog">
      <div class="ev-section-header">
        <button
          type="button"
          class="ev-section-header__toggle interactive-row"
          :aria-expanded="enginesExpanded"
          aria-controls="ev-section-engines-body"
          @click="enginesExpanded = !enginesExpanded"
        >
          <div class="ev-section-title">
            <i class="pi pi-server" aria-hidden="true" />
            <h2>Runtime engines</h2>
          </div>
          <i
            :class="[
              'pi',
              'ev-section-chevron',
              enginesExpanded ? 'pi-chevron-up' : 'pi-chevron-down',
            ]"
            aria-hidden="true"
          />
        </button>
        <div class="ev-section-actions">
          <Button
            icon="pi pi-refresh"
            text
            severity="secondary"
            size="small"
            aria-label="Refresh engines"
            @click="refreshEnginesOverview"
          />
        </div>
      </div>
      <Transition name="ev-collapse">
        <div v-if="enginesExpanded" id="ev-section-engines-body" class="ev-section-body">
          <div class="engine-filters">
            <label for="engine-task-filter"
              >Task
              <select id="engine-task-filter" v-model="engineTaskFilter">
                <option value="all">All tasks</option>
                <option value="text">Text</option>
                <option value="embeddings">Embeddings</option>
                <option value="audio">Audio</option>
              </select>
            </label>
            <label for="engine-hardware-filter"
              >Hardware
              <select id="engine-hardware-filter" v-model="engineHardwareFilter">
                <option value="all">Any hardware</option>
                <option value="cpu">CPU-capable</option>
                <option value="sm70">SM70 / V100</option>
              </select>
            </label>
            <button type="button" class="engine-filters__reset" @click="resetEngineFilters">
              Reset filters
            </button>
          </div>
          <div class="engine-grid">
            <EngineCard
              v-for="card in engineSummaryCards"
              :key="card.id"
              :style="{ order: presentation(card.id).order }"
              v-show="presentation(card.id).visible"
              :disabled="card.disabled"
              v-tooltip.top="card.tooltip"
              @open="card.disabled ? undefined : openEngineModal(card.id)"
              :title="card.title"
              :count="card.count"
              :cta="presentation(card.id).cta"
            >
              <template #mark>
                <span
                  v-if="card.mark.text"
                  :class="['engine-mark', card.mark.className]"
                  aria-hidden="true"
                  >{{ card.mark.text }}</span
                >
                <i v-else :class="['engine-card-icon', card.mark.icon]" />
              </template>
              <template v-if="card.badge" #badge>
                <Tag :value="card.badge.value" :severity="card.badge.severity" />
              </template>
              <div
                class="engine-card-version-line"
                :title="card.active ? card.active.version : undefined"
              >
                <Tag
                  v-if="card.active"
                  :value="engineVersionDisplay(card.active.version)"
                  severity="success"
                  class="engine-version-tag"
                />
                <EngineStatusTag v-else :versions="card.versions" :engine-id="card.id" />
              </div>
              <div
                v-if="card.update?.update_available"
                class="engine-card-status engine-card-status--warning"
              >
                Update available: {{ card.updateText }}
              </div>
              <div v-else class="engine-card-status">{{ card.subtitle }}</div>
            </EngineCard>

            <!-- YAML campaign S6: dynamic cards for SERVER-BACKED
                 (YAML-onboarded) engines — payload-driven, zero view edits
                 to onboard an engine, no entries in the client registry. -->
            <GenericEnginePanel
              v-for="descriptor in serverBackedEngineDescriptors"
              :key="descriptor.id"
              :descriptor="descriptor"
            />
          </div>
        </div>
      </Transition>
    </section>

    <Dialog v-model:visible="engineDialogVisible" modal maximizable class="dialog-width-lg">
      <template #header>
        <EngineDialogHeader v-if="nativeDialog" :title="nativeDialog.title">
          <template #leading>
            <span :class="['engine-mark', nativeDialog.markClass]" aria-hidden="true">{{
              nativeDialog.markText
            }}</span>
          </template>
          <template #tags>
            <span
              class="engine-dialog-tag-clip"
              :title="nativeDialog.activeVersion ? nativeDialog.activeVersion.version : undefined"
            >
              <Tag
                v-if="nativeDialog.activeVersion"
                :value="engineVersionDisplay(nativeDialog.activeVersion.version)"
                severity="success"
                class="engine-version-tag"
              />
              <EngineStatusTag
                v-else
                :versions="nativeDialog.versions"
                :engine-id="nativeDialog.statusEngineId"
              />
            </span>
          </template>
          <template v-if="nativeDialog.settings" #actions>
            <Button
              icon="pi pi-cog"
              text
              severity="secondary"
              size="small"
              aria-label="Build settings"
              v-tooltip.top="'Build settings'"
              @click="nativeDialog.onOpenSettings()"
            />
          </template>
          <template #more>
            <Button
              icon="pi pi-refresh"
              label="Reload versions"
              text
              severity="secondary"
              size="small"
              aria-label="Reload versions"
              v-tooltip.top="'Reload versions'"
              @click="enginesStore.fetchLlamaVersions()"
            />
            <Button
              icon="pi pi-book"
              label="Rescan CLI parameters"
              text
              severity="secondary"
              size="small"
              aria-label="Rescan CLI parameters"
              v-tooltip.top="'Rescan CLI parameters (--help)'"
              :loading="paramScanLoading === nativeDialog.engineId"
              @click="rescanEngineCliParams(nativeDialog.engineId)"
            />
          </template>
        </EngineDialogHeader>
        <EngineDialogHeader v-else-if="venvDialogHeader" :title="venvDialogHeader.title">
          <template #leading>
            <i :class="`pi ${venvDialogHeader.icon}`" aria-hidden="true" />
          </template>
          <template #tags>
            <span
              class="engine-dialog-tag-clip"
              :title="
                venvDialogHeader.activeVersion ? venvDialogHeader.activeVersion.version : undefined
              "
            >
              <Tag
                v-if="venvDialogHeader.activeVersion"
                :value="engineVersionDisplay(venvDialogHeader.activeVersion.version)"
                severity="success"
                class="engine-version-tag"
              />
              <EngineStatusTag
                v-else
                :versions="venvDialogHeader.versions"
                :engine-id="venvDialogHeader.engineId"
              />
            </span>
          </template>
          <template #actions>
            <Button
              icon="pi pi-cog"
              text
              severity="secondary"
              size="small"
              :aria-label="venvDialogHeader.settingsAriaLabel"
              v-tooltip.top="venvDialogHeader.settingsAriaLabel"
              @click="venvDialogHeader.onOpenSettings()"
            />
          </template>
          <template #more>
            <Button
              icon="pi pi-refresh"
              label="Reload versions and status"
              text
              severity="secondary"
              size="small"
              :aria-label="venvDialogHeader.buttonA11y ? 'Reload versions and status' : undefined"
              v-tooltip.top="venvDialogHeader.buttonA11y ? 'Reload versions and status' : undefined"
              @click="venvDialogHeader.onRefresh()"
            />
            <Button
              icon="pi pi-book"
              label="Rescan CLI parameters"
              text
              severity="secondary"
              size="small"
              :aria-label="venvDialogHeader.buttonA11y ? 'Rescan CLI parameters' : undefined"
              v-tooltip.top="
                venvDialogHeader.buttonA11y ? 'Rescan CLI parameters (--help)' : undefined
              "
              :loading="paramScanLoading === venvDialogHeader.engineId"
              @click="rescanEngineCliParams(venvDialogHeader.engineId)"
            />
          </template>
        </EngineDialogHeader>
        <EngineDialogHeader
          v-else-if="ENGINE_DESCRIPTORS[selectedEngine]?.configPanel === 'audio'"
          title="audio.cpp"
        >
          <template #leading>
            <span class="engine-mark engine-mark--audio" aria-hidden="true">A</span>
          </template>
          <template #tags>
            <span
              class="engine-dialog-tag-clip"
              :title="activeAudioCpp ? activeAudioCpp.version : undefined"
            >
              <Tag
                v-if="activeAudioCpp"
                :value="engineVersionDisplay(activeAudioCpp.version)"
                severity="success"
                class="engine-version-tag"
              />
              <EngineStatusTag
                v-else
                :versions="enginesStore.audioCppVersions"
                engine-id="audio.cpp"
              />
            </span>
          </template>
          <template #actions>
            <Button
              icon="pi pi-cog"
              text
              severity="secondary"
              size="small"
              aria-label="Build settings"
              v-tooltip.top="'Build settings'"
              @click="openAudioCppBuildSettings"
            />
          </template>
          <template #more>
            <Button
              icon="pi pi-refresh"
              label="Reload versions and status"
              text
              severity="secondary"
              size="small"
              aria-label="Reload versions and status"
              v-tooltip.top="'Reload versions and status'"
              @click="refreshEngineDialogStatus(audioEngineId)"
            />
            <Button
              icon="pi pi-book"
              label="Rescan audio.cpp capabilities"
              text
              severity="secondary"
              size="small"
              aria-label="Rescan audio.cpp capabilities"
              v-tooltip.top="'Rescan audio.cpp capabilities'"
              :loading="paramScanLoading === audioEngineId"
              @click="rescanEngineCliParams(audioEngineId)"
            />
          </template>
        </EngineDialogHeader>
      </template>
      <section v-if="nativeDialog" class="ev-section ev-section--modal">
        <div class="ev-section-body engine-modal-body">
          <EngineBuildSettingsHint
            v-if="nativeDialog.hintEngineKey"
            :key="nativeDialog.hintKey"
            :engine-key="nativeDialog.hintEngineKey"
            @open-settings="nativeDialog.onOpenSettings()"
          />
          <p v-if="nativeDialog.noteRaw" class="build-note build-note--info">
            {{ nativeDialog.noteRaw }}
          </p>
          <EngineCheckUpdatesCta
            :loading="nativeDialog.checking"
            :hint="nativeDialog.checkHint"
            @check="nativeDialog.onCheck()"
          />
          <EngineUpdateBanner
            :available="!!nativeDialog.updateInfo?.update_available"
            :checked="!!nativeDialog.updateInfo"
            :latest-version="nativeDialog.updateInfo?.latest_version"
            :current-version="nativeDialog.updateInfo?.current_version"
            :link-url="nativeDialog.updateInfo?.release_url"
            :link-label="nativeDialog.updateLinkLabel"
            :updating="nativeDialog.updating"
            :update-tooltip="nativeDialog.updateTooltip"
            @update="nativeDialog.onUpdate()"
          />
          <EngineInstallPanel :subtitle="nativeDialog.installSubtitle">
            <Button
              v-for="btn in nativeDialog.installButtons"
              :key="btn.label"
              :label="btn.label"
              :icon="btn.icon"
              :severity="btn.severity"
              outlined
              :loading="btn.loading"
              :disabled="btn.disabled"
              @click="btn.onClick()"
            />
          </EngineInstallPanel>
          <EngineActiveStatus :rows="nativeDialog.statusRows" />
          <EngineVersionsBlock>
            <VersionTable
              :versions="nativeDialog.versions"
              :activating="activating"
              :activating-message="activatingMessage"
              :syncing="syncingVersion"
              :retrying="retryingVersion"
              :deleting="deletingVersion"
              :can-verify="nativeDialog.canVerify"
              :empty-message="nativeDialog.emptyMessage"
              @activate="activateVersion"
              @sync="syncVersion"
              @retry="retryVersion"
              @edit-config="openVersionBuildConfig"
              @verify="verifyEngineVersion"
              @delete="confirmDeleteVersion"
            />
          </EngineVersionsBlock>
        </div>
      </section>

      <section v-else-if="venvDialog" class="ev-section ev-section--modal">
        <div class="ev-section-body engine-modal-body">
          <EngineBuildSettingsHint
            :key="venvDialog.hintKey"
            :engine-key="venvDialog.engineId"
            @open-settings="venvDialog.onOpenSettings()"
          />
          <EngineCheckUpdatesCta
            :loading="venvDialog.checking"
            :hint="venvDialog.checkHint"
            @check="venvDialog.onCheck()"
          />
          <EngineUpdateBanner
            :available="!!venvDialog.updateInfo?.update_available"
            :checked="!!venvDialog.updateInfo"
            :latest-version="
              venvDialog.updateInfo?.latest_version
                ? `v${venvDialog.updateInfo.latest_version}`
                : ''
            "
            :current-version="
              venvDialog.updateInfo?.current_version
                ? `v${venvDialog.updateInfo.current_version}`
                : 'none'
            "
            :link-url="venvDialog.updateLinkUrl"
            :link-label="venvDialog.updateLinkLabel"
            :updating="venvDialog.updating"
            :update-tooltip="venvDialog.updateTooltip"
            @update="venvDialog.onUpdate()"
          />
          <EngineNote v-if="venvDialog.note">{{ venvDialog.note }}</EngineNote>
          <EngineInstallPanel :subtitle="venvDialog.installSubtitle">
            <Button
              :label="venvDialog.installPrimaryLabel"
              icon="pi pi-download"
              severity="success"
              outlined
              @click="venvDialog.onInstallPrimary()"
            />
            <Button
              label="From source"
              icon="pi pi-code"
              severity="info"
              outlined
              @click="venvDialog.onInstallSource()"
            />
          </EngineInstallPanel>
          <EngineActiveStatus :rows="venvDialog.statusRows" />
          <EngineVersionsBlock>
            <VersionTable
              :versions="venvDialog.versions"
              :activating="activating"
              :activating-message="activatingMessage"
              :syncing="syncingVersion"
              :retrying="retryingVersion"
              :deleting="deletingVersion"
              empty-message="No versions yet. Install one using the options above."
              @activate="activateVersion"
              @sync="syncVersion"
              @retry="retryVersion"
              @edit-config="openVersionBuildConfig"
              @delete="confirmDeleteVersion"
            />
          </EngineVersionsBlock>
        </div>
      </section>

      <SglangEnginePanel
        v-else-if="sglangPanelEngineId"
        :key="sglangPanelEngineId"
        :ref="(el) => setSglangPanelRef(sglangPanelEngineId, el)"
        :engine-id="sglangPanelEngineId"
      />

      <section
        v-else-if="ENGINE_DESCRIPTORS[selectedEngine]?.configPanel === 'audio'"
        class="ev-section ev-section--modal"
      >
        <div class="ev-section-body engine-modal-body">
          <EngineBuildSettingsHint
            :key="`audio-hint-${hintRevAudio}`"
            :engine-key="audioEngineId"
            @open-settings="openAudioCppBuildSettings"
          />
          <EngineCheckUpdatesCta :loading="checkingAudioCpp" @check="checkAudioCppUpdates" />
          <EngineUpdateBanner
            :available="!!audioCppUpdateInfo?.update_available"
            :checked="!!audioCppUpdateInfo"
            :latest-version="formatEngineUpdateVersion(audioCppUpdateInfo?.latest_version)"
            :current-version="
              audioCppUpdateInfo?.tracking_ref || audioCppStatus?.tracking_ref || ''
            "
            :link-url="
              audioCppUpdateInfo?.latest_release?.html_url ||
              audioCppUpdateInfo?.latest_commit?.html_url ||
              ''
            "
            :link-label="
              audioCppUpdateInfo?.latest_release?.html_url ? 'View release' : 'View commit'
            "
            :updating="audioCppUpdating"
            :update-tooltip="audioCppUpdateTooltip"
            @update="updateAudioCpp"
          >
            <template #message>
              <template v-if="audioCppUpdateInfo?.latest_release?.tag_name">
                Update available (release
                <strong>{{ formatEngineUpdateVersion(audioCppUpdateInfo.latest_version) }}</strong
                >)
              </template>
              <template v-else>
                Update available on
                <strong>{{
                  audioCppUpdateInfo?.tracking_ref || audioCppStatus?.tracking_ref || 'tracked ref'
                }}</strong
                >:
                <strong>{{ formatEngineUpdateVersion(audioCppUpdateInfo?.latest_version) }}</strong>
              </template>
            </template>
          </EngineUpdateBanner>

          <EngineInstallPanel
            subtitle="Add a new build from the latest release or any git repo. Each build is a version you can activate."
          >
            <Button
              label="From release"
              icon="pi pi-tag"
              severity="success"
              outlined
              :loading="audioCppReleaseInstalling"
              :disabled="audioCppReleaseInstalling || audioCppSourceInstalling"
              @click="installAudioLatestRelease"
            />
            <Button
              label="From source"
              icon="pi pi-code"
              severity="info"
              outlined
              :loading="audioCppSourceInstalling"
              :disabled="audioCppReleaseInstalling || audioCppSourceInstalling"
              @click="openAudioCppSourceDialog"
            />
          </EngineInstallPanel>
          <EngineActiveStatus :rows="audioCppActiveStatusRows" />
          <EngineVersionsBlock>
            <VersionTable
              :versions="enginesStore.audioCppVersions"
              :activating="activating"
              :activating-message="activatingMessage"
              :syncing="syncingVersion"
              :retrying="retryingVersion"
              :deleting="deletingVersion"
              empty-message="No versions yet. Install one using the options above."
              @activate="activateVersion"
              @sync="syncVersion"
              @retry="retryVersion"
              @edit-config="openVersionBuildConfig"
              @delete="confirmDeleteVersion"
            />
          </EngineVersionsBlock>
        </div>
      </section>
    </Dialog>

    <!-- ── LMDeploy Build Settings Dialog ─────────────────── -->
    <EngineInstallSettingsDialog
      v-model:visible="lmdeployBuildDialogVisible"
      title="Build settings — LMDeploy"
      note="Saved defaults for PyPI and source installs. Use Save settings to store without installing, or Install from source to build now."
      version-key="pip_version"
      version-label="Default PyPI version"
      version-hint="Used by From PyPI when the version field is left blank."
      repo-placeholder="https://github.com/InternLM/lmdeploy.git"
      :form="lmdeployBuildForm"
      :compatibility="lmdeployCompatibility"
      :notice="documentSaveNotice"
      :saving="savingLmdeployBuildSettings"
      :save-disabled="documentSaveHeld"
      :install-loading="lmdeployInstalling"
      install-label="Install from source"
      @field="(key, value) => (lmdeployBuildForm[key] = value)"
      @retry="retryDocumentSave"
      @refresh="refreshDocumentSave"
      @save="saveLmdeployBuildSettingsOnly"
      @install="installLmdeployFromBuildSettings"
    />

    <EngineInstallSettingsDialog
      v-model:visible="onecatVllmBuildDialogVisible"
      title="Build settings — 1Cat-vLLM"
      note="Saved defaults for release wheels and source builds. Use Save settings to store without installing, or Build from source to compile now."
      version-key="release_version"
      version-label="Default release version"
      version-hint="Used by From release when the version field is left blank."
      repo-placeholder="https://github.com/1CatAI/1Cat-vLLM.git"
      extra-note="Source builds compile SM70 CUDA kernels and require an NVIDIA GPU plus the CUDA 12.8 toolkit."
      :form="onecatVllmBuildForm"
      :notice="documentSaveNotice"
      :saving="savingOnecatVllmBuildSettings"
      :save-disabled="documentSaveHeld"
      :install-loading="onecatVllmInstalling"
      install-label="Build from source"
      @field="(key, value) => (onecatVllmBuildForm[key] = value)"
      @retry="retryDocumentSave"
      @refresh="refreshDocumentSave"
      @save="saveOnecatVllmBuildSettingsOnly"
      @install="installOnecatVllmFromBuildSettings"
    />

    <EngineBuildOptionsDialog
      v-model:visible="audioCppBuildDialogVisible"
      :header="audioCppBuildDialogHeader"
      :form="audioCppBuildForm"
      config-key="build_config"
      :fields="audioBuildFields"
      :categories="audioBuildDialogCategories"
      :build-types="audioCppBuildTypeOptions"
      :notice="documentSaveNotice"
      :options-loading="audioCppOptionsLoading"
      :frozen="Boolean(editingAudioVersion)"
      :frozen-version="String(editingAudioVersion?.version || '')"
      :actions="audioBuildDialogActions"
      @field="applyAudioBuildField"
      @option="applyAudioBuildOption"
      @retry="retryDocumentSave"
      @refresh="refreshDocumentSave"
      @save="saveAudioBuildSettingsFromDialog"
      @rebuild="saveAudioVersionBuildConfigAndRebuild"
      @build="buildAudioCpp"
    />

    <!-- ── audio.cpp Install from Source Dialog ───────────── -->
    <Dialog
      v-model:visible="audioCppSourceDialogVisible"
      header="Build audio.cpp from source"
      modal
      class="dialog-width-md"
    >
      <div class="dialog-body">
        <div class="form-field">
          <label>Repo URL</label>
          <InputText
            v-model="audioCppSourceRepo"
            placeholder="https://github.com/0xShug0/audio.cpp.git"
            class="w-full"
          />
          <small>Official repo or any fork with the same layout.</small>
        </div>
        <div class="form-field">
          <label>Tag / branch / commit</label>
          <InputText v-model="audioCppSourceRef" placeholder="main" class="w-full" />
          <small
            >Checked out before CMake build. Uses your saved build settings (gear in the
            header).</small
          >
        </div>
      </div>
      <template #footer>
        <Button
          label="Cancel"
          severity="secondary"
          outlined
          @click="audioCppSourceDialogVisible = false"
        />
        <Button
          label="Build from source"
          icon="pi pi-code"
          severity="info"
          :loading="audioCppSourceInstalling"
          :disabled="audioCppSourceInstalling"
          @click="installAudioCppFromSource"
        />
      </template>
    </Dialog>

    <Dialog
      v-model:visible="verifyDialogVisible"
      header="Verify engine binaries"
      modal
      class="dialog-width-sm"
    >
      <div class="dialog-body">
        <p v-if="verifyLoading">Checking llama-server, llama-cli, and llama-quantize…</p>
        <p v-else-if="verifyError" role="alert">{{ verifyError }}</p>
        <template v-else-if="verifyResult">
          <p>
            {{
              verifyResult.all_available
                ? 'All required commands are present.'
                : 'Some required commands are missing.'
            }}
          </p>
          <ul class="verify-commands">
            <li v-for="(ok, name) in verifyResult.verification || {}" :key="name">
              <span>{{ name }}: {{ ok ? 'present' : 'missing' }}</span>
              <code v-if="verifyResult.commands?.[name]">{{ verifyResult.commands[name] }}</code>
            </li>
          </ul>
        </template>
      </div>
      <template #footer>
        <Button label="Close" severity="secondary" outlined @click="verifyDialogVisible = false" />
      </template>
    </Dialog>

    <!-- ── Build Settings Dialog ─────────────────────────── -->
    <EngineBuildOptionsDialog
      v-model:visible="buildDialogVisible"
      :header="llamaBuildDialogHeader"
      :form="buildForm"
      config-key="buildConfig"
      :fields="llamaBuildFields"
      :categories="llamaBuildDialogCategories"
      :build-types="buildTypeOptions"
      build-type-placeholder="Release"
      :notes="llamaBuildNotes"
      :notice="documentSaveNotice"
      :options-loading="buildOptionsLoading"
      :frozen="Boolean(editingVersion)"
      :frozen-version="String(editingVersion?.version || '')"
      :actions="llamaBuildDialogActions"
      @field="applyLlamaBuildField"
      @option="applyLlamaBuildOption"
      @retry="retryDocumentSave"
      @refresh="refreshDocumentSave"
      @save="saveLlamaBuildSettingsFromDialog"
      @rebuild="saveLlamaVersionBuildConfigAndRebuild"
      @build="doStartBuild"
    />

    <!-- ── llama.cpp Install from Source Dialog ────────────── -->
    <Dialog
      v-model:visible="llamaCppSourceDialogVisible"
      header="Build llama.cpp from source"
      modal
      class="dialog-width-md"
    >
      <div class="dialog-body">
        <div class="form-field">
          <label>Repo URL</label>
          <InputText
            v-model="llamaCppSourceRepo"
            placeholder="https://github.com/ggerganov/llama.cpp.git"
            class="w-full"
          />
          <small>Official repo or any fork with the same layout.</small>
        </div>
        <div class="form-field">
          <label>Tag / branch / commit</label>
          <InputText
            v-model="llamaCppSourceRef"
            placeholder="vX.Y.Z, bNNNN, or master"
            class="w-full"
          />
          <small
            >Stable releases are vX.Y.Z. Nightly tags are still bNNNN (master commit count, not a
            SHA). A branch or commit SHA works too.</small
          >
        </div>
      </div>
      <template #footer>
        <Button
          label="Cancel"
          severity="secondary"
          outlined
          @click="llamaCppSourceDialogVisible = false"
        />
        <Button
          label="Build from source"
          icon="pi pi-code"
          severity="info"
          :loading="llamaCppSourceInstalling"
          :disabled="llamaCppSourceInstalling"
          @click="installLlamaCppFromSource"
        />
      </template>
    </Dialog>

    <!-- ── ik_llama.cpp Install from Source Dialog ─────────── -->
    <Dialog
      v-model:visible="ikLlamaSourceDialogVisible"
      header="Build ik_llama.cpp from source"
      modal
      class="dialog-width-md"
    >
      <div class="dialog-body">
        <div class="form-field">
          <label>Repo URL</label>
          <InputText
            v-model="ikLlamaSourceRepo"
            placeholder="https://github.com/ikawrakow/ik_llama.cpp.git"
            class="w-full"
          />
          <small>Official repo or any fork with the same layout.</small>
        </div>
        <div class="form-field">
          <label>Branch / commit</label>
          <InputText v-model="ikLlamaSourceRef" placeholder="main" class="w-full" />
          <small
            >Checked out before CMake build. Uses your saved build settings (gear in the header).
            Release tags are not used for ik_llama.cpp.</small
          >
        </div>
      </div>
      <template #footer>
        <Button
          label="Cancel"
          severity="secondary"
          outlined
          @click="ikLlamaSourceDialogVisible = false"
        />
        <Button
          label="Build from source"
          icon="pi pi-code"
          severity="info"
          :loading="ikLlamaSourceInstalling"
          :disabled="ikLlamaSourceInstalling"
          @click="installIkLlamaFromSource"
        />
      </template>
    </Dialog>

    <!-- ── LMDeploy Install from pip Dialog ───────────────── -->
    <Dialog
      v-model:visible="lmPipDialogVisible"
      header="Install LMDeploy from pip"
      modal
      class="dialog-width-sm"
    >
      <div class="dialog-body">
        <div class="form-field">
          <label>Version</label>
          <InputText v-model="lmdeployPipVersion" placeholder="Blank = latest" class="w-full" />
          <small>Leave blank to install the latest from PyPI.</small>
        </div>
      </div>
      <template #footer>
        <Button label="Cancel" severity="secondary" outlined @click="lmPipDialogVisible = false" />
        <Button
          label="Install"
          icon="pi pi-download"
          severity="success"
          :loading="lmdeployInstalling"
          :disabled="lmdeployInstalling"
          @click="installLmdeployPip"
        />
      </template>
    </Dialog>

    <!-- ── LMDeploy Install from Source Dialog ─────────────── -->
    <Dialog
      v-model:visible="lmSourceDialogVisible"
      header="Install LMDeploy from Source"
      modal
      class="dialog-width-md"
    >
      <div class="dialog-body">
        <div class="form-field">
          <label>Repo URL</label>
          <InputText
            v-model="lmSourceRepo"
            placeholder="https://github.com/InternLM/lmdeploy.git"
            class="w-full"
          />
        </div>
        <div class="form-field">
          <label>Branch</label>
          <InputText v-model="lmSourceBranch" placeholder="main" class="w-full" />
        </div>
      </div>
      <template #footer>
        <Button
          label="Cancel"
          severity="secondary"
          outlined
          @click="lmSourceDialogVisible = false"
        />
        <Button
          label="Install from Source"
          icon="pi pi-code"
          severity="info"
          :loading="lmdeployInstalling"
          :disabled="lmdeployInstalling"
          @click="installLmdeploySource"
        />
      </template>
    </Dialog>

    <!-- ── 1Cat-vLLM Install from Release Dialog ───────────── -->
    <Dialog
      v-model:visible="ovllmReleaseDialogVisible"
      header="Install 1Cat-vLLM from Release"
      modal
      class="dialog-width-sm"
    >
      <div class="dialog-body">
        <div class="form-field">
          <label>Release version</label>
          <InputText v-model="ovllmReleaseVersion" placeholder="Blank = latest" class="w-full" />
          <small
            >Leave blank to install the latest GitHub release. Downloads prebuilt CUDA 12.8
            wheels.</small
          >
        </div>
      </div>
      <template #footer>
        <Button
          label="Cancel"
          severity="secondary"
          outlined
          @click="ovllmReleaseDialogVisible = false"
        />
        <Button
          label="Install"
          icon="pi pi-download"
          severity="success"
          :loading="onecatVllmInstalling"
          :disabled="onecatVllmInstalling"
          @click="installOnecatVllmRelease"
        />
      </template>
    </Dialog>

    <!-- ── 1Cat-vLLM Install from Source Dialog ────────────── -->
    <Dialog
      v-model:visible="ovllmSourceDialogVisible"
      header="Build 1Cat-vLLM from Source"
      modal
      class="dialog-width-md"
    >
      <div class="dialog-body">
        <div class="form-field">
          <label>Repo URL</label>
          <InputText
            v-model="ovllmSourceRepo"
            placeholder="https://github.com/1CatAI/1Cat-vLLM.git"
            class="w-full"
          />
        </div>
        <div class="form-field">
          <label>Branch</label>
          <InputText v-model="ovllmSourceBranch" placeholder="main" class="w-full" />
        </div>
        <div class="form-field">
          <small
            >Source builds compile SM70 CUDA kernels and require an NVIDIA GPU plus the CUDA 12.8
            toolkit. This can take a long time.</small
          >
        </div>
      </div>
      <template #footer>
        <Button
          label="Cancel"
          severity="secondary"
          outlined
          @click="ovllmSourceDialogVisible = false"
        />
        <Button
          label="Build from Source"
          icon="pi pi-code"
          severity="info"
          :loading="onecatVllmInstalling"
          :disabled="onecatVllmInstalling"
          @click="installOnecatVllmSource"
        />
      </template>
    </Dialog>
  </div>
</template>

<script setup>
import EngineCard from '@/components/engines/EngineCard.vue'
import GenericEnginePanel from '@/components/engines/GenericEnginePanel.vue'
import { useQueryState } from '@/composables/useQueryState'
import { ref, computed, onMounted, onUnmounted, watch } from 'vue'
import { useConfirm } from 'primevue/useconfirm'
import { useToast } from 'primevue/usetoast'
import Button from 'primevue/button'
import Tag from 'primevue/tag'
import Dialog from 'primevue/dialog'
import InputText from 'primevue/inputtext'
import PageHeader from '@/components/common/PageHeader.vue'
import EngineDialogHeader from '@/components/system/EngineDialogHeader.vue'
import EngineCheckUpdatesCta from '@/components/system/EngineCheckUpdatesCta.vue'
import EngineBuildSettingsHint from '@/components/system/EngineBuildSettingsHint.vue'
import EngineBuildOptionsDialog from '@/components/system/EngineBuildOptionsDialog.vue'
import EngineInstallPanel from '@/components/system/EngineInstallPanel.vue'
import EngineInstallSettingsDialog from '@/components/system/EngineInstallSettingsDialog.vue'
import EngineUpdateBanner from '@/components/system/EngineUpdateBanner.vue'
import EngineActiveStatus from '@/components/system/EngineActiveStatus.vue'
import EngineVersionsBlock from '@/components/system/EngineVersionsBlock.vue'
import EngineNote from '@/components/system/EngineNote.vue'
import SglangEnginePanel from '@/components/system/SglangEnginePanel.vue'
import VersionTable from '@/components/system/VersionTable.vue'
import {
  noteDocumentSaveFailure,
  saveHeldForRefresh,
  classifyPersistenceError,
} from '@/composables/persistenceOutcome'
import { requireSingleConfirmation } from '@/composables/singleConfirm'
import { engineCardCta, engineCardOrder, engineMatchesFilters } from '@/composables/engineReadiness'
import { activeVersionDeletePlan } from '@/composables/engineVersionDelete'
import { lmdeployCompatibilityFor } from '@/utils/lmdeployCompatibility'
import {
  AUDIO_ENGINE_IDS,
  ENGINE_DESCRIPTORS,
  SG_PANEL_ENGINE_IDS,
  engineIdForManager,
  engineLabel,
  presentationForEngine,
  serverBackedEngines,
} from '@/engines/registry'
import { FALLBACK_BUILD_DEFAULTS, FALLBACK_BUILD_TYPES } from '@/engines/buildDefaults'
import { versionDeleteErrorText, versionDeleteRetry } from '@/composables/actionConfirmation'
import EngineStatusTag from '@/components/system/EngineStatusTag.vue'
import { useEnginesStore } from '@/stores/engines'
import { useProgressStore } from '@/stores/progress'

const enginesStore = useEnginesStore()

// R88: statuses live in the store's keyed map; the view keeps named computeds
// (template-friendly) instead of the removed per-engine status refs.
const audioEngineId = AUDIO_ENGINE_IDS[0] || ''
const audioCppStatus = computed(() => enginesStore.statusForEngine(audioEngineId))
const lmdeployStatus = computed(() => enginesStore.statusForEngine('lmdeploy'))
const onecatVllmStatus = computed(() => enginesStore.statusForEngine('1cat_vllm'))
const progressStore = useProgressStore()
const confirm = useConfirm()

// YAML campaign S6: the dynamic card section — purely payload-driven. Engines
// that already have a client descriptor render through their pinned card; a
// YAML-onboarded engine with no local entry gets this generic card + dialog.
const serverBackedEngineDescriptors = computed(() =>
  serverBackedEngines(enginesStore.engineDescriptors).filter(
    (descriptor) => !Object.hasOwn(ENGINE_DESCRIPTORS, descriptor.id),
  ),
)
const toast = useToast()
const documentSaveNotice = ref(null)
const documentSaveHeld = computed(() => saveHeldForRefresh(documentSaveNotice.value))

function onDocumentSaveError(error) {
  const outcome = noteDocumentSaveFailure(toast, error)
  if (outcome) documentSaveNotice.value = outcome
  return Boolean(outcome)
}

function retryDocumentSave() {
  if (lmdeployBuildDialogVisible.value) void saveLmdeployBuildSettingsOnly()
  else if (onecatVllmBuildDialogVisible.value) void saveOnecatVllmBuildSettingsOnly()
  else if (audioCppBuildDialogVisible.value && editingAudioVersion.value)
    void saveAudioVersionBuildConfigOnly()
  else if (audioCppBuildDialogVisible.value) void saveAudioCppBuildSettingsOnly()
  else if (buildDialogVisible.value && editingVersion.value) void saveLlamaVersionBuildConfigOnly()
  else if (buildDialogVisible.value) void saveBuildSettingsOnly()
}

async function refreshDocumentSave() {
  const draftNotice = documentSaveNotice.value
  try {
    if (lmdeployBuildDialogVisible.value) {
      await applyLmdeployBuildSettings(await enginesStore.fetchLmdeployBuildSettings())
    } else if (onecatVllmBuildDialogVisible.value) {
      await applyOnecatVllmBuildSettings(await enginesStore.fetchOnecatVllmBuildSettings())
    } else if (audioCppBuildDialogVisible.value && !editingAudioVersion.value) {
      const saved = await enginesStore.fetchAudioCppBuildSettings()
      const split = splitAudioCppSettings(saved)
      const base = _defaultAudioBuildConfig()
      audioCppBuildForm.value.repository_url = split.repository_url
      audioCppBuildForm.value.source_ref = split.tracking_ref || audioCppBuildForm.value.source_ref
      audioCppBuildForm.value.build_config = { ...base, ...split.build_config }
    } else if (buildDialogVisible.value && !editingVersion.value) {
      await openBuildDialog(buildTarget.value)
    } else {
      throw new Error('reload failed')
    }
    documentSaveNotice.value = null
  } catch {
    documentSaveNotice.value =
      draftNotice?.committed === true
        ? {
            ...draftNotice,
            detail:
              'The document was replaced, but it could not be reloaded. Your edits are still here. Refresh before trying again.',
            refresh: true,
            retry: false,
          }
        : classifyPersistenceError(new Error('reload failed'))
  }
}

// ── System metrics ─────────────────────────────────────────
const enginesExpanded = ref(true)

const engineDialogVisible = ref(false)
const selectedEngine = ref('llama_cpp')
const paramScanLoading = ref(null)

const sglangPanelRefs = {}
function setSglangPanelRef(engineId, element) {
  if (!engineId) return
  if (element) sglangPanelRefs[engineId] = element
  else delete sglangPanelRefs[engineId]
}
const sglangPanelEngineId = computed(() => {
  const engineId = selectedEngine.value
  return engineId && ENGINE_DESCRIPTORS[engineId]?.panelFamily === 'sglang' ? engineId : ''
})

async function rescanEngineCliParams(engine) {
  paramScanLoading.value = engine
  try {
    const data = await enginesStore.scanEngineParams(engine)
    if (data?.ok) {
      toast.add({
        severity: 'success',
        summary: 'CLI parameters scanned',
        detail: `Indexed ${data.param_count ?? 0} options for ${engine}. Open the progress card for the help scan log.`,
        life: 4500,
      })
    } else {
      toast.add({
        severity: 'warn',
        summary: 'Scan failed',
        detail: data?.scan_error || 'Unknown error',
        life: 6000,
      })
    }
  } catch (e) {
    toast.add({
      severity: 'error',
      summary: 'Scan failed',
      detail: e?.message || String(e),
      life: 5000,
    })
  } finally {
    paramScanLoading.value = null
  }
}

// Dialog-open probes keyed by engine. Declarative replacement for the old
// 7-branch if-chain: every family except llama.cpp and ik_llama refreshes
// the shared version list first (their branches skipped it — preserved),
// and the SGLang panel family is generated from the registry instead of
// spelled out. Pinned by EnginesView.tasks.test.js (dispatch + cross-absence).
function withVersionListRefresh(probe) {
  return () => {
    enginesStore.fetchLlamaVersions()
    probe()
  }
}

const ENGINE_OPEN_STRATEGIES = {
  llama: () => checkLlamaCppUpdates(),
  ik: () => checkIkLlamaUpdates(),
  unsloth: withVersionListRefresh(checkUnslothLlamaUpdates),
  lmdeploy: withVersionListRefresh(checkLmdeployUpdates),
  onecat: withVersionListRefresh(checkOnecatVllmUpdates),
  sglang: (engineId) => withVersionListRefresh(() => enginesStore.fetchSglangStatus(engineId)),
  audio: () => {
    enginesStore.fetchLlamaVersions()
    enginesStore.fetchAudioCppStatus()
    checkAudioCppUpdates()
  },
}

const ENGINE_DIALOG_OPEN = Object.fromEntries(
  Object.entries(ENGINE_DESCRIPTORS).map(([engineId, descriptor]) => [
    engineId,
    () => ENGINE_OPEN_STRATEGIES[descriptor.openStrategy]?.(engineId),
  ]),
)

const ENGINE_STATUS_REFRESH = {
  lmdeploy: () => enginesStore.fetchLmdeployStatus(),
  onecat: () => enginesStore.fetchOnecatVllmStatus(),
  sglang: (engineId) => enginesStore.fetchSglangStatus(engineId),
  audio: () => {
    enginesStore.fetchLlamaVersions()
    return enginesStore.fetchAudioCppStatus()
  },
}

function openEngineModal(engineKey) {
  selectedEngine.value = engineKey
  engineDialogVisible.value = true
  ENGINE_DIALOG_OPEN[engineKey]?.()
}

function refreshEngineDialogStatus(engineKey) {
  const strategy = ENGINE_DESCRIPTORS[engineKey]?.statusRefresh
  const handler = ENGINE_STATUS_REFRESH[strategy]
  if (handler) return handler(engineKey)
  return enginesStore.fetchLlamaVersions()
}

async function refreshEnginesOverview() {
  await Promise.allSettled([
    enginesStore.fetchEngineDescriptors(),
    enginesStore.fetchLlamaVersions(),
    enginesStore.fetchServerBackedStatuses(),
    enginesStore.fetchLmdeployStatus(),
    enginesStore.fetchOnecatVllmStatus(),
    enginesStore.fetchSglangStatus('sglang'),
    enginesStore.fetchSglangStatus('sglang_v100'),
    enginesStore.fetchSglangStatus('vllm'),
    enginesStore.fetchAudioCppStatus(),
    checkLlamaCppUpdates(),
    checkIkLlamaUpdates(),
    checkUnslothLlamaUpdates(),
    checkLmdeployUpdates(),
    checkOnecatVllmUpdates(),
    checkAudioCppUpdates(),
  ])
}

/** Short label for cards/headers; put full `version` in title/tooltip. */
function engineVersionDisplay(version) {
  if (version == null) return ''
  const s = String(version).trim()
  if (!s) return ''
  const iso = s.match(/^([\w.+~-]+)-(\d{4}-\d{2}-\d{2})T/)
  if (iso) {
    return `${iso[1]} · ${iso[2]}`
  }
  const max = 22
  if (s.length <= max) return s
  return `${s.slice(0, max - 1)}…`
}

// ── Active versions ────────────────────────────────────────
const activeLlamaCpp = computed(() => enginesStore.llamaVersions.find((v) => v.is_active) ?? null)
const activeIkLlama = computed(() => enginesStore.ikLlamaVersions.find((v) => v.is_active) ?? null)
const activeUnslothLlama = computed(
  () => (enginesStore.unslothLlamaVersions || []).find((v) => v.is_active) ?? null,
)
const activeLmdeploy = computed(
  () => enginesStore.lmdeployVersions.find((v) => v.is_active) ?? null,
)
const activeOnecatVllm = computed(
  () => enginesStore.onecatVllmVersions.find((v) => v.is_active) ?? null,
)
const activeSglang = computed(
  () => (enginesStore.sglangVersions || []).find((v) => v.is_active) ?? null,
)
const activeSglangV100 = computed(
  () => (enginesStore.sglangV100Versions || []).find((v) => v.is_active) ?? null,
)
const activeSglangSxm2 = computed(
  () => (enginesStore.sglangSxm2Versions || []).find((v) => v.is_active) ?? null,
)
const activeSglangSm70 = computed(
  () => (enginesStore.sglangSm70Versions || []).find((v) => v.is_active) ?? null,
)
const activeVllm = computed(
  () => (enginesStore.vllmVersions || []).find((v) => v.is_active) ?? null,
)
const selectedSglangActive = computed(() =>
  selectedEngine.value === 'sglang_v100'
    ? activeSglangV100.value
    : selectedEngine.value === 'sglang_sxm2'
      ? activeSglangSxm2.value
      : selectedEngine.value === 'sglang_sm70'
        ? activeSglangSm70.value
        : selectedEngine.value === 'vllm'
          ? activeVllm.value
          : activeSglang.value,
)
const selectedSglangVersions = computed(() =>
  selectedEngine.value === 'sglang_v100'
    ? enginesStore.sglangV100Versions || []
    : selectedEngine.value === 'sglang_sxm2'
      ? enginesStore.sglangSxm2Versions || []
      : selectedEngine.value === 'sglang_sm70'
        ? enginesStore.sglangSm70Versions || []
        : selectedEngine.value === 'vllm'
          ? enginesStore.vllmVersions || []
          : enginesStore.sglangVersions || [],
)
const activeAudioCpp = computed(
  () => enginesStore.audioCppVersions.find((v) => v.is_active) ?? null,
)
const engineTaskFilter = ref('all')
const engineHardwareFilter = ref('all')
useQueryState({
  task: {
    get: () => engineTaskFilter.value,
    set: (v) => {
      engineTaskFilter.value = v
    },
    parse: (v) => (['text', 'embeddings', 'audio'].includes(v) ? v : 'all'),
    default: 'all',
  },
  hardware: {
    get: () => engineHardwareFilter.value,
    set: (v) => {
      engineHardwareFilter.value = v
    },
    parse: (v) => (['cpu', 'sm70'].includes(v) ? v : 'all'),
    default: 'all',
  },
})

const engineActive = {
  llama_cpp: activeLlamaCpp,
  ik_llama: activeIkLlama,
  unsloth_llama: activeUnslothLlama,
  lmdeploy: activeLmdeploy,
  '1cat_vllm': activeOnecatVllm,
  sglang: activeSglang,
  sglang_v100: activeSglangV100,
  sglang_sxm2: activeSglangSxm2,
  sglang_sm70: activeSglangSm70,
  vllm: activeVllm,
  [audioEngineId]: activeAudioCpp,
}

function resetEngineFilters() {
  engineTaskFilter.value = 'all'
  engineHardwareFilter.value = 'all'
}

function versionsFor(engineId) {
  // R82: the descriptor declares which store list holds this engine's rows
  // (`versionsKey`); no per-engine map in the view.
  const key = ENGINE_DESCRIPTORS[engineId]?.versionsKey
  const list = key ? enginesStore[key] : undefined
  return Array.isArray(list) ? list : []
}

const ENGINE_CARD_ORDER = computed(() => Object.keys(ENGINE_DESCRIPTORS))

// Update-info refs are declared later in setup; resolve them lazily so the
// summary cards stay one descriptor-driven loop.
const ENGINE_UPDATE_SOURCES = {
  llamaCpp: () => llamaCppUpdateInfo.value,
  ikLlama: () => ikLlamaUpdateInfo.value,
  unslothLlama: () => unslothLlamaUpdateInfo.value,
  lmdeploy: () => lmdeployUpdateInfo.value,
  onecatVllm: () => onecatVllmUpdateInfo.value,
  audioCpp: () => audioCppUpdateInfo.value,
}

const ENGINE_UPDATE_FORMATTERS = {
  plain: (value) => String(value ?? ''),
  v: (value) => `v${value ?? ''}`,
  version: (value) => formatEngineUpdateVersion(value),
}

const engineSummaryCards = computed(() =>
  ENGINE_CARD_ORDER.value.map((engineId) => {
    const descriptor = presentationForEngine(engineId, enginesStore.engineDescriptors)
    const card = descriptor.card || {}
    const versions = versionsFor(engineId)
    const active = engineActive[engineId]?.value || null
    const update = card.update ? ENGINE_UPDATE_SOURCES[card.update.key]?.() || null : null
    const enabled = card.featureFlag ? featureEnabled(card.featureFlag) : true
    const disabled = !enabled
    const featureBadge = card.featureFlag ? featureBadgeFor(card.featureFlag) : null
    let badge = card.badge || null
    if (badge?.kind === 'maturity' && featureBadge) {
      badge = enabled
        ? { value: maturityTag(card.featureFlag, featureBadge), severity: 'success' }
        : { value: featureBadge.disabledLabel || 'Disabled', severity: 'secondary' }
    }
    return {
      id: engineId,
      title: descriptor.label,
      versions,
      count: versions.length,
      active,
      mark: card.mark || {},
      badge,
      subtitle: card.subtitle || '',
      update,
      updateText:
        update?.update_available && card.update
          ? ENGINE_UPDATE_FORMATTERS[card.update.formatter](update.latest_version)
          : '',
      disabled,
      tooltip: disabled
        ? featureBadge?.disabledTooltip
        : featureBadge
          ? maturityTooltip(card.featureFlag, featureBadge)
          : undefined,
    }
  }),
)

function presentation(engineId) {
  const descriptor = (enginesStore.engineDescriptors || []).find((item) => item.id === engineId)
  const versions = versionsFor(engineId)
  const active = Boolean(engineActive[engineId]?.value)
  const installed = (versions || []).length > 0
  const filters = { task: engineTaskFilter.value, hardware: engineHardwareFilter.value }
  return {
    order: descriptor ? engineCardOrder(descriptor) : active ? 0 : installed ? 1 : 2,
    cta: engineCardCta(descriptor, { installed, active }),
    visible: !descriptor || engineMatchesFilters(descriptor, filters),
  }
}

function cmakeBackendBadge(version) {
  const cfg = version?.build_config || {}
  if (cfg.cuda || cfg.backend === 'cuda' || cfg.enable_cuda) return 'CUDA'
  return ''
}

function cmakeActiveStatusRows(version) {
  if (!version) return []
  const displayType = version.type || version.install_type || 'source'
  const rows = [
    {
      label: 'Install type:',
      tag: displayType,
      tagSeverity: displayType === 'fork' || version.is_fork ? 'warning' : 'info',
    },
  ]
  if (version.binary_path) {
    rows.push({ label: 'Binary:', code: version.binary_path })
  }
  const sourceBits = [
    version.source_repo,
    version.source_ref || version.source_branch || version.source_commit,
  ].filter(Boolean)
  if (sourceBits.length) {
    rows.push({
      label: 'Source:',
      code: sourceBits.length === 2 ? `${sourceBits[0]} @ ${sourceBits[1]}` : sourceBits[0],
    })
  }
  const backend = cmakeBackendBadge(version)
  if (backend) {
    rows.push({ label: 'Backend:', tag: backend, tagSeverity: 'secondary' })
  }
  return rows
}

const llamaCppActiveStatusRows = computed(() => cmakeActiveStatusRows(activeLlamaCpp.value))
const ikLlamaActiveStatusRows = computed(() => cmakeActiveStatusRows(activeIkLlama.value))
const unslothLlamaActiveStatusRows = computed(() => {
  const rows = cmakeActiveStatusRows(activeUnslothLlama.value)
  const flavor = activeUnslothLlama.value?.build_config?.flavor
  if (flavor) {
    rows.push({ label: 'Asset:', tag: flavor, tagSeverity: 'secondary' })
  }
  return rows
})

const lmdeployActiveStatusRows = computed(() => {
  const status = lmdeployStatus.value
  const rows = []
  if (activeLmdeploy.value || status.venv_path) {
    const displayType =
      activeLmdeploy.value?.type ||
      activeLmdeploy.value?.install_type ||
      status.install_type ||
      'pip'
    rows.push({
      label: 'Install type:',
      tag: displayType,
      tagSeverity: displayType === 'fork' || activeLmdeploy.value?.is_fork ? 'warning' : 'info',
    })
  }
  if (status.venv_path) {
    rows.push({ label: 'Venv:', code: status.venv_path })
  }
  if (status.source_repo) {
    rows.push({
      label: 'Source:',
      code: `${status.source_repo}${status.source_branch ? ` (${status.source_branch})` : ''}`,
    })
  }
  if (status.last_error) {
    rows.push({ label: 'Last error:', code: status.last_error, error: true })
  }
  return rows
})

const onecatVllmActiveStatusRows = computed(() => {
  const status = onecatVllmStatus.value
  const rows = []
  if (activeOnecatVllm.value || status.venv_path) {
    const displayType =
      activeOnecatVllm.value?.type ||
      activeOnecatVllm.value?.install_type ||
      status.install_type ||
      'release'
    rows.push({
      label: 'Install type:',
      tag: displayType,
      tagSeverity: displayType === 'fork' || activeOnecatVllm.value?.is_fork ? 'warning' : 'info',
    })
  }
  if (status.venv_path) {
    rows.push({ label: 'Venv:', code: status.venv_path })
  }
  if (status.source_repo) {
    rows.push({
      label: 'Source:',
      code: `${status.source_repo}${status.source_branch ? ` (${status.source_branch})` : ''}`,
    })
  }
  if (status.last_error) {
    rows.push({ label: 'Last error:', code: status.last_error, error: true })
  }
  return rows
})

const audioCppActiveStatusRows = computed(() => {
  const active = audioCppStatus.value?.active
  if (!active) return []
  const rows = []
  if (active.install_type || active.type) {
    const displayType = active.type || active.install_type || 'source'
    rows.push({
      label: 'Install type:',
      tag: displayType,
      tagSeverity: displayType === 'fork' || active.is_fork ? 'warning' : 'info',
    })
  }
  const binary = active.server_binary_path || active.binary_path || active.cli_binary_path
  if (binary) {
    rows.push({ label: 'Binary:', code: binary })
  }
  if (active.source_repo) {
    rows.push({
      label: 'Source:',
      code: active.source_ref ? `${active.source_repo} @ ${active.source_ref}` : active.source_repo,
    })
  }
  const backend = cmakeBackendBadge(active)
  if (backend) {
    rows.push({ label: 'Backend:', tag: backend, tagSeverity: 'secondary' })
  }
  if (audioCppStatus.value?.models_root) {
    rows.push({
      label: 'Models:',
      code: audioCppStatus.value.models_root,
      tag: audioCppStatus.value.model_manager_ready
        ? 'Model manager ready'
        : 'Model manager unavailable',
      tagSeverity: audioCppStatus.value.model_manager_ready ? 'success' : 'warning',
      tagTooltip: audioCppStatus.value.model_manager_ready
        ? 'Prefers model_manager_v2.py (model_specs). Legacy manager remains for assemble/convert packages.'
        : 'Activate an audio.cpp build that includes model_manager_v2.py or a legacy model_manager*.py',
    })
  }
  return rows
})
function featureEnabled(featureFlag) {
  const descriptor = (enginesStore.engineDescriptors || []).find(
    (engine) => engine.id === featureFlag,
  )
  return descriptor?.enabled !== false
}

function featureBadgeFor(featureFlag) {
  return ENGINE_DESCRIPTORS[featureFlag]?.featureBadge || null
}

function maturitySurfacesFor(featureFlag) {
  const descriptor = (enginesStore.engineDescriptors || []).find(
    (engine) => engine.id === featureFlag,
  )
  return descriptor?.maturity_surfaces || {}
}

function maturityTag(featureFlag, featureBadge) {
  const surfaces = maturitySurfacesFor(featureFlag)
  for (const rule of featureBadge.tagRules || []) {
    if (surfaces[rule.surface] === rule.equals) return rule.label
  }
  return featureBadge.fallbackLabel || ''
}

function maturityTooltip(featureFlag, featureBadge) {
  const surfaces = maturitySurfacesFor(featureFlag)
  const parts = []
  for (const field of featureBadge.tooltipFields || []) {
    if (surfaces[field.surface]) parts.push(`${field.label}: ${surfaces[field.surface]}`)
  }
  return parts.length ? parts.join(' · ') : featureBadge.fallbackTooltip || ''
}

// ── Version activate / delete ──────────────────────────────
const activating = ref(null)
const activatingMessage = ref('')
const syncingVersion = ref(null)
const retryingVersion = ref(null)
const deletingVersion = ref(null)

async function activateVersion(versionId) {
  activating.value = versionId
  activatingMessage.value = ''
  try {
    await enginesStore.activateVersion(versionId, {
      onStage: (message) => {
        activatingMessage.value = message
      },
    })
    toast.add({ severity: 'success', summary: 'Version activated', life: 3000 })
  } catch (e) {
    toast.add({ severity: 'error', summary: 'Failed', detail: e.message, life: 4000 })
  } finally {
    activating.value = null
    activatingMessage.value = ''
  }
}

async function syncVersion(versionId) {
  syncingVersion.value = versionId
  try {
    await enginesStore.syncVersion(versionId)
    toast.add({
      severity: 'success',
      summary: 'Sync started',
      detail: 'Track progress in notifications',
      life: 3500,
    })
  } catch (e) {
    toast.add({
      severity: 'error',
      summary: 'Sync failed',
      detail: e?.response?.data?.detail || e.message,
      life: 5000,
    })
  } finally {
    syncingVersion.value = null
  }
}

async function retryVersion(versionId) {
  retryingVersion.value = versionId
  try {
    await enginesStore.retryVersion(versionId)
    toast.add({
      severity: 'success',
      summary: 'Retry started',
      detail: 'Track progress in notifications',
      life: 3500,
    })
  } catch (e) {
    toast.add({
      severity: 'error',
      summary: 'Retry failed',
      detail: e?.response?.data?.detail || e.message,
      life: 5000,
    })
  } finally {
    retryingVersion.value = null
  }
}

function findListedVersion(versionId) {
  const allVersions = [
    ...(enginesStore.llamaVersions || []),
    ...(enginesStore.ikLlamaVersions || []),
    ...(enginesStore.unslothLlamaVersions || []),
    ...(enginesStore.lmdeployVersions || []),
    ...(enginesStore.onecatVllmVersions || []),
    ...(enginesStore.sglangVersions || []),
    ...(enginesStore.sglangV100Versions || []),
    ...(enginesStore.sglangSxm2Versions || []),
    ...(enginesStore.sglangSm70Versions || []),
    ...(enginesStore.vllmVersions || []),
    ...(enginesStore.audioCppVersions || []),
  ]
  return allVersions.find((v) => (v.id ?? v.version) === versionId) || null
}

function versionLooksLikeCommitRef(value) {
  return /^[0-9a-f]{7,40}$/i.test(String(value || '').trim())
}

function versionLooksLikeReleaseTag(value) {
  return /^(?:v?\d+(?:\.\d+){1,}(?:[-+][0-9A-Za-z._-]+)?|b\d+)$/i.test(String(value || '').trim())
}

function listedVersionBranch(version) {
  const branch = String(version?.source_branch || '').trim()
  if (branch) return branch
  const ref = String(version?.source_ref || '').trim()
  const refType = String(version?.source_ref_type || '')
    .trim()
    .toLowerCase()
  if (refType === 'branch' && ref) return ref
  if (refType === 'commit' || refType === 'release') return ''
  if (ref && !versionLooksLikeCommitRef(ref) && !versionLooksLikeReleaseTag(ref)) return ref
  return ''
}

function listedVersionUnusable(version) {
  return ['building', 'failed', 'cancelled', 'broken'].includes(
    String(version?.build_status || '')
      .trim()
      .toLowerCase(),
  )
}

function versionRebuildAction(version) {
  if (!version || String(version.build_status || '').toLowerCase() === 'building') return ''
  if (
    version.retryable === true ||
    ['failed', 'cancelled', 'broken'].includes(
      String(version.build_status || '')
        .trim()
        .toLowerCase(),
    )
  ) {
    return 'retry'
  }
  const installType = String(version.install_type || version.type || '').toLowerCase()
  if (
    ['source', 'fork', 'patched', 'local'].includes(installType) &&
    listedVersionBranch(version) &&
    !listedVersionUnusable(version)
  ) {
    return 'sync'
  }
  return ''
}

async function performVersionDelete(versionId, params) {
  deletingVersion.value = versionId
  try {
    await enginesStore.deleteVersion(versionId, params)
    toast.add({ severity: 'info', summary: 'Version deleted', life: 3000 })
  } catch (error) {
    const retry = versionDeleteRetry(error)
    if (retry) {
      requireSingleConfirmation(confirm, {
        message: retry.message,
        header: 'Confirm Delete',
        icon: 'pi pi-exclamation-triangle',
        acceptClass: 'p-button-danger',
        accept: () => performVersionDelete(versionId, retry.params),
      })
      return
    }
    toast.add({
      severity: 'error',
      summary: 'Failed',
      detail: versionDeleteErrorText(error),
      life: 4000,
    })
  } finally {
    deletingVersion.value = null
  }
}

function confirmDeleteVersion(versionId) {
  const version = findListedVersion(versionId) || { id: versionId, version: versionId }
  const plan = activeVersionDeletePlan(version)
  if (!plan.allowed) {
    toast.add({
      severity: 'warn',
      summary: 'Engine is in use',
      detail: plan.message,
      life: 5000,
    })
    return
  }

  requireSingleConfirmation(confirm, {
    message: plan.message,
    header: 'Confirm Delete',
    icon: 'pi pi-exclamation-triangle',
    acceptClass: 'p-button-danger',
    accept: () => performVersionDelete(versionId),
  })
}

// ── Update check (shared normalizer for llama/ik_llama API shape) ─────────
function normalizeLlamaUpdateInfo(raw, currentVersion, commitUrlPrefix) {
  if (!raw?.latest_release && !raw?.latest_commit) return null
  const latestVersion = raw.latest_release?.tag_name || raw.latest_commit?.sha?.slice(0, 8) || null
  const releaseUrl =
    raw.latest_release?.html_url ||
    (raw.latest_commit ? `${commitUrlPrefix}/commit/${raw.latest_commit.sha}` : null)
  const current = currentVersion || 'none'
  const updateAvailable = latestVersion && current !== latestVersion
  return {
    update_available: updateAvailable,
    latest_version: latestVersion,
    release_url: releaseUrl,
    current_version: current,
    available_tags:
      raw.available_tags || (raw.latest_release?.tag_name ? [raw.latest_release.tag_name] : []),
  }
}

function currentComparableLlamaVersion(version) {
  if (!version) return null
  if (version.source_ref_type === 'branch') {
    return version.source_commit || version.source_ref || version.version
  }
  return version.source_ref || version.source_commit || version.version
}

const checkingLlamaCpp = ref(false)
const llamaCppUpdateInfo = ref(null)
const updatingEngine = ref(null)

/** Bumps when build settings open so EngineBuildSettingsHint remounts and hides after LS dismiss. */
const hintRevLlama = ref(0)
const hintRevIk = ref(0)
const hintRevLmdeploy = ref(0)
const hintRevOnecat = ref(0)

const BUILD_HINT_LS_KEY = 'lcs.engine.buildSettingsHintDismissed.v1'

function persistBuildHintDismissed(engineKey) {
  const k = String(engineKey || 'llama_cpp')
  try {
    const raw = localStorage.getItem(BUILD_HINT_LS_KEY)
    const o = raw ? JSON.parse(raw) : {}
    o[k] = true
    localStorage.setItem(BUILD_HINT_LS_KEY, JSON.stringify(o))
  } catch {
    /* ignore */
  }
}

const llamaCppSourceDialogVisible = ref(false)
const llamaCppSourceRepo = ref('https://github.com/ggerganov/llama.cpp.git')
const llamaCppSourceRef = ref('master')
const llamaCppSourceInstalling = ref(false)
const llamaReleaseInstalling = ref(false)

const ikLlamaSourceDialogVisible = ref(false)
const ikLlamaSourceRepo = ref('https://github.com/ikawrakow/ik_llama.cpp.git')
const ikLlamaSourceRef = ref('main')
const ikLlamaSourceInstalling = ref(false)
const ikTipInstalling = ref(false)

async function getMergedCmakeBuildConfig(engineId) {
  await ensureBuildOptionsCatalog(engineId)
  const base = _defaultBuildConfig()
  try {
    const saved = await fetchEngineBuildSettings(engineId)
    const raw = saved && typeof saved === 'object' ? { ...saved } : {}
    delete raw.tracking_ref
    delete raw.repository_url
    Object.assign(base, raw)
  } catch {
    // use defaults
  }
  if (engineId === 'ik_llama') {
    base.build_examples = true
  }
  return base
}

async function installLlamaLatestRelease() {
  llamaReleaseInstalling.value = true
  try {
    await enginesStore.updateEngine('llama_cpp')
    toast.add({
      severity: 'success',
      summary: 'Build started',
      detail:
        'Building the latest stable release (vX.Y.Z) with your saved build settings. Track progress in notifications.',
      life: 3500,
    })
  } catch (e) {
    toast.add({ severity: 'error', summary: 'Failed', detail: e.message, life: 4000 })
  } finally {
    llamaReleaseInstalling.value = false
  }
}

async function openLlamaCppSourceDialog() {
  try {
    const saved = await fetchEngineBuildSettings('llama_cpp')
    if (saved?.tracking_ref) {
      llamaCppSourceRef.value = saved.tracking_ref
    }
    if (saved?.repository_url) {
      llamaCppSourceRepo.value = saved.repository_url
    }
  } catch {
    /* keep defaults */
  }
  llamaCppSourceDialogVisible.value = true
}

async function installLlamaCppFromSource() {
  llamaCppSourceInstalling.value = true
  try {
    const config = await getMergedCmakeBuildConfig('llama_cpp')
    const ref = (llamaCppSourceRef.value || 'master').trim()
    const repo = (llamaCppSourceRepo.value || '').trim()
    const payload = {
      commit_sha: ref,
      repository_source: 'llama.cpp',
      build_config: config,
      auto_activate: false,
      source_ref_type: inferSourceRefType(ref),
    }
    if (repo) {
      payload.repository_url = repo
    }
    await enginesStore.buildSource(payload)
    llamaCppSourceDialogVisible.value = false
    toast.add({
      severity: 'success',
      summary: 'Build started',
      detail: 'Track progress in notifications',
      life: 3000,
    })
  } catch (e) {
    toast.add({ severity: 'error', summary: 'Build failed', detail: e.message, life: 4000 })
  } finally {
    llamaCppSourceInstalling.value = false
  }
}

async function installIkFromTip() {
  ikTipInstalling.value = true
  try {
    await enginesStore.updateEngine('ik_llama')
    toast.add({
      severity: 'success',
      summary: 'Build started',
      detail:
        'Building tip of main with your saved build settings. Track progress in notifications.',
      life: 3500,
    })
  } catch (e) {
    toast.add({ severity: 'error', summary: 'Failed', detail: e.message, life: 4000 })
  } finally {
    ikTipInstalling.value = false
  }
}

async function openIkLlamaSourceDialog() {
  try {
    const saved = await fetchEngineBuildSettings('ik_llama')
    if (saved?.tracking_ref) {
      ikLlamaSourceRef.value = saved.tracking_ref
    }
    if (saved?.repository_url) {
      ikLlamaSourceRepo.value = saved.repository_url
    }
  } catch {
    /* keep defaults */
  }
  ikLlamaSourceDialogVisible.value = true
}

async function installIkLlamaFromSource() {
  ikLlamaSourceInstalling.value = true
  try {
    const config = await getMergedCmakeBuildConfig('ik_llama')
    const ref = (ikLlamaSourceRef.value || 'main').trim()
    const repo = (ikLlamaSourceRepo.value || '').trim()
    const payload = {
      commit_sha: ref,
      repository_source: 'ik_llama.cpp',
      build_config: config,
      auto_activate: false,
      source_ref_type: inferSourceRefType(ref),
    }
    if (repo) {
      payload.repository_url = repo
    }
    await enginesStore.buildSource(payload)
    ikLlamaSourceDialogVisible.value = false
    toast.add({
      severity: 'success',
      summary: 'Build started',
      detail: 'Track progress in notifications',
      life: 3000,
    })
  } catch (e) {
    toast.add({ severity: 'error', summary: 'Build failed', detail: e.message, life: 4000 })
  } finally {
    ikLlamaSourceInstalling.value = false
  }
}

async function checkLlamaCppUpdates() {
  checkingLlamaCpp.value = true
  try {
    const raw = await enginesStore.checkLlamaCppUpdates()
    llamaCppUpdateInfo.value = normalizeLlamaUpdateInfo(
      raw,
      currentComparableLlamaVersion(activeLlamaCpp.value),
      'https://github.com/ggerganov/llama.cpp',
    )
  } catch (e) {
    toast.add({
      severity: 'warn',
      summary: 'Could not check updates',
      detail: e.message,
      life: 3000,
    })
  } finally {
    checkingLlamaCpp.value = false
  }
}

const checkingIkLlama = ref(false)
const ikLlamaUpdateInfo = ref(null)

async function checkIkLlamaUpdates() {
  checkingIkLlama.value = true
  try {
    const raw = await enginesStore.checkIkLlamaUpdates()
    ikLlamaUpdateInfo.value = normalizeLlamaUpdateInfo(
      raw,
      currentComparableLlamaVersion(activeIkLlama.value),
      'https://github.com/ikawrakow/ik_llama.cpp',
    )
  } catch (e) {
    toast.add({
      severity: 'warn',
      summary: 'Could not check updates',
      detail: e.message,
      life: 3000,
    })
  } finally {
    checkingIkLlama.value = false
  }
}

const checkingUnslothLlama = ref(false)
const unslothLlamaUpdateInfo = ref(null)
const unslothLlamaInstalling = ref(false)

function normalizeUnslothUpdateInfo(raw, currentVersion) {
  const latest = raw?.latest_version || null
  const current = raw?.current_version || currentVersion || 'none'
  const available = raw?.update_available
  return {
    update_available:
      typeof available === 'boolean'
        ? available
        : Boolean(latest && current !== 'none' && latest !== current),
    latest_version: latest,
    release_url: raw?.release_url || raw?.url || 'https://github.com/unslothai/llama.cpp/releases',
    current_version: current,
  }
}

async function checkUnslothLlamaUpdates() {
  checkingUnslothLlama.value = true
  try {
    const raw = await enginesStore.checkUnslothLlamaUpdates()
    unslothLlamaUpdateInfo.value = normalizeUnslothUpdateInfo(
      raw,
      activeUnslothLlama.value?.source_ref || activeUnslothLlama.value?.version,
    )
  } catch (e) {
    toast.add({
      severity: 'warn',
      summary: 'Could not check updates',
      detail: e.message,
      life: 3000,
    })
  } finally {
    checkingUnslothLlama.value = false
  }
}

async function installUnslothLlamaRelease() {
  unslothLlamaInstalling.value = true
  try {
    await enginesStore.installUnslothLlama({})
    toast.add({
      severity: 'success',
      summary: 'Install started',
      detail: 'Downloading the Unsloth llama-server mix. Track progress in notifications.',
      life: 3500,
    })
    await enginesStore.fetchLlamaVersions()
  } catch (e) {
    toast.add({
      severity: 'error',
      summary: 'Failed',
      detail: e?.response?.data?.detail || e.message,
      life: 4000,
    })
  } finally {
    unslothLlamaInstalling.value = false
  }
}

// ── Build from source dialog ───────────────────────────────
const buildDialogVisible = ref(false)
const buildTarget = ref('llama_cpp')
const buildUi = computed(() => ENGINE_DESCRIPTORS[buildTarget.value]?.buildUi || {})
const building = ref(false)
const savingBuildSettings = ref(false)
const buildCapabilities = ref(null)
const buildCapabilitiesError = ref('')
const verifyDialogVisible = ref(false)
const verifyLoading = ref(false)
const verifyError = ref('')
const verifyResult = ref(null)
const editingVersion = ref(null)
const editingAudioVersion = ref(null)
const buildingVersionConfig = ref(false)

const llamaBuildDialogHeader = computed(() => {
  const engineLabel = buildTarget.value === 'ik_llama' ? 'ik_llama.cpp' : 'llama.cpp'
  if (editingVersion.value?.version) {
    return `CMake settings — ${engineLabel} · ${editingVersion.value.version}`
  }
  return `Build settings — ${engineLabel}`
})

const audioCppBuildDialogHeader = computed(() => {
  if (editingAudioVersion.value?.version) {
    return `CMake settings — audio.cpp · ${editingAudioVersion.value.version}`
  }
  return 'Build settings — audio.cpp'
})

const buildTypeOptions = FALLBACK_BUILD_TYPES.map((value) => ({ label: value, value }))

// Fallback catalog if GET /build-options fails: generated from the backend
// build-option catalogs (scripts/export-build-defaults.py, --check in gates),
// so this can no longer drift from the backend defaults.

const buildOptionsCatalog = ref({
  categories: [],
  defaults: { ...FALLBACK_BUILD_DEFAULTS },
  build_types: [...FALLBACK_BUILD_TYPES],
})
const buildOptionsLoading = ref(false)
const buildOptionsCatalogEngine = ref(null)

const buildForm = ref({
  commitSha: '',
  versionSuffix: '',
  buildConfig: { ...FALLBACK_BUILD_DEFAULTS },
})

const visibleBuildCategories = computed(() => {
  const cats = buildOptionsCatalog.value?.categories || []
  return cats
    .filter((cat) => {
      if (cat.id === 'advanced') return false
      if (!cat.requires) return true
      return !!buildForm.value?.buildConfig?.[cat.requires]
    })
    .map((cat) => ({
      ...cat,
      // Default unknown categories to collapsed so the dialog stays calm
      collapsed: cat.collapsed !== false && cat.id !== 'backends',
    }))
})

function optionVisible(opt) {
  return !opt.requires || !!buildForm.value?.buildConfig?.[opt.requires]
}

function visibleOptions(cat) {
  return (cat.options || []).filter(optionVisible)
}

const llamaBuildFields = computed(() => [
  {
    key: 'commitSha',
    label: 'Ref (tag / branch / commit)',
    placeholder: buildUi.value.refPlaceholder || 'master',
    hint: buildUi.value.refHint || '',
  },
  {
    key: 'versionSuffix',
    label: 'Build Name Suffix',
    optional: true,
    placeholder: 'e.g. my-build',
    hint: 'Appended to version name. Defaults to timestamp if empty.',
  },
])

const llamaBuildNotes = computed(() => {
  if (buildCapabilitiesError.value) {
    return [{ text: buildCapabilitiesError.value, role: 'status' }]
  }
  if (buildCapabilities.value) {
    return [
      {
        text:
          `${capabilityLine('CUDA', buildCapabilities.value.cuda)}. ` +
          `${capabilityLine('OpenBLAS', buildCapabilities.value.openblas)}.`,
        tone: 'info',
        role: 'status',
      },
    ]
  }
  return []
})

const llamaBuildDialogCategories = computed(() =>
  visibleBuildCategories.value.map((cat) => {
    const isBackends = cat.id === 'backends'
    const options = (
      isBackends ? (cat.options || []).filter(optionVisible) : visibleOptions(cat)
    ).map((opt) => {
      const primary = opt.primary !== false
      return {
        ...opt,
        disabled: isBackends
          ? primary && backendToggleDisabled(opt.key)
          : Boolean(buildUi.value.disableBuildExamples && opt.key === 'build_examples'),
        disabledReason: isBackends && primary ? backendBlockReason(opt.key) : '',
        placeholder: opt.key === 'cuda_architectures' ? 'e.g. 86;89 (blank = auto)' : '',
      }
    })
    let note = ''
    if (isBackends) note = buildUi.value.backendNote || ''
    else if (cat.id === 'artifacts') note = buildUi.value.artifactsNote || ''
    return {
      ...cat,
      layout: isBackends ? 'grid' : 'advanced',
      note,
      options,
    }
  }),
)

const llamaBuildDialogActions = computed(() => {
  const version = editingVersion.value
  if (version) {
    const action = versionRebuildAction(version)
    return {
      save: {
        label: 'Save',
        loading: savingBuildSettings.value,
        disabled: documentSaveHeld.value,
      },
      ...(action
        ? {
            rebuild: {
              label: action === 'retry' ? 'Save & retry' : 'Save & rebuild',
              loading: savingBuildSettings.value || buildingVersionConfig.value,
            },
          }
        : {}),
    }
  }
  return {
    save: {
      label: 'Save settings',
      loading: savingBuildSettings.value,
      disabled: documentSaveHeld.value,
    },
    primary: { label: 'Build now', icon: 'pi pi-cog', loading: building.value },
  }
})

function applyLlamaBuildField(key, value) {
  buildForm.value[key] = value
}

function applyLlamaBuildOption(key, value) {
  buildForm.value.buildConfig[key] = value
}

function saveLlamaBuildSettingsFromDialog() {
  return editingVersion.value ? saveLlamaVersionBuildConfigOnly() : saveBuildSettingsOnly()
}

function _defaultBuildConfig() {
  const defaults = { ...(buildOptionsCatalog.value?.defaults || FALLBACK_BUILD_DEFAULTS) }
  return { ...FALLBACK_BUILD_DEFAULTS, ...defaults }
}

async function ensureBuildOptionsCatalog(engineId) {
  const engine = engineId === 'ik_llama' ? 'ik_llama' : 'llama_cpp'
  if (
    (buildOptionsCatalog.value?.categories || []).length &&
    buildOptionsCatalogEngine.value === engine
  ) {
    return
  }
  buildOptionsLoading.value = true
  try {
    const data = await enginesStore.fetchBuildOptionsForEngine(engine)
    if (data?.categories?.length) {
      buildOptionsCatalog.value = data
      buildOptionsCatalogEngine.value = engine
    }
  } catch {
    // keep fallback
  } finally {
    buildOptionsLoading.value = false
  }
}

function inferSourceRefType(ref) {
  const value = String(ref || '').trim()
  if (/^[0-9a-f]{40}$/i.test(value)) return 'commit'
  // audio.cpp GitHub Releases use release-X.Y(.Z); also legacy v* / b* tags
  if (
    /^(?:release-\d+(?:\.\d+)*(?:[-+][0-9A-Za-z._-]*)?|v?\d+(?:\.\d+){1,}(?:[-+][0-9A-Za-z._-]+)?|b\d+)$/i.test(
      value,
    )
  ) {
    return 'release'
  }
  return 'branch'
}

/** Show full release tags; shorten commit SHAs. */
function formatEngineUpdateVersion(value) {
  const s = String(value || '').trim()
  if (/^[0-9a-f]{7,40}$/i.test(s)) return s.slice(0, 8)
  return s
}

async function fetchEngineBuildSettings(engineId) {
  return await enginesStore.fetchBuildSettings(engineId)
}

async function saveEngineBuildSettings(engineId, settings) {
  return await enginesStore.saveBuildSettings(engineId, settings)
}

async function updateEngineWithSavedSettings(engineId) {
  return await enginesStore.updateEngine(engineId)
}

function llamaBuildSettingsPayload(config) {
  const ref = String(buildForm.value.commitSha || '').trim()
  const refType = ref ? inferSourceRefType(ref) : ''
  const payload = { ...config }
  // Persist branch/tag as tracking ref; bare commits are one-off and not tracked.
  if (ref && refType !== 'commit') {
    payload.tracking_ref = ref
  } else {
    payload.tracking_ref = ''
  }
  return payload
}

function capabilityForBackend(key) {
  if (key === 'enable_cuda') return buildCapabilities.value?.cuda
  if (key === 'enable_blas') return buildCapabilities.value?.openblas
  return null
}

function backendToggleDisabled(key) {
  const cap = capabilityForBackend(key)
  return Boolean(cap && cap.available === false)
}

function backendBlockReason(key) {
  const cap = capabilityForBackend(key)
  if (!cap || cap.available !== false) return ''
  const reason = String(cap.reason || '').trim()
  return reason || 'Not available on this machine.'
}

function lockUnavailableBackends() {
  const config = buildForm.value?.buildConfig
  const caps = buildCapabilities.value
  if (!config || !caps) return
  if (caps.cuda?.available === false) config.enable_cuda = false
  if (caps.openblas?.available === false) config.enable_blas = false
}

function capabilityLine(name, cap) {
  if (!cap || typeof cap !== 'object') return `${name}: unknown`
  const state = cap.available
    ? cap.recommended
      ? 'available, recommended'
      : 'available'
    : 'not available'
  const reason = String(cap.reason || '').trim()
  return reason ? `${name}: ${state} (${reason})` : `${name}: ${state}`
}

function axiosDetail(error, fallback) {
  const detail = error?.response?.data?.detail
  if (typeof detail === 'string' && detail.trim()) return detail
  return error?.message || fallback
}

async function loadBuildCapabilities() {
  buildCapabilitiesError.value = ''
  const load = enginesStore.fetchBuildCapabilities
  if (typeof load !== 'function') return
  try {
    buildCapabilities.value = await load()
    lockUnavailableBackends()
  } catch {
    buildCapabilities.value = null
    buildCapabilitiesError.value = 'Could not check CUDA and OpenBLAS on this machine.'
  }
}

async function verifyEngineVersion(versionId) {
  verifyDialogVisible.value = true
  verifyLoading.value = true
  verifyError.value = ''
  verifyResult.value = null
  try {
    verifyResult.value = await enginesStore.verifyLlamaVersion(versionId)
  } catch (error) {
    verifyError.value = axiosDetail(error, 'Could not verify this build.')
  } finally {
    verifyLoading.value = false
  }
}

async function openBuildDialog(engineKey) {
  editingVersion.value = null
  buildTarget.value = engineKey
  const engineId = engineKey === 'ik_llama' ? 'ik_llama' : 'llama_cpp'
  const updateInfo = engineKey === 'ik_llama' ? ikLlamaUpdateInfo.value : llamaCppUpdateInfo.value
  await ensureBuildOptionsCatalog(engineId)
  const baseConfig = _defaultBuildConfig()
  let trackingRef = ''
  try {
    const saved = await fetchEngineBuildSettings(engineId)
    const raw = saved && typeof saved === 'object' ? { ...saved } : {}
    trackingRef = String(raw.tracking_ref || '').trim()
    delete raw.tracking_ref
    delete raw.repository_url
    Object.assign(baseConfig, raw)
  } catch {
    // Ignore, fall back to defaults
  }
  // ik_llama.cpp requires Build examples (server is in examples/)
  if (engineKey === 'ik_llama') {
    baseConfig.build_examples = true
  }
  buildForm.value.commitSha =
    trackingRef || updateInfo?.latest_version || (engineKey === 'ik_llama' ? 'main' : 'master')
  buildForm.value.versionSuffix = ''
  buildForm.value.buildConfig = baseConfig
  persistBuildHintDismissed(engineKey)
  if (engineKey === 'ik_llama') {
    hintRevIk.value += 1
  } else {
    hintRevLlama.value += 1
  }
  void loadBuildCapabilities()
  buildDialogVisible.value = true
}

function closeLlamaBuildDialog() {
  buildDialogVisible.value = false
  editingVersion.value = null
}

function storedLlamaConfigToUi(raw) {
  const mapped = _defaultBuildConfig()
  if (!raw || typeof raw !== 'object') return mapped
  for (const [key, value] of Object.entries(raw)) {
    if (key === 'repository_source' || key === 'tracking_ref' || key === 'repository_url') continue
    if (key in mapped) {
      mapped[key] = value
      continue
    }
    if (String(key).startsWith('enable_')) {
      const uiKey = String(key).slice('enable_'.length)
      if (uiKey in mapped) mapped[uiKey] = value
    }
  }
  return mapped
}

async function openVersionBuildConfig(versionId) {
  const version = findListedVersion(versionId)
  if (!version) {
    toast.add({
      severity: 'warn',
      summary: 'Version not found',
      detail: 'Refresh engines and try again.',
      life: 3000,
    })
    return
  }
  const id = String(version.id || '')
  const engine = id.includes(':') ? id.split(':')[0] : ''
  if (engine === audioEngineId) {
    await openAudioVersionBuildConfig(version)
    return
  }
  if (engine !== 'llama_cpp' && engine !== 'ik_llama') {
    toast.add({
      severity: 'warn',
      summary: 'No CMake config',
      detail: 'This engine does not freeze a per-build CMake config.',
      life: 3000,
    })
    return
  }
  editingAudioVersion.value = null
  audioCppBuildDialogVisible.value = false
  editingVersion.value = version
  buildTarget.value = engine === 'ik_llama' ? 'ik_llama' : 'llama_cpp'
  await ensureBuildOptionsCatalog(buildTarget.value)
  const baseConfig = storedLlamaConfigToUi(version.build_config)
  if (engine === 'ik_llama') baseConfig.build_examples = true
  buildForm.value.commitSha = ''
  buildForm.value.versionSuffix = ''
  buildForm.value.buildConfig = baseConfig
  void loadBuildCapabilities()
  buildDialogVisible.value = true
}

async function persistEditingVersionBuildConfig() {
  const version = editingVersion.value
  if (!version) return null
  const config = { ...buildForm.value.buildConfig }
  return await enginesStore.saveVersionBuildConfig(version.id ?? version.version, config)
}

async function saveLlamaVersionBuildConfigOnly() {
  savingBuildSettings.value = true
  try {
    await persistEditingVersionBuildConfig()
    closeLlamaBuildDialog()
    documentSaveNotice.value = null
    toast.add({
      severity: 'success',
      summary: 'Build config saved',
      detail: 'This version will use the new CMake options on the next rebuild.',
      life: 3000,
    })
  } catch (e) {
    if (!onDocumentSaveError(e)) {
      toast.add({
        severity: 'error',
        summary: 'Save failed',
        detail: e?.response?.data?.detail || e.message,
        life: 4000,
      })
    }
  } finally {
    savingBuildSettings.value = false
  }
}

async function saveLlamaVersionBuildConfigAndRebuild() {
  const version = editingVersion.value
  if (!version) return
  const action = versionRebuildAction(version)
  savingBuildSettings.value = true
  buildingVersionConfig.value = true
  try {
    const versionId = version.id ?? version.version
    await persistEditingVersionBuildConfig()
    closeLlamaBuildDialog()
    if (action === 'retry') {
      await retryVersion(versionId)
    } else if (action === 'sync') {
      await syncVersion(versionId)
    } else {
      toast.add({
        severity: 'success',
        summary: 'Build config saved',
        detail: 'This version will use the new CMake options on the next rebuild.',
        life: 3000,
      })
    }
  } catch (e) {
    toast.add({
      severity: 'error',
      summary: 'Save failed',
      detail: e?.response?.data?.detail || e.message,
      life: 4000,
    })
  } finally {
    savingBuildSettings.value = false
    buildingVersionConfig.value = false
  }
}

async function doStartBuild() {
  building.value = true
  try {
    const repoSource = buildTarget.value === 'ik_llama' ? 'ik_llama.cpp' : 'llama.cpp'
    const engineId = buildTarget.value === 'ik_llama' ? 'ik_llama' : 'llama_cpp'
    const config = { ...buildForm.value.buildConfig }
    // Persist settings before triggering a manual build (full config + tracking ref)
    await saveEngineBuildSettings(engineId, llamaBuildSettingsPayload(config))
    await enginesStore.buildSource({
      commit_sha:
        buildForm.value.commitSha || (buildTarget.value === 'ik_llama' ? 'main' : 'master'),
      repository_source: repoSource,
      version_suffix: buildForm.value.versionSuffix || undefined,
      build_config: config,
      auto_activate: false,
      source_ref_type: inferSourceRefType(
        buildForm.value.commitSha || (buildTarget.value === 'ik_llama' ? 'main' : 'master'),
      ),
    })
    buildDialogVisible.value = false
    toast.add({
      severity: 'success',
      summary: 'Build started',
      detail: 'Track progress in notifications',
      life: 3000,
    })
  } catch (e) {
    toast.add({ severity: 'error', summary: 'Build failed', detail: e.message, life: 4000 })
  } finally {
    building.value = false
  }
}

async function saveBuildSettingsOnly() {
  const engineId = buildTarget.value === 'ik_llama' ? 'ik_llama' : 'llama_cpp'
  const config = { ...buildForm.value.buildConfig }
  savingBuildSettings.value = true
  try {
    await saveEngineBuildSettings(engineId, llamaBuildSettingsPayload(config))
    buildDialogVisible.value = false
    documentSaveNotice.value = null
    toast.add({
      severity: 'success',
      summary: 'Build settings saved',
      detail: 'Options stored without building.',
      life: 2500,
    })
  } catch (e) {
    if (!onDocumentSaveError(e)) {
      toast.add({ severity: 'error', summary: 'Save failed', detail: e.message, life: 4000 })
    }
  } finally {
    savingBuildSettings.value = false
  }
}

async function doUpdateEngine(engineKey) {
  const updateInfo = engineKey === 'ik_llama' ? ikLlamaUpdateInfo.value : llamaCppUpdateInfo.value
  if (!updateInfo?.latest_version) {
    toast.add({
      severity: 'warn',
      summary: 'No update available',
      detail: 'Check for updates first.',
      life: 3000,
    })
    return
  }
  const engineId = engineKey === 'ik_llama' ? 'ik_llama' : 'llama_cpp'
  updatingEngine.value = engineKey
  try {
    await updateEngineWithSavedSettings(engineId)
    toast.add({
      severity: 'success',
      summary: 'Update started',
      detail: 'Build in progress — track in notifications.',
      life: 3000,
    })
  } catch (e) {
    toast.add({ severity: 'error', summary: 'Update failed', detail: e.message, life: 4000 })
  } finally {
    updatingEngine.value = null
  }
}

// ── audio.cpp ──────────────────────────────────────────────
const checkingAudioCpp = ref(false)
const audioCppUpdateInfo = ref(null)
const audioCppUpdating = ref(false)
const hintRevAudio = ref(0)
const audioCppReleaseInstalling = ref(false)
const audioCppSourceInstalling = ref(false)
const audioCppSourceDialogVisible = ref(false)
const audioCppSourceRepo = ref('https://github.com/0xShug0/audio.cpp.git')
const audioCppSourceRef = ref('')

const audioCppBuilding = ref(false)
const savingAudioCppBuildSettings = ref(false)
const audioCppBuildDialogVisible = ref(false)
const audioCppOptionsLoading = ref(false)
const audioCppOptionsCatalog = ref({ categories: [], defaults: {}, build_types: [] })

const AUDIO_FALLBACK_DEFAULTS = {
  build_type: 'RelWithDebInfo',
  cuda: false,
  native_cpu: true,
  openmp: true,
  cuda_graphs: true,
  llamafile: true,
  cpu_all_variants: false,
  build_tests: false,
  build_extended_tests: false,
  build_model_tests: false,
  build_examples: false,
  build_warmbench: false,
  deployment_build: false,
  native_model_manager: false,
  use_system_openssl: false,
  build_server_frontends: false,
  build_c_api: false,
  static_espeak: false,
  model_set: 'full',
  models: '',
  jobs: 0,
  custom_cmake_args: '',
  cflags: '',
  cxxflags: '',
  backend: 'cpu',
}

const audioCppBuildTypeOptions = [
  { label: 'RelWithDebInfo', value: 'RelWithDebInfo' },
  { label: 'Release', value: 'Release' },
  { label: 'Debug', value: 'Debug' },
  { label: 'MinSizeRel', value: 'MinSizeRel' },
]

function _defaultAudioBuildConfig() {
  return {
    ...AUDIO_FALLBACK_DEFAULTS,
    ...(audioCppOptionsCatalog.value?.defaults || {}),
  }
}

const audioCppBuildForm = ref({
  repository_url: 'https://github.com/0xShug0/audio.cpp.git',
  source_ref: '',
  versionSuffix: '',
  build_config: _defaultAudioBuildConfig(),
})

const visibleAudioBuildCategories = computed(() => {
  const cats = audioCppOptionsCatalog.value?.categories || []
  return cats.filter((cat) => {
    if (cat.id === 'advanced') return false
    if (!cat.requires) return true
    return audioOptionParentEnabled(cat.requires)
  })
})

function audioOptionParentEnabled(requires) {
  const cfg = audioCppBuildForm.value?.build_config || {}
  if (requires === 'cuda_or_hip') return !!cfg.cuda
  if (requires === 'model_set_custom') return cfg.model_set === 'custom'
  return !!cfg[requires]
}

function visibleAudioOptions(cat) {
  return (cat.options || []).filter(
    (opt) => !opt.requires || audioOptionParentEnabled(opt.requires),
  )
}

function audioBackendDisabled(key) {
  const supported = new Set(audioCppStatus.value?.supported_build_backends || ['cpu', 'cuda'])
  return !supported.has(key)
}

const audioBuildFields = computed(() => [
  {
    key: 'repository_url',
    label: 'Repo URL',
    placeholder: 'https://github.com/0xShug0/audio.cpp.git',
    hint: 'Official repo or any fork with the same layout. Saved with Build settings.',
  },
  {
    key: 'source_ref',
    label: 'Ref (tag / branch / commit)',
    placeholder: audioCppStatus.value?.tracking_ref || 'main',
    hint:
      'Used when you Build now. Branch and tag refs become the Update tracking ref. ' +
      'Building a commit installs that tip but leaves tracking on the previous branch/tag, ' +
      'so Update will not follow the detached commit.',
  },
  {
    key: 'versionSuffix',
    label: 'Build Name Suffix',
    optional: true,
    placeholder: 'e.g. my-build',
    hint: 'Appended to version name. Defaults to timestamp if empty.',
  },
])

const audioBuildDialogCategories = computed(() =>
  visibleAudioBuildCategories.value.map((cat) => {
    const isBackends = cat.id === 'backends'
    return {
      ...cat,
      layout: isBackends ? 'grid' : 'advanced',
      options: (isBackends ? cat.options || [] : visibleAudioOptions(cat)).map((opt) => ({
        ...opt,
        disabled: isBackends ? audioBackendDisabled(opt.key) : false,
      })),
    }
  }),
)

const audioBuildDialogActions = computed(() => {
  const version = editingAudioVersion.value
  if (version) {
    const action = versionRebuildAction(version)
    return {
      save: {
        label: 'Save',
        loading: savingAudioCppBuildSettings.value,
        disabled: documentSaveHeld.value,
      },
      ...(action
        ? {
            rebuild: {
              label: action === 'retry' ? 'Save & retry' : 'Save & rebuild',
              loading: savingAudioCppBuildSettings.value || buildingVersionConfig.value,
            },
          }
        : {}),
    }
  }
  return {
    save: {
      label: 'Save settings',
      loading: savingAudioCppBuildSettings.value,
      disabled: documentSaveHeld.value,
    },
    primary: { label: 'Build now', icon: 'pi pi-cog', loading: audioCppBuilding.value },
  }
})

function applyAudioBuildField(key, value) {
  audioCppBuildForm.value[key] = value
}

function applyAudioBuildOption(key, value) {
  audioCppBuildForm.value.build_config[key] = value
}

function saveAudioBuildSettingsFromDialog() {
  return editingAudioVersion.value
    ? saveAudioVersionBuildConfigOnly()
    : saveAudioCppBuildSettingsOnly()
}

const audioCppUpdateTooltip = computed(() => {
  const releaseTag = audioCppUpdateInfo.value?.latest_release?.tag_name
  if (releaseTag) {
    return `Build source at latest GitHub release tag ${releaseTag}`
  }
  const ref =
    audioCppUpdateInfo.value?.tracking_ref || audioCppStatus.value?.tracking_ref || 'tracked ref'
  return `Sync or rebuild tip of ${ref}`
})

function splitAudioCppSettings(saved) {
  const raw = saved && typeof saved === 'object' ? saved : {}
  const tracking_ref = raw.tracking_ref || ''
  const repository_url = raw.repository_url || 'https://github.com/0xShug0/audio.cpp.git'
  const build_config = { ..._defaultAudioBuildConfig() }
  for (const key of Object.keys(build_config)) {
    if (key in raw) build_config[key] = raw[key]
  }
  // Legacy backend string — only CUDA is supported
  if (raw.backend === 'cuda' && !build_config.cuda) {
    build_config.cuda = true
  }
  return { tracking_ref, repository_url, build_config }
}

async function ensureAudioCppOptionsCatalog() {
  if ((audioCppOptionsCatalog.value?.categories || []).length) return
  audioCppOptionsLoading.value = true
  try {
    const data = await enginesStore.fetchBuildOptionsForEngine(audioEngineId)
    if (data?.categories?.length) {
      audioCppOptionsCatalog.value = data
    }
  } catch {
    // keep fallback
  } finally {
    audioCppOptionsLoading.value = false
  }
}

async function openAudioCppBuildSettings() {
  editingAudioVersion.value = null
  await ensureAudioCppOptionsCatalog()
  const base = _defaultAudioBuildConfig()
  try {
    const saved = await enginesStore.fetchAudioCppBuildSettings()
    const split = splitAudioCppSettings(saved)
    audioCppBuildForm.value.repository_url = split.repository_url
    audioCppBuildForm.value.source_ref =
      split.tracking_ref || audioCppStatus.value?.tracking_ref || 'main'
    audioCppBuildForm.value.build_config = { ...base, ...split.build_config }
  } catch {
    audioCppBuildForm.value.build_config = base
    audioCppBuildForm.value.source_ref = audioCppStatus.value?.tracking_ref || 'main'
  }
  audioCppBuildForm.value.versionSuffix = ''
  persistBuildHintDismissed(audioEngineId)
  hintRevAudio.value += 1
  audioCppBuildDialogVisible.value = true
}

function closeAudioCppBuildDialog() {
  audioCppBuildDialogVisible.value = false
  editingAudioVersion.value = null
}

async function openAudioVersionBuildConfig(version) {
  editingVersion.value = null
  buildDialogVisible.value = false
  editingAudioVersion.value = version
  await ensureAudioCppOptionsCatalog()
  const base = _defaultAudioBuildConfig()
  const split = splitAudioCppSettings(version?.build_config || {})
  audioCppBuildForm.value.build_config = { ...base, ...split.build_config }
  audioCppBuildForm.value.versionSuffix = ''
  audioCppBuildDialogVisible.value = true
}

async function persistEditingAudioVersionBuildConfig() {
  const version = editingAudioVersion.value
  if (!version) return null
  const buildConfig = { ...audioCppBuildForm.value.build_config }
  return await enginesStore.saveVersionBuildConfig(version.id ?? version.version, buildConfig)
}

async function saveAudioVersionBuildConfigOnly() {
  savingAudioCppBuildSettings.value = true
  try {
    await persistEditingAudioVersionBuildConfig()
    closeAudioCppBuildDialog()
    documentSaveNotice.value = null
    toast.add({
      severity: 'success',
      summary: 'Build config saved',
      detail: 'This version will use the new CMake options on the next rebuild.',
      life: 3000,
    })
  } catch (e) {
    if (!onDocumentSaveError(e)) {
      toast.add({
        severity: 'error',
        summary: 'Save failed',
        detail: e?.response?.data?.detail || e.message,
        life: 4000,
      })
    }
  } finally {
    savingAudioCppBuildSettings.value = false
  }
}

async function saveAudioVersionBuildConfigAndRebuild() {
  const version = editingAudioVersion.value
  if (!version) return
  const action = versionRebuildAction(version)
  savingAudioCppBuildSettings.value = true
  buildingVersionConfig.value = true
  try {
    const versionId = version.id ?? version.version
    await persistEditingAudioVersionBuildConfig()
    closeAudioCppBuildDialog()
    if (action === 'retry') {
      await retryVersion(versionId)
    } else if (action === 'sync') {
      await syncVersion(versionId)
    } else {
      toast.add({
        severity: 'success',
        summary: 'Build config saved',
        detail: 'This version will use the new CMake options on the next rebuild.',
        life: 3000,
      })
    }
  } catch (e) {
    toast.add({
      severity: 'error',
      summary: 'Save failed',
      detail: e?.response?.data?.detail || e.message,
      life: 4000,
    })
  } finally {
    savingAudioCppBuildSettings.value = false
    buildingVersionConfig.value = false
  }
}

function audioCppSettingsPayloadFromForm() {
  const buildConfig = { ...audioCppBuildForm.value.build_config }
  const sourceRef = String(audioCppBuildForm.value.source_ref || '').trim()
  const sourceRefType = sourceRef ? inferSourceRefType(sourceRef) : 'branch'
  return {
    ...buildConfig,
    tracking_ref: sourceRef && sourceRefType !== 'commit' ? sourceRef : undefined,
    repository_url:
      String(audioCppBuildForm.value.repository_url || '').trim() ||
      'https://github.com/0xShug0/audio.cpp.git',
  }
}

async function saveAudioCppBuildSettingsOnly() {
  savingAudioCppBuildSettings.value = true
  try {
    await enginesStore.saveAudioCppBuildSettings(audioCppSettingsPayloadFromForm())
    await enginesStore.fetchAudioCppStatus()
    audioCppBuildDialogVisible.value = false
    documentSaveNotice.value = null
    toast.add({
      severity: 'success',
      summary: 'audio.cpp build settings saved',
      detail: 'CMake options were stored without building.',
      life: 3000,
    })
  } catch (e) {
    if (!onDocumentSaveError(e)) {
      toast.add({
        severity: 'error',
        summary: 'Save failed',
        detail: e?.response?.data?.detail || e.message,
        life: 4000,
      })
    }
  } finally {
    savingAudioCppBuildSettings.value = false
  }
}

async function checkAudioCppUpdates() {
  checkingAudioCpp.value = true
  try {
    audioCppUpdateInfo.value = await enginesStore.checkAudioCppUpdates()
  } catch (e) {
    audioCppUpdateInfo.value = null
    toast.add({
      severity: 'warn',
      summary: 'Could not check audio.cpp updates',
      detail: e?.response?.data?.detail || e.message,
      life: 3500,
    })
  } finally {
    checkingAudioCpp.value = false
  }
}

async function buildAudioCpp() {
  audioCppBuilding.value = true
  try {
    const buildConfig = { ...audioCppBuildForm.value.build_config }
    const sourceRef = audioCppBuildForm.value.source_ref || 'main'
    const sourceRefType = inferSourceRefType(sourceRef)
    await enginesStore.saveAudioCppBuildSettings(audioCppSettingsPayloadFromForm())
    await enginesStore.buildAudioCppSource({
      repository_url:
        audioCppBuildForm.value.repository_url || 'https://github.com/0xShug0/audio.cpp.git',
      source_ref: sourceRef,
      source_ref_type: sourceRefType,
      version_suffix: audioCppBuildForm.value.versionSuffix || undefined,
      build_config: buildConfig,
      auto_activate: true,
    })
    audioCppBuildDialogVisible.value = false
    await enginesStore.fetchAudioCppStatus()
    toast.add({
      severity: 'success',
      summary: 'audio.cpp build started',
      detail: 'Track progress in notifications',
      life: 3000,
    })
  } catch (e) {
    toast.add({
      severity: 'error',
      summary: 'audio.cpp build failed',
      detail: e?.response?.data?.detail || e.message,
      life: 5000,
    })
  } finally {
    audioCppBuilding.value = false
  }
}

async function openAudioCppSourceDialog() {
  try {
    const saved = await enginesStore.fetchAudioCppBuildSettings()
    const split = splitAudioCppSettings(saved)
    audioCppSourceRepo.value = split.repository_url
    audioCppSourceRef.value = split.tracking_ref || audioCppStatus.value?.tracking_ref || 'main'
  } catch {
    audioCppSourceRepo.value = 'https://github.com/0xShug0/audio.cpp.git'
    audioCppSourceRef.value = audioCppStatus.value?.tracking_ref || 'main'
  }
  audioCppSourceDialogVisible.value = true
}

async function installAudioLatestRelease() {
  audioCppReleaseInstalling.value = true
  try {
    const saved = await enginesStore.fetchAudioCppBuildSettings()
    const split = splitAudioCppSettings(saved)
    const data = await enginesStore.updateAudioCpp({
      from_release: true,
      build_config: split.build_config,
      repository_url: split.repository_url,
    })
    await enginesStore.fetchAudioCppStatus()
    toast.add({
      severity: 'info',
      summary: data?.prebuilt
        ? 'Installing audio.cpp release'
        : 'Building latest audio.cpp release',
      detail: data?.prebuilt
        ? data.prebuilt_reason ||
          'Downloading the matching prebuilt. Track progress in notifications.'
        : data?.prebuilt_skipped ||
          'Using your saved build settings. Track progress in notifications.',
      life: 4000,
    })
  } catch (e) {
    toast.add({
      severity: 'error',
      summary: 'Release install failed',
      detail: e?.response?.data?.detail || e.message,
      life: 5000,
    })
  } finally {
    audioCppReleaseInstalling.value = false
  }
}

async function installAudioCppFromSource() {
  // Saving build settings is a separate document write. It does not mean the
  // source build started or finished.
  const repo = String(audioCppSourceRepo.value || '').trim()
  const ref = String(audioCppSourceRef.value || '').trim()
  if (!repo || !ref) {
    toast.add({
      severity: 'warn',
      summary: 'Repo and ref required',
      detail: 'Enter a repository URL and tag/branch/commit.',
      life: 3000,
    })
    return
  }
  audioCppSourceInstalling.value = true
  try {
    const saved = await enginesStore.fetchAudioCppBuildSettings()
    const split = splitAudioCppSettings(saved)
    const sourceRefType = inferSourceRefType(ref)
    await enginesStore.saveAudioCppBuildSettings({
      ...split.build_config,
      repository_url: repo,
      tracking_ref: sourceRefType !== 'commit' ? ref : split.tracking_ref,
    })
    await enginesStore.buildAudioCppSource({
      repository_url: repo,
      source_ref: ref,
      source_ref_type: sourceRefType,
      build_config: split.build_config,
      auto_activate: true,
    })
    audioCppSourceDialogVisible.value = false
    await enginesStore.fetchAudioCppStatus()
    toast.add({
      severity: 'success',
      summary: 'audio.cpp build started',
      detail: 'Track progress in notifications',
      life: 3000,
    })
  } catch (e) {
    toast.add({
      severity: 'error',
      summary: 'audio.cpp build failed',
      detail: e?.response?.data?.detail || e.message,
      life: 5000,
    })
  } finally {
    audioCppSourceInstalling.value = false
  }
}

async function updateAudioCpp() {
  audioCppUpdating.value = true
  try {
    const saved = await enginesStore.fetchAudioCppBuildSettings()
    const split = splitAudioCppSettings(saved)
    const preferRelease = Boolean(audioCppUpdateInfo.value?.latest_release?.tag_name)
    const data = await enginesStore.updateAudioCpp({
      build_config: split.build_config,
      repository_url: split.repository_url,
      ...(preferRelease
        ? { from_release: true }
        : { source_ref: split.tracking_ref || audioCppStatus.value?.tracking_ref }),
    })
    await enginesStore.fetchAudioCppStatus()
    toast.add({
      severity: 'success',
      summary: data?.sync
        ? 'audio.cpp sync started'
        : data?.prebuilt
          ? 'Installing audio.cpp release'
          : 'audio.cpp update started',
      detail: data?.prebuilt_reason || data?.prebuilt_skipped || 'Track progress in notifications',
      life: 3500,
    })
  } catch (e) {
    toast.add({
      severity: 'error',
      summary: 'audio.cpp update failed',
      detail: e?.response?.data?.detail || e.message,
      life: 5000,
    })
  } finally {
    audioCppUpdating.value = false
  }
}

// ── LMDeploy ───────────────────────────────────────────────
const lm = computed(() => lmdeployStatus.value)
const lmdeployPipVersion = ref('')
const lmSourceRepo = ref('https://github.com/InternLM/lmdeploy.git')
const lmSourceBranch = ref('main')
const lmdeployInstalling = ref(false)
const checkingLmdeploy = ref(false)
const lmdeployUpdateInfo = ref(null)
const lmPipDialogVisible = ref(false)
const lmSourceDialogVisible = ref(false)
const lmdeployBuildDialogVisible = ref(false)
const savingLmdeployBuildSettings = ref(false)
const lmdeployBuildForm = ref({
  source_repo: 'https://github.com/InternLM/lmdeploy.git',
  source_branch: 'main',
  pip_version: '',
  compat_sm70: false,
})

const lmdeployCompatibility = computed(() => lmdeployCompatibilityFor(enginesStore.gpuInfo))

async function applyLmdeployBuildSettings(saved) {
  const s = saved && typeof saved === 'object' ? saved : {}
  lmdeployBuildForm.value = {
    source_repo: s.source_repo || 'https://github.com/InternLM/lmdeploy.git',
    source_branch: s.source_branch || 'main',
    pip_version: s.pip_version || '',
    compat_sm70: s.compat_sm70 === true || s.compat_sm70 === 'true',
  }
  lmSourceRepo.value = lmdeployBuildForm.value.source_repo
  lmSourceBranch.value = lmdeployBuildForm.value.source_branch
  lmdeployPipVersion.value = lmdeployBuildForm.value.pip_version
}

async function openLmdeployBuildSettings() {
  try {
    await enginesStore.fetchGpuInfo()
  } catch {
    /* Keep the dialog usable if hardware detection is unavailable. */
  }
  try {
    const saved = await enginesStore.fetchLmdeployBuildSettings()
    await applyLmdeployBuildSettings(saved)
  } catch {
    await applyLmdeployBuildSettings({})
  }
  persistBuildHintDismissed('lmdeploy')
  hintRevLmdeploy.value += 1
  lmdeployBuildDialogVisible.value = true
}

const updatingLmdeploy = ref(false)

async function doUpdateLmdeploy() {
  const latest = lmdeployUpdateInfo.value?.latest_version
  if (!latest) {
    toast.add({
      severity: 'warn',
      summary: 'No update available',
      detail: 'Check for updates first.',
      life: 3000,
    })
    return
  }
  updatingLmdeploy.value = true
  try {
    await enginesStore.installLmdeploy({ version: String(latest) })
    toast.add({
      severity: 'success',
      summary: 'Update started',
      detail: `Installing LMDeploy v${latest} — track progress in notifications.`,
      life: 3500,
    })
  } catch (e) {
    toast.add({ severity: 'error', summary: 'Update failed', detail: e.message, life: 4000 })
  } finally {
    updatingLmdeploy.value = false
  }
}

async function saveLmdeployBuildSettingsOnly() {
  savingLmdeployBuildSettings.value = true
  try {
    const saved = await enginesStore.saveLmdeployBuildSettings({ ...lmdeployBuildForm.value })
    await applyLmdeployBuildSettings(saved)
    lmdeployBuildDialogVisible.value = false
    documentSaveNotice.value = null
    toast.add({
      severity: 'success',
      summary: 'Build settings saved',
      detail: 'LMDeploy defaults stored without installing.',
      life: 2500,
    })
  } catch (e) {
    if (!onDocumentSaveError(e)) {
      toast.add({ severity: 'error', summary: 'Save failed', detail: e.message, life: 4000 })
    }
  } finally {
    savingLmdeployBuildSettings.value = false
  }
}

async function installLmdeployFromBuildSettings() {
  // A stored settings document is not evidence this install finished.
  lmdeployInstalling.value = true
  try {
    await enginesStore.saveLmdeployBuildSettings({ ...lmdeployBuildForm.value })
    await enginesStore.installLmdeployFromSource({
      repo_url: lmdeployBuildForm.value.source_repo,
      branch: lmdeployBuildForm.value.source_branch,
    })
    await applyLmdeployBuildSettings(lmdeployBuildForm.value)
    lmdeployBuildDialogVisible.value = false
    toast.add({
      severity: 'success',
      summary: 'Install from source started',
      detail: 'Track progress in notifications',
      life: 3000,
    })
  } catch (e) {
    toast.add({ severity: 'error', summary: 'Failed', detail: e.message, life: 4000 })
  } finally {
    lmdeployInstalling.value = false
  }
}

async function openLmdeployPipDialog() {
  try {
    await applyLmdeployBuildSettings(await enginesStore.fetchLmdeployBuildSettings())
  } catch {
    /* keep current fields */
  }
  lmPipDialogVisible.value = true
}

async function openLmdeploySourceDialog() {
  try {
    await applyLmdeployBuildSettings(await enginesStore.fetchLmdeployBuildSettings())
  } catch {
    /* keep current fields */
  }
  lmSourceDialogVisible.value = true
}

async function checkLmdeployUpdates() {
  checkingLmdeploy.value = true
  try {
    const raw = await enginesStore.checkLmdeployUpdates()
    const current = activeLmdeploy.value?.version || lm.value?.version || null
    const latest = raw?.latest_version || null
    const updateAvailable = latest && current !== latest
    lmdeployUpdateInfo.value = {
      update_available: updateAvailable,
      latest_version: latest,
      current_version: current,
    }
  } catch (e) {
    toast.add({
      severity: 'warn',
      summary: 'Could not check updates',
      detail: e.message,
      life: 3000,
    })
  } finally {
    checkingLmdeploy.value = false
  }
}

async function installLmdeployPip() {
  lmdeployInstalling.value = true
  try {
    await enginesStore.saveLmdeployBuildSettings({
      ...lmdeployBuildForm.value,
      pip_version: lmdeployPipVersion.value || '',
      source_repo: lmSourceRepo.value || lmdeployBuildForm.value.source_repo,
      source_branch: lmSourceBranch.value || lmdeployBuildForm.value.source_branch,
    })
    await enginesStore.installLmdeploy(
      lmdeployPipVersion.value ? { version: lmdeployPipVersion.value } : {},
    )
    lmPipDialogVisible.value = false
    toast.add({
      severity: 'success',
      summary: 'LMDeploy install started',
      detail: 'Track progress in notifications',
      life: 3000,
    })
  } catch (e) {
    toast.add({ severity: 'error', summary: 'Failed', detail: e.message, life: 4000 })
  } finally {
    // Ensure the dialog spinner is cleared even if the request fails.
    lmdeployInstalling.value = false
    lmPipDialogVisible.value = false
  }
}

async function installLmdeploySource() {
  lmdeployInstalling.value = true
  try {
    await enginesStore.saveLmdeployBuildSettings({
      ...lmdeployBuildForm.value,
      source_repo: lmSourceRepo.value,
      source_branch: lmSourceBranch.value,
      pip_version: lmdeployPipVersion.value || lmdeployBuildForm.value.pip_version,
    })
    await enginesStore.installLmdeployFromSource({
      repo_url: lmSourceRepo.value,
      branch: lmSourceBranch.value,
    })
    toast.add({
      severity: 'success',
      summary: 'Install from source started',
      detail: 'Track progress in notifications',
      life: 3000,
    })
  } catch (e) {
    toast.add({ severity: 'error', summary: 'Failed', detail: e.message, life: 4000 })
  } finally {
    lmdeployInstalling.value = false
    // Ensure the modal doesn't stay in a "loading" state after failure.
    lmSourceDialogVisible.value = false
  }
}

// ── 1Cat-vLLM ──────────────────────────────────────────────
const ovllm = computed(() => onecatVllmStatus.value)
const ovllmReleaseVersion = ref('')
const ovllmSourceRepo = ref('https://github.com/1CatAI/1Cat-vLLM.git')
const ovllmSourceBranch = ref('main')
const onecatVllmInstalling = ref(false)
const checkingOnecatVllm = ref(false)
const onecatVllmUpdateInfo = ref(null)
const ovllmReleaseDialogVisible = ref(false)
const ovllmSourceDialogVisible = ref(false)
const onecatVllmBuildDialogVisible = ref(false)
const savingOnecatVllmBuildSettings = ref(false)
const onecatVllmBuildForm = ref({
  source_repo: 'https://github.com/1CatAI/1Cat-vLLM.git',
  source_branch: 'main',
  release_version: '',
})

async function applyOnecatVllmBuildSettings(saved) {
  const s = saved && typeof saved === 'object' ? saved : {}
  onecatVllmBuildForm.value = {
    source_repo: s.source_repo || 'https://github.com/1CatAI/1Cat-vLLM.git',
    source_branch: s.source_branch || 'main',
    release_version: s.release_version || '',
  }
  ovllmSourceRepo.value = onecatVllmBuildForm.value.source_repo
  ovllmSourceBranch.value = onecatVllmBuildForm.value.source_branch
  ovllmReleaseVersion.value = onecatVllmBuildForm.value.release_version
}

async function openOnecatVllmBuildSettings() {
  try {
    const saved = await enginesStore.fetchOnecatVllmBuildSettings()
    await applyOnecatVllmBuildSettings(saved)
  } catch {
    await applyOnecatVllmBuildSettings({})
  }
  persistBuildHintDismissed('1cat_vllm')
  hintRevOnecat.value += 1
  onecatVllmBuildDialogVisible.value = true
}

const updatingOnecatVllm = ref(false)

async function doUpdateOnecatVllm() {
  const latest = onecatVllmUpdateInfo.value?.latest_version
  if (!latest) {
    toast.add({
      severity: 'warn',
      summary: 'No update available',
      detail: 'Check for updates first.',
      life: 3000,
    })
    return
  }
  updatingOnecatVllm.value = true
  try {
    await enginesStore.installOnecatVllm({ version: String(latest) })
    toast.add({
      severity: 'success',
      summary: 'Update started',
      detail: `Installing 1Cat-vLLM v${latest} — track progress in notifications.`,
      life: 3500,
    })
  } catch (e) {
    toast.add({ severity: 'error', summary: 'Update failed', detail: e.message, life: 4000 })
  } finally {
    updatingOnecatVllm.value = false
  }
}

async function saveOnecatVllmBuildSettingsOnly() {
  savingOnecatVllmBuildSettings.value = true
  try {
    const saved = await enginesStore.saveOnecatVllmBuildSettings({ ...onecatVllmBuildForm.value })
    await applyOnecatVllmBuildSettings(saved)
    onecatVllmBuildDialogVisible.value = false
    documentSaveNotice.value = null
    toast.add({
      severity: 'success',
      summary: 'Build settings saved',
      detail: '1Cat-vLLM defaults stored without installing.',
      life: 2500,
    })
  } catch (e) {
    if (!onDocumentSaveError(e)) {
      toast.add({ severity: 'error', summary: 'Save failed', detail: e.message, life: 4000 })
    }
  } finally {
    savingOnecatVllmBuildSettings.value = false
  }
}

async function installOnecatVllmFromBuildSettings() {
  // A stored settings document is not evidence this install finished.
  onecatVllmInstalling.value = true
  try {
    await enginesStore.saveOnecatVllmBuildSettings({ ...onecatVllmBuildForm.value })
    await enginesStore.installOnecatVllmFromSource({
      repo_url: onecatVllmBuildForm.value.source_repo,
      branch: onecatVllmBuildForm.value.source_branch,
    })
    await applyOnecatVllmBuildSettings(onecatVllmBuildForm.value)
    onecatVllmBuildDialogVisible.value = false
    toast.add({
      severity: 'success',
      summary: 'Source build started',
      detail: 'Track progress in notifications',
      life: 3000,
    })
  } catch (e) {
    toast.add({ severity: 'error', summary: 'Failed', detail: e.message, life: 4000 })
  } finally {
    onecatVllmInstalling.value = false
  }
}

async function openOnecatVllmReleaseDialog() {
  try {
    await applyOnecatVllmBuildSettings(await enginesStore.fetchOnecatVllmBuildSettings())
  } catch {
    /* keep current fields */
  }
  ovllmReleaseDialogVisible.value = true
}

async function openOnecatVllmSourceDialog() {
  try {
    await applyOnecatVllmBuildSettings(await enginesStore.fetchOnecatVllmBuildSettings())
  } catch {
    /* keep current fields */
  }
  ovllmSourceDialogVisible.value = true
}

async function checkOnecatVllmUpdates() {
  checkingOnecatVllm.value = true
  try {
    const raw = await enginesStore.checkOnecatVllmUpdates()
    const current = activeOnecatVllm.value?.version || ovllm.value?.version || null
    const latest = raw?.latest_version || null
    const updateAvailable = latest && current !== latest
    onecatVllmUpdateInfo.value = {
      update_available: updateAvailable,
      latest_version: latest,
      current_version: current,
    }
  } catch (e) {
    toast.add({
      severity: 'warn',
      summary: 'Could not check updates',
      detail: e.message,
      life: 3000,
    })
  } finally {
    checkingOnecatVllm.value = false
  }
}

async function installOnecatVllmRelease() {
  onecatVllmInstalling.value = true
  try {
    await enginesStore.saveOnecatVllmBuildSettings({
      ...onecatVllmBuildForm.value,
      release_version: ovllmReleaseVersion.value || '',
      source_repo: ovllmSourceRepo.value || onecatVllmBuildForm.value.source_repo,
      source_branch: ovllmSourceBranch.value || onecatVllmBuildForm.value.source_branch,
    })
    await enginesStore.installOnecatVllm(
      ovllmReleaseVersion.value ? { version: ovllmReleaseVersion.value } : {},
    )
    toast.add({
      severity: 'success',
      summary: '1Cat-vLLM install started',
      detail: 'Track progress in notifications',
      life: 3000,
    })
  } catch (e) {
    toast.add({ severity: 'error', summary: 'Failed', detail: e.message, life: 4000 })
  } finally {
    onecatVllmInstalling.value = false
    ovllmReleaseDialogVisible.value = false
  }
}

async function installOnecatVllmSource() {
  onecatVllmInstalling.value = true
  try {
    await enginesStore.saveOnecatVllmBuildSettings({
      ...onecatVllmBuildForm.value,
      source_repo: ovllmSourceRepo.value,
      source_branch: ovllmSourceBranch.value,
      release_version: ovllmReleaseVersion.value || onecatVllmBuildForm.value.release_version,
    })
    await enginesStore.installOnecatVllmFromSource({
      repo_url: ovllmSourceRepo.value,
      branch: ovllmSourceBranch.value,
    })
    toast.add({
      severity: 'success',
      summary: 'Source build started',
      detail: 'Track progress in notifications',
      life: 3000,
    })
  } catch (e) {
    toast.add({ severity: 'error', summary: 'Failed', detail: e.message, life: 4000 })
  } finally {
    onecatVllmInstalling.value = false
    ovllmSourceDialogVisible.value = false
  }
}

// ── Shared native llama-family dialog (llama.cpp / ik_llama / Unsloth) ──
// The three near-duplicate templates (header + body) are ONE template pair
// now, parameterized by this per-engine config. Real divergences stay as
// data: the engine-mark leading (no pi icon), unsloth's MISSING settings
// action, its raw <p> note instead of the hint component, raw (un-prefixed)
// banner versions vs the venv `v` prefix, ik's "View commit" link label,
// per-engine install button sets, and VersionTable can-verify (llama/ik
// only). Byte-pinned by EnginesView.tasks.test.js ("native dialog…").
const NATIVE_DIALOG_CONFIG = {
  llama_cpp: {
    markClass: 'engine-mark--llama',
    markText: 'L',
    statusEngineId: 'llama.cpp',
    settings: true,
    active: activeLlamaCpp,
    versions: () => enginesStore.llamaVersions,
    hintEngineKey: 'llama_cpp',
    hintKey: () => `llama-hint-${hintRevLlama.value}`,
    noteRaw: '',
    checking: checkingLlamaCpp,
    checkHint: 'Compare the installed ref to the latest stable release (vX.Y.Z).',
    updateInfo: llamaCppUpdateInfo,
    updating: () => updatingEngine.value === 'llama_cpp',
    updateTooltip: 'Update using saved build settings',
    installSubtitle:
      'Add a build of the latest stable release (vX.Y.Z) or any git ref. Each build is a version you can activate.',
    installButtons: () => [
      {
        label: 'From release',
        icon: 'pi pi-tag',
        severity: 'success',
        loading: llamaReleaseInstalling.value,
        disabled: llamaReleaseInstalling.value || llamaCppSourceInstalling.value,
        onClick: installLlamaLatestRelease,
      },
      {
        label: 'From source',
        icon: 'pi pi-code',
        severity: 'info',
        loading: llamaCppSourceInstalling.value,
        disabled: llamaReleaseInstalling.value || llamaCppSourceInstalling.value,
        onClick: openLlamaCppSourceDialog,
      },
    ],
    statusRows: llamaCppActiveStatusRows,
    canVerify: true,
    emptyMessage: 'No versions yet. Install one using the options above.',
    onCheck: () => checkLlamaCppUpdates(),
    onOpenSettings: () => openBuildDialog('llama_cpp'),
    onUpdate: () => doUpdateEngine('llama_cpp'),
  },
  ik_llama: {
    markClass: 'engine-mark--ik',
    markText: 'IK',
    statusEngineId: 'ik_llama.cpp',
    settings: true,
    active: activeIkLlama,
    versions: () => enginesStore.ikLlamaVersions,
    hintEngineKey: 'ik_llama',
    hintKey: () => `ik-hint-${hintRevIk.value}`,
    noteRaw: '',
    checking: checkingIkLlama,
    checkHint: 'Compare the installed tip to the latest commit on main.',
    updateInfo: ikLlamaUpdateInfo,
    updateLinkLabel: 'View commit',
    updating: () => updatingEngine.value === 'ik_llama',
    updateTooltip: 'Rebuild tip of main using saved build settings',
    installSubtitle:
      'ik_llama.cpp has no release tags here. Build the tip of main, or any git repo/ref. Each build is a version you can activate.',
    installButtons: () => [
      {
        label: 'From tip',
        icon: 'pi pi-bolt',
        severity: 'success',
        loading: ikTipInstalling.value,
        disabled: ikTipInstalling.value || ikLlamaSourceInstalling.value,
        onClick: installIkFromTip,
      },
      {
        label: 'From source',
        icon: 'pi pi-code',
        severity: 'info',
        loading: ikLlamaSourceInstalling.value,
        disabled: ikTipInstalling.value || ikLlamaSourceInstalling.value,
        onClick: openIkLlamaSourceDialog,
      },
    ],
    statusRows: ikLlamaActiveStatusRows,
    canVerify: true,
    emptyMessage: 'No versions yet. Install one using the options above.',
    onCheck: () => checkIkLlamaUpdates(),
    onOpenSettings: () => openBuildDialog('ik_llama'),
    onUpdate: () => doUpdateEngine('ik_llama'),
  },
  unsloth_llama: {
    markClass: 'engine-mark--unsloth',
    markText: 'US',
    statusEngineId: 'unsloth_llama',
    settings: false,
    active: activeUnslothLlama,
    versions: () => enginesStore.unslothLlamaVersions,
    hintEngineKey: '',
    hintKey: () => '',
    noteRaw:
      "Installs Unsloth's published llama-server: an upstream llama.cpp nightly plus their " +
      'pinned patch mix. This is the inference binary, not Unsloth Studio.',
    checking: checkingUnslothLlama,
    checkHint: 'Compare the active mix tag to the latest unslothai/llama.cpp prebuilt.',
    updateInfo: unslothLlamaUpdateInfo,
    updating: () => unslothLlamaInstalling.value,
    updateTooltip: 'Install the latest Unsloth llama-server mix',
    installSubtitle:
      'Download a Linux x64 prebuilt (CUDA 12/13 portable when a matching toolkit is active, otherwise CPU). Each release is a version you can activate.',
    installButtons: () => [
      {
        label: 'Install prebuilt',
        icon: 'pi pi-download',
        severity: 'success',
        loading: unslothLlamaInstalling.value,
        disabled: unslothLlamaInstalling.value,
        onClick: installUnslothLlamaRelease,
      },
    ],
    statusRows: unslothLlamaActiveStatusRows,
    canVerify: false,
    emptyMessage: 'No versions yet. Install a prebuilt using the option above.',
    onCheck: () => checkUnslothLlamaUpdates(),
    onOpenSettings: () => {},
    onUpdate: () => installUnslothLlamaRelease(),
  },
}

const nativeDialog = computed(() => {
  const engineId = selectedEngine.value
  const cfg = Object.hasOwn(NATIVE_DIALOG_CONFIG, engineId) ? NATIVE_DIALOG_CONFIG[engineId] : null
  if (!cfg) return null
  return {
    engineId,
    title: ENGINE_DESCRIPTORS[engineId].label,
    markClass: cfg.markClass,
    markText: cfg.markText,
    statusEngineId: cfg.statusEngineId,
    settings: cfg.settings,
    activeVersion: cfg.active.value,
    versions: cfg.versions(),
    hintEngineKey: cfg.hintEngineKey,
    hintKey: cfg.hintKey(),
    noteRaw: cfg.noteRaw,
    checking: cfg.checking.value,
    checkHint: cfg.checkHint,
    updateInfo: cfg.updateInfo.value,
    updateLinkLabel: cfg.updateLinkLabel,
    updating: cfg.updating(),
    updateTooltip: cfg.updateTooltip,
    installSubtitle: cfg.installSubtitle,
    installButtons: cfg.installButtons(),
    statusRows: cfg.statusRows.value,
    canVerify: cfg.canVerify,
    emptyMessage: cfg.emptyMessage,
    onCheck: cfg.onCheck,
    onOpenSettings: cfg.onOpenSettings,
    onUpdate: cfg.onUpdate,
  }
})

// ── Shared venv dialog (LMDeploy + 1Cat-vLLM) ────────────────
// The two non-panel venv dialogs were near-duplicate templates; they are
// ONE template pair now, parameterized by this per-engine config plus
// registry data (label, dialog icon, store versions key). Per-engine
// divergences (update link/label, install copy, V100 note, hint-rev and
// state refs) stay as data here. Byte-pinned by EnginesView.tasks.test.js
// ("lmdeploy dialog…" / "1Cat dialog…").

// The installer-panel family (SGLang, its SM70 forks and the
// OpenAI-compatible server engine) shares the dialog HEADER shape
// (title/icon/tag/settings/reload/rescan) but its body is SglangEnginePanel and
// its actions drive the panel ref. Declared once so a new panel engine is one
// entry, not five repeated fields. Panel family: no aria/tooltip on the
// Reload/Rescan buttons — a real divergence preserved via buttonA11y.
const SGLANG_PANEL_DIALOG = {
  panel: true,
  settingsAriaLabel: 'Install settings',
  buttonA11y: false,
  active: selectedSglangActive,
  versions: selectedSglangVersions,
}

const VENV_DIALOG_CONFIG = {
  lmdeploy: {
    settingsAriaLabel: 'Build settings',
    buttonA11y: true,
    checkHint: 'Compare the active install to the latest PyPI release.',
    updateLinkUrl: 'https://pypi.org/project/lmdeploy/',
    updateLinkLabel: 'View on PyPI',
    updateTooltip: 'Install the latest PyPI version as a new environment',
    installSubtitle:
      'Add a new Python environment from PyPI or a git source. Each install is a version you can activate.',
    installPrimaryLabel: 'From PyPI',
    note: '',
    checking: checkingLmdeploy,
    updateInfo: lmdeployUpdateInfo,
    updating: updatingLmdeploy,
    hintRev: hintRevLmdeploy,
    active: activeLmdeploy,
    statusRows: lmdeployActiveStatusRows,
    onCheck: () => checkLmdeployUpdates(),
    onOpenSettings: () => openLmdeployBuildSettings(),
    onRefresh: () => refreshEngineDialogStatus('lmdeploy'),
    onUpdate: () => doUpdateLmdeploy(),
    onInstallPrimary: () => openLmdeployPipDialog(),
    onInstallSource: () => openLmdeploySourceDialog(),
  },
  '1cat_vllm': {
    settingsAriaLabel: 'Build settings',
    buttonA11y: true,
    checkHint: 'Compare the active install to the latest GitHub release.',
    updateLinkUrl: 'https://github.com/1CatAI/1Cat-vLLM/releases/latest',
    // No updateLinkLabel: the banner's default ("View release") applies.
    updateTooltip: 'Install the latest release wheels as a new environment',
    installSubtitle:
      'Add a new environment from prebuilt release wheels (recommended) or build from source. Each install is a version you can activate.',
    installPrimaryLabel: 'From release',
    note: 'vLLM fork for Tesla V100 (SM70). Release installs use a prebuilt CUDA 12.8 wheel. Source builds need an SM70 GPU and the CUDA 12.8 toolkit.',
    checking: checkingOnecatVllm,
    updateInfo: onecatVllmUpdateInfo,
    updating: updatingOnecatVllm,
    hintRev: hintRevOnecat,
    active: activeOnecatVllm,
    statusRows: onecatVllmActiveStatusRows,
    onCheck: () => checkOnecatVllmUpdates(),
    onOpenSettings: () => openOnecatVllmBuildSettings(),
    onRefresh: () => refreshEngineDialogStatus('1cat_vllm'),
    onUpdate: () => doUpdateOnecatVllm(),
    onInstallPrimary: () => openOnecatVllmReleaseDialog(),
    onInstallSource: () => openOnecatVllmSourceDialog(),
  },
  sglang: {
    ...SGLANG_PANEL_DIALOG,
    onOpenSettings: () => sglangPanelRefs.sglang?.openSettings(),
    onRefresh: () => sglangPanelRefs.sglang?.refresh(),
  },
  sglang_v100: {
    ...SGLANG_PANEL_DIALOG,
    onOpenSettings: () => sglangPanelRefs.sglang_v100?.openSettings(),
    onRefresh: () => sglangPanelRefs.sglang_v100?.refresh(),
  },
  sglang_sxm2: {
    ...SGLANG_PANEL_DIALOG,
    onOpenSettings: () => sglangPanelRefs.sglang_sxm2?.openSettings(),
    onRefresh: () => sglangPanelRefs.sglang_sxm2?.refresh(),
  },
  sglang_sm70: {
    ...SGLANG_PANEL_DIALOG,
    onOpenSettings: () => sglangPanelRefs.sglang_sm70?.openSettings(),
    onRefresh: () => sglangPanelRefs.sglang_sm70?.refresh(),
  },
  vllm: {
    ...SGLANG_PANEL_DIALOG,
    onOpenSettings: () => sglangPanelRefs.vllm?.openSettings(),
    onRefresh: () => sglangPanelRefs.vllm?.refresh(),
  },
}

const venvDialogHeader = computed(() => {
  const engineId = selectedEngine.value
  const cfg = Object.hasOwn(VENV_DIALOG_CONFIG, engineId) ? VENV_DIALOG_CONFIG[engineId] : null
  if (!cfg) return null
  const descriptor = ENGINE_DESCRIPTORS[engineId]
  return {
    engineId,
    title: descriptor.label,
    icon: descriptor.dialogIcon,
    settingsAriaLabel: cfg.settingsAriaLabel,
    buttonA11y: cfg.buttonA11y,
    activeVersion: cfg.active.value,
    versions: cfg.versions ? cfg.versions.value : enginesStore[descriptor.versionsKey],
    onOpenSettings: cfg.onOpenSettings,
    onRefresh: cfg.onRefresh,
  }
})

const venvDialog = computed(() => {
  const engineId = selectedEngine.value
  const cfg = Object.hasOwn(VENV_DIALOG_CONFIG, engineId) ? VENV_DIALOG_CONFIG[engineId] : null
  if (!cfg || cfg.panel) return null
  const descriptor = ENGINE_DESCRIPTORS[engineId]
  return {
    engineId,
    title: descriptor.label,
    icon: descriptor.dialogIcon,
    versions: enginesStore[descriptor.versionsKey],
    hintKey: `${engineId}-hint-${cfg.hintRev.value}`,
    activeVersion: cfg.active.value,
    checking: cfg.checking.value,
    checkHint: cfg.checkHint,
    updateInfo: cfg.updateInfo.value,
    updating: cfg.updating.value,
    updateLinkUrl: cfg.updateLinkUrl,
    updateLinkLabel: cfg.updateLinkLabel,
    updateTooltip: cfg.updateTooltip,
    note: cfg.note,
    installSubtitle: cfg.installSubtitle,
    installPrimaryLabel: cfg.installPrimaryLabel,
    statusRows: cfg.statusRows.value,
    onCheck: cfg.onCheck,
    onOpenSettings: cfg.onOpenSettings,
    onUpdate: cfg.onUpdate,
    onInstallPrimary: cfg.onInstallPrimary,
    onInstallSource: cfg.onInstallSource,
  }
})

// ── Lifecycle ──────────────────────────────────────────────
let unsubscribeTaskUpdated = null

watch(buildDialogVisible, (visible) => {
  if (!visible) editingVersion.value = null
})

watch(audioCppBuildDialogVisible, (visible) => {
  if (!visible) editingAudioVersion.value = null
})

onMounted(() => {
  enginesStore.fetchAll()
  unsubscribeTaskUpdated = progressStore.subscribe('task_updated', async (task) => {
    if (task?.status !== 'completed' && task?.status !== 'failed') return

    const manager = task?.metadata?.manager

    if (manager === 'cuda') {
      await enginesStore.fetchCudaStatus()
      return
    }

    const managerEngineId = engineIdForManager(manager)
    if (
      manager &&
      !managerEngineId &&
      serverBackedEngines(enginesStore.engineDescriptors).some((d) => d.id === manager)
    ) {
      // YAML campaign S6: task completion for a YAML-onboarded engine —
      // refresh its dynamic status + version rows (no local registry id).
      await Promise.allSettled([
        enginesStore.venvFetchStatus(manager),
        enginesStore.fetchLlamaVersions(),
      ])
      return
    }
    if (managerEngineId) {
      // One branch for every generic-router engine (registry-owned manager
      // ids). The two hand-rolled branches it replaces differed only in
      // toast detail fallback, life, and which status fetch ran — kept
      // byte-identical as data below (pinned in EnginesView.tasks.test.js).
      const isPanelFamily = SG_PANEL_ENGINE_IDS.includes(managerEngineId)
      if (task.status === 'failed') {
        toast.add({
          severity: 'error',
          summary: `${engineLabel(manager)} install failed`,
          detail:
            task.message ||
            (isPanelFamily
              ? 'Engine operation failed'
              : `${engineLabel(manager)} operation failed`),
          life: isPanelFamily ? 6000 : 5000,
        })
      }
      const statusRefresh = isPanelFamily
        ? enginesStore.fetchSglangStatus(managerEngineId)
        : managerEngineId === '1cat_vllm'
          ? enginesStore.fetchOnecatVllmStatus()
          : enginesStore.fetchLmdeployStatus()
      await Promise.allSettled([statusRefresh, enginesStore.fetchLlamaVersions()])
      return
    }

    if (manager === 'unsloth_llama') {
      if (task.status === 'failed') {
        toast.add({
          severity: 'error',
          summary: `${engineLabel(manager)} install failed`,
          detail: task.message || 'Engine operation failed',
          life: 6000,
        })
      }
      await enginesStore.fetchLlamaVersions()
      return
    }

    if (task?.type === 'build') {
      const refreshTasks = [enginesStore.fetchLlamaVersions(), enginesStore.fetchSystemStatus()]
      if (task.metadata?.engine === audioEngineId) {
        refreshTasks.push(enginesStore.fetchAudioCppStatus())
      }
      await Promise.allSettled(refreshTasks)
    }
  })
})

onUnmounted(() => {
  if (unsubscribeTaskUpdated) unsubscribeTaskUpdated()
})
</script>

<style scoped>
/* layout: .page-shell.page-shell--relaxed */

/* ── Collapse transition ─────────────────────────────── */
.ev-collapse-enter-active,
.ev-collapse-leave-active {
  transition: all 0.2s ease;
  overflow: hidden;
}
.ev-collapse-enter-from,
.ev-collapse-leave-to {
  max-height: 0;
  opacity: 0;
}
.ev-collapse-enter-to,
.ev-collapse-leave-from {
  max-height: 600px;
  opacity: 1;
}

/* ── Section ─────────────────────────────────────────── */
.ev-section {
  background: color-mix(in srgb, var(--bg-card) 94%, transparent);
  border: 1px solid var(--border-primary);
  border-radius: var(--radius-xl);
  overflow: hidden;
  box-shadow: var(--shadow-sm);
}

.ev-section-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 0.5rem;
  flex-wrap: wrap;
  background: color-mix(in srgb, var(--bg-surface) 76%, transparent);
  border-bottom: 1px solid var(--border-primary);
  user-select: none;
}

.ev-section-header__toggle {
  flex: 1;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 0.5rem;
  min-width: 0;
  margin: 0;
  padding: 0.75rem 0 0.75rem 1.25rem;
  border: none;
  background: transparent;
  font: inherit;
  color: inherit;
  cursor: pointer;
  text-align: left;
}

.ev-section-chevron {
  flex-shrink: 0;
  margin-right: 0.5rem;
  color: var(--text-secondary);
}

.ev-section-actions {
  display: flex;
  align-items: center;
  gap: 0.25rem;
  padding: 0.45rem 1.25rem 0.45rem 0;
  flex-shrink: 0;
}

.ev-section-title {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  flex: 1;
  min-width: 0;
}

/* Modal subpanels: title + actions (no toggle button) */
.ev-section-header > .ev-section-title:first-child {
  padding: 0.75rem 0 0.75rem 1.25rem;
}

.ev-section-title h2 {
  font-size: 1rem;
  font-weight: 600;
  margin: 0;
  line-height: 1.25;
  color: var(--text-primary);
}

:deep(.engine-mark) {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  min-width: 1.8rem;
  height: 1.8rem;
  padding: 0 0.45rem;
  border-radius: var(--radius-sm);
  font-size: var(--text-xs);
  font-weight: 700;
  line-height: 1;
  letter-spacing: 0.04em;
  color: var(--text-secondary);
  background: var(--bg-tertiary);
  border: 1px solid var(--border-primary);
}

.ev-section-body {
  padding: 1.25rem;
  background: linear-gradient(
    180deg,
    color-mix(in srgb, var(--accent-blue) 3%, transparent),
    transparent 12rem
  );
}

/* ── Engines overview ───────────────────────────────────── */
.engine-filters {
  display: flex;
  flex-wrap: wrap;
  gap: 0.75rem;
  align-items: end;
  margin-bottom: 1rem;
  padding: 0.75rem;
  border: 1px solid var(--border-primary);
  border-radius: var(--radius-lg);
  background: color-mix(in srgb, var(--bg-surface) 74%, transparent);
  font-size: 0.85rem;
}

.engine-filters select {
  display: block;
  margin-top: 0.375rem;
  min-height: 2.75rem;
  max-width: 100%;
  padding: 0.5rem 0.75rem;
  border: 1px solid var(--border-secondary);
  border-radius: var(--radius-md);
  background: var(--bg-secondary);
  color: var(--text-primary);
}

.engine-filters__reset {
  min-height: 2.75rem;
  padding: 0.5rem;
  border: none;
  background: transparent;
  color: var(--accent-cyan);
  font: inherit;
  cursor: pointer;
}

.engine-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(min(100%, 17rem), 1fr));
  gap: 0.85rem;
}

:deep(.engine-card) {
  display: flex;
  flex-direction: column;
  min-height: 11.5rem;
  appearance: none;
  border: 1px solid var(--border-primary);
  background: var(--bg-card);
  border-radius: var(--radius-lg);
  padding: 1rem;
  text-align: left;
  color: inherit;
  cursor: pointer;
  box-shadow: var(--shadow-sm);
  transition:
    border-color 0.15s ease,
    transform 0.15s ease,
    background 0.15s ease;
}

:deep(.engine-card):hover {
  border-color: var(--accent-cyan);
  background: var(--bg-surface);
  box-shadow: var(--shadow-sm);
}

:deep(.engine-card-cta) {
  display: inline-flex;
  align-items: center;
  gap: 0.35rem;
  align-self: flex-start;
  margin-top: auto;
  padding-top: 0.9rem;
  color: var(--accent-cyan);
  font-size: var(--text-xs);
  font-weight: 700;
}

:deep(.engine-card-cta)::after {
  content: '\2192';
  transition: transform var(--transition-fast);
}

:deep(.engine-card:hover .engine-card-cta)::after {
  transform: translateX(0.2rem);
}

:deep(.engine-card):focus {
  outline: none;
}

:deep(.engine-card):focus-visible {
  outline: 2px solid var(--accent-cyan);
  outline-offset: 2px;
}

:deep(.engine-card-head) {
  display: flex;
  align-items: flex-start;
  justify-content: flex-start;
  gap: 0.75rem;
}

:deep(.engine-card-version-line) {
  min-width: 0;
  max-width: 100%;
  margin-bottom: 0.45rem;
}

:deep(.engine-card-version-line .p-tag) {
  max-width: 100%;
}

:deep(.engine-card-version-line .p-tag-value) {
  display: block;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.engine-dialog-tag-clip {
  display: inline-flex;
  min-width: 0;
  max-width: min(16rem, 42vw);
  vertical-align: middle;
}

.engine-dialog-tag-clip :deep(.p-tag) {
  max-width: 100%;
}

.engine-dialog-tag-clip :deep(.p-tag-value) {
  display: block;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

:deep(.engine-card-title) {
  display: flex;
  align-items: center;
  gap: 0.65rem;
  min-width: 0;
}

:deep(.engine-card-name) {
  font-size: 1rem;
  font-weight: 680;
}

:deep(.engine-card-meta) {
  font-size: var(--text-xs);
  color: var(--text-secondary);
  margin-top: 0.1rem;
}

:deep(.engine-card-icon) {
  display: inline-grid;
  place-items: center;
  font-size: 0.875rem;
  color: var(--text-secondary);
  width: 1.8rem;
  height: 1.8rem;
  border: 1px solid var(--border-primary);
  border-radius: var(--radius-sm);
  background: var(--bg-tertiary);
  flex-shrink: 0;
}

:deep(.engine-card-body) {
  margin-top: 0.8rem;
}

:deep(.engine-card-status) {
  font-size: 0.82rem;
  color: var(--text-secondary);
}

:deep(.engine-card-status--warning) {
  color: var(--status-warning);
  font-weight: 600;
}

/* ── Actions ─────────────────────────────────────────── */
.ev-actions {
  display: flex;
  gap: 0.5rem;
  align-items: center;
  flex-wrap: wrap;
  margin-bottom: 0.75rem;
}

.ev-subsection {
  margin-top: 1.25rem;
}
.ev-subsection h4 {
  font-size: var(--text-xs);
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: var(--text-secondary, #9ca3af);
  margin: 0 0 0.5rem;
}

.ev-form {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}

.form-row {
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.form-row label {
  font-size: 0.875rem;
  width: 88px;
  flex-shrink: 0;
  color: var(--text-secondary);
}

.form-input {
  flex: 1;
}
.form-input-short {
  width: 140px;
}

/* ── Status details ──────────────────────────────────── */
.status-detail {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  font-size: 0.875rem;
  margin-bottom: 0.75rem;
  flex-wrap: wrap;
}

.detail-label {
  color: var(--text-secondary);
  flex-shrink: 0;
}

.detail-label.detail-label--error {
  color: var(--status-error);
}
.ml {
  margin-left: 0.75rem;
}

code {
  background: var(--bg-surface);
  padding: 0.1em 0.4em;
  border-radius: 0.25rem;
  font-size: 0.8rem;
  font-family: monospace;
  word-break: break-all;
}

/* ── Version list (CUDA / table-like) ───────────────── */
.ev-version-list {
  display: flex;
  flex-direction: column;
  gap: 0.25rem;
  margin-bottom: 0.75rem;
}

.ev-version-row {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.5rem 0.75rem;
  background: var(--bg-surface);
  border: 1px solid var(--border-primary);
  border-radius: var(--radius-md);
  font-size: 0.875rem;
}

.ev-version-row .version-name {
  flex: 1;
  margin: 0;
}

.empty-state-mini {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.75rem;
  color: var(--text-secondary);
  font-size: 0.875rem;
  margin-bottom: 0.75rem;
}

.empty-state-mini i {
  color: var(--text-muted);
}

.cuda-version-select {
  min-width: 160px;
}
.lm-version-input {
  width: 220px;
}

/* ── Update banners ──────────────────────────────────── */
.update-banner,
.update-current {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.5rem 0.75rem;
  border-radius: var(--radius-md);
  font-size: 0.875rem;
  margin-bottom: 0.75rem;
}

.update-banner {
  background: var(--status-warning-soft);
  border: 1px solid rgba(245, 158, 11, 0.3);
  color: var(--status-warning);
}

.update-current {
  background: var(--status-success-soft);
  border: 1px solid rgba(16, 185, 129, 0.3);
  color: var(--status-success);
}

.update-link {
  color: inherit;
  margin-left: 0.5rem;
  text-decoration: underline;
  opacity: 0.8;
}

/* ── Dialog ──────────────────────────────────────────── */
.dialog-body {
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
}

.dialog-loading {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 0.75rem;
  padding: 2rem 0;
  color: var(--text-secondary);
}

.engine-modal-body {
  display: flex;
  flex-direction: column;
  gap: 1rem;
  padding-top: 0.35rem;
}

.ev-section--modal {
  border: 0;
  background: transparent;
}

.form-field {
  display: flex;
  flex-direction: column;
  gap: 0.25rem;
}

.form-field label {
  font-size: 0.875rem;
  font-weight: 500;
  color: var(--text-secondary);
}

.form-field small {
  font-size: var(--text-xs);
  color: var(--text-secondary);
}
.optional {
  font-weight: 400;
  opacity: 0.6;
}

.asset-list {
  display: flex;
  flex-direction: column;
  gap: 0.25rem;
  max-height: 240px;
  overflow-y: auto;
}

.asset-option {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.4rem 0.6rem;
  border-radius: var(--radius-md, 0.5rem);
  cursor: pointer;
  border: 1px solid transparent;
  transition: background 0.15s;
}

.asset-option:hover {
  background: var(--bg-surface);
}
.asset-option.selected {
  background: var(--bg-surface);
  border-color: var(--accent-cyan);
}

.asset-name {
  flex: 1;
  font-size: 0.8rem;
  font-family: monospace;
}
.asset-size {
  font-size: var(--text-xs);
  color: var(--text-secondary);
}

.build-note {
  font-size: 0.8rem;
  padding: 0.5rem 0.6rem;
  border-radius: 6px;
  margin-bottom: 0.5rem;
}
.build-note--info {
  background: var(--status-info-soft);
  color: var(--text-primary);
  border: 1px solid var(--border-primary);
}
.build-note strong {
  font-weight: 600;
}

@media (max-width: 768px) {
  .ev-section-body {
    padding: 0.85rem;
  }

  .ev-section-header__toggle {
    flex: 1 1 12rem;
    padding-left: 0.85rem;
  }

  .ev-section-actions {
    padding-right: 0.65rem;
    margin-left: auto;
  }
}

/* The catalog is the page's primary content, without another enclosing card. */
.engine-catalog {
  border: 0;
  border-radius: 0;
  background: transparent;
  box-shadow: none;
  overflow: visible;
}
.engine-catalog > .ev-section-header {
  background: transparent;
  border-bottom: 0;
}
.engine-catalog .ev-section-header__toggle {
  padding-left: 0;
}
.engine-catalog .ev-section-actions {
  padding-right: 0;
}
.engine-catalog .ev-section-body {
  padding: 0;
  background: transparent;
}
.engine-catalog .engine-filters {
  padding: 0 0 1.25rem;
  background: transparent;
  border: 0;
  border-radius: 0;
}
.engine-catalog .engine-grid {
  grid-template-columns: repeat(auto-fill, minmax(min(100%, 19rem), 1fr));
  gap: 0.75rem;
}

.verify-commands {
  list-style: none;
  margin: 0.75rem 0 0;
  padding: 0;
  display: grid;
  gap: 0.45rem;
}

.verify-commands code {
  display: block;
  margin-top: 0.15rem;
  font-size: var(--text-xs);
  word-break: break-all;
}
</style>
