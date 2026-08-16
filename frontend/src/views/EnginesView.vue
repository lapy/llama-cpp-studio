<template>
  <div class="engines-view page-shell page-shell--relaxed">

    <!-- ── System Info ─────────────────────────────────────── -->
    <section class="ev-section">
      <div class="ev-section-header">
        <button
          type="button"
          class="ev-section-header__toggle interactive-row"
          :aria-expanded="systemExpanded"
          aria-controls="ev-section-system-body"
          @click="systemExpanded = !systemExpanded"
        >
          <div class="ev-section-title">
            <i class="pi pi-desktop" aria-hidden="true" />
            <h2>System</h2>
          </div>
          <i :class="['pi', 'ev-section-chevron', systemExpanded ? 'pi-chevron-up' : 'pi-chevron-down']" aria-hidden="true" />
        </button>
        <div class="ev-section-actions">
          <Button icon="pi pi-refresh" text severity="secondary" size="small"
            :loading="enginesStore.loading" @click="enginesStore.fetchSystemStatus()" />
        </div>
      </div>
      <Transition name="ev-collapse">
        <div v-if="systemExpanded" id="ev-section-system-body" class="ev-section-body">
          <div class="ev-system-layout">
            <div class="metrics-grid metrics-grid--resources" role="region" aria-label="CPU, memory, and disk">
              <div class="metric-card">
                <i class="pi pi-desktop metric-icon" />
                <div class="metric-data">
                  <div class="metric-label">CPU</div>
                  <div class="metric-value">{{ (sys.cpu_percent || 0).toFixed(1) }}%</div>
                  <ProgressBar :value="sys.cpu_percent || 0" :showValue="false" class="metric-bar" />
                </div>
              </div>
              <div class="metric-card">
                <i class="pi pi-database metric-icon" />
                <div class="metric-data">
                  <div class="metric-label">Memory</div>
                  <div class="metric-value">
                    {{ formatBytesIEC(memUsedBytes) }} / {{ formatBytesIEC(sys.memory?.total) }} ({{ memPercent }}%)
                  </div>
                  <ProgressBar :value="memPercent" :showValue="false" class="metric-bar" />
                </div>
              </div>
              <div class="metric-card">
                <i class="pi pi-save metric-icon" />
                <div class="metric-data">
                  <div class="metric-label">Disk</div>
                  <div class="metric-value">
                    {{ formatBytesIEC(sys.disk?.used) }} / {{ formatBytesIEC(sys.disk?.total) }} ({{ diskPercent }}%)
                  </div>
                  <ProgressBar :value="diskPercent" :showValue="false" class="metric-bar" />
                </div>
              </div>
            </div>

            <div class="cuda-toolkit-region" role="region" aria-labelledby="cuda-toolkit-heading">
              <div class="cuda-toolkit-main">
                <div class="cuda-toolkit-main__icon" aria-hidden="true">
                  <i class="pi pi-bolt" />
                </div>
                <div class="cuda-toolkit-main__body">
                  <h3 id="cuda-toolkit-heading" class="cuda-toolkit-main__title">CUDA Toolkit</h3>
                  <p class="cuda-toolkit-main__status">
                    <template v-if="cuda.installed">CUDA {{ cuda.version || '?' }}</template>
                    <template v-else>Not installed</template>
                  </p>
                  <p class="cuda-toolkit-main__hint">
                    <template v-if="cuda.installed_versions?.length">
                      {{ cuda.installed_versions.length }} version{{ cuda.installed_versions.length === 1 ? '' : 's' }} detected
                    </template>
                    <template v-else-if="cuda.cuda_path">
                      {{ cuda.cuda_path }}
                    </template>
                    <template v-else>
                      Build support and toolkit management
                    </template>
                  </p>
                </div>
                <div class="cuda-toolkit-main__actions">
                  <Button icon="pi pi-refresh" text severity="secondary" size="small"
                    v-tooltip.top="'Reload CUDA status'"
                    @click.stop="enginesStore.fetchCudaStatus()" />
                  <Button label="Install" icon="pi pi-download" severity="success" outlined size="small"
                    @click.stop="cudaInstallDialogVisible = true" />
                </div>
              </div>

              <div
                v-if="cuda.installed || cuda.installed_versions?.length"
                class="cuda-toolkit-details"
              >
                <div v-if="cuda.installed" class="status-detail">
                  <span class="detail-label">CUDA path</span>
                  <code>{{ cuda.cuda_path || 'unknown' }}</code>
                </div>

                <div v-if="cuda.installed_versions?.length" class="ev-version-list">
                  <div v-for="v in cuda.installed_versions" :key="v.version" class="ev-version-row">
                    <code class="version-name">CUDA {{ v.version }}</code>
                    <Tag v-if="v.is_current" value="Active" severity="success" />
                    <Button icon="pi pi-trash" text severity="danger" size="small"
                      @click="confirmUninstallCuda(v.version)" />
                  </div>
                </div>
              </div>
            </div>

            <div
              v-if="gpus.length"
              class="metrics-grid metrics-grid--gpus"
              role="region"
              aria-label="GPU memory"
            >
              <div v-for="(gpuItem, idx) in gpus" :key="gpuItem.index ?? gpuItem.uuid ?? gpuItem.name ?? idx" class="metric-card">
                <i class="pi pi-bolt metric-icon" />
                <div class="metric-data">
                  <div class="metric-label">GPU — {{ gpuItem.name }}</div>
                  <div class="metric-value">
                    {{ formatBytesIEC(gpuVramUsedBytes(gpuItem)) }} /
                    {{ formatBytesIEC(gpuVramTotalBytes(gpuItem)) }} VRAM
                  </div>
                  <ProgressBar :value="gpuPercent(gpuItem)" :showValue="false" class="metric-bar" />
                </div>
              </div>
            </div>
          </div>
        </div>
      </Transition>
    </section>

    <!-- ── Virtual models & profiles (llama-swap) ─────────── -->
    <section id="ev-section-routing" class="ev-section">
      <div class="ev-section-header">
        <button
          type="button"
          class="ev-section-header__toggle interactive-row"
          :aria-expanded="routingExpanded"
          aria-controls="ev-section-routing-body"
          @click="routingExpanded = !routingExpanded"
        >
          <div class="ev-section-title">
            <i class="pi pi-sitemap" aria-hidden="true" />
            <h2>Virtual models &amp; profiles</h2>
          </div>
          <i :class="['pi', 'ev-section-chevron', routingExpanded ? 'pi-chevron-up' : 'pi-chevron-down']" aria-hidden="true" />
        </button>
        <div class="ev-section-actions">
          <Button
            icon="pi pi-refresh"
            text
            severity="secondary"
            size="small"
            :loading="routingPanel?.loading"
            v-tooltip.top="'Reload virtual models & profiles'"
            aria-label="Reload virtual models and profiles"
            @click="routingPanel?.reload()"
          />
          <Button
            v-if="routingPanel?.showApplyLlamaSwap"
            label="Apply"
            icon="pi pi-bolt"
            size="small"
            severity="warning"
            :loading="routingPanel?.applying"
            :disabled="routingPanel?.saving || routingPanel?.applying"
            v-tooltip.top="'Regenerate llama-swap-config.yaml and reload the proxy (stops all loaded models)'"
            @click="routingPanel?.applyConfig()"
          />
          <Button
            label="Save"
            icon="pi pi-save"
            size="small"
            :loading="routingPanel?.saving"
            :disabled="!routingPanel?.dirty || routingPanel?.saving"
            @click="routingPanel?.save()"
          />
        </div>
      </div>
      <Transition name="ev-collapse">
        <div v-if="routingExpanded" id="ev-section-routing-body" class="ev-section-body">
          <SwapRoutingPanel ref="routingPanel" />
        </div>
      </Transition>
    </section>

    <!-- ── Engines Overview ───────────────────────────────── -->
    <section class="ev-section">
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
            <h2>Engines</h2>
          </div>
          <i :class="['pi', 'ev-section-chevron', enginesExpanded ? 'pi-chevron-up' : 'pi-chevron-down']" aria-hidden="true" />
        </button>
        <div class="ev-section-actions">
          <Button icon="pi pi-refresh" text severity="secondary" size="small"
            @click="refreshEnginesOverview" />
        </div>
      </div>
      <Transition name="ev-collapse">
        <div v-if="enginesExpanded" id="ev-section-engines-body" class="ev-section-body">
          <div class="engine-grid">
            <button type="button" class="engine-card" @click="openEngineModal('llama_cpp')">
              <div class="engine-card-head">
                <div class="engine-card-title">
                  <span class="engine-mark engine-mark--llama" aria-hidden="true">L</span>
                  <div>
                    <div class="engine-card-name">llama.cpp</div>
                    <div class="engine-card-meta">{{ enginesStore.llamaVersions.length }} version{{ enginesStore.llamaVersions.length === 1 ? '' : 's' }}</div>
                  </div>
                </div>
              </div>
              <div class="engine-card-body">
                <div
                  class="engine-card-version-line"
                  :title="activeLlamaCpp ? activeLlamaCpp.version : undefined"
                >
                  <Tag
                    v-if="activeLlamaCpp"
                    :value="engineVersionDisplay(activeLlamaCpp.version)"
                    severity="success"
                    class="engine-version-tag"
                  />
                  <Tag v-else value="No Active" severity="warning" class="engine-version-tag" />
                </div>
                <div v-if="llamaCppUpdateInfo?.update_available" class="engine-card-status engine-card-status--warning">
                  Update available: {{ llamaCppUpdateInfo.latest_version }}
                </div>
                <div v-else class="engine-card-status">
                  GGUF inference · CMake builds
                </div>
              </div>
            </button>

            <button type="button" class="engine-card" @click="openEngineModal('ik_llama')">
              <div class="engine-card-head">
                <div class="engine-card-title">
                  <span class="engine-mark engine-mark--ik" aria-hidden="true">IK</span>
                  <div>
                    <div class="engine-card-name">ik_llama.cpp</div>
                    <div class="engine-card-meta">{{ enginesStore.ikLlamaVersions.length }} version{{ enginesStore.ikLlamaVersions.length === 1 ? '' : 's' }}</div>
                  </div>
                </div>
              </div>
              <div class="engine-card-body">
                <div
                  class="engine-card-version-line"
                  :title="activeIkLlama ? activeIkLlama.version : undefined"
                >
                  <Tag
                    v-if="activeIkLlama"
                    :value="engineVersionDisplay(activeIkLlama.version)"
                    severity="success"
                    class="engine-version-tag"
                  />
                  <Tag v-else value="No Active" severity="warning" class="engine-version-tag" />
                </div>
                <div v-if="ikLlamaUpdateInfo?.update_available" class="engine-card-status engine-card-status--warning">
                  Update available: {{ ikLlamaUpdateInfo.latest_version }}
                </div>
                <div v-else class="engine-card-status">
                  GGUF inference · IQK · tracks main
                </div>
              </div>
            </button>

            <button type="button" class="engine-card" @click="openEngineModal('lmdeploy')">
              <div class="engine-card-head">
                <div class="engine-card-title">
                  <i class="pi pi-server engine-card-icon" />
                  <div>
                    <div class="engine-card-name">LMDeploy</div>
                    <div class="engine-card-meta">{{ enginesStore.lmdeployVersions.length }} version{{ enginesStore.lmdeployVersions.length === 1 ? '' : 's' }}</div>
                  </div>
                </div>
              </div>
              <div class="engine-card-body">
                <div
                  class="engine-card-version-line"
                  :title="activeLmdeploy ? activeLmdeploy.version : undefined"
                >
                  <Tag
                    v-if="activeLmdeploy"
                    :value="engineVersionDisplay(activeLmdeploy.version)"
                    severity="success"
                    class="engine-version-tag"
                  />
                  <Tag v-else value="No Active" severity="warning" class="engine-version-tag" />
                </div>
                <div v-if="lmdeployUpdateInfo?.update_available" class="engine-card-status engine-card-status--warning">
                  Update available: v{{ lmdeployUpdateInfo.latest_version }}
                </div>
                <div v-else class="engine-card-status">
                  HF / safetensors · Python env
                </div>
              </div>
            </button>

            <button type="button" class="engine-card" @click="openEngineModal('1cat_vllm')">
              <div class="engine-card-head">
                <div class="engine-card-title">
                  <i class="pi pi-bolt engine-card-icon" />
                  <div>
                    <div class="engine-card-name">1Cat-vLLM</div>
                    <div class="engine-card-meta">{{ enginesStore.onecatVllmVersions.length }} version{{ enginesStore.onecatVllmVersions.length === 1 ? '' : 's' }}</div>
                  </div>
                </div>
              </div>
              <div class="engine-card-body">
                <div
                  class="engine-card-version-line"
                  :title="activeOnecatVllm ? activeOnecatVllm.version : undefined"
                >
                  <Tag
                    v-if="activeOnecatVllm"
                    :value="engineVersionDisplay(activeOnecatVllm.version)"
                    severity="success"
                    class="engine-version-tag"
                  />
                  <Tag v-else value="No Active" severity="warning" class="engine-version-tag" />
                </div>
                <div v-if="onecatVllmUpdateInfo?.update_available" class="engine-card-status engine-card-status--warning">
                  Update available: v{{ onecatVllmUpdateInfo.latest_version }}
                </div>
                <div v-else class="engine-card-status">
                  vLLM SM70 · CUDA 12.8 wheels
                </div>
              </div>
            </button>

            <button
              type="button"
              class="engine-card"
              :disabled="!audioCppFeatureEnabled"
              v-tooltip.top="audioCppFeatureEnabled ? audioCppMaturityTooltip : 'Disabled by AUDIO_CPP_ENABLED'"
              @click="audioCppFeatureEnabled && openEngineModal('audio_cpp')"
            >
              <div class="engine-card-head">
                <div class="engine-card-title">
                  <span class="engine-mark engine-mark--audio" aria-hidden="true">A</span>
                  <div>
                    <div class="engine-card-name">audio.cpp</div>
                    <div class="engine-card-meta">{{ enginesStore.audioCppVersions.length }} version{{ enginesStore.audioCppVersions.length === 1 ? '' : 's' }}</div>
                  </div>
                  <Tag
                    :value="audioCppFeatureEnabled ? audioCppMaturityTag : 'Disabled'"
                    :severity="audioCppFeatureEnabled ? 'success' : 'secondary'"
                  />
                </div>
              </div>
              <div class="engine-card-body">
                <div class="engine-card-version-line" :title="activeAudioCpp ? activeAudioCpp.version : undefined">
                  <Tag
                    v-if="activeAudioCpp"
                    :value="engineVersionDisplay(activeAudioCpp.version)"
                    severity="success"
                    class="engine-version-tag"
                  />
                  <Tag v-else value="No Active" severity="warning" class="engine-version-tag" />
                </div>
                <div v-if="audioCppUpdateInfo?.update_available" class="engine-card-status engine-card-status--warning">
                  Update available: {{ formatEngineUpdateVersion(audioCppUpdateInfo.latest_version) }}
                </div>
                <div v-else class="engine-card-status">
                  Speech/ASR and generic tasks via llama-swap
                </div>
              </div>
            </button>
          </div>
        </div>
      </Transition>
    </section>

    <!-- ── CUDA Install Dialog ────────────────────────────── -->
    <Dialog v-model:visible="cudaInstallDialogVisible" header="Install CUDA Toolkit" modal class="dialog-width-xs">
      <div class="dialog-body">
        <div class="form-field">
          <label>Version</label>
          <Dropdown v-model="cudaInstallVersion" :options="cudaVersionOptions"
            placeholder="Select version…" class="w-full" />
        </div>
      </div>
      <template #footer>
        <Button label="Cancel" severity="secondary" outlined @click="cudaInstallDialogVisible = false" />
        <Button label="Install" icon="pi pi-download" severity="success"
          :disabled="!cudaInstallVersion" :loading="cudaInstalling"
          @click="installCuda" />
      </template>
    </Dialog>

    <Dialog v-model:visible="engineDialogVisible"
      modal maximizable
      class="dialog-width-lg">
      <template #header>
        <EngineDialogHeader v-if="selectedEngine === 'llama_cpp'" title="llama.cpp">
          <template #leading>
            <span class="engine-mark engine-mark--llama" aria-hidden="true">L</span>
          </template>
          <template #tags>
            <span
              class="engine-dialog-tag-clip"
              :title="activeLlamaCpp ? activeLlamaCpp.version : undefined"
            >
              <Tag
                v-if="activeLlamaCpp"
                :value="engineVersionDisplay(activeLlamaCpp.version)"
                severity="success"
                class="engine-version-tag"
              />
              <Tag
                v-else-if="enginesStore.llamaVersions.length"
                value="No Active"
                severity="warning"
                class="engine-version-tag"
              />
            </span>
          </template>
          <template #actions>
            <Button icon="pi pi-sliders-h" text severity="info" size="small"
              v-tooltip.top="'Build settings'"
              @click="openBuildDialog('llama_cpp')" />
            <Button icon="pi pi-refresh" text severity="secondary" size="small"
              v-tooltip.top="'Reload versions'"
              @click="enginesStore.fetchLlamaVersions()" />
            <Button icon="pi pi-list" text severity="secondary" size="small"
              v-tooltip.top="'Rescan CLI parameters (--help)'"
              :loading="paramScanLoading === 'llama_cpp'"
              @click="rescanEngineCliParams('llama_cpp')" />
          </template>
        </EngineDialogHeader>
        <EngineDialogHeader v-else-if="selectedEngine === 'ik_llama'" title="ik_llama.cpp">
          <template #leading>
            <span class="engine-mark engine-mark--ik" aria-hidden="true">IK</span>
          </template>
          <template #tags>
            <span
              class="engine-dialog-tag-clip"
              :title="activeIkLlama ? activeIkLlama.version : undefined"
            >
              <Tag
                v-if="activeIkLlama"
                :value="engineVersionDisplay(activeIkLlama.version)"
                severity="success"
                class="engine-version-tag"
              />
              <Tag
                v-else-if="enginesStore.ikLlamaVersions.length"
                value="No Active"
                severity="warning"
                class="engine-version-tag"
              />
            </span>
          </template>
          <template #actions>
            <Button icon="pi pi-sliders-h" text severity="info" size="small"
              v-tooltip.top="'Build settings'"
              @click="openBuildDialog('ik_llama')" />
            <Button icon="pi pi-refresh" text severity="secondary" size="small"
              v-tooltip.top="'Reload versions'"
              @click="enginesStore.fetchLlamaVersions()" />
            <Button icon="pi pi-list" text severity="secondary" size="small"
              v-tooltip.top="'Rescan CLI parameters (--help)'"
              :loading="paramScanLoading === 'ik_llama'"
              @click="rescanEngineCliParams('ik_llama')" />
          </template>
        </EngineDialogHeader>
        <EngineDialogHeader v-else-if="selectedEngine === 'lmdeploy'" title="LMDeploy">
          <template #leading>
            <i class="pi pi-server" aria-hidden="true" />
          </template>
          <template #tags>
            <span
              class="engine-dialog-tag-clip"
              :title="activeLmdeploy ? activeLmdeploy.version : undefined"
            >
              <Tag
                v-if="activeLmdeploy"
                :value="engineVersionDisplay(activeLmdeploy.version)"
                severity="success"
                class="engine-version-tag"
              />
              <Tag
                v-else-if="enginesStore.lmdeployVersions.length"
                value="No Active"
                severity="warning"
                class="engine-version-tag"
              />
            </span>
          </template>
          <template #actions>
            <Button icon="pi pi-sliders-h" text severity="info" size="small"
              v-tooltip.top="'Build settings'"
              @click="openLmdeployBuildSettings" />
            <Button icon="pi pi-refresh" text severity="secondary" size="small"
              v-tooltip.top="'Reload versions and status'"
              @click="enginesStore.fetchLlamaVersions(); enginesStore.fetchLmdeployStatus()" />
            <Button icon="pi pi-list" text severity="secondary" size="small"
              v-tooltip.top="'Rescan CLI parameters (--help)'"
              :loading="paramScanLoading === 'lmdeploy'"
              @click="rescanEngineCliParams('lmdeploy')" />
          </template>
        </EngineDialogHeader>
        <EngineDialogHeader v-else-if="selectedEngine === '1cat_vllm'" title="1Cat-vLLM">
          <template #leading>
            <i class="pi pi-bolt" aria-hidden="true" />
          </template>
          <template #tags>
            <span
              class="engine-dialog-tag-clip"
              :title="activeOnecatVllm ? activeOnecatVllm.version : undefined"
            >
              <Tag
                v-if="activeOnecatVllm"
                :value="engineVersionDisplay(activeOnecatVllm.version)"
                severity="success"
                class="engine-version-tag"
              />
              <Tag
                v-else-if="enginesStore.onecatVllmVersions.length"
                value="No Active"
                severity="warning"
                class="engine-version-tag"
              />
            </span>
          </template>
          <template #actions>
            <Button icon="pi pi-sliders-h" text severity="info" size="small"
              v-tooltip.top="'Build settings'"
              @click="openOnecatVllmBuildSettings" />
            <Button icon="pi pi-refresh" text severity="secondary" size="small"
              v-tooltip.top="'Reload versions and status'"
              @click="enginesStore.fetchLlamaVersions(); enginesStore.fetchOnecatVllmStatus()" />
            <Button icon="pi pi-list" text severity="secondary" size="small"
              v-tooltip.top="'Rescan CLI parameters (--help)'"
              :loading="paramScanLoading === '1cat_vllm'"
              @click="rescanEngineCliParams('1cat_vllm')" />
          </template>
        </EngineDialogHeader>
        <EngineDialogHeader v-else-if="selectedEngine === 'audio_cpp'" title="audio.cpp">
          <template #leading>
            <span class="engine-mark engine-mark--audio" aria-hidden="true">A</span>
          </template>
          <template #tags>
            <span class="engine-dialog-tag-clip" :title="activeAudioCpp ? activeAudioCpp.version : undefined">
              <Tag
                v-if="activeAudioCpp"
                :value="engineVersionDisplay(activeAudioCpp.version)"
                severity="success"
                class="engine-version-tag"
              />
              <Tag
                v-else-if="enginesStore.audioCppVersions.length"
                value="No Active"
                severity="warning"
                class="engine-version-tag"
              />
            </span>
          </template>
          <template #actions>
            <Button icon="pi pi-sliders-h" text severity="info" size="small"
              v-tooltip.top="'Build settings'"
              @click="openAudioCppBuildSettings" />
            <Button icon="pi pi-refresh" text severity="secondary" size="small"
              v-tooltip.top="'Reload versions and status'"
              @click="enginesStore.fetchLlamaVersions(); enginesStore.fetchAudioCppStatus()" />
            <Button icon="pi pi-list" text severity="secondary" size="small"
              v-tooltip.top="'Rescan audio.cpp capabilities'"
              :loading="paramScanLoading === 'audio_cpp'"
              @click="rescanEngineCliParams('audio_cpp')" />
          </template>
        </EngineDialogHeader>
      </template>
      <section v-if="selectedEngine === 'llama_cpp'" class="ev-section ev-section--modal">
        <div class="ev-section-body engine-modal-body">
          <EngineBuildSettingsHint
            :key="`llama-hint-${hintRevLlama}`"
            engine-key="llama_cpp"
            @open-settings="openBuildDialog('llama_cpp')"
          />
          <EngineCheckUpdatesCta
            :loading="checkingLlamaCpp"
            @check="checkLlamaCppUpdates"
          />
          <EngineUpdateBanner
            :available="!!llamaCppUpdateInfo?.update_available"
            :checked="!!llamaCppUpdateInfo"
            :latest-version="llamaCppUpdateInfo?.latest_version"
            :current-version="llamaCppUpdateInfo?.current_version"
            :link-url="llamaCppUpdateInfo?.release_url"
            :updating="updatingEngine === 'llama_cpp'"
            update-tooltip="Update using saved build settings"
            @update="doUpdateEngine('llama_cpp')"
          />
          <EngineInstallPanel
            subtitle="Add a new build from the latest GitHub release tag or any git repo. Each build is a version you can activate."
          >
            <Button label="From release" icon="pi pi-tag" severity="success" outlined
              :loading="llamaReleaseInstalling"
              :disabled="llamaReleaseInstalling || llamaCppSourceInstalling"
              @click="installLlamaLatestRelease" />
            <Button label="From source" icon="pi pi-code" severity="info" outlined
              :loading="llamaCppSourceInstalling"
              :disabled="llamaReleaseInstalling || llamaCppSourceInstalling"
              @click="openLlamaCppSourceDialog" />
          </EngineInstallPanel>
          <EngineActiveStatus :rows="llamaCppActiveStatusRows" />
          <EngineVersionsBlock>
            <VersionTable
              :versions="enginesStore.llamaVersions"
              :activating="activating"
              :syncing="syncingVersion"
              empty-message="No versions yet. Install one using the options above."
              @activate="activateVersion"
              @sync="syncVersion"
              @delete="confirmDeleteVersion"
            />
          </EngineVersionsBlock>
        </div>
      </section>

      <section v-else-if="selectedEngine === 'ik_llama'" class="ev-section ev-section--modal">
        <div class="ev-section-body engine-modal-body">
          <EngineBuildSettingsHint
            :key="`ik-hint-${hintRevIk}`"
            engine-key="ik_llama"
            @open-settings="openBuildDialog('ik_llama')"
          />
          <EngineCheckUpdatesCta
            :loading="checkingIkLlama"
            hint="Compare the installed tip to the latest commit on main."
            @check="checkIkLlamaUpdates"
          />
          <EngineUpdateBanner
            :available="!!ikLlamaUpdateInfo?.update_available"
            :checked="!!ikLlamaUpdateInfo"
            :latest-version="ikLlamaUpdateInfo?.latest_version"
            :current-version="ikLlamaUpdateInfo?.current_version"
            :link-url="ikLlamaUpdateInfo?.release_url"
            link-label="View commit"
            :updating="updatingEngine === 'ik_llama'"
            update-tooltip="Rebuild tip of main using saved build settings"
            @update="doUpdateEngine('ik_llama')"
          />
          <EngineInstallPanel
            subtitle="ik_llama.cpp has no release tags here. Build the tip of main, or any git repo/ref. Each build is a version you can activate."
          >
            <Button label="From tip" icon="pi pi-bolt" severity="success" outlined
              :loading="ikTipInstalling"
              :disabled="ikTipInstalling || ikLlamaSourceInstalling"
              @click="installIkFromTip" />
            <Button label="From source" icon="pi pi-code" severity="info" outlined
              :loading="ikLlamaSourceInstalling"
              :disabled="ikTipInstalling || ikLlamaSourceInstalling"
              @click="openIkLlamaSourceDialog" />
          </EngineInstallPanel>
          <EngineActiveStatus :rows="ikLlamaActiveStatusRows" />
          <EngineVersionsBlock>
            <VersionTable
              :versions="enginesStore.ikLlamaVersions"
              :activating="activating"
              :syncing="syncingVersion"
              empty-message="No versions yet. Install one using the options above."
              @activate="activateVersion"
              @sync="syncVersion"
              @delete="confirmDeleteVersion"
            />
          </EngineVersionsBlock>
        </div>
      </section>

      <section v-else-if="selectedEngine === 'lmdeploy'" class="ev-section ev-section--modal">
        <div class="ev-section-body engine-modal-body">
          <EngineBuildSettingsHint
            :key="`lm-hint-${hintRevLmdeploy}`"
            engine-key="lmdeploy"
            @open-settings="openLmdeployBuildSettings"
          />
          <EngineCheckUpdatesCta
            :loading="checkingLmdeploy"
            hint="Compare the active install to the latest PyPI release."
            @check="checkLmdeployUpdates"
          />
          <EngineUpdateBanner
            :available="!!lmdeployUpdateInfo?.update_available"
            :checked="!!lmdeployUpdateInfo"
            :latest-version="lmdeployUpdateInfo?.latest_version ? `v${lmdeployUpdateInfo.latest_version}` : ''"
            :current-version="lmdeployUpdateInfo?.current_version ? `v${lmdeployUpdateInfo.current_version}` : 'none'"
            link-url="https://pypi.org/project/lmdeploy/"
            link-label="View on PyPI"
            :updating="updatingLmdeploy"
            update-tooltip="Install the latest PyPI version as a new environment"
            @update="doUpdateLmdeploy"
          />
          <EngineInstallPanel
            subtitle="Add a new Python environment from PyPI or a git source. Each install is a version you can activate."
          >
            <Button label="From PyPI" icon="pi pi-download" severity="success" outlined
              @click="openLmdeployPipDialog" />
            <Button label="From source" icon="pi pi-code" severity="info" outlined
              @click="openLmdeploySourceDialog" />
          </EngineInstallPanel>
          <EngineActiveStatus :rows="lmdeployActiveStatusRows" />
          <EngineVersionsBlock>
            <VersionTable
              :versions="enginesStore.lmdeployVersions"
              :activating="activating"
              :syncing="syncingVersion"
              empty-message="No versions yet. Install one using the options above."
              @activate="activateVersion"
              @sync="syncVersion"
              @delete="confirmDeleteVersion"
            />
          </EngineVersionsBlock>
        </div>
      </section>

      <section v-else-if="selectedEngine === '1cat_vllm'" class="ev-section ev-section--modal">
        <div class="ev-section-body engine-modal-body">
          <EngineBuildSettingsHint
            :key="`ovllm-hint-${hintRevOnecat}`"
            engine-key="1cat_vllm"
            @open-settings="openOnecatVllmBuildSettings"
          />
          <EngineCheckUpdatesCta
            :loading="checkingOnecatVllm"
            hint="Compare the active install to the latest GitHub release."
            @check="checkOnecatVllmUpdates"
          />
          <EngineUpdateBanner
            :available="!!onecatVllmUpdateInfo?.update_available"
            :checked="!!onecatVllmUpdateInfo"
            :latest-version="onecatVllmUpdateInfo?.latest_version ? `v${onecatVllmUpdateInfo.latest_version}` : ''"
            :current-version="onecatVllmUpdateInfo?.current_version ? `v${onecatVllmUpdateInfo.current_version}` : 'none'"
            link-url="https://github.com/1CatAI/1Cat-vLLM/releases/latest"
            :updating="updatingOnecatVllm"
            update-tooltip="Install the latest release wheels as a new environment"
            @update="doUpdateOnecatVllm"
          />
          <EngineNote>
            vLLM fork for Tesla V100 / SM70. Release installs pull prebuilt CUDA 12.8 wheels
            (<code>flash_attn_v100</code> + <code>vllm</code>); source builds require an SM70 GPU and the CUDA 12.8 toolkit.
          </EngineNote>
          <EngineInstallPanel
            subtitle="Add a new environment from prebuilt release wheels (recommended) or build from source. Each install is a version you can activate."
          >
            <Button label="From release" icon="pi pi-download" severity="success" outlined
              @click="openOnecatVllmReleaseDialog" />
            <Button label="From source" icon="pi pi-code" severity="info" outlined
              @click="openOnecatVllmSourceDialog" />
          </EngineInstallPanel>
          <EngineActiveStatus :rows="onecatVllmActiveStatusRows" />
          <EngineVersionsBlock>
            <VersionTable
              :versions="enginesStore.onecatVllmVersions"
              :activating="activating"
              :syncing="syncingVersion"
              empty-message="No versions yet. Install one using the options above."
              @activate="activateVersion"
              @sync="syncVersion"
              @delete="confirmDeleteVersion"
            />
          </EngineVersionsBlock>
        </div>
      </section>

      <section v-else-if="selectedEngine === 'audio_cpp'" class="ev-section ev-section--modal">
        <div class="ev-section-body engine-modal-body">
          <EngineBuildSettingsHint
            :key="`audio-hint-${hintRevAudio}`"
            engine-key="audio_cpp"
            @open-settings="openAudioCppBuildSettings"
          />
          <EngineCheckUpdatesCta
            :loading="checkingAudioCpp"
            @check="checkAudioCppUpdates"
          />
          <EngineUpdateBanner
            :available="!!audioCppUpdateInfo?.update_available"
            :checked="!!audioCppUpdateInfo"
            :latest-version="formatEngineUpdateVersion(audioCppUpdateInfo?.latest_version)"
            :current-version="audioCppUpdateInfo?.tracking_ref || enginesStore.audioCppStatus?.tracking_ref || ''"
            :link-url="audioCppUpdateInfo?.latest_release?.html_url || audioCppUpdateInfo?.latest_commit?.html_url || ''"
            :link-label="audioCppUpdateInfo?.latest_release?.html_url ? 'View release' : 'View commit'"
            :updating="audioCppUpdating"
            :update-tooltip="audioCppUpdateTooltip"
            @update="updateAudioCpp"
          >
            <template #message>
              <template v-if="audioCppUpdateInfo?.latest_release?.tag_name">
                Update available (release
                <strong>{{ formatEngineUpdateVersion(audioCppUpdateInfo.latest_version) }}</strong>)
              </template>
              <template v-else>
                Update available on
                <strong>{{ audioCppUpdateInfo?.tracking_ref || enginesStore.audioCppStatus?.tracking_ref || 'tracked ref' }}</strong>:
                <strong>{{ formatEngineUpdateVersion(audioCppUpdateInfo?.latest_version) }}</strong>
              </template>
            </template>
          </EngineUpdateBanner>

          <EngineNote
            v-if="enginesStore.audioCppStatus?.active && (enginesStore.audioCppStatus?.contract_changed || audioCppDeltaHasChanges)"
            severity="warning"
          >
            <span>
              The active audio.cpp CLI/help contract fingerprint changed since the previous scan.
              New loaders or options may be available
              <template v-if="(enginesStore.audioCppStatus?.families || []).length">
                ({{ enginesStore.audioCppStatus.families.length }} families
                <template v-if="(enginesStore.audioCppStatus?.tasks || []).length">
                  · {{ enginesStore.audioCppStatus.tasks.join(', ') }}
                </template>)
              </template>.
              Rescan capabilities, then review affected model configs.
            </span>
            <ul v-if="audioCppDeltaSummary.length" class="audio-cpp-delta-list">
              <li v-for="line in audioCppDeltaSummary" :key="line">{{ line }}</li>
            </ul>
            <div v-if="showAffectedAudioModels && affectedAudioModels.length" class="audio-cpp-affected">
              <div class="audio-cpp-affected__title">Affected models</div>
              <ul>
                <li v-for="model in affectedAudioModels" :key="model.id">
                  <button
                    type="button"
                    class="audio-cpp-affected__link"
                    @click="$router.push(`/models/${encodeURIComponent(model.id)}/config`)"
                  >
                    {{ model.name || model.id }}
                  </button>
                  <span v-if="model.family || model.task" class="audio-cpp-affected__meta">
                    {{ [model.family, model.task].filter(Boolean).join(' · ') }}
                  </span>
                </li>
              </ul>
            </div>
            <template #actions>
              <Button
                label="Rescan CLI"
                icon="pi pi-list"
                size="small"
                severity="warning"
                outlined
                :loading="paramScanLoading === 'audio_cpp'"
                @click="rescanEngineCliParams('audio_cpp')"
              />
              <Button
                label="Review affected models"
                icon="pi pi-list-check"
                size="small"
                severity="warning"
                :disabled="!affectedAudioModels.length"
                @click="showAffectedAudioModels = !showAffectedAudioModels"
              />
              <Button
                label="Open Models"
                icon="pi pi-box"
                size="small"
                severity="secondary"
                text
                @click="$router.push('/models')"
              />
            </template>
          </EngineNote>

          <EngineInstallPanel
            subtitle="Add a new build from the latest release or any git repo. Each build is a version you can activate."
          >
            <Button label="From release" icon="pi pi-tag" severity="success" outlined
              :loading="audioCppReleaseInstalling"
              :disabled="audioCppReleaseInstalling || audioCppSourceInstalling"
              @click="installAudioLatestRelease" />
            <Button label="From source" icon="pi pi-code" severity="info" outlined
              :loading="audioCppSourceInstalling"
              :disabled="audioCppReleaseInstalling || audioCppSourceInstalling"
              @click="openAudioCppSourceDialog" />
          </EngineInstallPanel>
          <EngineActiveStatus :rows="audioCppActiveStatusRows" />
          <EngineVersionsBlock>
            <VersionTable
              :versions="enginesStore.audioCppVersions"
              :activating="activating"
              :syncing="syncingVersion"
              empty-message="No versions yet. Install one using the options above."
              @activate="activateVersion"
              @sync="syncVersion"
              @delete="confirmDeleteVersion"
            />
          </EngineVersionsBlock>
        </div>
      </section>
    </Dialog>

    <!-- ── LMDeploy Build Settings Dialog ─────────────────── -->
    <Dialog v-model:visible="lmdeployBuildDialogVisible"
      header="Build settings — LMDeploy"
      modal class="build-settings-dialog dialog-width-md">
      <div class="dialog-body build-settings-body">
        <p class="build-note build-note--info">
          Saved defaults for PyPI and source installs. Use <strong>Save settings</strong> to store
          without installing, or <strong>Install from source</strong> to build now.
        </p>
        <div class="form-field">
          <label>Default PyPI version <span class="optional">(optional)</span></label>
          <InputText v-model="lmdeployBuildForm.pip_version" placeholder="Blank = latest" class="w-full" />
          <small>Used by From PyPI when the version field is left blank.</small>
        </div>
        <div class="form-field">
          <label>Source repo URL</label>
          <InputText v-model="lmdeployBuildForm.source_repo" placeholder="https://github.com/InternLM/lmdeploy.git" class="w-full" />
        </div>
        <div class="form-field">
          <label>Source branch</label>
          <InputText v-model="lmdeployBuildForm.source_branch" placeholder="main" class="w-full" />
        </div>
      </div>
      <template #footer>
        <Button label="Cancel" severity="secondary" outlined @click="lmdeployBuildDialogVisible = false" />
        <Button label="Save settings" icon="pi pi-save" severity="secondary"
          :loading="savingLmdeployBuildSettings"
          @click="saveLmdeployBuildSettingsOnly" />
        <Button label="Install from source" icon="pi pi-code" severity="info"
          :loading="lmdeployInstalling" @click="installLmdeployFromBuildSettings" />
      </template>
    </Dialog>

    <!-- ── 1Cat-vLLM Build Settings Dialog ───────────────── -->
    <Dialog v-model:visible="onecatVllmBuildDialogVisible"
      header="Build settings — 1Cat-vLLM"
      modal class="build-settings-dialog dialog-width-md">
      <div class="dialog-body build-settings-body">
        <p class="build-note build-note--info">
          Saved defaults for release wheels and source builds. Use <strong>Save settings</strong> to store
          without installing, or <strong>Build from source</strong> to compile now.
        </p>
        <div class="form-field">
          <label>Default release version <span class="optional">(optional)</span></label>
          <InputText v-model="onecatVllmBuildForm.release_version" placeholder="Blank = latest" class="w-full" />
          <small>Used by From release when the version field is left blank.</small>
        </div>
        <div class="form-field">
          <label>Source repo URL</label>
          <InputText v-model="onecatVllmBuildForm.source_repo" placeholder="https://github.com/1CatAI/1Cat-vLLM.git" class="w-full" />
        </div>
        <div class="form-field">
          <label>Source branch</label>
          <InputText v-model="onecatVllmBuildForm.source_branch" placeholder="main" class="w-full" />
        </div>
        <div class="form-field">
          <small>Source builds compile SM70 CUDA kernels and require an NVIDIA GPU plus the CUDA 12.8 toolkit.</small>
        </div>
      </div>
      <template #footer>
        <Button label="Cancel" severity="secondary" outlined @click="onecatVllmBuildDialogVisible = false" />
        <Button label="Save settings" icon="pi pi-save" severity="secondary"
          :loading="savingOnecatVllmBuildSettings"
          @click="saveOnecatVllmBuildSettingsOnly" />
        <Button label="Build from source" icon="pi pi-code" severity="info"
          :loading="onecatVllmInstalling" @click="installOnecatVllmFromBuildSettings" />
      </template>
    </Dialog>

    <!-- ── audio.cpp Build Settings Dialog ───────────────── -->
    <Dialog v-model:visible="audioCppBuildDialogVisible"
      header="Build settings — audio.cpp"
      modal class="build-settings-dialog dialog-width-md">
      <div class="dialog-body build-settings-body">
        <div class="form-field">
          <label>Repo URL</label>
          <InputText v-model="audioCppBuildForm.repository_url"
            placeholder="https://github.com/0xShug0/audio.cpp.git"
            class="w-full" />
          <small>Official repo or any fork with the same layout. Saved with Build settings.</small>
        </div>
        <div class="form-field">
          <label>Ref (tag / branch / commit)</label>
          <InputText v-model="audioCppBuildForm.source_ref"
            :placeholder="enginesStore.audioCppStatus?.tracking_ref || 'main'"
            class="w-full" />
          <small>
            Used when you Build now. Branch and tag refs become the Update tracking ref.
            Building a commit installs that tip but leaves tracking on the previous branch/tag,
            so Update will not follow the detached commit.
          </small>
        </div>
        <div class="form-field">
          <label>Build Name Suffix <span class="optional">(optional)</span></label>
          <InputText v-model="audioCppBuildForm.versionSuffix" placeholder="e.g. my-build" class="w-full" />
          <small>Appended to version name. Defaults to timestamp if empty.</small>
        </div>
        <div class="form-field">
          <label>Build type</label>
          <Dropdown v-model="audioCppBuildForm.build_config.build_type"
            :options="audioCppBuildTypeOptions"
            optionLabel="label"
            optionValue="value"
            class="w-full" />
        </div>

        <div v-if="audioCppOptionsLoading" class="build-note build-note--info">Loading build options…</div>

        <template v-for="cat in visibleAudioBuildCategories" :key="cat.id">
          <div v-if="cat.id === 'backends'" class="form-field">
            <label class="build-options-section">{{ cat.label }}</label>
            <div class="toggle-grid">
              <div
                v-for="opt in (cat.options || [])"
                :key="opt.key"
                class="toggle-row"
              >
                <InputSwitch
                  v-model="audioCppBuildForm.build_config[opt.key]"
                  :disabled="audioBackendDisabled(opt.key)"
                  @update:modelValue="onAudioBackendToggle(opt.key)"
                />
                <div>
                  <span class="opt-label">{{ opt.label }}</span>
                  <small class="opt-desc">{{ opt.desc }}</small>
                </div>
              </div>
            </div>
          </div>

          <details v-else class="build-options-details">
            <summary class="build-options-section">
              {{ cat.label }}
              <span class="build-advanced-hint">advanced</span>
            </summary>
            <div class="toggle-grid">
              <template v-for="opt in visibleAudioOptions(cat)" :key="opt.key">
                <div v-if="opt.type === 'bool'" class="toggle-row">
                  <InputSwitch v-model="audioCppBuildForm.build_config[opt.key]" />
                  <div>
                    <span class="opt-label">{{ opt.label }}</span>
                    <small class="opt-desc">{{ opt.desc }}</small>
                  </div>
                </div>
                <div v-else-if="opt.type === 'int'" class="opt-string-field">
                  <span class="opt-label">{{ opt.label }}</span>
                  <small class="opt-desc">{{ opt.desc }}</small>
                  <InputNumber v-model="audioCppBuildForm.build_config[opt.key]" :min="0" :max="256" class="w-full mt-1" />
                </div>
                <div v-else class="opt-string-field">
                  <span class="opt-label">{{ opt.label }}</span>
                  <small class="opt-desc">{{ opt.desc }}</small>
                  <Dropdown
                    v-if="opt.type === 'enum'"
                    v-model="audioCppBuildForm.build_config[opt.key]"
                    :options="opt.enum_values || []"
                    class="w-full mt-1"
                  />
                  <InputText v-else v-model="audioCppBuildForm.build_config[opt.key]" class="w-full mt-1" />
                </div>
              </template>
            </div>
          </details>
        </template>

        <details class="build-options-details">
          <summary class="build-options-section">
            Custom flags
            <span class="build-advanced-hint">advanced</span>
          </summary>
          <div class="form-field">
            <label>Custom CMake args <span class="optional">(optional)</span></label>
            <InputText v-model="audioCppBuildForm.build_config.custom_cmake_args"
              placeholder="e.g. -DFOO=ON -DBAR=OFF" class="w-full" />
          </div>
          <div class="form-field">
            <label>CFLAGS / CXXFLAGS <span class="optional">(optional)</span></label>
            <div class="flags-row">
              <InputText v-model="audioCppBuildForm.build_config.cflags" placeholder="CFLAGS" class="flex-1" />
              <InputText v-model="audioCppBuildForm.build_config.cxxflags" placeholder="CXXFLAGS" class="flex-1" />
            </div>
          </div>
        </details>
      </div>
      <template #footer>
        <Button label="Cancel" severity="secondary" outlined @click="audioCppBuildDialogVisible = false" />
        <Button label="Save settings" icon="pi pi-save" severity="secondary"
          :loading="savingAudioCppBuildSettings"
          @click="saveAudioCppBuildSettingsOnly" />
        <Button label="Build now" icon="pi pi-cog" severity="info"
          :loading="audioCppBuilding" @click="buildAudioCpp" />
      </template>
    </Dialog>

    <!-- ── audio.cpp Install from Source Dialog ───────────── -->
    <Dialog v-model:visible="audioCppSourceDialogVisible" header="Build audio.cpp from source" modal class="dialog-width-md">
      <div class="dialog-body">
        <div class="form-field">
          <label>Repo URL</label>
          <InputText v-model="audioCppSourceRepo" placeholder="https://github.com/0xShug0/audio.cpp.git" class="w-full" />
          <small>Official repo or any fork with the same layout.</small>
        </div>
        <div class="form-field">
          <label>Tag / branch / commit</label>
          <InputText v-model="audioCppSourceRef" placeholder="main" class="w-full" />
          <small>Checked out before CMake build. Uses your saved build settings (gear in the header).</small>
        </div>
      </div>
      <template #footer>
        <Button label="Cancel" severity="secondary" outlined @click="audioCppSourceDialogVisible = false" />
        <Button label="Build from source" icon="pi pi-code" severity="info"
          :loading="audioCppSourceInstalling" :disabled="audioCppSourceInstalling"
          @click="installAudioCppFromSource" />
      </template>
    </Dialog>

    <!-- ── Build Settings Dialog ─────────────────────────── -->
    <Dialog v-model:visible="buildDialogVisible"
      :header="`Build settings — ${buildTarget === 'ik_llama' ? 'ik_llama.cpp' : 'llama.cpp'}`"
      modal class="build-settings-dialog dialog-width-md">
      <div class="dialog-body build-settings-body">
        <div class="form-field">
          <label>Ref (tag / branch / commit)</label>
          <InputText v-model="buildForm.commitSha"
            :placeholder="buildTarget === 'ik_llama' ? 'main or commit SHA' : 'master'"
            class="w-full" />
          <small v-if="buildTarget === 'ik_llama'">
            Use a branch or commit. ik_llama.cpp does not ship releases or tags here; check for updates and “build latest” track the tip of <code>main</code>.
          </small>
          <small v-else>
            Use a release tag, branch, or commit. Latest detected release is used by default when available.
          </small>
        </div>
        <div class="form-field">
          <label>Build Name Suffix <span class="optional">(optional)</span></label>
          <InputText v-model="buildForm.versionSuffix" placeholder="e.g. my-build" class="w-full" />
          <small>Appended to version name. Defaults to timestamp if empty.</small>
        </div>
        <div class="form-field">
          <label>Build type</label>
          <Dropdown v-model="buildForm.buildConfig.build_type"
            :options="buildTypeOptions"
            optionLabel="label"
            optionValue="value"
            placeholder="Release"
            class="w-full" />
        </div>

        <div v-if="buildOptionsLoading" class="build-note build-note--info">Loading build options…</div>

        <template v-for="cat in visibleBuildCategories" :key="cat.id">
          <!-- Primary backends stay open; niche backends nested under More -->
          <div v-if="cat.id === 'backends'" class="form-field">
            <div v-if="buildTarget === 'ik_llama'" class="build-note build-note--info">
              ik_llama.cpp uses IQK kernels and <code>GGML_HIPBLAS</code> / <code>GGML_CUDA_USE_GRAPHS</code> naming. Examples must stay on (server lives there).
            </div>
            <label class="build-options-section">{{ cat.label }}</label>
            <div class="toggle-grid">
              <div
                v-for="opt in primaryBackendOptions(cat)"
                :key="opt.key"
                class="toggle-row"
              >
                <InputSwitch v-model="buildForm.buildConfig[opt.key]" />
                <div>
                  <span class="opt-label">{{ opt.label }}</span>
                  <small class="opt-desc">{{ opt.desc }}</small>
                </div>
              </div>
            </div>
            <details v-if="extraBackendOptions(cat).length" class="build-options-details">
              <summary class="build-options-section">More backends</summary>
              <div class="toggle-grid">
                <div
                  v-for="opt in extraBackendOptions(cat)"
                  :key="opt.key"
                  class="toggle-row"
                >
                  <InputSwitch v-model="buildForm.buildConfig[opt.key]" />
                  <div>
                    <span class="opt-label">{{ opt.label }}</span>
                    <small class="opt-desc">{{ opt.desc }}</small>
                  </div>
                </div>
              </div>
            </details>
          </div>

          <!-- Everything else: collapsed by default when marked advanced -->
          <details
            v-else
            class="build-options-details"
          >
            <summary class="build-options-section">
              {{ cat.label }}
              <span class="build-advanced-hint">advanced</span>
            </summary>
            <div v-if="cat.id === 'artifacts' && buildTarget === 'ik_llama'" class="build-note build-note--info">
              For ik_llama.cpp, <strong>Examples</strong> is required (server binary lives in examples).
            </div>
            <div class="toggle-grid">
              <template v-for="opt in visibleOptions(cat)" :key="opt.key">
                <div v-if="opt.type === 'bool'" class="toggle-row">
                  <InputSwitch
                    v-model="buildForm.buildConfig[opt.key]"
                    :disabled="buildTarget === 'ik_llama' && opt.key === 'build_examples'"
                  />
                  <div>
                    <span class="opt-label">{{ opt.label }}</span>
                    <small class="opt-desc">{{ opt.desc }}</small>
                  </div>
                </div>
                <div v-else class="opt-string-field">
                  <span class="opt-label">{{ opt.label }}</span>
                  <small class="opt-desc">{{ opt.desc }}</small>
                  <Dropdown
                    v-if="opt.type === 'enum'"
                    v-model="buildForm.buildConfig[opt.key]"
                    :options="opt.enum_values || []"
                    class="w-full mt-1"
                  />
                  <InputText
                    v-else
                    v-model="buildForm.buildConfig[opt.key]"
                    class="w-full mt-1"
                    :placeholder="opt.key === 'cuda_architectures' ? 'e.g. 86;89 (blank = auto)' : ''"
                  />
                </div>
              </template>
            </div>
          </details>
        </template>

        <details class="build-options-details">
          <summary class="build-options-section">
            Custom flags
            <span class="build-advanced-hint">advanced</span>
          </summary>
          <div class="form-field">
            <label>Custom CMake args <span class="optional">(optional)</span></label>
            <InputText v-model="buildForm.buildConfig.custom_cmake_args"
              placeholder="e.g. -DFOO=ON -DBAR=OFF" class="w-full" />
          </div>
          <div class="form-field">
            <label>CFLAGS / CXXFLAGS <span class="optional">(optional)</span></label>
            <div class="flags-row">
              <InputText v-model="buildForm.buildConfig.cflags" placeholder="CFLAGS" class="flex-1" />
              <InputText v-model="buildForm.buildConfig.cxxflags" placeholder="CXXFLAGS" class="flex-1" />
            </div>
          </div>
        </details>
      </div>
      <template #footer>
        <Button label="Cancel" severity="secondary" outlined @click="buildDialogVisible = false" />
        <Button label="Save settings" icon="pi pi-save" severity="secondary"
          :loading="savingBuildSettings"
          @click="saveBuildSettingsOnly" />
        <Button label="Build now" icon="pi pi-cog" severity="info"
          :loading="building" @click="doStartBuild" />
      </template>
    </Dialog>

    <!-- ── llama.cpp Install from Source Dialog ────────────── -->
    <Dialog v-model:visible="llamaCppSourceDialogVisible" header="Build llama.cpp from source" modal class="dialog-width-md">
      <div class="dialog-body">
        <div class="form-field">
          <label>Repo URL</label>
          <InputText v-model="llamaCppSourceRepo" placeholder="https://github.com/ggerganov/llama.cpp.git" class="w-full" />
          <small>Official repo or any fork with the same layout.</small>
        </div>
        <div class="form-field">
          <label>Tag / branch / commit</label>
          <InputText v-model="llamaCppSourceRef" placeholder="master" class="w-full" />
          <small>Checked out before CMake build. Uses your saved build settings (gear in the header).</small>
        </div>
      </div>
      <template #footer>
        <Button label="Cancel" severity="secondary" outlined @click="llamaCppSourceDialogVisible = false" />
        <Button label="Build from source" icon="pi pi-code" severity="info"
          :loading="llamaCppSourceInstalling" :disabled="llamaCppSourceInstalling"
          @click="installLlamaCppFromSource" />
      </template>
    </Dialog>

    <!-- ── ik_llama.cpp Install from Source Dialog ─────────── -->
    <Dialog v-model:visible="ikLlamaSourceDialogVisible" header="Build ik_llama.cpp from source" modal class="dialog-width-md">
      <div class="dialog-body">
        <div class="form-field">
          <label>Repo URL</label>
          <InputText v-model="ikLlamaSourceRepo" placeholder="https://github.com/ikawrakow/ik_llama.cpp.git" class="w-full" />
          <small>Official repo or any fork with the same layout.</small>
        </div>
        <div class="form-field">
          <label>Branch / commit</label>
          <InputText v-model="ikLlamaSourceRef" placeholder="main" class="w-full" />
          <small>Checked out before CMake build. Uses your saved build settings (gear in the header). Release tags are not used for ik_llama.cpp.</small>
        </div>
      </div>
      <template #footer>
        <Button label="Cancel" severity="secondary" outlined @click="ikLlamaSourceDialogVisible = false" />
        <Button label="Build from source" icon="pi pi-code" severity="info"
          :loading="ikLlamaSourceInstalling" :disabled="ikLlamaSourceInstalling"
          @click="installIkLlamaFromSource" />
      </template>
    </Dialog>

    <!-- ── LMDeploy Install from pip Dialog ───────────────── -->
    <Dialog v-model:visible="lmPipDialogVisible" header="Install LMDeploy from pip" modal class="dialog-width-sm">
      <div class="dialog-body">
        <div class="form-field">
          <label>Version</label>
          <InputText v-model="lmdeployPipVersion" placeholder="Blank = latest" class="w-full" />
          <small>Leave blank to install the latest from PyPI.</small>
        </div>
      </div>
      <template #footer>
        <Button label="Cancel" severity="secondary" outlined @click="lmPipDialogVisible = false" />
        <Button label="Install" icon="pi pi-download" severity="success"
          :loading="lmdeployInstalling" :disabled="lmdeployInstalling"
          @click="installLmdeployPip" />
      </template>
    </Dialog>

    <!-- ── LMDeploy Install from Source Dialog ─────────────── -->
    <Dialog v-model:visible="lmSourceDialogVisible" header="Install LMDeploy from Source" modal class="dialog-width-md">
      <div class="dialog-body">
        <div class="form-field">
          <label>Repo URL</label>
          <InputText v-model="lmSourceRepo" placeholder="https://github.com/InternLM/lmdeploy.git" class="w-full" />
        </div>
        <div class="form-field">
          <label>Branch</label>
          <InputText v-model="lmSourceBranch" placeholder="main" class="w-full" />
        </div>
      </div>
      <template #footer>
        <Button label="Cancel" severity="secondary" outlined @click="lmSourceDialogVisible = false" />
        <Button label="Install from Source" icon="pi pi-code" severity="info"
          :loading="lmdeployInstalling" :disabled="lmdeployInstalling"
          @click="installLmdeploySource" />
      </template>
    </Dialog>

    <!-- ── 1Cat-vLLM Install from Release Dialog ───────────── -->
    <Dialog v-model:visible="ovllmReleaseDialogVisible" header="Install 1Cat-vLLM from Release" modal class="dialog-width-sm">
      <div class="dialog-body">
        <div class="form-field">
          <label>Release version</label>
          <InputText v-model="ovllmReleaseVersion" placeholder="Blank = latest" class="w-full" />
          <small>Leave blank to install the latest GitHub release. Downloads prebuilt CUDA 12.8 wheels.</small>
        </div>
      </div>
      <template #footer>
        <Button label="Cancel" severity="secondary" outlined @click="ovllmReleaseDialogVisible = false" />
        <Button label="Install" icon="pi pi-download" severity="success"
          :loading="onecatVllmInstalling" :disabled="onecatVllmInstalling"
          @click="installOnecatVllmRelease" />
      </template>
    </Dialog>

    <!-- ── 1Cat-vLLM Install from Source Dialog ────────────── -->
    <Dialog v-model:visible="ovllmSourceDialogVisible" header="Build 1Cat-vLLM from Source" modal class="dialog-width-md">
      <div class="dialog-body">
        <div class="form-field">
          <label>Repo URL</label>
          <InputText v-model="ovllmSourceRepo" placeholder="https://github.com/1CatAI/1Cat-vLLM.git" class="w-full" />
        </div>
        <div class="form-field">
          <label>Branch</label>
          <InputText v-model="ovllmSourceBranch" placeholder="main" class="w-full" />
        </div>
        <div class="form-field">
          <small>Source builds compile SM70 CUDA kernels and require an NVIDIA GPU plus the CUDA 12.8 toolkit. This can take a long time.</small>
        </div>
      </div>
      <template #footer>
        <Button label="Cancel" severity="secondary" outlined @click="ovllmSourceDialogVisible = false" />
        <Button label="Build from Source" icon="pi pi-code" severity="info"
          :loading="onecatVllmInstalling" :disabled="onecatVllmInstalling"
          @click="installOnecatVllmSource" />
      </template>
    </Dialog>

  </div>
</template>

<script setup>
import { ref, computed, nextTick, onMounted, onUnmounted } from 'vue'
import { useRoute } from 'vue-router'
import { useConfirm } from 'primevue/useconfirm'
import { useToast } from 'primevue/usetoast'
import Button from 'primevue/button'
import Tag from 'primevue/tag'
import ProgressBar from 'primevue/progressbar'
import Dialog from 'primevue/dialog'
import Dropdown from 'primevue/dropdown'
import InputText from 'primevue/inputtext'
import InputNumber from 'primevue/inputnumber'
import InputSwitch from 'primevue/inputswitch'
import Checkbox from 'primevue/checkbox'
import EngineDialogHeader from '@/components/system/EngineDialogHeader.vue'
import EngineCheckUpdatesCta from '@/components/system/EngineCheckUpdatesCta.vue'
import EngineBuildSettingsHint from '@/components/system/EngineBuildSettingsHint.vue'
import EngineInstallPanel from '@/components/system/EngineInstallPanel.vue'
import EngineUpdateBanner from '@/components/system/EngineUpdateBanner.vue'
import EngineActiveStatus from '@/components/system/EngineActiveStatus.vue'
import EngineVersionsBlock from '@/components/system/EngineVersionsBlock.vue'
import EngineNote from '@/components/system/EngineNote.vue'
import VersionTable from '@/components/system/VersionTable.vue'
import SwapRoutingPanel from '@/components/system/SwapRoutingPanel.vue'
import { useEnginesStore } from '@/stores/engines'
import { useProgressStore } from '@/stores/progress'
import { formatBytesIEC } from '@/utils/formatting'

const enginesStore = useEnginesStore()
const progressStore = useProgressStore()
const route = useRoute()
const confirm = useConfirm()
const toast = useToast()

// ── System metrics ─────────────────────────────────────────
const systemExpanded = ref(true)
const enginesExpanded = ref(true)
const routingExpanded = ref(true)
const routingPanel = ref(null)

function focusRoutingSection() {
  const hash = String(route?.hash || '').toLowerCase()
  if (
    hash !== '#ev-section-routing' &&
    hash !== '#ev-section-routing-body' &&
    hash !== '#routing'
  ) {
    return
  }
  routingExpanded.value = true
  void nextTick(() => {
    document.getElementById('ev-section-routing')?.scrollIntoView({
      behavior: 'smooth',
      block: 'start',
    })
  })
}
const engineDialogVisible = ref(false)
const selectedEngine = ref('llama_cpp')
const paramScanLoading = ref(null)

async function rescanEngineCliParams(engine) {
  paramScanLoading.value = engine
  try {
    const data = await enginesStore.scanEngineParams(engine)
    if (data?.ok) {
      toast.add({
        severity: 'success',
        summary: 'CLI parameters scanned',
        detail: `Indexed ${data.param_count ?? 0} options for ${engine}.`,
        life: 3500,
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

function openEngineModal(engineKey) {
  selectedEngine.value = engineKey
  engineDialogVisible.value = true
  if (engineKey === 'llama_cpp') {
    checkLlamaCppUpdates()
  } else if (engineKey === 'ik_llama') {
    checkIkLlamaUpdates()
  } else if (engineKey === 'lmdeploy') {
    enginesStore.fetchLlamaVersions()
    checkLmdeployUpdates()
  } else if (engineKey === '1cat_vllm') {
    enginesStore.fetchLlamaVersions()
    checkOnecatVllmUpdates()
  } else if (engineKey === 'audio_cpp') {
    enginesStore.fetchLlamaVersions()
    enginesStore.fetchAudioCppStatus()
    checkAudioCppUpdates()
  }
}

async function refreshEnginesOverview() {
  await Promise.allSettled([
    enginesStore.fetchLlamaVersions(),
    enginesStore.fetchLmdeployStatus(),
    enginesStore.fetchOnecatVllmStatus(),
    enginesStore.fetchAudioCppStatus(),
    checkLlamaCppUpdates(),
    checkIkLlamaUpdates(),
    checkLmdeployUpdates(),
    checkOnecatVllmUpdates(),
    checkAudioCppUpdates(),
  ])
}

const sys = computed(() => {
  const s = enginesStore.systemStatus
  return s?.system || s || {}
})

const gpus = computed(() => enginesStore.gpuInfo?.gpus ?? [])

const memUsedBytes = computed(() => {
  const m = sys.value.memory
  const total = m?.total ?? 0
  if (m?.used != null) return m.used
  if (total > 0 && m?.available != null) return total - m.available
  return 0
})

const memPercent = computed(() => {
  const m = sys.value.memory
  const total = m?.total ?? 0
  const used = memUsedBytes.value
  return total > 0 ? Math.round((used / total) * 100) : 0
})

const diskPercent = computed(() => {
  const d = sys.value.disk
  const used = d?.used ?? 0
  const total = d?.total ?? 0
  return total > 0 ? Math.round((used / total) * 100) : 0
})

/** VRAM from /api/gpu-info: gpus[].memory.{used,total} in bytes */
function gpuVramUsedBytes(g) {
  if (g?.memory?.used != null) return Number(g.memory.used)
  return 0
}

function gpuVramTotalBytes(g) {
  if (g?.memory?.total != null) return Number(g.memory.total)
  return 0
}

function gpuPercent(g) {
  const used = gpuVramUsedBytes(g)
  const total = gpuVramTotalBytes(g)
  return total > 0 ? Math.round((used / total) * 100) : 0
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
const activeLlamaCpp = computed(() => enginesStore.llamaVersions.find(v => v.is_active) ?? null)
const activeIkLlama = computed(() => enginesStore.ikLlamaVersions.find(v => v.is_active) ?? null)
const activeLmdeploy = computed(() => enginesStore.lmdeployVersions.find(v => v.is_active) ?? null)
const activeOnecatVllm = computed(() => enginesStore.onecatVllmVersions.find(v => v.is_active) ?? null)
const activeAudioCpp = computed(() => enginesStore.audioCppVersions.find(v => v.is_active) ?? null)

function cmakeBackendBadge(version) {
  const cfg = version?.build_config || {}
  if (cfg.cuda || cfg.backend === 'cuda' || cfg.enable_cuda) return 'CUDA'
  if (cfg.hip || cfg.backend === 'hip') return 'HIP'
  if (cfg.vulkan || cfg.backend === 'vulkan') return 'Vulkan'
  if (cfg.metal || cfg.backend === 'metal') return 'Metal'
  if (cfg.backend) return String(cfg.backend).toUpperCase()
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

const lmdeployActiveStatusRows = computed(() => {
  const status = enginesStore.lmdeployStatus || {}
  const rows = []
  if (activeLmdeploy.value || status.venv_path) {
    const displayType =
      activeLmdeploy.value?.type
      || activeLmdeploy.value?.install_type
      || status.install_type
      || 'pip'
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
  const status = enginesStore.onecatVllmStatus || {}
  const rows = []
  if (activeOnecatVllm.value || status.venv_path) {
    const displayType =
      activeOnecatVllm.value?.type
      || activeOnecatVllm.value?.install_type
      || status.install_type
      || 'release'
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
  const active = enginesStore.audioCppStatus?.active
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
      code: active.source_ref
        ? `${active.source_repo} @ ${active.source_ref}`
        : active.source_repo,
    })
  }
  const backend = cmakeBackendBadge(active)
  if (backend) {
    rows.push({ label: 'Backend:', tag: backend, tagSeverity: 'secondary' })
  }
  if (enginesStore.audioCppStatus?.models_root) {
    rows.push({
      label: 'Models:',
      code: enginesStore.audioCppStatus.models_root,
      tag: enginesStore.audioCppStatus.model_manager_ready
        ? 'Model manager ready'
        : 'Model manager unavailable',
      tagSeverity: enginesStore.audioCppStatus.model_manager_ready ? 'success' : 'warning',
      tagTooltip: enginesStore.audioCppStatus.model_manager_ready
        ? 'Prefers model_manager_v2.py (model_specs). Legacy manager remains for assemble/convert packages.'
        : 'Activate an audio.cpp build that includes model_manager_v2.py or a legacy model_manager*.py',
    })
  }
  return rows
})
const audioCppFeatureEnabled = computed(() => {
  const descriptor = (enginesStore.engineDescriptors || []).find(engine => engine.id === 'audio_cpp')
  return descriptor?.enabled !== false
})

const audioCppMaturitySurfaces = computed(() => {
  const descriptor = (enginesStore.engineDescriptors || []).find(engine => engine.id === 'audio_cpp')
  return descriptor?.maturity_surfaces || {}
})

const audioCppMaturityTag = computed(() => {
  const surfaces = audioCppMaturitySurfaces.value
  if (surfaces.speech_asr === 'stable') return 'Speech/ASR ready'
  if (surfaces.generic_tasks === 'limited') return 'Limited'
  return 'Audio'
})

const audioCppMaturityTooltip = computed(() => {
  const surfaces = audioCppMaturitySurfaces.value
  const parts = []
  if (surfaces.speech_asr) parts.push(`Speech/ASR: ${surfaces.speech_asr}`)
  if (surfaces.generic_tasks) parts.push(`Generic tasks: ${surfaces.generic_tasks}`)
  if (surfaces.heuristic_discovery) {
    parts.push(`Heuristic discovery: ${surfaces.heuristic_discovery}`)
  }
  return parts.length
    ? parts.join(' · ')
    : 'Native audio engine for prepared bundles'
})

// ── Version activate / delete ──────────────────────────────
const activating = ref(null)
const syncingVersion = ref(null)

async function activateVersion(versionId) {
  activating.value = versionId
  try {
    await enginesStore.activateVersion(versionId)
    toast.add({ severity: 'success', summary: 'Version activated', life: 3000 })
  } catch (e) {
    toast.add({ severity: 'error', summary: 'Failed', detail: e.message, life: 4000 })
  } finally {
    activating.value = null
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

function confirmDeleteVersion(versionId) {
  const allVersions = [
    ...(enginesStore.llamaVersions || []),
    ...(enginesStore.ikLlamaVersions || []),
    ...(enginesStore.lmdeployVersions || []),
    ...(enginesStore.onecatVllmVersions || []),
    ...(enginesStore.audioCppVersions || []),
  ]
  const version = allVersions.find(v => (v.id ?? v.version) === versionId)
  if (version?.is_active) {
    toast.add({
      severity: 'warn',
      summary: 'Cannot delete active version',
      detail: 'Activate another engine version before deleting this one.',
      life: 3000,
    })
    return
  }

  confirm.require({
    message: `Delete version "${versionId}"?`,
    header: 'Confirm Delete',
    icon: 'pi pi-exclamation-triangle',
    acceptClass: 'p-button-danger',
    accept: async () => {
      try {
        await enginesStore.deleteVersion(versionId)
        toast.add({ severity: 'info', summary: 'Version deleted', life: 3000 })
      } catch (e) {
        toast.add({ severity: 'error', summary: 'Failed', detail: e.message, life: 4000 })
      }
    },
  })
}

// ── Update check (shared normalizer for llama/ik_llama API shape) ─────────
function normalizeLlamaUpdateInfo(raw, currentVersion, commitUrlPrefix) {
  if (!raw?.latest_release && !raw?.latest_commit) return null
  const latestVersion = raw.latest_release?.tag_name || raw.latest_commit?.sha?.slice(0, 8) || null
  const releaseUrl = raw.latest_release?.html_url ||
    (raw.latest_commit ? `${commitUrlPrefix}/commit/${raw.latest_commit.sha}` : null)
  const current = currentVersion || 'none'
  const updateAvailable = latestVersion && current !== latestVersion
  return {
    update_available: updateAvailable,
    latest_version: latestVersion,
    release_url: releaseUrl,
    current_version: current,
    available_tags: raw.available_tags || (raw.latest_release?.tag_name ? [raw.latest_release.tag_name] : []),
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

async function getMergedLlamaCppBuildConfig() {
  return getMergedCmakeBuildConfig('llama_cpp')
}

async function installLlamaLatestRelease() {
  llamaReleaseInstalling.value = true
  try {
    await enginesStore.updateEngine('llama_cpp')
    toast.add({
      severity: 'success',
      summary: 'Build started',
      detail: 'Building the latest GitHub release with your saved build settings. Track progress in notifications.',
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
    toast.add({ severity: 'success', summary: 'Build started', detail: 'Track progress in notifications', life: 3000 })
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
      detail: 'Building tip of main with your saved build settings. Track progress in notifications.',
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
    toast.add({ severity: 'success', summary: 'Build started', detail: 'Track progress in notifications', life: 3000 })
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
    toast.add({ severity: 'warn', summary: 'Could not check updates', detail: e.message, life: 3000 })
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
    toast.add({ severity: 'warn', summary: 'Could not check updates', detail: e.message, life: 3000 })
  } finally {
    checkingIkLlama.value = false
  }
}

// ── Build from source dialog ───────────────────────────────
const buildDialogVisible = ref(false)
const buildTarget = ref('llama_cpp')
const building = ref(false)
const savingBuildSettings = ref(false)

const buildTypeOptions = [
  { label: 'Release', value: 'Release' },
  { label: 'Debug', value: 'Debug' },
  { label: 'RelWithDebInfo', value: 'RelWithDebInfo' },
  { label: 'MinSizeRel', value: 'MinSizeRel' },
]

/** Fallback catalog if GET /build-options fails (kept in sync with backend defaults). */
const FALLBACK_BUILD_DEFAULTS = {
  build_type: 'Release',
  cuda: false,
  hip: false,
  vulkan: false,
  metal: false,
  sycl: false,
  opencl: false,
  musa: false,
  webgpu: false,
  rpc: false,
  blas: false,
  openblas: false,
  flash_attention: false,
  cuda_fa: true,
  cuda_graphs: true,
  build_common: true,
  build_tests: true,
  build_tools: true,
  build_examples: true,
  build_server: true,
  build_app: true,
  build_ui: true,
  use_prebuilt_ui: true,
  install_tools: true,
  install_tests: true,
  openssl: true,
  backend_dl: false,
  cpu_all_variants: false,
  lto: false,
  native: true,
  ccache: true,
  openmp: true,
  cpu: true,
  custom_cmake_args: '',
  cuda_architectures: '',
  blas_vendor: 'OpenBLAS',
  cflags: '',
  cxxflags: '',
}

const buildOptionsCatalog = ref({ categories: [], defaults: { ...FALLBACK_BUILD_DEFAULTS }, build_types: ['Release', 'Debug', 'RelWithDebInfo', 'MinSizeRel'] })
const buildOptionsLoading = ref(false)
const buildOptionsCatalogEngine = ref(null)

const buildForm = ref({
  commitSha: '',
  versionSuffix: '',
  buildConfig: { ...FALLBACK_BUILD_DEFAULTS },
})

const visibleBuildCategories = computed(() => {
  const cats = buildOptionsCatalog.value?.categories || []
  return cats.filter((cat) => {
    if (cat.id === 'advanced') return false
    if (!cat.requires) return true
    return !!buildForm.value?.buildConfig?.[cat.requires]
  }).map((cat) => ({
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

function primaryBackendOptions(cat) {
  return (cat.options || []).filter((o) => o.primary !== false && optionVisible(o))
}

function extraBackendOptions(cat) {
  return (cat.options || []).filter((o) => o.primary === false && optionVisible(o))
}

function _defaultBuildConfig() {
  const defaults = { ...(buildOptionsCatalog.value?.defaults || FALLBACK_BUILD_DEFAULTS) }
  return { ...FALLBACK_BUILD_DEFAULTS, ...defaults }
}

async function ensureBuildOptionsCatalog(engineId) {
  const engine = engineId === 'ik_llama' ? 'ik_llama' : 'llama_cpp'
  if (
    (buildOptionsCatalog.value?.categories || []).length
    && buildOptionsCatalogEngine.value === engine
  ) {
    return
  }
  buildOptionsLoading.value = true
  try {
    const data = await enginesStore.fetchBuildOptions(engine)
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
  if (/^(?:release-\d+(?:\.\d+)*(?:[-+][0-9A-Za-z._-]*)?|v?\d+(?:\.\d+){1,}(?:[-+][0-9A-Za-z._-]+)?|b\d+)$/i.test(value)) {
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

async function openBuildDialog(engineKey) {
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
  // Legacy openblas → blas
  if (baseConfig.openblas && !baseConfig.blas) {
    baseConfig.blas = true
    if (!baseConfig.blas_vendor) baseConfig.blas_vendor = 'OpenBLAS'
  }
  buildForm.value.commitSha =
    trackingRef
    || updateInfo?.latest_version
    || (engineKey === 'ik_llama' ? 'main' : 'master')
  buildForm.value.versionSuffix = ''
  buildForm.value.buildConfig = baseConfig
  persistBuildHintDismissed(engineKey)
  if (engineKey === 'ik_llama') {
    hintRevIk.value += 1
  } else {
    hintRevLlama.value += 1
  }
  buildDialogVisible.value = true
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
      commit_sha: buildForm.value.commitSha || (buildTarget.value === 'ik_llama' ? 'main' : 'master'),
      repository_source: repoSource,
      version_suffix: buildForm.value.versionSuffix || undefined,
      build_config: config,
      auto_activate: false,
      source_ref_type: inferSourceRefType(buildForm.value.commitSha || (buildTarget.value === 'ik_llama' ? 'main' : 'master')),
    })
    buildDialogVisible.value = false
    toast.add({ severity: 'success', summary: 'Build started', detail: 'Track progress in notifications', life: 3000 })
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
    toast.add({
      severity: 'success',
      summary: 'Build settings saved',
      detail: 'Options stored without building.',
      life: 2500,
    })
  } catch (e) {
    toast.add({ severity: 'error', summary: 'Save failed', detail: e.message, life: 4000 })
  } finally {
    savingBuildSettings.value = false
  }
}

async function doUpdateEngine(engineKey) {
  const updateInfo = engineKey === 'ik_llama' ? ikLlamaUpdateInfo.value : llamaCppUpdateInfo.value
  if (!updateInfo?.latest_version) {
    toast.add({ severity: 'warn', summary: 'No update available', detail: 'Check for updates first.', life: 3000 })
    return
  }
  const engineId = engineKey === 'ik_llama' ? 'ik_llama' : 'llama_cpp'
  updatingEngine.value = engineKey
  try {
    await updateEngineWithSavedSettings(engineId)
    toast.add({ severity: 'success', summary: 'Update started', detail: 'Build in progress — track in notifications.', life: 3000 })
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
const showAffectedAudioModels = ref(false)
const hintRevAudio = ref(0)
const audioCppReleaseInstalling = ref(false)
const audioCppSourceInstalling = ref(false)
const audioCppSourceDialogVisible = ref(false)
const audioCppSourceRepo = ref('https://github.com/0xShug0/audio.cpp.git')
const audioCppSourceRef = ref('')

const audioCppCapabilityDelta = computed(() => (
  enginesStore.audioCppStatus?.capability_delta || {
    added_families: [],
    removed_families: [],
    added_tasks: [],
    removed_tasks: [],
    families_without_tasks: [],
    warnings: [],
  }
))

const audioCppDeltaHasChanges = computed(() => {
  const delta = audioCppCapabilityDelta.value
  const grade = enginesStore.audioCppStatus?.contract_grade
  return Boolean(
    (delta.added_families || []).length
    || (delta.removed_families || []).length
    || (delta.added_tasks || []).length
    || (delta.removed_tasks || []).length
    || (delta.families_without_tasks || []).length
    || (delta.warnings || []).length
    || (grade && grade !== 'full'),
  )
})

const audioCppDeltaSummary = computed(() => {
  const delta = audioCppCapabilityDelta.value
  const lines = []
  const grade = enginesStore.audioCppStatus?.contract_grade
  if (grade) {
    lines.push(`Contract grade: ${grade}`)
  }
  if (enginesStore.audioCppStatus?.discovery_source || enginesStore.audioCppStatus?.catalog_source) {
    lines.push(
      `Sources: loaders=${enginesStore.audioCppStatus?.discovery_source || 'unknown'}`
      + ` · catalog=${enginesStore.audioCppStatus?.catalog_source || 'unknown'}`,
    )
  }
  if ((delta.added_families || []).length) {
    lines.push(`Added families: ${delta.added_families.join(', ')}`)
  }
  if ((delta.removed_families || []).length) {
    lines.push(`Removed families: ${delta.removed_families.join(', ')}`)
  }
  if ((delta.added_tasks || []).length) {
    lines.push(`Added tasks: ${delta.added_tasks.join(', ')}`)
  }
  if ((delta.removed_tasks || []).length) {
    lines.push(`Removed tasks: ${delta.removed_tasks.join(', ')}`)
  }
  if ((delta.families_without_tasks || []).length) {
    lines.push(`Families without tasks: ${delta.families_without_tasks.join(', ')}`)
  }
  for (const warning of (delta.warnings || enginesStore.audioCppStatus?.contract_warnings || [])) {
    if (warning) lines.push(String(warning))
  }
  return lines
})

const affectedAudioModels = computed(() => (
  Array.isArray(enginesStore.audioCppStatus?.affected_models)
    ? enginesStore.audioCppStatus.affected_models
    : []
))
const audioCppBuilding = ref(false)
const savingAudioCppBuildSettings = ref(false)
const audioCppBuildDialogVisible = ref(false)
const audioCppOptionsLoading = ref(false)
const audioCppOptionsCatalog = ref({ categories: [], defaults: {}, build_types: [] })

const AUDIO_FALLBACK_DEFAULTS = {
  build_type: 'RelWithDebInfo',
  cuda: false,
  hip: false,
  vulkan: false,
  metal: false,
  native_cpu: true,
  openmp: true,
  cuda_graphs: true,
  llamafile: true,
  cpu_all_variants: false,
  build_tests: false,
  build_examples: false,
  build_warmbench: false,
  deployment_build: false,
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
  if (requires === 'cuda_or_hip') return !!(cfg.cuda || cfg.hip)
  if (requires === 'model_set_custom') return cfg.model_set === 'custom'
  return !!cfg[requires]
}

function visibleAudioOptions(cat) {
  return (cat.options || []).filter((opt) => !opt.requires || audioOptionParentEnabled(opt.requires))
}

function audioBackendDisabled(key) {
  const supported = new Set(enginesStore.audioCppStatus?.supported_build_backends || ['cpu', 'cuda', 'hip', 'vulkan'])
  return !supported.has(key)
}

function onAudioBackendToggle(key) {
  const cfg = audioCppBuildForm.value.build_config
  // CUDA ↔ HIP mutual exclusion
  if (key === 'cuda' && cfg.cuda) cfg.hip = false
  if (key === 'hip' && cfg.hip) cfg.cuda = false
}

const audioCppUpdateTooltip = computed(() => {
  const releaseTag = audioCppUpdateInfo.value?.latest_release?.tag_name
  if (releaseTag) {
    return `Build source at latest GitHub release tag ${releaseTag}`
  }
  const ref =
    audioCppUpdateInfo.value?.tracking_ref ||
    enginesStore.audioCppStatus?.tracking_ref ||
    'tracked ref'
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
  // Legacy backend string
  if (raw.backend && !build_config.cuda && !build_config.hip && !build_config.vulkan && !build_config.metal) {
    if (['cuda', 'hip', 'vulkan', 'metal'].includes(raw.backend)) {
      build_config[raw.backend] = true
    }
  }
  return { tracking_ref, repository_url, build_config }
}

async function ensureAudioCppOptionsCatalog() {
  if ((audioCppOptionsCatalog.value?.categories || []).length) return
  audioCppOptionsLoading.value = true
  try {
    const data = await enginesStore.fetchAudioCppBuildOptions()
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
  await ensureAudioCppOptionsCatalog()
  const base = _defaultAudioBuildConfig()
  try {
    const saved = await enginesStore.fetchAudioCppBuildSettings()
    const split = splitAudioCppSettings(saved)
    audioCppBuildForm.value.repository_url = split.repository_url
    audioCppBuildForm.value.source_ref =
      split.tracking_ref || enginesStore.audioCppStatus?.tracking_ref || 'main'
    audioCppBuildForm.value.build_config = { ...base, ...split.build_config }
  } catch {
    audioCppBuildForm.value.build_config = base
    audioCppBuildForm.value.source_ref =
      enginesStore.audioCppStatus?.tracking_ref || 'main'
  }
  audioCppBuildForm.value.versionSuffix = ''
  persistBuildHintDismissed('audio_cpp')
  hintRevAudio.value += 1
  audioCppBuildDialogVisible.value = true
}

function audioCppSettingsPayloadFromForm() {
  const buildConfig = { ...audioCppBuildForm.value.build_config }
  const sourceRef = String(audioCppBuildForm.value.source_ref || '').trim()
  const sourceRefType = sourceRef ? inferSourceRefType(sourceRef) : 'branch'
  return {
    ...buildConfig,
    tracking_ref: sourceRef && sourceRefType !== 'commit' ? sourceRef : undefined,
    repository_url: String(audioCppBuildForm.value.repository_url || '').trim()
      || 'https://github.com/0xShug0/audio.cpp.git',
  }
}

async function saveAudioCppBuildSettingsOnly() {
  savingAudioCppBuildSettings.value = true
  try {
    await enginesStore.saveAudioCppBuildSettings(audioCppSettingsPayloadFromForm())
    await enginesStore.fetchAudioCppStatus()
    audioCppBuildDialogVisible.value = false
    toast.add({
      severity: 'success',
      summary: 'audio.cpp build settings saved',
      detail: 'CMake options were stored without building.',
      life: 3000,
    })
  } catch (e) {
    toast.add({
      severity: 'error',
      summary: 'Save failed',
      detail: e?.response?.data?.detail || e.message,
      life: 4000,
    })
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
      repository_url: audioCppBuildForm.value.repository_url || 'https://github.com/0xShug0/audio.cpp.git',
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
    audioCppSourceRef.value = split.tracking_ref || enginesStore.audioCppStatus?.tracking_ref || 'main'
  } catch {
    audioCppSourceRepo.value = 'https://github.com/0xShug0/audio.cpp.git'
    audioCppSourceRef.value = enginesStore.audioCppStatus?.tracking_ref || 'main'
  }
  audioCppSourceDialogVisible.value = true
}

async function installAudioLatestRelease() {
  audioCppReleaseInstalling.value = true
  try {
    const saved = await enginesStore.fetchAudioCppBuildSettings()
    const split = splitAudioCppSettings(saved)
    await enginesStore.updateAudioCpp({
      from_release: true,
      build_config: split.build_config,
      repository_url: split.repository_url,
    })
    await enginesStore.fetchAudioCppStatus()
    toast.add({
      severity: 'info',
      summary: 'Building latest audio.cpp release',
      detail: 'Using your saved build settings. Track progress in notifications.',
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
        : { source_ref: split.tracking_ref || enginesStore.audioCppStatus?.tracking_ref }),
    })
    await enginesStore.fetchAudioCppStatus()
    toast.add({
      severity: 'success',
      summary: data?.sync ? 'audio.cpp sync started' : 'audio.cpp update started',
      detail: 'Track progress in notifications',
      life: 3500,
    })
    if (enginesStore.audioCppStatus?.contract_changed) {
      toast.add({
        severity: 'info',
        summary: 'audio.cpp capabilities may have changed',
        detail: 'CLI/help contract fingerprint differs from the previous scan.',
        life: 4500,
      })
    }
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

// ── CUDA ───────────────────────────────────────────────────
const cuda = computed(() => enginesStore.cudaStatus || {})
const cudaVersionOptions = ['12.9', '12.8', '12.7', '12.6', '12.5', '12.4', '12.3', '12.2', '12.1', '12.0', '11.9', '11.8']
const cudaInstallVersion = ref(null)
const cudaInstalling = ref(false)
const cudaInstallDialogVisible = ref(false)

async function installCuda() {
  cudaInstalling.value = true
  try {
    await enginesStore.installCuda({ version: cudaInstallVersion.value })
    cudaInstallDialogVisible.value = false
    toast.add({ severity: 'success', summary: 'CUDA install started', detail: 'Track progress in notifications', life: 3000 })
    await enginesStore.fetchCudaStatus()
  } catch (e) {
    toast.add({ severity: 'error', summary: 'Failed', detail: e.message, life: 4000 })
  } finally {
    cudaInstalling.value = false
  }
}

function confirmUninstallCuda(version) {
  confirm.require({
    message: `Uninstall CUDA ${version}?`,
    header: 'Confirm Uninstall',
    icon: 'pi pi-exclamation-triangle',
    acceptClass: 'p-button-danger',
    accept: async () => {
      try {
        await enginesStore.uninstallCuda({ version })
        toast.add({ severity: 'info', summary: `CUDA ${version} uninstalled`, life: 3000 })
      } catch (e) {
        toast.add({ severity: 'error', summary: 'Failed', detail: e.message, life: 4000 })
      }
    },
  })
}

// ── LMDeploy ───────────────────────────────────────────────
const lm = computed(() => enginesStore.lmdeployStatus || {})
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
})

async function applyLmdeployBuildSettings(saved) {
  const s = saved && typeof saved === 'object' ? saved : {}
  lmdeployBuildForm.value = {
    source_repo: s.source_repo || 'https://github.com/InternLM/lmdeploy.git',
    source_branch: s.source_branch || 'main',
    pip_version: s.pip_version || '',
  }
  lmSourceRepo.value = lmdeployBuildForm.value.source_repo
  lmSourceBranch.value = lmdeployBuildForm.value.source_branch
  lmdeployPipVersion.value = lmdeployBuildForm.value.pip_version
}

async function openLmdeployBuildSettings() {
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
    toast.add({ severity: 'warn', summary: 'No update available', detail: 'Check for updates first.', life: 3000 })
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
    toast.add({
      severity: 'success',
      summary: 'Build settings saved',
      detail: 'LMDeploy defaults stored without installing.',
      life: 2500,
    })
  } catch (e) {
    toast.add({ severity: 'error', summary: 'Save failed', detail: e.message, life: 4000 })
  } finally {
    savingLmdeployBuildSettings.value = false
  }
}

async function installLmdeployFromBuildSettings() {
  lmdeployInstalling.value = true
  try {
    await enginesStore.saveLmdeployBuildSettings({ ...lmdeployBuildForm.value })
    await enginesStore.installLmdeployFromSource({
      repo_url: lmdeployBuildForm.value.source_repo,
      branch: lmdeployBuildForm.value.source_branch,
    })
    await applyLmdeployBuildSettings(lmdeployBuildForm.value)
    lmdeployBuildDialogVisible.value = false
    toast.add({ severity: 'success', summary: 'Install from source started', detail: 'Track progress in notifications', life: 3000 })
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
    toast.add({ severity: 'warn', summary: 'Could not check updates', detail: e.message, life: 3000 })
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
    await enginesStore.installLmdeploy(lmdeployPipVersion.value ? { version: lmdeployPipVersion.value } : {})
    lmPipDialogVisible.value = false
    toast.add({ severity: 'success', summary: 'LMDeploy install started', detail: 'Track progress in notifications', life: 3000 })
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
    toast.add({ severity: 'success', summary: 'Install from source started', detail: 'Track progress in notifications', life: 3000 })
  } catch (e) {
    toast.add({ severity: 'error', summary: 'Failed', detail: e.message, life: 4000 })
  } finally {
    lmdeployInstalling.value = false
    // Ensure the modal doesn't stay in a "loading" state after failure.
    lmSourceDialogVisible.value = false
  }
}

// ── 1Cat-vLLM ──────────────────────────────────────────────
const ovllm = computed(() => enginesStore.onecatVllmStatus || {})
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
    toast.add({ severity: 'warn', summary: 'No update available', detail: 'Check for updates first.', life: 3000 })
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
    toast.add({
      severity: 'success',
      summary: 'Build settings saved',
      detail: '1Cat-vLLM defaults stored without installing.',
      life: 2500,
    })
  } catch (e) {
    toast.add({ severity: 'error', summary: 'Save failed', detail: e.message, life: 4000 })
  } finally {
    savingOnecatVllmBuildSettings.value = false
  }
}

async function installOnecatVllmFromBuildSettings() {
  onecatVllmInstalling.value = true
  try {
    await enginesStore.saveOnecatVllmBuildSettings({ ...onecatVllmBuildForm.value })
    await enginesStore.installOnecatVllmFromSource({
      repo_url: onecatVllmBuildForm.value.source_repo,
      branch: onecatVllmBuildForm.value.source_branch,
    })
    await applyOnecatVllmBuildSettings(onecatVllmBuildForm.value)
    onecatVllmBuildDialogVisible.value = false
    toast.add({ severity: 'success', summary: 'Source build started', detail: 'Track progress in notifications', life: 3000 })
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
    toast.add({ severity: 'warn', summary: 'Could not check updates', detail: e.message, life: 3000 })
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
    await enginesStore.installOnecatVllm(ovllmReleaseVersion.value ? { version: ovllmReleaseVersion.value } : {})
    toast.add({ severity: 'success', summary: '1Cat-vLLM install started', detail: 'Track progress in notifications', life: 3000 })
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
    toast.add({ severity: 'success', summary: 'Source build started', detail: 'Track progress in notifications', life: 3000 })
  } catch (e) {
    toast.add({ severity: 'error', summary: 'Failed', detail: e.message, life: 4000 })
  } finally {
    onecatVllmInstalling.value = false
    ovllmSourceDialogVisible.value = false
  }
}

// ── Lifecycle ──────────────────────────────────────────────
let unsubscribeTaskUpdated = null

onMounted(() => {
  enginesStore.fetchAll()
  focusRoutingSection()
  unsubscribeTaskUpdated = progressStore.subscribe('task_updated', async (task) => {
    if (task?.status !== 'completed' && task?.status !== 'failed') return

    const manager = task?.metadata?.manager

    if (manager === 'cuda') {
      await enginesStore.fetchCudaStatus()
      return
    }

    if (manager === 'lmdeploy' || manager === 'onecat_vllm') {
      if (task.status === 'failed') {
        const detail = task.message || `${manager === 'lmdeploy' ? 'LMDeploy' : '1Cat-vLLM'} operation failed`
        toast.add({
          severity: 'error',
          summary: manager === 'lmdeploy' ? 'LMDeploy install failed' : '1Cat-vLLM install failed',
          detail,
          life: 5000,
        })
      }
      await Promise.allSettled([
        manager === 'lmdeploy' ? enginesStore.fetchLmdeployStatus() : enginesStore.fetchOnecatVllmStatus(),
        enginesStore.fetchLlamaVersions(),
      ])
      return
    }

    if (task?.type === 'build') {
      const refreshTasks = [
        enginesStore.fetchLlamaVersions(),
        enginesStore.fetchSystemStatus(),
      ]
      if (task.metadata?.engine === 'audio_cpp') {
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
.ev-collapse-leave-active { transition: all 0.2s ease; overflow: hidden; }
.ev-collapse-enter-from,
.ev-collapse-leave-to    { max-height: 0; opacity: 0; }
.ev-collapse-enter-to,
.ev-collapse-leave-from  { max-height: 600px; opacity: 1; }

/* ── Section ─────────────────────────────────────────── */
.ev-section {
  background: var(--bg-card);
  border: 1px solid var(--border-primary);
  border-radius: var(--radius-lg);
  overflow: hidden;
}

.ev-section-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 0.5rem;
  background: var(--bg-surface);
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

.engine-mark {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  min-width: 1.8rem;
  height: 1.8rem;
  padding: 0 0.45rem;
  border-radius: 999px;
  font-size: 0.72rem;
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

.ev-section-body {
  padding: 1.25rem;
}

.ev-system-layout {
  display: flex;
  flex-direction: column;
  gap: 1.25rem;
}

/* ── CUDA: one panel (no nested metric-card / double border) ───────────────── */
.cuda-toolkit-region {
  border-radius: var(--radius-md);
  border: 1px solid var(--border-primary);
  background: var(--bg-surface);
  overflow: hidden;
}

.cuda-toolkit-main {
  display: grid;
  grid-template-columns: auto 1fr auto;
  gap: 0.875rem 1rem;
  align-items: start;
  padding: 1rem 1.125rem;
}

@media (max-width: 640px) {
  .cuda-toolkit-main {
    grid-template-columns: auto 1fr;
    grid-template-rows: auto auto;
  }
  .cuda-toolkit-main__actions {
    grid-column: 1 / -1;
    justify-self: end;
  }
}

.cuda-toolkit-main__icon {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 2.5rem;
  height: 2.5rem;
  flex-shrink: 0;
  border-radius: var(--radius-md);
  background: color-mix(in srgb, var(--accent-cyan) 18%, transparent);
  color: var(--accent-cyan);
  font-size: 1.25rem;
}

.cuda-toolkit-main__body {
  min-width: 0;
}

.cuda-toolkit-main__title {
  margin: 0;
  font-size: 0.72rem;
  font-weight: 600;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: var(--text-secondary);
}

.cuda-toolkit-main__status {
  margin: 0.35rem 0 0;
  font-size: 1rem;
  font-weight: 600;
  line-height: 1.25;
  color: var(--text-primary);
}

.cuda-toolkit-main__hint {
  margin: 0.35rem 0 0;
  font-size: 0.8125rem;
  line-height: 1.45;
  color: var(--text-secondary);
}

.cuda-toolkit-main__actions {
  display: flex;
  flex-wrap: wrap;
  gap: 0.35rem;
  align-items: center;
  justify-content: flex-end;
  padding-top: 0.1rem;
}

.cuda-toolkit-details {
  padding: 0.75rem 1.125rem 1rem;
  border-top: 1px solid var(--border-primary);
}

/* ── Metrics ─────────────────────────────────────────── */
.metrics-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
  gap: 0.75rem;
}

.metric-card {
  display: flex;
  gap: 0.5rem;
  align-items: flex-start;
  background: var(--bg-surface);
  padding: 0.75rem;
  border-radius: var(--radius-md);
  border: 1px solid var(--border-primary);
}

.metric-card--actionable {
  flex-direction: row;
}

.metric-icon { font-size: 1.5rem; flex-shrink: 0; line-height: 1; color: var(--accent-cyan); }
.metric-data { flex: 1; min-width: 0; }
.metric-label { font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.05em; color: var(--text-secondary); margin-bottom: 0.2rem; }
.metric-value { font-size: 0.875rem; font-weight: 600; }
.metric-subvalue {
  margin-top: 0.25rem;
  font-size: 0.8rem;
  color: var(--text-secondary);
  word-break: break-word;
}
.metric-bar { margin-top: 0.5rem; }
.metric-actions {
  display: flex;
  gap: 0.4rem;
  flex-wrap: wrap;
  margin-top: 0.6rem;
}
/* No text inside the bar so low percentages don’t get clipped; value is shown above */

.system-subpanel {
  margin-top: 1rem;
}

/* ── Engines overview ───────────────────────────────────── */
.engine-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
  gap: 0.75rem;
}

.engine-card {
  appearance: none;
  border: 1px solid var(--border-primary);
  background: var(--bg-surface);
  border-radius: var(--radius-md);
  padding: 0.9rem;
  text-align: left;
  color: inherit;
  cursor: pointer;
  transition: border-color 0.15s ease, transform 0.15s ease, background 0.15s ease;
}

.engine-card:hover {
  border-color: var(--accent-cyan);
  background: color-mix(in srgb, var(--bg-surface) 88%, var(--accent-cyan) 12%);
  transform: translateY(-1px);
}

.engine-card:focus {
  outline: none;
}

.engine-card:focus-visible {
  outline: 2px solid var(--accent-cyan);
  outline-offset: 2px;
}

.engine-card-head {
  display: flex;
  align-items: flex-start;
  justify-content: flex-start;
  gap: 0.75rem;
}

.engine-card-version-line {
  min-width: 0;
  max-width: 100%;
  margin-bottom: 0.45rem;
}

.engine-card-version-line :deep(.p-tag) {
  max-width: 100%;
}

.engine-card-version-line :deep(.p-tag-value) {
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

.engine-card-title {
  display: flex;
  align-items: center;
  gap: 0.65rem;
  min-width: 0;
}

.engine-card-name {
  font-size: 0.95rem;
  font-weight: 600;
}

.engine-card-meta {
  font-size: 0.78rem;
  color: var(--text-secondary);
  margin-top: 0.1rem;
}

.engine-card-icon {
  font-size: 1.25rem;
  color: var(--accent-cyan);
  width: 1.8rem;
  text-align: center;
  flex-shrink: 0;
}

.engine-card-body {
  margin-top: 0.8rem;
}

.engine-card-status {
  font-size: 0.82rem;
  color: var(--text-secondary);
}

.engine-card-status--warning {
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

.ev-subsection { margin-top: 1.25rem; }
.ev-subsection h4 {
  font-size: 0.75rem;
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

.form-input      { flex: 1; }
.form-input-short { width: 140px; }

/* ── Status details ──────────────────────────────────── */
.status-detail {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  font-size: 0.875rem;
  margin-bottom: 0.75rem;
  flex-wrap: wrap;
}

.detail-label { color: var(--text-secondary); flex-shrink: 0; }

.detail-label.detail-label--error {
  color: var(--status-error);
}
.ml { margin-left: 0.75rem; }

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

.ev-version-row .version-name { flex: 1; margin: 0; }

.empty-state-mini {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.75rem;
  color: var(--text-secondary);
  font-size: 0.875rem;
  margin-bottom: 0.75rem;
}

.empty-state-mini i { color: var(--text-muted); }

.cuda-version-select { min-width: 160px; }
.lm-version-input { width: 220px; }

/* ── Update banners ──────────────────────────────────── */
.update-banner, .update-current {
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

.audio-cpp-delta-list,
.audio-cpp-affected ul {
  margin: 0;
  padding-left: 1.1rem;
}

.audio-cpp-affected__title {
  font-weight: 600;
  margin-bottom: 0.25rem;
}

.audio-cpp-affected__link {
  background: none;
  border: 0;
  padding: 0;
  color: inherit;
  text-decoration: underline;
  cursor: pointer;
  font: inherit;
}

.audio-cpp-affected__meta {
  margin-left: 0.35rem;
  opacity: 0.8;
  font-size: 0.75rem;
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

.form-field small { font-size: 0.75rem; color: var(--text-secondary); }
.optional { font-weight: 400; opacity: 0.6; }

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

.asset-option:hover { background: var(--bg-surface); }
.asset-option.selected {
  background: var(--bg-surface);
  border-color: var(--accent-cyan);
}

.asset-name { flex: 1; font-size: 0.8rem; font-family: monospace; }
.asset-size { font-size: 0.75rem; color: var(--text-secondary); }

.toggle-grid { display: flex; flex-direction: column; gap: 0.5rem; }

.toggle-row {
  display: flex;
  align-items: flex-start;
  gap: 0.75rem;
}

.opt-label { font-size: 0.875rem; font-weight: 500; display: block; }
.opt-desc  { font-size: 0.75rem; color: var(--text-secondary); display: block; }

.opt-string-field {
  display: flex;
  flex-direction: column;
  gap: 0.15rem;
  padding: 0.15rem 0;
}

.build-settings-body {
  max-height: min(70vh, 640px);
  overflow-y: auto;
  padding-right: 0.25rem;
}

.build-options-details {
  margin-bottom: 0.65rem;
  border: 1px solid var(--surface-border);
  border-radius: 6px;
  padding: 0.35rem 0.6rem 0.5rem;
  background: var(--surface-50, transparent);
}

.build-options-details > summary {
  cursor: pointer;
  list-style: none;
  margin-bottom: 0.15rem;
  user-select: none;
}

.build-options-details > summary::-webkit-details-marker { display: none; }
.build-options-details > summary::before {
  content: '▸ ';
  color: var(--text-secondary);
}
.build-options-details[open] > summary::before { content: '▾ '; }
.build-options-details[open] > summary { margin-bottom: 0.45rem; }

.build-advanced-hint {
  margin-left: 0.4rem;
  font-size: 0.7rem;
  font-weight: 500;
  color: var(--text-secondary);
  opacity: 0.85;
  text-transform: uppercase;
  letter-spacing: 0.03em;
}

.build-options-section {
  font-size: 0.8rem;
  font-weight: 600;
  color: var(--text-secondary);
  margin-bottom: 0.25rem;
  display: block;
}
.flags-row { display: flex; gap: 0.5rem; }

.build-note {
  font-size: 0.8rem;
  padding: 0.5rem 0.6rem;
  border-radius: 6px;
  margin-bottom: 0.5rem;
}
.build-note--info {
  background: var(--surface-100);
  color: var(--text-color);
  border: 1px solid var(--surface-border);
}
.build-note strong { font-weight: 600; }
</style>
