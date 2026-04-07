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

              <div class="cuda-toolkit-progress-slot">
                <ProgressTracker
                  section-title="Install progress"
                  type="install"
                  metadata-key="target"
                  metadata-value="cuda"
                  :show-completed="true"
                />
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
                  Manage builds, updates, activation, and versions
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
                  Manage builds, updates, activation, and versions
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
                  Manage installs, updates, activation, and versions
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
            <Button icon="pi pi-refresh" text severity="secondary" size="small"
              v-tooltip.top="'Reload versions and status'"
              @click="enginesStore.fetchLlamaVersions(); enginesStore.fetchLmdeployStatus()" />
            <Button icon="pi pi-list" text severity="secondary" size="small"
              v-tooltip.top="'Rescan CLI parameters (--help)'"
              :loading="paramScanLoading === 'lmdeploy'"
              @click="rescanEngineCliParams('lmdeploy')" />
          </template>
        </EngineDialogHeader>
      </template>
      <section v-if="selectedEngine === 'llama_cpp'" class="ev-section ev-section--modal">
        <div class="ev-section-body">
          <EngineBuildSettingsHint
            :key="`llama-hint-${hintRevLlama}`"
            engine-key="llama_cpp"
            @open-settings="openBuildDialog('llama_cpp')"
          />
          <EngineCheckUpdatesCta
            :loading="checkingLlamaCpp"
            @check="checkLlamaCppUpdates"
          />
          <div v-if="llamaCppUpdateInfo?.update_available" class="update-banner">
            <i class="pi pi-arrow-up-right" />
            Update available: <strong>{{ llamaCppUpdateInfo.latest_version }}</strong>
            <a :href="llamaCppUpdateInfo.release_url" target="_blank" class="update-link">View release</a>
            <Button icon="pi pi-arrow-circle-up" text severity="success" size="small"
              v-tooltip.top="'Update using saved build settings'"
              :loading="updatingEngine === 'llama_cpp'"
              @click="doUpdateEngine('llama_cpp')" />
          </div>
          <div v-else-if="llamaCppUpdateInfo" class="update-current">
            <i class="pi pi-check" /> Up to date ({{ llamaCppUpdateInfo.current_version }})
          </div>

          <div class="lmdeploy-install-panel">
            <div class="lmdeploy-install-panel__head">
              <span class="lmdeploy-install-panel__title">Install</span>
              <span class="lmdeploy-install-panel__subtitle">Add a new build from the latest release or any git repo; each build is a version you can activate.</span>
            </div>
            <div class="lmdeploy-install-panel__actions">
              <Button label="From release" icon="pi pi-tag" severity="success" outlined
                :loading="llamaReleaseInstalling"
                :disabled="llamaReleaseInstalling || llamaCppSourceInstalling"
                @click="installLlamaLatestRelease" />
              <Button label="From source" icon="pi pi-code" severity="info" outlined
                :loading="llamaCppSourceInstalling"
                :disabled="llamaReleaseInstalling || llamaCppSourceInstalling"
                @click="llamaCppSourceDialogVisible = true" />
            </div>
          </div>

          <ProgressTracker
            section-title="Build progress"
            type="build"
            metadata-key="repository_source"
            metadata-value="llama.cpp"
            :show-completed="true"
          />

          <VersionTable
            :versions="enginesStore.llamaVersions"
            :activating="activating"
            empty-message="No versions yet. Install one using the options above."
            @activate="activateVersion"
            @delete="confirmDeleteVersion"
          />
        </div>
      </section>

      <section v-else-if="selectedEngine === 'ik_llama'" class="ev-section ev-section--modal">
        <div class="ev-section-body">
          <EngineBuildSettingsHint
            :key="`ik-hint-${hintRevIk}`"
            engine-key="ik_llama"
            @open-settings="openBuildDialog('ik_llama')"
          />
          <EngineCheckUpdatesCta
            :loading="checkingIkLlama"
            @check="checkIkLlamaUpdates"
          />
          <div v-if="ikLlamaUpdateInfo?.update_available" class="update-banner">
            <i class="pi pi-arrow-up-right" />
            Update available: <strong>{{ ikLlamaUpdateInfo.latest_version }}</strong>
            <a :href="ikLlamaUpdateInfo.release_url" target="_blank" class="update-link">View</a>
            <Button icon="pi pi-arrow-circle-up" text severity="success" size="small"
              v-tooltip.top="'Update using saved build settings'"
              :loading="updatingEngine === 'ik_llama'"
              @click="doUpdateEngine('ik_llama')" />
          </div>
          <div v-else-if="ikLlamaUpdateInfo" class="update-current">
            <i class="pi pi-check" /> Up to date ({{ ikLlamaUpdateInfo.current_version }})
          </div>

          <ProgressTracker
            section-title="Build progress"
            type="build"
            metadata-key="repository_source"
            metadata-value="ik_llama.cpp"
            :show-completed="true"
          />

          <VersionTable
            :versions="enginesStore.ikLlamaVersions"
            :activating="activating"
            @activate="activateVersion"
            @delete="confirmDeleteVersion"
          />
        </div>
      </section>

      <section v-else-if="selectedEngine === 'lmdeploy'" class="ev-section ev-section--modal ev-section--lmdeploy">
        <div class="ev-section-body lmdeploy-modal-body">
          <EngineCheckUpdatesCta
            :loading="checkingLmdeploy"
            @check="checkLmdeployUpdates"
          />
          <div v-if="lmdeployUpdateInfo?.update_available" class="update-banner">
            <i class="pi pi-arrow-up-right" aria-hidden="true" />
            Update available: <strong>v{{ lmdeployUpdateInfo.latest_version }}</strong>
            <a href="https://pypi.org/project/lmdeploy/" target="_blank" class="update-link">View on PyPI</a>
          </div>
          <div v-else-if="lmdeployUpdateInfo" class="update-current">
            <i class="pi pi-check" aria-hidden="true" /> Up to date (v{{ lmdeployUpdateInfo.current_version || 'none' }})
          </div>

          <div class="lmdeploy-install-panel">
            <div class="lmdeploy-install-panel__head">
              <span class="lmdeploy-install-panel__title">Install</span>
              <span class="lmdeploy-install-panel__subtitle">Add a new environment; each install is a separate version you can activate.</span>
            </div>
            <div class="lmdeploy-install-panel__actions">
              <Button label="From PyPI" icon="pi pi-download" severity="success" outlined
                @click="lmPipDialogVisible = true" />
              <Button label="From source" icon="pi pi-code" severity="info" outlined
                @click="lmSourceDialogVisible = true" />
            </div>
          </div>

          <ProgressTracker
            section-title="Install progress"
            type="install"
            metadata-key="target"
            metadata-value="lmdeploy"
            :show-completed="true"
          />

          <div v-if="activeLmdeploy && lm.venv_path" class="status-detail">
            <span class="detail-label">Active install:</span>
            <Tag :value="activeLmdeploy.install_type || lm.install_type || 'pip'" severity="info" />
            <template v-if="lm.venv_path">
              <span class="detail-label ml">Venv:</span>
              <code>{{ lm.venv_path }}</code>
            </template>
          </div>
          <div v-if="lm.source_repo" class="status-detail">
            <span class="detail-label">Source:</span>
            <code>{{ lm.source_repo }} ({{ lm.source_branch }})</code>
          </div>

          <div v-if="lm.last_error" class="status-detail">
            <span class="detail-label detail-label--error">Last error:</span>
            <code>{{ lm.last_error }}</code>
          </div>

          <div class="lmdeploy-versions-block">
            <div class="lmdeploy-versions-heading">Installed versions</div>
            <VersionTable
              :versions="enginesStore.lmdeployVersions"
              :activating="activating"
              empty-message="No versions yet. Install one using the options above."
              @activate="activateVersion"
              @delete="confirmDeleteVersion"
            />
          </div>
        </div>
      </section>
    </Dialog>

    <!-- ── Build Settings Dialog ─────────────────────────── -->
    <Dialog v-model:visible="buildDialogVisible"
      :header="`Build settings — ${buildTarget === 'ik_llama' ? 'ik_llama.cpp' : 'llama.cpp'}`"
      modal class="build-settings-dialog dialog-width-md">
      <div class="dialog-body">
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
        <div class="form-field">
          <label class="build-options-section">GPU &amp; backends</label>
          <div class="toggle-grid">
            <div v-for="opt in buildOptionsGpu" :key="opt.key" class="toggle-row">
              <InputSwitch v-model="buildForm.buildConfig[opt.key]" />
              <div>
                <span class="opt-label">{{ opt.label }}</span>
                <small class="opt-desc">{{ opt.desc }}</small>
              </div>
            </div>
          </div>
        </div>
        <div class="form-field">
          <label class="build-options-section">Build artifacts</label>
          <div v-if="buildTarget === 'ik_llama'" class="build-note build-note--info">
            For ik_llama.cpp, <strong>Examples</strong> is required (server binary lives in examples).
          </div>
          <div class="toggle-grid">
            <div v-for="opt in buildOptionsArtifacts" :key="opt.key" class="toggle-row">
              <InputSwitch
                v-model="buildForm.buildConfig[opt.key]"
                :disabled="buildTarget === 'ik_llama' && opt.key === 'build_examples'"
              />
              <div>
                <span class="opt-label">{{ opt.label }}</span>
                <small class="opt-desc">{{ opt.desc }}</small>
              </div>
            </div>
          </div>
        </div>
        <div class="form-field">
          <label class="build-options-section">GGML / CPU options</label>
          <div class="toggle-grid">
            <div v-for="opt in buildOptionsGGML" :key="opt.key" class="toggle-row">
              <InputSwitch v-model="buildForm.buildConfig[opt.key]" />
              <div>
                <span class="opt-label">{{ opt.label }}</span>
                <small class="opt-desc">{{ opt.desc }}</small>
              </div>
            </div>
          </div>
        </div>
        <div v-if="buildForm.buildConfig.cuda" class="form-field">
          <label>CUDA Architectures <span class="optional">(optional)</span></label>
          <InputText v-model="buildForm.buildConfig.cuda_architectures"
            placeholder="e.g. 86;89 (blank = auto)" class="w-full" />
        </div>
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

  </div>
</template>

<script setup>
import { ref, computed, onMounted, onUnmounted } from 'vue'
import { useConfirm } from 'primevue/useconfirm'
import { useToast } from 'primevue/usetoast'
import Button from 'primevue/button'
import Tag from 'primevue/tag'
import ProgressBar from 'primevue/progressbar'
import Dialog from 'primevue/dialog'
import Dropdown from 'primevue/dropdown'
import InputText from 'primevue/inputtext'
import InputSwitch from 'primevue/inputswitch'
import ProgressTracker from '@/components/common/ProgressTracker.vue'
import EngineDialogHeader from '@/components/system/EngineDialogHeader.vue'
import EngineCheckUpdatesCta from '@/components/system/EngineCheckUpdatesCta.vue'
import EngineBuildSettingsHint from '@/components/system/EngineBuildSettingsHint.vue'
import VersionTable from '@/components/system/VersionTable.vue'
import { useEnginesStore } from '@/stores/engines'
import { useProgressStore } from '@/stores/progress'
import { formatBytesIEC } from '@/utils/formatting'

const enginesStore = useEnginesStore()
const progressStore = useProgressStore()
const confirm = useConfirm()
const toast = useToast()

// ── System metrics ─────────────────────────────────────────
const systemExpanded = ref(true)
const enginesExpanded = ref(true)
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
  }
}

async function refreshEnginesOverview() {
  await Promise.allSettled([
    enginesStore.fetchLlamaVersions(),
    enginesStore.fetchLmdeployStatus(),
    checkLlamaCppUpdates(),
    checkIkLlamaUpdates(),
    checkLmdeployUpdates(),
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

// ── Version activate / delete ──────────────────────────────
const activating = ref(null)

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

function confirmDeleteVersion(versionId) {
  const allVersions = [
    ...(enginesStore.llamaVersions || []),
    ...(enginesStore.ikLlamaVersions || []),
    ...(enginesStore.lmdeployVersions || []),
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

const checkingLlamaCpp = ref(false)
const llamaCppUpdateInfo = ref(null)
const updatingEngine = ref(null)

/** Bumps when build settings open so EngineBuildSettingsHint remounts and hides after LS dismiss. */
const hintRevLlama = ref(0)
const hintRevIk = ref(0)

const BUILD_HINT_LS_KEY = 'lcs.engine.buildSettingsHintDismissed.v1'

function persistBuildHintDismissed(engineKey) {
  const k = engineKey === 'ik_llama' ? 'ik_llama' : 'llama_cpp'
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

async function getMergedLlamaCppBuildConfig() {
  const base = _defaultBuildConfig()
  try {
    const saved = await fetchEngineBuildSettings('llama_cpp')
    Object.assign(base, saved || {})
  } catch {
    // use defaults
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
      detail: 'Building the latest GitHub release with your saved build settings.',
      life: 3500,
    })
  } catch (e) {
    toast.add({ severity: 'error', summary: 'Failed', detail: e.message, life: 4000 })
  } finally {
    llamaReleaseInstalling.value = false
  }
}

async function installLlamaCppFromSource() {
  llamaCppSourceInstalling.value = true
  try {
    const config = await getMergedLlamaCppBuildConfig()
    const ref = (llamaCppSourceRef.value || 'master').trim()
    const repo = (llamaCppSourceRepo.value || '').trim()
    const payload = {
      commit_sha: ref,
      repository_source: 'llama.cpp',
      build_config: config,
      auto_activate: false,
    }
    if (repo) {
      payload.repository_url = repo
    }
    await enginesStore.buildSource(payload)
    llamaCppSourceDialogVisible.value = false
    toast.add({ severity: 'success', summary: 'Build started', detail: 'Track progress below', life: 3000 })
  } catch (e) {
    toast.add({ severity: 'error', summary: 'Build failed', detail: e.message, life: 4000 })
  } finally {
    llamaCppSourceInstalling.value = false
  }
}

async function checkLlamaCppUpdates() {
  checkingLlamaCpp.value = true
  try {
    const raw = await enginesStore.checkLlamaCppUpdates()
    llamaCppUpdateInfo.value = normalizeLlamaUpdateInfo(
      raw,
      activeLlamaCpp.value?.source_ref || activeLlamaCpp.value?.source_commit || activeLlamaCpp.value?.version,
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
      activeIkLlama.value?.source_ref || activeIkLlama.value?.source_commit || activeIkLlama.value?.version,
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
const buildForm = ref({
  commitSha: '',
  versionSuffix: '',
  buildConfig: _defaultBuildConfig(),
})

const buildTypeOptions = [
  { label: 'Release', value: 'Release' },
  { label: 'Debug', value: 'Debug' },
  { label: 'RelWithDebInfo', value: 'RelWithDebInfo' },
  { label: 'MinSizeRel', value: 'MinSizeRel' },
]

const buildOptionsGpu = [
  { key: 'cuda', label: 'CUDA', desc: 'GGML_CUDA=on' },
  { key: 'flash_attention', label: 'Flash Attention', desc: 'GGML_CUDA_FA_ALL_QUANTS=on (requires CUDA)' },
  { key: 'openblas', label: 'OpenBLAS', desc: 'GGML_BLAS=on (CPU acceleration)' },
]

const buildOptionsArtifacts = [
  { key: 'build_common', label: 'Common lib', desc: 'LLAMA_BUILD_COMMON=on' },
  { key: 'build_tests', label: 'Tests', desc: 'LLAMA_BUILD_TESTS=on' },
  { key: 'build_tools', label: 'Tools', desc: 'LLAMA_BUILD_TOOLS=on' },
  { key: 'build_examples', label: 'Examples', desc: 'LLAMA_BUILD_EXAMPLES=on' },
  { key: 'build_server', label: 'Server', desc: 'LLAMA_BUILD_SERVER=on (required for serving)' },
  { key: 'install_tools', label: 'Install tools', desc: 'LLAMA_TOOLS_INSTALL=on' },
]

const buildOptionsGGML = [
  { key: 'native', label: 'Native CPU', desc: 'GGML_NATIVE=on' },
  { key: 'backend_dl', label: 'Backend DL', desc: 'GGML_BACKEND_DL=on' },
  { key: 'cpu_all_variants', label: 'CPU all variants', desc: 'GGML_CPU_ALL_VARIANTS=on' },
  { key: 'lto', label: 'LTO', desc: 'GGML_LTO=on (link-time optimization)' },
]

function _defaultBuildConfig() {
  return {
    build_type: 'Release',
    cuda: false,
    openblas: false,
    flash_attention: false,
    build_common: true,
    build_tests: true,
    build_tools: true,
    build_examples: true,
    build_server: true,
    install_tools: true,
    backend_dl: false,
    cpu_all_variants: false,
    lto: false,
    native: true,
    custom_cmake_args: '',
    cuda_architectures: '',
    cflags: '',
    cxxflags: '',
  }
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

async function openBuildDialog(engineKey) {
  buildTarget.value = engineKey
  const engineId = engineKey === 'ik_llama' ? 'ik_llama' : 'llama_cpp'
  const updateInfo = engineKey === 'ik_llama' ? ikLlamaUpdateInfo.value : llamaCppUpdateInfo.value
  const baseConfig = _defaultBuildConfig()
  try {
    const saved = await fetchEngineBuildSettings(engineId)
    Object.assign(baseConfig, saved || {})
  } catch {
    // Ignore, fall back to defaults
  }
  // ik_llama.cpp requires Build examples (server is in examples/)
  if (engineKey === 'ik_llama') {
    baseConfig.build_examples = true
  }
  buildForm.value.commitSha = updateInfo?.latest_version || (engineKey === 'ik_llama' ? 'main' : 'master')
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
    // Persist settings before triggering a manual build (full config)
    await saveEngineBuildSettings(engineId, config)
    await enginesStore.buildSource({
      commit_sha: buildForm.value.commitSha || (buildTarget.value === 'ik_llama' ? 'main' : 'master'),
      repository_source: repoSource,
      version_suffix: buildForm.value.versionSuffix || undefined,
      build_config: config,
      auto_activate: false,
    })
    buildDialogVisible.value = false
    toast.add({ severity: 'success', summary: 'Build started', detail: 'Track progress below', life: 3000 })
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
    await saveEngineBuildSettings(engineId, config)
    buildDialogVisible.value = false
    toast.add({ severity: 'success', summary: 'Build settings saved', life: 2500 })
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
    toast.add({ severity: 'success', summary: 'Update started', detail: 'Build in progress, track below.', life: 3000 })
  } catch (e) {
    toast.add({ severity: 'error', summary: 'Update failed', detail: e.message, life: 4000 })
  } finally {
    updatingEngine.value = null
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
    toast.add({ severity: 'success', summary: 'CUDA install started', detail: 'Track progress below', life: 3000 })
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
    await enginesStore.installLmdeploy(lmdeployPipVersion.value ? { version: lmdeployPipVersion.value } : {})
    lmPipDialogVisible.value = false
    toast.add({ severity: 'success', summary: 'LMDeploy install started', detail: 'Track progress below', life: 3000 })
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
    await enginesStore.installLmdeployFromSource({
      repo_url: lmSourceRepo.value,
      branch: lmSourceBranch.value,
    })
    toast.add({ severity: 'success', summary: 'Install from source started', detail: 'Track progress below', life: 3000 })
  } catch (e) {
    toast.add({ severity: 'error', summary: 'Failed', detail: e.message, life: 4000 })
  } finally {
    lmdeployInstalling.value = false
    // Ensure the modal doesn't stay in a "loading" state after failure.
    lmSourceDialogVisible.value = false
  }
}

// ── Lifecycle ──────────────────────────────────────────────
let unsubscribeCudaStatus = null
let unsubscribeLmdeployStatus = null
let unsubscribeTaskUpdated = null

onMounted(() => {
  enginesStore.fetchAll()
  unsubscribeCudaStatus = progressStore.subscribe('cuda_install_status', async (payload) => {
    if (payload?.status === 'completed' || payload?.status === 'failed') {
      await enginesStore.fetchCudaStatus()
    }
  })
  unsubscribeLmdeployStatus = progressStore.subscribe('lmdeploy_install_status', async (payload) => {
    if (payload?.status === 'completed' || payload?.status === 'failed') {
      if (payload?.status === 'failed') {
        const detail = payload?.message ? String(payload.message) : 'LMDeploy install from source failed';
        toast.add({ severity: 'error', summary: 'LMDeploy install failed', detail, life: 5000 })
      }
      await Promise.allSettled([
        enginesStore.fetchLmdeployStatus(),
        enginesStore.fetchLlamaVersions(),
      ])
    }
  })
  unsubscribeTaskUpdated = progressStore.subscribe('task_updated', async (task) => {
    if (task?.type !== 'build') return
    if (task.status !== 'completed' && task.status !== 'failed') return
    await Promise.allSettled([
      enginesStore.fetchLlamaVersions(),
      enginesStore.fetchSystemStatus(),
    ])
  })
})

onUnmounted(() => {
  if (unsubscribeCudaStatus) unsubscribeCudaStatus()
  if (unsubscribeLmdeployStatus) unsubscribeLmdeployStatus()
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

.cuda-toolkit-progress-slot :deep(.progress-tracker) {
  margin: 0;
  padding: 0.75rem 1.125rem 0;
  border-top: 1px solid var(--border-primary);
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

/* LMDeploy engine dialog: single title in dialog header, structured body */
.ev-section--lmdeploy {
  display: flex;
  flex-direction: column;
  gap: 0;
}

.lmdeploy-modal-body {
  display: flex;
  flex-direction: column;
  gap: 1rem;
  padding-top: 0.35rem;
}

.lmdeploy-install-panel {
  background: var(--bg-surface);
  border: 1px solid var(--border-primary);
  border-radius: var(--radius-lg);
  padding: 1rem 1.15rem;
}

.lmdeploy-install-panel__head {
  margin-bottom: 0.85rem;
}

.lmdeploy-install-panel__title {
  display: block;
  font-size: 0.7rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: var(--text-secondary);
  margin-bottom: 0.35rem;
}

.lmdeploy-install-panel__subtitle {
  display: block;
  font-size: 0.8125rem;
  line-height: 1.45;
  color: var(--text-secondary);
}

.lmdeploy-install-panel__actions {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 0.65rem;
}

@media (max-width: 520px) {
  .lmdeploy-install-panel__actions {
    grid-template-columns: 1fr;
  }
}

.lmdeploy-versions-block {
  display: flex;
  flex-direction: column;
  gap: 0.35rem;
}

.lmdeploy-versions-heading {
  font-size: 0.7rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: var(--text-secondary);
  margin: 0;
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
