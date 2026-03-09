<template>
  <div class="engines-view">

    <!-- ── System Info ─────────────────────────────────────── -->
    <section class="ev-section">
      <div class="ev-section-header" @click="systemExpanded = !systemExpanded">
        <div class="ev-section-title">
          <i class="pi pi-desktop" />
          <h2>System</h2>
        </div>
        <div class="ev-section-actions">
          <Button icon="pi pi-refresh" text severity="secondary" size="small"
            :loading="enginesStore.loading" @click.stop="enginesStore.fetchSystemStatus()" />
          <i :class="['pi', systemExpanded ? 'pi-chevron-up' : 'pi-chevron-down']" />
        </div>
      </div>
      <Transition name="ev-collapse">
        <div v-if="systemExpanded" class="ev-section-body">
          <div class="metrics-grid">
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
                  {{ formatBytesIEC(sys.memory?.used) }} / {{ formatBytesIEC(sys.memory?.total) }} ({{ memPercent }}%)
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
            <div class="metric-card metric-card--actionable">
              <i class="pi pi-bolt metric-icon" />
              <div class="metric-data">
                <div class="metric-label">CUDA Toolkit</div>
                <div class="metric-value">
                  <template v-if="cuda.installed">CUDA {{ cuda.version || '?' }}</template>
                  <template v-else>Not Installed</template>
                </div>
                <div class="metric-subvalue">
                  <template v-if="cuda.installed_versions?.length">
                    {{ cuda.installed_versions.length }} version{{ cuda.installed_versions.length === 1 ? '' : 's' }} detected
                  </template>
                  <template v-else-if="cuda.cuda_path">
                    {{ cuda.cuda_path }}
                  </template>
                  <template v-else>
                    Build support and toolkit management
                  </template>
                </div>
                <div class="metric-actions">
                  <Button icon="pi pi-refresh" text severity="secondary" size="small"
                    v-tooltip.top="'Reload CUDA status'"
                    @click.stop="enginesStore.fetchCudaStatus()" />
                  <Button label="Install" icon="pi pi-download" severity="success" outlined size="small"
                    @click.stop="cudaInstallDialogVisible = true" />
                </div>
              </div>
            </div>
            <div v-for="(gpuItem, idx) in gpus" :key="gpuItem.index ?? gpuItem.uuid ?? gpuItem.name ?? idx" class="metric-card">
              <i class="pi pi-bolt metric-icon" />
              <div class="metric-data">
                <div class="metric-label">GPU — {{ gpuItem.name }}</div>
                <div class="metric-value">
                  {{ formatBytesIEC(gpuItem.memory_used_mb * 1048576) }} /
                  {{ formatBytesIEC(gpuItem.memory_total_mb * 1048576) }} VRAM
                </div>
                <ProgressBar :value="gpuPercent(gpuItem)" :showValue="false" class="metric-bar" />
              </div>
            </div>
          </div>
          <div class="system-subpanel">
            <ProgressTracker type="install" metadata-key="target" metadata-value="cuda" />

            <div v-if="cuda.installed" class="status-detail">
              <span class="detail-label">CUDA Path:</span>
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
      </Transition>
    </section>

    <!-- ── Engines Overview ───────────────────────────────── -->
    <section class="ev-section">
      <div class="ev-section-header" @click="enginesExpanded = !enginesExpanded">
        <div class="ev-section-title">
          <i class="pi pi-server" />
          <h2>Engines</h2>
        </div>
        <div class="ev-section-actions">
          <Button icon="pi pi-refresh" text severity="secondary" size="small"
            @click.stop="refreshEnginesOverview" />
          <i :class="['pi', enginesExpanded ? 'pi-chevron-up' : 'pi-chevron-down']" />
        </div>
      </div>
      <Transition name="ev-collapse">
        <div v-if="enginesExpanded" class="ev-section-body">
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
                <Tag v-if="activeLlamaCpp" :value="activeLlamaCpp.version" severity="success" />
                <Tag v-else value="No Active" severity="warning" />
              </div>
              <div class="engine-card-body">
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
                <Tag v-if="activeIkLlama" :value="activeIkLlama.version" severity="success" />
                <Tag v-else value="No Active" severity="warning" />
              </div>
              <div class="engine-card-body">
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
                    <div class="engine-card-meta">{{ lm.installed ? 'Installed' : 'Not installed' }}</div>
                  </div>
                </div>
                <Tag v-if="lm.installed" :value="`v${lm.version || '?'}`" severity="success" />
                <Tag v-else value="Not Installed" severity="secondary" />
              </div>
              <div class="engine-card-body">
                <div v-if="lmdeployUpdateInfo?.update_available" class="engine-card-status engine-card-status--warning">
                  Update available: v{{ lmdeployUpdateInfo.latest_version }}
                </div>
                <div v-else class="engine-card-status">
                  Manage installs, updates, and removal
                </div>
              </div>
            </button>
          </div>
        </div>
      </Transition>
    </section>

    <!-- ── CUDA Install Dialog ────────────────────────────── -->
    <Dialog v-model:visible="cudaInstallDialogVisible" header="Install CUDA Toolkit" modal :style="{ width: '400px' }">
      <div class="dialog-body">
        <div class="form-field">
          <label>Version</label>
          <Dropdown v-model="cudaInstallVersion" :options="cudaVersionOptions"
            placeholder="Select version…" style="width:100%" />
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
      :header="engineDialogTitle"
      modal maximizable
      :style="{ width: '960px' }">
      <section v-if="selectedEngine === 'llama_cpp'" class="ev-section ev-section--modal">
        <div class="ev-section-header">
          <div class="ev-section-title">
            <span class="engine-mark engine-mark--llama" aria-hidden="true">L</span>
            <h2>llama.cpp</h2>
            <Tag v-if="activeLlamaCpp" :value="activeLlamaCpp.version" severity="success" />
            <Tag v-else-if="enginesStore.llamaVersions.length" value="No Active" severity="warning" />
          </div>
          <div class="ev-section-actions">
            <Button icon="pi pi-sliders-h" text severity="info" size="small"
              v-tooltip.top="'Build settings'"
              @click="openBuildDialog('llama_cpp')" />
            <Button icon="pi pi-arrow-up-right" text severity="info" size="small"
              v-tooltip.top="'Check for updates'"
              :loading="checkingLlamaCpp" @click="checkLlamaCppUpdates" />
            <Button icon="pi pi-refresh" text severity="secondary" size="small"
              v-tooltip.top="'Reload versions'"
              @click="enginesStore.fetchLlamaVersions()" />
          </div>
        </div>
        <div class="ev-section-body">
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

          <ProgressTracker type="build" metadata-key="repository_source" metadata-value="llama.cpp" />

          <VersionTable
            :versions="enginesStore.llamaVersions"
            :activating="activating"
            @activate="activateVersion"
            @delete="confirmDeleteVersion"
          />
        </div>
      </section>

      <section v-else-if="selectedEngine === 'ik_llama'" class="ev-section ev-section--modal">
        <div class="ev-section-header">
          <div class="ev-section-title">
            <span class="engine-mark engine-mark--ik" aria-hidden="true">IK</span>
            <h2>ik_llama.cpp</h2>
            <Tag v-if="activeIkLlama" :value="activeIkLlama.version" severity="success" />
            <Tag v-else-if="enginesStore.ikLlamaVersions.length" value="No Active" severity="warning" />
          </div>
          <div class="ev-section-actions">
            <Button icon="pi pi-sliders-h" text severity="info" size="small"
              v-tooltip.top="'Build settings'"
              @click="openBuildDialog('ik_llama')" />
            <Button icon="pi pi-arrow-up-right" text severity="info" size="small"
              v-tooltip.top="'Check for updates'"
              :loading="checkingIkLlama" @click="checkIkLlamaUpdates" />
            <Button icon="pi pi-refresh" text severity="secondary" size="small"
              v-tooltip.top="'Reload versions'"
              @click="enginesStore.fetchLlamaVersions()" />
          </div>
        </div>
        <div class="ev-section-body">
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

          <ProgressTracker type="build" metadata-key="repository_source" metadata-value="ik_llama.cpp" />

          <VersionTable
            :versions="enginesStore.ikLlamaVersions"
            :activating="activating"
            @activate="activateVersion"
            @delete="confirmDeleteVersion"
          />
        </div>
      </section>

      <section v-else-if="selectedEngine === 'lmdeploy'" class="ev-section ev-section--modal">
        <div class="ev-section-header">
          <div class="ev-section-title">
            <i class="pi pi-server" />
            <h2>LMDeploy</h2>
            <Tag v-if="lm.installed" :value="`v${lm.version || '?'}`" severity="success" />
            <Tag v-else value="Not Installed" severity="secondary" />
          </div>
          <div class="ev-section-actions">
            <Button icon="pi pi-arrow-up-right" text severity="info" size="small"
              v-tooltip.top="'Check for updates'"
              :loading="checkingLmdeploy" @click="checkLmdeployUpdates" />
            <Button icon="pi pi-refresh" text severity="secondary" size="small"
              v-tooltip.top="'Reload LMDeploy status'"
              @click="enginesStore.fetchLmdeployStatus()" />
          </div>
        </div>
        <div class="ev-section-body">
          <div v-if="lmdeployUpdateInfo?.update_available" class="update-banner">
            <i class="pi pi-arrow-up-right" />
            Update available: <strong>v{{ lmdeployUpdateInfo.latest_version }}</strong>
            <a href="https://pypi.org/project/lmdeploy/" target="_blank" class="update-link">View on PyPI</a>
          </div>
          <div v-else-if="lmdeployUpdateInfo" class="update-current">
            <i class="pi pi-check" /> Up to date (v{{ lmdeployUpdateInfo.current_version || 'none' }})
          </div>

          <ProgressTracker type="install" metadata-key="target" metadata-value="lmdeploy" />

          <div v-if="lm.installed" class="status-detail">
            <span class="detail-label">Install type:</span>
            <Tag :value="lm.install_type || 'pip'" severity="info" />
            <template v-if="lm.venv_path">
              <span class="detail-label ml">Venv:</span>
              <code>{{ lm.venv_path }}</code>
            </template>
          </div>
          <div v-if="lm.source_repo" class="status-detail">
            <span class="detail-label">Source:</span>
            <code>{{ lm.source_repo }} ({{ lm.source_branch }})</code>
          </div>

          <div class="ev-actions">
            <Button label="Install from pip" icon="pi pi-download" severity="success" outlined size="small"
              :disabled="lm.installed" @click="lmPipDialogVisible = true" />
            <Button label="Install from Source" icon="pi pi-code" severity="info" outlined size="small"
              :disabled="lm.installed" @click="lmSourceDialogVisible = true" />
          </div>

          <div v-if="lm.installed" class="ev-actions" style="margin-top:1rem; border-top:1px solid var(--border-primary); padding-top:1rem">
            <Button label="Remove LMDeploy" icon="pi pi-trash" severity="danger" outlined
              :loading="lmdeployRemoving" @click="confirmRemoveLmdeploy" />
          </div>
        </div>
      </section>
    </Dialog>

    <!-- ── Build Settings Dialog ─────────────────────────── -->
    <Dialog v-model:visible="buildDialogVisible"
      :header="`Build settings — ${buildTarget === 'ik_llama' ? 'ik_llama.cpp' : 'llama.cpp'}`"
      modal :style="{ width: '620px' }" class="build-settings-dialog">
      <div class="dialog-body">
        <div class="form-field">
          <label>Ref (tag / branch / commit)</label>
          <InputText v-model="buildForm.commitSha"
            :placeholder="buildTarget === 'ik_llama' ? 'main' : 'master'"
            style="width:100%" />
          <small>Use a release tag, branch, or commit. Latest detected release is used by default when available.</small>
        </div>
        <div class="form-field">
          <label>Build Name Suffix <span class="optional">(optional)</span></label>
          <InputText v-model="buildForm.versionSuffix" placeholder="e.g. my-build" style="width:100%" />
          <small>Appended to version name. Defaults to timestamp if empty.</small>
        </div>
        <div class="form-field">
          <label>Build type</label>
          <Dropdown v-model="buildForm.buildConfig.build_type"
            :options="buildTypeOptions"
            optionLabel="label"
            optionValue="value"
            placeholder="Release"
            style="width:100%" />
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
            placeholder="e.g. 86;89 (blank = auto)" style="width:100%" />
        </div>
        <div class="form-field">
          <label>Custom CMake args <span class="optional">(optional)</span></label>
          <InputText v-model="buildForm.buildConfig.custom_cmake_args"
            placeholder="e.g. -DFOO=ON -DBAR=OFF" style="width:100%" />
        </div>
        <div class="form-field">
          <label>CFLAGS / CXXFLAGS <span class="optional">(optional)</span></label>
          <div class="flags-row">
            <InputText v-model="buildForm.buildConfig.cflags" placeholder="CFLAGS" style="flex:1" />
            <InputText v-model="buildForm.buildConfig.cxxflags" placeholder="CXXFLAGS" style="flex:1" />
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

    <!-- ── LMDeploy Install from pip Dialog ───────────────── -->
    <Dialog v-model:visible="lmPipDialogVisible" header="Install LMDeploy from pip" modal :style="{ width: '420px' }">
      <div class="dialog-body">
        <div class="form-field">
          <label>Version</label>
          <InputText v-model="lmdeployPipVersion" placeholder="Blank = latest" style="width:100%" />
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
    <Dialog v-model:visible="lmSourceDialogVisible" header="Install LMDeploy from Source" modal :style="{ width: '480px' }">
      <div class="dialog-body">
        <div class="form-field">
          <label>Repo URL</label>
          <InputText v-model="lmSourceRepo" placeholder="https://github.com/InternLM/lmdeploy.git" style="width:100%" />
        </div>
        <div class="form-field">
          <label>Branch</label>
          <InputText v-model="lmSourceBranch" placeholder="main" style="width:100%" />
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
import axios from 'axios'
import { useConfirm } from 'primevue/useconfirm'
import { useToast } from 'primevue/usetoast'
import Button from 'primevue/button'
import Tag from 'primevue/tag'
import ProgressBar from 'primevue/progressbar'
import ProgressSpinner from 'primevue/progressspinner'
import Dialog from 'primevue/dialog'
import Dropdown from 'primevue/dropdown'
import InputText from 'primevue/inputtext'
import InputSwitch from 'primevue/inputswitch'
import ProgressTracker from '@/components/common/ProgressTracker.vue'
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

const engineDialogTitle = computed(() => {
  if (selectedEngine.value === 'ik_llama') return 'ik_llama.cpp'
  if (selectedEngine.value === 'lmdeploy') return 'LMDeploy'
  return 'llama.cpp'
})

function openEngineModal(engineKey) {
  selectedEngine.value = engineKey
  engineDialogVisible.value = true
  if (engineKey === 'llama_cpp') {
    checkLlamaCppUpdates()
  } else if (engineKey === 'ik_llama') {
    checkIkLlamaUpdates()
  } else if (engineKey === 'lmdeploy') {
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

const memPercent = computed(() => {
  const m = sys.value.memory
  const used = m?.used ?? 0
  const total = m?.total ?? 0
  return total > 0 ? Math.round((used / total) * 100) : 0
})

const diskPercent = computed(() => {
  const d = sys.value.disk
  const used = d?.used ?? 0
  const total = d?.total ?? 0
  return total > 0 ? Math.round((used / total) * 100) : 0
})

function gpuPercent(g) {
  const used = g?.memory_used_mb ?? 0
  const total = g?.memory_total_mb ?? 0
  return total > 0 ? Math.round((used / total) * 100) : 0
}

// ── Active versions ────────────────────────────────────────
const activeLlamaCpp = computed(() => enginesStore.llamaVersions.find(v => v.is_active) ?? null)
const activeIkLlama = computed(() => enginesStore.ikLlamaVersions.find(v => v.is_active) ?? null)

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
  if (typeof enginesStore.fetchBuildSettings === 'function') {
    return await enginesStore.fetchBuildSettings(engineId)
  }
  const { data } = await axios.get('/api/llama-versions/build-settings', {
    params: { engine: engineId },
  })
  return data
}

async function saveEngineBuildSettings(engineId, settings) {
  if (typeof enginesStore.saveBuildSettings === 'function') {
    return await enginesStore.saveBuildSettings(engineId, settings)
  }
  const { data } = await axios.put('/api/llama-versions/build-settings', settings, {
    params: { engine: engineId },
  })
  return data
}

async function updateEngineWithSavedSettings(engineId) {
  if (typeof enginesStore.updateEngine === 'function') {
    return await enginesStore.updateEngine(engineId)
  }
  const { data } = await axios.post('/api/llama-versions/update', {
    engine: engineId,
  })
  return data
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
const lmdeployRemoving = ref(false)
const checkingLmdeploy = ref(false)
const lmdeployUpdateInfo = ref(null)
const lmPipDialogVisible = ref(false)
const lmSourceDialogVisible = ref(false)

async function checkLmdeployUpdates() {
  checkingLmdeploy.value = true
  try {
    const raw = await enginesStore.checkLmdeployUpdates()
    const current = lm.value?.version || null
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
    lmdeployInstalling.value = false
  }
}

async function installLmdeploySource() {
  lmdeployInstalling.value = true
  try {
    await enginesStore.installLmdeployFromSource({
      repo_url: lmSourceRepo.value,
      branch: lmSourceBranch.value,
    })
    lmSourceDialogVisible.value = false
    toast.add({ severity: 'success', summary: 'Install from source started', detail: 'Track progress below', life: 3000 })
  } catch (e) {
    toast.add({ severity: 'error', summary: 'Failed', detail: e.message, life: 4000 })
  } finally {
    lmdeployInstalling.value = false
  }
}

function confirmRemoveLmdeploy() {
  confirm.require({
    message: 'Remove LMDeploy from the venv?',
    header: 'Confirm Remove',
    icon: 'pi pi-exclamation-triangle',
    acceptClass: 'p-button-danger',
    accept: async () => {
      lmdeployRemoving.value = true
      try {
        await enginesStore.removeLmdeploy()
        toast.add({ severity: 'info', summary: 'LMDeploy removed', life: 3000 })
      } catch (e) {
        toast.add({ severity: 'error', summary: 'Failed', detail: e.message, life: 4000 })
      } finally {
        lmdeployRemoving.value = false
      }
    },
  })
}

// ── Lifecycle ──────────────────────────────────────────────
let unsubscribeCudaStatus = null
let unsubscribeLmdeployStatus = null

onMounted(() => {
  enginesStore.fetchAll()
  unsubscribeCudaStatus = progressStore.subscribe('cuda_install_status', async (payload) => {
    if (payload?.status === 'completed' || payload?.status === 'failed') {
      await enginesStore.fetchCudaStatus()
    }
  })
  unsubscribeLmdeployStatus = progressStore.subscribe('lmdeploy_install_status', async (payload) => {
    if (payload?.status === 'completed' || payload?.status === 'failed') {
      await enginesStore.fetchLmdeployStatus()
    }
  })
})

onUnmounted(() => {
  if (unsubscribeCudaStatus) unsubscribeCudaStatus()
  if (unsubscribeLmdeployStatus) unsubscribeLmdeployStatus()
})
</script>

<style scoped>
.engines-view {
  max-width: 960px;
  margin: 0 auto;
  padding: var(--spacing-lg);
  display: flex;
  flex-direction: column;
  gap: var(--spacing-lg);
}

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
  padding: 0.75rem 1.25rem;
  background: var(--bg-surface);
  border-bottom: 1px solid var(--border-primary);
  cursor: default;
  user-select: none;
}

.ev-section-title {
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.ev-section-title h2 {
  font-size: 1rem;
  font-weight: 600;
  margin: 0;
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

.ev-section-actions {
  display: flex;
  align-items: center;
  gap: 0.25rem;
}

.ev-section-body {
  padding: 1.25rem;
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

.engine-card-head {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 0.75rem;
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
