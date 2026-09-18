<template>
  <section class="sglang-panel">
    <EngineBuildSettingsHint
      :key="`${engineId}-${hintRevision}`"
      :engine-key="engineId"
      @open-settings="openSettings"
    />

    <EngineCheckUpdatesCta
      :loading="checking"
      :hint="isV100 ? 'Compare the active checkout to the V100 fork main branch.' : 'Compare the active install to the latest PyPI release.'"
      @check="checkUpdates"
    />
    <EngineUpdateBanner
      :available="Boolean(updateInfo?.update_available)"
      :checked="Boolean(updateInfo)"
      :latest-version="shortVersion(updateInfo?.latest_version)"
      :current-version="shortVersion(updateInfo?.current_version)"
      :link-url="updateInfo?.url || projectUrl"
      :link-label="isV100 ? 'View commit' : 'View on PyPI'"
      :updating="installing"
      :update-tooltip="isV100 ? 'Install the latest fork source as a new environment' : 'Install the latest PyPI version as a new environment'"
      @update="installUpdate"
    />

    <EngineNote v-if="isV100">
      Specialized SGLang fork for NVIDIA V100 / SM70. Studio supplies its versioned Python
      environment and managed CUDA 12.8 toolkit; the fork installer supplies only its pinned
      Python dependencies, patched components, SM70 kernels, and smoke tests.
    </EngineNote>
    <EngineNote v-else-if="isVllm">
      Vanilla vLLM with its OpenAI-compatible server. Studio provides an isolated,
      versioned Python environment and binds source builds to the active managed CUDA toolkit.
    </EngineNote>
    <EngineNote v-else>
      Upstream SGLang with OpenAI-compatible serving. Current upstream releases require Python
      3.10+ and the supported CUDA runtime for that release.
    </EngineNote>

    <EngineInstallPanel :subtitle="installSubtitle">
      <Button
        v-if="!isV100"
        :label="isVllm ? 'Install vLLM' : 'From PyPI'"
        icon="pi pi-download"
        severity="success"
        outlined
        :loading="installing"
        @click="openPipDialog"
      />
      <Button
        :label="isV100 ? 'Build V100 fork' : 'From source'"
        icon="pi pi-code"
        severity="info"
        outlined
        :loading="installing"
        @click="openSourceDialog"
      />
      <Button
        label="Settings"
        icon="pi pi-cog"
        severity="secondary"
        outlined
        @click="openSettings"
      />
    </EngineInstallPanel>

    <EngineActiveStatus :rows="activeRows" />
    <EngineVersionsBlock>
      <VersionTable
        :versions="versions"
        :activating="activating"
        :syncing="syncing"
        :retrying="retrying"
        empty-message="No versions yet. Install one using the options above."
        @activate="activateVersion"
        @sync="syncVersion"
        @retry="retryVersion"
        @delete="confirmDelete"
      />
    </EngineVersionsBlock>

    <Dialog
      v-model:visible="settingsVisible"
      :header="`Install settings — ${label}`"
      modal
      class="dialog-width-md"
    >
      <div class="dialog-body">
        <div v-if="!isV100" class="form-field">
          <label>Default PyPI version <span class="optional">(optional)</span></label>
          <InputText v-model="form.pip_version" placeholder="Blank = latest" class="w-full" />
        </div>
        <div class="form-field">
          <label>Source repo URL</label>
          <InputText v-model="form.source_repo" :placeholder="defaultRepo" class="w-full" />
        </div>
        <div class="form-field">
          <label>Source branch</label>
          <InputText v-model="form.source_branch" placeholder="main" class="w-full" />
        </div>
      </div>
      <template #footer>
        <Button label="Cancel" severity="secondary" outlined @click="settingsVisible = false" />
        <Button label="Save settings" icon="pi pi-save" :loading="saving" @click="saveSettings" />
      </template>
    </Dialog>

    <Dialog
      v-if="!isV100"
      v-model:visible="pipVisible"
      :header="isVllm ? 'Install vLLM from PyPI' : 'Install SGLang from PyPI'"
      modal
      class="dialog-width-sm"
    >
      <div class="dialog-body">
        <div class="form-field">
          <label>Version</label>
          <InputText v-model="pipVersion" placeholder="Blank = latest" class="w-full" />
          <small v-if="isVllm">Installs the selected official vLLM release in a fresh Studio virtual environment.</small>
          <small v-else>Pre-release dependencies are allowed, matching upstream installation guidance.</small>
        </div>
      </div>
      <template #footer>
        <Button label="Cancel" severity="secondary" outlined @click="pipVisible = false" />
        <Button label="Install" icon="pi pi-download" severity="success" :loading="installing" @click="installPip" />
      </template>
    </Dialog>

    <Dialog
      v-model:visible="sourceVisible"
      :header="isV100 ? 'Build SGLang V100 fork' : `Install ${label} from source`"
      modal
      class="dialog-width-md"
    >
      <div class="dialog-body">
        <div class="form-field">
          <label>Repo URL</label>
          <InputText v-model="sourceRepo" :placeholder="defaultRepo" class="w-full" />
        </div>
        <div class="form-field">
          <label>Branch</label>
          <InputText v-model="sourceBranch" placeholder="main" class="w-full" />
        </div>
        <small v-if="isV100">
          Runs the fork's SM70 build and validation flow in a Studio-managed Python environment.
          Install CUDA 12.8 from the NVIDIA CUDA engine card first.
        </small>
      </div>
      <template #footer>
        <Button label="Cancel" severity="secondary" outlined @click="sourceVisible = false" />
        <Button :label="isV100 ? 'Build' : 'Install'" icon="pi pi-code" severity="info" :loading="installing" @click="installSource" />
      </template>
    </Dialog>
  </section>
</template>

<script setup>
import { computed, onMounted, ref } from 'vue'
import { useConfirm } from 'primevue/useconfirm'
import { useToast } from 'primevue/usetoast'
import Button from 'primevue/button'
import Dialog from 'primevue/dialog'
import InputText from 'primevue/inputtext'
import EngineActiveStatus from './EngineActiveStatus.vue'
import EngineBuildSettingsHint from './EngineBuildSettingsHint.vue'
import EngineCheckUpdatesCta from './EngineCheckUpdatesCta.vue'
import EngineInstallPanel from './EngineInstallPanel.vue'
import EngineNote from './EngineNote.vue'
import EngineUpdateBanner from './EngineUpdateBanner.vue'
import EngineVersionsBlock from './EngineVersionsBlock.vue'
import VersionTable from './VersionTable.vue'
import { useEnginesStore } from '@/stores/engines'

const props = defineProps({
  engineId: {
    type: String,
    required: true,
    validator: value => ['sglang', 'sglang_v100', 'vllm'].includes(value),
  },
})

const store = useEnginesStore()
const toast = useToast()
const confirm = useConfirm()

const isV100 = computed(() => props.engineId === 'sglang_v100')
const isVllm = computed(() => props.engineId === 'vllm')
const label = computed(() => (isV100.value ? 'SGLang V100' : isVllm.value ? 'vLLM' : 'SGLang'))
const defaultRepo = computed(() => (
  isV100.value
    ? 'https://github.com/haohervchb/sglang-V100.git'
    : isVllm.value ? 'https://github.com/vllm-project/vllm.git' : 'https://github.com/sgl-project/sglang.git'
))
const projectUrl = computed(() => (
  isV100.value
    ? 'https://github.com/haohervchb/sglang-V100'
    : isVllm.value ? 'https://pypi.org/project/vllm/' : 'https://pypi.org/project/sglang/'
))
const versions = computed(() => (
  isV100.value ? store.sglangV100Versions : isVllm.value ? store.vllmVersions : store.sglangVersions
))
const status = computed(() => (
  isV100.value ? store.sglangV100Status : isVllm.value ? store.vllmStatus : store.sglangStatus
))
const active = computed(() => versions.value.find(version => version.is_active) || null)
const installSubtitle = computed(() => (
  isV100.value
    ? 'Build a versioned SM70 environment from the maintained V100 fork.'
    : 'Add a versioned environment from PyPI or a git source checkout.'
))
const activeRows = computed(() => {
  const rows = []
  if (active.value || status.value?.venv_path) {
    const kind = active.value?.type || active.value?.install_type || status.value?.install_type || (isV100.value ? 'source' : 'pip')
    rows.push({ label: 'Install type:', tag: kind, tagSeverity: kind === 'fork' ? 'warning' : 'info' })
  }
  if (status.value?.venv_path) rows.push({ label: 'Venv:', code: status.value.venv_path })
  const cudaVersion = active.value?.cuda_version || status.value?.cuda_version
  const cudaPath = active.value?.cuda_path || status.value?.cuda_path
  if (cudaVersion || cudaPath) {
    rows.push({ label: 'Studio CUDA:', code: [cudaVersion, cudaPath].filter(Boolean).join(' · ') })
  }
  if (status.value?.source_repo) {
    rows.push({ label: 'Source:', code: `${status.value.source_repo}${status.value.source_branch ? ` (${status.value.source_branch})` : ''}` })
  }
  if (status.value?.last_error) rows.push({ label: 'Last error:', code: status.value.last_error, error: true })
  return rows
})

const checking = ref(false)
const installing = ref(false)
const saving = ref(false)
const activating = ref(null)
const syncing = ref(null)
const retrying = ref(null)
const updateInfo = ref(null)
const settingsVisible = ref(false)
const pipVisible = ref(false)
const sourceVisible = ref(false)
const pipVersion = ref('')
const sourceRepo = ref('')
const sourceBranch = ref('main')
const hintRevision = ref(0)
const form = ref({ source_repo: '', source_branch: 'main', pip_version: '' })

function detail(error) {
  return error?.response?.data?.detail || error?.message || String(error)
}

function shortVersion(value) {
  const text = String(value || '')
  return text.length > 18 ? `${text.slice(0, 17)}…` : text
}

function applySettings(settings = {}) {
  form.value = {
    source_repo: settings.source_repo || defaultRepo.value,
    source_branch: settings.source_branch || 'main',
    pip_version: settings.pip_version || '',
  }
  sourceRepo.value = form.value.source_repo
  sourceBranch.value = form.value.source_branch
  pipVersion.value = form.value.pip_version
}

async function loadSettings() {
  try {
    applySettings(await store.fetchSglangBuildSettings(props.engineId))
  } catch {
    applySettings()
  }
}

async function openSettings() {
  await loadSettings()
  hintRevision.value += 1
  settingsVisible.value = true
}

async function openPipDialog() {
  await loadSettings()
  pipVisible.value = true
}

async function openSourceDialog() {
  await loadSettings()
  sourceVisible.value = true
}

async function saveSettings() {
  saving.value = true
  try {
    applySettings(await store.saveSglangBuildSettings(props.engineId, { ...form.value }))
    settingsVisible.value = false
    toast.add({ severity: 'success', summary: 'Settings saved', detail: `${label.value} install defaults updated.`, life: 2500 })
  } catch (error) {
    toast.add({ severity: 'error', summary: 'Save failed', detail: detail(error), life: 5000 })
  } finally {
    saving.value = false
  }
}

async function checkUpdates() {
  checking.value = true
  try {
    updateInfo.value = await store.checkSglangUpdates(props.engineId)
  } catch (error) {
    toast.add({ severity: 'warn', summary: 'Could not check updates', detail: detail(error), life: 4000 })
  } finally {
    checking.value = false
  }
}

async function afterInstall(message) {
  toast.add({ severity: 'success', summary: message, detail: 'Track progress in notifications.', life: 3500 })
  pipVisible.value = false
  sourceVisible.value = false
}

async function installPip() {
  installing.value = true
  try {
    await store.saveSglangBuildSettings(props.engineId, { ...form.value, pip_version: pipVersion.value })
    await store.installSglang(props.engineId, pipVersion.value ? { version: pipVersion.value } : {})
    await afterInstall(`${label.value} install started`)
  } catch (error) {
    toast.add({ severity: 'error', summary: 'Install failed', detail: detail(error), life: 6000 })
  } finally {
    installing.value = false
  }
}

async function installSource() {
  installing.value = true
  try {
    await store.saveSglangBuildSettings(props.engineId, {
      ...form.value,
      source_repo: sourceRepo.value,
      source_branch: sourceBranch.value,
    })
    await store.installSglangFromSource(props.engineId, {
      repo_url: sourceRepo.value,
      branch: sourceBranch.value,
    })
    await afterInstall(`${label.value} source install started`)
  } catch (error) {
    toast.add({ severity: 'error', summary: 'Install failed', detail: detail(error), life: 6000 })
  } finally {
    installing.value = false
  }
}

async function installUpdate() {
  await loadSettings()
  if (isV100.value) {
    return installSource()
  }
  pipVersion.value = updateInfo.value?.latest_version || ''
  return installPip()
}

async function activateVersion(version) {
  activating.value = version.id
  try {
    await store.activateVersion(version.id)
    toast.add({ severity: 'success', summary: 'Version activated', detail: version.version, life: 2500 })
  } catch (error) {
    toast.add({ severity: 'error', summary: 'Activation failed', detail: detail(error), life: 5000 })
  } finally {
    activating.value = null
  }
}

async function syncVersion(version) {
  syncing.value = version.id
  try {
    await store.syncVersion(version.id)
    toast.add({ severity: 'success', summary: 'Source sync started', detail: 'Track progress in notifications.', life: 3000 })
  } catch (error) {
    toast.add({ severity: 'error', summary: 'Sync failed', detail: detail(error), life: 5000 })
  } finally {
    syncing.value = null
  }
}

async function retryVersion(version) {
  retrying.value = version.id
  try {
    await store.retryVersion(version.id)
    toast.add({ severity: 'success', summary: 'Retry started', detail: 'Track progress in notifications.', life: 3000 })
  } catch (error) {
    toast.add({ severity: 'error', summary: 'Retry failed', detail: detail(error), life: 5000 })
  } finally {
    retrying.value = null
  }
}

function confirmDelete(version) {
  confirm.require({
    header: 'Delete engine version',
    message: `Delete ${label.value} ${version.version}?`,
    icon: 'pi pi-exclamation-triangle',
    acceptClass: 'p-button-danger',
    accept: async () => {
      try {
        await store.deleteVersion(version.id)
        toast.add({ severity: 'info', summary: 'Version deleted', detail: version.version, life: 2500 })
      } catch (error) {
        toast.add({ severity: 'error', summary: 'Delete failed', detail: detail(error), life: 5000 })
      }
    },
  })
}

async function refresh() {
  await Promise.allSettled([
    store.fetchLlamaVersions(),
    store.fetchSglangStatus(props.engineId),
    checkUpdates(),
  ])
}

defineExpose({ openSettings, refresh })
onMounted(refresh)
</script>

<style scoped>
.sglang-panel {
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
  padding: 1rem 1.25rem 1.25rem;
}
</style>
