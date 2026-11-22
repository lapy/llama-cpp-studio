<template>
  <section class="hardware-cuda" aria-labelledby="cuda-toolkit-heading">
    <div class="hardware-subhead">
      <div class="hardware-subhead__copy">
        <h3 id="cuda-toolkit-heading" class="page-subheading">CUDA Toolkit</h3>
        <p class="hardware-subhead__status">
          <template v-if="cuda.installed">CUDA {{ cuda.version || '?' }}</template>
          <template v-else>Not installed</template>
          <span class="hardware-subhead__hint"> · {{ cudaHint }}</span>
        </p>
      </div>
      <div class="hardware-subhead__actions">
        <Button
          icon="pi pi-refresh"
          text
          severity="secondary"
          size="small"
          aria-label="Reload CUDA status"
          v-tooltip.top="'Reload CUDA status'"
          @click="reloadStatus"
        />
        <Button
          label="Install CUDA"
          icon="pi pi-download"
          severity="success"
          outlined
          size="small"
          @click="cudaInstallDialogVisible = true"
        />
      </div>
    </div>

    <div v-if="cuda.installed || cuda.installed_versions?.length" class="hardware-cuda__details">
      <div v-if="cuda.installed" class="hardware-detail">
        <span class="hardware-detail__label">CUDA path</span>
        <code class="hardware-detail__value">{{ cuda.cuda_path || 'unknown' }}</code>
      </div>

      <ul v-if="cuda.installed_versions?.length" class="hardware-version-list">
        <li
          v-for="installed in cuda.installed_versions"
          :key="installed.version"
          class="ev-version-row"
        >
          <span class="version-identity">
            <code class="version-name">CUDA {{ installed.version }}</code>
            <span v-if="installed.size_bytes != null" class="version-size">
              {{ formatBytesIEC(installed.size_bytes) }}
            </span>
          </span>
          <Tag v-if="installed.is_current" value="Active" severity="success" />
          <Button
            v-else
            label="Activate"
            icon="pi pi-play"
            size="small"
            severity="success"
            outlined
            :loading="cudaActivating === installed.version"
            :disabled="Boolean(cudaActivating)"
            @click="activateCuda(installed.version)"
          />
          <Button
            icon="pi pi-trash"
            text
            severity="danger"
            size="small"
            :aria-label="`Uninstall CUDA ${installed.version}`"
            @click="confirmUninstallCuda(installed.version)"
          />
          <p v-if="cudaWarningFor(installed.version)" class="cuda-compat-warning" role="status">
            <i class="pi pi-exclamation-triangle" aria-hidden="true" />
            <span>{{ cudaWarningFor(installed.version)?.message }}</span>
          </p>
        </li>
      </ul>
    </div>

    <div v-if="installerCacheBytes > 0" class="hardware-cuda__cache">
      <span class="cuda-toolkit-cache__label">
        Installer cache: {{ formatBytesIEC(installerCacheBytes) }} ({{
          cuda.installer_cache?.files ?? 0
        }}
        files)
      </span>
      <Button
        label="Clean installer cache"
        icon="pi pi-trash"
        severity="danger"
        outlined
        size="small"
        :loading="cudaCleaning"
        @click="confirmCleanupCudaInstallers"
      />
    </div>
  </section>

  <Dialog
    v-model:visible="cudaInstallDialogVisible"
    header="Install CUDA Toolkit"
    modal
    class="dialog-width-xs"
  >
    <div class="dialog-stack">
      <div class="page-form page-form--single">
        <div class="page-form__field">
          <label class="page-form__label" for="cuda-install-version">Version</label>
          <Select
            v-model="cudaInstallVersion"
            inputId="cuda-install-version"
            :options="cudaVersionOptions"
            placeholder="Select version…"
            class="w-full"
            :aria-describedby="pickerCudaWarnings.length ? 'cuda-install-warnings' : undefined"
          />
          <div
            v-if="pickerCudaWarnings.length"
            id="cuda-install-warnings"
            class="cuda-compat-warnings"
          >
            <p
              v-for="warning in pickerCudaWarnings"
              :key="warning.version"
              class="cuda-compat-warning"
              role="status"
            >
              <i class="pi pi-exclamation-triangle" aria-hidden="true" />
              <span>{{ warning.message }}</span>
            </p>
          </div>
          <p v-else class="page-form__hint">
            Installed builds are activated from the CUDA Toolkit list.
          </p>
        </div>
      </div>
    </div>
    <template #footer>
      <div class="dialog-actions">
        <Button
          label="Cancel"
          severity="secondary"
          outlined
          @click="cudaInstallDialogVisible = false"
        />
        <Button
          label="Install"
          icon="pi pi-download"
          severity="success"
          :disabled="!cudaInstallVersion"
          :loading="cudaInstalling"
          @click="installCuda"
        />
      </div>
    </template>
  </Dialog>
</template>

<script setup>
import { computed, ref } from 'vue'
import Button from 'primevue/button'
import Dialog from 'primevue/dialog'
import Select from 'primevue/select'
import Tag from 'primevue/tag'
import { useToast } from 'primevue/usetoast'
import { useConfirm } from 'primevue/useconfirm'
import { requireSingleConfirmation } from '@/composables/singleConfirm'
import { watchDialogFocus } from '@/composables/useFocusReturn'
import { useEnginesStore } from '@/stores/engines'
import { cudaCompatibilityWarnings } from '@/utils/cudaCompatibility'
import { formatBytesIEC } from '@/utils/formatting'

const enginesStore = useEnginesStore()
const toast = useToast()
const confirm = useConfirm()

const cuda = computed(() => enginesStore.cudaStatus || {})
const gpus = computed(() => enginesStore.gpuInfo?.gpus ?? [])
const installerCacheBytes = computed(() => Number(cuda.value.installer_cache?.bytes) || 0)
const cudaVersionOptions = computed(() => cuda.value.available_versions || [])
const cudaInstallVersion = ref(null)

const cudaHint = computed(() => {
  if (cuda.value.installed_versions?.length) {
    const count = cuda.value.installed_versions.length
    return `${count} version${count === 1 ? '' : 's'} detected`
  }
  if (cuda.value.cuda_path) return cuda.value.cuda_path
  return 'Build support and toolkit management'
})

const cudaInstalling = ref(false)
const cudaActivating = ref(null)
const cudaCleaning = ref(false)
const cudaInstallDialogVisible = ref(false)
watchDialogFocus(cudaInstallDialogVisible)

function reloadStatus() {
  void enginesStore.fetchCudaStatus()
}

// Guidance only: an incompatible version can still be installed or activated.
const cudaVersionWarnings = computed(() =>
  cudaCompatibilityWarnings(
    [
      ...(cuda.value.available_versions || []),
      ...(cuda.value.installed_versions || []).map((entry) => entry.version),
    ],
    gpus.value,
  ),
)

function cudaWarningFor(version) {
  return cudaVersionWarnings.value.find((warning) => warning.version === version) || null
}

const pickerCudaWarnings = computed(() => {
  const selected = cudaInstallVersion.value ? cudaWarningFor(cudaInstallVersion.value) : null
  if (selected) return [selected]
  return (cuda.value.available_versions || [])
    .map((version) => cudaWarningFor(version))
    .filter(Boolean)
})

/** Typed CUDA failures answer `detail = {code, message}`; legacy ones a string. */
function cudaErrorDetail(error) {
  const detail = error?.response?.data?.detail
  if (detail && typeof detail === 'object' && !Array.isArray(detail)) {
    return detail.message || detail.code || error.message
  }
  return detail || error.message
}

async function installCuda() {
  cudaInstalling.value = true
  try {
    await enginesStore.installCuda({ version: cudaInstallVersion.value })
    cudaInstallDialogVisible.value = false
    toast.add({
      severity: 'success',
      summary: 'CUDA install started',
      detail: 'Track progress in notifications',
      life: 3000,
    })
    await enginesStore.fetchCudaStatus()
  } catch (error) {
    toast.add({ severity: 'error', summary: 'Failed', detail: cudaErrorDetail(error), life: 4000 })
  } finally {
    cudaInstalling.value = false
  }
}

async function activateCuda(version) {
  cudaActivating.value = version
  try {
    await enginesStore.activateCuda({ version })
    toast.add({ severity: 'success', summary: `CUDA ${version} activated`, life: 3000 })
  } catch (error) {
    toast.add({ severity: 'error', summary: 'Failed', detail: cudaErrorDetail(error), life: 4000 })
  } finally {
    cudaActivating.value = null
  }
}

function confirmUninstallCuda(version) {
  requireSingleConfirmation(confirm, {
    message: `Uninstall CUDA ${version}?`,
    header: 'Confirm Uninstall',
    icon: 'pi pi-exclamation-triangle',
    acceptClass: 'p-button-danger',
    accept: async () => {
      try {
        await enginesStore.uninstallCuda({ version })
        toast.add({ severity: 'info', summary: `CUDA ${version} uninstalled`, life: 3000 })
      } catch (error) {
        toast.add({
          severity: 'error',
          summary: 'Failed',
          detail: cudaErrorDetail(error),
          life: 4000,
        })
      }
    },
  })
}

function confirmCleanupCudaInstallers() {
  requireSingleConfirmation(confirm, {
    message: 'Delete the downloaded CUDA installer files? Installed toolkits are kept.',
    header: 'Clean Installer Cache',
    icon: 'pi pi-exclamation-triangle',
    acceptClass: 'p-button-danger',
    accept: async () => {
      cudaCleaning.value = true
      try {
        const result = await enginesStore.cleanupCudaInstallers()
        toast.add({
          severity: 'success',
          summary: 'Installer cache cleaned',
          detail: `${formatBytesIEC(result?.freed_bytes ?? 0)} freed`,
          life: 3000,
        })
      } catch (error) {
        toast.add({
          severity: 'error',
          summary: 'Failed',
          detail: cudaErrorDetail(error),
          life: 4000,
        })
      } finally {
        cudaCleaning.value = false
      }
    },
  })
}
</script>

<style scoped>
.hardware-cuda {
  display: grid;
  gap: 0.75rem;
  min-width: 0;
}

.hardware-subhead {
  display: flex;
  flex-wrap: wrap;
  align-items: flex-start;
  justify-content: space-between;
  gap: 0.75rem;
}

.hardware-subhead__copy {
  display: grid;
  gap: 0.25rem;
  min-width: 0;
}

.hardware-subhead__status {
  margin: 0;
  color: var(--text-secondary);
  font-size: var(--text-sm);
  overflow-wrap: anywhere;
}

.hardware-subhead__hint {
  color: var(--text-secondary);
  font-size: var(--text-xs);
}

.hardware-subhead__actions {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 0.5rem;
}

.hardware-cuda__details {
  display: grid;
  gap: 0.75rem;
  min-width: 0;
}

.hardware-detail {
  display: grid;
  gap: 0.35rem;
  min-width: 0;
}

.hardware-detail__label {
  color: var(--text-secondary);
  font-size: var(--text-xs);
}

.hardware-detail__value {
  overflow-wrap: anywhere;
}

.hardware-version-list {
  margin: 0;
  padding: 0;
  list-style: none;
}

.ev-version-row {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  justify-content: space-between;
  gap: 0.75rem;
  padding: 0.6rem 0;
  border-top: 1px solid var(--border-primary);
}

.version-identity {
  display: inline-flex;
  align-items: baseline;
  gap: 0.5rem;
  min-width: 0;
}

.version-size {
  color: var(--text-secondary);
  font-size: var(--text-xs);
  font-variant-numeric: tabular-nums;
}

.cuda-compat-warnings {
  display: grid;
  gap: 0.35rem;
  margin-top: 0.5rem;
}

.cuda-compat-warning {
  display: flex;
  align-items: flex-start;
  gap: 0.4rem;
  margin: 0;
  color: var(--status-warning);
  font-size: var(--text-xs);
  line-height: 1.4;
}

.ev-version-row .cuda-compat-warning {
  flex-basis: 100%;
}

.cuda-compat-warning .pi {
  margin-top: 0.15rem;
}

.hardware-cuda__cache {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  justify-content: space-between;
  gap: 0.75rem;
  padding-top: 0.75rem;
  border-top: 1px solid var(--border-primary);
}

.cuda-toolkit-cache__label {
  color: var(--text-secondary);
  font-size: var(--text-xs);
  font-variant-numeric: tabular-nums;
  overflow-wrap: anywhere;
}
</style>
