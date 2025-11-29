<template>
  <div class="llama-manager">
    <BaseCard>
      <template #header>
        <div class="card-header">
          <div class="header-content">
            <h2 class="card-title">llama.cpp Version Manager</h2>
            <div class="active-version-header">
              <Tag 
                v-if="activeVersion"
                :value="`Active: ${activeVersion.version}`" 
                severity="success"
                class="active-header-badge"
              />
              <Tag 
                v-else-if="systemStore.llamaVersions.length > 0"
                value="No Active Version" 
                severity="warning"
                class="active-header-badge"
              />
            </div>
          </div>
          <div class="header-actions">
            <Button 
              icon="pi pi-refresh" 
              @click="refreshVersions"
              :loading="refreshingVersions"
              severity="secondary"
              text
              v-tooltip.top="'Refresh installed versions'"
            />
            <Button 
              label="Check for Updates"
              icon="pi pi-search" 
              @click="checkUpdates"
              :loading="checkingUpdates"
              severity="info"
              outlined
              v-tooltip.top="'Check for available updates'"
            />
          </div>
        </div>
      </template>

      <!-- Build Progress -->
      <BuildProgress />

      <!-- Update Information -->
      <UpdateInfo
        v-if="updateInfo"
        :update-info="updateInfo"
        :installing-release="installingRelease"
        :building-source="buildingSource"
        @install-release="openReleaseDialog"
        @build-source="showBuildDialog"
      />

      <!-- Installed Versions -->
      <VersionList
        :versions="systemStore.llamaVersions"
        :activating="activating"
        @activate="activateVersion"
        @delete="confirmDeleteVersion"
      />
    </BaseCard>

    <!-- Release Install Dialog -->
    <ReleaseDialog
      v-model:visible="releaseDialogVisible"
      :release-tag="selectedReleaseTag"
      @installed="handleReleaseInstalled"
      @hide="handleReleaseDialogHide"
    />

    <!-- Build from Source Dialog -->
    <BuildDialog
      v-model:visible="buildDialogVisible"
      :build-capabilities="buildCapabilities"
      :cuda-status="cudaStatus"
      @build="handleBuild"
      @show-cuda-install="showCudaInstallDialog"
    />

    <!-- CUDA Installation Dialog -->
    <CudaInstallDialog
      v-model:visible="cudaInstallDialogVisible"
      :cuda-status="cudaStatus"
      :cuda-install-progress="cudaInstallProgress"
      :cuda-install-logs="cudaInstallLogs"
      @install="handleCudaInstall"
    />
  </div>
</template>

<script setup>
// Vue
import { ref, onMounted, computed } from 'vue'

// PrimeVue
import Button from 'primevue/button'
import Tag from 'primevue/tag'
import { useConfirm } from 'primevue/useconfirm'

// Third-party
import { toast } from 'vue3-toastify'

// Stores
import { useSystemStore } from '@/stores/system'
import { useWebSocketStore } from '@/stores/websocket'

// Components
import BuildProgress from '@/components/BuildProgress.vue'
import BaseCard from '@/components/common/BaseCard.vue'
import UpdateInfo from '@/components/system/LlamaCppManager/UpdateInfo.vue'
import VersionList from '@/components/system/LlamaCppManager/VersionList.vue'
import ReleaseDialog from '@/components/system/LlamaCppManager/ReleaseDialog.vue'
import BuildDialog from '@/components/system/LlamaCppManager/BuildDialog.vue'
import CudaInstallDialog from '@/components/system/LlamaCppManager/CudaInstallDialog.vue'

const systemStore = useSystemStore()
const wsStore = useWebSocketStore()
const confirm = useConfirm()

const updateInfo = ref(null)
const activeVersion = computed(() => {
  return systemStore.llamaVersions.find(version => version.is_active)
})
const checkingUpdates = ref(false)
const refreshingVersions = ref(false)
const installingRelease = ref(false)
const buildingSource = ref(false)
const buildDialogVisible = ref(false)
const activating = ref(null)

// CUDA installation
const cudaStatus = ref(null)
const cudaInstallDialogVisible = ref(false)
const cudaInstallProgress = ref(null)
const cudaInstallLogs = ref([])

// Release dialog
const releaseDialogVisible = ref(false)
const selectedReleaseTag = ref(null)

// Build capabilities
const buildCapabilities = ref(null)

const fetchBuildCapabilities = async () => {
  try {
    const response = await fetch('/api/llama-versions/build-capabilities')
    if (response.ok) {
      buildCapabilities.value = await response.json()
    }
  } catch (error) {
    console.error('Failed to fetch build capabilities:', error)
  }
}

const fetchCudaStatus = async () => {
  try {
    cudaStatus.value = await systemStore.getCudaStatus()
  } catch (error) {
    console.error('Failed to fetch CUDA status:', error)
  }
}

const checkUpdates = async () => {
  checkingUpdates.value = true
  try {
    updateInfo.value = await systemStore.checkUpdates()
    toast.success('Updates checked successfully')
  } catch (error) {
    let errorMessage = 'Failed to check for updates'
    if (error.response?.status === 429) {
      errorMessage = 'GitHub API rate limit exceeded. Please try again later.'
    } else if (error.response?.status === 404) {
      errorMessage = 'GitHub repository not found'
    } else if (error.response?.data?.detail) {
      errorMessage = error.response.data.detail
    }
    toast.error(errorMessage)
  } finally {
    checkingUpdates.value = false
  }
}

const refreshVersions = async () => {
  refreshingVersions.value = true
  try {
    await systemStore.fetchLlamaVersions()
    toast.success('Versions refreshed successfully')
  } catch (error) {
    toast.error('Failed to refresh versions')
  } finally {
    refreshingVersions.value = false
  }
}

const openReleaseDialog = (tagName) => {
  selectedReleaseTag.value = tagName
  releaseDialogVisible.value = true
}

const handleReleaseDialogHide = () => {
  selectedReleaseTag.value = null
}

const handleReleaseInstalled = () => {
  releaseDialogVisible.value = false
  selectedReleaseTag.value = null
}

const showBuildDialog = async () => {
  await fetchBuildCapabilities()
  buildDialogVisible.value = true
}

const handleBuild = () => {
  buildingSource.value = true
  buildDialogVisible.value = false
  buildingSource.value = false
}

const activateVersion = async (versionId) => {
  activating.value = versionId
  try {
    await systemStore.activateVersion(versionId)
    toast.success('Version activated successfully')
    await systemStore.fetchLlamaVersions()
  } catch (error) {
    toast.error('Failed to activate version')
  } finally {
    activating.value = null
  }
}

const confirmDeleteVersion = (version) => {
  confirm.require({
    message: `Are you sure you want to delete version "${version.version}"? This will remove the binary files and cannot be undone.`,
    header: 'Delete Version',
    icon: 'pi pi-exclamation-triangle',
    rejectLabel: 'Cancel',
    acceptLabel: 'Delete',
    accept: async () => {
      try {
        await systemStore.deleteVersion(version.id)
        toast.success(`${version.version} has been deleted`)
      } catch (error) {
        toast.error('Failed to delete version')
      }
    }
  })
}

const showCudaInstallDialog = async () => {
  await fetchCudaStatus()
  cudaInstallDialogVisible.value = true
  cudaInstallProgress.value = null
  cudaInstallLogs.value = []
}

const handleCudaInstall = async (version) => {
  // Installation started, WebSocket will handle updates
}

// WebSocket handlers for CUDA installation
const handleCudaInstallStatus = (data) => {
  if (data.status === 'completed') {
    cudaInstallProgress.value = { progress: 100, message: 'Installation completed!' }
    toast.success(data.message || 'CUDA installation completed')
    fetchCudaStatus()
  } else if (data.status === 'failed') {
    toast.error(data.message || 'CUDA installation failed')
  }
}

const handleCudaInstallProgress = (data) => {
  cudaInstallProgress.value = {
    progress: data.progress || 0,
    message: data.message || 'Installing...',
    stage: data.stage || 'install'
  }
}

const handleCudaInstallLog = (data) => {
  if (data.line) {
    cudaInstallLogs.value.push(data.line)
    if (cudaInstallLogs.value.length > 100) {
      cudaInstallLogs.value = cudaInstallLogs.value.slice(-100)
    }
  }
}

onMounted(async () => {
  // Subscribe to CUDA installation events
  wsStore.subscribe('cuda_install_status', handleCudaInstallStatus)
  wsStore.subscribe('cuda_install_progress', handleCudaInstallProgress)
  wsStore.subscribe('cuda_install_log', handleCudaInstallLog)

  await fetchCudaStatus()
  await systemStore.fetchLlamaVersions()
  await fetchBuildCapabilities()
  
  if (!wsStore.isConnected) {
    wsStore.connect()
  }
})
</script>

<style scoped>
.llama-manager {
  max-width: 1400px;
  margin: 0 auto;
}

.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 0;
  padding-bottom: 0;
  border-bottom: none;
}

.header-content {
  display: flex;
  align-items: center;
  gap: 1rem;
}

.active-version-header {
  margin-left: 1rem;
}

.active-header-badge {
  font-size: 0.875rem;
  font-weight: 600;
}
</style>
