<template>
  <div class="llama-manager">
    <div class="card">
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

      <!-- Build Progress -->
      <BuildProgress />

      <!-- Update Information -->
      <div v-if="updateInfo" class="update-info">
        <h3>Available Updates</h3>
        <div class="update-cards">
          <div class="update-card">
            <div class="update-header">
              <h4>Latest Release</h4>
              <Tag :value="updateInfo.latest_release?.tag_name || 'N/A'" severity="info" />
            </div>
            <p v-if="updateInfo.latest_release">
              Published: {{ formatDate(updateInfo.latest_release.published_at) }}
            </p>
            <Button 
              label="Install Release"
              icon="pi pi-download"
              @click="installRelease(updateInfo.latest_release.tag_name)"
              :loading="installingRelease"
              :disabled="!updateInfo.latest_release"
            />
          </div>
          
          <div class="update-card">
            <div class="update-header">
              <h4>Latest Source</h4>
              <Tag :value="updateInfo.latest_commit?.sha?.substring(0, 8) || 'N/A'" severity="success" />
            </div>
            <p v-if="updateInfo.latest_commit">
              {{ updateInfo.latest_commit.message }}
            </p>
            <Button 
              label="Build from Source"
              icon="pi pi-code"
              @click="showBuildDialog"
              :loading="buildingSource"
            />
          </div>
        </div>
      </div>

      <!-- Installed Versions -->
      <div class="installed-versions">
        <h3>Installed Versions</h3>
        <div v-if="systemStore.llamaVersions.length === 0" class="empty-state">
          <i class="pi pi-code" style="font-size: 3rem; color: var(--text-color-secondary);"></i>
          <h4>No Versions Installed</h4>
          <p>Install a release or build from source to get started.</p>
        </div>
        
        <div v-else class="version-list">
          <div 
            v-for="version in systemStore.llamaVersions" 
            :key="version.id"
            class="version-card"
            :class="{ 'active-version': version.is_active }"
          >
            <div class="version-header">
              <div class="version-info">
                <div class="version-title">
                  <h4>{{ version.version }}</h4>
                  <div v-if="version.is_active" class="active-indicators">
                    <Tag 
                      value="ACTIVE" 
                      severity="success"
                      class="active-badge"
                    />
                    <i class="pi pi-check-circle active-icon"></i>
                  </div>
                </div>
                <div class="version-meta">
                  <Tag 
                    :value="version.install_type" 
                    :severity="getInstallTypeSeverity(version.install_type)"
                  />
                  <span class="install-date">
                    Installed: {{ formatDate(version.installed_at) }}
                  </span>
                </div>
              </div>
              <div class="version-actions">
                <Button 
                  v-if="!version.is_active"
                  icon="pi pi-check"
                  @click="activateVersion(version.id)"
                  severity="success"
                  size="small"
                  text
                  :loading="activating === version.id"
                />
                <Button 
                  icon="pi pi-trash"
                  severity="danger"
                  outlined
                  @click="confirmDeleteVersion(version)"
                  :disabled="version.is_active"
                />
              </div>
            </div>
            
            <div v-if="version.source_commit" class="version-details">
              <p><strong>Commit:</strong> {{ version.source_commit }}</p>
            </div>
            
            <div v-if="version.patches && version.patches.length > 0" class="version-patches">
              <p><strong>Patches Applied:</strong></p>
              <ul>
                <li v-for="patch in version.patches" :key="patch">
                  <a :href="patch" target="_blank">{{ patch }}</a>
                </li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Build from Source Dialog -->
    <Dialog 
      v-model:visible="buildDialogVisible" 
      header="Build from Source"
      :modal="true"
      :style="{ width: '70vw', maxWidth: '900px' }"
      :draggable="false"
      :resizable="false"
      class="build-dialog"
    >
      <div class="build-form">
        <div class="dialog-section">
          <h4 class="section-title">Source Information</h4>
          <div class="form-field full-width">
            <label>Commit SHA or Branch *</label>
            <InputText 
              v-model="buildForm.commitSha"
              placeholder="master or commit hash"
            />
            <small>Default: master (latest stable)</small>
          </div>
          
          <div class="form-field full-width">
            <label>Patches (Optional)</label>
            <Textarea 
              v-model="buildForm.patches"
              rows="3"
              placeholder="One patch URL per line"
            />
            <small>GitHub PR URLs or raw patch file URLs</small>
          </div>
        </div>
        
        <div class="dialog-section">
          <h4 class="section-title">Build Configuration</h4>
          <div class="form-row">
            <div class="form-field">
              <label>Build Type</label>
              <Dropdown 
                v-model="buildForm.buildType"
                :options="['Release', 'Debug', 'RelWithDebInfo']"
                placeholder="Select build type"
              />
              <small>Release recommended for production</small>
            </div>
          </div>
        </div>
        
        <div class="dialog-section">
          <h4 class="section-title">GPU Backends</h4>
          <div class="checkbox-group">
            <div class="checkbox-item">
              <Checkbox 
                v-model="buildForm.enableCuda" 
                :binary="true"
              />
              <div class="checkbox-label">
                <span>CUDA</span>
                <small class="capability-info" :class="getCapabilityClass(buildCapabilities?.cuda)">
                  {{ buildCapabilities?.cuda?.reason || 'Not detected' }}
                </small>
              </div>
            </div>
            
            <div class="checkbox-item">
              <Checkbox 
                v-model="buildForm.enableVulkan" 
                :binary="true"
              />
              <div class="checkbox-label">
                <span>Vulkan</span>
                <small class="capability-info" :class="getCapabilityClass(buildCapabilities?.vulkan)">
                  {{ buildCapabilities?.vulkan?.reason || 'Not detected' }}
                </small>
              </div>
            </div>
            
            <div class="checkbox-item">
              <Checkbox 
                v-model="buildForm.enableMetal" 
                :binary="true"
              />
              <div class="checkbox-label">
                <span>Metal</span>
                <small class="capability-info" :class="getCapabilityClass(buildCapabilities?.metal)">
                  {{ buildCapabilities?.metal?.reason || 'Not detected' }}
                </small>
              </div>
            </div>
            
            <div class="checkbox-item">
              <Checkbox 
                v-model="buildForm.enableOpenBLAS" 
                :binary="true"
              />
              <div class="checkbox-label">
                <span>OpenBLAS</span>
                <small class="capability-info" :class="getCapabilityClass(buildCapabilities?.openblas)">
                  {{ buildCapabilities?.openblas?.reason || 'Not detected' }}
                </small>
              </div>
            </div>
            
            <div class="checkbox-item">
              <Checkbox 
                v-model="buildForm.enableFlashAttention" 
                :binary="true"
              />
              <div class="checkbox-label">
                <span>Flash Attention (FA)</span>
                <small class="capability-info" :class="buildForm.enableCuda ? 'text-blue-500' : 'text-gray-500'">
                  Requires CUDA - enables V cache quantization
                </small>
              </div>
            </div>
          </div>
        </div>
        
        <Accordion class="advanced-accordion">
          <AccordionTab header="Advanced Options">
            <div class="form-field">
              <label>Custom CMake Arguments</label>
              <InputText 
                v-model="buildForm.customCmakeArgs"
                placeholder="-DGGML_CUBLAS=ON -DGGML_VULKAN=ON"
              />
              <small>Additional CMake flags</small>
            </div>
            <div class="form-row">
              <div class="form-field">
                <label>CFLAGS</label>
                <InputText v-model="buildForm.cflags" placeholder="-O3 -march=native"/>
              </div>
              <div class="form-field">
                <label>CXXFLAGS</label>
                <InputText v-model="buildForm.cxxflags" placeholder="-O3 -march=native"/>
              </div>
            </div>
          </AccordionTab>
        </Accordion>
      </div>
      
      <template #footer>
        <Button 
          label="Cancel" 
          icon="pi pi-times" 
          @click="buildDialogVisible = false"
          severity="secondary"
          text
        />
        <Button 
          label="Build" 
          icon="pi pi-code" 
          @click="buildFromSource"
          :loading="buildingSource"
        />
      </template>
    </Dialog>
  </div>
</template>

<script setup>
import { ref, onMounted, computed } from 'vue'
import { useSystemStore } from '@/stores/system'
import { useWebSocketStore } from '@/stores/websocket'
import { toast } from 'vue3-toastify'
import { useConfirm } from 'primevue/useconfirm'
import Button from 'primevue/button'
import InputText from 'primevue/inputtext'
import Card from 'primevue/card'
import Tag from 'primevue/tag'
import Dialog from 'primevue/dialog'
import Textarea from 'primevue/textarea'
import Dropdown from 'primevue/dropdown'
import Checkbox from 'primevue/checkbox'
import Accordion from 'primevue/accordion'
import AccordionTab from 'primevue/accordiontab'
import BuildProgress from '@/components/BuildProgress.vue'

const systemStore = useSystemStore()
const wsStore = useWebSocketStore()
const confirm = useConfirm()

const updateInfo = ref(null)

// Computed property to get the currently active version
const activeVersion = computed(() => {
  return systemStore.llamaVersions.find(version => version.is_active)
})
const checkingUpdates = ref(false)
const refreshingVersions = ref(false)
const installingRelease = ref(false)
const buildingSource = ref(false)
const buildDialogVisible = ref(false)
const activating = ref(null)

// Build configuration
const buildForm = ref({
  commitSha: 'master',
  patches: '',
  buildType: 'Release',
  enableCuda: true,
  enableVulkan: false,
  enableMetal: false,
  enableOpenBLAS: false,
  enableFlashAttention: false,
  customCmakeArgs: '',
  cflags: '',
  cxxflags: ''
})

const buildCapabilities = ref(null)
const loadingCapabilities = ref(false)

const fetchBuildCapabilities = async () => {
  loadingCapabilities.value = true
  try {
    const response = await fetch('/api/llama-versions/build-capabilities')
    if (response.ok) {
      buildCapabilities.value = await response.json()
      // Auto-enable recommended backends
      if (buildCapabilities.value?.cuda?.recommended) {
        buildForm.value.enableCuda = true
      }
      if (buildCapabilities.value?.vulkan?.recommended) {
        buildForm.value.enableVulkan = true
      }
      if (buildCapabilities.value?.metal?.recommended) {
        buildForm.value.enableMetal = true
      }
      if (buildCapabilities.value?.openblas?.recommended) {
        buildForm.value.enableOpenBLAS = true
      }
    }
  } catch (error) {
    console.error('Failed to fetch build capabilities:', error)
  } finally {
    loadingCapabilities.value = false
  }
}

const getCapabilityClass = (capability) => {
  if (!capability) return 'text-gray-500'
  return capability.available ? 'text-green-500' : 'text-gray-500'
}

onMounted(async () => {
  await systemStore.fetchLlamaVersions()
  await fetchBuildCapabilities()
  // Ensure WebSocket connection is established
  if (!wsStore.isConnected) {
    wsStore.connect()
  }
})

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

const installRelease = async (tagName) => {
  if (!tagName) return
  
  installingRelease.value = true
  try {
    await systemStore.installRelease(tagName)
    toast.success(`Installing release ${tagName}`)
    // Refresh the versions list after installation starts
    await systemStore.fetchLlamaVersions()
  } catch (error) {
    let errorMessage = 'Failed to install release'
    if (error.response?.data?.detail) {
      errorMessage = error.response.data.detail
    } else if (error.message) {
      errorMessage = error.message
    }
    
    toast.error(errorMessage)
  } finally {
    installingRelease.value = false
  }
}

const showBuildDialog = async () => {
  // Fetch fresh capabilities before showing dialog
  await fetchBuildCapabilities()
  
  buildForm.value = {
    commitSha: 'master',
    patches: '',
    buildType: 'Release',
    enableCuda: buildCapabilities.value?.cuda?.recommended || false,
    enableVulkan: buildCapabilities.value?.vulkan?.recommended || false,
    enableMetal: buildCapabilities.value?.metal?.recommended || false,
    enableOpenBLAS: buildCapabilities.value?.openblas?.recommended || false,
    enableFlashAttention: false,
    customCmakeArgs: '',
    cflags: '',
    cxxflags: ''
  }
  buildDialogVisible.value = true
}

const buildFromSource = async () => {
  buildingSource.value = true
  buildDialogVisible.value = false
  
  try {
    const patches = buildForm.value.patches
      .split('\n')
      .map(line => line.trim())
      .filter(line => line)
    
    // Build configuration object - backend is source of truth
    const buildConfig = {
      build_type: buildForm.value.buildType || 'Release',
      enable_cuda: buildForm.value.enableCuda || false,
      enable_vulkan: buildForm.value.enableVulkan || false,
      enable_metal: buildForm.value.enableMetal || false,
      enable_openblas: buildForm.value.enableOpenBLAS || false,
      enable_flash_attention: buildForm.value.enableFlashAttention || false,
      custom_cmake_args: buildForm.value.customCmakeArgs || '',
      cflags: buildForm.value.cflags || '',
      cxxflags: buildForm.value.cxxflags || ''
    }
    
    await systemStore.buildSource(buildForm.value.commitSha, patches, buildConfig)
    
    // Build progress will be shown in the BuildProgress component via WebSocket
    
  } catch (error) {
    toast.error('Failed to start build from source')
  } finally {
    buildingSource.value = false
  }
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

const getInstallTypeSeverity = (type) => {
  switch (type) {
    case 'release': return 'info'
    case 'source': return 'success'
    case 'patched': return 'warning'
    default: return 'secondary'
  }
}

const formatDate = (dateString) => {
  if (!dateString) return 'Unknown'
  return new Date(dateString).toLocaleDateString()
}
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
  margin-bottom: 2rem;
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

.update-info {
  margin-bottom: 2rem;
}

.update-cards {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: 1rem;
  margin-top: 1rem;
}

.update-card {
  background: var(--gradient-card);
  border: 1px solid var(--border-primary);
  border-radius: var(--radius-xl);
  padding: var(--spacing-xl);
  box-shadow: var(--shadow-md);
  transition: all var(--transition-normal);
  position: relative;
  overflow: hidden;
}

.update-card::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  height: 3px;
  background: var(--gradient-primary);
  opacity: 0;
  transition: opacity var(--transition-normal);
}

.update-card:hover {
  transform: translateY(-3px);
  box-shadow: var(--shadow-lg);
}

.update-card:hover::before {
  opacity: 1;
}

.update-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 0.5rem;
}

.update-header h4 {
  margin: 0;
  color: var(--text-primary);
  font-weight: 700;
  font-size: 1.1rem;
}

.update-card p {
  margin: var(--spacing-md) 0 var(--spacing-lg);
  font-size: 0.9rem;
  color: var(--text-secondary);
  line-height: 1.5;
}

.installed-versions {
  margin-top: 2rem;
}

.empty-state {
  text-align: center;
  padding: var(--spacing-3xl) var(--spacing-xl);
  color: var(--text-secondary);
  background: var(--gradient-surface);
  border-radius: var(--radius-xl);
  border: 2px dashed var(--border-secondary);
  margin: var(--spacing-xl) 0;
  position: relative;
  overflow: hidden;
}

.empty-state::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  height: 2px;
  background: var(--gradient-primary);
  opacity: 0.3;
}

.empty-state i {
  font-size: 3rem !important;
  background: var(--gradient-primary);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  margin-bottom: var(--spacing-lg);
}

.empty-state h4 {
  margin: var(--spacing-lg) 0 var(--spacing-md);
  color: var(--text-primary);
  font-weight: 700;
  font-size: 1.3rem;
}

.version-list {
  display: flex;
  flex-direction: column;
  gap: 1rem;
  margin-top: 1rem;
}

.version-card {
  background: var(--gradient-card);
  border: 1px solid var(--border-primary);
  border-radius: var(--radius-xl);
  padding: var(--spacing-xl);
  box-shadow: var(--shadow-md);
  transition: all var(--transition-normal);
  position: relative;
  overflow: hidden;
  backdrop-filter: blur(10px);
  animation: fadeIn 0.6s ease-out;
}

.version-card::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  height: 3px;
  background: var(--gradient-primary);
  opacity: 0;
  transition: opacity var(--transition-normal);
}

.version-card:hover {
  transform: translateY(-5px) scale(1.02);
  box-shadow: var(--shadow-lg), var(--glow-primary);
}

.version-card:hover::before {
  opacity: 1;
}

.version-card.active-version {
  background: var(--gradient-card);
  border: 2px solid var(--status-success);
  box-shadow: var(--shadow-lg), var(--glow-success);
}

.version-card.active-version::before {
  background: var(--gradient-success);
  opacity: 1;
}

.version-card.active-version:hover {
  box-shadow: var(--shadow-xl), var(--glow-success);
}

.version-header {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  margin-bottom: 0.5rem;
}

.version-title {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  margin-bottom: 0.5rem;
}

.version-title h4 {
  margin: 0;
  color: var(--text-primary);
  font-weight: 700;
  font-size: 1.2rem;
}

.active-indicators {
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.active-badge {
  font-size: 0.75rem;
  font-weight: 600;
  animation: pulse 2s infinite;
}

.active-icon {
  color: #22c55e;
  font-size: 1.25rem;
  animation: pulse 2s infinite;
}

@keyframes pulse {
  0%, 100% {
    opacity: 1;
  }
  50% {
    opacity: 0.7;
  }
}

.version-info h4 {
  margin: 0 0 var(--spacing-sm);
  color: var(--text-primary);
  font-weight: 700;
  font-size: 1.1rem;
}

.version-meta {
  display: flex;
  align-items: center;
  gap: 1rem;
}

.install-date {
  font-size: 0.9rem;
  color: var(--text-secondary);
  font-weight: 500;
}

.version-details,
.version-patches {
  margin-top: var(--spacing-md);
  font-size: 0.9rem;
  color: var(--text-secondary);
  line-height: 1.5;
}

.version-patches ul {
  margin: 0.5rem 0 0 1rem;
}

.version-patches a {
  color: var(--accent-cyan);
  text-decoration: none;
  font-weight: 500;
}

.version-patches a:hover {
  text-decoration: underline;
  color: var(--accent-blue);
}

/* Enhanced Dialog styling */
:deep(.build-dialog .p-dialog-content) {
  padding: 2rem 1.5rem !important;
  overflow-y: auto !important;
  max-height: calc(80vh - 120px) !important;
  background: transparent !important;
}

.build-form {
  display: flex;
  flex-direction: column;
  gap: 1.5rem;
}

.dialog-section {
  display: flex;
  flex-direction: column;
  gap: 1rem;
  padding: 1.5rem;
  background: var(--gradient-card);
  border: 1px solid var(--border-primary);
  border-radius: var(--radius-xl);
  transition: all var(--transition-normal);
  position: relative;
  overflow: hidden;
  backdrop-filter: blur(10px);
  animation: fadeIn 0.6s ease-out;
}

.dialog-section::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  height: 3px;
  background: var(--gradient-primary);
  opacity: 0;
  transition: opacity var(--transition-normal);
}

.dialog-section:hover {
  border-color: var(--accent-cyan);
  box-shadow: var(--shadow-lg), var(--glow-primary);
  transform: translateY(-2px);
}

.dialog-section:hover::before {
  opacity: 1;
}

.section-title {
  margin: 0 0 0.75rem 0;
  color: var(--text-primary);
  font-size: 1.2rem;
  font-weight: 700;
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.section-title::before {
  content: '';
  width: 4px;
  height: 1.5rem;
  background: var(--gradient-primary);
  border-radius: 2px;
}

.form-row {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 1rem;
}

.form-field {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}

.form-field.full-width {
  grid-column: 1 / -1;
}

.form-field label {
  font-weight: 600;
  color: var(--text-primary);
  font-size: 0.9rem;
}

.form-field small {
  color: var(--text-secondary);
  font-size: 0.8rem;
  margin-top: 0.25rem;
}

.advanced-accordion {
  margin-top: 0.5rem;
}

.checkbox-group {
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
}

.checkbox-item {
  display: flex;
  align-items: flex-start;
  gap: 0.75rem;
  padding: 1rem;
  background: var(--gradient-surface);
  border-radius: var(--radius-lg);
  border: 1px solid var(--border-primary);
  transition: all var(--transition-normal);
  position: relative;
  overflow: hidden;
}

.checkbox-item::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  height: 2px;
  background: var(--gradient-primary);
  opacity: 0;
  transition: opacity var(--transition-normal);
}

.checkbox-item:hover {
  background: var(--gradient-card);
  border-color: var(--accent-cyan);
  transform: translateX(4px);
  box-shadow: var(--shadow-md);
}

.checkbox-item:hover::before {
  opacity: 1;
}

.checkbox-label {
  display: flex;
  flex-direction: column;
  gap: 0.25rem;
  flex: 1;
}

.checkbox-label span {
  font-weight: 600;
  color: var(--text-primary);
  font-size: 0.9rem;
}

.capability-info {
  color: var(--text-secondary);
  font-size: 0.8rem;
  font-weight: 400;
  line-height: 1.4;
}

.capability-info.text-green-500 {
  color: #22c55e;
}

.capability-info.text-gray-500 {
  color: #6b7280;
}

.build-progress {
  display: flex;
  flex-direction: column;
  gap: var(--spacing-lg);
}

.build-message {
  margin: 0;
  font-weight: 600;
  color: var(--text-primary);
  font-size: 1rem;
}

.build-log {
  background: var(--bg-surface);
  border-radius: var(--radius-lg);
  padding: var(--spacing-lg);
  max-height: 300px;
  overflow-y: auto;
  border: 1px solid var(--border-primary);
}

.build-log pre {
  margin: 0;
  font-family: 'Courier New', monospace;
  font-size: 0.9rem;
  white-space: pre-wrap;
  word-break: break-word;
  color: var(--text-primary);
}

@media (max-width: 768px) {
  .update-cards {
    grid-template-columns: 1fr;
  }
  
  .version-header {
    flex-direction: column;
    gap: 1rem;
  }
  
  .version-meta {
    flex-direction: column;
    align-items: flex-start;
    gap: 0.5rem;
  }
}
</style>
