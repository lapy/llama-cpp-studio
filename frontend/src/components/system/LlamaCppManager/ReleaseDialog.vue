<template>
  <BaseDialog
    :visible="visible"
    :header="selectedReleaseTag ? `Install Release ${selectedReleaseTag}` : 'Install Release'"
    :modal="true"
    :dialog-style="{ width: '60vw', maxWidth: '750px' }"
    :draggable="false"
    :resizable="false"
    dialog-class="release-dialog"
    @update:visible="handleVisibleChange"
    @hide="handleHide"
  >
    <div class="release-dialog-body">
      <div v-if="loading" class="release-assets-loading">
        <ProgressSpinner style="width: 48px; height: 48px" strokeWidth="4" />
        <span>Loading release artifacts…</span>
      </div>
      
      <div v-else-if="error" class="release-assets-error">
        <i class="pi pi-exclamation-triangle"></i>
        <p>{{ error }}</p>
        <div v-if="skippedAssets.length" class="skipped-artifacts">
          <h5>Filtered Out</h5>
          <ul>
            <li v-for="asset in skippedAssets" :key="asset.id || asset.name">
              <span class="skipped-name">{{ asset.name }}</span>
              <span class="skipped-reason">{{ asset.compatibility_reason || 'Incompatible with container' }}</span>
            </li>
          </ul>
        </div>
      </div>
      
      <div v-else class="release-asset-list">
        <div 
          v-for="asset in assets" 
          :key="asset.id" 
          :class="['release-asset-option', { selected: selectedAssetId === asset.id }]"
          @click="selectedAssetId = asset.id"
        >
          <div class="asset-option-header">
            <RadioButton 
              :inputId="`release-asset-${asset.id}`"
              :value="asset.id"
              v-model="selectedAssetId"
            />
            <label :for="`release-asset-${asset.id}`" class="asset-label">
              <span class="asset-name">{{ asset.name }}</span>
            </label>
            <span class="asset-size">{{ formatBytes(asset.size) }}</span>
          </div>
          <div v-if="asset.features && asset.features.length" class="asset-features">
            <Tag 
              v-for="feature in asset.features"
              :key="feature"
              severity="info"
              class="asset-feature-tag"
            >
              {{ feature }}
            </Tag>
          </div>
          <div class="asset-meta">
            <span class="archive-type">{{ (asset.archive_type || '').toUpperCase() }}</span>
            <span 
              v-if="asset.download_count !== undefined && asset.download_count !== null" 
              class="download-count"
            >
              {{ asset.download_count }} downloads
            </span>
          </div>
        </div>
        
        <div v-if="skippedAssets.length" class="skipped-artifacts">
          <h5>Filtered Out</h5>
          <ul>
            <li v-for="asset in skippedAssets" :key="asset.id || asset.name">
              <span class="skipped-name">{{ asset.name }}</span>
              <span class="skipped-reason">{{ asset.compatibility_reason || 'Incompatible with container' }}</span>
            </li>
          </ul>
        </div>
      </div>
    </div>
    
    <template #footer>
      <Button 
        label="Cancel" 
        icon="pi pi-times" 
        @click="handleCancel"
        severity="secondary"
        text
      />
      <Button 
        label="Install" 
        icon="pi pi-download" 
        @click="handleInstall"
        :loading="installing"
        :disabled="!selectedAssetId || assets.length === 0 || loading"
      />
    </template>
  </BaseDialog>
</template>

<script setup>
import { ref, computed, watch } from 'vue'
import { useSystemStore } from '@/stores/system'
import { toast } from 'vue3-toastify'
import { formatBytes } from '@/utils/formatting'
import Button from 'primevue/button'
import Tag from 'primevue/tag'
import RadioButton from 'primevue/radiobutton'
import ProgressSpinner from 'primevue/progressspinner'
import BaseDialog from '@/components/common/BaseDialog.vue'

const props = defineProps({
  visible: {
    type: Boolean,
    default: false
  },
  releaseTag: {
    type: String,
    default: null
  }
})

const emit = defineEmits(['update:visible', 'installed', 'hide'])

const systemStore = useSystemStore()

const loading = ref(false)
const error = ref(null)
const assets = ref([])
const skippedAssets = ref([])
const selectedAssetId = ref(null)
const installing = ref(false)

const selectedAsset = computed(() => {
  return assets.value.find(asset => asset.id === selectedAssetId.value) || null
})

const resetState = () => {
  assets.value = []
  skippedAssets.value = []
  error.value = null
  loading.value = false
  selectedAssetId.value = null
}

const loadReleaseAssets = async (tagName) => {
  if (!tagName) return
  
  resetState()
  loading.value = true
  
  try {
    const data = await systemStore.fetchReleaseAssets(tagName)
    assets.value = (data?.assets || []).map(asset => ({
      ...asset,
      id: asset.id !== undefined && asset.id !== null ? Number(asset.id) : asset.id
    }))
    skippedAssets.value = (data?.skipped_assets || []).map(asset => ({
      ...asset,
      id: asset.id !== undefined && asset.id !== null ? Number(asset.id) : asset.id
    }))
    
    if (assets.value.length === 0) {
      error.value = 'No compatible artifacts were found for this release in the current container.'
    } else {
      const defaultId = data?.default_asset_id
      if (defaultId !== undefined && defaultId !== null) {
        selectedAssetId.value = Number(defaultId)
      } else {
        selectedAssetId.value = assets.value[0]?.id ?? null
      }
    }
  } catch (err) {
    if (err.response?.data?.detail) {
      error.value = err.response.data.detail
    } else if (err.message) {
      error.value = err.message
    } else {
      error.value = 'Failed to load release artifacts.'
    }
  } finally {
    loading.value = false
  }
}

const handleVisibleChange = (value) => {
  emit('update:visible', value)
  if (value && props.releaseTag) {
    loadReleaseAssets(props.releaseTag)
  }
}

const handleHide = () => {
  resetState()
  emit('hide')
}

const handleCancel = () => {
  emit('update:visible', false)
}

const handleInstall = async () => {
  const tagName = props.releaseTag
  const asset = selectedAsset.value
  
  if (!tagName) return
  if (!asset) {
    toast.error('Please select an artifact to install.')
    return
  }
  
  installing.value = true
  let installSucceeded = false
  
  try {
    await systemStore.installRelease(tagName, asset.id)
    installSucceeded = true
    const assetLabel = asset.name ? ` (${asset.name})` : ''
    toast.success(`Installing release ${tagName}${assetLabel}`)
    await systemStore.fetchLlamaVersions()
    emit('installed', { tagName, asset })
  } catch (err) {
    let errorMessage = 'Failed to install release'
    let detail = err.response?.data?.detail
    if (typeof detail === 'string') {
      const trimmedDetail = detail.startsWith('400:') ? detail.substring(4).trim() : detail
      if (trimmedDetail.toLowerCase().includes('version already installed')) {
        errorMessage = 'That release artifact is already installed. Select a different artifact or remove the existing installation first.'
      } else if (trimmedDetail.length > 0) {
        errorMessage = trimmedDetail
      }
    } else if (detail) {
      errorMessage = detail
    } else if (err.message) {
      errorMessage = err.message
    }
    
    toast.error(errorMessage)
  } finally {
    installing.value = false
    if (installSucceeded) {
      emit('update:visible', false)
      resetState()
    }
  }
}

watch(() => props.releaseTag, (newTag) => {
  if (props.visible && newTag) {
    loadReleaseAssets(newTag)
  }
})

watch(() => props.visible, (newVisible) => {
  if (newVisible && props.releaseTag) {
    loadReleaseAssets(props.releaseTag)
  } else if (!newVisible) {
    resetState()
  }
})
</script>

<style scoped>
.release-dialog :deep(.p-dialog-content) {
  padding-top: 0;
}

.release-dialog-body {
  display: flex;
  flex-direction: column;
  gap: var(--spacing-lg);
}

.release-assets-loading {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: var(--spacing-md);
  padding: var(--spacing-2xl) 0;
  color: var(--text-secondary);
}

.release-assets-error {
  text-align: center;
  padding: var(--spacing-xl) var(--spacing-lg);
  color: var(--text-secondary);
}

.release-assets-error i {
  font-size: 1.5rem;
  color: var(--status-warning);
  margin-bottom: var(--spacing-sm);
  display: block;
}

.release-assets-error p {
  margin: 0;
}

.release-asset-list {
  display: flex;
  flex-direction: column;
  gap: var(--spacing-md);
  max-height: 60vh;
  overflow-y: auto;
  padding-right: 0.5rem;
}

.release-asset-option {
  border: 1px solid var(--border-primary);
  border-radius: var(--radius-lg);
  padding: var(--spacing-lg);
  background: var(--bg-card);
  transition: border-color var(--transition-fast), box-shadow var(--transition-fast), transform var(--transition-fast);
  cursor: pointer;
}

.release-asset-option:hover {
  border-color: var(--primary-color);
  box-shadow: var(--shadow-sm);
  transform: translateY(-1px);
}

.release-asset-option.selected {
  border-color: var(--primary-color);
  box-shadow: var(--shadow-sm);
}

.asset-option-header {
  display: flex;
  align-items: center;
  gap: var(--spacing-md);
  justify-content: space-between;
}

.asset-label {
  flex: 1;
  display: flex;
  flex-direction: column;
  gap: 0.25rem;
}

.asset-name {
  font-weight: 600;
  color: var(--text-primary);
  word-break: break-word;
}

.asset-size {
  font-size: 0.85rem;
  color: var(--text-secondary);
  white-space: nowrap;
}

.asset-features {
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
  margin-top: var(--spacing-md);
}

.asset-feature-tag {
  font-size: 0.75rem;
  font-weight: 600;
}

.asset-meta {
  margin-top: var(--spacing-md);
  display: flex;
  gap: var(--spacing-lg);
  font-size: 0.8rem;
  color: var(--text-secondary);
}

.archive-type {
  text-transform: uppercase;
  letter-spacing: 0.05em;
}

.download-count {
  color: var(--text-secondary);
}

.skipped-artifacts {
  margin-top: var(--spacing-lg);
  border-top: 1px solid var(--border-primary);
  padding-top: var(--spacing-md);
  text-align: left;
}

.skipped-artifacts h5 {
  margin: 0 0 var(--spacing-sm);
  font-size: 0.9rem;
  font-weight: 600;
  color: var(--text-secondary);
}

.skipped-artifacts ul {
  margin: 0;
  padding-left: 1rem;
  display: flex;
  flex-direction: column;
  gap: 0.4rem;
}

.skipped-name {
  font-weight: 600;
  color: var(--text-secondary);
}

.skipped-reason {
  display: block;
  font-size: 0.8rem;
  color: var(--text-secondary);
}
</style>

