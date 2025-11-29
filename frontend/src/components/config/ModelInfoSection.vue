<template>
  <div class="model-info-section">
    <div class="model-info">
      <h2 class="card-title">{{ model?.huggingface_id || model?.name || 'Model Configuration' }}</h2>
      <div class="model-tags">
        <span class="model-tag tag-size">{{ formatFileSize(model?.file_size) }}</span>
        <span class="model-tag tag-quantization">{{ model?.quantization }}</span>
        <span class="model-tag tag-type">{{ model?.model_type }}</span>
        <span v-if="modelLayerInfo?.architecture && modelLayerInfo.architecture !== model?.model_type"
          class="model-tag tag-architecture">
          {{ modelLayerInfo.architecture }}
        </span>
        <span v-if="modelLayerInfo?.layer_count" class="model-tag tag-layers">
          {{ modelLayerInfo.layer_count }} layers
        </span>
        <span v-if="isEmbeddingModel" class="model-tag tag-pipeline">Embedding</span>
      </div>
      <div class="embedding-notice" v-if="isEmbeddingModel">
        <i class="pi pi-database"></i>
        <div>
          <strong>Embedding model detected</strong>
          <p>This model automatically exposes the /v1/embeddings endpoint via llama.cpp.</p>
        </div>
      </div>
    </div>
    <div class="header-actions">
      <div class="action-buttons">
        <slot name="actions">
          <Button 
            label="Quick Start" 
            icon="pi pi-bolt" 
            size="small"
            severity="info"
            outlined
            @click="$emit('quick-start')"
            v-tooltip="'Choose a preset, use the wizard, or let Smart Auto optimize for you'"
          />
          <Button 
            icon="pi pi-refresh" 
            @click="$emit('regenerate-info')" 
            :loading="regeneratingInfo"
            severity="secondary" 
            size="small" 
            outlined
            v-tooltip="'Regenerate model information from GGUF metadata'" 
          />
          <Button 
            label="Save Config" 
            icon="pi pi-save" 
            @click="$emit('save-config')" 
            :loading="saveLoading"
            severity="success" 
            size="small" 
          />
        </slot>
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed } from 'vue'
import { formatFileSize } from '@/utils/formatting'
import Button from 'primevue/button'

const props = defineProps({
  model: {
    type: Object,
    default: null
  },
  modelLayerInfo: {
    type: Object,
    default: null
  },
  hasHfMetadata: {
    type: Boolean,
    default: false
  },
  hfMetadata: {
    type: Object,
    default: null
  },
  hfMetadataLoading: {
    type: Boolean,
    default: false
  },
  regeneratingInfo: {
    type: Boolean,
    default: false
  },
  saveLoading: {
    type: Boolean,
    default: false
  }
})

defineEmits(['quick-start', 'regenerate-info', 'save-config'])

const isEmbeddingModel = computed(() => {
  return props.model?.pipeline_tag === 'feature-extraction' || 
         props.hfMetadata?.pipeline_tag === 'feature-extraction' ||
         props.model?.model_type?.toLowerCase().includes('embedding')
})
</script>

<style scoped>
.model-info-section {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  gap: 1rem;
}

.model-info {
  flex: 1;
}

.model-tags {
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
  margin: 0.75rem 0;
}

.embedding-notice {
  display: flex;
  gap: 0.75rem;
  padding: 1rem;
  background: var(--status-info-soft);
  border-radius: var(--radius-md);
  border-left: 3px solid var(--accent-blue);
  margin-top: 1rem;
}

.embedding-notice i {
  font-size: 1.5rem;
  color: var(--accent-blue);
}

.embedding-notice strong {
  display: block;
  margin-bottom: 0.25rem;
  color: var(--text-primary);
}

.embedding-notice p {
  margin: 0;
  font-size: 0.875rem;
  color: var(--text-secondary);
}

.header-actions {
  display: flex;
  align-items: center;
}

.action-buttons {
  display: flex;
  gap: 0.5rem;
  flex-wrap: wrap;
}

@media (max-width: 768px) {
  .model-info-section {
    flex-direction: column;
  }
  
  .action-buttons {
    width: 100%;
  }
  
  .action-buttons .p-button {
    flex: 1;
  }
}
</style>

