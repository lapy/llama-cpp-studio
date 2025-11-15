<template>
  <div class="model-grid">
    <div
      v-for="modelGroup in modelGroups"
      :key="modelGroup.huggingface_id"
      class="model-card"
    >
      <div class="model-card-header">
        <div>
          <div class="model-name">{{ modelGroup.huggingface_id }}</div>
          <div class="model-tags">
            <span class="model-tag tag-type">{{ modelGroup.model_type }}</span>
            <span class="model-tag tag-count">{{ modelGroup.quantizations.length }} quantizations</span>
          </div>
        </div>
        <div class="model-status">
          <span
            :class="[
              'status-indicator',
              hasRunningQuantization(modelGroup) ? 'status-running' : 'status-stopped',
              { 'llama-swap-running': hasLlamaSwapQuantization(modelGroup) }
            ]"
          >
            <i :class="hasLlamaSwapQuantization(modelGroup) ? 'pi pi-share-alt' : (hasRunningQuantization(modelGroup) ? 'pi pi-play' : 'pi pi-pause')"></i>
            {{ getModelStatusText(modelGroup) }}
          </span>
        </div>
      </div>

      <div class="quantization-list">
        <div
          v-for="quantization in modelGroup.quantizations"
          :key="quantization.id"
          class="quantization-item"
          :class="{ selected: selectedQuantization[modelGroup.huggingface_id] === quantization.id }"
        >
          <div class="quantization-info">
            <div class="quantization-name">
              {{ quantization.quantization }}
              <Button
                v-if="quantization.is_active && quantization.proxy_name"
                icon="pi pi-external-link"
                @click="openUpstreamUrl(quantization.proxy_name)"
                severity="info"
                size="small"
                text
                class="upstream-link"
                v-tooltip.top="getUpstreamUrl(quantization.proxy_name)"
              />
            </div>
            <div class="quantization-details">
              <span class="quantization-size">{{ formatFileSize(quantization.file_size) }}</span>
              <span
                v-if="quantization.is_active"
                class="quantization-status running"
                :class="{ 'llama-swap-running': quantization.llama_swap_status === 'running' }"
              >
                <i :class="quantization.llama_swap_status === 'running' ? 'pi pi-share-alt' : 'pi pi-play'"></i>
                Running
              </span>
            </div>
          </div>
          <div class="quantization-actions">
            <Button
              icon="pi pi-check"
              @click="emitSelectQuantization(modelGroup.huggingface_id, quantization.id)"
              :class="{ 'p-button-outlined': selectedQuantization[modelGroup.huggingface_id] !== quantization.id }"
              size="small"
              severity="success"
              text
            />
            <Button
              icon="pi pi-trash"
              @click="emitDeleteQuantization(quantization)"
              severity="danger"
              size="small"
              text
            />
          </div>
        </div>
      </div>

      <div class="model-actions">
        <div class="action-group">
          <Button
            v-if="!hasRunningQuantization(modelGroup)"
            label="Start"
            icon="pi pi-play"
            @click="emitStart(modelGroup)"
            :loading="startingModels[selectedQuantization[modelGroup.huggingface_id]]"
            :disabled="!selectedQuantization[modelGroup.huggingface_id]"
            severity="success"
            size="small"
          />
          <Button
            v-else
            label="Stop"
            icon="pi pi-stop"
            @click="emitStop(modelGroup)"
            :loading="stoppingModels[getRunningQuantizationId(modelGroup)]"
            severity="danger"
            size="small"
          />
          <Button
            label="Configure"
            icon="pi pi-cog"
            @click="emitConfigure(modelGroup)"
            :disabled="!selectedQuantization[modelGroup.huggingface_id]"
            severity="secondary"
            size="small"
            outlined
          />
        </div>
        <div class="action-group">
          <Button
            label="Delete All"
            icon="pi pi-trash"
            @click="emitDeleteGroup(modelGroup)"
            severity="danger"
            size="small"
            outlined
          />
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import Button from 'primevue/button'

const props = defineProps({
  modelGroups: {
    type: Array,
    default: () => []
  },
  selectedQuantization: {
    type: Object,
    default: () => ({})
  },
  startingModels: {
    type: Object,
    default: () => ({})
  },
  stoppingModels: {
    type: Object,
    default: () => ({})
  }
})

const emit = defineEmits([
  'select-quantization',
  'start',
  'stop',
  'configure',
  'delete-quantization',
  'delete-group'
])

const hasRunningQuantization = (modelGroup) => {
  return modelGroup.quantizations?.some(q => q.is_active)
}

const hasLlamaSwapQuantization = (modelGroup) => {
  return modelGroup.quantizations?.some(q => q.llama_swap_status === 'running')
}

const getModelStatusText = (modelGroup) => {
  return hasRunningQuantization(modelGroup) ? 'Running' : 'Stopped'
}

const getRunningQuantizationId = (modelGroup) => {
  const running = modelGroup.quantizations?.find(q => q.is_active)
  return running ? running.id : null
}

const emitSelectQuantization = (huggingfaceId, quantizationId) => {
  emit('select-quantization', { huggingfaceId, quantizationId })
}

const emitStart = (modelGroup) => {
  emit('start', modelGroup)
}

const emitStop = (modelGroup) => {
  emit('stop', {
    modelGroup,
    quantizationId: getRunningQuantizationId(modelGroup)
  })
}

const emitConfigure = (modelGroup) => {
  emit('configure', modelGroup)
}

const emitDeleteQuantization = (quantization) => {
  emit('delete-quantization', quantization)
}

const emitDeleteGroup = (modelGroup) => {
  emit('delete-group', modelGroup)
}

const formatFileSize = (bytes) => {
  if (!bytes || bytes === 0) return '0 B'
  const k = 1024
  const sizes = ['B', 'KB', 'MB', 'GB', 'TB']
  const i = Math.floor(Math.log(bytes) / Math.log(k))
  return `${(bytes / Math.pow(k, i)).toFixed(2)} ${sizes[i]}`
}

const getUpstreamUrl = (proxyName) => {
  const host = window.location.hostname
  const port = '2000'
  return `http://${host}:${port}/upstream/${proxyName}/`
}

const openUpstreamUrl = (proxyName) => {
  const url = getUpstreamUrl(proxyName)
  window.open(url, '_blank')
}
</script>

<style scoped>
.model-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
  gap: var(--spacing-md);
}

.model-card {
  background: var(--gradient-card);
  border: 1px solid var(--border-primary);
  border-radius: var(--radius-xl);
  padding: var(--spacing-lg);
  transition: all var(--transition-normal);
  box-shadow: var(--shadow-md);
  position: relative;
  overflow: hidden;
  backdrop-filter: blur(10px);
  animation: fadeIn 0.6s ease-out;
}

.model-card::before {
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

.model-card:hover {
  box-shadow: var(--shadow-lg), var(--glow-primary);
  transform: translateY(-5px) scale(1.02);
  border-color: var(--accent-cyan);
}

.model-card:hover::before {
  opacity: 1;
}

.model-card-header {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  margin-bottom: var(--spacing-sm);
}

.model-name {
  font-weight: 700;
  color: var(--text-primary);
  margin-bottom: var(--spacing-sm);
  font-size: 1.1rem;
  line-height: 1.3;
}

.model-tags {
  display: flex;
  gap: var(--spacing-xs);
  flex-wrap: wrap;
}

.model-tag {
  font-size: 0.75rem;
  padding: 2px 8px;
  border-radius: var(--radius-sm);
  background: var(--bg-surface);
  color: var(--text-secondary);
  border: 1px solid var(--border-secondary);
}

.model-tag.tag-type {
  text-transform: capitalize;
}

.model-tag.tag-count {
  background: rgba(14, 165, 233, 0.1);
  color: var(--accent-cyan);
  border-color: rgba(14, 165, 233, 0.2);
}

.model-status {
  display: flex;
  align-items: center;
}

.status-indicator {
  display: flex;
  align-items: center;
  gap: var(--spacing-xs);
  padding: var(--spacing-xs) var(--spacing-sm);
  border-radius: var(--radius-sm);
  font-size: 0.75rem;
  font-weight: 500;
}

.status-running {
  background: rgba(16, 185, 129, 0.1);
  color: var(--accent-green);
  border: 1px solid rgba(16, 185, 129, 0.2);
}

.status-stopped {
  background: var(--bg-surface);
  color: var(--text-secondary);
  border: 1px solid var(--border-secondary);
}

.quantization-list {
  display: flex;
  flex-direction: column;
  gap: var(--spacing-sm);
  margin: var(--spacing-sm) 0;
}

.quantization-item {
  border: 1px solid var(--border-secondary);
  border-radius: var(--radius-lg);
  padding: var(--spacing-sm);
  background: rgba(255, 255, 255, 0.05);
  transition: border-color var(--transition-fast), transform var(--transition-fast);
}

.quantization-item.selected {
  border-color: var(--accent-cyan);
  transform: translateY(-2px);
  box-shadow: var(--shadow-sm);
}

.quantization-info {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: var(--spacing-sm);
}

.quantization-name {
  font-weight: 600;
  color: var(--text-primary);
  display: flex;
  align-items: center;
  gap: var(--spacing-xs);
}

.quantization-details {
  display: flex;
  align-items: center;
  gap: var(--spacing-xs);
  font-size: 0.8rem;
  color: var(--text-secondary);
}

.quantization-size {
  font-weight: 500;
}

.quantization-status.running {
  display: inline-flex;
  align-items: center;
  gap: var(--spacing-xxs);
  padding: 2px 6px;
  border-radius: var(--radius-sm);
  background: rgba(16, 185, 129, 0.15);
  color: var(--accent-green);
  font-size: 0.75rem;
}

.quantization-actions {
  display: flex;
  gap: var(--spacing-xs);
  margin-top: var(--spacing-xs);
}

.upstream-link {
  padding: 0;
  height: auto;
}

.model-actions {
  display: flex;
  justify-content: space-between;
  gap: var(--spacing-sm);
  margin-top: var(--spacing-md);
}

.action-group {
  display: flex;
  gap: var(--spacing-xs);
}
</style>

