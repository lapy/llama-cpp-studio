<template>
  <div class="hf-defaults-card">
    <div class="hf-defaults-header">
      <div class="hf-defaults-title">
        <i class="pi pi-sliders-h"></i>
        <div>
          <strong>Hugging Face Defaults</strong>
          <span>Original repo recommendations</span>
        </div>
      </div>
      <i v-if="loading" class="pi pi-spin pi-spinner hf-defaults-spinner"></i>
    </div>
    <div class="hf-defaults-body">
      <div class="hf-defaults-row" v-if="hfMetadata?.pipeline_tag">
        <span class="hf-defaults-label">Pipeline</span>
        <span class="hf-defaults-value">{{ hfMetadata.pipeline_tag }}</span>
      </div>
      <div class="hf-defaults-row" v-if="hfContextValue">
        <span class="hf-defaults-label">Context Size</span>
        <span class="hf-defaults-value">{{ hfContextValue.toLocaleString() }} tokens</span>
      </div>
      <div class="hf-defaults-grid" v-if="hfDefaultsList.length">
        <div
          v-for="item in hfDefaultsList"
          :key="item.label"
          class="hf-default-chip"
        >
          <span class="chip-label">{{ item.label }}</span>
          <span class="chip-value">{{ item.value }}</span>
        </div>
      </div>
      <div v-if="hfLayerInfoList.length" class="hf-defaults-subtitle">
        Embedding Parameters
      </div>
      <div class="hf-defaults-grid" v-if="hfLayerInfoList.length">
        <div
          v-for="item in hfLayerInfoList"
          :key="`layer-${item.label}`"
          class="hf-default-chip"
        >
          <span class="chip-label">{{ item.label }}</span>
          <span class="chip-value">{{ item.value }}</span>
        </div>
      </div>
      <div class="hf-defaults-row" v-if="hfMetadata?.license">
        <span class="hf-defaults-label">License</span>
        <span class="hf-defaults-value">{{ hfMetadata.license }}</span>
      </div>
    </div>
  </div>
</template>

<script setup>
defineProps({
  hfMetadata: {
    type: Object,
    default: null
  },
  hfContextValue: {
    type: Number,
    default: null
  },
  hfDefaultsList: {
    type: Array,
    default: () => []
  },
  hfLayerInfoList: {
    type: Array,
    default: () => []
  },
  loading: {
    type: Boolean,
    default: false
  }
})
</script>

<style scoped>
.hf-defaults-card {
  margin-top: 1rem;
  padding: 1rem;
  background: var(--bg-surface);
  border: 1px solid var(--border-primary);
  border-radius: var(--radius-lg);
}

.hf-defaults-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 1rem;
}

.hf-defaults-title {
  display: flex;
  align-items: center;
  gap: 0.75rem;
}

.hf-defaults-title i {
  font-size: 1.25rem;
  color: var(--accent-cyan);
}

.hf-defaults-title strong {
  display: block;
  color: var(--text-primary);
  font-size: 0.9rem;
}

.hf-defaults-title span {
  display: block;
  color: var(--text-secondary);
  font-size: 0.75rem;
}

.hf-defaults-spinner {
  color: var(--accent-cyan);
}

.hf-defaults-body {
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
}

.hf-defaults-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0.5rem 0;
  border-bottom: 1px solid var(--border-primary);
}

.hf-defaults-row:last-child {
  border-bottom: none;
}

.hf-defaults-label {
  font-weight: 600;
  color: var(--text-secondary);
  font-size: 0.875rem;
}

.hf-defaults-value {
  color: var(--text-primary);
  font-weight: 500;
  font-size: 0.875rem;
}

.hf-defaults-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
  gap: 0.5rem;
}

.hf-default-chip {
  display: flex;
  flex-direction: column;
  padding: 0.5rem;
  background: var(--bg-card);
  border: 1px solid var(--border-primary);
  border-radius: var(--radius-md);
}

.chip-label {
  font-size: 0.75rem;
  color: var(--text-secondary);
  font-weight: 500;
}

.chip-value {
  font-size: 0.875rem;
  color: var(--text-primary);
  font-weight: 600;
  margin-top: 0.25rem;
}

.hf-defaults-subtitle {
  margin-top: 0.5rem;
  font-size: 0.8rem;
  font-weight: 600;
  color: var(--text-secondary);
  text-transform: uppercase;
  letter-spacing: 0.05em;
}
</style>

