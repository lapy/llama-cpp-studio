<template>
  <div class="tab-content">
    <div class="tab-section">
      <h4 class="tab-section-title">
        <i class="pi pi-memory"></i>
        Context & Memory
      </h4>
      <ConfigField 
        label="Context Size" 
        :tooltip="contextSizeTooltip"
        :help-text="`Max context length (max: ${maxContextSize.toLocaleString()})`"
      >
        <template #input>
          <SliderInput 
            v-model="config.ctx_size" 
            :min="512" 
            :max="maxContextSize" 
            :recommended="recommendedContextSize" 
            @input="$emit('update-vram-estimate')" 
          />
        </template>
        <template #validation>
          <div v-if="contextSizeValidation" class="inline-validation" :class="contextSizeValidation.type">
            <i class="pi pi-exclamation-triangle"></i>
            <span>{{ contextSizeValidation.message }}</span>
          </div>
        </template>
      </ConfigField>
      <ConfigField 
        label="Batch Size" 
        :tooltip="batchSizeTooltip"
        :help-text="`Parallel tokens (max: ${maxBatchSize})`"
      >
        <template #input>
          <SliderInput 
            v-model="config.batch_size" 
            :min="1" 
            :max="maxBatchSize" 
            :recommended="recommendedBatchSize" 
            @input="$emit('update-vram-estimate')" 
          />
        </template>
        <template #validation>
          <div v-if="batchSizeValidation" class="inline-validation" :class="batchSizeValidation.type">
            <i class="pi pi-exclamation-triangle"></i>
            <span>{{ batchSizeValidation.message }}</span>
          </div>
        </template>
      </ConfigField>
      <ConfigField label="U-Batch Size" :help-text="`Unified batch (max: ${maxBatchSize})`">
        <template #input>
          <SliderInput v-model="config.ubatch_size" :min="1" :max="maxBatchSize" />
        </template>
      </ConfigField>
      <ConfigField label="No Memory Map" help-text="Disable mmap">
        <template #input>
          <Checkbox v-model="config.no_mmap" binary />
        </template>
      </ConfigField>
      <ConfigField label="Mlock" help-text="Lock model in RAM (prevent swapping)">
        <template #input>
          <Checkbox v-model="config.mlock" binary />
        </template>
      </ConfigField>
    </div>
  </div>
</template>

<script setup>
import ConfigField from './ConfigField.vue'
import SliderInput from '@/components/SliderInput.vue'
import Checkbox from 'primevue/checkbox'

defineProps({
  config: {
    type: Object,
    required: true
  },
  maxContextSize: {
    type: Number,
    default: 131072
  },
  maxBatchSize: {
    type: Number,
    default: 512
  },
  recommendedContextSize: {
    type: Number,
    default: null
  },
  recommendedBatchSize: {
    type: Number,
    default: null
  },
  contextSizeValidation: {
    type: Object,
    default: null
  },
  batchSizeValidation: {
    type: Object,
    default: null
  },
  contextSizeTooltip: {
    type: String,
    default: ''
  },
  batchSizeTooltip: {
    type: String,
    default: ''
  }
})

defineEmits(['update-vram-estimate'])
</script>

<style scoped>
.tab-content {
  display: flex;
  flex-direction: column;
  gap: 1.5rem;
}

.tab-section {
  display: flex;
  flex-direction: column;
  gap: 1rem;
}

.tab-section-title {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  margin: 0 0 0.75rem 0;
  color: var(--text-primary);
  font-size: 1.1rem;
  font-weight: 600;
}

.tab-section-title i {
  color: var(--accent-cyan);
}

.inline-validation {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  margin-top: 0.5rem;
  font-size: 0.875rem;
}

.inline-validation.error {
  color: var(--status-error);
}

.inline-validation.warning {
  color: var(--status-warning);
}

.inline-validation.success {
  color: var(--status-success);
}
</style>

