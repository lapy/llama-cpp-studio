<template>
  <div class="tab-content">
    <div class="tab-section">
      <h4 class="tab-section-title">
        <i class="pi pi-microchip"></i>
        Model Loading
      </h4>
      <ConfigField 
        v-if="!cpuOnlyMode" 
        label="GPU Layers" 
        :tooltip="gpuLayersTooltip"
        :help-text="`Layers offloaded to GPU (max: ${maxGpuLayers})`"
      >
        <template #input>
          <SliderInput 
            v-model="config.n_gpu_layers" 
            :min="0" 
            :max="maxGpuLayers" 
            :recommended="recommendedGpuLayers" 
            :disabled="!gpuAvailable"
            @input="$emit('update-vram-estimate')" 
          />
        </template>
        <template #validation>
          <div v-if="gpuLayersValidation" class="inline-validation" :class="gpuLayersValidation.type">
            <i :class="gpuLayersValidation.type === 'error' ? 'pi pi-times-circle' : 'pi pi-check-circle'"></i>
            <span>{{ gpuLayersValidation.message }}</span>
          </div>
        </template>
      </ConfigField>
      <ConfigField v-if="!cpuOnlyMode" label="Main GPU" help-text="Primary GPU">
        <template #input>
          <Dropdown 
            v-model="config.main_gpu" 
            :options="gpuOptions" 
            optionLabel="label" 
            optionValue="value"
            placeholder="Select GPU" 
            :disabled="!gpuAvailable" 
          />
        </template>
      </ConfigField>
      <ConfigField v-if="!cpuOnlyMode" label="Tensor Split" help-text="Multi-GPU ratios">
        <template #input>
          <InputText v-model="config.tensor_split" placeholder="0.5,0.5" :disabled="!gpuAvailable" />
        </template>
      </ConfigField>
      <ConfigField label="CPU Threads" help-text="CPU threads for computation">
        <template #input>
          <SliderInput v-model="config.threads" :min="1" :max="cpuThreads" />
        </template>
      </ConfigField>
    </div>
  </div>
</template>

<script setup>
import ConfigField from './ConfigField.vue'
import SliderInput from '@/components/SliderInput.vue'
import InputText from 'primevue/inputtext'
import Dropdown from 'primevue/dropdown'

defineProps({
  config: {
    type: Object,
    required: true
  },
  cpuOnlyMode: {
    type: Boolean,
    default: false
  },
  gpuAvailable: {
    type: Boolean,
    default: true
  },
  maxGpuLayers: {
    type: Number,
    default: 32
  },
  recommendedGpuLayers: {
    type: Number,
    default: null
  },
  cpuThreads: {
    type: Number,
    default: 8
  },
  gpuOptions: {
    type: Array,
    default: () => []
  },
  gpuLayersValidation: {
    type: Object,
    default: null
  },
  gpuLayersTooltip: {
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

.inline-validation.success {
  color: var(--status-success);
}
</style>

