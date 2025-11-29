<template>
  <div class="essential-settings-section">
    <div class="tab-section">
      <h4 class="tab-section-title">
        <i class="pi pi-microchip"></i>
        Model Loading
      </h4>
      <ConfigField 
        v-if="!systemStore.gpuInfo.cpu_only_mode" 
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
      <ConfigField v-if="!systemStore.gpuInfo.cpu_only_mode" label="Main GPU" help-text="Primary GPU">
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
      <ConfigField v-if="!systemStore.gpuInfo.cpu_only_mode" label="Tensor Split" help-text="Multi-GPU ratios">
        <template #input>
          <InputText 
            v-model="config.tensor_split" 
            placeholder="0.5,0.5" 
            :disabled="!gpuAvailable" 
          />
        </template>
      </ConfigField>
      <ConfigField label="CPU Threads" help-text="CPU threads for computation">
        <template #input>
          <SliderInput 
            v-model="config.threads" 
            :min="1" 
            :max="systemStore.gpuInfo.cpu_threads" 
          />
        </template>
      </ConfigField>
    </div>
  </div>
</template>

<script setup>
// PrimeVue
import Dropdown from 'primevue/dropdown'
import InputText from 'primevue/inputtext'

// Components
import ConfigField from '@/components/config/ConfigField.vue'
import SliderInput from '@/components/SliderInput.vue'

// Stores
import { useSystemStore } from '@/stores/system'
import { computed } from 'vue'

const props = defineProps({
  config: {
    type: Object,
    required: true
  },
  maxGpuLayers: {
    type: Number,
    required: true
  },
  recommendedGpuLayers: {
    type: Number,
    default: null
  },
  gpuLayersTooltip: {
    type: String,
    default: ''
  },
  gpuLayersValidation: {
    type: Object,
    default: null
  },
  gpuAvailable: {
    type: Boolean,
    default: true
  }
})

const emit = defineEmits(['update-vram-estimate'])

const systemStore = useSystemStore()

// GPU options
const gpuOptions = computed(() => {
  return Array.from({ length: systemStore.gpuInfo.device_count }, (_, i) => ({
    label: `GPU ${i}`,
    value: i
  }))
})
</script>

<style scoped>
.essential-settings-section {
  display: flex;
  flex-direction: column;
  gap: var(--spacing-lg);
}

.tab-section {
  display: flex;
  flex-direction: column;
  gap: var(--spacing-md);
}

.tab-section-title {
  display: flex;
  align-items: center;
  gap: var(--spacing-sm);
  margin: 0 0 var(--spacing-md) 0;
  font-size: 1rem;
  font-weight: 600;
  color: var(--text-primary);
}

.inline-validation {
  display: flex;
  align-items: center;
  gap: var(--spacing-xs);
  margin-top: var(--spacing-xs);
  font-size: 0.875rem;
}

.inline-validation.warning {
  color: var(--status-warning);
}

.inline-validation.error {
  color: var(--status-error);
}

.inline-validation.success {
  color: var(--status-success);
}
</style>
