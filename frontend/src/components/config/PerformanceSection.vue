<template>
  <div class="performance-section">
    <div class="tab-section">
      <h4 class="tab-section-title">
        <i class="pi pi-tachometer"></i>
        Performance Tuning
      </h4>
      <ConfigField label="Batch Threads" help-text="Threads for batch processing">
        <template #input>
          <SliderInput v-model="config.threads_batch" :min="1" :max="systemStore.gpuInfo.cpu_threads" />
        </template>
      </ConfigField>
      <ConfigField label="Parallel" :help-text="`Parallel processing (max: ${maxParallel})`">
        <template #input>
          <SliderInput v-model="config.parallel" :min="1" :max="maxParallel" />
        </template>
        <template #validation>
          <div v-if="parallelValidation" class="inline-validation" :class="parallelValidation.type">
            <i class="pi pi-exclamation-triangle"></i>
            <span>{{ parallelValidation.message }}</span>
          </div>
        </template>
      </ConfigField>
      <ConfigField v-if="!systemStore.gpuInfo.cpu_only_mode" label="Flash Attention" 
                  help-text="Enable flash attn (enables V cache quantization)">
        <template #input>
          <Checkbox v-model="config.flash_attn" binary :disabled="!gpuAvailable" />
        </template>
      </ConfigField>
      <ConfigField v-if="!systemStore.gpuInfo.cpu_only_mode" label="Low VRAM" 
                  help-text="Optimize for low VRAM usage">
        <template #input>
          <Checkbox v-model="config.low_vram" binary :disabled="!gpuAvailable" />
        </template>
      </ConfigField>
      <ConfigField label="Continuous Batching" help-text="Enable continuous/dynamic batching">
        <template #input>
          <Checkbox v-model="config.cont_batching" binary />
        </template>
      </ConfigField>
      <ConfigField label="No KV Offload" help-text="Disable KV cache offloading">
        <template #input>
          <Checkbox v-model="config.no_kv_offload" binary />
        </template>
      </ConfigField>
      <ConfigField label="Logits All" help-text="Return logits for all tokens">
        <template #input>
          <Checkbox v-model="config.logits_all" binary />
        </template>
      </ConfigField>
      <ConfigField label="Embedding Mode" help-text="Enable embedding generation mode">
        <template #input>
          <Checkbox v-model="config.embedding" binary :disabled="isEmbeddingModel" />
        </template>
      </ConfigField>
    </div>
    
    <div class="tab-section">
      <h4 class="tab-section-title">
        <i class="pi pi-database"></i>
        KV Cache Optimization
      </h4>
      <div v-if="!config.flash_attn && (config.cache_type_v && config.cache_type_v !== 'f16')"
        class="flash-attention-warning">
        <i class="pi pi-exclamation-triangle"></i>
        <div class="warning-content">
          <strong>Flash Attention Required</strong>
          <p>V cache quantization requires llama.cpp compiled with Flash Attention support (flag:
            -DGGML_CUDA_FA_ALL_QUANTS=ON). Recompile your llama.cpp version or disable V cache quantization.</p>
        </div>
      </div>
      <ConfigField label="K Cache Type" help-text="Key cache quantization (reduces memory usage)">
        <template #input>
          <Dropdown v-model="config.cache_type_k" :options="kvCacheOptions" optionLabel="label"
            optionValue="value" placeholder="Select K cache type" />
        </template>
      </ConfigField>
      <ConfigField 
        v-if="config.flash_attn && !systemStore.gpuInfo.cpu_only_mode && isCacheTypeVSupported"
        label="V Cache Type" 
        help-text="Value cache quantization (requires Flash Attention)"
      >
        <template #input>
          <Dropdown v-model="config.cache_type_v" :options="kvCacheOptions" optionLabel="label"
            optionValue="value" placeholder="Select V cache type" />
        </template>
      </ConfigField>
    </div>
    
    <div v-if="modelLayerInfo?.is_moe" class="tab-section">
      <h4 class="tab-section-title">
        <i class="pi pi-sitemap"></i>
        MoE Expert Offloading
      </h4>
      <ConfigField label="Offload Pattern" help-text="Control which MoE layers go to CPU/GPU">
        <template #input>
          <Dropdown v-model="config.moe_offload_pattern" :options="moeOffloadPatterns" optionLabel="label"
            optionValue="value" @change="handleMoEPatternChange" />
        </template>
      </ConfigField>
      <ConfigField label="Custom Offload Pattern" full-width
                  help-text="Advanced regex pattern for -ot parameter (leave empty to use pattern above)">
        <template #input>
          <InputText v-model="config.moe_offload_custom" placeholder="e.g., .ffn_.*_exps.=CPU" />
        </template>
      </ConfigField>
      <ConfigField label="Expert Info">
        <template #input>
          <div class="expert-info">
            <span>{{ modelLayerInfo.expert_count }} experts</span>
            <span>·</span>
            <span>{{ modelLayerInfo.experts_used_count }} active per token</span>
          </div>
        </template>
      </ConfigField>
    </div>
  </div>
</template>

<script setup>
// Vue
import { computed } from 'vue'

// PrimeVue
import Checkbox from 'primevue/checkbox'
import Dropdown from 'primevue/dropdown'
import InputText from 'primevue/inputtext'

// Components
import ConfigField from '@/components/config/ConfigField.vue'
import SliderInput from '@/components/SliderInput.vue'

// Stores
import { useSystemStore } from '@/stores/system'

const props = defineProps({
  config: {
    type: Object,
    required: true
  },
  modelLayerInfo: {
    type: Object,
    default: null
  },
  maxParallel: {
    type: Number,
    required: true
  },
  parallelValidation: {
    type: Object,
    default: null
  },
  gpuAvailable: {
    type: Boolean,
    default: true
  },
  isEmbeddingModel: {
    type: Boolean,
    default: false
  },
  isCacheTypeVSupported: {
    type: Boolean,
    default: false
  }
})

const systemStore = useSystemStore()

// KV cache options
const kvCacheOptions = [
  { label: 'No quantization (use llama.cpp default)', value: null },
  { label: 'FP32 (full precision)', value: 'f32' },
  { label: 'FP16 (half precision)', value: 'f16' },
  { label: 'BF16 (bfloat16)', value: 'bf16' },
  { label: 'Q8_0 (8-bit)', value: 'q8_0' },
  { label: 'Q5_1 (5-bit high quality)', value: 'q5_1' },
  { label: 'Q5_0 (5-bit)', value: 'q5_0' },
  { label: 'Q4_1 (4-bit high quality)', value: 'q4_1' },
  { label: 'Q4_0 (4-bit)', value: 'q4_0' },
  { label: 'IQ4_NL (4-bit non-linear)', value: 'iq4_nl' }
]

// MoE offload patterns
const moeOffloadPatterns = [
  { label: 'None', value: 'none' },
  { label: 'All MoE layers to CPU', value: 'all' },
  { label: 'Up/Down projections to CPU', value: 'up_down' },
  { label: 'Up projection only to CPU', value: 'up' },
  { label: 'Custom pattern', value: 'custom' }
]

// Handle MoE pattern change
const handleMoEPatternChange = () => {
  // Automatically set the custom pattern based on selection
  switch (props.config.moe_offload_pattern) {
    case 'all':
      props.config.moe_offload_custom = '.ffn_.*_exps.=CPU'
      break
    case 'up_down':
      props.config.moe_offload_custom = '.ffn_(up|down)_exps.=CPU'
      break
    case 'up':
      props.config.moe_offload_custom = '.ffn_(up)_exps.=CPU'
      break
    case 'none':
      props.config.moe_offload_custom = ''
      break
    case 'custom':
      // User will input custom pattern
      break
  }
}
</script>

<style scoped>
.performance-section {
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

.flash-attention-warning {
  display: flex;
  gap: var(--spacing-sm);
  padding: var(--spacing-md);
  background: var(--status-warning-soft);
  border: 1px solid var(--status-warning);
  border-radius: var(--radius-md);
  color: var(--status-warning);
  margin-bottom: var(--spacing-md);
}

.flash-attention-warning i {
  font-size: 1.25rem;
  flex-shrink: 0;
}

.warning-content {
  flex: 1;
}

.warning-content strong {
  display: block;
  margin-bottom: var(--spacing-xs);
}

.warning-content p {
  margin: 0;
  font-size: 0.875rem;
  line-height: 1.5;
}

.expert-info {
  display: flex;
  align-items: center;
  gap: var(--spacing-sm);
  color: var(--text-secondary);
  font-size: 0.875rem;
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
</style>
