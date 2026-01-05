<template>
  <div class="tab-content">
    <div class="tab-section">
      <h4 class="tab-section-title">
        <i class="pi pi-cog"></i>
        Sampling Parameters
      </h4>
      <ConfigField label="Max Predict" help-text="Max tokens (-1=unlimited)">
        <template #input>
          <InputNumber v-model="config.n_predict" :min="-1" :max="2048" />
        </template>
      </ConfigField>
      <ConfigField 
        label="Temperature" 
        :tooltip="temperatureTooltip"
        :help-text="getTemperatureTooltip()"
      >
        <template #input>
          <SliderInput 
            v-model="config.temp" 
            :min="0.1" 
            :max="2.0" 
            :step="0.1" 
            :maxFractionDigits="1"
            :markers="[
              { value: 0.3, label: 'Code', color: 'blue' },
              { value: 0.8, label: 'Chat', color: 'green' },
              { value: 1.5, label: 'Creative', color: 'purple' }
            ]"
            :recommended="recommendedTemperature"
          />
        </template>
      </ConfigField>
      <ConfigField 
        label="Top-K" 
        :tooltip="topKTooltip"
        :help-text="getTopKTooltip()"
      >
        <template #input>
          <SliderInput 
            v-model="config.top_k" 
            :min="1" 
            :max="maxTopK"
            :recommended="recommendedTopK"
          />
        </template>
      </ConfigField>
      <ConfigField 
        label="Top-P" 
        :tooltip="topPTooltip"
        :help-text="getTopPTooltip()"
      >
        <template #input>
          <SliderInput 
            v-model="config.top_p" 
            :min="0.1" 
            :max="1.0" 
            :step="0.1" 
            :maxFractionDigits="1"
            :recommended="recommendedTopP"
          />
        </template>
      </ConfigField>
      <ConfigField 
        label="Repeat Penalty" 
        :tooltip="repeatPenaltyTooltip"
        :help-text="getRepeatPenaltyTooltip()"
      >
        <template #input>
          <SliderInput 
            v-model="config.repeat_penalty" 
            :min="0.5" 
            :max="2.0" 
            :step="0.05"
            :maxFractionDigits="2"
            :recommended="null"
          />
        </template>
      </ConfigField>
    </div>
    <div class="tab-section">
      <h4 class="tab-section-title">
        <i class="pi pi-sliders-h"></i>
        Advanced Generation Options
      </h4>
      <ConfigField v-if="isMinPSupported" label="Min-P">
        <template #input>
          <SliderInput v-model="config.min_p" :min="0.0" :max="1.0" :step="0.05" :maxFractionDigits="2" />
        </template>
      </ConfigField>
      <ConfigField v-if="isTypicalPSupported" label="Typical-P">
        <template #input>
          <SliderInput v-model="config.typical_p" :min="0.0" :max="1.0" :step="0.05" :maxFractionDigits="2" />
        </template>
      </ConfigField>
      <ConfigField v-if="isTfsZSupported" label="TFS-Z">
        <template #input>
          <SliderInput v-model="config.tfs_z" :min="0.0" :max="1.0" :step="0.05" :maxFractionDigits="2" />
        </template>
      </ConfigField>
      <ConfigField v-if="isPresencePenaltySupported" label="Presence Penalty">
        <template #input>
          <SliderInput v-model="config.presence_penalty" :min="0.0" :max="2.0" :step="0.1"
            :maxFractionDigits="1" />
        </template>
      </ConfigField>
      <ConfigField v-if="isFrequencyPenaltySupported" label="Frequency Penalty">
        <template #input>
          <SliderInput v-model="config.frequency_penalty" :min="0.0" :max="2.0" :step="0.1"
            :maxFractionDigits="1" />
        </template>
      </ConfigField>
      <ConfigField label="Mirostat Mode">
        <template #input>
          <Dropdown v-model="config.mirostat"
            :options="[{ label: 'Off (0)', value: 0 }, { label: 'Mirostat (1)', value: 1 }, { label: 'Mirostat 2.0 (2)', value: 2 }]"
            optionLabel="label" optionValue="value" />
        </template>
      </ConfigField>
      <ConfigField v-if="config.mirostat > 0" label="Mirostat Eta">
        <template #input>
          <InputNumber v-model="config.mirostat_eta" :min="0" :max="10" :step="0.1" />
        </template>
      </ConfigField>
      <ConfigField v-if="config.mirostat > 0" label="Mirostat Tau">
        <template #input>
          <InputNumber v-model="config.mirostat_tau" :min="0" :max="10" :step="0.1" />
        </template>
      </ConfigField>
    </div>
  </div>
</template>

<script setup>
import { computed } from 'vue'
import ConfigField from './ConfigField.vue'
import SliderInput from '@/components/SliderInput.vue'
import InputNumber from 'primevue/inputnumber'
import Dropdown from 'primevue/dropdown'

const props = defineProps({
  config: {
    type: Object,
    required: true
  },
  maxTopK: {
    type: Number,
    default: 200
  },
  recommendedTemperature: {
    type: Number,
    default: null
  },
  recommendedTopK: {
    type: Number,
    default: null
  },
  recommendedTopP: {
    type: Number,
    default: null
  },
  isMinPSupported: {
    type: Boolean,
    default: false
  },
  isTypicalPSupported: {
    type: Boolean,
    default: false
  },
  isTfsZSupported: {
    type: Boolean,
    default: false
  },
  isPresencePenaltySupported: {
    type: Boolean,
    default: false
  },
  isFrequencyPenaltySupported: {
    type: Boolean,
    default: false
  },
  temperatureTooltip: {
    type: String,
    default: ''
  },
  topKTooltip: {
    type: String,
    default: ''
  },
  topPTooltip: {
    type: String,
    default: ''
  },
  repeatPenaltyTooltip: {
    type: String,
    default: ''
  }
})

const getTemperatureTooltip = () => {
  return props.temperatureTooltip || 'Controls randomness. Lower = more deterministic, Higher = more creative'
}

const getTopKTooltip = () => {
  return props.topKTooltip || 'Consider top K tokens. Lower = more focused, Higher = more diverse'
}

const getTopPTooltip = () => {
  return props.topPTooltip || 'Nucleus sampling. Lower = more focused, Higher = more diverse'
}

const getRepeatPenaltyTooltip = () => {
  return props.repeatPenaltyTooltip || 'Penalize repetition. Higher = less repetition'
}
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
</style>

