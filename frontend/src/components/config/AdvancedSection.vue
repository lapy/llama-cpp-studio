<template>
  <div class="advanced-section">
    <div class="tab-section">
      <h4 class="tab-section-title">
        <i class="pi pi-wrench"></i>
        RoPE & YARN Settings
      </h4>
      <ConfigField label="RoPE Freq Base" help-text="RoPE frequency base">
        <template #input>
          <InputNumber v-model="config.rope_freq_base" :min="0" :max="100000" />
        </template>
      </ConfigField>
      <ConfigField label="RoPE Freq Scale" help-text="RoPE frequency scale">
        <template #input>
          <InputNumber v-model="config.rope_freq_scale" :min="0" :max="100" :step="0.1" :maxFractionDigits="1" />
        </template>
      </ConfigField>
      <ConfigField label="YARN Ext Factor" help-text="YARN extension factor">
        <template #input>
          <InputNumber v-model="config.yarn_ext_factor" :min="0" :max="100" :step="0.1" :maxFractionDigits="1" />
        </template>
      </ConfigField>
      <ConfigField label="YARN Attn Factor" help-text="YARN attention factor">
        <template #input>
          <InputNumber v-model="config.yarn_attn_factor" :min="0" :max="100" :step="0.1" :maxFractionDigits="1" />
        </template>
      </ConfigField>
      <ConfigField label="RoPE Scaling" help-text="RoPE scaling type">
        <template #input>
          <InputText v-model="config.rope_scaling" placeholder="linear, yarn" />
        </template>
      </ConfigField>
      <ConfigField label="YAML Config" help-text="Extra YAML config" full-width>
        <template #input>
          <Textarea v-model="config.yaml" rows="3" placeholder="Additional YAML configuration" />
        </template>
      </ConfigField>
    </div>
  </div>
</template>

<script setup>
// PrimeVue
import InputNumber from 'primevue/inputnumber'
import InputText from 'primevue/inputtext'
import Textarea from 'primevue/textarea'

// Components
import ConfigField from '@/components/config/ConfigField.vue'

const props = defineProps({
  config: {
    type: Object,
    required: true
  }
})
</script>

<style scoped>
.advanced-section {
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
</style>
