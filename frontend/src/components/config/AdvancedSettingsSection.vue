<template>
  <div class="tab-content">
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
import ConfigField from './ConfigField.vue'
import InputNumber from 'primevue/inputnumber'
import InputText from 'primevue/inputtext'
import Textarea from 'primevue/textarea'

defineProps({
  config: {
    type: Object,
    required: true
  }
})
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

