<template>
  <div class="custom-args-section">
    <ConfigField label="Custom Arguments" help-text="Additional command-line arguments" full-width>
      <template #input>
        <Textarea v-model="config.customArgs" rows="4" placeholder="Enter custom llama.cpp arguments..." />
      </template>
    </ConfigField>
  </div>
</template>

<script setup>
// PrimeVue
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
.custom-args-section {
  display: flex;
  flex-direction: column;
  gap: var(--spacing-md);
}
</style>
