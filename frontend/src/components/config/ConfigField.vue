<template>
  <div class="config-field" :class="{ 'full-width': fullWidth }">
    <div v-if="label" class="label-wrapper">
      <label :for="fieldId" :id="`label-${fieldId}`">{{ label }}</label>
      <SettingsTooltip 
        v-if="tooltip"
        :title="label"
        :description="tooltip.description || ''"
        :when-to-adjust="tooltip.whenToAdjust"
        :tradeoffs="tooltip.tradeoffs || []"
        :recommended="tooltip.recommended"
        :ranges="tooltip.ranges || []"
      />
    </div>
    <div :aria-labelledby="label ? `label-${fieldId}` : undefined" :aria-describedby="getAriaDescribedBy()">
      <slot name="input"></slot>
    </div>
    <small v-if="helpText" :id="`help-${fieldId}`" class="help-text">{{ helpText }}</small>
    <slot name="validation"></slot>
  </div>
</template>

<script setup>
import { computed } from 'vue'
import SettingsTooltip from './SettingsTooltip.vue'

const props = defineProps({
  label: {
    type: String,
    default: null
  },
  helpText: {
    type: String,
    default: null
  },
  fullWidth: {
    type: Boolean,
    default: false
  },
  tooltip: {
    type: Object,
    default: null
  },
  fieldId: {
    type: String,
    default: () => `field-${Math.random().toString(36).substr(2, 9)}`
  }
})

const getAriaDescribedBy = () => {
  const ids = []
  if (props.helpText) {
    ids.push(`help-${props.fieldId}`)
  }
  return ids.length > 0 ? ids.join(' ') : undefined
}
</script>

<style scoped>
.config-field {
  display: flex;
  flex-direction: column;
  gap: var(--spacing-sm);
  min-width: 0;
  width: 100%;
}

.config-field.full-width {
  grid-column: 1 / -1;
}

.label-wrapper {
  display: flex;
  align-items: center;
  gap: var(--spacing-xs);
}

.config-field label {
  font-weight: 500;
  color: var(--text-primary);
  font-size: 0.9rem;
  margin: 0;
}

.config-field small {
  color: var(--text-secondary);
  font-size: 0.75rem;
  line-height: 1.3;
}
</style>

