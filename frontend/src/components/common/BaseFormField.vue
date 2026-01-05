<template>
  <div class="form-field" :class="fieldClass">
    <label v-if="label" :for="inputId">
      {{ label }}
      <span v-if="required" class="required-indicator">*</span>
    </label>
    <slot>
      <!-- Default slot for input component -->
    </slot>
    <small v-if="helpText" class="help-text">{{ helpText }}</small>
    <small v-if="error" class="error-text">{{ error }}</small>
  </div>
</template>

<script setup>
defineProps({
  label: {
    type: String,
    default: ''
  },
  helpText: {
    type: String,
    default: ''
  },
  error: {
    type: String,
    default: ''
  },
  required: {
    type: Boolean,
    default: false
  },
  inputId: {
    type: String,
    default: ''
  },
  fieldClass: {
    type: String,
    default: ''
  }
})
</script>

<style scoped>
.required-indicator {
  color: var(--status-error);
  margin-left: 0.25rem;
}

.help-text {
  color: var(--text-secondary);
  font-size: 0.8rem;
  margin-top: var(--spacing-xs);
}

.error-text {
  color: var(--status-error);
  font-size: 0.8rem;
  margin-top: var(--spacing-xs);
}
</style>

