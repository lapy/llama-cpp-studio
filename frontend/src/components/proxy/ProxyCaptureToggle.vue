<template>
  <label class="proxy-checkbox-field">
    <span class="proxy-checkbox-field__control">
      <input
        :checked="modelValue"
        type="checkbox"
        data-capture-request-payloads
        @change="emit('update:modelValue', $event.target.checked)"
      />
      <span>Capture request payloads for replay</span>
    </span>
    <small class="hint">
      Memory only, last 200 requests; bodies are capped at 256 KiB and sensitive headers are
      redacted. Never written to history. Applies after a proxy restart.
    </small>
  </label>
</template>

<script setup>
defineProps({
  modelValue: { type: Boolean, default: false },
})
const emit = defineEmits(['update:modelValue'])
</script>

<style scoped>
.proxy-checkbox-field {
  display: flex;
  flex-direction: column;
  gap: 0.3rem;
  grid-column: 1 / -1;
}

.proxy-checkbox-field__control {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  min-height: 2.125rem;
  font-size: 0.8125rem;
}

.proxy-checkbox-field__control input {
  width: 1rem;
  height: 1rem;
  accent-color: var(--accent-cyan);
}

@media (max-width: 640px), (pointer: coarse) {
  .proxy-checkbox-field__control {
    min-height: 2.75rem;
  }
}
</style>
