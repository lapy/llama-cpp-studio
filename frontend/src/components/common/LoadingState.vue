<template>
  <div
    class="page-loading"
    :class="{ 'page-loading--inline': inline }"
    role="status"
    aria-live="polite"
  >
    <ProgressSpinner
      :class="small ? 'page-loading__spinner page-loading__spinner--sm' : 'page-loading__spinner'"
      :strokeWidth="small ? '4' : undefined"
    />
    <span class="page-loading__text">{{ message }}</span>
  </div>
</template>

<script setup>
import ProgressSpinner from 'primevue/progressspinner'

defineProps({
  message: {
    type: String,
    default: 'Loading…',
  },
  /** Horizontal layout (e.g. search “Searching…”) */
  inline: {
    type: Boolean,
    default: false,
  },
  small: {
    type: Boolean,
    default: false,
  },
})
</script>
