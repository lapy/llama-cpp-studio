<template>
  <StateBanner
    :severity="severity"
    icon="pi-key"
    class="huggingface-token-notice"
    :message="message"
  >
    <template #actions>
      <Button :label="actionLabel" size="small" text @click="$emit('open')" />
      <Button
        v-if="dismissible"
        label="Dismiss"
        size="small"
        text
        severity="secondary"
        @click="$emit('dismiss')"
      />
    </template>
  </StateBanner>
</template>

<script setup>
import Button from 'primevue/button'
import StateBanner from '@/components/common/StateBanner.vue'

defineProps({
  message: {
    type: String,
    default: 'A Hugging Face token is required for gated models.',
  },
  actionLabel: {
    type: String,
    default: 'Set token',
  },
  severity: {
    type: String,
    default: 'info',
    validator: (value) => ['info', 'warning', 'error', 'success'].includes(value),
  },
  /** Search shows a per-session dismiss control; the library does not. */
  dismissible: {
    type: Boolean,
    default: false,
  },
})

defineEmits(['open', 'dismiss'])
</script>
