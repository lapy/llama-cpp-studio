<template>
  <div
    :class="['state-banner', `state-banner--${severity}`]"
    :role="resolvedRole"
    :aria-live="resolvedRole === 'status' ? 'polite' : undefined"
  >
    <i v-if="resolvedIcon" :class="['pi', resolvedIcon, 'state-banner__icon']" aria-hidden="true" />
    <div class="state-banner__message">
      <slot>{{ message }}</slot>
    </div>
    <div v-if="$slots.actions" class="state-banner__actions">
      <slot name="actions" />
    </div>
  </div>
</template>

<script setup>
import { computed } from 'vue'

const ICONS = {
  info: 'pi-info-circle',
  success: 'pi-check-circle',
  warning: 'pi-exclamation-triangle',
  error: 'pi-exclamation-circle',
}

const props = defineProps({
  severity: {
    type: String,
    default: 'warning',
    validator: (value) => ['info', 'success', 'warning', 'error'].includes(value),
  },
  message: {
    type: String,
    default: '',
  },
  /** Override the default icon; pass an empty string to hide it. */
  icon: {
    type: String,
    default: undefined,
  },
  /**
   * Announcement semantics. Errors are alert by default; everything else is
   * a polite status. Pass "alert" for a failure that must interrupt, or
   * "status" for a recoverable warning that can wait.
   */
  role: {
    type: String,
    default: '',
    validator: (value) => ['', 'status', 'alert', 'note', 'none'].includes(value),
  },
})

const resolvedRole = computed(() => props.role || (props.severity === 'error' ? 'alert' : 'status'))
const resolvedIcon = computed(() => (props.icon === undefined ? ICONS[props.severity] : props.icon))
</script>
