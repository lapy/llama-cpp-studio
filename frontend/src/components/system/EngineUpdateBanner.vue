<template>
  <div v-if="available" class="engine-update-banner">
    <i class="pi pi-arrow-up-right" aria-hidden="true" />
    <span class="engine-update-banner__copy">
      <slot name="message">
        Update available:
        <strong>{{ latestLabel }}</strong>
      </slot>
      <a
        v-if="linkUrl"
        :href="linkUrl"
        target="_blank"
        rel="noopener noreferrer"
        class="engine-update-banner__link"
      >{{ linkLabel }}</a>
    </span>
    <Button
      v-if="showUpdateAction"
      icon="pi pi-arrow-circle-up"
      text
      severity="success"
      size="small"
      v-tooltip.top="updateTooltip"
      :loading="updating"
      @click="$emit('update')"
    />
  </div>
  <div v-else-if="checked" class="engine-update-current">
    <i class="pi pi-check" aria-hidden="true" />
    Up to date
    <template v-if="currentLabel"> ({{ currentLabel }})</template>
  </div>
</template>

<script setup>
import { computed } from 'vue'
import Button from 'primevue/button'

const props = defineProps({
  /** True when an update is available. */
  available: { type: Boolean, default: false },
  /** True after a successful check (shows “Up to date” when not available). */
  checked: { type: Boolean, default: false },
  latestVersion: { type: [String, Number], default: '' },
  currentVersion: { type: [String, Number], default: '' },
  linkUrl: { type: String, default: '' },
  linkLabel: { type: String, default: 'View release' },
  showUpdateAction: { type: Boolean, default: true },
  updating: { type: Boolean, default: false },
  updateTooltip: {
    type: String,
    default: 'Update using saved build settings',
  },
})

defineEmits(['update'])

const latestLabel = computed(() => String(props.latestVersion || '').trim())
const currentLabel = computed(() => String(props.currentVersion || '').trim())
</script>

<style scoped>
.engine-update-banner,
.engine-update-current {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 0.5rem 0.75rem;
  padding: 0.65rem 0.85rem;
  border-radius: var(--radius-lg);
  font-size: 0.875rem;
  margin-bottom: 0.65rem;
}

.engine-update-banner {
  background: var(--status-warning-soft, rgba(245, 158, 11, 0.1));
  border: 1px solid rgba(245, 158, 11, 0.3);
  color: var(--status-warning, #f59e0b);
}

.engine-update-current {
  background: var(--status-success-soft, rgba(16, 185, 129, 0.1));
  border: 1px solid rgba(16, 185, 129, 0.3);
  color: var(--status-success, #10b981);
}

.engine-update-banner__copy {
  flex: 1;
  min-width: 12rem;
}

.engine-update-banner__link {
  margin-left: 0.35rem;
  color: var(--accent-cyan, #60a5fa);
  text-decoration: underline;
}
</style>
