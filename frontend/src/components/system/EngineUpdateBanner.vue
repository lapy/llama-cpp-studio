<template>
  <StateBanner v-if="available" class="engine-update-banner" severity="warning">
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
      >{{ linkLabel }}</a
    >
    <template v-if="showUpdateAction" #actions>
      <Button
        label="Update"
        icon="pi pi-arrow-circle-up"
        severity="success"
        outlined
        size="small"
        :aria-label="latestLabel ? `Update to ${latestLabel}` : 'Update engine'"
        v-tooltip.top="updateTooltip"
        :loading="updating"
        @click="$emit('update')"
      />
    </template>
  </StateBanner>
  <StateBanner v-else-if="checked" class="engine-update-current" severity="success">
    Up to date
    <template v-if="currentLabel"> ({{ currentLabel }})</template>
  </StateBanner>
</template>

<script setup>
import { computed } from 'vue'
import Button from 'primevue/button'
import StateBanner from '@/components/common/StateBanner.vue'

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
  margin-bottom: 0;
  font-size: var(--text-sm);
}

.engine-update-banner__link {
  color: inherit;
  font-weight: 600;
  text-decoration: underline;
}
</style>
