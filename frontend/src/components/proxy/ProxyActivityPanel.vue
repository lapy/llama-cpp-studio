<template>
  <PageSection
    title="Recent activity"
    description="Lifecycle events from the native proxy, newest first. Refreshes every few seconds."
  >
    <template #meta>
      <Tag :value="`${activity.length} events`" severity="secondary" />
    </template>
    <ul v-if="activity.length" class="activity-list">
      <li v-for="(item, index) in activityDesc" :key="`${item.type}-${item.created_at}-${index}`">
        <span class="activity-time">{{ formatTime(item.timestamp) }}</span>
        <Tag :value="item.type" :severity="activitySeverity(item.type)" />
        <span class="activity-text">{{ activityText(item) }}</span>
      </li>
    </ul>
    <EmptyState
      v-else
      icon="pi pi-bolt"
      title="No proxy activity recorded"
      description="Lifecycle events appear here as the proxy starts, loads, and drains engines."
    />
  </PageSection>
</template>

<script setup>
import { computed } from 'vue'
import Tag from 'primevue/tag'
import EmptyState from '@/components/common/EmptyState.vue'
import PageSection from '@/components/common/PageSection.vue'
import { formatTime } from '@/utils/proxyRequestPresentation'

const props = defineProps({
  activity: { type: Array, default: () => [] },
})

const activityDesc = computed(() => [...props.activity].reverse())

function activitySeverity(type) {
  return (
    {
      ready: 'success',
      starting: 'info',
      loading_progress: 'info',
      drain_timeout: 'warn',
      vram_warning: 'warn',
      vram_rejected: 'danger',
      vram_unverified: 'info',
      stopped: 'secondary',
    }[type] || 'secondary'
  )
}

function activityText(item) {
  const bits = []
  if (item.model) bits.push(item.model)
  if (item.type === 'loading_progress' && item.loading_progress != null)
    bits.push(`${Math.round(item.loading_progress)}% loaded`)
  if (item.type === 'stopped' && item.draining) bits.push('while draining')
  if (item.type.startsWith('vram_') && item.reason) bits.push(item.reason)
  if (item.pid) bits.push(`pid ${item.pid}`)
  return bits.join(' · ') || item.type
}
</script>

<style scoped>
.activity-list {
  display: grid;
  gap: 0.35rem;
  max-height: 20rem;
  margin: 0;
  padding: 0;
  overflow: auto;
  list-style: none;
}

.activity-list li {
  display: flex;
  align-items: center;
  gap: 0.7rem;
  padding: 0.3rem 0;
  border-bottom: 1px dashed color-mix(in srgb, var(--border-primary) 55%, transparent);
  font-size: var(--text-sm);
}

.activity-time {
  color: var(--text-secondary);
  font-variant-numeric: tabular-nums;
  white-space: nowrap;
}

.activity-text {
  min-width: 0;
  overflow-wrap: anywhere;
}
</style>
