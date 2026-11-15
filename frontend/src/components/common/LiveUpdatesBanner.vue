<template>
  <div v-if="visible" class="state-banner live-updates-banner" role="status" aria-live="polite">
    <i class="pi pi-exclamation-triangle" aria-hidden="true" />
    <span>Live updates paused — retrying. Tasks may still be running.</span>
    <span class="live-updates-banner__actions">
      <button type="button" class="live-updates-banner__retry" @click="retry">Retry</button>
      <button type="button" class="live-updates-banner__dismiss" @click="dismiss">Dismiss</button>
    </span>
  </div>
</template>
<script setup>
import { computed, ref, watch } from 'vue'
import { useProgressStore } from '@/stores/progress'
import { useNow } from '@/composables/useNow'
import { LIVE_UPDATES_PAUSED_DELAY_MS } from '@/utils/liveUpdates'

const progressStore = useProgressStore()
const now = useNow(1000)
const dismissed = ref(false)

const visible = computed(() => {
  if (dismissed.value) return false
  if (progressStore.connectionState === 'live') return false
  if (!progressStore.hasActiveTasks) return false
  const since = progressStore.disconnectedSince
  if (since == null) return false
  return now.value - since >= LIVE_UPDATES_PAUSED_DELAY_MS
})

// A successful reconnect clears the outage (and a previous Dismiss), so the
// next outage starts from a clean slate.
watch(
  () => progressStore.connectionState,
  (state) => {
    if (state === 'live') dismissed.value = false
  },
)

function retry() {
  progressStore.retryConnection()
}

function dismiss() {
  dismissed.value = true
}
</script>
<style scoped>
.live-updates-banner {
  margin: 0 0 0.75rem;
}
.live-updates-banner__actions {
  display: inline-flex;
  gap: 0.5rem;
  margin-left: auto;
}
.live-updates-banner__actions button {
  min-height: 1.9rem;
  padding: 0 0.55rem;
  border: 1px solid currentcolor;
  border-radius: var(--radius-xs);
  background: transparent;
  color: var(--text-primary);
  font-size: 0.85rem;
  font-weight: 600;
  cursor: pointer;
}
.live-updates-banner__actions button:hover {
  background: var(--bg-card-hover);
}
</style>
