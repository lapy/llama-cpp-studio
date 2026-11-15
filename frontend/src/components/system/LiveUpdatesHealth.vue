<template>
  <div class="live-updates-health">
    <dt>Live updates</dt>
    <dd>
      <span class="live-updates-health__state">
        <i :class="['pi', statusIcon, statusClass]" aria-hidden="true" />
        <span role="status" aria-live="polite">{{ presentation.stateLabel }}</span>
      </span>
      <span class="live-updates-health__detail">{{ presentation.eventAgeLabel }}</span>
      <span class="live-updates-health__detail">{{ presentation.attemptLabel }}</span>
      <button
        type="button"
        class="live-updates-health__retry"
        aria-label="Retry live updates"
        @click="retryConnection"
      >
        Retry
      </button>
    </dd>
  </div>
</template>
<script setup>
import { computed } from 'vue'
import { useProgressStore } from '@/stores/progress'
import { useNow } from '@/composables/useNow'
import { liveConnectionPresentation } from '@/utils/liveUpdates'

const progressStore = useProgressStore()
const now = useNow(1000)

const presentation = computed(() =>
  liveConnectionPresentation(
    progressStore.connectionState,
    progressStore.reconnectAttempt,
    progressStore.lastEventAt,
    now.value,
  ),
)

const statusIcon = computed(() => {
  if (presentation.value.isLive) return 'pi-check-circle'
  return progressStore.connectionState === 'offline' ? 'pi-exclamation-circle' : 'pi-clock'
})

const statusClass = computed(() => {
  if (presentation.value.isLive) return 'live-updates-health__icon--ok'
  return progressStore.connectionState === 'offline'
    ? 'live-updates-health__icon--error'
    : 'live-updates-health__icon--warn'
})

function retryConnection() {
  progressStore.retryConnection()
}
</script>
<style scoped>
.live-updates-health dt {
  color: var(--text-secondary);
  font-size: 0.85rem;
}
.live-updates-health dd {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 0.35rem 0.6rem;
  margin: 0.35rem 0 0;
}
.live-updates-health__state {
  display: inline-flex;
  align-items: center;
  gap: 0.35rem;
  font-weight: 600;
}
.live-updates-health__detail {
  color: var(--text-secondary);
  font-size: 0.85rem;
}
.live-updates-health__retry {
  min-height: 1.9rem;
  padding: 0 0.55rem;
  border: 1px solid var(--border-primary);
  border-radius: var(--radius-xs);
  background: transparent;
  color: var(--text-primary);
  font-size: 0.85rem;
  cursor: pointer;
}
.live-updates-health__retry:hover {
  background: var(--bg-card-hover);
}
.live-updates-health__icon--ok {
  color: var(--status-success);
}
.live-updates-health__icon--warn {
  color: var(--status-warning);
}
.live-updates-health__icon--error {
  color: var(--status-error);
}
</style>
