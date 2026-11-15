<template>
  <footer class="layout-footer">
    <div class="footer-content">
      <span class="footer-version">Studio v{{ appVersion }}</span>
      <span class="live-status" role="group" aria-label="Live updates">
        <i :class="['pi', statusIcon, statusClass]" aria-hidden="true" />
        <button
          v-if="progressStore.connectionState === 'offline'"
          type="button"
          class="live-status__retry"
          @click="retryConnection"
        >
          {{ presentation.statusText }}
        </button>
        <template v-else-if="!presentation.isLive">
          <span class="live-status__text">{{ presentation.statusText }}</span>
          <button
            type="button"
            class="live-status__retry"
            aria-label="Retry live updates"
            @click="retryConnection"
          >
            Retry
          </button>
        </template>
        <span v-else class="live-status__text">{{ presentation.statusText }}</span>
        <span class="sr-only" role="status" aria-live="polite">{{
          presentation.stateAnnouncement
        }}</span>
      </span>
      <span v-if="needsAttention" class="footer-attention">Needs attention</span>
    </div>
  </footer>
</template>
<script setup>
import { computed } from 'vue'
import { useEnginesStore } from '@/stores/engines'
import { useProgressStore } from '@/stores/progress'
import { useNow } from '@/composables/useNow'
import { liveConnectionPresentation } from '@/utils/liveUpdates'

const progressStore = useProgressStore()
const systemStore = useEnginesStore()
const now = useNow(1000)
const appVersion = typeof __APP_VERSION__ !== 'undefined' ? __APP_VERSION__ : '1.0.0'

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
  if (presentation.value.isLive) return 'footer-status--ok'
  return progressStore.connectionState === 'offline'
    ? 'footer-status--error'
    : 'footer-status--warn'
})

const needsAttention = computed(() =>
  Boolean(
    systemStore.systemStatus?.persistence?.saturated ||
    systemStore.systemStatus?.persistence?.latest_failure,
  ),
)

function retryConnection() {
  progressStore.retryConnection()
}
</script>
<style scoped>
.footer-content {
  justify-content: flex-start;
  flex-direction: row;
  align-items: center;
  flex-wrap: nowrap;
  gap: 0.65rem;
}
.footer-version {
  min-width: 0;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  color: var(--text-secondary);
}
.live-status {
  display: inline-flex;
  align-items: center;
  gap: 0.35rem;
  margin-left: auto;
  color: var(--text-secondary);
}
.live-status__text {
  white-space: nowrap;
}
.live-status__retry {
  min-height: 1.75rem;
  padding: 0 0.45rem;
  border: 1px solid currentcolor;
  border-radius: var(--radius-xs);
  background: transparent;
  color: var(--text-secondary);
  font-size: var(--text-xs);
  font-weight: 600;
  line-height: 1.2;
  cursor: pointer;
}
.live-status__retry:hover {
  background: var(--bg-card-hover);
  color: var(--text-primary);
}
.footer-status--ok {
  color: var(--status-success);
}
.footer-status--warn,
.footer-attention {
  color: var(--status-warning);
}
.footer-status--error {
  color: var(--status-error);
}

.footer-attention {
  flex-shrink: 0;
  white-space: nowrap;
  font-weight: 600;
}

@media (max-width: 900px) {
  .layout-footer {
    padding: 0.45rem 0.75rem;
  }
  .footer-content {
    font-size: var(--text-xs);
  }
}
</style>
