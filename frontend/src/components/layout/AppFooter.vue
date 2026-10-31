<template>
  <footer class="layout-footer">
    <div class="footer-content">
      <span class="footer-version">llama.cpp Studio v{{ appVersion }}</span>
      <span class="live-status"
        ><i
          :class="[
            'pi',
            progressStore.isConnected
              ? 'pi-check-circle footer-status--ok'
              : 'pi-clock footer-status--warn',
          ]"
          aria-hidden="true"
        />{{ progressStore.isConnected ? 'Live' : 'Reconnecting…' }}</span
      >
      <RouterLink class="footer-tools" to="/system"
        ><i class="pi pi-sliders-h" aria-hidden="true" /> System<span
          v-if="needsAttention"
          class="footer-tools__badge"
          >Needs attention</span
        ></RouterLink
      >
    </div>
  </footer>
</template>
<script setup>
import { computed } from 'vue'
import { useEnginesStore } from '@/stores/engines'
import { useProgressStore } from '@/stores/progress'
const progressStore = useProgressStore()
const systemStore = useEnginesStore()
const appVersion = typeof __APP_VERSION__ !== 'undefined' ? __APP_VERSION__ : '1.0.0'
const needsAttention = computed(() =>
  Boolean(
    systemStore.systemStatus?.persistence?.saturated ||
    systemStore.systemStatus?.persistence?.latest_failure,
  ),
)
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
.footer-status--ok {
  color: var(--status-success);
}
.footer-status--warn,
.footer-tools__badge {
  color: var(--status-warning);
}
.footer-tools {
  flex-shrink: 0;
  min-height: 1.5rem;
  display: inline-flex;
  align-items: center;
  gap: 0.4rem;
  color: var(--text-secondary);
}

@media (max-width: 900px) {
  .layout-footer {
    padding: 0.45rem 0.75rem;
  }
  .footer-content {
    font-size: 0.68rem;
  }
  .footer-tools__badge {
    display: none;
  }
}
</style>
