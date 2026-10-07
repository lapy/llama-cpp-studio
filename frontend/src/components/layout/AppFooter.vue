<template>
  <footer class="layout-footer">
    <div class="footer-content">
      <span class="footer-version">llama.cpp Studio v{{ appVersion }}</span>
      <div
        class="live-status"
        :title="progressStore.isConnected ? 'Live updates (SSE)' : 'Reconnecting…'"
      >
        <i
          v-if="progressStore.isConnected"
          class="pi pi-check-circle footer-status footer-status--ok"
          aria-hidden="true"
        />
        <i v-else class="pi pi-clock footer-status footer-status--warn" aria-hidden="true" />
        <span>{{ progressStore.isConnected ? 'Live' : 'Reconnecting…' }}</span>
      </div>
      <div class="footer-diagnostics">
        <span v-if="persistence.saturated" role="status"
          >Queue full ({{ persistence.pending_store_writes }}/{{
            persistence.max_pending_store_writes
          }})</span
        >
        <span v-if="persistence.latest_failure" role="status">{{ persistenceFailureLabel }}</span>
        <span>{{ proxyHealthLabel }}</span>
        <span>{{ runtimeLabel }}</span>
        <a href="/api/diagnostics/bundle">Download diagnostics</a>
      </div>
    </div>
  </footer>
</template>

<script setup>
import { computed, onBeforeUnmount, onMounted, ref } from 'vue'
import { useEnginesStore } from '@/stores/engines'
import { useProgressStore } from '@/stores/progress'

const progressStore = useProgressStore()
const systemStore = useEnginesStore()
const appVersion = typeof __APP_VERSION__ !== 'undefined' ? __APP_VERSION__ : '1.0.0'
const clock = ref(Date.now())
let clockTimer = null

onMounted(() => {
  clockTimer = setInterval(() => {
    clock.value = Date.now()
  }, 15000)
})

onBeforeUnmount(() => {
  if (clockTimer) clearInterval(clockTimer)
})

const persistence = computed(() => systemStore.systemStatus?.persistence || {})
const runtime = computed(() => systemStore.systemStatus?.runtime_observation || {})

const persistenceFailureLabel = computed(() => {
  const failure = persistence.value.latest_failure
  if (!failure) return ''
  if (failure.code === 'STORE_QUEUE_FULL') {
    return 'Persistence queue is full. Retry after the current writes finish.'
  }
  if (failure.code === 'STORE_WRITE_FAILED') {
    if (failure.committed === true) {
      return 'The document was replaced, but acknowledgement failed. Refresh before trying again.'
    }
    if (failure.committed === false) {
      return 'The save was not stored. The previous state is unchanged.'
    }
    return 'The save outcome could not be established. Refresh before trying again.'
  }
  return 'A persistence error occurred. Its details were not exported.'
})

function ageSeconds(iso) {
  const then = Date.parse(iso || '')
  if (Number.isNaN(then)) return null
  return Math.max(0, Math.round((clock.value - then) / 1000))
}

const proxyHealthLabel = computed(() => {
  const observed =
    systemStore.systemStatus?.proxy_status?.health_observed_at ||
    systemStore.systemStatus?.proxy_status?.observed_at
  const age = ageSeconds(observed)
  return age == null ? 'Proxy health unknown' : `Proxy health ${age}s`
})

const runtimeLabel = computed(() => {
  const quality = String(runtime.value.quality || '')
  if (!runtime.value.observed_at || quality === 'unreachable' || quality === 'unknown') {
    return 'No running-model observation'
  }
  const age = ageSeconds(runtime.value.observed_at)
  return age == null ? 'Runtime age unknown' : `Runtime ${age}s`
})
</script>

<style scoped>
.footer-status--ok {
  color: var(--status-success);
}

.footer-status--warn {
  color: var(--status-warning);
}

.live-status {
  gap: 0.35rem;
  padding: 0.24rem 0.55rem;
  border: 1px solid var(--border-primary);
  border-radius: 999px;
  background: color-mix(in srgb, var(--bg-surface) 72%, transparent);
  color: var(--text-muted);
}

.footer-version {
  color: var(--text-secondary);
  white-space: nowrap;
}

.footer-diagnostics {
  display: flex;
  flex-wrap: wrap;
  justify-content: flex-end;
  gap: 0.3rem 0.7rem;
  max-width: 100%;
}
</style>
