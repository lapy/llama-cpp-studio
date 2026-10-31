<template>
  <div class="system-page page-shell">
    <PageHeader
      title="System"
      description="Service health, configuration backups, and change history."
    />
    <nav class="system-tabs" aria-label="System sections">
      <RouterLink to="/system" :aria-current="activeSection === 'diagnostics' ? 'page' : undefined"
        >Diagnostics</RouterLink
      >
      <RouterLink
        to="/system#config-backup"
        :aria-current="activeSection === 'backup' ? 'page' : undefined"
        >Backup &amp; restore</RouterLink
      >
      <RouterLink
        to="/system#history"
        :aria-current="activeSection === 'history' ? 'page' : undefined"
        >Change history</RouterLink
      >
    </nav>
    <section
      v-if="activeSection === 'diagnostics'"
      class="system-card"
      aria-labelledby="service-health-heading"
    >
      <div class="system-section-head">
        <h2 id="service-health-heading">Service health</h2>
        <a class="system-action" href="/api/diagnostics/bundle"
          ><i class="pi pi-download" aria-hidden="true" /> Download diagnostics</a
        >
      </div>
      <dl class="health-grid">
        <div>
          <dt>Live updates</dt>
          <dd>{{ progressStore.isConnected ? 'Connected' : 'Reconnecting…' }}</dd>
        </div>
        <div>
          <dt>Proxy health</dt>
          <dd>{{ proxyHealthLabel }}</dd>
        </div>
        <div>
          <dt>Model observation</dt>
          <dd>{{ runtimeLabel }}</dd>
        </div>
        <div>
          <dt>Settings writes</dt>
          <dd>
            {{ persistence.pending_store_writes || 0 }} pending<span v-if="persistence.saturated">
              · Queue full ({{ persistence.pending_store_writes }}/{{
                persistence.max_pending_store_writes
              }})</span
            >
          </dd>
        </div>
      </dl>
      <p v-if="needsAttention" class="system-error" role="alert">
        {{
          persistenceFailureLabel ||
          'Settings write queue is full. Wait for current writes to finish.'
        }}
      </p>
      <p class="system-note">
        The diagnostic bundle contains redacted service information for troubleshooting.
      </p>
    </section>
    <div v-else-if="activeSection === 'backup'" class="system-card">
      <ConfigBackupPanel :show-history="false" />
    </div>
    <div v-else class="system-card"><ConfigHistoryPanel heading-tag="h2" /></div>
  </div>
</template>
<script setup>
import { computed, onBeforeUnmount, onMounted, ref } from 'vue'
import { useRoute } from 'vue-router'
import { useEnginesStore } from '@/stores/engines'
import { useProgressStore } from '@/stores/progress'
import ConfigBackupPanel from '@/components/system/ConfigBackupPanel.vue'
import ConfigHistoryPanel from '@/components/system/ConfigHistoryPanel.vue'
import PageHeader from '@/components/common/PageHeader.vue'

const progressStore = useProgressStore()
const systemStore = useEnginesStore()
const route = useRoute()
const clock = ref(Date.now())
const activeSection = computed(() =>
  route.hash === '#config-backup'
    ? 'backup'
    : route.hash === '#history'
      ? 'history'
      : 'diagnostics',
)
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
const needsAttention = computed(() =>
  Boolean(persistence.value.saturated || persistence.value.latest_failure),
)

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
  return age == null ? 'Not observed' : `Checked ${age}s ago`
})

const runtimeLabel = computed(() => {
  const quality = String(runtime.value.quality || '')
  if (!runtime.value.observed_at || quality === 'unreachable' || quality === 'unknown') {
    return 'No running-model observation'
  }
  const age = ageSeconds(runtime.value.observed_at)
  return age == null ? 'Not observed' : `Observed ${age}s ago`
})
</script>
<style scoped>
.system-tabs {
  display: flex;
  gap: 0.25rem;
  border-bottom: 1px solid var(--border-primary);
  overflow-x: auto;
}
.system-tabs a {
  padding: 0.65rem 0.9rem;
  white-space: nowrap;
  color: var(--text-secondary);
  border-bottom: 2px solid transparent;
}
.system-tabs a[aria-current] {
  color: var(--accent-primary);
  border-color: var(--accent-primary);
}
.system-card {
  padding: 1rem;
  border: 1px solid var(--border-primary);
  background: var(--bg-card);
  border-radius: var(--radius-md);
  min-width: 0;
}
.system-section-head {
  display: flex;
  justify-content: space-between;
  align-items: center;
  flex-wrap: wrap;
  gap: 0.75rem;
}
.system-section-head h2 {
  font-size: 1rem;
  margin: 0;
}
.system-action {
  display: inline-flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.5rem 0.75rem;
  border: 1px solid var(--border-primary);
  border-radius: var(--radius-sm);
}
.health-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(12rem, 1fr));
  gap: 1rem;
  margin: 1.25rem 0;
}
.health-grid dt,
.system-note {
  color: var(--text-secondary);
  font-size: 0.85rem;
}
.health-grid dd {
  margin: 0.35rem 0 0;
}
.system-note {
  margin: 0;
}
.system-error {
  color: var(--status-warning);
}
</style>
