<template>
  <div class="system-page page-shell">
    <PageHeader
      title="System"
      description="Inspect hardware and storage, check service health, and manage configuration backups and change history for this Studio host."
    />
    <SectionTabs
      :items="sectionTabs"
      :model-value="activeSection"
      aria-label="System sections"
      responsive-grid
    />

    <template v-if="activeSection === 'diagnostics'">
      <NeedsAttentionPanel />
      <PageSection
        id="service-health"
        title="Service health"
        description="Live state for the API, the settings write queue, and the change feed."
      >
        <template #actions>
          <Button
            label="Refresh status"
            icon="pi pi-refresh"
            severity="secondary"
            outlined
            size="small"
            :loading="refreshing"
            @click="refreshStatus"
          />
          <a class="page-action-link" href="/api/diagnostics/bundle">
            <i class="pi pi-download" aria-hidden="true" />
            <span>Download diagnostics</span>
          </a>
        </template>
        <dl class="health-grid">
          <LiveUpdatesHealth />
          <div class="health-item">
            <dt>Proxy health</dt>
            <dd>{{ proxyHealthLabel }}</dd>
          </div>
          <div class="health-item">
            <dt>Model observation</dt>
            <dd>{{ runtimeLabel }}</dd>
          </div>
          <div class="health-item">
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
        <StateBanner v-if="needsAttention" severity="error">
          {{
            persistenceFailureLabel ||
            'Settings write queue is full. Wait for current writes to finish.'
          }}
        </StateBanner>
        <p class="page-form__hint">
          The diagnostic bundle contains redacted service information for troubleshooting.
        </p>
      </PageSection>
    </template>
    <HardwarePanel v-else-if="activeSection === 'hardware'" />
    <StoragePanel v-else-if="activeSection === 'storage'" />
    <ConfigBackupPanel v-else-if="activeSection === 'backup'" :show-history="false" />
    <ConfigHistoryPanel v-else heading-tag="h2" />
  </div>
</template>

<script setup>
import { computed, onBeforeUnmount, onMounted, ref } from 'vue'
import { useRoute } from 'vue-router'
import Button from 'primevue/button'
import { useEnginesStore } from '@/stores/engines'
import ConfigBackupPanel from '@/components/system/ConfigBackupPanel.vue'
import ConfigHistoryPanel from '@/components/system/ConfigHistoryPanel.vue'
import HardwarePanel from '@/components/system/HardwarePanel.vue'
import StoragePanel from '@/components/system/StoragePanel.vue'
import LiveUpdatesHealth from '@/components/system/LiveUpdatesHealth.vue'
import NeedsAttentionPanel from '@/components/system/NeedsAttentionPanel.vue'
import PageHeader from '@/components/common/PageHeader.vue'
import PageSection from '@/components/common/PageSection.vue'
import SectionTabs from '@/components/common/SectionTabs.vue'
import StateBanner from '@/components/common/StateBanner.vue'

const sectionTabs = [
  { id: 'hardware', label: 'Hardware', to: '/system#hardware' },
  { id: 'storage', label: 'Storage', to: '/system#storage' },
  { id: 'diagnostics', label: 'Diagnostics', to: '/system' },
  { id: 'backup', label: 'Backup & restore', to: '/system#config-backup' },
  { id: 'history', label: 'Change history', to: '/system#history' },
]

const systemStore = useEnginesStore()
const route = useRoute()
const clock = ref(Date.now())
const refreshing = ref(false)
const activeSection = computed(() =>
  route.hash === '#hardware'
    ? 'hardware'
    : route.hash === '#storage'
      ? 'storage'
      : route.hash === '#config-backup'
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

async function refreshStatus() {
  if (refreshing.value) return
  refreshing.value = true
  try {
    await systemStore.fetchSystemStatus()
  } catch {
    // The banner already reports the persistence failure; a read failure keeps
    // the last known state on screen.
  } finally {
    refreshing.value = false
  }
}

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
.health-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(min(100%, 12rem), 1fr));
  gap: 1rem;
  margin: 0;
}

.health-item dt {
  color: var(--text-secondary);
  font-size: var(--text-xs);
}

.health-item dd {
  margin: 0.3rem 0 0;
  overflow-wrap: anywhere;
}

@media (max-width: 640px) {
  .health-grid {
    grid-template-columns: minmax(0, 1fr);
  }
}
</style>
