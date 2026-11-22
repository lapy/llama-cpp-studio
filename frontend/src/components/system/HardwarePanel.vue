<template>
  <PageSection
    id="hardware"
    title="Hardware resources"
    description="CPU, memory, disk, CUDA toolkit and GPU memory for this host. Readings refresh every second while this section is open."
  >
    <template #actions>
      <Button
        label="Refresh"
        icon="pi pi-refresh"
        severity="secondary"
        outlined
        size="small"
        :loading="refreshing"
        @click="refresh"
      />
    </template>

    <StateBanner v-if="error" severity="warning">
      {{ error }}
      <template #actions>
        <Button
          label="Retry"
          icon="pi pi-refresh"
          severity="secondary"
          outlined
          size="small"
          :loading="refreshing"
          @click="refresh"
        />
      </template>
    </StateBanner>

    <HardwareMetrics />
    <HardwareCudaPanel />
  </PageSection>
</template>

<script setup>
import { onBeforeUnmount, onMounted, ref } from 'vue'
import Button from 'primevue/button'
import PageSection from '@/components/common/PageSection.vue'
import StateBanner from '@/components/common/StateBanner.vue'
import { useEnginesStore } from '@/stores/engines'
import HardwareCudaPanel from './HardwareCudaPanel.vue'
import HardwareMetrics from './HardwareMetrics.vue'

const enginesStore = useEnginesStore()
const refreshing = ref(false)
const error = ref('')

async function refresh() {
  refreshing.value = true
  const results = await Promise.allSettled([
    enginesStore.fetchSystemStatus(),
    enginesStore.fetchCudaStatus(),
  ])
  error.value = results.some((result) => result.status === 'rejected')
    ? 'Some hardware information could not be refreshed. Showing the last available readings.'
    : ''
  refreshing.value = false
}

let statsTimer = null
let statsInFlight = false
let statsTick = 0
let panelMounted = true

async function pollSystemStats() {
  if (!panelMounted || statsInFlight || refreshing.value || document.hidden) return
  statsInFlight = true
  try {
    await enginesStore.fetchSystemStats()
    statsTick += 1
    if (statsTick % 5 === 0) await enginesStore.fetchGpuInfo()
  } catch {
    // Keep the last good readings; a transient read failure is not announced
    // once per second.
  } finally {
    statsInFlight = false
  }
}

onMounted(async () => {
  await refresh()
  if (!panelMounted) return
  statsTimer = window.setInterval(pollSystemStats, 1000)
})

onBeforeUnmount(() => {
  panelMounted = false
  if (statsTimer) window.clearInterval(statsTimer)
  statsTimer = null
})
</script>
