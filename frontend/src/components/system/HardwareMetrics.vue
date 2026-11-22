<template>
  <div class="hardware-metrics">
    <section class="hardware-block" aria-labelledby="hardware-utilization-heading">
      <h3 id="hardware-utilization-heading" class="page-subheading">CPU, memory and disk</h3>
      <ul class="hardware-stat-grid">
        <li class="hardware-stat">
          <span class="hardware-stat__label"
            ><i class="pi pi-desktop" aria-hidden="true" /> CPU</span
          >
          <span class="hardware-stat__value">
            {{ metricsKnown ? `${Number(sys.cpu_percent || 0).toFixed(1)}%` : 'Unknown' }}
          </span>
          <ProgressBar
            v-if="metricsKnown"
            :value="sys.cpu_percent || 0"
            :showValue="false"
            class="hardware-bar"
            :aria-label="`CPU usage ${Number(sys.cpu_percent || 0).toFixed(0)} percent`"
          />
        </li>
        <li class="hardware-stat">
          <span class="hardware-stat__label"
            ><i class="pi pi-database" aria-hidden="true" /> Memory</span
          >
          <span class="hardware-stat__value">
            <template v-if="metricsKnown && sys.memory">
              {{ formatBytesIEC(memUsedBytes) }} / {{ formatBytesIEC(sys.memory?.total) }} ({{
                memPercent
              }}%)
            </template>
            <template v-else>Unknown</template>
          </span>
          <ProgressBar
            v-if="metricsKnown && sys.memory"
            :value="memPercent"
            :showValue="false"
            class="hardware-bar"
            :aria-label="`Memory usage ${memPercent} percent`"
          />
        </li>
        <li class="hardware-stat">
          <span class="hardware-stat__label"><i class="pi pi-save" aria-hidden="true" /> Disk</span>
          <span class="hardware-stat__value">
            <template v-if="metricsKnown && sys.disk">
              {{ formatBytesIEC(sys.disk?.used) }} / {{ formatBytesIEC(sys.disk?.total) }} ({{
                diskPercent
              }}%)
            </template>
            <template v-else>Unknown</template>
          </span>
          <ProgressBar
            v-if="metricsKnown && sys.disk"
            :value="diskPercent"
            :showValue="false"
            class="hardware-bar"
            :aria-label="`Disk usage ${diskPercent} percent`"
          />
        </li>
      </ul>
    </section>

    <section v-if="gpus.length" class="hardware-block" aria-labelledby="hardware-gpu-heading">
      <h3 id="hardware-gpu-heading" class="page-subheading">GPU memory</h3>
      <ul class="hardware-gpu-list">
        <li
          v-for="(gpuItem, idx) in gpus"
          :key="gpuItem.index ?? gpuItem.uuid ?? gpuItem.name ?? idx"
          class="hardware-gpu"
        >
          <div class="hardware-gpu__head">
            <span class="hardware-gpu__name">GPU — {{ gpuItem.name }}</span>
            <span class="hardware-gpu__vram">
              {{ formatBytesIEC(gpuVramUsedBytes(gpuItem)) }} /
              {{ formatBytesIEC(gpuVramTotalBytes(gpuItem)) }} VRAM
            </span>
          </div>
          <ProgressBar
            :value="gpuPercent(gpuItem)"
            :showValue="false"
            class="hardware-bar"
            :aria-label="`${gpuItem.name} memory usage ${gpuPercent(gpuItem)} percent`"
          />
        </li>
      </ul>
    </section>
  </div>
</template>

<script setup>
import { computed } from 'vue'
import ProgressBar from 'primevue/progressbar'
import { useEnginesStore } from '@/stores/engines'
import { formatBytesIEC } from '@/utils/formatting'

const enginesStore = useEnginesStore()

const sys = computed(() => {
  const status = enginesStore.systemStatus
  return status?.system || status || {}
})

const metricsKnown = computed(() => {
  const status = enginesStore.systemStatus
  if (!status || typeof status !== 'object' || !Object.keys(status).length) return false
  return Boolean(status.system || status.cpu_percent != null || status.memory || status.disk)
})

const gpus = computed(() => enginesStore.gpuInfo?.gpus ?? [])

const memUsedBytes = computed(() => {
  const memory = sys.value.memory
  const total = memory?.total ?? 0
  if (memory?.used != null) return memory.used
  if (total > 0 && memory?.available != null) return total - memory.available
  return 0
})

const memPercent = computed(() => {
  const total = sys.value.memory?.total ?? 0
  return total > 0 ? Math.round((memUsedBytes.value / total) * 100) : 0
})

const diskPercent = computed(() => {
  const disk = sys.value.disk
  const used = disk?.used ?? 0
  const total = disk?.total ?? 0
  return total > 0 ? Math.round((used / total) * 100) : 0
})

/** VRAM from /api/gpu-info: gpus[].memory.{used,total} in bytes. */
function gpuVramUsedBytes(gpu) {
  return gpu?.memory?.used != null ? Number(gpu.memory.used) : 0
}

function gpuVramTotalBytes(gpu) {
  return gpu?.memory?.total != null ? Number(gpu.memory.total) : 0
}

function gpuPercent(gpu) {
  const used = gpuVramUsedBytes(gpu)
  const total = gpuVramTotalBytes(gpu)
  return total > 0 ? Math.round((used / total) * 100) : 0
}
</script>

<style scoped>
.hardware-metrics {
  display: grid;
  gap: 1.25rem;
  min-width: 0;
}

.hardware-block {
  display: grid;
  gap: 0.6rem;
  min-width: 0;
}

.hardware-stat-grid,
.hardware-gpu-list {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(min(100%, 15rem), 1fr));
  gap: 1rem;
  margin: 0;
  padding: 0;
  list-style: none;
}

.hardware-stat,
.hardware-gpu {
  display: grid;
  gap: 0.45rem;
  align-content: start;
  min-width: 0;
}

.hardware-stat__label {
  display: inline-flex;
  align-items: center;
  gap: 0.4rem;
  color: var(--text-secondary);
  font-size: var(--text-xs);
}

.hardware-stat__label .pi {
  color: var(--accent-primary);
}

.hardware-stat__value,
.hardware-gpu__vram {
  font-size: var(--text-base);
  font-variant-numeric: tabular-nums;
  overflow-wrap: anywhere;
}

.hardware-gpu {
  grid-template-columns: minmax(0, 1fr);
}

.hardware-gpu__head {
  display: flex;
  flex-wrap: wrap;
  align-items: baseline;
  justify-content: space-between;
  gap: 0.15rem 0.75rem;
  min-width: 0;
}

.hardware-gpu__name {
  font-size: var(--text-sm);
  font-weight: 600;
  overflow-wrap: anywhere;
}

.hardware-gpu__vram {
  color: var(--text-secondary);
  font-size: var(--text-xs);
}
</style>
