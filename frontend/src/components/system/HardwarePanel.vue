<template>
  <section class="hardware-panel" aria-labelledby="hardware-heading">
    <div class="hardware-heading">
      <h2 id="hardware-heading">Hardware resources</h2>
      <Button
        label="Refresh"
        icon="pi pi-refresh"
        size="small"
        severity="secondary"
        outlined
        :loading="refreshing"
        @click="refresh"
      />
    </div>
    <p v-if="error" role="alert" class="hardware-error">{{ error }}</p>
    <div class="ev-system-dashboard">
      <div
        class="metrics-grid metrics-grid--resources"
        role="region"
        aria-label="CPU, memory, and disk"
      >
        <div class="metric-card">
          <i class="pi pi-desktop metric-icon" />
          <div class="metric-data">
            <div class="metric-label">CPU</div>
            <div class="metric-value">
              {{ metricsKnown ? `${Number(sys.cpu_percent || 0).toFixed(1)}%` : 'Unknown' }}
            </div>
            <ProgressBar
              v-if="metricsKnown"
              :value="sys.cpu_percent || 0"
              :showValue="false"
              class="metric-bar"
            />
          </div>
        </div>
        <div class="metric-card">
          <i class="pi pi-database metric-icon" />
          <div class="metric-data">
            <div class="metric-label">Memory</div>
            <div class="metric-value">
              <template v-if="metricsKnown && sys.memory">
                {{ formatBytesIEC(memUsedBytes) }} / {{ formatBytesIEC(sys.memory?.total) }} ({{
                  memPercent
                }}%)
              </template>
              <template v-else>Unknown</template>
            </div>
            <ProgressBar
              v-if="metricsKnown && sys.memory"
              :value="memPercent"
              :showValue="false"
              class="metric-bar"
            />
          </div>
        </div>
        <div class="metric-card">
          <i class="pi pi-save metric-icon" />
          <div class="metric-data">
            <div class="metric-label">Disk</div>
            <div class="metric-value">
              <template v-if="metricsKnown && sys.disk">
                {{ formatBytesIEC(sys.disk?.used) }} / {{ formatBytesIEC(sys.disk?.total) }} ({{
                  diskPercent
                }}%)
              </template>
              <template v-else>Unknown</template>
            </div>
            <ProgressBar
              v-if="metricsKnown && sys.disk"
              :value="diskPercent"
              :showValue="false"
              class="metric-bar"
            />
          </div>
        </div>
      </div>

      <div class="cuda-toolkit-region" role="region" aria-labelledby="cuda-toolkit-heading">
        <div class="cuda-toolkit-main">
          <div class="cuda-toolkit-main__icon" aria-hidden="true">
            <i class="pi pi-bolt" />
          </div>
          <div class="cuda-toolkit-main__body">
            <h3 id="cuda-toolkit-heading" class="cuda-toolkit-main__title">CUDA Toolkit</h3>
            <p class="cuda-toolkit-main__status">
              <template v-if="cuda.installed">CUDA {{ cuda.version || '?' }}</template>
              <template v-else>Not installed</template>
            </p>
            <p class="cuda-toolkit-main__hint">
              <template v-if="cuda.installed_versions?.length">
                {{ cuda.installed_versions.length }} version{{
                  cuda.installed_versions.length === 1 ? '' : 's'
                }}
                detected
              </template>
              <template v-else-if="cuda.cuda_path">
                {{ cuda.cuda_path }}
              </template>
              <template v-else> Build support and toolkit management </template>
            </p>
          </div>
          <div class="cuda-toolkit-main__actions">
            <Button
              icon="pi pi-refresh"
              text
              severity="secondary"
              size="small"
              aria-label="Reload CUDA status"
              v-tooltip.top="'Reload CUDA status'"
              @click.stop="enginesStore.fetchCudaStatus()"
            />
            <Button
              label="Install"
              aria-label="Install CUDA"
              icon="pi pi-download"
              severity="success"
              outlined
              size="small"
              @click.stop="cudaInstallDialogVisible = true"
            />
          </div>
        </div>

        <div v-if="cuda.installed || cuda.installed_versions?.length" class="cuda-toolkit-details">
          <div v-if="cuda.installed" class="status-detail">
            <span class="detail-label">CUDA path</span>
            <code>{{ cuda.cuda_path || 'unknown' }}</code>
          </div>

          <div v-if="cuda.installed_versions?.length" class="ev-version-list">
            <div v-for="v in cuda.installed_versions" :key="v.version" class="ev-version-row">
              <code class="version-name">CUDA {{ v.version }}</code>
              <Tag v-if="v.is_current" value="Active" severity="success" />
              <Button
                icon="pi pi-trash"
                text
                severity="danger"
                size="small"
                @click="confirmUninstallCuda(v.version)"
              />
            </div>
          </div>
        </div>
      </div>

      <div
        v-if="gpus.length"
        class="metrics-grid metrics-grid--gpus"
        role="region"
        aria-label="GPU memory"
      >
        <div
          v-for="(gpuItem, idx) in gpus"
          :key="gpuItem.index ?? gpuItem.uuid ?? gpuItem.name ?? idx"
          class="metric-card"
        >
          <i class="pi pi-bolt metric-icon" />
          <div class="metric-data">
            <div class="metric-label">GPU — {{ gpuItem.name }}</div>
            <div class="metric-value">
              {{ formatBytesIEC(gpuVramUsedBytes(gpuItem)) }} /
              {{ formatBytesIEC(gpuVramTotalBytes(gpuItem)) }} VRAM
            </div>
            <ProgressBar :value="gpuPercent(gpuItem)" :showValue="false" class="metric-bar" />
          </div>
        </div>
      </div>
    </div>
  </section>
  <!-- ── CUDA Install Dialog ────────────────────────────── -->
  <Dialog
    v-model:visible="cudaInstallDialogVisible"
    header="Install CUDA Toolkit"
    modal
    class="dialog-width-xs"
  >
    <div class="dialog-body">
      <div class="form-field">
        <label>Version</label>
        <Select
          v-model="cudaInstallVersion"
          :options="cudaVersionOptions"
          placeholder="Select version…"
          class="w-full"
        />
      </div>
    </div>
    <template #footer>
      <Button
        label="Cancel"
        severity="secondary"
        outlined
        @click="cudaInstallDialogVisible = false"
      />
      <Button
        label="Install"
        icon="pi pi-download"
        severity="success"
        :disabled="!cudaInstallVersion"
        :loading="cudaInstalling"
        @click="installCuda"
      />
    </template>
  </Dialog>
</template>
<script setup>
import { computed, onBeforeUnmount, onMounted, ref } from 'vue'
import Button from 'primevue/button'
import Dialog from 'primevue/dialog'
import Select from 'primevue/select'
import Tag from 'primevue/tag'
import ProgressBar from 'primevue/progressbar'
import { useToast } from 'primevue/usetoast'
import { useConfirm } from 'primevue/useconfirm'
import { requireSingleConfirmation } from '@/composables/singleConfirm'
import { useEnginesStore } from '@/stores/engines'
const enginesStore = useEnginesStore(),
  toast = useToast(),
  confirm = useConfirm()
const refreshing = ref(false),
  error = ref('')
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
  } catch (_) {
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
const sys = computed(() => {
  const s = enginesStore.systemStatus
  return s?.system || s || {}
})

const metricsKnown = computed(() => {
  const status = enginesStore.systemStatus
  if (!status || typeof status !== 'object' || !Object.keys(status).length) return false
  return Boolean(status.system || status.cpu_percent != null || status.memory || status.disk)
})

const gpus = computed(() => enginesStore.gpuInfo?.gpus ?? [])

const memUsedBytes = computed(() => {
  const m = sys.value.memory
  const total = m?.total ?? 0
  if (m?.used != null) return m.used
  if (total > 0 && m?.available != null) return total - m.available
  return 0
})

const memPercent = computed(() => {
  const m = sys.value.memory
  const total = m?.total ?? 0
  const used = memUsedBytes.value
  return total > 0 ? Math.round((used / total) * 100) : 0
})

const diskPercent = computed(() => {
  const d = sys.value.disk
  const used = d?.used ?? 0
  const total = d?.total ?? 0
  return total > 0 ? Math.round((used / total) * 100) : 0
})

/** VRAM from /api/gpu-info: gpus[].memory.{used,total} in bytes */
function gpuVramUsedBytes(g) {
  if (g?.memory?.used != null) return Number(g.memory.used)
  return 0
}

function gpuVramTotalBytes(g) {
  if (g?.memory?.total != null) return Number(g.memory.total)
  return 0
}

function gpuPercent(g) {
  const used = gpuVramUsedBytes(g)
  const total = gpuVramTotalBytes(g)
  return total > 0 ? Math.round((used / total) * 100) : 0
}

// ── CUDA ───────────────────────────────────────────────────
const cuda = computed(() => enginesStore.cudaStatus || {})
const cudaVersionOptions = [
  '13.0',
  '12.9',
  '12.8',
  '12.7',
  '12.6',
  '12.5',
  '12.4',
  '12.3',
  '12.2',
  '12.1',
  '12.0',
  '11.9',
  '11.8',
]
const cudaInstallVersion = ref(null)
const cudaInstalling = ref(false)
const cudaInstallDialogVisible = ref(false)

async function installCuda() {
  cudaInstalling.value = true
  try {
    await enginesStore.installCuda({ version: cudaInstallVersion.value })
    cudaInstallDialogVisible.value = false
    toast.add({
      severity: 'success',
      summary: 'CUDA install started',
      detail: 'Track progress in notifications',
      life: 3000,
    })
    await enginesStore.fetchCudaStatus()
  } catch (e) {
    toast.add({ severity: 'error', summary: 'Failed', detail: e.message, life: 4000 })
  } finally {
    cudaInstalling.value = false
  }
}

function confirmUninstallCuda(version) {
  requireSingleConfirmation(confirm, {
    message: `Uninstall CUDA ${version}?`,
    header: 'Confirm Uninstall',
    icon: 'pi pi-exclamation-triangle',
    acceptClass: 'p-button-danger',
    accept: async () => {
      try {
        await enginesStore.uninstallCuda({ version })
        toast.add({ severity: 'info', summary: `CUDA ${version} uninstalled`, life: 3000 })
      } catch (e) {
        toast.add({ severity: 'error', summary: 'Failed', detail: e.message, life: 4000 })
      }
    },
  })
}

function formatBytesIEC(value) {
  if (value == null) return 'Unknown'
  const units = ['B', 'KiB', 'MiB', 'GiB', 'TiB']
  let n = Number(value),
    index = 0
  while (n >= 1024 && index < units.length - 1) {
    n /= 1024
    index++
  }
  return `${n.toFixed(index ? 1 : 0)} ${units[index]}`
}
</script>
<style scoped>
.hardware-panel,
.ev-system-dashboard {
  display: grid;
  gap: 1.25rem;
}
.hardware-heading,
.cuda-toolkit-main,
.cuda-toolkit-main__actions,
.ev-version-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 0.75rem;
  flex-wrap: wrap;
}
h2,
h3,
p {
  margin: 0;
}
h2 {
  font-size: 1.125rem;
}
h3 {
  font-size: 1rem;
}
.metrics-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(min(100%, 15rem), 1fr));
  gap: 1rem;
}
.metric-card,
.cuda-toolkit-region {
  padding: 1.25rem;
  background: var(--bg-card);
  border: 1px solid var(--border-primary);
  border-radius: var(--radius-md);
}
.metric-card {
  display: flex;
  gap: 0.75rem;
  align-items: flex-start;
}
.metric-data {
  flex: 1;
  min-width: 0;
  display: grid;
  gap: 0.5rem;
}
.metric-icon {
  color: var(--accent-primary);
  margin-top: 0.25rem;
}
.metric-label,
.detail-label,
.cuda-toolkit-main__hint {
  color: var(--text-secondary);
  font-size: var(--text-xs);
}
.metric-value {
  font-size: 1rem;
  font-variant-numeric: tabular-nums;
}
.cuda-toolkit-main__body {
  flex: 1;
  min-width: 0;
  display: grid;
  gap: 0.35rem;
}
.cuda-toolkit-details {
  margin-top: 1rem;
  display: grid;
  gap: 0.75rem;
}
.status-detail {
  display: grid;
  gap: 0.35rem;
  overflow-wrap: anywhere;
}
.ev-version-row {
  padding-block: 0.5rem;
  border-top: 1px solid var(--border-primary);
}
.hardware-error {
  color: var(--status-warning);
}
</style>
