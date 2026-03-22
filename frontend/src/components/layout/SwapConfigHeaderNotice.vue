<template>
  <div v-if="showTrigger" class="swap-notice-wrap">
    <button
      type="button"
      class="swap-notice-trigger"
      :aria-expanded="modalVisible"
      aria-haspopup="dialog"
      :aria-label="ariaLabel"
      v-tooltip.bottom="'On-disk llama-swap config differs from the studio — review and apply when ready'"
      @click="openModal"
    >
      <i class="pi pi-bell" aria-hidden="true" />
      <span class="swap-notice-trigger__badge" aria-hidden="true" />
    </button>
    <Dialog
      v-model:visible="modalVisible"
      modal
      dismissable-mask
      class="swap-notice-dialog dialog-width-md"
      :header="dialogTitle"
      @show="onDialogShow"
    >
      <div class="swap-notice-dialog__body">
        <p v-if="!modalLoading && stillPending" class="swap-notice-lead">
          Your saved models and engine settings no longer match
          <code class="swap-notice-code">llama-swap-config.yaml</code>
          on disk. Apply when you are ready to rewrite the file and restart the proxy.
        </p>

        <div v-if="modalLoading" class="swap-notice-loading">
          <i class="pi pi-spin pi-spinner" aria-hidden="true" />
          <span>Checking configuration…</span>
        </div>

        <template v-else-if="stillPending">
          <div v-if="changes.length" class="swap-notice-section">
            <h3 class="swap-notice-section__title">Summary of differences</h3>
            <ul class="swap-notice-changes">
              <li v-for="(line, idx) in changes" :key="idx">{{ line }}</li>
            </ul>
          </div>
          <p v-else class="swap-notice-muted">
            The on-disk file differs from what the studio would generate, but no line-by-line summary was returned.
          </p>

          <Message severity="warn" :closable="false" class="swap-notice-warn">
            <span>
              Applying updates <code>llama-swap-config.yaml</code> and reloads the llama-swap proxy.
              <strong>All currently loaded models will stop.</strong>
            </span>
          </Message>
        </template>

        <div v-else class="swap-notice-in-sync">
          <i class="pi pi-check-circle swap-notice-in-sync__icon" aria-hidden="true" />
          <p class="swap-notice-in-sync__text">
            Configuration is already in sync. You can close this dialog.
          </p>
        </div>
      </div>

      <template #footer>
        <Button
          label="Close"
          severity="secondary"
          outlined
          @click="modalVisible = false"
        />
        <Button
          v-if="!modalLoading && stillPending"
          label="Apply configuration"
          icon="pi pi-check"
          severity="success"
          :loading="applying"
          :disabled="applying"
          @click="onApply"
        />
      </template>
    </Dialog>
  </div>
</template>

<script setup>
import { ref, computed, watch } from 'vue'
import Dialog from 'primevue/dialog'
import Button from 'primevue/button'
import Message from 'primevue/message'
import { useToast } from 'primevue/usetoast'
import { useEnginesStore } from '@/stores/engines'

const enginesStore = useEnginesStore()
const toast = useToast()

const modalVisible = ref(false)
const modalLoading = ref(false)
const applying = ref(false)

const swapState = computed(() => enginesStore.swapConfigPending)

const showTrigger = computed(
  () => Boolean(swapState.value?.applicable && swapState.value?.pending)
)

const changes = computed(() => swapState.value?.changes ?? [])

/** After refresh inside the dialog, pending may clear — avoid showing stale “apply”. */
const stillPending = computed(
  () => Boolean(swapState.value?.applicable && swapState.value?.pending)
)

const dialogTitle = computed(() =>
  stillPending.value ? 'llama-swap config out of sync' : 'llama-swap configuration'
)

const ariaLabel = computed(() => {
  const n = changes.value.length
  if (n > 0) {
    return `llama-swap configuration out of sync, ${n} change${n === 1 ? '' : 's'} listed — open details`
  }
  return 'llama-swap configuration out of sync — open details'
})

function openModal() {
  modalVisible.value = true
}

async function onDialogShow() {
  modalLoading.value = true
  try {
    await enginesStore.fetchSwapConfigPending()
  } finally {
    modalLoading.value = false
  }
}

watch(showTrigger, (visible) => {
  if (!visible) modalVisible.value = false
})

function formatErr(e) {
  const d = e?.response?.data?.detail
  if (Array.isArray(d)) return d.map((x) => x.msg || String(x)).join(' ')
  if (typeof d === 'string') return d
  return e?.message || 'Unknown error'
}

async function onApply() {
  applying.value = true
  try {
    await enginesStore.applySwapConfig()
    toast.add({
      severity: 'success',
      summary: 'Configuration applied',
      detail: 'llama-swap config was regenerated and the proxy reloaded.',
      life: 4000,
    })
    modalVisible.value = false
  } catch (e) {
    toast.add({
      severity: 'error',
      summary: 'Apply failed',
      detail: formatErr(e),
      life: 5000,
    })
  } finally {
    applying.value = false
  }
}
</script>

<style scoped>
.swap-notice-wrap {
  display: inline-flex;
  align-items: center;
}

.swap-notice-trigger {
  position: relative;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 2.5rem;
  height: 2.5rem;
  padding: 0;
  border: 1px solid color-mix(in srgb, var(--accent-amber, #f59e0b) 45%, var(--border-primary));
  border-radius: 999px;
  background: color-mix(in srgb, var(--accent-amber, #f59e0b) 12%, var(--bg-surface));
  color: color-mix(in srgb, var(--accent-amber, #fbbf24) 85%, var(--text-primary));
  cursor: pointer;
  transition:
    border-color 0.15s ease,
    background 0.15s ease,
    transform 0.15s ease,
    box-shadow 0.15s ease;
}

.swap-notice-trigger:hover {
  border-color: color-mix(in srgb, var(--accent-amber, #f59e0b) 70%, var(--border-primary));
  background: color-mix(in srgb, var(--accent-amber, #f59e0b) 18%, var(--bg-surface));
  transform: translateY(-1px);
  box-shadow: 0 0 0 1px color-mix(in srgb, var(--accent-amber, #f59e0b) 25%, transparent);
}

.swap-notice-trigger:focus-visible {
  outline: 2px solid var(--accent-cyan);
  outline-offset: 2px;
}

.swap-notice-trigger .pi {
  font-size: 1.05rem;
}

.swap-notice-trigger__badge {
  position: absolute;
  top: 0.2rem;
  right: 0.25rem;
  width: 0.45rem;
  height: 0.45rem;
  border-radius: 999px;
  background: var(--accent-amber, #f59e0b);
  box-shadow: 0 0 0 2px var(--bg-card, #0f1419);
}

.swap-notice-dialog__body {
  display: flex;
  flex-direction: column;
  gap: 1rem;
}

.swap-notice-lead {
  margin: 0;
  font-size: 0.9rem;
  line-height: 1.55;
  color: var(--text-secondary);
}

.swap-notice-code {
  font-size: 0.82em;
  padding: 0.1em 0.35em;
  border-radius: 4px;
  background: var(--bg-surface, rgba(0, 0, 0, 0.2));
  border: 1px solid var(--border-primary);
  color: var(--text-primary);
}

.swap-notice-loading {
  display: flex;
  align-items: center;
  gap: 0.6rem;
  font-size: 0.88rem;
  color: var(--text-secondary);
  padding: 0.5rem 0;
}

.swap-notice-loading .pi-spinner {
  font-size: 1.25rem;
  color: var(--accent-cyan);
}

.swap-notice-section__title {
  margin: 0 0 0.4rem;
  font-size: 0.72rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: var(--text-muted, var(--text-secondary));
}

.swap-notice-changes {
  margin: 0;
  padding: 0.65rem 0.85rem;
  padding-left: 1.35rem;
  max-height: min(40vh, 16rem);
  overflow-y: auto;
  font-size: 0.82rem;
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, monospace;
  line-height: 1.5;
  color: var(--text-primary);
  background: var(--bg-surface, rgba(0, 0, 0, 0.2));
  border: 1px solid var(--border-primary);
  border-radius: var(--radius-md, 8px);
}

.swap-notice-changes li {
  margin-bottom: 0.2rem;
}

.swap-notice-muted {
  margin: 0;
  font-size: 0.85rem;
  color: var(--text-secondary);
  line-height: 1.5;
}

.swap-notice-warn {
  margin: 0;
}

.swap-notice-warn :deep(code) {
  font-size: 0.85em;
  padding: 0.05em 0.3em;
  border-radius: 3px;
  background: rgba(0, 0, 0, 0.15);
}

.swap-notice-in-sync {
  display: flex;
  align-items: flex-start;
  gap: 0.65rem;
  padding: 0.35rem 0;
}

.swap-notice-in-sync__icon {
  font-size: 1.5rem;
  color: var(--status-success);
  flex-shrink: 0;
  margin-top: 0.05rem;
}

.swap-notice-in-sync__text {
  margin: 0;
  font-size: 0.9rem;
  line-height: 1.5;
  color: var(--text-secondary);
}
</style>
