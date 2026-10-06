<template>
  <div v-if="showTrigger" class="swap-notice-wrap">
    <button
      type="button"
      class="swap-notice-trigger"
      :aria-expanded="modalVisible"
      aria-haspopup="dialog"
      :aria-label="ariaLabel"
      v-tooltip.bottom="'Studio state differs from llama-swap-config.yaml on disk — open to review and apply'"
      @click="openModal"
    >
      <span class="swap-notice-trigger__pulse" aria-hidden="true" />
      <i class="pi pi-exclamation-triangle swap-notice-trigger__icon" aria-hidden="true" />
      <span class="swap-notice-trigger__label">Apply llama-swap config</span>
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
        <p v-if="!modalLoading && stillPending && selectiveLaunch" class="swap-notice-lead">
          These saved launch settings can be applied without reloading the proxy.
          Other loaded models keep their processes.
        </p>
        <p v-else-if="!modalLoading && stillPending" class="swap-notice-lead">
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

          <div v-if="actionLines.length" class="swap-notice-section">
            <h3 class="swap-notice-section__title">Reviewed actions</h3>
            <ul class="swap-notice-changes">
              <li v-for="(line, idx) in actionLines" :key="`action-${idx}`">{{ line }}</li>
            </ul>
          </div>
          <p v-if="withheld" class="swap-notice-confirm" role="alert">
            {{ withheld.message }}
            Apply again confirms that earlier attempt and continues this one.
          </p>
          <Message v-if="!selectiveLaunch" severity="warn" :closable="false" class="swap-notice-warn">
            <span>
              Applying updates <code>llama-swap-config.yaml</code> and reloads the llama-swap proxy.
              <strong>Reload proxy — affects all loaded models.</strong>
            </span>
          </Message>
          <Message v-else severity="info" :closable="false" class="swap-notice-warn">
            <span>
              Restarting one model can interrupt that model's requests. It does not unload the others.
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
          :label="applyLabel"
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
import { withheldConfirmation } from '@/composables/actionConfirmation'

const enginesStore = useEnginesStore()
const toast = useToast()

const modalVisible = ref(false)
const modalLoading = ref(false)
const applying = ref(false)
const withheld = ref(null)

const staleState = computed(() => enginesStore.swapConfigStale)
const pendingState = computed(() => enginesStore.swapConfigPending)

const showTrigger = computed(
  () => Boolean(staleState.value?.applicable && staleState.value?.stale)
)

const changes = computed(() => pendingState.value?.changes ?? [])

const selectiveLaunch = computed(
  () =>
    Boolean(pendingState.value?.launch_manifests) &&
    !pendingState.value?.requires_proxy_reload &&
    Array.isArray(pendingState.value?.models) &&
    pendingState.value.models.length > 0,
)

const actionLines = computed(() => {
  const rows = pendingState.value?.models
  if (!Array.isArray(rows)) return []
  return rows.map((row) => {
    if (row.action === 'global_proxy' || row.requires_proxy_reload) {
      return `Reload proxy — affects all loaded models (${row.model_id})`
    }
    if (row.running && row.action === 'restart_now') return `Restart model ${row.model_id}`
    if (row.action === 'publish_next_start') return `Use new settings on next start for ${row.model_id}`
    return (row.reasons && row.reasons[0]) || row.action
  })
})

const applyLabel = computed(() => {
  if (withheld.value) return 'Apply again'
  return selectiveLaunch.value ? 'Apply launch changes' : 'Apply configuration'
})

/** After refresh inside the dialog, pending may clear — avoid showing stale “apply”. */
const stillPending = computed(
  () => Boolean(pendingState.value?.applicable && pendingState.value?.pending)
)

const dialogTitle = computed(() =>
  stillPending.value ? 'llama-swap config out of sync' : 'llama-swap configuration'
)

const ariaLabel = computed(() => {
  const n = changes.value.length
  if (n > 0) {
    return `llama-swap configuration out of sync, ${n} change${n === 1 ? '' : 's'} listed — open to apply`
  }
  return 'llama-swap configuration out of sync — open to apply'
})

function openModal() {
  modalVisible.value = true
}

async function onDialogShow() {
  modalLoading.value = true
  try {
    await enginesStore.fetchSwapConfigPending()
    await enginesStore.fetchSwapConfigStale()
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
  if (d && typeof d === 'object' && typeof d.message === 'string') return d.message
  return e?.message || 'Unknown error'
}

async function onApply() {
  applying.value = true
  try {
    if (selectiveLaunch.value) {
      const rows = pendingState.value.models || []
      const data = await enginesStore.applyLaunchChanges({
        mode: 'restart_now',
        models: rows.map((row) => ({
          catalog_id: row.catalog_id,
          expected_desired_revision: row.desired_revision,
          expected_published_revision: row.published_revision,
          idempotency_key: `${pendingState.value.plan_id}:${row.model_id}:restart_now`,
        })),
      })
      const failed = (data?.results || []).find((row) => !row.ok)
      toast.add({
        severity: failed ? 'warn' : 'success',
        summary: failed ? 'Launch apply stopped' : 'Launch settings applied',
        detail: failed
          ? failed.message || 'A model apply failed. Earlier models in the list were left on their new revisions.'
          : 'Each selected model was updated without reloading the proxy.',
        life: 5000,
      })
    } else {
      const confirmation = withheld.value
        ? {
            confirm_operation_id: withheld.value.confirm_operation_id,
            confirm_state: withheld.value.confirm_state,
          }
        : null
      await enginesStore.applySwapConfig(confirmation)
      toast.add({
        severity: 'success',
        summary: 'Configuration applied',
        detail: 'llama-swap config was regenerated and the proxy reloaded.',
        life: 4000,
      })
    }
    withheld.value = null
    modalVisible.value = false
  } catch (e) {
    const held = withheldConfirmation(e)
    if (held) {
      withheld.value = held
      return
    }
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
  gap: 0.5rem;
  min-height: 2.65rem;
  padding: 0.35rem 0.95rem 0.35rem 0.75rem;
  border: 2px solid #f97316;
  border-radius: 999px;
  background: linear-gradient(
    135deg,
    color-mix(in srgb, #ea580c 92%, #fff 8%) 0%,
    color-mix(in srgb, #f59e0b 88%, #fef08a 12%) 100%
  );
  color: #1c0a00;
  font-size: 0.8125rem;
  font-weight: 800;
  letter-spacing: 0.02em;
  text-transform: uppercase;
  cursor: pointer;
  box-shadow:
    0 0 0 1px rgba(255, 255, 255, 0.35) inset,
    0 2px 14px rgba(234, 88, 12, 0.55),
    0 0 28px rgba(251, 146, 60, 0.45);
  animation: swap-notice-breathe 1.35s ease-in-out infinite;
  transition:
    transform 0.15s ease,
    filter 0.15s ease;
}

.swap-notice-trigger:hover {
  transform: translateY(-1px) scale(1.02);
  filter: brightness(1.06);
}

.swap-notice-trigger:focus-visible {
  outline: 3px solid var(--accent-cyan, #22d3ee);
  outline-offset: 3px;
}

.swap-notice-trigger__pulse {
  position: absolute;
  inset: -4px;
  border-radius: inherit;
  pointer-events: none;
  border: 2px solid rgba(251, 146, 60, 0.9);
  animation: swap-notice-ring 1.35s ease-out infinite;
}

.swap-notice-trigger__icon {
  font-size: 1.1rem;
  filter: drop-shadow(0 1px 1px rgba(0, 0, 0, 0.25));
}

.swap-notice-trigger__label {
  max-width: 14rem;
  line-height: 1.2;
  text-shadow: 0 1px 0 rgba(255, 255, 255, 0.35);
}

@media (max-width: 768px) {
  .swap-notice-trigger {
    min-height: 2.5rem;
    padding: 0.35rem 0.55rem;
    gap: 0;
  }

  .swap-notice-trigger__label {
    position: absolute;
    width: 1px;
    height: 1px;
    padding: 0;
    margin: -1px;
    overflow: hidden;
    clip: rect(0, 0, 0, 0);
    white-space: nowrap;
    border: 0;
  }
}

@keyframes swap-notice-breathe {
  0%,
  100% {
    box-shadow:
      0 0 0 1px rgba(255, 255, 255, 0.35) inset,
      0 2px 14px rgba(234, 88, 12, 0.55),
      0 0 22px rgba(251, 146, 60, 0.4);
  }
  50% {
    box-shadow:
      0 0 0 1px rgba(255, 255, 255, 0.45) inset,
      0 4px 22px rgba(234, 88, 12, 0.75),
      0 0 40px rgba(253, 186, 116, 0.65);
  }
}

@keyframes swap-notice-ring {
  0% {
    transform: scale(1);
    opacity: 0.85;
  }
  70% {
    transform: scale(1.08);
    opacity: 0;
  }
  100% {
    transform: scale(1.12);
    opacity: 0;
  }
}

.swap-notice-dialog__body {
  display: flex;
  flex-direction: column;
  gap: 1rem;
}

.swap-notice-confirm {
  margin: 0;
  padding: 0.65rem 0.75rem;
  border-radius: var(--radius-md, 0.5rem);
  background: var(--bg-tertiary);
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
