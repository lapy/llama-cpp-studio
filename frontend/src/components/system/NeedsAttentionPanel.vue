<template>
  <section
    id="attention"
    class="attention-card"
    aria-labelledby="operations-attention-heading"
    :aria-busy="loading ? 'true' : 'false'"
  >
    <div class="attention-head">
      <div class="attention-heading">
        <h2 id="operations-attention-heading">Operations needing attention</h2>
        <p class="attention-intro">
          These operations were interrupted or their outcome could not be established. Studio did
          not run them again.
        </p>
      </div>
      <button type="button" class="attention-refresh" :disabled="loading" @click="refresh">
        <i :class="loading ? 'pi pi-spin pi-spinner' : 'pi pi-refresh'" aria-hidden="true" />{{
          loading ? 'Checking…' : 'Refresh'
        }}
      </button>
    </div>

    <p v-if="error" class="attention-error" role="alert">{{ error }}</p>
    <p v-if="loading && !operations.length" class="attention-empty" role="status">
      Checking for interrupted operations…
    </p>
    <p v-else-if="!operations.length && !error" class="attention-empty">
      No interrupted operations.
    </p>

    <ul v-else-if="operations.length" class="attention-list">
      <li
        v-for="(operation, index) in operations"
        :key="operation.operation_id"
        class="attention-item"
      >
        <div class="attention-item__head">
          <span class="attention-status" :data-status="operation.status">
            {{ statusLabel(operation.status) }}
          </span>
          <h3 class="attention-item__title">{{ operation.description }}</h3>
          <time
            v-if="operationIso(operation)"
            class="attention-item__time"
            :datetime="operationIso(operation)"
          >
            {{ operationTime(operation) }}
          </time>
        </div>
        <p class="attention-item__meta">
          {{ operation.kind }}<span aria-hidden="true"> · </span
          ><span class="attention-item__id">{{ operation.operation_id }}</span>
        </p>
        <p class="attention-item__message">{{ operation.message }}</p>
        <p class="attention-item__guidance">{{ guidance(operation) }}</p>
        <div class="attention-item__actions">
          <button
            type="button"
            class="attention-action"
            :aria-expanded="isExpanded(operation) ? 'true' : 'false'"
            :aria-controls="logDomId(index)"
            @click="toggleLogs(operation)"
          >
            <i
              :class="isExpanded(operation) ? 'pi pi-chevron-up' : 'pi pi-chevron-down'"
              aria-hidden="true"
            />{{ isExpanded(operation) ? 'Hide logs' : 'View logs' }}
          </button>
          <button
            v-if="canAcknowledge(operation)"
            type="button"
            class="attention-action attention-action--quiet"
            :aria-label="`Acknowledge ${operation.description}`"
            title="Stop listing this operation. It never started, so a retry is already safe."
            @click="acknowledgeOperation(operation)"
          >
            Acknowledge
          </button>
        </div>
        <div v-if="isExpanded(operation)" :id="logDomId(index)" class="attention-item__logs">
          <LogOutput
            :lines="progressStore.getTaskLogs(operation.operation_id)"
            :loading="progressStore.isTaskLogsLoading(operation.operation_id)"
            aria-label="Interrupted operation log"
            empty-message="No retained output for this operation."
          />
        </div>
      </li>
    </ul>
  </section>
</template>

<script setup>
import { onMounted, ref } from 'vue'
import LogOutput from '@/components/common/LogOutput.vue'
import { useOperationAttention } from '@/composables/useOperationAttention'

const STATUS_LABELS = {
  interrupted: 'Interrupted',
  unknown: 'Outcome unknown',
}

const expanded = ref({})
const { operations, loading, error, refresh, canAcknowledge, acknowledge, progressStore } =
  useOperationAttention()

onMounted(() => {
  void refresh()
})

function statusLabel(status) {
  return STATUS_LABELS[status] || 'Needs attention'
}

function operationDate(operation) {
  const seconds = Number(operation?.updated_at)
  if (!Number.isFinite(seconds) || seconds <= 0) return null
  return new Date(seconds * 1000)
}

function operationIso(operation) {
  return operationDate(operation)?.toISOString() || ''
}

function operationTime(operation) {
  const date = operationDate(operation)
  if (!date) return ''
  return date.toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short' })
}

function guidance(operation) {
  if (operation?.needs_confirmation) {
    return (
      'Studio will not run this again on its own. Retry the original action from its page; ' +
      'Studio asks you to confirm this operation and its current state before anything runs.'
    )
  }
  return (
    'The action had not started when Studio restarted, so nothing was replayed. ' +
    'It can be retried from its page without confirming this operation.'
  )
}

function isExpanded(operation) {
  return Boolean(expanded.value[operation?.operation_id])
}

function logDomId(index) {
  return `operation-attention-logs-${index}`
}

function toggleLogs(operation) {
  const id = operation?.operation_id
  if (!id) return
  const next = !isExpanded(operation)
  expanded.value = { ...expanded.value, [id]: next }
  if (next) void progressStore.hydrateTaskLogs(id)
}

async function acknowledgeOperation(operation) {
  const acknowledged = await acknowledge(operation)
  if (!acknowledged || !operation?.operation_id) return
  const { [operation.operation_id]: _removed, ...rest } = expanded.value
  expanded.value = rest
}
</script>

<style scoped>
.attention-card {
  min-width: 0;
}

.attention-head {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  flex-wrap: wrap;
  gap: 0.75rem;
}

.attention-head h2 {
  margin: 0;
  font-size: 1rem;
}

.attention-heading {
  min-width: 0;
}

.attention-intro {
  margin: 0.35rem 0 0;
  color: var(--text-secondary);
  font-size: 0.85rem;
  max-width: 55ch;
}

.attention-refresh {
  display: inline-flex;
  align-items: center;
  gap: 0.4rem;
  min-height: 2.25rem;
  padding: 0.4rem 0.7rem;
  border: 1px solid var(--border-primary);
  border-radius: var(--radius-sm);
  background: transparent;
  color: var(--text-primary);
  font: inherit;
  font-size: 0.85rem;
  cursor: pointer;
}

.attention-refresh:disabled {
  cursor: wait;
  opacity: 0.7;
}

.attention-refresh:hover:not(:disabled) {
  background: var(--bg-card-hover, rgba(255, 255, 255, 0.06));
}

.attention-error {
  margin: 0.85rem 0 0;
  color: var(--status-warning);
}

.attention-empty {
  margin: 0.85rem 0 0;
  color: var(--text-secondary);
  font-size: 0.85rem;
}

.attention-list {
  display: flex;
  flex-direction: column;
  gap: 0.65rem;
  margin: 0.9rem 0 0;
  padding: 0;
  list-style: none;
}

.attention-item {
  display: flex;
  flex-direction: column;
  gap: 0.35rem;
  min-width: 0;
  padding: 0.7rem 0.8rem;
  border: 1px solid var(--border-primary);
  border-left: 3px solid var(--status-warning, #d97706);
  border-radius: var(--radius-sm);
  background: var(--bg-tertiary);
}

.attention-item__head {
  display: flex;
  align-items: baseline;
  flex-wrap: wrap;
  gap: 0.5rem;
  min-width: 0;
}

.attention-status {
  flex: 0 0 auto;
  padding: 0.1rem 0.45rem;
  border-radius: 999px;
  border: 1px solid var(--border-primary);
  color: var(--text-secondary);
  font-size: var(--text-xs, 0.72rem);
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.04em;
}

.attention-status[data-status='unknown'] {
  border-color: var(--status-warning, #d97706);
  color: var(--status-warning, #d97706);
}

.attention-item__title {
  flex: 1 1 14rem;
  min-width: 0;
  margin: 0;
  font-size: 0.9rem;
  overflow-wrap: anywhere;
}

.attention-item__time {
  flex: 0 0 auto;
  color: var(--text-secondary);
  font-size: var(--text-xs, 0.72rem);
  font-variant-numeric: tabular-nums;
}

.attention-item__meta {
  margin: 0;
  color: var(--text-secondary);
  font-size: var(--text-xs, 0.72rem);
  overflow-wrap: anywhere;
}

.attention-item__id {
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
}

.attention-item__message {
  margin: 0;
  font-size: 0.85rem;
  color: var(--text-primary);
}

.attention-item__guidance {
  margin: 0;
  font-size: 0.8rem;
  color: var(--text-secondary);
}

.attention-item__actions {
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
  margin-top: 0.15rem;
}

.attention-action {
  display: inline-flex;
  align-items: center;
  gap: 0.35rem;
  min-height: 2.25rem;
  padding: 0.35rem 0.65rem;
  border: 1px solid var(--border-primary);
  border-radius: var(--radius-sm);
  background: transparent;
  color: var(--text-primary);
  font: inherit;
  font-size: 0.82rem;
  cursor: pointer;
}

.attention-action:hover {
  background: var(--bg-card-hover, rgba(255, 255, 255, 0.06));
}

.attention-action--quiet {
  color: var(--text-secondary);
}

.attention-action:focus-visible,
.attention-refresh:focus-visible {
  outline: 2px solid var(--accent-primary, #60a5fa);
  outline-offset: 2px;
}

.attention-item__logs {
  margin-top: 0.35rem;
}

@media (max-width: 600px) {
  .attention-item {
    padding: 0.6rem;
  }

  .attention-item__time {
    flex-basis: 100%;
  }

  .attention-action {
    flex: 1 1 auto;
    justify-content: center;
  }
}
</style>
