<template>
  <PageSection
    id="attention"
    title="Operations needing attention"
    description="These operations were interrupted or their outcome could not be established. Studio did not run them again."
    :aria-busy="loading ? 'true' : 'false'"
  >
    <template #actions>
      <Button
        label="Refresh"
        icon="pi pi-refresh"
        severity="secondary"
        outlined
        size="small"
        :loading="loading"
        @click="refresh"
      />
    </template>

    <StateBanner v-if="error" severity="error">
      {{ error }}
      <template #actions>
        <Button
          label="Retry"
          icon="pi pi-refresh"
          severity="secondary"
          outlined
          size="small"
          :loading="loading"
          @click="refresh"
        />
      </template>
    </StateBanner>

    <LoadingState
      v-if="loading && !operations.length"
      message="Checking for interrupted operations…"
      inline
    />
    <StateBanner v-else-if="!operations.length && !error" severity="success">
      No interrupted operations. Nothing is waiting for recovery.
    </StateBanner>

    <ul v-if="operations.length" class="attention-list">
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
          <Button
            :label="isExpanded(operation) ? 'Hide logs' : 'View logs'"
            :icon="isExpanded(operation) ? 'pi pi-chevron-up' : 'pi pi-chevron-down'"
            severity="secondary"
            outlined
            size="small"
            class="attention-action"
            :aria-expanded="isExpanded(operation) ? 'true' : 'false'"
            :aria-controls="logDomId(index)"
            @click="toggleLogs(operation)"
          />
          <Button
            v-if="canAcknowledge(operation)"
            label="Acknowledge"
            severity="secondary"
            text
            size="small"
            class="attention-action attention-action--quiet"
            :aria-label="`Acknowledge ${operation.description}`"
            title="Stop listing this operation. It never started, so a retry is already safe."
            @click="acknowledgeOperation(operation)"
          />
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
  </PageSection>
</template>

<script setup>
import { onMounted, ref } from 'vue'
import Button from 'primevue/button'
import LoadingState from '@/components/common/LoadingState.vue'
import LogOutput from '@/components/common/LogOutput.vue'
import PageSection from '@/components/common/PageSection.vue'
import StateBanner from '@/components/common/StateBanner.vue'
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
.attention-list {
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
  margin: 0;
  padding: 0;
  list-style: none;
}

.attention-item {
  display: flex;
  flex-direction: column;
  gap: 0.35rem;
  min-width: 0;
  padding: 0 0 0.75rem 0.75rem;
  border-left: 3px solid var(--status-warning);
  border-bottom: 1px solid var(--border-primary);
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
  border: 1px solid var(--border-primary);
  border-radius: 999px;
  color: var(--text-secondary);
  font-size: var(--text-xs);
  font-weight: 650;
}

.attention-status[data-status='unknown'] {
  border-color: var(--status-warning);
  color: var(--status-warning);
}

.attention-item__title {
  flex: 1 1 14rem;
  min-width: 0;
  margin: 0;
  font-size: var(--text-base);
  overflow-wrap: anywhere;
}

.attention-item__time {
  flex: 0 0 auto;
  color: var(--text-secondary);
  font-size: var(--text-xs);
  font-variant-numeric: tabular-nums;
}

.attention-item__meta {
  margin: 0;
  color: var(--text-secondary);
  font-size: var(--text-xs);
  overflow-wrap: anywhere;
}

.attention-item__id {
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
}

.attention-item__message {
  margin: 0;
  font-size: var(--text-sm);
  color: var(--text-primary);
}

.attention-item__guidance {
  margin: 0;
  font-size: var(--text-xs);
  color: var(--text-secondary);
}

.attention-item__actions {
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
  margin-top: 0.15rem;
}

.attention-item__logs {
  margin-top: 0.35rem;
}

@media (max-width: 640px) {
  .attention-item__time {
    flex-basis: 100%;
  }

  .attention-item__actions :deep(.p-button) {
    flex: 1 1 auto;
    justify-content: center;
  }
}
</style>
