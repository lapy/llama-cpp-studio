<template>
  <PageSection
    variant="plain"
    title="Requests"
    :description="`Live and retained, newest first · ${total} matching`"
  >
    <RequestFiltersBar
      :filters="filters"
      :keys="keys"
      :states="states"
      :loading="loading"
      @update:filter="(key, value) => $emit('update:filter', key, value)"
      @apply="$emit('apply')"
      @reset="$emit('reset')"
    />
    <StateBanner v-if="error" severity="error" role="status">
      {{ error }}
      <template #actions>
        <Button label="Retry" size="small" severity="secondary" outlined @click="$emit('apply')" />
      </template>
    </StateBanner>
    <div class="table-shell">
      <div class="table-shell__scroll request-table">
        <table v-if="requests.length">
          <thead>
            <tr>
              <th>Started</th>
              <th>Client / model</th>
              <th>Method / path</th>
              <th>State / status</th>
              <th>Queue / load</th>
              <th>TTFT / total</th>
              <th>Tokens in (cached) / out</th>
              <th scope="col"><span class="sr-only">Actions</span></th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="request in requests" :key="request.id">
              <td
                class="request-time"
                :title="formatTime(request.started_at || request.completed_at)"
              >
                {{ formatClock(request.started_at || request.completed_at) }}
              </td>
              <td>
                <span class="request-primary">{{
                  request.model || request.requested_model || 'Unresolved model'
                }}</span
                ><small class="request-secondary">{{
                  request.key_name || 'Unidentified client'
                }}</small>
              </td>
              <td>
                <span class="request-method">{{ request.method }}</span
                ><code class="request-path">{{ request.path }}</code>
              </td>
              <td>
                <Tag
                  :value="request.state || 'completed'"
                  :severity="requestSeverity(request)"
                /><br />{{ request.status || 'In progress'
                }}<small v-if="request.queue_position">
                  · queue #{{ request.queue_position }}</small
                >
              </td>
              <td>{{ metric(request.queue_ms) }} / {{ metric(request.loading_ms) }}</td>
              <td :title="ttftDetail(request, semantics)">
                {{ ttftLabel(request) }} / {{ metric(request.latency_ms) }}
              </td>
              <td :title="tokenDetail(request)">
                {{ request.input_tokens ?? '—'
                }}<template v-if="request.input_tokens_cached != null"
                  >({{ request.input_tokens_cached }})</template
                >
                / {{ request.output_tokens ?? '—'
                }}<small class="usage-source">{{ request.usage_source || 'unknown' }}</small>
              </td>
              <td>
                <Button
                  label="Details"
                  size="small"
                  text
                  @click="$emit('select-request', request)"
                />
              </td>
            </tr>
          </tbody>
        </table>
        <div v-else class="table-shell__state">
          <LoadingState v-if="loading" small inline message="Loading requests…" />
          <EmptyState
            v-else
            icon="pi pi-filter"
            :title="hasFilters ? 'No requests match these filters' : 'No requests recorded yet'"
            :description="
              hasFilters
                ? 'Clear the filters to search the retained request history again.'
                : 'Requests appear here as clients use the inference gateway.'
            "
          >
            <Button v-if="hasFilters" label="Reset filters" size="small" @click="$emit('reset')" />
            <Button
              v-else
              label="Refresh"
              size="small"
              severity="secondary"
              outlined
              @click="$emit('apply')"
            />
          </EmptyState>
        </div>
      </div>
      <div v-if="total" class="table-shell__footer">
        <span class="request-range"
          >{{ total ? offset + 1 : 0 }}–{{ Math.min(offset + requests.length, total) }} of
          {{ total }}</span
        >
        <div class="pagination-actions">
          <Button
            label="Previous 100"
            size="small"
            severity="secondary"
            outlined
            :disabled="offset === 0 || loading"
            @click="$emit('page', -100)"
          />
          <Button
            label="Next 100"
            size="small"
            severity="secondary"
            outlined
            :disabled="offset + 100 >= total || loading"
            @click="$emit('page', 100)"
          />
        </div>
      </div>
    </div>
    <RequestInspectorPanel
      :visible="Boolean(selectedRequest)"
      :request="selectedRequest"
      @update:visible="(value) => (value ? null : $emit('select-request', null))"
    />
  </PageSection>
</template>

<script setup>
import { computed } from 'vue'
import Button from 'primevue/button'
import Tag from 'primevue/tag'
import EmptyState from '@/components/common/EmptyState.vue'
import LoadingState from '@/components/common/LoadingState.vue'
import PageSection from '@/components/common/PageSection.vue'
import StateBanner from '@/components/common/StateBanner.vue'
import RequestFiltersBar from '@/components/proxy/RequestFiltersBar.vue'
import RequestInspectorPanel from '@/components/proxy/RequestInspectorPanel.vue'
import {
  formatClock,
  formatTime,
  metric,
  requestSeverity,
  tokenDetail,
  ttftDetail,
  ttftLabel,
} from '@/utils/proxyRequestPresentation'

const props = defineProps({
  requests: { type: Array, default: () => [] },
  total: { type: Number, default: 0 },
  offset: { type: Number, default: 0 },
  loading: { type: Boolean, default: false },
  error: { type: String, default: '' },
  filters: { type: Object, required: true },
  keys: { type: Array, default: () => [] },
  states: { type: Array, default: () => [] },
  selectedRequest: { type: Object, default: null },
  semantics: { type: Object, default: () => ({}) },
})

defineEmits(['update:filter', 'apply', 'reset', 'page', 'select-request'])

const hasFilters = computed(() => Object.values(props.filters).some((value) => value !== ''))
</script>

<style scoped>
.request-table table {
  min-width: 48rem;
  font-size: var(--text-sm);
}

.request-table tbody tr:hover {
  background: var(--hover-bg);
}

.request-primary {
  display: block;
  max-width: 17rem;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  font-weight: 550;
}

.request-secondary {
  display: block;
  margin-top: 0.2rem;
  color: var(--text-secondary);
}

.request-method {
  display: inline-block;
  margin-right: 0.5rem;
  color: var(--text-secondary);
  font-size: var(--text-xs);
  font-weight: 600;
}

.request-path {
  font-size: var(--text-xs);
}

.request-time {
  white-space: nowrap;
  color: var(--text-secondary);
  font-size: var(--text-xs);
}

.usage-source {
  display: block;
  color: var(--text-secondary);
}

.request-range {
  color: var(--text-secondary);
  font-size: var(--text-xs);
  font-variant-numeric: tabular-nums;
}

.pagination-actions {
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
}

@media (max-width: 640px), (pointer: coarse) {
  .table-shell__footer,
  .pagination-actions {
    flex-direction: column;
    align-items: stretch;
  }
}
</style>
