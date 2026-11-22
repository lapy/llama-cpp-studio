<template>
  <div class="proxy-overview">
    <PageSection variant="plain" title="Gateway overview" :description="summaryDescription">
      <template #actions>
        <Button
          label="Reset stats"
          aria-label="Reset stats"
          icon="pi pi-refresh"
          size="small"
          text
          severity="secondary"
          @click="$emit('reset-stats')"
        />
      </template>
      <dl class="proxy-summary metrics" aria-label="Gateway statistics">
        <div class="summary-tile">
          <dt><i class="pi pi-arrow-right-arrow-left" aria-hidden="true" />Requests</dt>
          <dd>{{ number(summary.requests) }}</dd>
          <dd class="summary-caption">
            {{ number(summary.input_tokens) }} tokens in · {{ number(summary.output_tokens) }} out
          </dd>
        </div>
        <div class="summary-tile">
          <dt><i class="pi pi-bolt" aria-hidden="true" />Active / queued</dt>
          <dd>
            {{ summary.active || 0 }} <span>/ {{ summary.queued || 0 }}</span>
          </dd>
          <dd class="summary-caption">
            {{ summary.loading || 0 }} loading · {{ summary.streaming || 0 }} streaming
          </dd>
        </div>
        <div class="summary-tile">
          <dt><i class="pi pi-clock" aria-hidden="true" />Latency p50 / p95</dt>
          <dd class="summary-latency">
            {{ metric(summary.p50_latency_ms) }}
            <span>/ {{ metric(summary.p95_latency_ms) }}</span>
          </dd>
          <dd class="summary-caption">
            TTFT p50 {{ metric(summary.p50_ttft_ms) }} · all retained outcomes in this window
          </dd>
        </div>
        <div class="summary-tile">
          <dt><i class="pi pi-exclamation-circle" aria-hidden="true" />HTTP 4xx / 5xx</dt>
          <dd>
            <button
              class="metric-link"
              :aria-label="`${summary.client_errors ?? '—'} HTTP 4xx requests — inspect`"
              @click="$emit('inspect-outcome', '4xx')"
            >
              {{ summary.client_errors ?? '—' }}</button
            ><span> / </span
            ><button
              class="metric-link"
              :aria-label="`${summary.server_errors ?? '—'} HTTP 5xx requests — inspect`"
              @click="$emit('inspect-outcome', '5xx')"
            >
              {{ summary.server_errors ?? '—' }}
            </button>
          </dd>
          <dd class="summary-caption">Client responses / server failures</dd>
        </div>
        <div class="summary-tile">
          <dt>
            <i class="pi pi-database" aria-hidden="true" />Input cache
            <i
              class="pi pi-info-circle metric-help"
              v-tooltip.top="metricNote('input_cache')"
              aria-hidden="true"
            />
          </dt>
          <dd>{{ percent(summary.cache_hit_ratio) }}</dd>
          <dd class="summary-caption">
            {{ number(summary.input_tokens_cached) }} cached of
            {{ number(summary.input_tokens_total) }} known input ·
            {{ number(summary.cache_known_requests) }} cache-known samples
          </dd>
        </div>
      </dl>
      <StateBanner v-if="summary.dropped_history" severity="warning" icon="">
        {{ summary.dropped_history }} request records could not be retained while history storage
        was unavailable.
      </StateBanner>
      <StateBanner v-if="notice" :severity="noticeSeverity" class="vram-notice">
        {{ notice.text }}
      </StateBanner>
      <details class="overview-analysis" :open="!compact">
        <summary>
          <i class="pi pi-chart-bar" aria-hidden="true" />Traffic &amp; gateway health<span
            >30-minute sample</span
          ><i class="pi pi-chevron-down" aria-hidden="true" />
        </summary>
        <div class="overview-analysis-grid">
          <TrafficOverview :requests="requests" :now="now" />
          <aside class="gateway-health" aria-labelledby="gateway-health-heading">
            <div class="health-heading">
              <h3 id="gateway-health-heading">Gateway health</h3>
              <span class="health-live"
                ><i :class="['health-dot', { 'health-dot--warning': !activeBackend }]" />{{
                  activeBackend ? 'Active' : 'Offline'
                }}</span
              >
            </div>
            <dl class="health-facts">
              <div>
                <dt>Engine fleet</dt>
                <dd>
                  {{ readyEngines }} ready<span v-if="failedEngines">
                    · {{ failedEngines }} failed</span
                  >
                </dd>
              </div>
              <div>
                <dt>Access</dt>
                <dd>
                  {{
                    settings.allow_anonymous === true
                      ? 'Anonymous enabled'
                      : settings.allow_anonymous === false
                        ? 'Key required'
                        : 'Not observed'
                  }}
                </dd>
              </div>
              <div>
                <dt>Usage authority</dt>
                <dd>
                  {{ number(summary.authoritative_usage_requests) }} confirmed ·
                  {{ number(summary.unknown_usage) }} unreported
                </dd>
              </div>
              <div>
                <dt>Cache coverage</dt>
                <dd>
                  {{ percent(summary.cache_hit_ratio) }} hit ratio ·
                  {{ percent(summary.cache_coverage) }} of known input
                </dd>
              </div>
            </dl>
            <div class="health-actions">
              <button type="button" @click="$emit('inspect-failures')">
                Inspect failures<i class="pi pi-arrow-right" aria-hidden="true" /></button
              ><button type="button" @click="$emit('open-settings')">
                Gateway settings<i class="pi pi-arrow-right" aria-hidden="true" />
              </button>
            </div>
          </aside>
        </div>
      </details>
    </PageSection>

    <PageSection variant="plain" title="Engine fleet">
      <template #meta>
        <Tag
          :value="`${liveRunning.length} running`"
          :severity="liveRunning.length ? 'success' : 'secondary'"
        />
      </template>
      <template #actions>
        <RouterLink :to="{ name: 'models' }"
          >Open Models <i class="pi pi-arrow-up-right" aria-hidden="true"
        /></RouterLink>
      </template>
      <div v-if="running.length" class="runtime-list">
        <article v-for="item in running" :key="`${item.model}-${item.pid}`" class="runtime">
          <div class="runtime-head">
            <div>
              <strong>{{ item.model }}</strong
              ><span
                >{{ item.engine || 'engine' }} · PID {{ item.pid }} · port {{ item.port }}</span
              >
            </div>
            <Tag
              :value="item.state"
              :severity="item.state === 'failed' ? 'danger' : item.ready ? 'success' : 'warn'"
            />
            <StateBanner
              v-if="item.failure_reason"
              severity="error"
              icon=""
              class="runtime-failure"
            >
              {{ item.failure_reason }}. Start the model from Models to retry.
            </StateBanner>
          </div>
          <ProgressBar
            v-if="!item.ready && item.loading_progress != null"
            :value="Math.round(item.loading_progress)"
            :show-value="false"
            aria-label="Model loading progress"
          />
          <dl class="metrics">
            <div>
              <dt>In-flight</dt>
              <dd>{{ item.inflight }}</dd>
            </div>
            <div>
              <dt>Requests</dt>
              <dd>{{ item.request_count || 0 }}</dd>
            </div>
            <div>
              <dt>HTTP 4xx / 5xx</dt>
              <dd>
                <button
                  class="metric-link"
                  :aria-label="`${item.client_error_count ?? '—'} / ${item.server_error_count ?? '—'} — Inspect failed requests for ${item.model}`"
                  @click="$emit('inspect-failures', item.model)"
                >
                  {{ item.client_error_count ?? '—' }} / {{ item.server_error_count ?? '—' }}
                </button>
              </dd>
            </div>
            <div>
              <dt>Latency p50 / p95</dt>
              <dd>{{ metric(item.p50_latency_ms) }} · {{ metric(item.p95_latency_ms) }}</dd>
            </div>
            <div>
              <dt>
                Lifetime throughput
                <i
                  class="pi pi-info-circle metric-help"
                  v-tooltip.top="metricNote('lifetime_output_tokens_per_second', semantics)"
                  aria-hidden="true"
                />
              </dt>
              <dd>{{ rate(item.lifetime_output_tokens_per_second) }} tok/s</dd>
            </div>
            <div>
              <dt>
                Rolling 10 s / 60 s
                <i
                  class="pi pi-info-circle metric-help"
                  v-tooltip.top="metricNote('rolling_60s_output_tokens_per_second', semantics)"
                  aria-hidden="true"
                />
              </dt>
              <dd>
                {{ rate(item.rolling_10s_output_tokens_per_second) }} /
                {{ rate(item.rolling_60s_output_tokens_per_second) }} tok/s
              </dd>
            </div>
            <div>
              <dt>
                Decode p50
                <i
                  class="pi pi-info-circle metric-help"
                  v-tooltip.top="metricNote('proxy_decode_tokens_per_second', semantics)"
                  aria-hidden="true"
                />
              </dt>
              <dd>{{ rate(item.p50_decode_tokens_per_second) }} tok/s</dd>
            </div>
            <div>
              <dt>TTFT p50 / p95</dt>
              <dd>{{ metric(item.p50_ttft_ms) }} · {{ metric(item.p95_ttft_ms) }}</dd>
            </div>
            <div>
              <dt>Output tokens · cache</dt>
              <dd>{{ number(item.output_tokens) }} · {{ percent(item.cache_hit_ratio) }}</dd>
              <dd class="summary-caption">
                {{ number(item.confirmed_requests) }} confirmed ·
                {{ number(item.partial_requests) }} partial
              </dd>
            </div>
            <div>
              <dt>Traffic</dt>
              <dd>
                {{ formatBytes(item.input_bytes) }} in · {{ formatBytes(item.output_bytes) }} out
              </dd>
            </div>
          </dl>
          <details class="runtime-details">
            <summary>Process details</summary>
            <dl class="metrics">
              <div>
                <dt>Revision</dt>
                <dd>
                  <code>{{ shortRevision(item.revision) }}</code>
                </dd>
              </div>
              <div>
                <dt>Installed build</dt>
                <dd>{{ formatBuildIdentity(item.engine_build) }}</dd>
              </div>
              <div>
                <dt>Started</dt>
                <dd>{{ formatTime(item.created_at) }}</dd>
              </div>
              <div v-if="item.vram">
                <dt>VRAM</dt>
                <dd>{{ vramSummary(item.vram, number) }}</dd>
              </div>
            </dl>
          </details>
          <div class="runtime-actions">
            <Button
              v-if="['failed', 'error'].includes(item.state)"
              label="Remove"
              icon="pi pi-trash"
              size="small"
              severity="danger"
              outlined
              aria-label="Remove failed engine instance"
              title="Remove failed engine instance and clear its status"
              @click="$emit('remove-failed', item)"
            />
            <Button
              :label="openLogs === item.model ? 'Hide logs' : 'View logs'"
              :icon="openLogs === item.model ? 'pi pi-eye-slash' : 'pi pi-list'"
              size="small"
              severity="secondary"
              outlined
              :aria-expanded="openLogs === item.model"
              @click="$emit('toggle-logs', item.model)"
            />
            <Button
              label="Requests"
              size="small"
              text
              severity="secondary"
              @click="$emit('inspect-requests', item.model)"
            />
            <RouterLink :to="modelLocation(item.model)" class="model-link"
              >Manage model <i class="pi pi-arrow-up-right" aria-hidden="true"
            /></RouterLink>
          </div>
          <LogOutput
            v-if="openLogs === item.model"
            :lines="logs[item.model] || []"
            :levels="ENGINE_LOG_LEVELS"
            :live="!['failed', 'stopped'].includes(item.state)"
            stream-aware
            downloadable
            clearable
            aria-label="Engine log output"
            empty-message="No engine output yet."
            @download="$emit('download-logs', item.model, $event)"
            @clear="$emit('clear-logs', item.model)"
          />
        </article>
      </div>
      <EmptyState
        v-else
        icon="pi pi-play"
        title="No engine processes running"
        description="Active engines appear here after inference traffic, or after starting a model from the model library."
      >
        <RouterLink :to="{ name: 'models' }" class="page-action-link">
          <i class="pi pi-play" aria-hidden="true" />
          Start a model
        </RouterLink>
      </EmptyState>
    </PageSection>
  </div>
</template>

<script setup>
import { computed, onMounted, onUnmounted, ref } from 'vue'
import { RouterLink } from 'vue-router'
import Button from 'primevue/button'
import ProgressBar from 'primevue/progressbar'
import Tag from 'primevue/tag'
import EmptyState from '@/components/common/EmptyState.vue'
import LogOutput from '@/components/common/LogOutput.vue'
import PageSection from '@/components/common/PageSection.vue'
import StateBanner from '@/components/common/StateBanner.vue'
import TrafficOverview from '@/components/common/TrafficOverview.vue'
import { ENGINE_LOG_LEVELS } from '@/utils/proxyLogLevels'
import {
  formatBuildIdentity,
  formatBytes,
  formatTime,
  metric,
  metricNote,
  number,
  percent,
  rate,
  shortRevision,
} from '@/utils/proxyRequestPresentation'
import { vramNotice as vramNoticeFor, vramSummary } from '@/utils/vramAdmission'

const props = defineProps({
  summary: { type: Object, default: () => ({}) },
  running: { type: Array, default: () => [] },
  liveRunning: { type: Array, default: () => [] },
  requests: { type: Array, default: () => [] },
  now: { type: Number, default: 0 },
  settings: { type: Object, default: () => ({}) },
  catalog: { type: Array, default: () => [] },
  vramSnapshot: { type: Object, default: () => ({}) },
  semantics: { type: Object, default: () => ({}) },
  activeBackend: { type: String, default: '' },
  openLogs: { type: String, default: '' },
  logs: { type: Object, default: () => ({}) },
})

defineEmits([
  'reset-stats',
  'inspect-outcome',
  'inspect-failures',
  'inspect-requests',
  'open-settings',
  'toggle-logs',
  'download-logs',
  'clear-logs',
  'remove-failed',
])

const compact = ref(false)
let media
function onMedia(event) {
  compact.value = Boolean(event.matches)
}
onMounted(() => {
  media = window.matchMedia?.('(max-width: 800px)')
  compact.value = Boolean(media?.matches)
  media?.addEventListener('change', onMedia)
})
onUnmounted(() => media?.removeEventListener('change', onMedia))

const readyEngines = computed(() => props.liveRunning.filter((item) => item.ready).length)
const failedEngines = computed(
  () => props.running.filter((item) => ['failed', 'error'].includes(item.state)).length,
)
const summaryDescription = computed(
  () =>
    `All HTTP traffic · since ${
      props.summary.since ? formatTime(props.summary.since) : 'the observation window began'
    }`,
)
const notice = computed(() => vramNoticeFor(props.vramSnapshot, number))
const noticeSeverity = computed(() => (notice.value?.severity === 'error' ? 'error' : 'warning'))

function modelLocation(proxyId) {
  const modelId = props.catalog.find((model) => model.id === proxyId)?.studio_model_id
  return modelId ? { name: 'model-config', params: { id: modelId } } : { name: 'models' }
}
</script>

<style scoped>
.proxy-overview {
  display: grid;
  gap: 1rem;
  min-width: 0;
}

.proxy-summary {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(min(100%, 12rem), 1fr));
  gap: 0.75rem;
  padding: 0;
  margin: 0;
}

.summary-tile {
  min-width: 0;
  padding: 0.9rem 1rem;
  border: 1px solid var(--border-primary);
  border-radius: var(--radius-md);
  background: var(--bg-card);
}

.summary-tile dt {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  font-size: var(--text-xs);
}

.summary-tile dt i {
  font-size: var(--text-xs);
  opacity: 0.7;
}

.summary-tile dd {
  padding: 0.35rem 0 0.2rem;
  font-size: clamp(1.25rem, 1.65vw, 1.8rem);
  font-weight: 600;
  letter-spacing: -0.04em;
  white-space: nowrap;
}

.summary-tile dd > span {
  color: var(--text-secondary);
  font-size: 0.7em;
  font-weight: 450;
}

.summary-tile dd.summary-caption {
  padding: 0;
  margin: 0;
  color: var(--text-secondary);
  font-size: var(--text-xs);
  font-weight: 400;
  letter-spacing: normal;
  white-space: normal;
}

.summary-latency {
  font-size: clamp(1rem, 1.3vw, 1.4rem) !important;
}

.metric-link {
  padding: 0;
  border: 0;
  border-bottom: 1px dotted var(--text-secondary);
  background: transparent;
  color: var(--text-primary);
  font: inherit;
  cursor: pointer;
}

.metric-help {
  margin-left: 0.25rem;
  color: var(--text-secondary);
  font-size: var(--text-xs);
  cursor: help;
}

.overview-analysis > summary {
  display: none;
}

.overview-analysis-grid {
  display: grid;
  grid-template-columns: minmax(0, 1.7fr) minmax(16rem, 1fr);
  gap: 1rem;
}

.gateway-health {
  min-width: 0;
  padding: 1rem 1.1rem;
  border: 1px solid var(--border-primary);
  border-radius: var(--radius-lg);
  background: var(--bg-card);
}

.health-heading {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 1rem;
}

.health-heading h3 {
  margin: 0;
  font-size: var(--text-base);
}

.health-live {
  display: inline-flex;
  align-items: center;
  gap: 0.4rem;
  color: var(--text-secondary);
  font-size: var(--text-xs);
}

.health-dot {
  width: 0.4rem;
  height: 0.4rem;
  border-radius: 50%;
  background: var(--status-success);
}

.health-dot--warning {
  background: var(--status-warning);
}

.health-facts {
  margin: 0.75rem 0;
  font-size: var(--text-xs);
}

.health-facts > div {
  display: flex;
  justify-content: space-between;
  gap: 0.5rem;
  padding: 0.4rem 0;
  border-bottom: 1px solid var(--border-primary);
}

.health-facts dt {
  color: var(--text-secondary);
}

.health-facts dd {
  margin: 0;
  font-variant-numeric: tabular-nums;
}

.health-facts dd span {
  color: var(--status-error);
}

.health-actions {
  display: flex;
  flex-wrap: wrap;
  justify-content: space-between;
  gap: 0.5rem;
}

.health-actions button {
  display: inline-flex;
  align-items: center;
  gap: 0.45rem;
  padding: 0.25rem 0;
  border: 0;
  background: transparent;
  color: var(--accent-primary);
  font-size: var(--text-xs);
  cursor: pointer;
}

.health-actions i {
  font-size: var(--text-xs);
}

.runtime-list {
  display: grid;
  gap: 0.6rem;
  min-width: 0;
}

.runtime {
  display: grid;
  gap: 0.65rem;
  min-width: 0;
  padding: 0.8rem;
  border: 1px solid var(--border-primary);
  border-radius: var(--radius-md);
}

.runtime-head {
  display: flex;
  flex-wrap: wrap;
  align-items: flex-start;
  justify-content: space-between;
  gap: 1rem;
}

.runtime-head div {
  display: grid;
  gap: 0.25rem;
  min-width: 0;
}

.runtime-head strong {
  overflow-wrap: anywhere;
}

.runtime-head span:not(.p-tag) {
  color: var(--text-secondary);
  font-size: var(--text-sm);
}

.runtime-head :deep(.state-banner) {
  flex: 1 1 100%;
  margin-bottom: 0;
}

.metrics {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(min(100%, 9rem), 1fr));
  gap: 0.6rem 1rem;
  margin: 0;
  font-size: var(--text-sm);
}

.metrics div {
  display: grid;
  gap: 0.2rem;
}

.metrics dt {
  color: var(--text-secondary);
}

.metrics dd {
  margin: 0;
  color: var(--text-primary);
  font-variant-numeric: tabular-nums;
}

.runtime-details {
  color: var(--text-secondary);
  font-size: var(--text-sm);
}

.runtime-details summary {
  padding: 0.25rem 0;
  cursor: pointer;
}

.runtime-details .metrics {
  margin-top: 0.5rem;
}

.runtime-actions {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 0.5rem;
}

.runtime-actions .model-link {
  margin-left: auto;
  font-size: var(--text-sm);
}

@media (max-width: 1100px) {
  .overview-analysis-grid {
    grid-template-columns: 1fr;
  }
}

@media (max-width: 800px) {
  .overview-analysis > summary {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.65rem 0.75rem;
    border: 1px solid var(--border-primary);
    border-radius: var(--radius-md);
    color: var(--text-secondary);
    font-size: var(--text-xs);
    cursor: pointer;
    list-style: none;
  }

  .overview-analysis > summary::-webkit-details-marker {
    display: none;
  }

  .overview-analysis > summary span {
    margin-left: auto;
    font-size: var(--text-xs);
  }

  .overview-analysis[open] > summary {
    margin-bottom: 0.75rem;
  }

  .runtime-head {
    gap: 0.5rem;
  }
}

@media (max-width: 600px) {
  .summary-tile {
    padding: 0.7rem;
  }

  .summary-tile dd {
    font-size: 1.3rem;
  }

  .proxy-overview {
    gap: 0.75rem;
  }
}

@media (max-width: 420px) {
  .proxy-summary,
  .metrics {
    grid-template-columns: 1fr;
  }
}
</style>
