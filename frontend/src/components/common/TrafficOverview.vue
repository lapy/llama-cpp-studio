<template>
  <section class="traffic-overview" aria-labelledby="traffic-heading">
    <div class="traffic-heading">
      <div>
        <h2 id="traffic-heading">Recent traffic</h2>
        <p>Last 30 minutes · newest 100 records</p>
      </div>
      <span class="sample-count">{{ total }} sampled</span>
    </div>
    <svg class="traffic-chart" viewBox="0 0 600 100" role="img" :aria-labelledby="chartId">
      <title :id="chartId">
        {{ total }} retained HTTP requests in the last 30 minutes. {{ totals.success }} successful,
        {{ totals.client }} HTTP 4xx, {{ totals.server }} HTTP 5xx, {{ totals.pending }} other or in
        progress. The sample is limited to the newest 100 retained records.
      </title>
      <path v-for="y in [16, 40, 64, 88]" :key="y" :d="`M0 ${y} H600`" class="chart-grid" />
      <g v-for="(bucket, index) in buckets" :key="index">
        <title>
          {{ bucketLabel(index) }}: {{ bucket.success }} successful, {{ bucket.client }} HTTP 4xx,
          {{ bucket.server }} HTTP 5xx, {{ bucket.pending }} other or in progress
        </title>
        <rect
          v-for="segment in segments(bucket)"
          :key="segment.kind"
          :x="index * 50 + 8"
          :y="88 - segment.top * scale"
          width="34"
          :height="segment.count * scale"
          rx="2"
          :class="`traffic-bar traffic-bar--${segment.kind}`"
        />
      </g>
    </svg>
    <div v-if="!total" class="traffic-empty">No retained traffic in this window.</div>
    <div class="chart-axis"><span>30 min ago</span><span>15 min ago</span><span>Now</span></div>
    <div class="chart-legend">
      <span
        ><i class="legend-dot legend-dot--success" />2xx / 3xx <b>{{ totals.success }}</b></span
      ><span
        ><i class="legend-dot legend-dot--client" />4xx <b>{{ totals.client }}</b></span
      ><span
        ><i class="legend-dot legend-dot--server" />5xx <b>{{ totals.server }}</b></span
      ><span
        ><i class="legend-dot legend-dot--pending" />Other / active
        <b>{{ totals.pending }}</b></span
      >
    </div>
  </section>
</template>
<script setup>
import { computed, useId } from 'vue'
const props = defineProps({
  requests: { type: Array, default: () => [] },
  now: { type: Number, required: true },
})
const chartId = useId()
const buckets = computed(() => {
  const rows = Array.from({ length: 12 }, () => ({ success: 0, client: 0, server: 0, pending: 0 }))
  for (const request of props.requests.slice(0, 100)) {
    const timestamp = Number(request.started_at || request.completed_at)
    if (!Number.isFinite(timestamp) || timestamp < props.now - 1800 || timestamp > props.now)
      continue
    const index = Math.min(11, Math.floor((timestamp - (props.now - 1800)) / 150))
    const status = Number(request.status)
    const kind =
      status >= 500
        ? 'server'
        : status >= 400
          ? 'client'
          : status >= 200 && status < 400
            ? 'success'
            : 'pending'
    rows[index][kind]++
  }
  return rows
})
const totals = computed(() =>
  buckets.value.reduce(
    (sum, bucket) => {
      for (const key of Object.keys(sum)) sum[key] += bucket[key]
      return sum
    },
    { success: 0, client: 0, server: 0, pending: 0 },
  ),
)
const total = computed(() => Object.values(totals.value).reduce((a, b) => a + b, 0))
const scale = computed(
  () =>
    72 /
    Math.max(1, ...buckets.value.map((bucket) => Object.values(bucket).reduce((a, b) => a + b, 0))),
)
function segments(bucket) {
  let top = 0
  return ['success', 'client', 'server', 'pending'].map((kind) => {
    top += bucket[kind]
    return { kind, count: bucket[kind], top }
  })
}
function bucketLabel(index) {
  return new Date((props.now - 1800 + index * 150) * 1000).toLocaleTimeString([], {
    hour: '2-digit',
    minute: '2-digit',
  })
}
</script>
<style scoped>
.traffic-overview {
  position: relative;
  padding: 1rem 1.1rem;
  min-width: 0;
  border: 1px solid var(--border-primary);
  border-radius: var(--radius-lg);
  background: var(--bg-card);
}
.traffic-heading {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  gap: 0.75rem;
}
.traffic-heading h2 {
  margin: 0;
  font-size: 0.9rem;
}
.traffic-heading p {
  margin: 0.25rem 0 0;
  color: var(--text-secondary);
  font-size: 0.75rem;
}
.sample-count {
  color: var(--text-secondary);
  font-size: 0.75rem;
  white-space: nowrap;
}
.traffic-chart {
  display: block;
  width: 100%;
  height: 7rem;
  margin-top: 0.6rem;
}
.chart-grid {
  fill: none;
  stroke: var(--border-primary);
  stroke-dasharray: 3 5;
}
.traffic-bar--success,
.legend-dot--success {
  fill: var(--status-success);
  background: var(--status-success);
}
.traffic-bar--client,
.legend-dot--client {
  fill: var(--status-warning);
  background: var(--status-warning);
}
.traffic-bar--server,
.legend-dot--server {
  fill: var(--status-error);
  background: var(--status-error);
}
.traffic-bar--pending,
.legend-dot--pending {
  fill: var(--accent-primary);
  background: var(--accent-primary);
}
.chart-axis {
  display: flex;
  justify-content: space-between;
  color: var(--text-secondary);
  font-size: 0.65rem;
}
.chart-legend {
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem 1rem;
  padding-top: 0.85rem;
  color: var(--text-secondary);
  font-size: 0.7rem;
}
.chart-legend span {
  display: inline-flex;
  align-items: center;
  gap: 0.35rem;
}
.chart-legend b {
  color: var(--text-primary);
  font-variant-numeric: tabular-nums;
}
.legend-dot {
  width: 0.4rem;
  height: 0.4rem;
  border-radius: 50%;
}
.traffic-empty {
  position: absolute;
  top: 47%;
  left: 0;
  right: 0;
  text-align: center;
  color: var(--text-secondary);
  font-size: 0.8rem;
}
</style>
