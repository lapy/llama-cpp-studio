<template>
  <section class="proxy-card gpu-admission-panel" aria-labelledby="proxy-vram-heading">
    <div class="section-head">
      <div>
        <h2 id="proxy-vram-heading">GPU admission</h2>
        <p>
          Refuse a start that cannot fit before a process spawns. Enforce rejects only a known
          deficit from an override or an observed load; weights-only estimates and failed probes
          admit with a visible decision. NVIDIA only; CPU mode is skipped.
        </p>
      </div>
      <Tag :value="probeLabel" :severity="probeSeverity" />
    </div>

    <div class="settings-grid compact-fields">
      <label>
        Mode
        <select
          class="field"
          :value="mode"
          aria-label="VRAM admission mode"
          @change="emit('update:mode', $event.target.value)"
        >
          <option value="off">Off — no preflight</option>
          <option value="warn">Warn — admit and record the deficit</option>
          <option value="enforce">Enforce — refuse a known deficit</option>
        </select>
        <small class="hint">Applied live; takes effect on the next model start.</small>
      </label>
      <label>
        Safety margin per GPU (MiB)
        <input
          class="field"
          type="number"
          min="0"
          max="1048576"
          step="128"
          :value="margin"
          aria-label="VRAM safety margin MiB"
          @input="
            emit('update:margin', $event.target.value === '' ? null : Number($event.target.value))
          "
        />
        <small class="hint">Reserved headroom subtracted on every selected GPU.</small>
      </label>
    </div>

    <div class="vram-estimates">
      <div class="vram-estimates-head">
        <h3>Per-model estimates</h3>
        <p>
          Overrides win over observed ready-time measurements and the model file-size lower bound.
          They are gateway settings, not a model configuration field.
        </p>
      </div>
      <div v-for="(row, index) in rows" :key="`${row.model}-${index}`" class="vram-row">
        <label class="sr-only" :for="`vram-model-${index}`"
          >Model for estimate {{ index + 1 }}</label
        >
        <select
          :id="`vram-model-${index}`"
          class="field"
          :value="row.model"
          @change="updateRow(index, { model: $event.target.value })"
        >
          <option value="">Select a model…</option>
          <option v-for="option in modelOptions(row.model)" :key="option" :value="option">
            {{ option }}
          </option>
        </select>
        <label class="sr-only" :for="`vram-mib-${index}`">
          Estimate for {{ row.model || `row ${index + 1}` }} in MiB
        </label>
        <input
          :id="`vram-mib-${index}`"
          class="field"
          type="number"
          min="1"
          :max="MAX_ESTIMATE_MIB"
          step="256"
          :value="row.mib"
          placeholder="MiB"
          @input="
            updateRow(index, {
              mib: $event.target.value === '' ? null : Number($event.target.value),
            })
          "
        />
        <Button
          icon="pi pi-times"
          text
          rounded
          severity="danger"
          size="small"
          aria-label="Remove VRAM estimate"
          @click="removeRow(index)"
        />
      </div>
      <div>
        <Button
          label="Add estimate"
          icon="pi pi-plus"
          size="small"
          severity="secondary"
          outlined
          @click="addRow"
        />
      </div>
    </div>

    <dl class="vram-state" role="status" aria-live="polite">
      <div>
        <dt>Probe</dt>
        <dd>
          {{ probeLabel }}
          <small v-if="probe.at">· {{ formatStamp(probe.at) }}</small>
          <small v-if="probe.error">· {{ probe.error }}</small>
        </dd>
      </div>
      <div>
        <dt>Observed estimates</dt>
        <dd>{{ observedLabel }}</dd>
      </div>
      <div>
        <dt>Last decision</dt>
        <dd>
          <template v-if="lastAdmission">
            <Tag
              :value="lastAdmission.decision"
              :severity="decisionSeverity(lastAdmission.decision)"
            />
            <span class="vram-decision-detail">{{ decisionDetail(lastAdmission) }}</span>
          </template>
          <span v-else>No admission decision recorded yet.</span>
        </dd>
      </div>
    </dl>
    <p v-if="lastAdmission?.reason" class="vram-reason">{{ lastAdmission.reason }}</p>
  </section>
</template>

<script setup>
import { computed } from 'vue'
import Button from 'primevue/button'
import Tag from 'primevue/tag'

const MAX_ESTIMATE_MIB = 4194304

const props = defineProps({
  mode: { type: String, default: 'off' },
  margin: { type: [Number, String], default: 1024 },
  rows: { type: Array, default: () => [] },
  catalog: { type: Array, default: () => [] },
  snapshot: { type: Object, default: () => ({}) },
})
const emit = defineEmits(['update:mode', 'update:margin', 'update:rows'])

const probe = computed(() => props.snapshot?.probe || {})
const lastAdmission = computed(() => props.snapshot?.last_admission || null)
const probeLabel = computed(
  () =>
    ({
      ok: 'Probe ready',
      failed: 'Probe failed',
      cpu: 'No NVIDIA GPU',
      never: 'Probe not run yet',
    })[probe.value.status] || 'Probe unknown',
)
const probeSeverity = computed(
  () =>
    ({ ok: 'success', failed: 'danger', cpu: 'secondary', never: 'secondary' })[
      probe.value.status
    ] || 'secondary',
)
const observedLabel = computed(() => {
  const observed = props.snapshot?.observed_estimates || {}
  const entries = Object.entries(observed)
  if (!entries.length) return 'No uncontended load observed yet.'
  return entries
    .map(([model, value]) => `${model} · ${Number(value.mib).toLocaleString()} MiB`)
    .join(', ')
})

function modelOptions(current) {
  const options = (props.catalog || []).map((model) => model.id).filter(Boolean)
  if (current && !options.includes(current)) options.push(current)
  return options
}

function updateRow(index, patch) {
  const next = props.rows.map((row, position) =>
    position === index ? { ...row, ...patch } : { ...row },
  )
  emit('update:rows', next)
}

function addRow() {
  emit('update:rows', [...props.rows, { model: '', mib: null }])
}

function removeRow(index) {
  emit(
    'update:rows',
    props.rows.filter((_, position) => position !== index),
  )
}

function decisionSeverity(decision) {
  return (
    {
      rejected: 'danger',
      warned: 'warn',
      unknown: 'info',
      admitted: 'success',
      skipped: 'secondary',
    }[decision] || 'secondary'
  )
}

function decisionDetail(decision) {
  const bits = []
  if (decision.model) bits.push(decision.model)
  if (decision.estimate_mib != null) {
    bits.push(`${Number(decision.estimate_mib).toLocaleString()} MiB ${decision.estimate_source}`)
  }
  if (decision.available_mib != null) {
    bits.push(`${Number(decision.available_mib).toLocaleString()} MiB available`)
  }
  if (decision.deficit_mib) bits.push(`${Number(decision.deficit_mib).toLocaleString()} MiB short`)
  return bits.join(' · ')
}

function formatStamp(value) {
  if (!value) return ''
  const date = new Date(value)
  return Number.isNaN(date.getTime()) ? String(value) : date.toLocaleString()
}
</script>

<style scoped>
.gpu-admission-panel {
  display: grid;
  gap: 1rem;
  min-width: 0;
  border: 1px solid var(--border-primary);
  background: var(--bg-card);
}

.section-head {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 1rem;
  flex-wrap: wrap;
}

.section-head h2 {
  margin: 0;
  font-size: 1.05rem;
}

.section-head p,
.vram-estimates-head p,
.vram-reason {
  color: var(--text-secondary);
  margin: 0.3rem 0 0;
}

.vram-estimates {
  display: grid;
  gap: 0.6rem;
}

.vram-estimates-head h3 {
  margin: 0;
  font-size: 0.95rem;
}

.vram-row {
  display: grid;
  grid-template-columns: minmax(0, 2fr) minmax(0, 1fr) auto;
  gap: 0.5rem;
  align-items: center;
}

.vram-state {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(min(100%, 14rem), 1fr));
  gap: 0.8rem;
  margin: 0;
  font-size: 0.82rem;
}

.vram-state dt {
  color: var(--text-secondary);
}

.vram-state dd {
  margin: 0.2rem 0 0;
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 0.35rem;
  overflow-wrap: anywhere;
}

.vram-decision-detail {
  font-variant-numeric: tabular-nums;
}

@media (max-width: 640px) {
  .vram-row {
    grid-template-columns: 1fr;
  }
}
</style>
