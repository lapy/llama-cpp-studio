<template>
  <PageSection
    id="proxy-vram"
    title="GPU admission"
    description="Refuse a start that cannot fit before a process spawns. Enforce rejects only a known deficit from an override or an observed load; weights-only estimates and failed probes admit with a visible decision. NVIDIA only; CPU mode is skipped."
  >
    <template #meta>
      <Tag :value="probeLabel" :severity="probeSeverity" />
    </template>

    <div class="page-form vram-controls">
      <label class="page-form__label">
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
        <small class="page-form__hint">Applied live; takes effect on the next model start.</small>
      </label>
      <label class="page-form__label">
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
        <small class="page-form__hint">Reserved headroom subtracted on every selected GPU.</small>
      </label>
    </div>

    <div class="vram-estimates">
      <div class="vram-estimates-head">
        <h3 class="page-subheading">Per-model estimates</h3>
        <p class="page-section__description">
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

    <div class="vram-status" role="status" aria-live="polite">
      <dl class="vram-state">
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
    </div>
    <p v-if="lastAdmission?.reason" class="vram-reason">{{ lastAdmission.reason }}</p>
  </PageSection>
</template>

<script setup>
import { computed } from 'vue'
import Button from 'primevue/button'
import Tag from 'primevue/tag'
import PageSection from '@/components/common/PageSection.vue'

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
/* Labels wrap their control for the accessible name. */
.page-form > label {
  display: grid;
  gap: 0.35rem;
  align-content: start;
}

.page-form > label .field {
  font-weight: 400;
}

.vram-estimates {
  display: grid;
  gap: 0.6rem;
  min-width: 0;
}

.vram-estimates-head {
  display: grid;
  gap: 0.25rem;
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
  font-size: var(--text-sm);
}

.vram-state dt {
  color: var(--text-secondary);
}

.vram-state dd {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 0.35rem;
  margin: 0.2rem 0 0;
  overflow-wrap: anywhere;
}

.vram-decision-detail {
  font-variant-numeric: tabular-nums;
}

.vram-reason {
  margin: 0;
  color: var(--text-secondary);
  font-size: var(--text-sm);
}

@media (max-width: 640px) {
  .vram-row {
    grid-template-columns: 1fr;
  }
}
</style>
