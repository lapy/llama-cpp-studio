<template>
  <section class="runtime-empty" aria-labelledby="runtime-empty-title">
    <div class="runtime-empty__icon" aria-hidden="true">
      <i class="pi pi-sliders-h" />
    </div>
    <div class="runtime-empty__copy">
      <h3 id="runtime-empty-title">No engine parameters yet</h3>
      <p v-if="scanError">
        Studio could not read <strong>{{ engineLabel }}</strong> CLI help:
        <span class="runtime-empty__error">{{ scanError }}</span>
      </p>
      <p v-else-if="scanPending">
        CLI parameters for <strong>{{ engineLabel }}</strong> are still loading. Activate the engine
        on the Engines page, then rescan.
      </p>
      <p v-else>
        Runtime metadata for <strong>{{ engineLabel }}</strong> is not available yet. Install and
        activate the engine, then rescan its CLI parameters.
      </p>
      <p class="runtime-empty__hint">
        Rescanning reads the engine's own help output, so parameter names and defaults stay
        accurate. Saved values are shown for reference.
      </p>
    </div>
    <div class="runtime-empty__actions">
      <Button
        label="Rescan parameters"
        icon="pi pi-refresh"
        size="small"
        :loading="rescanning"
        data-runtime-rescan
        @click="emit('rescan')"
      />
      <Button
        label="Open Engines"
        icon="pi pi-external-link"
        size="small"
        severity="secondary"
        outlined
        data-runtime-open-engines
        @click="emit('open-engines')"
      />
    </div>

    <div v-if="savedEntries.length" class="runtime-empty__saved">
      <h4>Saved settings</h4>
      <dl>
        <div v-for="entry in savedEntries" :key="entry.key" class="runtime-empty__saved-row">
          <dt>{{ entry.label }}</dt>
          <dd>{{ formatValue(entry.value) }}</dd>
        </div>
      </dl>
      <p class="runtime-empty__hint">Rescan to edit them safely.</p>
    </div>
  </section>
</template>

<script setup>
import Button from 'primevue/button'

defineProps({
  engineLabel: { type: String, default: 'this engine' },
  scanError: { type: String, default: '' },
  scanPending: { type: Boolean, default: false },
  savedEntries: { type: Array, default: () => [] },
  rescanning: { type: Boolean, default: false },
})

const emit = defineEmits(['rescan', 'open-engines'])

function formatValue(value) {
  if (value === true) return 'On'
  if (value === false) return 'Off'
  if (Array.isArray(value)) return value.join(', ')
  if (value && typeof value === 'object') return JSON.stringify(value)
  return String(value)
}
</script>

<style scoped>
.runtime-empty {
  display: grid;
  grid-template-columns: auto minmax(0, 1fr) auto;
  gap: 0.9rem 1rem;
  align-items: start;
  padding: 1rem;
  border: 1px solid var(--border-primary);
  border-radius: var(--radius-md);
  background: var(--bg-surface);
}

.runtime-empty__icon {
  display: grid;
  place-items: center;
  width: 2.25rem;
  height: 2.25rem;
  border-radius: var(--radius-sm);
  background: var(--bg-elevated);
  color: var(--text-secondary);
}

.runtime-empty__copy h3 {
  margin: 0 0 0.25rem;
  font-size: 0.95rem;
}

.runtime-empty__copy p {
  margin: 0;
  color: var(--text-secondary);
  font-size: 0.82rem;
  line-height: 1.45;
}

.runtime-empty__copy p + p {
  margin-top: 0.35rem;
}

.runtime-empty__error {
  color: var(--text-primary);
  overflow-wrap: anywhere;
}

.runtime-empty__hint {
  color: var(--text-muted);
}

.runtime-empty__actions {
  display: flex;
  gap: 0.4rem;
  flex-wrap: wrap;
  justify-content: flex-end;
}

.runtime-empty__saved {
  grid-column: 2 / -1;
  border-top: 1px solid var(--border-primary);
  padding-top: 0.65rem;
}

.runtime-empty__saved h4 {
  margin: 0 0 0.35rem;
  font-size: 0.78rem;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  color: var(--text-muted);
}

.runtime-empty__saved dl {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(min(100%, 11rem), 1fr));
  gap: 0.35rem 0.75rem;
  margin: 0;
}

.runtime-empty__saved-row {
  display: flex;
  justify-content: space-between;
  gap: 0.5rem;
  padding: 0.25rem 0;
  border-bottom: 1px solid var(--border-subtle, var(--border-primary));
  font-size: 0.8rem;
}

.runtime-empty__saved-row dt {
  color: var(--text-secondary);
}

.runtime-empty__saved-row dd {
  margin: 0;
  color: var(--text-primary);
  font-variant-numeric: tabular-nums;
  text-align: right;
  overflow-wrap: anywhere;
}

@media (max-width: 760px), (pointer: coarse) {
  .runtime-empty {
    grid-template-columns: auto minmax(0, 1fr);
  }

  .runtime-empty__actions {
    grid-column: 1 / -1;
    justify-content: flex-start;
  }

  .runtime-empty__saved {
    grid-column: 1 / -1;
  }
}
</style>
