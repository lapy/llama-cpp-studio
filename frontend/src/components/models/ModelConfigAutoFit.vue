<template>
  <section
    class="config-card config-auto-fit"
    aria-labelledby="config-auto-fit-title"
    data-auto-fit
  >
    <div class="config-auto-fit__head">
      <div class="config-auto-fit__heading">
        <h3 id="config-auto-fit-title" class="config-auto-fit__title">
          {{ active ? 'Auto-fit is active' : 'Auto-fit is overridden' }}
        </h3>
        <p class="config-auto-fit__summary">
          <template v-if="active">
            No context or GPU-layer overrides are set. The engine's
            <code>--fit</code> sizing stays in control and measures the model at load time.
          </template>
          <template v-else>
            The engine's <code>--fit</code> default sizes context and GPU layers from measured
            memory. These settings override the measured sizing:
          </template>
        </p>
      </div>
      <span class="config-auto-fit__badge">
        {{ active ? 'Engine-managed' : overrideCountLabel }}
      </span>
    </div>

    <ul
      v-if="overrides.length"
      class="config-auto-fit__list"
      aria-label="Settings overriding auto-fit"
    >
      <li
        v-for="override in overrides"
        :key="override.key"
        class="config-auto-fit__item"
        :data-auto-fit-key="override.key"
      >
        <div class="config-auto-fit__item-head">
          <span class="config-auto-fit__label">{{ override.label }}</span>
          <code class="config-auto-fit__current">{{ displayCurrent(override) }}</code>
          <span class="config-auto-fit__arrow" aria-hidden="true">→</span>
          <code class="config-auto-fit__target">{{ override.autoLabel }}</code>
        </div>
        <p class="config-auto-fit__reason">{{ override.reason }}</p>
      </li>
    </ul>

    <p v-if="defaultsText" class="config-auto-fit__defaults">{{ defaultsText }}</p>

    <div v-if="overrides.length" class="config-auto-fit__actions">
      <Button
        label="Restore auto settings"
        icon="pi pi-undo"
        size="small"
        data-auto-fit-restore
        @click="$emit('restore')"
      />
      <Button
        label="Review in Runtime"
        icon="pi pi-server"
        size="small"
        text
        severity="secondary"
        data-auto-fit-review
        @click="$emit('review')"
      />
    </div>
  </section>
</template>

<script setup>
import { computed } from 'vue'
import Button from 'primevue/button'

import { formatNumber } from '@/utils/formatting'

const props = defineProps({
  active: { type: Boolean, default: false },
  overrides: { type: Array, default: () => [] },
  defaults: { type: Object, default: () => ({ fit: null, fitCtx: null, fitTarget: null }) },
})

defineEmits(['restore', 'review'])

const overrideCountLabel = computed(() => {
  const count = props.overrides.length
  return `${count} override${count === 1 ? '' : 's'}`
})

/**
 * Quote only defaults that the engine's own parameter metadata reported. No
 * VRAM, context, layer, or thread number is ever invented here.
 */
const defaultsText = computed(() => {
  const parts = []
  const fit = props.defaults?.fit
  if (fit === 'on' || fit === true) parts.push('--fit on')
  if (props.defaults?.fitCtx) {
    parts.push(`--fit-ctx ${formatNumber(props.defaults.fitCtx)} tokens`)
  }
  if (props.defaults?.fitTarget) {
    parts.push(`--fit-target ${formatNumber(props.defaults.fitTarget)} MiB`)
  }
  return parts.length ? `Engine defaults: ${parts.join(' · ')}.` : ''
})

function capitalize(value) {
  const text = String(value ?? '')
  return text ? text.charAt(0).toUpperCase() + text.slice(1) : text
}

function displayCurrent(override) {
  if (override.key === 'fit') {
    if (override.current === true) return 'On'
    if (override.current === false) return 'Off'
    return capitalize(override.current)
  }
  if (override.key === 'ctx_size') return `${formatNumber(override.current)} tokens`
  if (override.key === 'n_gpu_layers') return `${formatNumber(override.current)} layers`
  return String(override.current)
}
</script>

<style scoped>
.config-auto-fit {
  display: flex;
  flex-direction: column;
  gap: 0.65rem;
}

.config-auto-fit__head {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  flex-wrap: wrap;
  gap: 0.35rem 1rem;
}

.config-auto-fit__heading {
  display: flex;
  flex-direction: column;
  gap: 0.2rem;
  min-width: 0;
}

.config-auto-fit__title {
  margin: 0;
  font-size: 0.9375rem;
  font-weight: 650;
  color: var(--text-primary);
}

.config-auto-fit__summary {
  margin: 0;
  max-width: 72ch;
  font-size: var(--text-xs);
  line-height: 1.45;
  color: var(--text-secondary);
}

.config-auto-fit__summary code,
.config-auto-fit__defaults code {
  font-size: 0.8em;
}

.config-auto-fit__badge {
  font-size: var(--text-xs);
  font-weight: 600;
  color: var(--text-secondary);
  white-space: nowrap;
}

.config-auto-fit__list {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
  margin: 0;
  padding: 0;
  list-style: none;
}

.config-auto-fit__item {
  padding: 0.55rem 0.65rem;
  border: 1px solid var(--border-primary);
  border-left: 3px solid var(--status-warning);
  border-radius: var(--radius-md);
  background: var(--bg-surface);
}

.config-auto-fit__item-head {
  display: flex;
  align-items: baseline;
  flex-wrap: wrap;
  gap: 0.25rem 0.55rem;
}

.config-auto-fit__label {
  font-size: 0.875rem;
  font-weight: 600;
  color: var(--text-primary);
}

.config-auto-fit__current {
  font-size: var(--text-xs);
  color: var(--status-warning);
  text-decoration: line-through;
}

.config-auto-fit__arrow {
  color: var(--text-secondary);
}

.config-auto-fit__target {
  padding: 0.05rem 0.4rem;
  border-radius: var(--radius-sm);
  background: rgba(34, 211, 238, 0.1);
  color: var(--accent-cyan);
  font-size: var(--text-xs);
  font-weight: 600;
}

.config-auto-fit__reason {
  margin: 0.25rem 0 0;
  font-size: var(--text-xs);
  line-height: 1.45;
  color: var(--text-secondary);
  overflow-wrap: anywhere;
}

.config-auto-fit__defaults {
  margin: 0;
  font-size: var(--text-xs);
  line-height: 1.4;
  color: var(--text-secondary);
}

.config-auto-fit__actions {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 0.5rem;
}

.config-auto-fit__actions :deep(button:focus-visible) {
  outline: 2px solid var(--accent-cyan);
  outline-offset: 2px;
}

@media (max-width: 640px) {
  .config-auto-fit__badge {
    white-space: normal;
  }
}
</style>
