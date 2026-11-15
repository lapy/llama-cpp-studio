<template>
  <div class="gpu-bind">
    <div class="section-label section-label--inline">
      GPUs
      <i
        class="pi pi-info-circle param-info"
        tabindex="0"
        aria-label="About GPU assignment"
        v-tooltip.top="
          'All GPUs follows the deployment. Choose GPUs to pick devices in the order you click them. CPU only hides every CUDA device. The first card you pick becomes device 0.'
        "
      />
    </div>
    <div
      id="gpu-mode"
      class="gpu-mode"
      role="radiogroup"
      aria-label="GPU assignment mode"
      @keydown="onGpuModeKeydown"
    >
      <button
        v-for="mode in gpuModeOptions"
        :key="mode.value"
        type="button"
        class="gpu-mode__option"
        role="radio"
        :class="{ selected: activeMode === mode.value }"
        :aria-checked="activeMode === mode.value ? 'true' : 'false'"
        :tabindex="activeMode === mode.value ? 0 : -1"
        :disabled="disabled"
        @click="selectMode(mode.value)"
      >
        {{ mode.label }}
      </button>
    </div>
    <div
      v-if="activeMode === 'selected'"
      class="gpu-card-grid"
      role="group"
      aria-label="Select NVIDIA GPUs for CUDA_VISIBLE_DEVICES"
    >
      <button
        v-for="gpu in gpuCards"
        :key="gpu.value"
        type="button"
        class="gpu-card"
        :class="{ selected: gpu.order != null }"
        :aria-pressed="gpu.order != null ? 'true' : 'false'"
        :disabled="disabled"
        @click="toggleGpuCard(gpu.value)"
      >
        <span class="gpu-card__index" aria-hidden="true">{{ gpu.index }}</span>
        <span class="gpu-card__body">
          <span class="gpu-card__name">{{ gpu.name }}</span>
          <span class="gpu-card__meta">{{
            gpu.order != null ? `Device ${gpu.order}` : 'Not used'
          }}</span>
        </span>
      </button>
    </div>
  </div>
</template>

<script setup>
import { computed } from 'vue'

const props = defineProps({
  gpus: { type: Array, default: () => [] },
  selected: { type: Array, default: () => [] },
  mode: { type: String, default: 'inherit' },
  disabled: { type: Boolean, default: false },
})

const emit = defineEmits(['update:mode', 'update:selected'])

const gpuModeOptions = [
  { label: 'All GPUs', value: 'inherit' },
  { label: 'Choose GPUs', value: 'selected' },
  { label: 'CPU only', value: 'cpu' },
]

const activeMode = computed(() => props.mode || 'inherit')

const gpuCards = computed(() => {
  const selected = Array.isArray(props.selected) ? props.selected : []
  return (props.gpus || []).map((gpu) => {
    const order = selected.indexOf(gpu.value)
    return { ...gpu, order: order >= 0 ? order : null }
  })
})

function selectMode(mode) {
  if (props.disabled) return
  emit('update:mode', mode)
}

function toggleGpuCard(value) {
  if (props.disabled) return
  const current = Array.isArray(props.selected) ? [...props.selected] : []
  const index = current.indexOf(value)
  if (index >= 0) current.splice(index, 1)
  else current.push(value)
  emit('update:selected', current)
}

function onGpuModeKeydown(event) {
  if (props.disabled) return
  const keys = ['ArrowRight', 'ArrowDown', 'ArrowLeft', 'ArrowUp', 'Home', 'End']
  if (!keys.includes(event.key)) return
  event.preventDefault()
  const index = Math.max(
    0,
    gpuModeOptions.findIndex((mode) => mode.value === activeMode.value),
  )
  const next =
    event.key === 'Home'
      ? 0
      : event.key === 'End'
        ? gpuModeOptions.length - 1
        : (index +
            (['ArrowRight', 'ArrowDown'].includes(event.key) ? 1 : -1) +
            gpuModeOptions.length) %
          gpuModeOptions.length
  emit('update:mode', gpuModeOptions[next].value)
  event.currentTarget.querySelectorAll('[role="radio"]')[next]?.focus()
}
</script>

<style scoped>
.gpu-bind {
  display: flex;
  flex-direction: column;
  align-items: stretch;
  gap: 0.65rem;
}

.gpu-mode {
  display: flex;
  flex-wrap: wrap;
  gap: 0.4rem;
}

.gpu-mode__option {
  padding: 0.22rem 0.65rem;
  border-radius: var(--radius-md);
  border: 1px solid var(--border-primary);
  background: transparent;
  color: var(--text-secondary);
  font: inherit;
  font-size: 0.8125rem;
  font-weight: 600;
  cursor: pointer;
}

.gpu-mode__option.selected {
  border-color: var(--accent-cyan);
  background: rgba(34, 211, 238, 0.1);
  color: var(--accent-cyan);
}

.gpu-card-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(14rem, 1fr));
  gap: 0.5rem;
}

.gpu-card {
  display: flex;
  align-items: center;
  gap: 0.6rem;
  min-width: 0;
  padding: 0.45rem 0.65rem;
  border-radius: var(--radius-md);
  border: 1px solid var(--border-primary);
  background: transparent;
  color: inherit;
  font: inherit;
  text-align: left;
  cursor: pointer;
}

.gpu-card.selected {
  border-color: var(--accent-cyan);
  background: rgba(34, 211, 238, 0.1);
}

.gpu-card__index {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
  width: 1.5rem;
  height: 1.5rem;
  border-radius: var(--radius-sm);
  border: 1px solid var(--border-primary);
  background: var(--bg-tertiary);
  color: var(--text-secondary);
  font-size: var(--text-xs);
  font-weight: 700;
}

.gpu-card__body {
  display: flex;
  flex-direction: column;
  min-width: 0;
  gap: 0.1rem;
}

.gpu-card__name {
  font-size: 0.8125rem;
  font-weight: 600;
  color: var(--text-primary);
}

.gpu-card__meta {
  font-size: var(--text-xs);
  color: var(--text-secondary);
}
</style>
