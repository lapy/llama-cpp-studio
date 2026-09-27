<template>
  <Button
    v-if="!isActive"
    icon="pi pi-play"
    text
    severity="success"
    size="small"
    :aria-label="isActive ? undefined : startLabel"
    :label="showLabel && !isActive ? 'Start' : undefined"
    :aria-busy="isPlayBusy"
    v-tooltip.top="playTooltip"
    :loading="isPlayBusy"
    :disabled="isPlayBusy"
    @click="onStart"
  />
  <Button
    v-else
    icon="pi pi-stop"
    text
    severity="warning"
    size="small"
    :aria-label="stopLabel"
    :label="showLabel ? 'Stop' : undefined"
    :aria-busy="isStopBusy"
    v-tooltip.top="stopTooltip"
    :loading="isStopBusy"
    @click="onStop"
  />
</template>

<script setup>
import { computed } from 'vue'
import Button from 'primevue/button'

const props = defineProps({
  isActive: {
    type: Boolean,
    default: false,
  },
  /** Server reports model slot is loading (VRAM / weights). */
  isProxyLoading: {
    type: Boolean,
    default: false,
  },
  isStarting: {
    type: Boolean,
    default: false,
  },
  isStopping: {
    type: Boolean,
    default: false,
  },
  /** Use inside clickable rows/headers so clicks don’t bubble (e.g. expand/collapse). */
  stopPropagation: {
    type: Boolean,
    default: false,
  },
  name: {
    type: String,
    default: '',
  },
  showLabel: {
    type: Boolean,
    default: false,
  },
})

const emit = defineEmits(['start', 'stop'])

/** Play: busy while the HTTP start is in flight or the proxy slot is still loading weights. */
const isPlayBusy = computed(() => props.isStarting || props.isProxyLoading)

const startLabel = computed(() => props.name ? `Start ${props.name}` : 'Start')
const stopLabel = computed(() => props.name ? `Stop ${props.name}` : 'Stop')

const playTooltip = computed(() => {
  if (props.isStarting) return 'Starting…'
  if (props.isProxyLoading) return 'Loading model…'
  return 'Start'
})

const stopTooltip = computed(() => {
  if (props.isStopping) return 'Stopping…'
  if (props.isProxyLoading) return 'Loading model…'
  return 'Stop'
})

const isStopBusy = computed(() => props.isStopping || props.isProxyLoading)

function onStart(e) {
  if (props.stopPropagation) {
    e.stopPropagation()
  }
  emit('start')
}

function onStop(e) {
  if (props.stopPropagation) {
    e.stopPropagation()
  }
  emit('stop')
}
</script>
