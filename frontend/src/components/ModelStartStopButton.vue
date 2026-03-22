<template>
  <Button
    v-if="!isActive"
    icon="pi pi-play"
    text
    severity="success"
    size="small"
    aria-label="Start"
    :aria-busy="isStarting"
    v-tooltip.top="playTooltip"
    :loading="isStarting"
    :disabled="isStarting"
    @click="onStart"
  />
  <Button
    v-else
    icon="pi pi-stop"
    text
    severity="warning"
    size="small"
    aria-label="Stop"
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
})

const emit = defineEmits(['start', 'stop'])

const playTooltip = computed(() => (props.isStarting ? 'Starting…' : 'Start'))

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
