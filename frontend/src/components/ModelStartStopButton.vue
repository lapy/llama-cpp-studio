<template>
  <Button
    v-if="!isActive"
    icon="pi pi-play"
    text
    severity="success"
    size="small"
    aria-label="Start"
    v-tooltip.top="'Start'"
    :loading="isStarting"
    @click="onStart"
  />
  <Button
    v-else
    icon="pi pi-stop"
    text
    severity="warning"
    size="small"
    aria-label="Stop"
    v-tooltip.top="'Stop'"
    :loading="isStopping"
    @click="onStop"
  />
</template>

<script setup>
import Button from 'primevue/button'

const props = defineProps({
  isActive: {
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
