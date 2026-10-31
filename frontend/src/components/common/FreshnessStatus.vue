<template>
  <span class="freshness-status">
    <span role="status">{{ status }}</span>
    <time v-if="updated" :datetime="new Date(updated).toISOString()">
      · checked {{ new Date(updated).toLocaleTimeString() }}</time
    >
  </span>
</template>
<script setup>
import { computed, onMounted, onUnmounted, ref } from 'vue'
const props = defineProps({ updated: Number, failed: Boolean, offline: Boolean })
const now = ref(Date.now())
let timer
onMounted(() => {
  timer = setInterval(() => {
    now.value = Date.now()
  }, 5000)
})
onUnmounted(() => clearInterval(timer))
const status = computed(() =>
  props.offline
    ? 'Offline · showing last received data'
    : props.failed
      ? 'Connection interrupted · retrying'
      : !props.updated
        ? 'Connecting…'
        : now.value - props.updated > 15000
          ? 'Stale · waiting for an update'
          : 'Live',
)
</script>
<style scoped>
.freshness-status {
  color: var(--text-secondary);
  font-size: 0.75rem;
}
</style>
