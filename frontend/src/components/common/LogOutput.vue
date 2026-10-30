<template>
  <pre
    ref="output"
    class="task-logs"
    tabindex="0"
    aria-label="Engine log output"
    @scroll="trackScroll"
    >{{ lines.length ? lines.join('\n') : emptyMessage }}</pre>
</template>

<script setup>
import { nextTick, ref, watch } from 'vue'
const props = defineProps({
  lines: { type: Array, default: () => [] },
  emptyMessage: { type: String, default: 'Waiting for output…' },
})
const output = ref(null)
const following = ref(true)
function trackScroll() {
  const el = output.value
  if (el) following.value = el.scrollHeight - el.scrollTop - el.clientHeight < 40
}
watch(
  () => props.lines,
  async () => {
    if (!following.value) return
    await nextTick()
    if (output.value) output.value.scrollTop = output.value.scrollHeight
  },
  { deep: true, immediate: true },
)
</script>

<style scoped>
.task-logs {
  margin: 0;
  padding: 0.75rem;
  border-radius: var(--radius-md, 0.5rem);
  border: 1px solid var(--border-primary, #2a2f45);
  background: var(--bg-card-hover, rgb(255 255 255 / 3%));
  color: var(--text-secondary, #d1d5db);
  font-size: 0.75rem;
  line-height: 1.5;
  white-space: pre-wrap;
  overflow-wrap: anywhere;
  max-height: min(42vh, 28rem);
  overflow: auto;
}
</style>
