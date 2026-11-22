<template>
  <StateBanner class="engine-note" :severity="severity">
    <slot />
    <template v-if="$slots.actions" #actions>
      <slot name="actions" />
    </template>
  </StateBanner>
</template>

<script setup>
/**
 * Inline engine guidance (V100/upstream notes). The visual treatment is
 * the shared StateBanner; this wrapper only fixes the engine severity scale
 * and keeps an actions slot.
 */
import StateBanner from '@/components/common/StateBanner.vue'

defineProps({
  severity: {
    type: String,
    default: 'info',
    validator: (value) => ['info', 'warning'].includes(value),
  },
})
</script>

<style scoped>
.engine-note {
  margin-bottom: 0;
  font-size: var(--text-sm);
  line-height: 1.5;
}

.engine-note :deep(code) {
  overflow-wrap: anywhere;
  font-size: var(--text-xs);
}
</style>
