<template>
  <div v-if="hasContent" class="engine-active-status">
    <div
      v-for="(row, idx) in visibleRows"
      :key="`${row.label}-${idx}`"
      class="engine-active-status__row"
    >
      <span
        class="engine-active-status__label"
        :class="{ 'engine-active-status__label--error': row.error }"
      >{{ row.label }}</span>
      <Tag
        v-if="row.tag"
        :value="row.tag"
        :severity="row.tagSeverity || 'info'"
        v-tooltip.bottom="row.tagTooltip || undefined"
      />
      <code v-if="row.code != null && row.code !== ''">{{ row.code }}</code>
      <span v-else-if="row.text">{{ row.text }}</span>
    </div>
    <slot />
  </div>
</template>

<script setup>
import { computed, useSlots } from 'vue'
import Tag from 'primevue/tag'

const props = defineProps({
  /**
   * Rows: { label, code?, text?, tag?, tagSeverity?, tagTooltip?, error? }
   */
  rows: {
    type: Array,
    default: () => [],
  },
})

const slots = useSlots()

const visibleRows = computed(() =>
  (props.rows || []).filter((row) => {
    if (!row || !row.label) return false
    return Boolean(row.tag || (row.code != null && row.code !== '') || row.text || row.error)
  }),
)

const hasContent = computed(
  () => visibleRows.value.length > 0 || Boolean(slots.default),
)
</script>

<style scoped>
.engine-active-status {
  display: flex;
  flex-direction: column;
  gap: 0.45rem;
}

.engine-active-status__row {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 0.4rem 0.55rem;
  font-size: 0.8125rem;
  line-height: 1.4;
  color: var(--text-primary);
}

.engine-active-status__label {
  color: var(--text-secondary);
  flex-shrink: 0;
}

.engine-active-status__label--error {
  color: var(--accent-red, #ef4444);
}

.engine-active-status__row code {
  font-size: 0.75rem;
  word-break: break-all;
}
</style>
