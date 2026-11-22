<template>
  <dl v-if="hasContent" class="engine-active-status">
    <div
      v-for="(row, idx) in visibleRows"
      :key="`${row.label}-${idx}`"
      class="engine-active-status__row"
    >
      <dt
        class="engine-active-status__label"
        :class="{ 'engine-active-status__label--error': row.error }"
      >
        {{ row.label }}
      </dt>
      <dd class="engine-active-status__value">
        <Tag
          v-if="row.tag"
          :value="row.tag"
          :severity="row.tagSeverity || 'info'"
          v-tooltip.bottom="row.tagTooltip || undefined"
        />
        <code v-if="row.code != null && row.code !== ''">{{ row.code }}</code>
        <span v-else-if="row.text">{{ row.text }}</span>
      </dd>
    </div>
    <slot />
  </dl>
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

const hasContent = computed(() => visibleRows.value.length > 0 || Boolean(slots.default))
</script>

<style scoped>
.engine-active-status {
  display: flex;
  flex-direction: column;
  gap: 0.45rem;
  margin: 0;
}

.engine-active-status__row {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 0.4rem 0.55rem;
  color: var(--text-primary);
  font-size: var(--text-sm);
  line-height: 1.4;
}

.engine-active-status__label {
  flex-shrink: 0;
  color: var(--text-secondary);
  font-weight: 400;
}

.engine-active-status__label--error {
  color: var(--status-error);
}

.engine-active-status__value {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 0.4rem;
  margin: 0;
  min-width: 0;
}

.engine-active-status__value code {
  overflow-wrap: anywhere;
  font-size: var(--text-xs);
}
</style>
