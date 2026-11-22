<template>
  <button
    type="button"
    class="engine-card"
    :disabled="disabled"
    aria-haspopup="dialog"
    @click="$emit('open')"
  >
    <div class="engine-card-head">
      <div class="engine-card-title">
        <template v-if="card">
          <span
            v-if="card.mark?.text"
            :class="['engine-mark', card.mark.className]"
            aria-hidden="true"
            >{{ card.mark.text }}</span
          >
          <i v-else :class="['engine-card-icon', card.mark?.icon]" aria-hidden="true" />
        </template>
        <slot v-else name="mark" />
        <div class="engine-card-name-group">
          <div class="engine-card-name">{{ displayTitle }}</div>
          <div class="engine-card-meta">
            {{ displayCount }} version{{ displayCount === 1 ? '' : 's' }}
          </div>
        </div>
        <Tag v-if="card?.badge" :value="card.badge.value" :severity="card.badge.severity" />
        <slot v-else name="badge" />
      </div>
    </div>
    <div class="engine-card-body">
      <template v-if="card">
        <div
          class="engine-card-version-line"
          :title="card.active ? card.active.version : undefined"
        >
          <Tag
            v-if="card.active"
            :value="card.activeVersionLabel"
            severity="success"
            class="engine-version-tag"
          />
          <EngineStatusTag v-else :versions="card.versions" :engine-id="card.id" />
        </div>
        <div
          v-if="card.update?.update_available"
          class="engine-card-status engine-card-status--warning"
        >
          Update available: {{ card.updateText }}
        </div>
        <div v-else class="engine-card-status">{{ card.subtitle }}</div>
      </template>
      <slot v-else />
    </div>
    <span class="engine-card-cta">{{ cta }}</span>
  </button>
</template>
<script setup>
import { computed } from 'vue'
import Tag from 'primevue/tag'
import EngineStatusTag from '@/components/system/EngineStatusTag.vue'

const props = defineProps({
  title: { type: String, default: '' },
  count: { type: Number, default: 0 },
  cta: String,
  disabled: Boolean,
  /** Runtime engine summary card; omitted by the generic engine family card. */
  card: { type: Object, default: null },
})
defineEmits(['open'])

const displayTitle = computed(() => (props.card ? props.card.title : props.title))
const displayCount = computed(() => (props.card ? props.card.count : props.count) || 0)
</script>

<style scoped>
.engine-card {
  display: flex;
  flex-direction: column;
  min-height: 11.5rem;
  padding: 1rem;
  appearance: none;
  border: 1px solid var(--border-primary);
  border-radius: var(--section-radius);
  background: var(--bg-card);
  box-shadow: var(--shadow-sm);
  color: inherit;
  font: inherit;
  text-align: left;
  cursor: pointer;
  transition:
    border-color 0.15s ease,
    transform 0.15s ease,
    background 0.15s ease;
}

.engine-card:hover {
  border-color: var(--accent-primary);
  background: var(--bg-surface);
}

.engine-card:focus {
  outline: none;
}

.engine-card:focus-visible {
  outline: 2px solid var(--accent-primary);
  outline-offset: 2px;
}

.engine-card:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.engine-card-head {
  display: flex;
  align-items: flex-start;
  justify-content: flex-start;
  gap: 0.75rem;
}

.engine-card-title {
  display: flex;
  align-items: center;
  gap: 0.65rem;
  min-width: 0;
}

.engine-mark {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  min-width: 1.8rem;
  height: 1.8rem;
  padding: 0 0.45rem;
  border: 1px solid var(--border-primary);
  border-radius: var(--radius-sm);
  background: var(--bg-tertiary);
  color: var(--text-secondary);
  font-size: var(--text-xs);
  font-weight: 700;
  line-height: 1;
  letter-spacing: 0.04em;
}

.engine-card-icon {
  display: inline-grid;
  flex-shrink: 0;
  place-items: center;
  width: 1.8rem;
  height: 1.8rem;
  border: 1px solid var(--border-primary);
  border-radius: var(--radius-sm);
  background: var(--bg-tertiary);
  color: var(--text-secondary);
  font-size: 0.875rem;
}

.engine-card-name-group {
  min-width: 0;
}

.engine-card-name {
  font-size: var(--text-base);
  font-weight: 680;
  line-height: 1.25;
}

.engine-card-meta {
  margin-top: 0.1rem;
  color: var(--text-secondary);
  font-size: var(--text-xs);
}

.engine-card-body {
  display: flex;
  flex-direction: column;
  gap: var(--section-header-gap);
  margin-top: 0.8rem;
  min-width: 0;
}

.engine-card-version-line {
  min-width: 0;
  max-width: 100%;
}

.engine-card-version-line :deep(.p-tag) {
  max-width: 100%;
}

.engine-card-version-line :deep(.p-tag-value) {
  display: block;
  overflow: hidden;
  font-size: var(--text-xs);
  text-overflow: ellipsis;
  white-space: nowrap;
}

.engine-card-status {
  color: var(--text-secondary);
  font-size: var(--text-sm);
  line-height: 1.4;
}

.engine-card-status--warning {
  color: var(--status-warning);
  font-weight: 600;
}

.engine-card-cta {
  display: inline-flex;
  align-items: center;
  align-self: flex-start;
  gap: 0.35rem;
  margin-top: auto;
  padding-top: 0.9rem;
  color: var(--accent-primary);
  font-size: var(--text-xs);
  font-weight: 700;
}

.engine-card-cta::after {
  content: '\2192';
  transition: transform var(--transition-fast);
}

.engine-card:hover .engine-card-cta::after {
  transform: translateX(0.2rem);
}

@media (max-width: 640px) {
  .engine-card {
    min-height: 0;
  }
}
</style>
