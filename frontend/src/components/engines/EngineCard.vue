<template>
  <button type="button" class="engine-card" :disabled="disabled" @click="$emit('open')">
    <div class="engine-card-head">
      <div class="engine-card-title">
        <template v-if="card">
          <span
            v-if="card.mark?.text"
            :class="['engine-mark', card.mark.className]"
            aria-hidden="true"
            >{{ card.mark.text }}</span
          >
          <i v-else :class="['engine-card-icon', card.mark?.icon]" />
        </template>
        <slot v-else name="mark" />
        <div>
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
