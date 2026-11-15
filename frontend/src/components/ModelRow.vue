<template>
  <div
    class="quant-row"
    :class="{ 'is-active': quant.is_active && runtimeState.key !== 'unknown' }"
  >
    <div class="quant-info">
      <label v-if="selectable" class="quant-select">
        <input
          type="checkbox"
          :checked="selected"
          :aria-label="`Select ${quant.quantization || quant.name || 'model'}`"
          @change="emit('select', quant.id)"
        />
      </label>
      <div class="quant-main">
        <div class="quant-heading">
          <code class="quant-name">{{ quant.quantization || quant.name }}</code>
          <span v-if="quant.file_size" class="file-size">
            {{ props.formatBytes(quant.file_size) }}
          </span>
          <Tag
            :value="runtimeState.label"
            :severity="runtimeState.severity"
            :title="runtimeState.detail"
            role="status"
          />
        </div>
        <div class="quant-tags">
          <Tag v-if="quant.family" :value="quant.family" severity="secondary" />
          <Tag
            v-for="task in quant.tasks || []"
            :key="`task-${task}`"
            :value="taskLabel(task)"
            severity="info"
          />
          <Tag
            v-if="!quant.quantization && quant.format"
            :value="formatLabel(quant.format)"
            severity="secondary"
          />
        </div>
      </div>
    </div>

    <div class="quant-actions">
      <ModelCardActions
        :model="quant"
        :name="quant.quantization || quant.name || ''"
        :is-starting="isStarting"
        :is-stopping="isStopping"
        stop-propagation
        @start="emit('start', quant.id)"
        @stop="emit('stop', quant.id)"
        @configure="emit('configure', quant.id)"
        @audio="emit('audio', quant.id)"
        @connect="emit('connect', quant)"
        @refresh="emit('refresh', quant.id)"
        @delete="emit('delete', quant.id)"
      />
    </div>

    <div v-if="quant.downloaded_at" class="quant-row__footer">
      <span class="downloaded-at"> Downloaded {{ props.formatDate(quant.downloaded_at) }} </span>
    </div>
  </div>
</template>

<script setup>
import { computed, toRefs } from 'vue'
import { modelRuntimeState, taskLabel, formatLabel } from '@/utils/modelPresentation'
import Tag from 'primevue/tag'
import ModelCardActions from '@/components/models/ModelCardActions.vue'

const props = defineProps({
  quant: {
    type: Object,
    required: true,
  },
  isStarting: {
    type: Boolean,
    default: false,
  },
  isStopping: {
    type: Boolean,
    default: false,
  },
  selectable: {
    type: Boolean,
    default: false,
  },
  selected: {
    type: Boolean,
    default: false,
  },
  formatBytes: {
    type: Function,
    required: true,
  },
  formatDate: {
    type: Function,
    required: true,
  },
})

const { quant } = toRefs(props)
const runtimeState = computed(() =>
  modelRuntimeState(quant.value, {
    isStarting: props.isStarting,
    isStopping: props.isStopping,
  }),
)

const emit = defineEmits([
  'start',
  'stop',
  'configure',
  'audio',
  'delete',
  'connect',
  'refresh',
  'select',
])
</script>

<style scoped>
.quant-select {
  display: inline-flex;
  align-items: center;
  flex: 0 0 auto;
  margin-right: 0.5rem;
}

.quant-select input {
  width: 1rem;
  height: 1rem;
  margin: 0;
  accent-color: var(--accent-cyan, #22d3ee);
}
</style>
