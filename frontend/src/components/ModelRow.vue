<template>
  <div class="quant-row" :class="{ 'is-active': quant.is_active }">
    <div class="quant-info">
      <div class="quant-main">
        <div class="quant-heading">
          <code class="quant-name">{{ quant.quantization || quant.name }}</code>
          <span v-if="quant.file_size" class="file-size">
            {{ props.formatBytes(quant.file_size) }}
          </span>
          <Tag v-if="proxyStatus === 'loading'" value="Loading" severity="warn" />
          <Tag v-else-if="proxyStatus === 'ready'" value="Ready" severity="success" />
          <Tag v-else-if="quant.is_active" value="Running" severity="success" />
        </div>
        <div class="quant-tags">
          <Tag v-if="quant.family" :value="quant.family" severity="secondary" />
          <Tag
            v-for="task in quant.tasks || []"
            :key="`task-${task}`"
            :value="task"
            severity="info"
          />
          <Tag
            v-if="!quant.quantization && quant.format"
            :value="quant.format"
            severity="secondary"
          />
        </div>
      </div>
    </div>

    <div class="quant-actions">
      <ModelStartStopButton
        :is-active="quant.is_active"
        :is-proxy-loading="proxyStatus === 'loading'"
        :is-starting="isStarting"
        :is-stopping="isStopping"
        :name="quant.quantization || quant.name || ''"
        show-label
        stop-propagation
        @start="emit('start', quant.id)"
        @stop="emit('stop', quant.id)"
      />
      <Button
        v-if="isAudioModel"
        icon="pi pi-volume-up"
        text
        severity="secondary"
        size="small"
        :aria-label="`Open audio for ${quant.quantization || quant.name || 'model'}`"
        v-tooltip.top="'Audio'"
        @click="emit('audio', quant.id)"
      />
      <Button
        v-if="quant.is_active && !isAudioModel"
        label="Connect"
        icon="pi pi-link"
        size="small"
        :aria-label="`Connect ${quant.quantization || quant.name || 'model'}`"
        @click="emit('connect', quant)"
      />
      <Button
        label="Configure"
        icon="pi pi-cog"
        text
        severity="secondary"
        size="small"
        :aria-label="`Configure ${quant.quantization || quant.name || 'model'}`"
        @click="emit('configure', quant.id)"
      />
      <details class="row-menu">
        <summary :aria-label="`More actions for ${quant.quantization || quant.name || 'model'}`">More</summary>
        <button type="button" @click="emit('delete', quant.id)">Delete</button>
      </details>
    </div>

    <div
      v-if="quant.downloaded_at"
      class="quant-row__footer"
    >
      <span class="downloaded-at">
        Downloaded {{ props.formatDate(quant.downloaded_at) }}
      </span>
    </div>
  </div>
</template>

<script setup>
import { computed, toRefs } from 'vue'
import Button from 'primevue/button'
import Tag from 'primevue/tag'
import ModelStartStopButton from '@/components/ModelStartStopButton.vue'

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
  formatBytes: {
    type: Function,
    required: true,
  },
  formatDate: {
    type: Function,
    required: true,
  },
})

const { quant, isStarting, isStopping } = toRefs(props)
const proxyStatus = computed(() => String(quant.value?.status || quant.value?.run_state || '').toLowerCase())
const isAudioModel = computed(() => {
  const engine = quant.value?.config?.engine || quant.value?.engine
  return engine === 'audio_cpp' || quant.value?.format === 'audio_cpp'
})

const emit = defineEmits(['start', 'stop', 'configure', 'audio', 'delete', 'connect'])
</script>

<style scoped>
.row-menu {
  position: relative;
}

.row-menu summary {
  list-style: none;
  cursor: pointer;
  min-height: 2rem;
  border: 1px solid var(--border-primary);
  border-radius: var(--radius-md);
  padding: 0.2rem 0.55rem;
  font-size: 0.8rem;
}

.row-menu summary::-webkit-details-marker {
  display: none;
}

.row-menu button {
  display: block;
  margin-top: 0.25rem;
  color: var(--status-error);
  background: var(--bg-secondary);
  border: 1px solid var(--border-primary);
  border-radius: var(--radius-md);
  cursor: pointer;
  font: inherit;
  padding: 0.25rem 0.5rem;
}
</style>
