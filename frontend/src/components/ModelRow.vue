<template>
  <div class="quant-row" :class="{ 'is-active': quant.is_active }">
    <div class="quant-info">
      <div class="quant-main">
        <code class="quant-name">{{ quant.quantization || quant.name }}</code>
        <Tag v-if="proxyStatus === 'loading'" value="Loading" severity="warning" />
        <Tag v-else-if="proxyStatus === 'ready'" value="Ready" severity="success" />
        <Tag v-else-if="quant.is_active" value="Running" severity="success" />
        <span v-if="quant.file_size" class="file-size">
          {{ props.formatBytes(quant.file_size) }}
        </span>
      </div>
    </div>

    <div class="quant-actions">
      <ModelStartStopButton
        :is-active="quant.is_active"
        :is-proxy-loading="proxyStatus === 'loading'"
        :is-starting="isStarting"
        :is-stopping="isStopping"
        stop-propagation
        @start="emit('start', quant.id)"
        @stop="emit('stop', quant.id)"
      />
      <Button
        icon="pi pi-cog"
        text
        severity="secondary"
        size="small"
        v-tooltip.top="'Configure'"
        @click="emit('configure', quant.id)"
      />
      <Button
        icon="pi pi-trash"
        text
        severity="danger"
        size="small"
        v-tooltip.top="'Delete'"
        @click="emit('delete', quant.id)"
      />
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

const emit = defineEmits(['start', 'stop', 'configure', 'delete'])
</script>
