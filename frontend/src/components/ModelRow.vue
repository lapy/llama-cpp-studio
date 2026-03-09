<template>
  <div class="quant-row" :class="{ 'is-active': quant.is_active }">
    <div class="quant-info">
      <div class="quant-main">
        <code class="quant-name">{{ quant.quantization || quant.name }}</code>
        <Tag v-if="quant.is_active" value="Running" severity="success" />
      </div>
      <div class="quant-sub">
        <span v-if="quant.file_size" class="file-size">
          {{ props.formatBytes(quant.file_size) }}
        </span>
        <span v-if="quant.downloaded_at" class="downloaded-at">
          Downloaded {{ props.formatDate(quant.downloaded_at) }}
        </span>
      </div>
    </div>

    <div class="quant-actions">
      <Button
        v-if="!quant.is_active"
        label="Start"
        icon="pi pi-play"
        size="small"
        severity="success"
        outlined
        :loading="isStarting"
        @click="emit('start', quant.id)"
      />
      <Button
        v-else
        label="Stop"
        icon="pi pi-stop"
        size="small"
        severity="warning"
        outlined
        :loading="isStopping"
        @click="emit('stop', quant.id)"
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
  </div>
</template>

<script setup>
import Button from 'primevue/button'
import Tag from 'primevue/tag'

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

const { quant, isStarting, isStopping } = props

const emit = defineEmits(['start', 'stop', 'configure', 'delete'])
</script>

