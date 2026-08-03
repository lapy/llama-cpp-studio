<template>
  <div v-if="hasContent" class="config-card">
    <div class="section-label">{{ title }}</div>
    <Message
      v-if="error"
      severity="error"
      :closable="false"
      class="config-scan-message"
    >
      {{ error }}
    </Message>
    <div
      v-for="clip in clips"
      :key="clip.id"
      class="task-audio-clip"
    >
      <div v-if="clip.label && clips.length > 1" class="section-hint">{{ clip.label }}</div>
      <audio :src="clip.url" controls class="audio-player" />
      <div class="audio-actions">
        <Button
          :label="downloadLabel(clip)"
          icon="pi pi-download"
          size="small"
          severity="secondary"
          outlined
          @click="$emit('download', clip)"
        />
      </div>
    </div>
    <pre v-if="text" class="audio-transcript">{{ text }}</pre>
  </div>
</template>

<script setup>
import { computed } from 'vue'
import Button from 'primevue/button'
import Message from 'primevue/message'

const props = defineProps({
  title: { type: String, default: 'Result' },
  error: { type: String, default: '' },
  clips: { type: Array, default: () => [] },
  text: { type: String, default: '' },
})

defineEmits(['download'])

const hasContent = computed(() =>
  Boolean(props.error || props.clips?.length || props.text),
)

function downloadLabel(clip) {
  if (props.clips.length > 1 && clip?.label) {
    return `Download ${clip.label}`
  }
  return 'Download WAV'
}
</script>

<style scoped>
.audio-actions {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 0.5rem;
  margin-top: 0.5rem;
}

.audio-player {
  display: block;
  width: 100%;
  max-width: 36rem;
}

.audio-transcript {
  margin: 0.75rem 0 0;
  padding: 0.75rem;
  border-radius: var(--radius-md, 0.5rem);
  border: 1px solid var(--border-primary, #2a2f45);
  background: rgba(0, 0, 0, 0.2);
  white-space: pre-wrap;
  word-break: break-word;
  font-size: 0.85rem;
  line-height: 1.45;
  max-height: 20rem;
  overflow: auto;
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
}

.task-audio-clip + .task-audio-clip {
  margin-top: 1rem;
  padding-top: 0.85rem;
  border-top: 1px solid var(--border-primary, #2a2f45);
}

.task-audio-clip:first-of-type {
  margin-top: 0.35rem;
}
</style>
