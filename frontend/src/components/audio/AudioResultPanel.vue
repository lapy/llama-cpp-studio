<template>
  <div
    class="audio-result-region"
    role="status"
    aria-live="polite"
    :aria-busy="loading ? 'true' : 'false'"
  >
    <PageSection v-if="hasContent" :title="title">
      <StateBanner v-if="error" severity="error" icon="">{{ error }}</StateBanner>
      <div v-for="clip in clips" :key="clip.id" class="task-audio-clip">
        <div v-if="clip.label && clips.length > 1" class="page-subheading">{{ clip.label }}</div>
        <audio :src="clip.url" controls class="audio-player" />
        <div class="clip-actions">
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
    </PageSection>
  </div>
</template>

<script setup>
import { computed } from 'vue'
import Button from 'primevue/button'
import PageSection from '@/components/common/PageSection.vue'
import StateBanner from '@/components/common/StateBanner.vue'

const props = defineProps({
  title: { type: String, default: 'Result' },
  error: { type: String, default: '' },
  clips: { type: Array, default: () => [] },
  text: { type: String, default: '' },
  loading: { type: Boolean, default: false },
})

defineEmits(['download'])

const hasContent = computed(() => Boolean(props.error || props.clips?.length || props.text))

function downloadLabel(clip) {
  if (props.clips.length > 1 && clip?.label) {
    return `Download ${clip.label}`
  }
  return 'Download WAV'
}
</script>

<style scoped>
.clip-actions {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 0.5rem;
  margin-top: 0.5rem;
}

.audio-player {
  display: block;
  width: 100%;
}

.audio-transcript {
  max-height: 20rem;
  margin: 0.75rem 0 0;
  padding: 0.75rem;
  overflow: auto;
  border: 1px solid var(--border-primary);
  border-radius: var(--radius-md);
  background: color-mix(in srgb, var(--bg-surface) 65%, transparent);
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
  font-size: var(--text-sm);
  line-height: 1.45;
  white-space: pre-wrap;
  word-break: break-word;
}

.task-audio-clip + .task-audio-clip {
  margin-top: 1rem;
  padding-top: 0.85rem;
  border-top: 1px solid var(--border-primary);
}

.task-audio-clip:first-of-type {
  margin-top: 0.35rem;
}
</style>
