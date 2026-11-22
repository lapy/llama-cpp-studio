<template>
  <div
    class="audio-result-region"
    role="status"
    aria-live="polite"
    :aria-busy="loading ? 'true' : 'false'"
  >
    <PageSection v-if="hasContent" title="Transcript">
      <StateBanner v-if="error" severity="error" icon="">{{ error }}</StateBanner>
      <ul v-if="batchResults.length" class="asr-batch-results">
        <li v-for="(row, index) in batchResults" :key="`${row.name}-${index}`">
          <strong>{{ row.name }}</strong>
          <pre class="audio-transcript">{{ row.text }}</pre>
        </li>
      </ul>
      <pre v-else-if="text" class="audio-transcript">{{ text }}</pre>
      <pre v-if="metadata" class="audio-transcript">{{ metadata }}</pre>
    </PageSection>
  </div>
</template>

<script setup>
import { computed } from 'vue'
import PageSection from '@/components/common/PageSection.vue'
import StateBanner from '@/components/common/StateBanner.vue'

const props = defineProps({
  error: { type: String, default: '' },
  loading: { type: Boolean, default: false },
  text: { type: String, default: '' },
  metadata: { type: String, default: '' },
  batchResults: { type: Array, default: () => [] },
})

const hasContent = computed(() =>
  Boolean(props.error || props.text || props.metadata || props.batchResults.length),
)
</script>

<style scoped>
.asr-batch-results {
  display: grid;
  gap: 0.5rem;
  margin: 0;
  padding: 0;
  list-style: none;
}

.asr-batch-results strong {
  font-size: var(--text-sm);
}

.audio-transcript {
  max-height: 20rem;
  margin: 0;
  padding: 0.75rem;
  overflow: auto;
  border: 1px solid var(--border-primary);
  border-radius: var(--radius-md);
  background: var(--surface-inset, color-mix(in srgb, var(--bg-surface) 65%, transparent));
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
  font-size: var(--text-sm);
  line-height: 1.45;
  white-space: pre-wrap;
  word-break: break-word;
}
</style>
