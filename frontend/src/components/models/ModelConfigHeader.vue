<template>
  <PageHeader
    :title="title"
    description="Review the engine, runtime parameters, names, and launch settings for this model."
  >
    <template #start>
      <Button
        icon="pi pi-arrow-left"
        text
        severity="secondary"
        aria-label="Back to Models"
        @click="$emit('back')"
      />
    </template>
    <template #meta>
      <Tag :value="formatLabel(model.format || 'gguf')" severity="info" />
      <Tag v-if="model.quantization" :value="model.quantization" severity="secondary" />
      <Tag v-if="model.family" :value="model.family" severity="secondary" />
      <Tag
        v-for="task in model.tasks || []"
        :key="task"
        :value="taskLabel(task)"
        severity="success"
      />
      <a
        v-if="model.huggingface_id"
        :href="`https://huggingface.co/${model.huggingface_id}`"
        target="_blank"
        rel="noopener noreferrer"
        class="config-header__repo"
      >
        <i class="pi pi-external-link" aria-hidden="true" /> {{ model.huggingface_id }}
      </a>
    </template>
    <template v-if="!loading" #actions>
      <Button
        v-if="canCheckForUpdates"
        label="Check for updates"
        icon="pi pi-refresh"
        size="small"
        severity="secondary"
        outlined
        :loading="refreshingModel"
        :disabled="refreshingModel"
        @click="$emit('check-for-updates')"
      />
      <Button
        v-if="isAudioEngine"
        label="Open Audio"
        icon="pi pi-volume-up"
        size="small"
        severity="secondary"
        outlined
        @click="$emit('open-audio')"
      />
    </template>
  </PageHeader>
</template>

<script setup>
import { computed } from 'vue'
import Button from 'primevue/button'
import Tag from 'primevue/tag'

import PageHeader from '@/components/common/PageHeader.vue'
import { formatLabel, taskLabel } from '@/utils/modelPresentation'

const props = defineProps({
  model: { type: Object, required: true },
  loading: { type: Boolean, default: false },
  canCheckForUpdates: { type: Boolean, default: false },
  refreshingModel: { type: Boolean, default: false },
  isAudioEngine: { type: Boolean, default: false },
})

defineEmits(['back', 'check-for-updates', 'open-audio'])

const title = computed(
  () => props.model.display_name || props.model.base_model_name || 'Model configuration',
)
</script>

<style scoped>
.config-header__repo {
  display: inline-flex;
  align-items: center;
  gap: 0.25rem;
  color: var(--accent-primary);
  font-size: var(--text-sm);
  text-decoration: none;
  overflow-wrap: anywhere;
}

.config-header__repo:hover {
  text-decoration: underline;
}
</style>
