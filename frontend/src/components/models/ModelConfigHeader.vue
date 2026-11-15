<template>
  <PageHeader>
    <template #start>
      <Button
        icon="pi pi-arrow-left"
        text
        severity="secondary"
        aria-label="Back to Models"
        @click="$emit('back')"
      />
    </template>
    <template #title>
      <div class="config-page-title">
        <h1 class="page-title">{{ model.display_name || model.base_model_name }}</h1>
        <div class="header-meta">
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
            class="hf-link"
          >
            <i class="pi pi-external-link" /> {{ model.huggingface_id }}
          </a>
        </div>
      </div>
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
      <Tag v-if="hasUnsavedChanges" value="Unsaved changes" severity="warn" class="unsaved-tag" />
    </template>
  </PageHeader>
</template>

<script setup>
import Button from 'primevue/button'
import Tag from 'primevue/tag'

import PageHeader from '@/components/common/PageHeader.vue'
import { formatLabel, taskLabel } from '@/utils/modelPresentation'

defineProps({
  model: { type: Object, required: true },
  loading: { type: Boolean, default: false },
  canCheckForUpdates: { type: Boolean, default: false },
  refreshingModel: { type: Boolean, default: false },
  isAudioEngine: { type: Boolean, default: false },
  hasUnsavedChanges: { type: Boolean, default: false },
})

defineEmits(['back', 'check-for-updates', 'open-audio'])
</script>

<style scoped>
.config-page-title {
  display: flex;
  flex-direction: row;
  flex-wrap: wrap;
  align-items: center;
  gap: 0.4rem 0.75rem;
  min-width: 0;
}

.header-meta {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  flex-wrap: wrap;
}

.hf-link {
  font-size: 0.875rem;
  color: var(--accent-cyan, #22d3ee);
  text-decoration: none;
  display: flex;
  align-items: center;
  gap: 0.25rem;
}

.hf-link:hover {
  text-decoration: underline;
}

.unsaved-tag {
  font-size: var(--text-xs);
}
</style>
