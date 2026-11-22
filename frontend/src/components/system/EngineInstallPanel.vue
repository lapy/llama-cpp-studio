<template>
  <PageSection
    class="engine-install-panel"
    variant="compact"
    heading-tag="h3"
    :title="title"
    :description="subtitle"
  >
    <div class="engine-install-panel__actions">
      <slot />
    </div>
  </PageSection>
</template>

<script setup>
import PageSection from '@/components/common/PageSection.vue'

defineProps({
  title: { type: String, default: 'Install' },
  subtitle: { type: String, default: '' },
})
</script>

<style scoped>
.engine-install-panel {
  margin-bottom: 0;
}

.engine-install-panel__actions {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(min(100%, 11rem), 1fr));
  gap: 0.65rem;
}
</style>
