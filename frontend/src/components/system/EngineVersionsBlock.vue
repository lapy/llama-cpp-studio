<template>
  <PageSection class="engine-versions-block" variant="plain" heading-tag="h3" :title="heading">
    <slot />
  </PageSection>
</template>

<script setup>
import PageSection from '@/components/common/PageSection.vue'

defineProps({
  heading: { type: String, default: 'Installed versions' },
})
</script>

<style scoped>
.engine-versions-block {
  margin: 0;
}

.engine-versions-block :deep(.page-section__header) {
  margin-bottom: 0.35rem;
}
</style>
