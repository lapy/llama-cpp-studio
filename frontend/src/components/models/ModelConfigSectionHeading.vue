<template>
  <div class="config-section-heading">
    <h2 class="config-section-heading__title">{{ title }}</h2>
    <p v-if="description" class="config-section-heading__description">{{ description }}</p>
  </div>
</template>

<script setup>
defineProps({
  title: { type: String, required: true },
  description: { type: String, default: '' },
})
</script>

<style scoped>
.config-section-heading {
  display: flex;
  flex-direction: column;
  gap: 0.2rem;
}

.config-section-heading__title {
  margin: 0;
  font-size: clamp(1rem, 1.2vw, 1.15rem);
  font-weight: 600;
  letter-spacing: -0.01em;
  color: var(--text-primary);
}

.config-section-heading__description {
  margin: 0;
  max-width: 72ch;
  font-size: var(--text-sm);
  line-height: 1.5;
  color: var(--text-secondary);
}
</style>
