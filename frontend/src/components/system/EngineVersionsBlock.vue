<template>
  <div class="engine-versions-block">
    <div class="engine-versions-block__heading">{{ heading }}</div>
    <slot />
  </div>
</template>

<script setup>
defineProps({
  heading: { type: String, default: 'Installed versions' },
})
</script>

<style scoped>
.engine-versions-block {
  display: flex;
  flex-direction: column;
  gap: 0.35rem;
}

.engine-versions-block__heading {
  font-size: 0.7rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: var(--text-secondary);
  margin: 0;
}
</style>
