<template>
  <div class="page-empty" role="status">
    <i :class="['page-empty__icon', icon]" aria-hidden="true" />
    <h3 class="page-empty__title">{{ title }}</h3>
    <p v-if="description" class="page-empty__desc">{{ description }}</p>
    <div v-if="$slots.default" class="page-empty__actions">
      <slot />
    </div>
  </div>
</template>

<script setup>
defineProps({
  icon: {
    type: String,
    default: 'pi pi-inbox',
  },
  title: {
    type: String,
    required: true,
  },
  description: {
    type: String,
    default: '',
  },
})
</script>
