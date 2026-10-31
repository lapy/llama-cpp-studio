<template>
  <div class="page-empty" role="status">
    <i :class="['page-empty__icon', icon]" aria-hidden="true" />
    <component :is="headingTag" class="page-empty__title">{{ title }}</component>
    <p v-if="description" class="page-empty__desc">{{ description }}</p>
    <div v-if="$slots.default" class="page-empty__actions">
      <slot />
    </div>
  </div>
</template>

<script setup>
defineProps({
  headingTag: { type: String, default: 'h2', validator: (value) => /^h[2-6]$/.test(value) },
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
