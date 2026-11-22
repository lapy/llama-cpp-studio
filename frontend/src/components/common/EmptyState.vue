<template>
  <div class="page-empty" :role="role">
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
  headingTag: { type: String, default: 'h2', validator: (value) => /^h[1-6]$/.test(value) },
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
  /** Use "alert" when the empty state is a failed load rather than no data. */
  role: {
    type: String,
    default: 'status',
    validator: (value) => ['status', 'alert', 'note'].includes(value),
  },
})
</script>
