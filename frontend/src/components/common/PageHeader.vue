<template>
  <header class="page-header">
    <div class="page-header__start">
      <slot name="start" />
      <div class="page-header__title-group">
        <slot name="title">
          <h1 v-if="title" class="page-title">{{ title }}</h1>
        </slot>
        <div v-if="$slots.meta" class="page-header__meta">
          <slot name="meta" />
        </div>
      </div>
    </div>
    <div v-if="$slots.actions" class="page-header__actions">
      <slot name="actions" />
    </div>
  </header>
</template>

<script setup>
defineProps({
  title: {
    type: String,
    default: '',
  },
})
</script>
