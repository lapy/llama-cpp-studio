<template>
  <div
    :class="['page-toolbar', plain ? 'page-toolbar--plain' : '']"
    :role="label ? 'group' : undefined"
    :aria-label="label || undefined"
  >
    <div v-if="$slots.search" class="page-toolbar__search">
      <slot name="search" />
    </div>
    <div v-if="$slots.default" class="page-toolbar__group">
      <slot />
    </div>
    <div v-if="$slots.summary" class="page-toolbar__summary">
      <slot name="summary" />
    </div>
    <div v-if="$slots.actions" class="page-toolbar__actions">
      <slot name="actions" />
    </div>
  </div>
</template>

<script setup>
defineProps({
  /** Accessible name for the toolbar group, e.g. "Library filters". */
  label: {
    type: String,
    default: '',
  },
  /** Drop the surface chrome when the toolbar sits inside an existing card. */
  plain: {
    type: Boolean,
    default: false,
  },
})
</script>
