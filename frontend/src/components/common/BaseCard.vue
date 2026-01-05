<template>
  <div class="card" :class="cardClass">
    <div v-if="$slots.header" class="card-header">
      <slot name="header"></slot>
    </div>
    <div class="card-content">
      <slot></slot>
    </div>
    <div v-if="$slots.footer" class="card-footer">
      <slot name="footer"></slot>
    </div>
  </div>
</template>

<script setup>
defineProps({
  cardClass: {
    type: String,
    default: ''
  }
})
</script>

<style scoped>
.card-content {
  /* Content area styling handled by global .card class */
}

.card-footer {
  margin-top: var(--spacing-lg);
  padding-top: var(--spacing-md);
  border-top: 1px solid var(--border-primary);
}
</style>

