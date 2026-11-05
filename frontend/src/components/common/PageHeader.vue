<template>
  <header class="page-header">
    <div class="page-header__start">
      <slot name="start" />
      <div class="page-header__title-group">
        <div class="page-header__copy">
          <slot name="title">
            <h1 v-if="title" class="page-title">{{ title }}</h1>
          </slot>
        </div>
        <div v-if="$slots.meta" class="page-header__meta">
          <slot name="meta" />
        </div>
      </div>
      <details v-if="description" class="page-header__help" @keydown.esc.stop.prevent="closeHelp">
        <summary :aria-label="`About ${title || 'this page'}`">
          <i class="pi pi-info-circle" aria-hidden="true" />
        </summary>
        <p class="page-header__description">{{ description }}</p>
      </details>
    </div>
    <div v-if="$slots.actions" class="page-header__actions">
      <slot name="actions" />
    </div>
  </header>
</template>

<script setup>
function closeHelp(event) {
  event.currentTarget.open = false
  event.currentTarget.querySelector('summary')?.focus()
}

defineProps({
  title: {
    type: String,
    default: '',
  },
  description: {
    type: String,
    default: '',
  },
})
</script>
