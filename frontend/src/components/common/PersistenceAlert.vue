<template>
  <p v-if="notice" class="persistence-notice" role="status">
    <strong>{{ notice.summary }}.</strong>
    {{ notice.detail }}
    <button v-if="notice.retry" type="button" @click="$emit('retry')">Try again</button>
    <button v-if="notice.refresh" type="button" @click="$emit('refresh')">Refresh</button>
  </p>
</template>

<script setup>
defineProps({
  notice: { type: Object, default: null },
})
defineEmits(['retry', 'refresh'])
</script>
