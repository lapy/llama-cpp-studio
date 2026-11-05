<template>
  <section v-if="failed" role="alert" class="page-shell">
    <h1>This view could not be displayed</h1>
    <p>Your workspace is still available. Retry this view or return to Models.</p>
    <Button label="Retry view" @click="retry" />
    <RouterLink to="/models">Return to Models</RouterLink>
  </section>
  <slot v-else :key="generation" />
</template>
<script setup>
import { onErrorCaptured, ref } from 'vue'
import { RouterLink } from 'vue-router'
import Button from 'primevue/button'
const failed = ref(false)
const generation = ref(0)
onErrorCaptured(() => {
  failed.value = true
  return false
})
function retry() {
  generation.value++
  failed.value = false
}
</script>
