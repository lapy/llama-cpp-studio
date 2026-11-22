<template>
  <PageToolbar class="engine-check-updates-cta" label="Engine updates" plain>
    <p class="engine-check-updates-cta__hint">{{ hint }}</p>
    <template #actions>
      <Button
        :label="label"
        icon="pi pi-arrow-up-right"
        severity="secondary"
        outlined
        size="small"
        :loading="loading"
        @click="$emit('check')"
      />
    </template>
  </PageToolbar>
</template>

<script setup>
import Button from 'primevue/button'
import PageToolbar from '@/components/common/PageToolbar.vue'

defineProps({
  loading: { type: Boolean, default: false },
  /** Primary action label (same for all engines). */
  label: { type: String, default: 'Check for updates' },
  /** Supporting line next to the button. */
  hint: {
    type: String,
    default: 'Compare the installed version to the latest upstream release.',
  },
})

defineEmits(['check'])
</script>

<style scoped>
.engine-check-updates-cta__hint {
  margin: 0;
  padding-bottom: 0.35rem;
  color: var(--text-secondary);
  font-size: var(--text-sm);
  line-height: 1.45;
}
</style>
