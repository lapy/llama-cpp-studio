<template>
  <Toast group="model-library" position="top-center" class="library-toast-host">
    <template #message="{ message }">
      <div class="library-toast">
        <i :class="['library-toast__icon', severityIcon(message.severity)]" aria-hidden="true" />
        <div class="library-toast__body">
          <span class="library-toast__summary">{{ message.summary }}</span>
          <div v-if="message.detail" class="library-toast__detail">{{ message.detail }}</div>
          <Button
            v-if="message.configureModelId"
            class="library-toast__configure"
            label="Configure"
            icon="pi pi-cog"
            size="small"
            text
            :aria-label="`Configure ${message.configureModelId}`"
            @click="emit('configure', message.configureModelId)"
          />
        </div>
      </div>
    </template>
  </Toast>
</template>

<script setup>
import Toast from 'primevue/toast'
import Button from 'primevue/button'

const emit = defineEmits(['configure'])

function severityIcon(severity) {
  if (severity === 'error') return 'pi pi-exclamation-circle'
  if (severity === 'warn') return 'pi pi-exclamation-triangle'
  if (severity === 'success') return 'pi pi-check-circle'
  return 'pi pi-info-circle'
}
</script>

<style scoped>
/* Keep the library action toast below the sticky header. */
:global(.library-toast-host.p-toast-top-center) {
  top: calc(3.5rem + 1rem) !important;
}

.library-toast {
  display: flex;
  align-items: flex-start;
  gap: 0.5rem;
  width: 100%;
}

.library-toast__icon {
  margin-top: 0.15rem;
  flex: 0 0 auto;
}

.library-toast__body {
  display: flex;
  flex-direction: column;
  gap: 0.15rem;
  min-width: 0;
  flex: 1 1 auto;
}

.library-toast__summary {
  font-weight: 600;
}

.library-toast__detail {
  font-size: var(--text-sm);
  color: var(--text-secondary);
  overflow-wrap: anywhere;
}

.library-toast__configure {
  align-self: flex-start;
  margin-top: 0.2rem;
  padding-left: 0;
  white-space: nowrap;
}
</style>
