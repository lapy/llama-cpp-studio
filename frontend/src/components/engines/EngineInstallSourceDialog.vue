<template>
  <Dialog v-model:visible="visible" :header="header" modal class="dialog-width-md">
    <div class="dialog-body">
      <div class="form-field">
        <label>Repo URL</label>
        <InputText v-model="repo" :placeholder="repoPlaceholder" class="w-full" />
        <small v-if="repoHint">{{ repoHint }}</small>
      </div>
      <div class="form-field">
        <label>{{ refLabel }}</label>
        <InputText v-model="sourceRef" :placeholder="refPlaceholder" class="w-full" />
        <small v-if="refHint">{{ refHint }}</small>
      </div>
      <div v-if="note" class="form-field">
        <small>{{ note }}</small>
      </div>
    </div>
    <template #footer>
      <Button label="Cancel" severity="secondary" outlined @click="visible = false" />
      <Button
        :label="installLabel"
        icon="pi pi-code"
        severity="info"
        :loading="installing"
        :disabled="installing"
        @click="emit('install')"
      />
    </template>
  </Dialog>
</template>

<script setup>
import Button from 'primevue/button'
import Dialog from 'primevue/dialog'
import InputText from 'primevue/inputtext'

const visible = defineModel('visible', { type: Boolean, default: false })
const repo = defineModel('repo', { type: String, default: '' })
const sourceRef = defineModel('sourceRef', { type: String, default: '' })

defineProps({
  header: { type: String, required: true },
  repoPlaceholder: { type: String, default: '' },
  repoHint: { type: String, default: '' },
  refLabel: { type: String, default: 'Tag / branch / commit' },
  refPlaceholder: { type: String, default: 'main' },
  refHint: { type: String, default: '' },
  note: { type: String, default: '' },
  installLabel: { type: String, default: 'Build from source' },
  installing: Boolean,
})

const emit = defineEmits(['install'])
</script>

<style scoped>
/* Match the engine dialogs' field rhythm (the global .form-field is looser). */
.dialog-body {
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
}

.form-field {
  display: flex;
  flex-direction: column;
  gap: 0.25rem;
}

.form-field label {
  font-size: 0.875rem;
  font-weight: 500;
  color: var(--text-secondary);
}

.form-field small {
  font-size: var(--text-xs);
  color: var(--text-secondary);
}
</style>
