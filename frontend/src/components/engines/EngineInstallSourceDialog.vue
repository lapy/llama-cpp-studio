<template>
  <Dialog v-model:visible="visible" :header="header" modal class="dialog-width-md">
    <div class="dialog-stack">
      <div class="page-form page-form--single">
        <div class="page-form__field">
          <label class="page-form__label" :for="repoFieldId">Repo URL</label>
          <InputText
            :id="repoFieldId"
            v-model="repo"
            :placeholder="repoPlaceholder"
            class="w-full"
          />
          <p v-if="repoHint" class="page-form__hint">{{ repoHint }}</p>
        </div>
        <div class="page-form__field">
          <label class="page-form__label" :for="refFieldId">{{ refLabel }}</label>
          <InputText
            :id="refFieldId"
            v-model="sourceRef"
            :placeholder="refPlaceholder"
            class="w-full"
          />
          <p v-if="refHint" class="page-form__hint">{{ refHint }}</p>
        </div>
      </div>
      <p v-if="note" class="page-form__hint">{{ note }}</p>
    </div>
    <template #footer>
      <div class="dialog-actions">
        <Button label="Cancel" severity="secondary" outlined @click="visible = false" />
        <Button
          :label="installLabel"
          icon="pi pi-code"
          severity="info"
          :loading="installing"
          :disabled="installing"
          @click="emit('install')"
        />
      </div>
    </template>
  </Dialog>
</template>

<script setup>
import { useId } from 'vue'
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

const repoFieldId = `engine-source-repo-${useId()}`
const refFieldId = `engine-source-ref-${useId()}`
</script>
