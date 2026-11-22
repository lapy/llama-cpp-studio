<template>
  <Dialog
    :visible="visible"
    modal
    header="Import prepared audio.cpp bundle"
    class="dialog-width-md"
    @update:visible="$emit('update:visible', $event)"
  >
    <div class="dialog-stack">
      <p class="page-section__description">
        The directory is copied into managed storage and accepted only when the active
        <code>audiocpp_cli</code> identifies a valid family and task.
      </p>
      <div class="page-form page-form--single">
        <label class="page-form__field">
          <span class="page-form__label">Local directory</span>
          <InputText v-model="sourcePath" placeholder="/data/models/my-audio-model" fluid />
        </label>
        <label class="page-form__field">
          <span class="page-form__label">Package ID (optional)</span>
          <InputText v-model="packageId" placeholder="Defaults to the directory name" fluid />
        </label>
        <label class="page-form__field">
          <span class="page-form__label">Family override (optional)</span>
          <InputText v-model="family" placeholder="Detected automatically" fluid />
        </label>
      </div>
    </div>
    <template #footer>
      <div class="dialog-actions">
        <Button label="Cancel" severity="secondary" text @click="$emit('update:visible', false)" />
        <Button
          label="Validate and import"
          icon="pi pi-folder-open"
          :loading="loading"
          :disabled="!sourcePath"
          @click="submit"
        />
      </div>
    </template>
  </Dialog>
</template>

<script setup>
import { ref, watch } from 'vue'
import Dialog from 'primevue/dialog'
import Button from 'primevue/button'
import InputText from 'primevue/inputtext'

const props = defineProps({
  visible: { type: Boolean, default: false },
  loading: { type: Boolean, default: false },
})

const emit = defineEmits(['update:visible', 'import'])

const sourcePath = ref('')
const packageId = ref('')
const family = ref('')

watch(
  () => props.visible,
  (open) => {
    if (!open) return
    sourcePath.value = ''
    packageId.value = ''
    family.value = ''
  },
)

function submit() {
  if (!sourcePath.value) return
  emit('import', {
    source_path: sourcePath.value,
    package_id: packageId.value,
    family: family.value,
  })
}
</script>
