<template>
  <Dialog
    :visible="visible"
    :header="title"
    modal
    class="build-settings-dialog dialog-width-md"
    @update:visible="$emit('update:visible', $event)"
  >
    <div class="dialog-body build-settings-body">
      <PersistenceAlert :notice="notice" @retry="$emit('retry')" @refresh="$emit('refresh')" />
      <p class="build-note build-note--info">{{ note }}</p>
      <div class="form-field">
        <label>{{ versionLabel }} <span class="optional">(optional)</span></label>
        <InputText
          :model-value="form[versionKey] || ''"
          :placeholder="versionPlaceholder"
          class="w-full"
          @update:model-value="$emit('field', versionKey, $event)"
        />
        <small>{{ versionHint }}</small>
      </div>
      <div class="form-field">
        <label>Source repo URL</label>
        <InputText
          :model-value="form.source_repo || ''"
          :placeholder="repoPlaceholder"
          class="w-full"
          @update:model-value="$emit('field', 'source_repo', $event)"
        />
      </div>
      <div class="form-field">
        <label>Source branch</label>
        <InputText
          :model-value="form.source_branch || ''"
          placeholder="main"
          class="w-full"
          @update:model-value="$emit('field', 'source_branch', $event)"
        />
      </div>
      <div v-if="compatibility?.show" class="form-field compatibility-options">
        <label>Compatibility</label>
        <small class="compatibility-note">{{ compatibility.note }}</small>
        <label
          v-for="option in compatibility.options"
          :key="option.key"
          class="compatibility-option"
        >
          <Checkbox
            :model-value="Boolean(form[option.key])"
            binary
            @update:model-value="$emit('field', option.key, $event)"
          />
          <span>
            {{ option.label }}
            <small v-if="option.hint" class="compatibility-hint">{{ option.hint }}</small>
          </span>
        </label>
      </div>
      <div v-if="extraNote" class="form-field">
        <small>{{ extraNote }}</small>
      </div>
    </div>
    <template #footer>
      <Button
        label="Cancel"
        severity="secondary"
        outlined
        @click="$emit('update:visible', false)"
      />
      <Button
        label="Save settings"
        icon="pi pi-save"
        severity="secondary"
        :loading="saving"
        :disabled="saveDisabled"
        @click="$emit('save')"
      />
      <Button
        :label="installLabel"
        icon="pi pi-code"
        severity="info"
        :loading="installLoading"
        @click="$emit('install')"
      />
    </template>
  </Dialog>
</template>

<script setup>
import Button from 'primevue/button'
import Checkbox from 'primevue/checkbox'
import Dialog from 'primevue/dialog'
import InputText from 'primevue/inputtext'
import PersistenceAlert from '@/components/common/PersistenceAlert.vue'

defineProps({
  visible: { type: Boolean, default: false },
  title: { type: String, required: true },
  note: { type: String, required: true },
  versionKey: { type: String, required: true },
  versionLabel: { type: String, required: true },
  versionHint: { type: String, default: '' },
  versionPlaceholder: { type: String, default: 'Blank = latest' },
  repoPlaceholder: { type: String, default: '' },
  extraNote: { type: String, default: '' },
  compatibility: { type: Object, default: null },
  form: { type: Object, required: true },
  notice: { type: Object, default: null },
  saving: { type: Boolean, default: false },
  saveDisabled: { type: Boolean, default: false },
  installLoading: { type: Boolean, default: false },
  installLabel: { type: String, default: 'Install from source' },
})

defineEmits(['update:visible', 'field', 'save', 'install', 'retry', 'refresh'])
</script>

<style scoped>
.compatibility-note {
  display: block;
  margin-bottom: 0.5rem;
  color: var(--text-color-secondary);
}
.compatibility-option {
  display: flex;
  gap: 0.5rem;
  align-items: flex-start;
  margin: 0.4rem 0;
}
.compatibility-hint {
  display: block;
  color: var(--text-color-secondary);
}
</style>
