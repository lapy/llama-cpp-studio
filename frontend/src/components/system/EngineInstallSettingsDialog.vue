<template>
  <Dialog
    :visible="visible"
    :header="title"
    modal
    class="build-settings-dialog dialog-width-md"
    @update:visible="$emit('update:visible', $event)"
  >
    <div class="dialog-stack build-settings-body">
      <PersistenceAlert :notice="notice" @retry="$emit('retry')" @refresh="$emit('refresh')" />
      <StateBanner severity="info" role="note" class="build-settings-note">{{ note }}</StateBanner>
      <div class="page-form">
        <div class="page-form__field">
          <label class="page-form__label" :for="versionFieldId">
            {{ versionLabel }} (optional)
          </label>
          <InputText
            :id="versionFieldId"
            :model-value="form[versionKey] || ''"
            :placeholder="versionPlaceholder"
            class="w-full"
            @update:model-value="$emit('field', versionKey, $event)"
          />
          <p class="page-form__hint">{{ versionHint }}</p>
        </div>
        <div class="page-form__field">
          <label class="page-form__label" :for="repoFieldId">Source repo URL</label>
          <InputText
            :id="repoFieldId"
            :model-value="form.source_repo || ''"
            :placeholder="repoPlaceholder"
            class="w-full"
            @update:model-value="$emit('field', 'source_repo', $event)"
          />
        </div>
        <div class="page-form__field">
          <label class="page-form__label" :for="branchFieldId">Source branch</label>
          <InputText
            :id="branchFieldId"
            :model-value="form.source_branch || ''"
            placeholder="main"
            class="w-full"
            @update:model-value="$emit('field', 'source_branch', $event)"
          />
        </div>
        <fieldset v-if="compatibility?.show" class="page-form__field page-form__field--full">
          <legend class="page-form__label">Compatibility</legend>
          <p class="page-form__hint">{{ compatibility.note }}</p>
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
              <span class="page-form__hint">{{ option.hint }}</span>
            </span>
          </label>
        </fieldset>
      </div>
      <p v-if="extraNote" class="page-form__hint">{{ extraNote }}</p>
    </div>
    <template #footer>
      <div class="dialog-actions">
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
      </div>
    </template>
  </Dialog>
</template>

<script setup>
import { useId } from 'vue'
import Button from 'primevue/button'
import Checkbox from 'primevue/checkbox'
import Dialog from 'primevue/dialog'
import InputText from 'primevue/inputtext'
import PersistenceAlert from '@/components/common/PersistenceAlert.vue'
import StateBanner from '@/components/common/StateBanner.vue'

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

const idSuffix = useId()
const versionFieldId = `engine-settings-version-${idSuffix}`
const repoFieldId = `engine-settings-repo-${idSuffix}`
const branchFieldId = `engine-settings-branch-${idSuffix}`
</script>

<style scoped>
.build-settings-note {
  margin-bottom: 0;
  font-size: var(--text-sm);
}

fieldset.page-form__field {
  margin: 0;
  padding: 0;
  border: 0;
}

fieldset legend.page-form__label {
  padding: 0;
}

.compatibility-option {
  display: flex;
  gap: 0.5rem;
  align-items: flex-start;
  margin: 0.4rem 0;
}
</style>
