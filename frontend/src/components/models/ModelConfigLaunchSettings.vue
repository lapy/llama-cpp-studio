<template>
  <div class="config-launch">
    <div v-if="showCompanionsCard" class="advanced-block">
      <h3 class="page-subheading">
        Companions
        <span
          class="pi pi-info-circle param-info"
          role="img"
          tabindex="0"
          aria-label="About companions"
          v-tooltip.top="
            'Attach mmproj, MTP, or DFlash weights. MTP and DFlash cannot be used together.'
          "
        />
      </h3>
      <div v-if="companionsLoading" class="companions-loading">Loading companion options…</div>
      <div v-else class="companions-grid">
        <div class="companion-field">
          <label class="page-form__label" for="config-mmproj">Projector (mmproj)</label>
          <div class="companion-field__row">
            <Select
              id="config-mmproj"
              v-model="selectedMmproj"
              :options="mmprojOptions"
              optionLabel="label"
              optionValue="value"
              class="w-full"
              :disabled="companionBusy"
              placeholder="None"
            />
            <Button
              label="Apply"
              icon="pi pi-save"
              size="small"
              severity="success"
              outlined
              :loading="companionBusyField === 'mmproj'"
              :disabled="companionBusy || !mmprojSelectionChanged"
              @click="applyCompanion('mmproj')"
            />
          </div>
        </div>
        <div class="companion-field">
          <label class="page-form__label" for="config-mtp">MTP draft</label>
          <div class="companion-field__row">
            <Select
              id="config-mtp"
              v-model="selectedMtp"
              :options="mtpOptions"
              optionLabel="label"
              optionValue="value"
              class="w-full"
              :disabled="companionBusy"
              placeholder="None"
              @update:model-value="onMtpSelected"
            />
            <Button
              label="Apply"
              icon="pi pi-save"
              size="small"
              severity="success"
              outlined
              :loading="companionBusyField === 'mtp'"
              :disabled="companionBusy || !mtpSelectionChanged"
              @click="applyCompanion('mtp')"
            />
          </div>
        </div>
        <div class="companion-field">
          <label class="page-form__label" for="config-dflash">DFlash draft</label>
          <div class="companion-field__row">
            <Select
              id="config-dflash"
              v-model="selectedDflash"
              :options="dflashOptions"
              optionLabel="label"
              optionValue="value"
              class="w-full"
              :disabled="companionBusy"
              placeholder="None"
              @update:model-value="onDflashSelected"
            />
            <Button
              label="Apply"
              icon="pi pi-save"
              size="small"
              severity="success"
              outlined
              :loading="companionBusyField === 'dflash'"
              :disabled="companionBusy || !dflashSelectionChanged"
              @click="applyCompanion('dflash')"
            />
          </div>
        </div>
        <div class="companion-field">
          <label class="page-form__label" for="config-dspark">DSpark draft</label>
          <div class="companion-field__row">
            <Select
              id="config-dspark"
              v-model="selectedDspark"
              :options="dsparkOptions"
              optionLabel="label"
              optionValue="value"
              class="w-full"
              :disabled="companionBusy"
              placeholder="None"
              @update:model-value="onDsparkSelected"
            />
            <Button
              label="Apply"
              icon="pi pi-save"
              size="small"
              severity="success"
              outlined
              :loading="companionBusyField === 'dspark'"
              :disabled="companionBusy || !dsparkSelectionChanged"
              @click="applyCompanion('dspark')"
            />
          </div>
        </div>
        <div v-if="(companionFiles.engram_files || []).length" class="companion-field">
          <label>Engram tables</label>
          <small class="config-muted-hint">
            These are swappable n-gram table shards; use the repo's set_split.py and rename the
            table to the target quant's table shard.
          </small>
          <ul class="engram-file-list">
            <li v-for="table in companionFiles.engram_files" :key="table.filename">
              <code>{{ table.label || table.filename }}</code>
              <span>{{ table.filename }}</span>
            </li>
          </ul>
        </div>
      </div>
    </div>

    <div class="page-form">
      <div class="page-form__field">
        <label class="page-form__label" for="config-stable-id">
          {{ stableIdLabel }}
        </label>
        <InputText
          id="config-stable-id"
          :model-value="stableModelId"
          readonly
          class="w-full"
          :aria-label="stableIdAria"
        />
        <p class="page-form__hint">
          Assigned by Studio and used by launch manifests, receipts, and runtime state.
        </p>
      </div>

      <div class="page-form__field">
        <label class="page-form__label" for="config-model-alias">
          {{ aliasLabel }}
          <span
            class="pi pi-info-circle param-info"
            role="img"
            tabindex="0"
            aria-label="About API alias"
            v-tooltip.top="aliasTooltip"
          />
        </label>
        <InputText
          id="config-model-alias"
          :model-value="modelAlias"
          placeholder="e.g. my-app-model"
          class="w-full"
          @update:model-value="emit('update:modelAlias', $event)"
        />
        <p class="page-form__hint">
          Optional friendly name clients send as <code>model</code>. Requests without it use the
          stable runtime id.
        </p>
      </div>
    </div>

    <RequestAliasesEditor
      :key="engine"
      :model-value="setParamsById"
      :base-name="requestAliasBase"
      @update:model-value="emit('update:setParamsById', $event)"
      @validity="emit('validity', $event)"
    />

    <div class="advanced-block">
      <h3 class="page-subheading">
        Custom arguments
        <span
          class="pi pi-info-circle param-info"
          role="img"
          tabindex="0"
          aria-label="About custom arguments"
          v-tooltip.top="'Raw CLI flags appended to the server command.'"
        />
      </h3>
      <Textarea
        :model-value="customArgs"
        rows="2"
        placeholder="e.g. --some-flag value --another-flag"
        class="w-full textarea-cli"
        autoResize
        @update:model-value="emit('update:customArgs', $event)"
      />
      <Button
        v-if="canImportCommand"
        label="Parse into parameters"
        icon="pi pi-sparkles"
        size="small"
        severity="secondary"
        outlined
        class="mt-2"
        type="button"
        @click="emit('parse-command', customArgs)"
      />
    </div>
    <div class="advanced-block">
      <h3 class="page-subheading">
        Environment
        <span
          class="pi pi-info-circle param-info"
          role="img"
          tabindex="0"
          aria-label="About environment variables"
          v-tooltip.top="
            'Environment variables passed to this model process. CUDA_VISIBLE_DEVICES is set from the GPU control when NVIDIA GPUs are detected. LLAMA_STUDIO_* keys are reserved and ignored.'
          "
        />
      </h3>
      <div v-for="item in envRows" :key="item.originalIndex" class="runtime-env-row">
        <InputText
          :model-value="item.row.key"
          placeholder="VAR_NAME"
          class="runtime-env-key"
          aria-label="Environment variable name"
          @update:model-value="
            (v) =>
              emit('update-env-row', { originalIndex: item.originalIndex, field: 'key', value: v })
          "
        />
        <Select
          :model-value="item.row.mode || 'set'"
          :options="envModeOptions"
          option-label="label"
          option-value="value"
          class="runtime-env-mode"
          aria-label="Environment variable mode"
          @update:model-value="
            (v) =>
              emit('update-env-row', { originalIndex: item.originalIndex, field: 'mode', value: v })
          "
        />
        <InputText
          v-if="(item.row.mode || 'set') !== 'unset'"
          :model-value="item.row.value"
          placeholder="value"
          class="runtime-env-value"
          aria-label="Environment variable value"
          @update:model-value="
            (v) =>
              emit('update-env-row', {
                originalIndex: item.originalIndex,
                field: 'value',
                value: v,
              })
          "
        />
        <Button
          icon="pi pi-trash"
          severity="danger"
          text
          rounded
          type="button"
          aria-label="Remove variable"
          @click="emit('remove-env-row', item.originalIndex)"
        />
      </div>
      <Button
        label="Add variable"
        icon="pi pi-plus"
        severity="secondary"
        outlined
        type="button"
        class="mt-2"
        @click="emit('add-env-row')"
      />
    </div>
    <div class="advanced-block">
      <h3 class="page-subheading">Command</h3>
      <div class="config-cmd-actions">
        <Button
          label="Live preview"
          icon="pi pi-eye"
          severity="secondary"
          outlined
          :loading="
            cmdPreview.unsavedLoading && cmdPreview.dialogVisible && cmdPreview.mode === 'unsaved'
          "
          @click="emit('open-cmd-preview', 'unsaved')"
        />
        <Button
          label="Saved command"
          icon="pi pi-terminal"
          severity="secondary"
          outlined
          :loading="cmdPreview.loading && cmdPreview.dialogVisible && cmdPreview.mode === 'saved'"
          @click="emit('open-cmd-preview', 'saved')"
        />
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed, ref, watch } from 'vue'
import Button from 'primevue/button'
import InputText from 'primevue/inputtext'
import Select from 'primevue/select'
import Textarea from 'primevue/textarea'
import RequestAliasesEditor from '@/components/RequestAliasesEditor.vue'
import { useToast } from 'primevue/usetoast'
import { useModelStore } from '@/stores/models'
import { ENGINE_DESCRIPTORS } from '@/engines/registry'
import { normalizeApiName } from '@/utils/modelPresentation'

const props = defineProps({
  model: { type: Object, default: null },
  engine: { type: String, default: '' },
  stableModelId: { type: String, default: '' },
  modelAlias: { type: String, default: '' },
  setParamsById: { type: Array, default: () => [] },
  customArgs: { type: String, default: '' },
  canImportCommand: { type: Boolean, default: false },
  envRows: { type: Array, default: () => [] },
  cmdPreview: { type: Object, default: () => ({}) },
})

const emit = defineEmits([
  'update:modelAlias',
  'update:setParamsById',
  'update:customArgs',
  'refresh-model',
  'validity',
  'parse-command',
  'open-cmd-preview',
  'add-env-row',
  'remove-env-row',
  'update-env-row',
])

const toast = useToast()
const modelStore = useModelStore()

const engineConfigCopy = computed(() => ENGINE_DESCRIPTORS[props.engine]?.configCopy || {})
const stableIdLabel = computed(() => engineConfigCopy.value.stableIdLabel || 'Stable ID')
const stableIdAria = computed(
  () => engineConfigCopy.value.stableIdAria || 'Stable Studio proxy model ID',
)
const aliasLabel = computed(() => engineConfigCopy.value.aliasLabel || 'API alias')
const aliasTooltip = computed(
  () =>
    engineConfigCopy.value.aliasTooltip ||
    'Optional id apps send as model. Must be unique. Running state uses the stable ID, not this alias.',
)
const requestAliasBase = computed(() => normalizeApiName(props.modelAlias, props.stableModelId))

const envModeOptions = [
  { label: 'Set value', value: 'set' },
  { label: 'Unset', value: 'unset' },
]

const companionsLoading = ref(false)
const companionBusyField = ref(null)
const companionFiles = ref({
  mmproj_files: [],
  mtp_files: [],
  dflash_files: [],
  dspark_files: [],
  engram_files: [],
})
const selectedMmproj = ref('')
const selectedMtp = ref('')
const selectedDflash = ref('')
const selectedDspark = ref('')

const modelFormat = computed(() =>
  String(props.model?.format || props.model?.model_format || 'gguf').toLowerCase(),
)
const showCompanionsCard = computed(
  () => modelFormat.value === 'gguf' && !!props.model?.huggingface_id,
)
const companionBusy = computed(() => !!companionBusyField.value)
const mmprojOptions = computed(() =>
  companionDropdownOptions(companionFiles.value.mmproj_files, 'mmproj'),
)
const mtpOptions = computed(() => companionDropdownOptions(companionFiles.value.mtp_files, 'mtp'))
const dflashOptions = computed(() =>
  companionDropdownOptions(companionFiles.value.dflash_files, 'dflash'),
)
const dsparkOptions = computed(() =>
  companionDropdownOptions(companionFiles.value.dspark_files, 'dspark'),
)
const mmprojSelectionChanged = computed(
  () => (selectedMmproj.value || '') !== (props.model?.mmproj_filename || ''),
)
const mtpSelectionChanged = computed(
  () => (selectedMtp.value || '') !== (props.model?.mtp_filename || ''),
)
const dflashSelectionChanged = computed(
  () => (selectedDflash.value || '') !== (props.model?.dflash_filename || ''),
)
const dsparkSelectionChanged = computed(
  () => (selectedDspark.value || '') !== (props.model?.dspark_filename || ''),
)

function formatAxiosDetail(e) {
  const d = e?.response?.data?.detail
  if (typeof d === 'string') return d
  if (d && typeof d === 'object' && typeof d.message === 'string') return d.message
  if (Array.isArray(d)) {
    return d
      .map((x) =>
        typeof x === 'object' && x?.msg
          ? `${Array.isArray(x.loc) ? x.loc.join('.') : ''}: ${x.msg}`.replace(/^\.\s*/, '')
          : String(x),
      )
      .filter(Boolean)
      .join('; ')
  }
  if (d && typeof d === 'object' && typeof d.msg === 'string') return d.msg
  return e?.message || 'Request failed'
}

function companionDropdownOptions(files = [], kind = 'mmproj') {
  const none = { label: 'None', value: '', size: 0 }
  const options = (files || []).map((file) => {
    const filename = file.filename || file.value || ''
    let label = file.label || filename
    if (kind === 'mmproj') {
      const m = filename.match(/(F16|F32|BF16|Q\d[_A-Z0-9]*)/i)
      label = m ? m[1].toUpperCase() : filename
    }
    return { label, value: filename, size: file.size || 0 }
  })
  options.sort((a, b) => a.label.localeCompare(b.label))
  return [none, ...options]
}

function syncCompanionSelectionsFromModel() {
  selectedMmproj.value = props.model?.mmproj_filename || ''
  selectedMtp.value = props.model?.mtp_filename || ''
  selectedDflash.value = props.model?.dflash_filename || ''
  selectedDspark.value = props.model?.dspark_filename || ''
}

async function loadCompanions() {
  if (!showCompanionsCard.value || !props.model?.id) {
    companionFiles.value = {
      mmproj_files: [],
      mtp_files: [],
      dflash_files: [],
      dspark_files: [],
      engram_files: [],
    }
    return
  }
  companionsLoading.value = true
  try {
    const data = await modelStore.fetchModelCompanions(props.model.id)
    companionFiles.value = {
      mmproj_files: data?.mmproj_files || [],
      mtp_files: data?.mtp_files || [],
      dflash_files: data?.dflash_files || [],
      dspark_files: data?.dspark_files || [],
      engram_files: data?.engram_files || [],
    }
    syncCompanionSelectionsFromModel()
  } catch (e) {
    companionFiles.value = {
      mmproj_files: [],
      mtp_files: [],
      dflash_files: [],
      dspark_files: [],
      engram_files: [],
    }
    toast.add({
      severity: 'warn',
      summary: 'Companions',
      detail: formatAxiosDetail(e) || 'Could not load companion files from Hugging Face.',
      life: 4000,
    })
  } finally {
    companionsLoading.value = false
  }
}

function onMtpSelected(value) {
  selectedMtp.value = value || ''
  if (selectedMtp.value) {
    selectedDflash.value = ''
    selectedDspark.value = ''
  }
}

function onDflashSelected(value) {
  selectedDflash.value = value || ''
  if (selectedDflash.value) {
    selectedMtp.value = ''
    selectedDspark.value = ''
  }
}

function onDsparkSelected(value) {
  selectedDspark.value = value || ''
  if (selectedDspark.value) {
    selectedMtp.value = ''
    selectedDflash.value = ''
  }
}

async function applyCompanion(kind) {
  if (!props.model?.id || companionBusy.value) return
  companionBusyField.value = kind
  try {
    let response
    if (kind === 'mmproj') {
      const opt = mmprojOptions.value.find((o) => o.value === (selectedMmproj.value || ''))
      response = await modelStore.updateModelProjector(
        props.model.id,
        selectedMmproj.value || null,
        opt?.size || 0,
      )
    } else if (kind === 'mtp') {
      const opt = mtpOptions.value.find((o) => o.value === (selectedMtp.value || ''))
      response = await modelStore.updateModelMtp(
        props.model.id,
        selectedMtp.value || null,
        opt?.size || 0,
      )
      if (selectedMtp.value) selectedDflash.value = ''
    } else if (kind === 'dflash') {
      const opt = dflashOptions.value.find((o) => o.value === (selectedDflash.value || ''))
      response = await modelStore.updateModelDflash(
        props.model.id,
        selectedDflash.value || null,
        opt?.size || 0,
      )
      if (selectedDflash.value) selectedMtp.value = ''
    } else {
      const opt = dsparkOptions.value.find((o) => o.value === (selectedDspark.value || ''))
      response = await modelStore.updateModelDspark(
        props.model.id,
        selectedDspark.value || null,
        opt?.size || 0,
      )
      if (selectedDspark.value) {
        selectedMtp.value = ''
        selectedDflash.value = ''
      }
    }

    await modelStore.fetchModels()
    emit('refresh-model')

    if (response?.applied) {
      toast.add({
        severity: 'success',
        summary: 'Companion updated',
        detail: response.message || 'Applied',
        life: 3000,
      })
    } else {
      toast.add({
        severity: 'success',
        summary: 'Download started',
        detail: response?.message || 'Track progress in notifications',
        life: 3000,
      })
    }
  } catch (e) {
    toast.add({
      severity: 'error',
      summary: 'Companion update failed',
      detail: formatAxiosDetail(e) || e.message,
      life: 4000,
    })
  } finally {
    companionBusyField.value = null
  }
}

watch(() => props.model, syncCompanionSelectionsFromModel, { immediate: true })
watch(
  () => props.model?.id,
  () => {
    void loadCompanions()
  },
  { immediate: true },
)
</script>

<style scoped>
.config-launch {
  display: flex;
  flex-direction: column;
  gap: 0.85rem;
  min-width: 0;
}

.runtime-env-row {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 0.5rem;
  margin-bottom: 0.5rem;
}

.runtime-env-row :deep(.runtime-env-key) {
  flex: 0 0 11rem;
  width: 11rem;
  max-width: 100%;
  min-width: 0;
}

.runtime-env-row :deep(.runtime-env-mode) {
  flex: 0 0 10.75rem;
  width: 10.75rem;
  max-width: 100%;
  min-width: 0;
}

.runtime-env-row :deep(.runtime-env-value) {
  flex: 1 1 12rem;
  width: auto;
  min-width: 8rem;
}

.config-cmd-actions {
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
}

.advanced-block {
  min-width: 0;
}

.config-launch > * + .advanced-block,
.config-launch > * + .page-form {
  padding-top: 0.75rem;
  border-top: 1px solid var(--border-primary);
}

.companions-loading {
  color: var(--text-secondary);
  font-size: 0.875rem;
}

.companions-grid {
  display: grid;
  grid-template-columns: 1fr;
  gap: 0.875rem;
}

@media (min-width: 900px) {
  .companions-grid {
    grid-template-columns: repeat(3, minmax(0, 1fr));
  }
}

.companion-field {
  min-width: 0;
}

.companion-field__row {
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
  align-items: center;
}

.companion-field__row .p-select {
  flex: 1;
  min-width: 0;
}

.engram-file-list {
  list-style: none;
  margin: 0.4rem 0 0;
  padding: 0;
  display: grid;
  gap: 0.3rem;
}

.engram-file-list li {
  display: flex;
  gap: 0.5rem;
  align-items: baseline;
  font-size: 0.75rem;
  color: var(--text-secondary);
  overflow-wrap: anywhere;
}

@media (max-width: 768px) {
  .companion-field__row {
    flex-wrap: wrap;
  }
}
</style>
