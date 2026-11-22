<template>
  <Dialog
    :visible="visible"
    modal
    :header="dialogHeader"
    class="dialog-width-md"
    @update:visible="$emit('update:visible', $event)"
  >
    <div class="dialog-stack">
      <p class="page-section__description">{{ help }}</p>
      <div class="page-form page-form--single">
        <template v-if="needsExternalInputs">
          <label class="page-form__field">
            <span class="page-form__label">
              Source file {{ inputsRequired ? '' : '(optional)' }}
            </span>
            <InputText v-model="options.source_file" placeholder="/data/input/model.pt" fluid />
            <span class="page-form__hint">
              Use a source file for converters that transform a local checkpoint.
            </span>
          </label>
          <div class="field-divider">or</div>
          <label class="page-form__field">
            <span class="page-form__label">
              Source directory {{ inputsRequired ? '' : '(optional)' }}
            </span>
            <InputText
              v-model="options.source_dir"
              placeholder="/data/input/prepared-source"
              fluid
            />
          </label>
          <label class="page-form__field">
            <span class="page-form__label">Output file (optional)</span>
            <InputText
              v-model="options.output_file"
              placeholder="/data/output/model.safetensors"
              fluid
            />
          </label>
          <label class="page-form__field">
            <span class="page-form__label">Variant (optional)</span>
            <InputText v-model="options.variant" placeholder="Package-specific variant" fluid />
          </label>
        </template>
        <label class="page-form__field">
          <span class="page-form__label">Family override (optional)</span>
          <InputText
            v-model="options.family"
            placeholder="Detected automatically from package / config.json"
            fluid
          />
          <span class="page-form__hint">
            Leave blank unless the detected audio family is wrong. Repository code is never
            executed.
          </span>
        </label>
      </div>
    </div>
    <template #footer>
      <div class="dialog-actions">
        <Button label="Cancel" severity="secondary" text @click="cancel" />
        <Button
          label="Start install"
          icon="pi pi-download"
          :disabled="inputsRequired && !options.source_file && !options.source_dir"
          @click="confirm"
        />
      </div>
    </template>
  </Dialog>
</template>

<script setup>
import { computed, reactive, watch } from 'vue'
import Dialog from 'primevue/dialog'
import Button from 'primevue/button'
import InputText from 'primevue/inputtext'

const props = defineProps({
  visible: { type: Boolean, default: false },
  result: { type: Object, default: null },
  variant: { type: Object, default: null },
})

const emit = defineEmits(['update:visible', 'confirm', 'cancel'])

const options = reactive({
  source_file: '',
  source_dir: '',
  output_file: '',
  variant: '',
  family: '',
})

const inputsRequired = computed(() => Boolean(props.variant?.external_inputs_required))
const needsExternalInputs = computed(() =>
  Boolean(props.variant?.external_inputs_required || props.variant?.external_inputs_optional),
)

const dialogHeader = computed(() =>
  needsExternalInputs.value ? 'Prepare audio.cpp package' : 'Install audio.cpp package',
)

const help = computed(() => {
  const variant = props.variant
  if (!variant) {
    return 'This package installs through the active audio.cpp model manager. Repository code is never executed.'
  }
  if (variant.external_inputs_required) {
    return (
      variant.operation_description ||
      'This converter requires a local source file or directory. Repository code is never executed.'
    )
  }
  if (variant.external_inputs_optional) {
    return (
      variant.operation_description ||
      'Optional local source override for this converter. Leave blank to download the default upstream assets via the legacy model manager.'
    )
  }
  return (
    variant.method_hint ||
    'Confirm install options. Family is usually auto-detected; override only if inspect fails.'
  )
})

watch(
  () => props.visible,
  (open) => {
    if (!open) return
    options.source_file = ''
    options.source_dir = ''
    options.output_file = ''
    options.variant = ''
    options.family = props.result?.family || ''
  },
  { immediate: true },
)

function confirm() {
  const payload = Object.fromEntries(
    Object.entries(options).filter(([, value]) => String(value || '').trim()),
  )
  emit('confirm', payload)
}

function cancel() {
  emit('cancel')
  emit('update:visible', false)
}
</script>

<style scoped>
.field-divider {
  margin: -0.25rem 0;
  color: var(--text-secondary);
  font-size: var(--text-xs);
  text-align: center;
  text-transform: uppercase;
}
</style>
