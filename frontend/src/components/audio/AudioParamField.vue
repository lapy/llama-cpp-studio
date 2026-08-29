<template>
  <Select
    v-if="options.length"
    :id="id"
    :model-value="modelValue"
    :options="options"
    optionLabel="label"
    optionValue="value"
    :placeholder="placeholder"
    :showClear="param.required !== true"
    class="param-input"
    :disabled="disabled"
    @update:model-value="$emit('update:modelValue', $event)"
  />
  <InputNumber
    v-else-if="param.type === 'int' || param.type === 'float'"
    :id="id"
    :model-value="modelValue"
    :min="param.minimum"
    :max="param.maximum"
    :minFractionDigits="param.type === 'float' ? 1 : 0"
    :maxFractionDigits="param.type === 'float' ? 6 : 0"
    class="param-input"
    :disabled="disabled"
    @update:model-value="$emit('update:modelValue', $event)"
  />
  <ToggleSwitch
    v-else-if="param.type === 'bool'"
    :input-id="id"
    :model-value="Boolean(modelValue)"
    :disabled="disabled"
    @update:model-value="$emit('update:modelValue', $event)"
  />
  <InputTags
    v-else-if="param.type === 'list' || param.value_kind === 'repeatable'"
    :id="id"
    :model-value="modelValue || []"
    delimiter=","
    class="param-input"
    :disabled="disabled"
    @update:model-value="$emit('update:modelValue', $event)"
  />
  <Textarea
    v-else-if="param.type === 'json'"
    :id="id"
    :model-value="jsonDisplay"
    rows="4"
    class="w-full textarea-cli param-input"
    :disabled="disabled"
    @update:model-value="$emit('update:json', $event)"
  />
  <InputText
    v-else
    :id="id"
    :model-value="modelValue"
    :placeholder="placeholder"
    class="param-input"
    :disabled="disabled"
    @update:model-value="$emit('update:modelValue', $event)"
  />
</template>

<script setup>
import { computed } from 'vue'
import Select from 'primevue/select'
import InputNumber from 'primevue/inputnumber'
import ToggleSwitch from 'primevue/toggleswitch'
import InputText from 'primevue/inputtext'
import InputTags from 'primevue/inputtags'
import Textarea from 'primevue/textarea'
import { jsonParamDisplay } from '@/composables/useAudioModelConfig'

const props = defineProps({
  id: { type: String, default: '' },
  param: { type: Object, required: true },
  modelValue: { type: [String, Number, Boolean, Array, Object], default: null },
  options: { type: Array, default: () => [] },
  disabled: { type: Boolean, default: false },
})

defineEmits(['update:modelValue', 'update:json'])

const placeholder = computed(() => (
  props.param.default != null ? String(props.param.default) : ''
))

const jsonDisplay = computed(() => jsonParamDisplay(props.modelValue))
</script>

<style scoped>
.param-input {
  width: 100%;
}
</style>
