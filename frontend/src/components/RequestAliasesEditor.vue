<template>
  <section class="request-aliases" aria-labelledby="request-aliases-heading">
    <div class="request-aliases__heading">
      <h3 id="request-aliases-heading">Request parameter aliases</h3>
      <Button
        label="Add alias"
        icon="pi pi-plus"
        size="small"
        severity="secondary"
        outlined
        @click="add"
      />
    </div>
    <p class="request-aliases__help">
      Use the same model with different request parameters. An alias overrides matching values in
      the request without loading another model. Saved aliases apply to new JSON inference requests.
      Set options such as thinking_effort in the JSON object. Use dotted keys for nested options,
      such as
      <code>chat_template_kwargs.enable_thinking</code>.
    </p>
    <div v-for="(row, index) in rows" :key="row.id" class="request-aliases__row">
      <div class="request-aliases__heading">
        <label :for="`request-alias-${row.id}`">Alias suffix</label>
        <Button
          icon="pi pi-trash"
          severity="danger"
          text
          :aria-label="`Remove alias ${index + 1}`"
          @click="remove(index)"
        />
      </div>
      <InputText
        :id="`request-alias-${row.id}`"
        v-model="row.sub_id"
        placeholder="e.g. thinking"
        class="w-full"
        @update:model-value="publish"
      />
      <small
        >Model ID: <code>{{ baseName }}{{ row.sub_id.trim() ? `:${row.sub_id.trim()}` : '' }}</code
        >. An empty suffix customizes the base name.</small
      >
      <label :for="`request-alias-params-${row.id}`">Request overrides (JSON object)</label>
      <Textarea
        :id="`request-alias-params-${row.id}`"
        v-model="row.json"
        rows="4"
        class="w-full textarea-cli"
        :aria-invalid="Boolean(errors[index])"
        :aria-describedby="errors[index] ? `request-alias-error-${row.id}` : undefined"
        @update:model-value="publish"
      />
      <small
        v-if="errors[index]"
        :id="`request-alias-error-${row.id}`"
        role="alert"
        class="request-aliases__error"
        >{{ errors[index] }}</small
      >
    </div>
  </section>
</template>
<script setup>
import { computed, ref, watch } from 'vue'
import Button from 'primevue/button'
import InputText from 'primevue/inputtext'
import Textarea from 'primevue/textarea'
const props = defineProps({
  modelValue: { type: Array, default: () => [] },
  baseName: { type: String, default: '' },
})
const emit = defineEmits(['update:modelValue', 'validity'])
const rows = ref([])
let serial = 0
let lastPublished = ''
const errors = computed(() =>
  rows.value.map((row) => {
    const suffix = row.sub_id.trim()
    if (suffix && !/^[a-zA-Z0-9._-]+$/.test(suffix))
      return 'Use letters, numbers, dots, underscores, or hyphens in the suffix.'
    if (rows.value.filter((other) => other.sub_id.trim() === suffix).length > 1)
      return 'Each alias suffix must be unique.'
    try {
      const value = JSON.parse(row.json)
      if (!value || typeof value !== 'object' || Array.isArray(value)) return 'Enter a JSON object.'
      if (Object.keys(value).some((key) => key.replace(/\?$/, '').split('.')[0] === 'model'))
        return 'The model parameter is managed by Studio.'
    } catch {
      return 'Enter valid JSON. Strings need double quotes.'
    }
    return ''
  }),
)
watch(
  () => props.modelValue,
  (value) => {
    if (JSON.stringify(value) === lastPublished && errors.value.every((error) => !error)) return
    rows.value = (value || []).map((item) => ({
      id: ++serial,
      sub_id: item.sub_id || '',
      json: JSON.stringify(item.params || {}, null, 2),
    }))
  },
  { immediate: true, deep: true },
)
watch(
  errors,
  (value) =>
    emit(
      'validity',
      value.every((error) => !error),
    ),
  { immediate: true },
)
function publish() {
  if (errors.value.some(Boolean)) return
  const value = rows.value.map((row) => ({
    sub_id: row.sub_id.trim(),
    params: JSON.parse(row.json),
  }))
  lastPublished = JSON.stringify(value)
  emit('update:modelValue', value)
}
function add() {
  let suffix = 'thinking'
  let count = 2
  while (rows.value.some((row) => row.sub_id === suffix)) suffix = `thinking-${count++}`
  rows.value.push({
    id: ++serial,
    sub_id: suffix,
    json: '{}',
  })
  publish()
}
function remove(index) {
  rows.value.splice(index, 1)
  publish()
}
</script>
<style scoped>
.request-aliases {
  display: grid;
  gap: 0.75rem;
}
.request-aliases__heading {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 0.75rem;
}
h3,
p {
  margin: 0;
}
h3 {
  font-size: 1rem;
}
.request-aliases__help,
small {
  color: var(--text-secondary);
  font-size: var(--text-xs);
  overflow-wrap: anywhere;
}
.request-aliases__row {
  display: grid;
  gap: 0.5rem;
  padding: 1rem;
  border: 1px solid var(--border-primary);
  border-radius: var(--radius-md);
  min-width: 0;
}
.request-aliases__error {
  color: var(--accent-red);
}
</style>
