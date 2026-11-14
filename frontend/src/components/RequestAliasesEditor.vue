<template>
  <section class="request-aliases" aria-labelledby="request-aliases-heading">
    <div class="request-aliases__heading">
      <div>
        <h3 id="request-aliases-heading">Request parameter aliases</h3>
        <p class="request-aliases__help">
          Expose the same model under another id with request overrides. Aliases apply to new JSON
          inference requests without loading another model.
        </p>
      </div>
      <Button
        label="Add alias"
        icon="pi pi-plus"
        size="small"
        severity="secondary"
        outlined
        @click="add"
      />
    </div>

    <p v-if="!rows.length" class="request-aliases__empty">
      No request aliases. Add one to preset parameters such as thinking mode or a chat template.
    </p>

    <article v-for="(row, index) in rows" :key="row.id" class="request-aliases__card">
      <header class="request-aliases__card-head">
        <div class="request-aliases__identity">
          <label :for="`request-alias-${row.id}`">Alias suffix</label>
          <InputText
            :id="`request-alias-${row.id}`"
            v-model="row.sub_id"
            placeholder="e.g. thinking"
            class="request-aliases__suffix"
            @update:model-value="publish"
          />
          <small>
            Model ID:
            <code>{{ baseName }}{{ row.sub_id.trim() ? `:${row.sub_id.trim()}` : '' }}</code>
          </small>
        </div>
        <Button
          icon="pi pi-trash"
          severity="danger"
          text
          size="small"
          :aria-label="`Remove alias ${index + 1}`"
          @click="remove(index)"
        />
      </header>

      <div class="request-aliases__overrides">
        <div class="request-aliases__overrides-head">
          <span>Request overrides</span>
          <div class="request-aliases__override-actions">
            <Button
              label="Add override"
              icon="pi pi-plus"
              size="small"
              severity="secondary"
              text
              @click="addParam(row)"
            />
            <Button
              label="Add chat template"
              icon="pi pi-comment"
              size="small"
              severity="secondary"
              text
              @click="addChatTemplate(row)"
            />
          </div>
        </div>

        <p v-if="!row.params.length" class="request-aliases__hint">
          No overrides yet. Use <strong>Add chat template</strong> to create a common override such
          as <code>enable_thinking</code>, or add any request parameter.
        </p>

        <div
          v-for="param in row.params"
          :key="param.id"
          class="request-aliases__override"
          :class="{ 'request-aliases__override--invalid': paramError(row, param) }"
        >
          <div class="request-aliases__override-main">
            <Select
              v-model="param.scope"
              :options="scopeOptions"
              option-label="label"
              option-value="value"
              aria-label="Override target"
              class="request-aliases__select request-aliases__select--scope"
              @update:model-value="onParamChanged(row, param)"
            />
            <div class="request-aliases__key">
              <span v-if="param.scope === 'chat_template'" class="request-aliases__key-prefix"
                >chat_template_kwargs.</span
              >
              <InputText
                v-model="param.name"
                :placeholder="
                  param.scope === 'chat_template' ? 'enable_thinking' : 'e.g. thinking_effort'
                "
                :aria-label="`Override name for alias ${index + 1}`"
                class="request-aliases__key-input"
                @update:model-value="publish"
              />
            </div>
            <Select
              v-model="param.type"
              :options="typeOptions"
              option-label="label"
              option-value="value"
              aria-label="Override value type"
              class="request-aliases__select request-aliases__select--type"
              @update:model-value="onParamChanged(row, param)"
            />
            <div class="request-aliases__value">
              <Select
                v-if="param.type === 'boolean'"
                v-model="param.value"
                :options="booleanOptions"
                option-label="label"
                option-value="value"
                aria-label="Boolean value"
                class="w-full"
                @update:model-value="publish"
              />
              <Textarea
                v-else-if="param.type === 'json'"
                v-model="param.value"
                rows="2"
                class="w-full textarea-cli"
                aria-label="JSON value"
                @update:model-value="publish"
              />
              <InputText
                v-else
                v-model="param.value"
                :inputmode="param.type === 'number' ? 'decimal' : undefined"
                :placeholder="param.type === 'number' ? 'e.g. 1024' : 'Value'"
                aria-label="Override value"
                class="w-full"
                @update:model-value="publish"
              />
            </div>
            <Button
              icon="pi pi-times"
              severity="danger"
              text
              size="small"
              :aria-label="`Remove override ${param.name || 'row'}`"
              @click="removeParam(row, param)"
            />
          </div>
          <small v-if="paramError(row, param)" class="request-aliases__error" role="alert">
            {{ paramError(row, param) }}
          </small>
        </div>
      </div>

      <small v-if="aliasError(row)" class="request-aliases__error" role="alert">
        {{ aliasError(row) }}
      </small>
    </article>
  </section>
</template>

<script setup>
import { computed, ref, watch } from 'vue'
import Button from 'primevue/button'
import InputText from 'primevue/inputtext'
import Select from 'primevue/select'
import Textarea from 'primevue/textarea'

const props = defineProps({
  modelValue: { type: Array, default: () => [] },
  baseName: { type: String, default: '' },
})
const emit = defineEmits(['update:modelValue', 'validity'])

const CHAT_TEMPLATE_PREFIX = 'chat_template_kwargs.'
const scopeOptions = [
  { label: 'Request', value: 'request' },
  { label: 'Chat template', value: 'chat_template' },
]
const typeOptions = [
  { label: 'Text', value: 'string' },
  { label: 'Number', value: 'number' },
  { label: 'On/Off', value: 'boolean' },
  { label: 'JSON', value: 'json' },
]
const booleanOptions = [
  { label: 'On', value: 'true' },
  { label: 'Off', value: 'false' },
]

const rows = ref([])
let serial = 0
let lastPublished = ''

function nextId() {
  serial += 1
  return serial
}

function inferParam(name, value) {
  let scope = 'request'
  let key = String(name || '')
  if (key.startsWith(CHAT_TEMPLATE_PREFIX)) {
    scope = 'chat_template'
    key = key.slice(CHAT_TEMPLATE_PREFIX.length)
  }
  let type = 'string'
  let text
  if (typeof value === 'boolean') {
    type = 'boolean'
    text = value ? 'true' : 'false'
  } else if (typeof value === 'number' && Number.isFinite(value)) {
    type = 'number'
    text = String(value)
  } else if (typeof value === 'object') {
    type = 'json'
    text = JSON.stringify(value, null, 2)
  } else if (value == null) {
    type = 'string'
    text = ''
  } else {
    text = String(value)
  }
  return { id: nextId(), scope, name: key, type, value: text }
}

function rowsFromParams(params) {
  if (!params || typeof params !== 'object' || Array.isArray(params)) return []
  return Object.entries(params).map(([name, value]) => inferParam(name, value))
}

function fullKey(param) {
  let name = String(param.name || '').trim()
  if (!name) return ''
  if (param.scope === 'chat_template') {
    if (name.startsWith(CHAT_TEMPLATE_PREFIX)) name = name.slice(CHAT_TEMPLATE_PREFIX.length)
    return `${CHAT_TEMPLATE_PREFIX}${name}`
  }
  return name
}

function paramValue(param) {
  if (param.type === 'boolean') return param.value === 'true'
  if (param.type === 'number') return Number(param.value)
  if (param.type === 'json') return JSON.parse(param.value)
  return String(param.value ?? '')
}

function aliasError(row) {
  const suffix = String(row.sub_id || '').trim()
  if (suffix && !/^[a-zA-Z0-9._-]+$/.test(suffix))
    return 'Use letters, numbers, dots, underscores, or hyphens in the suffix.'
  if (rows.value.filter((other) => String(other.sub_id || '').trim() === suffix).length > 1)
    return 'Each alias suffix must be unique.'
  return ''
}

function paramError(row, param) {
  const key = fullKey(param)
  if (!key) {
    return String(param.value ?? '').trim() ? 'Enter a parameter name.' : ''
  }
  if (!/^[a-zA-Z0-9_.-]+$/.test(key))
    return 'Use letters, numbers, dots, underscores, or hyphens only.'
  if (key.replace(/\?$/, '').split('.')[0] === 'model')
    return 'The model parameter is managed by Studio.'
  const duplicate = row.params.filter((other) => fullKey(other) === key).length > 1
  if (duplicate) return 'Each parameter name must be unique in this alias.'
  if (param.type === 'number' && !Number.isFinite(Number(param.value)))
    return 'Enter a valid number.'
  if (param.type === 'json') {
    try {
      JSON.parse(param.value)
    } catch {
      return 'Enter valid JSON. Strings need double quotes.'
    }
  }
  return ''
}

const errors = computed(() =>
  rows.value.map((row) => {
    const alias = aliasError(row)
    if (alias) return alias
    for (const param of row.params) {
      const error = paramError(row, param)
      if (error) return error
    }
    return ''
  }),
)

watch(
  () => props.modelValue,
  (value) => {
    if (JSON.stringify(value) === lastPublished && errors.value.every((error) => !error)) return
    rows.value = (value || []).map((item) => ({
      id: nextId(),
      sub_id: item.sub_id || '',
      params: rowsFromParams(item.params || {}),
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
  const value = rows.value.map((row) => {
    const params = {}
    for (const param of row.params) {
      const key = fullKey(param)
      if (!key) continue
      params[key] = paramValue(param)
    }
    return { sub_id: String(row.sub_id || '').trim(), params }
  })
  lastPublished = JSON.stringify(value)
  emit('update:modelValue', value)
}

function add() {
  let suffix = 'thinking'
  let count = 2
  while (rows.value.some((row) => String(row.sub_id || '').trim() === suffix)) {
    suffix = `thinking-${count++}`
  }
  rows.value.push({ id: nextId(), sub_id: suffix, params: [] })
  publish()
}

function remove(index) {
  rows.value.splice(index, 1)
  publish()
}

function addParam(row) {
  row.params.push({ id: nextId(), scope: 'request', name: '', type: 'string', value: '' })
}

function addChatTemplate(row) {
  row.params.push({
    id: nextId(),
    scope: 'chat_template',
    name: 'enable_thinking',
    type: 'boolean',
    value: 'true',
  })
  publish()
}

function removeParam(row, param) {
  const index = row.params.indexOf(param)
  if (index >= 0) row.params.splice(index, 1)
  publish()
}

function onParamChanged(row, param) {
  if (param.type === 'boolean' && !['true', 'false'].includes(String(param.value))) {
    param.value = 'true'
  }
  if (param.type === 'number' && String(param.value ?? '') === '') param.value = '0'
  publish()
}
</script>

<style scoped>
.request-aliases {
  display: grid;
  gap: 0.85rem;
}
.request-aliases__heading {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  gap: 1rem;
}
.request-aliases__heading > div {
  display: grid;
  gap: 0.3rem;
  min-width: 0;
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
  line-height: 1.45;
  overflow-wrap: anywhere;
}
.request-aliases__empty,
.request-aliases__hint {
  margin: 0;
  padding: 0.7rem 0.8rem;
  border: 1px dashed var(--border-primary);
  border-radius: var(--radius-md);
  color: var(--text-secondary);
  font-size: var(--text-xs);
  line-height: 1.45;
}
.request-aliases__card {
  display: grid;
  gap: 0.7rem;
  padding: 0.85rem;
  border: 1px solid var(--border-primary);
  border-radius: var(--radius-md);
  min-width: 0;
}
.request-aliases__card-head {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  gap: 0.75rem;
}
.request-aliases__identity {
  display: grid;
  gap: 0.3rem;
  min-width: 0;
  flex: 1;
}
.request-aliases__suffix {
  max-width: 20rem;
}
.request-aliases__overrides {
  display: grid;
  gap: 0.5rem;
  padding-top: 0.65rem;
  border-top: 1px solid var(--border-primary);
}
.request-aliases__overrides-head {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 0.75rem;
  flex-wrap: wrap;
  font-size: var(--text-xs);
  font-weight: 600;
}
.request-aliases__override-actions {
  display: flex;
  flex-wrap: wrap;
  gap: 0.15rem;
}
.request-aliases__override {
  display: grid;
  gap: 0.35rem;
  padding: 0.45rem;
  border: 1px solid var(--border-primary);
  border-radius: var(--radius-sm);
  background: var(--bg-surface);
}
.request-aliases__override--invalid {
  border-color: var(--status-error);
}
.request-aliases__override-main {
  display: grid;
  grid-template-columns: 7.5rem minmax(9rem, 1.2fr) 6rem minmax(9rem, 1.4fr) auto;
  align-items: center;
  gap: 0.4rem;
}
.request-aliases__key {
  display: flex;
  align-items: center;
  gap: 0.25rem;
  min-width: 0;
}
.request-aliases__key-prefix {
  color: var(--text-secondary);
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
  font-size: 0.68rem;
  white-space: nowrap;
}
.request-aliases__key-input {
  min-width: 5rem;
  width: 100%;
}
.request-aliases__select {
  width: 100%;
}
.request-aliases__value {
  min-width: 0;
}
.request-aliases__error {
  color: var(--status-error);
}

@media (max-width: 760px) {
  .request-aliases__override-main {
    grid-template-columns: 1fr 1fr;
  }
  .request-aliases__key,
  .request-aliases__value {
    grid-column: 1 / -1;
  }
}
</style>
