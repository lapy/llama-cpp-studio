<template>
  <Dialog
    :visible="visible"
    modal
    header="Import from command"
    class="dialog-width-lg parse-command-dialog"
    @update:visible="emit('update:visible', $event)"
  >
    <p class="parse-command-lead">
      Paste a server command, raw flags, or <code>KEY=value</code> environment assignments.
      Matching catalog parameters and llama-swap env vars appear below as you type.
      Studio-managed options (listen host/port, model path, alias, <code>LLAMA_STUDIO_*</code>)
      are never imported.
    </p>

    <div class="parse-command-field">
      <label for="parse-command-text">Command text</label>
      <Textarea
        id="parse-command-text"
        v-model="draft"
        rows="6"
        class="w-full textarea-cli parse-command-textarea"
        placeholder="e.g. CUDA_VISIBLE_DEVICES=0,1 sglang serve --tp-size 2"
        autoResize
      />
      <div v-if="customArgs" class="parse-command-seed">
        <Button
          label="Use Custom Arguments"
          icon="pi pi-copy"
          size="small"
          severity="secondary"
          text
          type="button"
          @click="draft = customArgs"
        />
      </div>
    </div>

    <Message
      v-if="parsed.parseError"
      severity="warn"
      :closable="false"
      class="parse-command-message"
    >
      {{ parsed.parseError }}
    </Message>
    <Message
      v-for="warning in parsed.warnings"
      :key="warning"
      severity="warn"
      :closable="false"
      class="parse-command-message"
    >
      {{ warning }}
    </Message>

    <template v-if="!draft.trim()">
      <Message severity="secondary" :closable="false" class="parse-command-message">
        Paste a command to preview what will be imported.
      </Message>
    </template>
    <template v-else>
      <div class="parse-command-section">
        <div class="section-label">
          Import preview
          <small class="section-hint">{{ importableRows.length }} parameter(s) can be applied</small>
        </div>
        <Message v-if="!importableRows.length && !envRows.length" severity="secondary" :closable="false">
          No catalog parameters or environment variables to import. Studio-owned, unknown, and deprecated items are listed below.
        </Message>
        <Message v-else-if="!importableRows.length" severity="secondary" :closable="false">
          No catalog parameters to import. Environment variables are listed below.
        </Message>
        <ul v-else class="parse-preview-list">
          <li v-for="row in importableRows" :key="row.key" class="parse-preview-item">
            <Checkbox
              :input-id="`import-${row.key}`"
              :model-value="selectedKeys.includes(row.key)"
              binary
              @update:model-value="toggleKey(row.key, $event)"
            />
            <label :for="`import-${row.key}`" class="parse-preview-main">
              <span class="parse-preview-title">
                {{ row.label }}
                <code>{{ row.key }}</code>
                <code class="parse-preview-flag">{{ row.sourceFlag }}</code>
                <Tag :value="changeLabel(row.change)" :severity="changeSeverity(row.change)" />
              </span>
              <span class="parse-preview-values">
                {{ formatCliValue(row.currentValue) }}
                <i class="pi pi-arrow-right" aria-hidden="true" />
                <strong>{{ formatCliValue(row.value, { emptyLabel: '—' }) }}</strong>
              </span>
            </label>
          </li>
        </ul>
      </div>

      <div v-if="envRows.length" class="parse-command-section">
        <div class="section-label">
          Environment variables
          <small class="section-hint">{{ envRows.length }} applied to llama-swap env</small>
        </div>
        <ul class="parse-preview-list">
          <li v-for="row in envRows" :key="`env-${row.key}`" class="parse-preview-item">
            <Checkbox
              :input-id="`import-env-${row.key}`"
              :model-value="selectedEnvKeys.includes(row.key)"
              binary
              @update:model-value="toggleEnv(row.key, $event)"
            />
            <label :for="`import-env-${row.key}`" class="parse-preview-main">
              <span class="parse-preview-title">
                {{ row.key }}
                <code>{{ row.key }}</code>
                <Tag :value="changeLabel(row.change)" :severity="changeSeverity(row.change)" />
              </span>
              <span class="parse-preview-values">
                {{ formatCliValue(row.currentValue) }}
                <i class="pi pi-arrow-right" aria-hidden="true" />
                <strong>{{ formatCliValue(row.value, { emptyLabel: '—' }) }}</strong>
              </span>
            </label>
          </li>
        </ul>
      </div>

      <div v-if="parsed.reserved.length" class="parse-command-section">
        <div class="section-label">
          Skipped — Studio managed
          <small class="section-hint">Not imported</small>
        </div>
        <ul class="parse-skip-list">
          <li v-for="(row, idx) in parsed.reserved" :key="`${row.flag}-${idx}`">
            <code>{{ joinCliTokens(row.tokens) }}</code>
            <span>{{ row.reason }}</span>
          </li>
        </ul>
      </div>

      <div v-if="parsed.unsupported.length" class="parse-command-section">
        <div class="section-label">
          Skipped — not in this build
          <small class="section-hint">Deprecated or unsupported; opt in to apply</small>
        </div>
        <ul class="parse-preview-list">
          <li v-for="row in parsed.unsupported" :key="`unsup-${row.key}`" class="parse-preview-item">
            <Checkbox
              :input-id="`import-unsup-${row.key}`"
              :model-value="selectedUnsupportedKeys.includes(row.key)"
              binary
              @update:model-value="toggleUnsupported(row.key, $event)"
            />
            <label :for="`import-unsup-${row.key}`" class="parse-preview-main">
              <span class="parse-preview-title">
                {{ row.param?.label || row.key }}
                <code>{{ row.key }}</code>
                <Tag value="Not in this build" severity="secondary" />
              </span>
              <span class="parse-preview-values">{{ row.reason }}</span>
            </label>
          </li>
        </ul>
      </div>

      <div v-if="parsed.unknown.length" class="parse-command-section">
        <div class="section-label">
          Unrecognized
          <small class="section-hint">Not in the current catalog</small>
        </div>
        <ul class="parse-skip-list">
          <li v-for="(row, idx) in parsed.unknown" :key="`unk-${idx}`">
            <code>{{ joinCliTokens(row.tokens) }}</code>
            <span>{{ row.reason }}</span>
          </li>
        </ul>
        <div class="parse-command-check">
          <ToggleSwitch v-model="appendUnknown" input-id="parse-append-unknown" />
          <label for="parse-append-unknown">Append unrecognized tokens to Custom Arguments</label>
        </div>
      </div>

      <div v-if="showReplaceCustomArgs" class="parse-command-check">
        <ToggleSwitch v-model="replaceCustomArgs" input-id="parse-replace-custom" />
        <label for="parse-replace-custom">
          Replace Custom Arguments with leftover unrecognized tokens
        </label>
      </div>
    </template>

    <template #footer>
      <Button
        label="Cancel"
        severity="secondary"
        outlined
        type="button"
        @click="emit('update:visible', false)"
      />
      <Button
        :label="applyLabel"
        icon="pi pi-check"
        :disabled="!canApply"
        type="button"
        @click="confirmApply"
      />
    </template>
  </Dialog>
</template>

<script setup>
import { computed, ref, watch } from 'vue'
import Button from 'primevue/button'
import Checkbox from 'primevue/checkbox'
import Dialog from 'primevue/dialog'
import Message from 'primevue/message'
import Tag from 'primevue/tag'
import Textarea from 'primevue/textarea'
import ToggleSwitch from 'primevue/toggleswitch'
import {
  buildEnvImportPreview,
  buildImportPreview,
  formatCliValue,
  joinCliTokens,
  parseCliCommand,
} from '@/utils/parseCliCommand'

const props = defineProps({
  visible: { type: Boolean, default: false },
  catalogParams: { type: Array, default: () => [] },
  currentValues: { type: Object, default: () => ({}) },
  currentEnv: { type: Object, default: () => ({}) },
  customArgs: { type: String, default: '' },
  seedText: { type: String, default: '' },
})

const emit = defineEmits(['update:visible', 'apply'])

const draft = ref('')
const selectedKeys = ref([])
const selectedUnsupportedKeys = ref([])
const selectedEnvKeys = ref([])
const appendUnknown = ref(false)
const replaceCustomArgs = ref(false)

const parsed = computed(() => parseCliCommand(draft.value, props.catalogParams))
const importableRows = computed(() => buildImportPreview(parsed.value, props.currentValues))
const envRows = computed(() => buildEnvImportPreview(parsed.value, props.currentEnv))

const showReplaceCustomArgs = computed(() => {
  const custom = String(props.customArgs || '').trim()
  return Boolean(custom) && custom === draft.value.trim()
})

const selectedParamCount = computed(() => {
  return selectedKeys.value.length + selectedUnsupportedKeys.value.length
})

const selectedCount = computed(() => selectedParamCount.value + selectedEnvKeys.value.length)

const canApply = computed(() => {
  if (!draft.value.trim()) return false
  if (selectedCount.value > 0) return true
  return appendUnknown.value || replaceCustomArgs.value
})

const applyLabel = computed(() => {
  const params = selectedParamCount.value
  const envs = selectedEnvKeys.value.length
  const parts = []
  if (params) parts.push(params === 1 ? '1 parameter' : `${params} parameters`)
  if (envs) parts.push(envs === 1 ? '1 env var' : `${envs} env vars`)
  if (!parts.length) return 'Apply leftovers'
  return `Apply ${parts.join(' and ')}`
})

watch(
  () => props.visible,
  (open) => {
    if (!open) return
    draft.value = props.seedText || ''
    selectedKeys.value = []
    selectedUnsupportedKeys.value = []
    selectedEnvKeys.value = []
    appendUnknown.value = false
    replaceCustomArgs.value = Boolean(
      String(props.customArgs || '').trim() &&
      String(props.seedText || '').trim() === String(props.customArgs || '').trim(),
    )
  },
)

watch(
  () => importableRows.value.map((row) => `${row.key}:${row.change}`).join('|'),
  (next, prev) => {
    const stillPresent = new Set(importableRows.value.map((row) => row.key))
    const keep = selectedKeys.value.filter((key) => stillPresent.has(key))
    const prevKeys = new Set((prev || '').split('|').map((part) => part.split(':')[0]).filter(Boolean))
    for (const row of importableRows.value) {
      if (!prevKeys.has(row.key) && row.change !== 'unchanged') keep.push(row.key)
    }
    selectedKeys.value = [...new Set(keep)]
  },
)

watch(
  () => parsed.value.unsupported.map((row) => row.key).join('|'),
  () => {
    const still = new Set(parsed.value.unsupported.map((row) => row.key))
    selectedUnsupportedKeys.value = selectedUnsupportedKeys.value.filter((key) => still.has(key))
  },
)

watch(
  () => envRows.value.map((row) => `${row.key}:${row.change}`).join('|'),
  (next, prev) => {
    const stillPresent = new Set(envRows.value.map((row) => row.key))
    const keep = selectedEnvKeys.value.filter((key) => stillPresent.has(key))
    const prevKeys = new Set((prev || '').split('|').map((part) => part.split(':')[0]).filter(Boolean))
    for (const row of envRows.value) {
      if (!prevKeys.has(row.key) && row.change !== 'unchanged') keep.push(row.key)
    }
    selectedEnvKeys.value = [...new Set(keep)]
  },
)

function toggleKey(key, checked) {
  if (checked) {
    if (!selectedKeys.value.includes(key)) selectedKeys.value = [...selectedKeys.value, key]
    return
  }
  selectedKeys.value = selectedKeys.value.filter((item) => item !== key)
}

function toggleEnv(key, checked) {
  if (checked) {
    if (!selectedEnvKeys.value.includes(key)) selectedEnvKeys.value = [...selectedEnvKeys.value, key]
    return
  }
  selectedEnvKeys.value = selectedEnvKeys.value.filter((item) => item !== key)
}

function toggleUnsupported(key, checked) {
  if (checked) {
    if (!selectedUnsupportedKeys.value.includes(key)) {
      selectedUnsupportedKeys.value = [...selectedUnsupportedKeys.value, key]
    }
    return
  }
  selectedUnsupportedKeys.value = selectedUnsupportedKeys.value.filter((item) => item !== key)
}

function changeLabel(change) {
  if (change === 'new') return 'New'
  if (change === 'update') return 'Update'
  return 'Unchanged'
}

function changeSeverity(change) {
  if (change === 'new') return 'success'
  if (change === 'update') return 'warn'
  return 'secondary'
}

function confirmApply() {
  const byKey = new Map(importableRows.value.map((row) => [row.key, row]))
  const unsupportedByKey = new Map(parsed.value.unsupported.map((row) => [row.key, row]))
  const params = []
  for (const key of selectedKeys.value) {
    const row = byKey.get(key)
    if (row) params.push({ key: row.key, value: row.value })
  }
  for (const key of selectedUnsupportedKeys.value) {
    const row = unsupportedByKey.get(key)
    if (row) params.push({ key: row.key, value: row.value })
  }

  const envByKey = new Map(envRows.value.map((row) => [row.key, row]))
  const env = []
  for (const key of selectedEnvKeys.value) {
    const row = envByKey.get(key)
    if (row) env.push({ key: row.key, value: row.value })
  }

  let customArgs
  const leftover = joinCliTokens(parsed.value.leftoverTokens)
  if (replaceCustomArgs.value) {
    customArgs = leftover
  } else if (appendUnknown.value && leftover) {
    const existing = String(props.customArgs || '').trim()
    customArgs = existing ? `${existing} ${leftover}` : leftover
  }

  emit('apply', { params, env, customArgs })
  emit('update:visible', false)
}
</script>

<style scoped>
.parse-command-lead {
  margin: 0 0 1rem;
  font-size: 0.875rem;
  color: var(--text-secondary, #9ca3af);
  line-height: 1.45;
}

.parse-command-field label {
  display: block;
  font-size: 0.8rem;
  margin-bottom: 0.35rem;
  color: var(--text-secondary, #9ca3af);
}

.parse-command-textarea {
  min-height: 7rem;
}

.parse-command-seed {
  margin-top: 0.25rem;
}

.parse-command-message {
  margin: 0.65rem 0;
}

.parse-command-section {
  margin: 1rem 0 0;
}

.parse-preview-list,
.parse-skip-list {
  list-style: none;
  margin: 0;
  padding: 0;
}

.parse-preview-item {
  display: flex;
  align-items: flex-start;
  gap: 0.65rem;
  padding: 0.55rem 0;
  border-bottom: 1px solid var(--border-primary, #2a2f45);
}

.parse-preview-item:last-child {
  border-bottom: none;
}

.parse-preview-main {
  display: flex;
  flex-direction: column;
  gap: 0.2rem;
  min-width: 0;
  cursor: pointer;
}

.parse-preview-title {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 0.35rem;
  font-size: 0.875rem;
}

.parse-preview-title code,
.parse-skip-list code {
  font-size: 0.7rem;
  padding: 0.1rem 0.35rem;
  border-radius: 0.25rem;
  background: rgba(0, 0, 0, 0.25);
  color: var(--text-secondary, #9ca3af);
}

.parse-preview-flag {
  opacity: 0.85;
}

.parse-preview-values {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 0.35rem;
  font-size: 0.8rem;
  color: var(--text-secondary, #9ca3af);
}

.parse-skip-list li {
  display: flex;
  flex-direction: column;
  gap: 0.15rem;
  padding: 0.4rem 0;
  font-size: 0.8rem;
  color: var(--text-secondary, #9ca3af);
}

.parse-command-check {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  margin-top: 0.75rem;
  font-size: 0.875rem;
}
</style>
