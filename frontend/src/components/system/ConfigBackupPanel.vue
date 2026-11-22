<template>
  <PageSection
    id="config-backup"
    title="Backup and restore"
    description="Back up preferences, model settings, and templates. Credentials, model files, and running state are excluded."
  >
    <template #actions>
      <Button
        label="Download backup"
        icon="pi pi-download"
        :disabled="busy || uncertain"
        :loading="downloading"
        @click="downloadBackup"
      />
    </template>

    <PageSection
      heading-tag="h3"
      variant="plain"
      title="Restore settings"
      description="Choose a backup to review changes. Restoring configuration does not publish it or restart models. Use Apply afterwards to activate saved changes."
    >
      <div class="page-form page-form--single">
        <div class="page-form__field">
          <label class="page-form__label" for="restore-backup-file">Select a backup</label>
          <input
            id="restore-backup-file"
            ref="fileInput"
            class="restore-file"
            type="file"
            accept="application/json,.json"
            :disabled="busy || uncertain"
            @change="onFile"
          />
          <p class="page-form__hint">
            Choose a JSON backup exported from Studio. Files larger than 1 MiB are rejected.
          </p>
        </div>
      </div>
    </PageSection>

    <StateBanner v-if="readError" class="restore-error" severity="error">
      {{ readError }}
      <template #actions>
        <Button
          label="Choose another backup"
          icon="pi pi-folder-open"
          severity="secondary"
          outlined
          size="small"
          @click="focusFileInput"
        />
      </template>
    </StateBanner>

    <PageSection
      v-if="preview"
      heading-tag="h3"
      variant="plain"
      title="Review changes"
      :description="preview.notice"
    >
      <p v-if="!preview.applicable" class="page-form__hint">
        Map each unresolved model to an existing model, or skip it, then update the preview.
      </p>
      <ul class="restore-items">
        <li v-for="item in preview.items" :key="`${item.kind}:${item.id}`" class="restore-item">
          <span class="restore-item-kind">{{ kindLabel(item.kind) }}</span>
          <span class="restore-item-id">{{ item.id }}</span>
          <span class="restore-item-action">{{ actionLabel(shownAction(item)) }}</span>
          <select
            v-if="item.action === 'unresolved' || mapping[item.id]"
            :aria-label="`Map ${item.id}`"
            :value="mapping[item.id] || ''"
            :disabled="busy || uncertain"
            @change="setMapping(item, $event.target.value)"
          >
            <option value="">Unresolved</option>
            <option v-for="model in localModels" :key="model.id" :value="model.id">
              {{ model.label }}
            </option>
          </select>
          <select
            v-if="item.action !== 'unresolved' || mapping[item.id]"
            :aria-label="`Decision for ${item.id}`"
            :value="decisionFor(item)"
            :disabled="busy || uncertain"
            @change="setDecision(item, $event.target.value)"
          >
            <option v-for="option in decisionOptions(item)" :key="option" :value="option">
              {{ actionLabel(option) }}
            </option>
          </select>
          <Button
            v-if="item.action === 'unresolved' && !mapping[item.id]"
            :label="`Skip ${item.id}`"
            severity="secondary"
            text
            size="small"
            :disabled="busy || uncertain"
            @click="setDecision(item, 'skip')"
          />
        </li>
      </ul>
      <p v-if="planLabel" class="page-form__hint">Plan {{ planLabel }}</p>
      <div class="page-form__actions">
        <Button
          label="Update preview"
          icon="pi pi-refresh"
          severity="secondary"
          outlined
          :disabled="busy || uncertain"
          :loading="previewing"
          @click="loadPreview"
        />
        <Button
          v-if="!uncertain"
          label="Restore saved settings"
          icon="pi pi-undo"
          severity="secondary"
          outlined
          :disabled="!canRestore"
          :loading="applying"
          @click="confirmRestore"
        />
        <Button
          label="Cancel restore"
          severity="secondary"
          text
          :disabled="busy || uncertain"
          @click="cancelPreview"
        />
      </div>
    </PageSection>

    <StateBanner v-if="message" class="restore-status" :severity="messageSeverity" role="status">
      {{ message }}
    </StateBanner>
    <div v-if="uncertain" class="page-form__actions">
      <Button
        label="Reconcile"
        icon="pi pi-history"
        severity="secondary"
        outlined
        :loading="reconciling"
        @click="reconcile"
      />
    </div>

    <ConfigHistoryPanel v-if="showHistory" heading-tag="h3" variant="plain" />
  </PageSection>
</template>

<script setup>
import { computed, nextTick, onMounted, ref } from 'vue'
import axios from 'axios'
import Button from 'primevue/button'
import PageSection from '@/components/common/PageSection.vue'
import StateBanner from '@/components/common/StateBanner.vue'
import ConfigHistoryPanel from './ConfigHistoryPanel.vue'

defineProps({ showHistory: { type: Boolean, default: true } })

const backup = ref(null)
const preview = ref(null)
const decisions = ref({ preferences: {}, models: {}, templates: {} })
const mapping = ref({})
const localModels = ref([])
const readError = ref('')
const message = ref('')
const messageSeverity = ref('info')
const previewing = ref(false)
const applying = ref(false)
const reconciling = ref(false)
const reading = ref(false)
const downloading = ref(false)
const fileInput = ref(null)
const busy = computed(
  () =>
    previewing.value || applying.value || reconciling.value || reading.value || downloading.value,
)
const dirty = ref(false)
const uncertain = ref(false)
const planId = ref(null)

const canRestore = computed(
  () =>
    Boolean(preview.value?.applicable && planId.value) &&
    !dirty.value &&
    !busy.value &&
    !uncertain.value,
)
const planLabel = computed(() => (planId.value ? planId.value.slice(0, 12) : ''))

function setMessage(text, severity = 'info') {
  message.value = text
  messageSeverity.value = severity
}

function focusFileInput() {
  fileInput.value?.focus()
}

function backupFilename(header) {
  const match = /filename="([^"]+)"/.exec(String(header || ''))
  const name = match?.[1] || 'studio-config-backup.json'
  return /^[\w.-]+\.json$/.test(name) ? name : 'studio-config-backup.json'
}

function cancelPreview() {
  if (busy.value || uncertain.value) return
  backup.value = null
  preview.value = null
  decisions.value = { preferences: {}, models: {}, templates: {} }
  mapping.value = {}
  readError.value = ''
  message.value = ''
  messageSeverity.value = 'info'
  dirty.value = false
  uncertain.value = false
  planId.value = null
  if (fileInput.value) fileInput.value.value = ''
  nextTick(() => fileInput.value?.focus())
}

function kindLabel(kind) {
  if (kind === 'preference') return 'Preference'
  if (kind === 'model') return 'Model'
  if (kind === 'template') return 'Template'
  return 'Routing'
}

function actionLabel(action) {
  if (action === 'add') return 'Addition'
  if (action === 'replace') return 'Replacement'
  if (action === 'skip') return 'Skipped'
  if (action === 'unresolved') return 'Unresolved model'
  return 'Keep existing'
}

function shownAction(item) {
  return decisionFor(item)
}

function decisionGroup(kind) {
  if (kind === 'preference') return 'preferences'
  if (kind === 'model') return 'models'
  if (kind === 'template') return 'templates'
  return 'template'
}

function decisionFor(item) {
  const chosen = decisions.value[decisionGroup(item.kind)]?.[item.id]
  if (chosen) return chosen
  if (item.action === 'unresolved' && mapping.value[item.id]) return 'keep'
  return item.action
}

function decisionOptions(item) {
  if (item.kind === 'model' && (item.action === 'unresolved' || mapping.value[item.id])) {
    return ['keep', 'replace', 'skip']
  }
  if (item.action === 'add') return ['add', 'skip']
  return ['keep', 'replace', 'skip']
}

function setDecision(item, value) {
  decisions.value[decisionGroup(item.kind)][item.id] = value
  if (value === 'skip') delete mapping.value[item.id]
  dirty.value = true
  message.value = ''
}

function setMapping(item, localId) {
  if (!localId) delete mapping.value[item.id]
  else mapping.value[item.id] = localId
  if (!decisions.value.models[item.id]) decisions.value.models[item.id] = 'keep'
  dirty.value = true
  message.value = ''
}

async function downloadBackup() {
  if (busy.value || uncertain.value) return
  downloading.value = true
  readError.value = ''
  try {
    const response = await axios.get('/api/config-backup', { responseType: 'blob' })
    const blob =
      response.data instanceof Blob
        ? response.data
        : new Blob([response.data], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = url
    link.download = backupFilename(response.headers?.['content-disposition'])
    document.body.appendChild(link)
    link.click()
    link.remove()
    URL.revokeObjectURL(url)
  } catch (error) {
    readError.value = await downloadError(error)
  } finally {
    downloading.value = false
  }
}

async function downloadError(error) {
  const data = error?.response?.data
  if (data && typeof data.detail === 'string') return data.detail
  if (typeof Blob !== 'undefined' && data instanceof Blob) {
    try {
      const parsed = JSON.parse(await data.text())
      if (typeof parsed?.detail === 'string') return parsed.detail
    } catch {
      /* The error body was not JSON. */
    }
  }
  return 'The backup could not be downloaded.'
}

async function onFile(event) {
  if (busy.value || uncertain.value) return
  const file = event.target.files?.[0]
  readError.value = ''
  message.value = ''
  uncertain.value = false
  preview.value = null
  planId.value = null
  backup.value = null
  if (!file) return
  if (file.size > 1_048_576) {
    readError.value = 'This backup exceeds the 1 MiB limit.'
    return
  }
  let parsed
  reading.value = true
  try {
    parsed = JSON.parse(await file.text())
  } catch {
    readError.value = 'This backup could not be read.'
    return
  } finally {
    reading.value = false
  }
  backup.value = parsed
  decisions.value = { preferences: {}, models: {}, templates: {} }
  mapping.value = {}
  dirty.value = false
  await loadPreview()
}

async function loadPreview() {
  if (!backup.value || busy.value || uncertain.value) return
  previewing.value = true
  readError.value = ''
  try {
    const { data } = await axios.post('/api/config-backup/preview', {
      backup: backup.value,
      decisions: decisions.value,
      mapping: mapping.value,
    })
    preview.value = data
    planId.value = data.plan_id
    dirty.value = false
    if (data.applicable === false) {
      setMessage('The preview has unresolved models. Map or skip them before restoring.', 'warning')
    }
  } catch (error) {
    preview.value = null
    planId.value = null
    readError.value = error?.response?.data?.detail || 'This backup was rejected.'
  } finally {
    previewing.value = false
  }
}

async function confirmRestore() {
  if (!canRestore.value) return
  applying.value = true
  uncertain.value = false
  try {
    const { data } = await axios.post('/api/config-backup/apply', {
      backup: backup.value,
      decisions: decisions.value,
      mapping: mapping.value,
      plan_id: planId.value,
    })
    if (data.outcome === 'completed') {
      setMessage(
        'Restore completed. Saved settings were updated. Running models were not restarted or published.',
        'success',
      )
    } else {
      setMessage(
        'The restore outcome could not be established. Reconcile before trying again.',
        'error',
      )
    }
    if (data.outcome !== 'completed') uncertain.value = true
    planId.value = null
    dirty.value = true
  } catch (error) {
    const data = error?.response?.data
    if (data?.code === 'BACKUP_STALE') {
      dirty.value = true
      setMessage(
        'The configuration changed after the preview. Update the preview before restoring.',
        'warning',
      )
      return
    }
    uncertain.value = true
    setMessage(
      data?.detail ||
        'The restore outcome could not be established. Reconcile before trying again.',
      'error',
    )
  } finally {
    applying.value = false
  }
}

async function reconcile() {
  if (busy.value || !uncertain.value) return
  reconciling.value = true
  try {
    const { data } = await axios.post('/api/config-backup/reconcile')
    uncertain.value = !['completed', 'pre_import', 'idle'].includes(data.outcome)
    planId.value = null
    dirty.value = true
    if (data.outcome === 'completed') {
      setMessage(
        'Recovery completed the restore. Running models were not restarted or published.',
        'success',
      )
    } else if (data.outcome === 'pre_import') {
      setMessage(
        'Recovery returned configuration to the pre-import state. Running models were not restarted or published.',
        'success',
      )
    } else if (data.outcome === 'idle') {
      setMessage(
        'No restore is pending. Update the preview to inspect the current configuration before a new restore.',
        'info',
      )
    } else {
      setMessage(
        'The restore outcome could not be established. Reconcile again before trying a new restore.',
        'error',
      )
    }
  } catch {
    uncertain.value = true
    setMessage(
      'The restore outcome could not be established. Reconcile again before trying a new restore.',
      'error',
    )
  } finally {
    reconciling.value = false
  }
}

onMounted(async () => {
  try {
    const { data } = await axios.get('/api/models')
    const rows = []
    for (const group of Array.isArray(data) ? data : []) {
      for (const quant of group.quantizations || []) {
        if (quant?.id) rows.push({ id: quant.id, label: quant.name || quant.id })
      }
    }
    localModels.value = rows
  } catch {
    localModels.value = []
  }
})
</script>

<style scoped>
.restore-file {
  width: 100%;
  max-width: 100%;
  min-width: 0;
}

.restore-file::file-selector-button {
  padding: 0.4rem 0.65rem;
  border: 1px solid var(--border-primary);
  border-radius: var(--radius-sm);
  background: var(--bg-tertiary);
  color: var(--text-primary);
  font: inherit;
  cursor: pointer;
}

.restore-items {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
  margin: 0;
  padding: 0;
  list-style: none;
}

.restore-item {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 0.5rem;
  min-width: 0;
}

.restore-item-id,
.restore-item-action {
  overflow-wrap: anywhere;
}

.restore-item select {
  max-width: 100%;
  min-height: var(--toolbar-control-height);
  padding: 0.35rem 0.6rem;
  border: 1px solid var(--border-secondary);
  border-radius: var(--radius-md);
  background: var(--bg-tertiary);
  color: var(--text-primary);
  font: inherit;
}
</style>
