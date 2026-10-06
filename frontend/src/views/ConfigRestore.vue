<template>
  <section class="restore-page">
    <h1>Restore configuration</h1>
    <p class="restore-notice">
      Restoring configuration does not publish it or restart models.
      Saved settings change only. Use Apply afterwards if a running model should change.
    </p>

    <label class="restore-file" for="restore-backup-file">
      Select a backup
      <input
        id="restore-backup-file"
        type="file"
        accept="application/json,.json"
        :disabled="busy || uncertain"
        @change="onFile"
      />
    </label>
    <p v-if="readError" class="restore-error" role="alert">{{ readError }}</p>

    <div v-if="preview" class="restore-preview">
      <p>{{ preview.notice }}</p>
      <p v-if="!preview.applicable">
        Map each unresolved model to an existing model, or skip it, then update the preview.
      </p>
      <ul class="restore-items">
        <li v-for="item in preview.items" :key="`${item.kind}:${item.id}`">
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
          <button
            v-if="item.action === 'unresolved' && !mapping[item.id]"
            type="button"
            :disabled="busy || uncertain"
            @click="setDecision(item, 'skip')"
          >
            Skip {{ item.id }}
          </button>
        </li>
      </ul>
      <p v-if="planLabel">Plan {{ planLabel }}</p>
      <div class="restore-actions">
        <button type="button" :disabled="busy || uncertain" @click="loadPreview">Update preview</button>
        <button
          v-if="!uncertain"
          type="button"
          :disabled="!canRestore"
          @click="confirmRestore"
        >
          Restore saved settings
        </button>
        <button type="button" :disabled="busy || uncertain" @click="cancelPreview">Cancel restore</button>
      </div>
    </div>

    <p v-if="message" class="restore-status" role="status">{{ message }}</p>
    <button v-if="uncertain" type="button" :disabled="busy" @click="reconcile">Reconcile</button>
  </section>
</template>

<script setup>
import { computed, nextTick, onMounted, ref } from 'vue'
import axios from 'axios'

const backup = ref(null)
const preview = ref(null)
const decisions = ref({ preferences: {}, models: {}, templates: {}, routing: {} })
const mapping = ref({})
const localModels = ref([])
const readError = ref('')
const message = ref('')
const previewing = ref(false)
const applying = ref(false)
const reconciling = ref(false)
const reading = ref(false)
const busy = computed(() => previewing.value || applying.value || reconciling.value || reading.value)
const dirty = ref(false)
const uncertain = ref(false)
const planId = ref(null)

const canRestore = computed(() => (
  Boolean(preview.value?.applicable && planId.value) && !dirty.value && !busy.value && !uncertain.value
))
const planLabel = computed(() => (planId.value ? planId.value.slice(0, 12) : ''))

function cancelPreview() {
  if (busy.value || uncertain.value) return
  backup.value = null
  preview.value = null
  decisions.value = { preferences: {}, models: {}, templates: {}, routing: {} }
  mapping.value = {}
  readError.value = ''
  message.value = ''
  dirty.value = false
  uncertain.value = false
  planId.value = null
  const input = document.getElementById('restore-backup-file')
  if (input) input.value = ''
  nextTick(() => input?.focus())
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
  return 'routing'
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
  decisions.value = { preferences: {}, models: {}, templates: {}, routing: {} }
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
      message.value = 'The preview has unresolved models. Map or skip them before restoring.'
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
    message.value = data.outcome === 'completed'
      ? 'Restore completed. Saved settings were updated. Running models were not restarted or published.'
      : 'The restore outcome could not be established. Reconcile before trying again.'
    if (data.outcome !== 'completed') uncertain.value = true
    planId.value = null
    dirty.value = true
  } catch (error) {
    const data = error?.response?.data
    if (data?.code === 'BACKUP_STALE') {
      dirty.value = true
      message.value = 'The configuration changed after the preview. Update the preview before restoring.'
      return
    }
    uncertain.value = true
    message.value = data?.detail || 'The restore outcome could not be established. Reconcile before trying again.'
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
    message.value = data.outcome === 'completed'
      ? 'Recovery completed the restore. Running models were not restarted or published.'
      : data.outcome === 'pre_import'
        ? 'Recovery returned configuration to the pre-import state. Running models were not restarted or published.'
        : data.outcome === 'idle'
          ? 'No restore is pending. Update the preview to inspect the current configuration before a new restore.'
        : 'The restore outcome could not be established. Reconcile again before trying a new restore.'
  } catch {
    uncertain.value = true
    message.value = 'The restore outcome could not be established. Reconcile again before trying a new restore.'
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
.restore-page {
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
  max-width: 40rem;
}

.restore-notice,
.restore-status {
  margin: 0;
}

.restore-error {
  color: var(--status-warning);
  margin: 0;
}

.restore-items {
  list-style: none;
  padding: 0;
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}

.restore-items li {
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
  align-items: center;
  min-width: 0;
}

.restore-item-id,
.restore-item-action {
  overflow-wrap: anywhere;
}

.restore-items select,
.restore-items button,
.restore-actions button {
  max-width: 100%;
}

.restore-actions {
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
}

.restore-file {
  display: flex;
  flex-direction: column;
  gap: 0.35rem;
}
</style>
