<template>
  <section class="generic-engine-panel">
    <EngineCard :title="label" :count="versionRows.length" :cta="cta" @click="openDialog">
      <template #badge>
        <span class="generic-engine-badge" :data-state="badgeState">{{ badgeText }}</span>
      </template>
      <div class="generic-engine-lines">
        <div v-if="busy">Operation running: {{ operationLabel }}</div>
        <div v-else-if="status.installed === true">Installed: {{ status.version || 'active' }}</div>
        <div v-else-if="Object.keys(status).length">Not installed</div>
        <div v-else>Status not loaded yet</div>
      </div>
    </EngineCard>

    <Dialog v-model:visible="visible" :header="label" modal class="dialog-width-md">
      <div class="dialog-stack generic-engine-dialog">
        <div class="generic-engine-status">
          <div v-for="line in statusLines" :key="line.label">
            <span class="generic-engine-key">{{ line.label }}</span>
            <code>{{ line.value }}</code>
          </div>
          <div v-if="busy" class="generic-engine-operation">
            Operation: {{ operationLabel }}
            <Button
              v-if="operationTaskId && serves('cancel')"
              label="Cancel"
              size="small"
              severity="secondary"
              outlined
              @click="cancelOperation"
            />
          </div>
        </div>

        <PageSection v-if="serves('install')" variant="plain" title="Install a release">
          <div class="generic-engine-row">
            <InputText
              v-model="versionInput"
              placeholder="version (leave empty for latest)"
              aria-label="Release version"
            />
            <Button
              v-if="serves('check-updates')"
              label="Check updates"
              size="small"
              severity="secondary"
              outlined
              :loading="checking"
              @click="checkUpdates"
            />
            <Button label="Install" size="small" :disabled="busy" @click="installRelease" />
          </div>
          <p v-if="updateText" class="generic-engine-updates">{{ updateText }}</p>
        </PageSection>

        <PageSection v-if="serves('install-source')" variant="plain" title="Install from source">
          <div class="generic-engine-row">
            <InputText
              v-model="repoInput"
              placeholder="repository URL"
              aria-label="Repository URL"
            />
            <InputText v-model="branchInput" placeholder="branch" aria-label="Branch" />
            <Button
              label="Install source"
              size="small"
              severity="secondary"
              outlined
              :disabled="busy"
              @click="installSource"
            />
          </div>
        </PageSection>

        <PageSection variant="plain" title="Versions">
          <ul v-if="versionRows.length" class="generic-engine-versions">
            <li v-for="row in versionRows" :key="row.id || row.version">
              <strong>{{ row.version }}</strong>
              <span> — {{ row.build_status || 'unknown' }}</span>
              <span v-if="row.error"> ({{ row.error }})</span>
              <span v-if="row.is_active" class="generic-engine-active">active</span>
              <Button
                v-else
                label="Use"
                size="small"
                text
                :disabled="busy"
                @click="activateVersionRow(row)"
              />
            </li>
          </ul>
          <EmptyState
            v-else
            heading-tag="h3"
            title="No versions installed yet"
            description="Install a release or a source build to create the first version."
          />
        </PageSection>

        <PageSection v-if="serves('build-settings')" variant="plain" title="Settings">
          <div class="page-form">
            <div class="page-form__field">
              <label class="page-form__label" :for="`${engineId}-pip-version`"
                >Pinned pip version</label
              >
              <InputText
                :id="`${engineId}-pip-version`"
                v-model="pipVersion"
                placeholder="pinned pip version"
              />
            </div>
            <div class="page-form__field">
              <label class="page-form__label" :for="`${engineId}-source-repo`"
                >Default source repo</label
              >
              <InputText
                :id="`${engineId}-source-repo`"
                v-model="sourceRepo"
                placeholder="default source repo"
              />
            </div>
            <div class="page-form__field">
              <label class="page-form__label" :for="`${engineId}-source-branch`"
                >Default branch</label
              >
              <InputText
                :id="`${engineId}-source-branch`"
                v-model="sourceBranch"
                placeholder="default branch"
              />
            </div>
          </div>
          <div class="dialog-actions">
            <Button
              label="Save settings"
              size="small"
              severity="secondary"
              outlined
              @click="saveSettings"
            />
          </div>
        </PageSection>
      </div>
      <template #footer>
        <div class="dialog-actions">
          <Button
            v-if="serves('remove')"
            label="Remove engine"
            severity="danger"
            text
            :disabled="busy"
            @click="removeEngine"
          />
        </div>
      </template>
    </Dialog>
  </section>
</template>

<script setup>
/**
 * YAML campaign S6 — dynamic engine card for SERVER-BACKED engines
 * (YAML-onboarded, marked generic_lifecycle on GET /api/engines).
 *
 * One component renders a complete install/manage journey for ANY such
 * engine — the fixture ships none of this UI; it only declares a
 * lifecycle. All actions go through the store's server-backing-aware
 * generic verb helpers, so behavior is identical for whatever verbs the
 * engine's YAML actually advertises. The nine generated engines render
 * through their own pinned cards and NEVER mount this component.
 */
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { useConfirm } from 'primevue/useconfirm'
import { useToast } from 'primevue/usetoast'
import Button from 'primevue/button'
import Dialog from 'primevue/dialog'
import InputText from 'primevue/inputtext'
import EmptyState from '@/components/common/EmptyState.vue'
import PageSection from '@/components/common/PageSection.vue'
import EngineCard from '@/components/engines/EngineCard.vue'
import { versionDeleteRetry } from '@/composables/actionConfirmation'
import { useEnginesStore } from '@/stores/engines'

const props = defineProps({
  descriptor: { type: Object, required: true },
})

const store = useEnginesStore()
const toast = useToast()
const confirm = useConfirm()

const engineId = computed(() => props.descriptor.id)
const label = computed(() => props.descriptor.label || props.descriptor.id)
const verbs = computed(() => props.descriptor.generic_verbs || [])
const serves = (verb) => verbs.value.includes(verb)

const status = computed(() => store.serverStatuses[engineId.value] || {})
const versionRows = computed(() => store.serverVersions[engineId.value] || [])
// REAL status contract (_stateful_status): ``operation`` is the running
// operation NAME (a string) or null; the cancel handle lives in the
// separate ``progress_task_id`` field — NOT nested inside operation.
const operation = computed(() => {
  const op = status.value.operation
  return typeof op === 'string' && op ? op : null
})
const busy = computed(() => Boolean(operation.value))
const operationLabel = operation
const operationTaskId = computed(() => status.value.progress_task_id || null)

const badgeState = computed(() =>
  busy.value ? 'busy' : status.value.installed === true ? 'ok' : 'off',
)
const badgeText = computed(() =>
  busy.value ? 'Working' : status.value.installed === true ? 'Installed' : 'Not installed',
)
const cta = computed(() => (versionRows.value.length ? 'Manage' : 'Install'))

const visible = ref(false)
const versionInput = ref('')
const repoInput = ref('')
const branchInput = ref('')
const pipVersion = ref('')
const sourceRepo = ref('')
const sourceBranch = ref('')
const updates = ref(null)
const checking = ref(false)

const statusLines = computed(() =>
  Object.entries(status.value)
    .filter(
      ([key, value]) =>
        key !== 'operation' &&
        value !== null &&
        value !== false &&
        value !== '' &&
        (typeof value === 'string' || typeof value === 'number' || typeof value === 'boolean'),
    )
    .map(([key, value]) => ({ label: `${key}:`, value: String(value) })),
)
const updateText = computed(() => {
  if (!updates.value) return ''
  return Object.entries(updates.value)
    .filter(([, value]) => typeof value !== 'object' || value === null)
    .map(([key, value]) => `${key}: ${value}`)
    .join(' · ')
})

let pollTimer = null
function startPolling() {
  if (pollTimer) return
  pollTimer = setInterval(async () => {
    try {
      await store.venvFetchStatus(engineId.value)
      if (!busy.value) {
        await store.fetchLlamaVersions()
        stopPolling()
      }
    } catch (error) {
      console.error(`Failed to refresh ${engineId.value}:`, error)
    }
  }, 4000)
}
function stopPolling() {
  if (pollTimer) {
    clearInterval(pollTimer)
    pollTimer = null
  }
}

async function openDialog() {
  visible.value = true
  if (serves('build-settings')) {
    try {
      const settings = await store.venvGetBuildSettings(engineId.value)
      pipVersion.value = settings.pip_version || ''
      sourceRepo.value = settings.source_repo || ''
      sourceBranch.value = settings.source_branch || ''
      repoInput.value = settings.source_repo || ''
      branchInput.value = settings.source_branch || ''
    } catch (error) {
      console.error(`Failed to load ${engineId.value} settings:`, error)
    }
  }
}

function notified(message) {
  toast.add({ severity: 'success', summary: label.value, detail: message, life: 4000 })
}
function failureText(error) {
  const detail = error?.response?.data?.detail
  if (detail && typeof detail === 'object' && !Array.isArray(detail)) {
    return detail.message || detail.code || 'Request failed'
  }
  return detail || error?.message || String(error)
}
function failed(error) {
  toast.add({
    severity: 'error',
    summary: label.value,
    detail: failureText(error),
    life: 6000,
  })
}

async function activateVersionRow(row) {
  const versionId = row.id || `${engineId.value}:${row.version}`
  try {
    await store.activateVersion(versionId)
    await store.venvFetchStatus(engineId.value)
    await store.fetchLlamaVersions()
    notified(`Activated ${row.version}`)
  } catch (error) {
    failed(error)
  }
}

async function checkUpdates() {
  checking.value = true
  try {
    updates.value = await store.venvCheckUpdates(engineId.value)
    const latest = updates.value?.latest_version || updates.value?.latest
    if (latest && !versionInput.value) versionInput.value = String(latest)
  } catch (error) {
    failed(error)
  } finally {
    checking.value = false
  }
}

async function installRelease() {
  try {
    const payload = {}
    if (versionInput.value) payload.version = versionInput.value
    const response = await store.venvInstall(engineId.value, payload)
    notified(response?.message || 'Installation started')
    startPolling()
  } catch (error) {
    failed(error)
  }
}

async function installSource() {
  try {
    const payload = {}
    if (repoInput.value) payload.repo_url = repoInput.value
    if (branchInput.value) payload.branch = branchInput.value
    const response = await store.venvInstallFromSource(engineId.value, payload)
    notified(response?.message || 'Source installation started')
    startPolling()
  } catch (error) {
    failed(error)
  }
}

async function saveSettings() {
  try {
    await store.venvSaveBuildSettings(engineId.value, {
      pip_version: pipVersion.value || '',
      source_repo: sourceRepo.value || '',
      source_branch: sourceBranch.value || '',
    })
    notified('Settings saved')
  } catch (error) {
    failed(error)
  }
}

async function cancelOperation() {
  try {
    await store.venvCancel(engineId.value, operationTaskId.value)
    await store.venvFetchStatus(engineId.value)
  } catch (error) {
    failed(error)
  }
}

async function performRemove(params) {
  try {
    await store.venvRemove(engineId.value, params)
    notified('Engine removed')
    startPolling()
  } catch (error) {
    const retry = versionDeleteRetry(error)
    if (!retry) {
      failed(error)
      return
    }
    confirm.require({
      header: `Retire launch references for ${label.value}?`,
      message: retry.message,
      accept: async () => {
        try {
          await store.venvRemove(engineId.value, retry.params)
          notified('Engine removed')
          startPolling()
        } catch (nextError) {
          failed(nextError)
        }
      },
    })
  }
}

function removeEngine() {
  confirm.require({
    header: `Remove ${label.value}`,
    message:
      'This removes the ACTIVE installed version and its version row. ' +
      'Other installed versions of this engine are kept.',
    accept: () => performRemove({}),
  })
}

onMounted(async () => {
  // A reload in the middle of an operation must still show progress: the
  // descriptor may arrive after the first status pass, so fetch it here and
  // let the busy watcher start polling as soon as the operation is visible.
  try {
    await store.venvFetchStatus(engineId.value)
  } catch (error) {
    console.error(`Failed to load ${engineId.value} status:`, error)
  }
  if (busy.value) startPolling()
})
watch(busy, (isBusy) => {
  if (isBusy) startPolling()
  else stopPolling()
})
onBeforeUnmount(stopPolling)
</script>

<style scoped>
.generic-engine-panel {
  order: 3;
  min-width: 0;
  width: 100%;
}

.generic-engine-panel :deep(.engine-card) {
  width: 100%;
}

.generic-engine-badge {
  padding: 0.1rem 0.5rem;
  border: 1px solid currentcolor;
  border-radius: 999px;
  font-size: var(--text-xs);
}

.generic-engine-badge[data-state='ok'] {
  color: var(--status-success);
}

.generic-engine-badge[data-state='busy'] {
  color: var(--status-warning);
}

.generic-engine-badge[data-state='off'] {
  color: var(--text-secondary);
}

.generic-engine-lines {
  color: var(--text-secondary);
  font-size: var(--text-sm);
}

.generic-engine-dialog {
  min-width: 0;
}

.generic-engine-status {
  display: flex;
  flex-direction: column;
  gap: 0.25rem;
  color: var(--text-secondary);
  font-size: var(--text-sm);
}

.generic-engine-operation {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 0.5rem;
}

.generic-engine-row {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 0.5rem;
}

.generic-engine-key {
  margin-right: 0.35rem;
  opacity: 0.75;
}

.generic-engine-versions {
  margin: 0;
  padding-left: 1.1rem;
  font-size: var(--text-sm);
}

.generic-engine-active {
  margin-left: 0.5rem;
  color: var(--status-success);
  font-size: var(--text-xs);
  text-transform: uppercase;
}

.generic-engine-updates {
  margin: 0;
  color: var(--text-secondary);
  font-size: var(--text-sm);
}
</style>
