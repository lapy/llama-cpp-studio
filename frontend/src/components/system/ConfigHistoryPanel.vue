<template>
  <section class="config-history" aria-labelledby="config-history-heading">
    <div class="history-heading">
      <div>
        <h3 id="config-history-heading">Recent changes</h3>
        <p>Studio keeps up to 100 private, local pre-change revisions.</p>
      </div>
      <button type="button" :disabled="busy" @click="refresh">Refresh history</button>
    </div>

    <p v-if="error" role="alert" class="history-error">{{ error }}</p>
    <p v-if="status" role="status">{{ status }}</p>
    <p v-if="!loading && !entries.length">No configuration changes have been recorded yet.</p>

    <div v-else class="history-layout">
      <ul class="history-list" aria-label="Configuration revisions">
        <li v-for="entry in entries" :key="entry.id">
          <button
            type="button"
            :class="{ selected: selectedId === entry.id }"
            :disabled="busy"
            @click="selectEntry(entry.id)"
          >
            <span>{{ documentLabel(entry.document) }}</span>
            <small>{{ formatDate(entry.created_at) }} · {{ entry.reason }}</small>
          </button>
        </li>
      </ul>

      <div v-if="comparison" class="history-diff">
        <h4>Changes since this revision</h4>
        <p v-if="!comparison.changes.length">The current document matches this revision.</p>
        <ul v-else>
          <li v-for="change in comparison.changes" :key="change.path">
            <code>{{ change.path }}</code>
            <span>{{ display(change.before) }} → {{ display(change.after) }}</span>
          </li>
        </ul>
        <div v-if="restorableScopes.length" class="history-scopes">
          <p>Restore one saved item. Running models are not restarted or published.</p>
          <button
            v-for="scope in restorableScopes"
            :key="`${scope.kind}:${scope.item_id}`"
            type="button"
            :disabled="busy"
            @click="restore(scope)"
          >
            Restore {{ scopeLabel(scope) }}
          </button>
        </div>
      </div>
    </div>
  </section>
</template>

<script setup>
import { computed, onMounted, ref } from 'vue'
import {
  configurationHistoryDiff,
  listConfigurationHistory,
  restoreConfigurationHistory,
} from '@/api/configuration'

const entries = ref([])
const comparison = ref(null)
const selectedId = ref('')
const loading = ref(false)
const restoring = ref(false)
const error = ref('')
const status = ref('')
const busy = computed(() => loading.value || restoring.value)

const restorableScopes = computed(() => {
  if (!comparison.value) return []
  const scopes = new Map()
  for (const change of comparison.value.changes || []) {
    const scope = scopeFor(comparison.value.document, change.path)
    if (scope) scopes.set(`${scope.kind}:${scope.item_id}`, scope)
  }
  return [...scopes.values()]
})

function scopeFor(document, path) {
  const parts = String(path || '').split('.')
  if (document === 'settings.yaml' && parts[0] && !/token|password|secret|key/i.test(parts[0])) {
    return { kind: 'preference', item_id: parts[0] }
  }
  if (document === 'models.yaml' && parts[0] === 'models' && parts[1] && parts[2] === 'config') {
    return { kind: 'model', item_id: parts[1] }
  }
  if (document === 'model_config_templates.yaml' && parts[0] === 'templates' && parts[1]) {
    return { kind: 'template', item_id: parts[1] }
  }
  if (document === 'llama_swap_routing.yaml' && parts[1]) {
    if (parts[0] === 'profiles') return { kind: 'profile', item_id: parts[1] }
    if (parts[0] === 'selectors') return { kind: 'selector', item_id: parts[1] }
  }
  return null
}

function scopeLabel(scope) {
  const names = {
    preference: 'preference',
    model: 'model settings',
    template: 'template',
    profile: 'profile',
    selector: 'selector',
  }
  return `${names[scope.kind]} ${scope.item_id}`
}

function documentLabel(document) {
  return (
    {
      'settings.yaml': 'Preferences',
      'models.yaml': 'Model settings',
      'model_config_templates.yaml': 'Templates',
      'llama_swap_routing.yaml': 'Routing',
    }[document] || document
  )
}

function formatDate(value) {
  const parsed = new Date(value)
  return Number.isNaN(parsed.getTime()) ? String(value) : parsed.toLocaleString()
}

function display(value) {
  if (value === null) return 'null'
  return String(value)
}

async function refresh() {
  if (busy.value) return
  loading.value = true
  error.value = ''
  try {
    entries.value = await listConfigurationHistory()
    if (selectedId.value && !entries.value.some((entry) => entry.id === selectedId.value)) {
      selectedId.value = ''
      comparison.value = null
    }
  } catch (requestError) {
    error.value =
      requestError?.response?.data?.detail || 'Configuration history could not be loaded.'
  } finally {
    loading.value = false
  }
}

async function selectEntry(entryId) {
  if (busy.value) return
  loading.value = true
  error.value = ''
  status.value = ''
  try {
    comparison.value = await configurationHistoryDiff(entryId)
    selectedId.value = entryId
  } catch (requestError) {
    error.value = requestError?.response?.data?.detail || 'That revision could not be compared.'
  } finally {
    loading.value = false
  }
}

async function restore(scope) {
  if (busy.value || !comparison.value) return
  const entryId = comparison.value.id
  restoring.value = true
  error.value = ''
  status.value = ''
  try {
    const result = await restoreConfigurationHistory(
      entryId,
      scope,
      comparison.value.current_revision,
    )
    status.value = result.notice
    try {
      entries.value = await listConfigurationHistory()
      if (entries.value.some((entry) => entry.id === entryId)) {
        comparison.value = await configurationHistoryDiff(entryId)
        selectedId.value = entryId
      } else {
        comparison.value = null
        selectedId.value = ''
      }
    } catch {
      error.value = 'The item was restored, but the current comparison could not be refreshed.'
    }
  } catch (requestError) {
    error.value = requestError?.response?.data?.detail || 'The saved item could not be restored.'
  } finally {
    restoring.value = false
  }
}

onMounted(refresh)
</script>

<style scoped>
.config-history {
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
  padding-top: 1rem;
  border-top: 1px solid var(--border-secondary);
}

.history-heading,
.history-layout,
.history-scopes {
  display: flex;
  gap: 0.75rem;
  align-items: flex-start;
  flex-wrap: wrap;
}

.config-history button {
  font: inherit;
  border: 1px solid var(--border-secondary);
  background: var(--bg-tertiary);
  color: var(--text-primary);
  border-radius: var(--radius-md);
  padding: 0.4rem 0.75rem;
}

.history-heading {
  justify-content: space-between;
}

.history-heading h3,
.history-heading p,
.history-diff h4,
.history-scopes p {
  margin: 0;
}

.history-list,
.history-diff ul {
  list-style: none;
  padding: 0;
  margin: 0;
}

.history-list {
  flex: 1 1 15rem;
  max-height: 22rem;
  overflow: auto;
}

.history-list button {
  width: 100%;
  display: flex;
  flex-direction: column;
  align-items: flex-start;
  text-align: left;
}

.history-list button.selected {
  border-color: var(--accent-primary);
}

.history-list small,
.history-diff span {
  color: var(--text-secondary);
  overflow-wrap: anywhere;
}

.history-diff {
  flex: 2 1 20rem;
  min-width: 0;
}

.history-diff li {
  display: grid;
  gap: 0.15rem;
  padding: 0.4rem 0;
  border-bottom: 1px solid var(--border-secondary);
}

.history-error {
  color: var(--status-warning);
}

@media (width <= 640px) {
  .history-layout {
    flex-direction: column;
  }

  .history-list,
  .history-diff {
    width: 100%;
  }
}
</style>
