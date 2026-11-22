<template>
  <PageSection
    id="history"
    :heading-tag="headingTag"
    :variant="variant"
    title="Recent changes"
    description="Studio keeps up to 100 private, local pre-change revisions."
  >
    <template #actions>
      <Button
        label="Refresh history"
        icon="pi pi-refresh"
        severity="secondary"
        outlined
        size="small"
        :loading="loading"
        :disabled="restoring"
        @click="refresh"
      />
    </template>

    <StateBanner v-if="error" severity="error">
      {{ error }}
      <template #actions>
        <Button
          label="Retry"
          icon="pi pi-refresh"
          severity="secondary"
          outlined
          size="small"
          :loading="loading"
          :disabled="restoring"
          @click="refresh"
        />
      </template>
    </StateBanner>
    <StateBanner v-if="status" severity="success" role="status">{{ status }}</StateBanner>

    <LoadingState v-if="loading && !entries.length" message="Loading change history…" inline />
    <EmptyState
      v-else-if="!entries.length && !error"
      icon="pi pi-history"
      title="No configuration changes recorded yet"
      description="Studio saves a private revision before each settings change. Change a setting and it appears here with a diff you can restore."
    >
      <Button label="Refresh history" icon="pi pi-refresh" size="small" @click="refresh" />
    </EmptyState>

    <div v-else-if="entries.length" class="history-layout">
      <ul class="history-list" aria-label="Configuration revisions">
        <li v-for="entry in entries" :key="entry.id">
          <button
            type="button"
            class="history-entry"
            :class="{ 'history-entry--selected': selectedId === entry.id }"
            :disabled="busy"
            @click="selectEntry(entry.id)"
          >
            <span>{{ documentLabel(entry.document) }}</span>
            <small>{{ formatDate(entry.created_at) }} · {{ entry.reason }}</small>
          </button>
        </li>
      </ul>

      <div v-if="comparison" class="history-diff">
        <h3 class="page-subheading">Changes since this revision</h3>
        <p v-if="!comparison.changes.length" class="page-form__hint">
          The current document matches this revision.
        </p>
        <ul v-else class="history-changes">
          <li v-for="change in comparison.changes" :key="change.path">
            <code>{{ change.path }}</code>
            <span>{{ display(change.before) }} → {{ display(change.after) }}</span>
          </li>
        </ul>
        <div v-if="restorableScopes.length" class="history-scopes">
          <p class="page-form__hint">
            Restore one saved item. Running models are not restarted or published.
          </p>
          <div class="page-form__actions">
            <Button
              v-for="scope in restorableScopes"
              :key="`${scope.kind}:${scope.item_id}`"
              :label="`Restore ${scopeLabel(scope)}`"
              icon="pi pi-undo"
              severity="secondary"
              outlined
              size="small"
              :disabled="busy"
              @click="restore(scope)"
            />
          </div>
        </div>
      </div>
    </div>
  </PageSection>
</template>

<script setup>
import { computed, onMounted, ref } from 'vue'
import Button from 'primevue/button'
import EmptyState from '@/components/common/EmptyState.vue'
import LoadingState from '@/components/common/LoadingState.vue'
import PageSection from '@/components/common/PageSection.vue'
import StateBanner from '@/components/common/StateBanner.vue'
import {
  configurationHistoryDiff,
  listConfigurationHistory,
  restoreConfigurationHistory,
} from '@/api/configuration'

defineProps({
  headingTag: { type: String, default: 'h3', validator: (value) => /^h[2-6]$/.test(value) },
  variant: {
    type: String,
    default: 'default',
    validator: (value) => ['default', 'plain', 'compact'].includes(value),
  },
})

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
.history-layout {
  display: flex;
  flex-wrap: wrap;
  align-items: flex-start;
  gap: 1rem;
  min-width: 0;
}

.history-list {
  flex: 1 1 15rem;
  max-height: 22rem;
  overflow: auto;
  margin: 0;
  padding: 0;
  list-style: none;
}

.history-list li + li {
  margin-top: 0.4rem;
}

.history-entry {
  display: flex;
  flex-direction: column;
  align-items: flex-start;
  gap: 0.15rem;
  width: 100%;
  min-height: 2.75rem;
  padding: 0.55rem 0.7rem;
  border: 1px solid var(--border-secondary);
  border-radius: var(--radius-md);
  background: var(--bg-tertiary);
  color: var(--text-primary);
  font: inherit;
  font-size: var(--text-sm);
  text-align: left;
  cursor: pointer;
}

.history-entry:hover:not(:disabled) {
  background: var(--hover-bg);
}

.history-entry:disabled {
  cursor: not-allowed;
  opacity: 0.6;
}

.history-entry--selected {
  border-color: var(--accent-primary);
}

.history-entry small {
  color: var(--text-secondary);
  overflow-wrap: anywhere;
}

.history-diff {
  flex: 2 1 20rem;
  min-width: 0;
  display: grid;
  gap: 0.5rem;
}

.history-changes {
  display: flex;
  flex-direction: column;
  gap: 0;
  margin: 0;
  padding: 0;
  list-style: none;
}

.history-changes li {
  display: grid;
  gap: 0.15rem;
  padding: 0.4rem 0;
  border-bottom: 1px solid var(--border-secondary);
}

.history-changes span {
  color: var(--text-secondary);
  overflow-wrap: anywhere;
}

.history-scopes {
  display: grid;
  gap: 0.5rem;
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
