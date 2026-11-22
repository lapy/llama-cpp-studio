<template>
  <section id="storage" class="storage-panel" aria-labelledby="storage-heading">
    <div class="storage-heading">
      <div>
        <h2 id="storage-heading">Storage</h2>
        <p class="storage-intro">
          Disk used by Studio's model store, caches and local configuration history. Cleanup only
          removes storage this surface owns; catalog models, engine installs, CUDA toolkits and the
          accounting database stay managed from their own screens.
        </p>
      </div>
      <button type="button" class="storage-refresh" :disabled="store.busy" @click="refresh">
        Refresh storage
      </button>
    </div>

    <p v-if="store.error" class="storage-error" role="alert">{{ store.error }}</p>
    <p v-if="store.status" class="storage-status" role="status" aria-live="polite">
      {{ store.status }}
    </p>

    <LoadingState v-if="store.loading && !store.inventory" message="Scanning storage…" />
    <p v-else-if="!store.inventory && !store.error" class="storage-note">
      No storage inventory has been scanned yet.
    </p>

    <div v-if="store.inventory" class="storage-body">
      <dl class="storage-filesystem">
        <div>
          <dt>Data root</dt>
          <dd class="storage-path">{{ store.inventory.filesystem.data_root }}</dd>
        </div>
        <div>
          <dt>Measured by Studio</dt>
          <dd>{{ formatBytesIEC(store.measuredBytes) }}</dd>
        </div>
        <div>
          <dt>Free on the data volume</dt>
          <dd>{{ formatBytesIEC(store.inventory.filesystem.free_bytes) }}</dd>
        </div>
        <div>
          <dt>Last scanned</dt>
          <dd>{{ generatedLabel }}</dd>
        </div>
      </dl>

      <p v-if="store.inventory.scan.quality === 'partial'" class="storage-warning" role="status">
        This scan is partial: sizes may be lower than reality. Refresh after the current activity
        finishes for a complete measurement.
      </p>
      <p v-if="store.inventory.scan.cached" class="storage-note">
        Served from the 10 second scan cache. Refresh for a new walk.
      </p>

      <div class="storage-categories">
        <article
          v-for="category in store.categories"
          :key="category.id"
          class="storage-category"
          :aria-labelledby="`storage-category-${category.id}`"
        >
          <div class="storage-category-heading">
            <div>
              <h3 :id="`storage-category-${category.id}`">{{ category.label }}</h3>
              <p class="storage-category-size">
                {{ formatBytesIEC(category.bytes) }}
                <span v-if="category.unmanaged_bytes">
                  · {{ formatBytesIEC(category.unmanaged_bytes) }} other files</span
                >
                · {{ category.unit_count }}
                {{ category.unit_count === 1 ? 'unit' : 'units' }}
              </p>
            </div>
            <span v-if="category.cleanable_bytes" class="storage-badge">
              {{ formatBytesIEC(category.cleanable_bytes) }} reclaimable
            </span>
            <span v-else class="storage-badge storage-badge--muted">
              {{ category.cleanable ? 'Nothing to reclaim' : 'Report only' }}
            </span>
          </div>

          <div
            class="storage-share"
            role="img"
            :aria-label="`${category.label} is ${sharePercent(category)}% of the storage Studio measured`"
          >
            <span class="storage-share-fill" :style="{ width: `${sharePercent(category)}%` }" />
          </div>

          <p class="storage-category-description">{{ category.description }}</p>

          <p v-if="category.blocked" class="storage-refusal">
            {{ category.blocked.message }}
            <RouterLink v-if="category.manage_path" :to="category.manage_path" class="storage-link">
              Manage {{ category.label.toLowerCase() }}
            </RouterLink>
          </p>

          <p v-if="!category.available" class="storage-note">
            No storage for this category was found in the current data root.
          </p>

          <details v-if="category.units.length" class="storage-units">
            <summary>Show {{ category.units.length }} units for {{ category.label }}</summary>
            <ul>
              <li v-for="unit in category.units" :key="unit.id">
                <span class="storage-unit-label">{{ unit.label }}</span>
                <span class="storage-unit-size">{{ formatBytesIEC(unit.bytes) }}</span>
                <span v-if="unit.blocked" class="storage-unit-blocked">
                  {{ unit.blocked.message }}
                  <RouterLink
                    v-if="unit.manage_path"
                    :to="unit.manage_path"
                    class="storage-link"
                    :aria-label="`Manage ${unit.label} from ${unit.manage_path}`"
                  >
                    Manage
                  </RouterLink>
                </span>
                <span v-else class="storage-unit-cleanable">Reclaimable</span>
              </li>
            </ul>
          </details>

          <p v-if="category.truncated" class="storage-note">
            Only the largest units are listed. Refresh after cleanup to see the rest.
          </p>

          <div v-if="category.id === 'history_backups'" class="storage-keep">
            <label for="storage-keep-history">Revisions to keep</label>
            <input
              id="storage-keep-history"
              v-model.number="historyKeep"
              type="number"
              min="1"
              max="100"
              step="1"
            />
            <span class="storage-note">Older revisions become reclaimable.</span>
          </div>

          <div v-if="category.cleanable" class="storage-category-actions">
            <button
              type="button"
              :disabled="store.busy || !category.cleanable_bytes"
              :aria-label="`Review cleanup for ${category.label}`"
              @click="startPreview(category)"
            >
              Review cleanup
            </button>
          </div>

          <div
            v-if="store.result && store.result.category === category.id"
            class="storage-result"
            role="status"
          >
            <p class="storage-result-heading">
              {{ resultHeading }} · {{ formatBytesIEC(store.result.freed_bytes) }} freed
            </p>
            <p>{{ store.result.notice }}</p>
            <ul v-if="store.result.removed.length">
              <li v-for="unit in store.result.removed" :key="`removed-${unit.id}`">
                Removed {{ unit.label }}
              </li>
            </ul>
            <ul v-if="store.result.failed.length" class="storage-result-failed">
              <li v-for="unit in store.result.failed" :key="`failed-${unit.id}`">
                {{ unit.label }}: {{ unit.message }}
              </li>
            </ul>
            <button type="button" @click="store.clearResult()">Dismiss result</button>
          </div>
        </article>
      </div>
    </div>

    <Dialog v-model:visible="confirmVisible" modal :header="planHeader" class="storage-dialog">
      <div v-if="store.plan" class="storage-plan">
        <p>
          {{ formatBytesIEC(store.plan.reclaimable_bytes) }} from
          {{ store.plan.units.length }}
          {{ store.plan.units.length === 1 ? 'unit' : 'units' }} will be deleted permanently.
        </p>
        <ul v-if="store.plan.units.length" class="storage-plan-units">
          <li v-for="unit in store.plan.units" :key="unit.id">
            <span class="storage-unit-label">{{ unit.label }}</span>
            <span class="storage-unit-size">{{ formatBytesIEC(unit.bytes) }}</span>
          </li>
        </ul>
        <div v-if="store.plan.refused.length" class="storage-plan-refused">
          <p>These units are refused and will stay:</p>
          <ul>
            <li v-for="unit in store.plan.refused" :key="unit.id">
              <span class="storage-unit-label">{{ unit.label }}</span>
              <span>{{ unit.message }}</span>
            </li>
          </ul>
        </div>
        <p class="storage-phrase">
          Confirmation phrase:
          <strong>{{ store.plan.confirmation_phrase }}</strong>
        </p>
        <label class="storage-confirm">
          <input v-model="confirmed" type="checkbox" :disabled="store.applying" />
          I understand the selected files are deleted permanently.
        </label>
        <div class="storage-plan-actions">
          <button
            type="button"
            :disabled="!confirmed || store.applying"
            :aria-label="`Delete the selected units from ${store.plan.category_label}`"
            @click="store.applyPlan()"
          >
            Delete selected
          </button>
          <button type="button" :disabled="store.applying" @click="cancelPlan">
            Cancel cleanup
          </button>
        </div>
      </div>
    </Dialog>
  </section>
</template>

<script setup>
import { computed, onMounted, ref, watch } from 'vue'
import Dialog from 'primevue/dialog'
import LoadingState from '@/components/common/LoadingState.vue'
import { useStorageStore } from '@/stores/storage'
import { formatBytesIEC } from '@/utils/formatting'

const store = useStorageStore()
const confirmed = ref(false)
const confirmVisible = ref(false)
const historyKeep = ref(50)

const generatedLabel = computed(() => {
  const generated = Date.parse(store.inventory?.generated_at || '')
  return Number.isNaN(generated) ? 'Unknown' : new Date(generated).toLocaleString()
})

const resultHeading = computed(() => {
  const outcome = store.result?.outcome
  if (outcome === 'completed') return 'Cleanup completed'
  if (outcome === 'partial') return 'Cleanup partial'
  return 'Cleanup refused'
})

watch(
  () => store.plan,
  (plan) => {
    confirmed.value = false
    confirmVisible.value = Boolean(plan)
  },
)

const planHeader = computed(() =>
  store.plan ? `Confirm cleanup for ${store.plan.category_label}` : 'Confirm cleanup',
)

watch(confirmVisible, (visible) => {
  if (!visible && store.plan) store.clearPlan()
})

function sharePercent(category) {
  const total = Number(store.measuredBytes || 0)
  const bytes = Number(category.bytes || 0)
  if (!total || !bytes) return 0
  return Math.max(1, Math.round((bytes / total) * 100))
}

function refresh() {
  store.loadInventory({ refresh: true })
}

function startPreview(category) {
  store.clearResult()
  const keep = category.id === 'history_backups' ? historyKeep.value : null
  store.previewCleanup(category.id, [], keep)
}

function cancelPlan() {
  store.clearPlan()
}

onMounted(() => {
  store.loadInventory()
})
</script>

<style scoped>
.storage-panel {
  display: flex;
  flex-direction: column;
  gap: 1rem;
  min-width: 0;
}

.storage-heading {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  flex-wrap: wrap;
  gap: 0.75rem;
}

.storage-heading h2 {
  margin: 0;
  font-size: 1rem;
}

.storage-intro {
  margin: 0.35rem 0 0;
  color: var(--text-secondary);
  font-size: 0.85rem;
  max-width: 60rem;
}

.storage-refresh,
.storage-category-actions button,
.storage-plan-actions button,
.storage-result button {
  font: inherit;
  border: 1px solid var(--border-secondary);
  background: var(--bg-tertiary);
  color: var(--text-primary);
  border-radius: var(--radius-md);
  padding: 0.4rem 0.75rem;
}

.storage-refresh:disabled,
.storage-category-actions button:disabled,
.storage-plan-actions button:disabled,
.storage-result button:disabled {
  cursor: not-allowed;
  opacity: 0.6;
}

.storage-error {
  color: var(--status-warning);
  margin: 0;
}

.storage-status {
  margin: 0;
  color: var(--text-secondary);
}

.storage-note,
.storage-warning,
.storage-category-description {
  margin: 0;
  color: var(--text-secondary);
  font-size: 0.85rem;
}

.storage-warning {
  color: var(--status-warning);
}

.storage-body,
.storage-categories {
  display: flex;
  flex-direction: column;
  gap: 1rem;
  min-width: 0;
}

.storage-filesystem {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(min(100%, 13rem), 1fr));
  gap: 0.85rem;
  margin: 0;
}

.storage-filesystem dt {
  color: var(--text-secondary);
  font-size: 0.8rem;
}

.storage-filesystem dd {
  margin: 0.25rem 0 0;
  overflow-wrap: anywhere;
}

.storage-path {
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
  font-size: 0.8rem;
}

.storage-category {
  display: flex;
  flex-direction: column;
  gap: 0.65rem;
  padding: 1rem;
  border: 1px solid var(--border-primary);
  border-radius: var(--radius-md);
  background: var(--bg-card);
  min-width: 0;
}

.storage-category-heading {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  flex-wrap: wrap;
  gap: 0.5rem;
}

.storage-category-heading h3 {
  margin: 0;
  font-size: 0.95rem;
}

.storage-category-size {
  margin: 0.25rem 0 0;
  color: var(--text-secondary);
  font-size: 0.85rem;
}

.storage-badge {
  font-size: 0.8rem;
  padding: 0.2rem 0.5rem;
  border: 1px solid var(--border-secondary);
  border-radius: 999px;
  white-space: nowrap;
}

.storage-badge--muted {
  color: var(--text-secondary);
}

.storage-share {
  height: 0.4rem;
  border-radius: 999px;
  background: var(--bg-tertiary);
  overflow: hidden;
}

.storage-share-fill {
  display: block;
  height: 100%;
  border-radius: inherit;
  background: var(--accent-primary);
}

.storage-refusal {
  margin: 0;
  color: var(--status-warning);
  font-size: 0.85rem;
  overflow-wrap: anywhere;
}

.storage-link {
  color: var(--accent-primary);
  margin-left: 0.35rem;
}

.storage-units summary {
  cursor: pointer;
  font-size: 0.85rem;
}

.storage-units ul,
.storage-plan-units,
.storage-plan-refused ul,
.storage-result ul {
  list-style: none;
  padding: 0;
  margin: 0.5rem 0 0;
  display: flex;
  flex-direction: column;
  gap: 0.4rem;
}

.storage-units li,
.storage-plan-units li {
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
  justify-content: space-between;
  align-items: baseline;
  padding-bottom: 0.35rem;
  border-bottom: 1px solid var(--border-secondary);
  font-size: 0.85rem;
  min-width: 0;
}

.storage-unit-label {
  overflow-wrap: anywhere;
}

.storage-unit-size {
  color: var(--text-secondary);
  white-space: nowrap;
}

.storage-unit-blocked {
  flex-basis: 100%;
  color: var(--status-warning);
  overflow-wrap: anywhere;
}

.storage-unit-cleanable {
  color: var(--text-secondary);
}

.storage-keep {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 0.5rem;
  font-size: 0.85rem;
}

.storage-keep input {
  width: 5rem;
  font: inherit;
  padding: 0.25rem 0.4rem;
  border: 1px solid var(--border-secondary);
  border-radius: var(--radius-sm);
  background: var(--bg-tertiary);
  color: var(--text-primary);
}

.storage-category-actions {
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
}

.storage-plan,
.storage-result {
  display: flex;
  flex-direction: column;
  gap: 0.6rem;
  padding: 0.85rem;
  border: 1px solid var(--border-secondary);
  border-radius: var(--radius-md);
  background: var(--bg-tertiary);
  font-size: 0.85rem;
}

.storage-plan h4,
.storage-plan p,
.storage-result p {
  margin: 0;
}

.storage-plan-refused,
.storage-result-failed {
  color: var(--status-warning);
}

.storage-phrase strong {
  overflow-wrap: anywhere;
}

.storage-confirm {
  display: flex;
  align-items: flex-start;
  gap: 0.5rem;
}

.storage-plan-actions {
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
}

.storage-result-heading {
  font-weight: 600;
}

@media (max-width: 600px) {
  .storage-category {
    padding: 0.75rem;
  }
}
</style>
