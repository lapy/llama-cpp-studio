<template>
  <PageSection
    id="storage"
    title="Storage"
    description="Disk used by Studio's model store, caches and local configuration history. Cleanup only removes storage this surface owns; catalog models, engine installs, CUDA toolkits and the accounting database stay managed from their own screens."
  >
    <template #actions>
      <Button
        label="Refresh storage"
        icon="pi pi-refresh"
        severity="secondary"
        outlined
        size="small"
        :loading="store.loading"
        @click="refresh"
      />
    </template>

    <StateBanner v-if="store.error && !store.plan" severity="error">
      {{ store.error }}
      <template #actions>
        <Button
          label="Retry"
          icon="pi pi-refresh"
          severity="secondary"
          outlined
          size="small"
          :loading="store.loading"
          @click="refresh"
        />
      </template>
    </StateBanner>

    <LoadingState v-if="store.loading && !store.inventory" message="Scanning storage…" />
    <EmptyState
      v-else-if="!store.inventory && !store.error"
      icon="pi pi-database"
      title="No storage inventory yet"
      description="Scan the data root to see how much disk each Studio category uses and what can be reclaimed."
    >
      <Button label="Scan storage" icon="pi pi-search" size="small" @click="refresh" />
    </EmptyState>

    <template v-if="store.inventory">
      <dl class="storage-filesystem">
        <div class="storage-filesystem__item">
          <dt>Data root</dt>
          <dd class="storage-path">{{ store.inventory.filesystem.data_root }}</dd>
        </div>
        <div class="storage-filesystem__item">
          <dt>Measured by Studio</dt>
          <dd>{{ formatBytesIEC(store.measuredBytes) }}</dd>
        </div>
        <div class="storage-filesystem__item">
          <dt>Free on the data volume</dt>
          <dd>{{ formatBytesIEC(store.inventory.filesystem.free_bytes) }}</dd>
        </div>
        <div class="storage-filesystem__item">
          <dt>Last scanned</dt>
          <dd>{{ generatedLabel }}</dd>
        </div>
      </dl>

      <StateBanner v-if="store.inventory.scan?.quality === 'partial'" severity="warning">
        This scan is partial: sizes may be lower than reality. Refresh after the current activity
        finishes for a complete measurement.
        <template #actions>
          <Button
            label="Refresh scan"
            icon="pi pi-refresh"
            severity="secondary"
            outlined
            size="small"
            :loading="store.loading"
            @click="refresh"
          />
        </template>
      </StateBanner>
      <StateBanner v-else-if="store.inventory.scan?.cached" severity="info">
        Served from the 10 second scan cache. Refresh for a new walk.
      </StateBanner>

      <StateBanner v-if="store.status && !store.result" severity="info">
        {{ store.status }}
      </StateBanner>

      <ul class="storage-categories">
        <li
          v-for="category in store.categories"
          :key="category.id"
          class="storage-category"
          :aria-labelledby="`storage-category-${category.id}`"
        >
          <div class="storage-category__head">
            <div class="storage-category__identity">
              <h3 :id="`storage-category-${category.id}`" class="page-subheading">
                {{ category.label }}
              </h3>
              <p class="storage-category__size">
                {{ formatBytesIEC(category.bytes) }}
                <span v-if="category.unmanaged_bytes">
                  · {{ formatBytesIEC(category.unmanaged_bytes) }} other files</span
                >
                · {{ category.unit_count }}
                {{ category.unit_count === 1 ? 'unit' : 'units' }}
              </p>
            </div>
            <Tag :value="badgeLabel(category)" :severity="badgeSeverity(category)" />
          </div>

          <div
            class="storage-share"
            role="img"
            :aria-label="`${category.label} is ${sharePercent(category)}% of the storage Studio measured`"
          >
            <span class="storage-share__fill" :style="{ width: `${sharePercent(category)}%` }" />
          </div>

          <p class="page-form__hint">{{ category.description }}</p>

          <StateBanner v-if="category.blocked" severity="warning">
            {{ category.blocked.message }}
            <template v-if="category.manage_path" #actions>
              <RouterLink :to="category.manage_path" class="page-action-link">
                Manage {{ category.label.toLowerCase() }}
              </RouterLink>
            </template>
          </StateBanner>

          <p v-if="!category.available" class="page-form__hint">
            No storage for this category was found in the current data root.
          </p>

          <details v-if="category.units.length" class="storage-units">
            <summary>Show {{ category.units.length }} units for {{ category.label }}</summary>
            <ul class="storage-unit-list">
              <li v-for="unit in category.units" :key="unit.id" class="storage-unit">
                <span class="storage-unit__label">{{ unit.label }}</span>
                <span class="storage-unit__size">{{ formatBytesIEC(unit.bytes) }}</span>
                <span v-if="unit.blocked" class="storage-unit__blocked">
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
                <span v-else class="storage-unit__cleanable">Reclaimable</span>
              </li>
            </ul>
          </details>

          <p v-if="category.truncated" class="page-form__hint">
            Only the largest units are listed. Refresh after cleanup to see the rest.
          </p>

          <div v-if="category.id === 'history_backups'" class="storage-keep">
            <label class="page-form__label" for="storage-keep-history">Revisions to keep</label>
            <InputText
              id="storage-keep-history"
              v-model="historyKeep"
              type="number"
              min="1"
              max="100"
              step="1"
            />
            <p class="page-form__hint">Older revisions become reclaimable.</p>
          </div>

          <div v-if="category.cleanable" class="storage-category__actions">
            <Button
              label="Review cleanup"
              icon="pi pi-trash"
              severity="secondary"
              outlined
              size="small"
              :disabled="store.busy || !category.cleanable_bytes"
              :loading="previewingId === category.id"
              :aria-label="`Review cleanup for ${category.label}`"
              @click="startPreview(category)"
            />
          </div>

          <StateBanner
            v-if="store.result && store.result.category === category.id"
            class="storage-result"
            :severity="resultSeverity"
            role="status"
          >
            <p class="storage-result__heading">
              {{ resultHeading }} · {{ formatBytesIEC(store.result.freed_bytes) }} freed
            </p>
            <p>{{ store.result.notice }}</p>
            <ul v-if="store.result.removed.length" class="storage-result__list">
              <li v-for="unit in store.result.removed" :key="`removed-${unit.id}`">
                Removed {{ unit.label }}
              </li>
            </ul>
            <ul
              v-if="store.result.failed.length"
              class="storage-result__list storage-result__failed"
            >
              <li v-for="unit in store.result.failed" :key="`failed-${unit.id}`">
                {{ unit.label }}: {{ unit.message }}
              </li>
            </ul>
            <template #actions>
              <Button
                label="Dismiss result"
                severity="secondary"
                text
                size="small"
                @click="store.clearResult()"
              />
            </template>
          </StateBanner>
        </li>
      </ul>
    </template>

    <Dialog
      v-model:visible="confirmVisible"
      modal
      :header="planHeader"
      class="dialog-width-sm"
      @hide="onDialogHide"
    >
      <div v-if="store.plan" class="dialog-stack">
        <StateBanner v-if="store.error" severity="error">{{ store.error }}</StateBanner>
        <p>
          {{ formatBytesIEC(store.plan.reclaimable_bytes) }} from
          {{ store.plan.units.length }}
          {{ store.plan.units.length === 1 ? 'unit' : 'units' }} will be deleted permanently.
        </p>
        <ul v-if="store.plan.units.length" class="storage-plan-units">
          <li v-for="unit in store.plan.units" :key="unit.id" class="storage-unit">
            <span class="storage-unit__label">{{ unit.label }}</span>
            <span class="storage-unit__size">{{ formatBytesIEC(unit.bytes) }}</span>
          </li>
        </ul>
        <div v-if="store.plan.refused.length" class="storage-plan-refused">
          <p>These units are refused and will stay:</p>
          <ul class="storage-plan-units">
            <li v-for="unit in store.plan.refused" :key="unit.id" class="storage-unit">
              <span class="storage-unit__label">{{ unit.label }}</span>
              <span>{{ unit.message }}</span>
            </li>
          </ul>
        </div>
        <p class="storage-phrase">
          Confirmation phrase:
          <strong>{{ store.plan.confirmation_phrase }}</strong>
        </p>
        <label class="storage-confirm">
          <Checkbox v-model="confirmed" binary :disabled="store.applying" />
          <span>I understand the selected files are deleted permanently.</span>
        </label>
      </div>
      <template #footer>
        <div class="dialog-actions">
          <Button
            label="Cancel cleanup"
            severity="secondary"
            outlined
            :disabled="store.applying"
            @click="cancelPlan"
          />
          <Button
            label="Delete selected"
            severity="danger"
            :disabled="!confirmed || store.applying"
            :loading="store.applying"
            :aria-label="`Delete the selected units from ${store.plan?.category_label || 'storage'}`"
            @click="applyPlan"
          />
        </div>
      </template>
    </Dialog>
  </PageSection>
</template>

<script setup>
import { computed, onMounted, ref, watch } from 'vue'
import Button from 'primevue/button'
import Checkbox from 'primevue/checkbox'
import Dialog from 'primevue/dialog'
import InputText from 'primevue/inputtext'
import Tag from 'primevue/tag'
import EmptyState from '@/components/common/EmptyState.vue'
import LoadingState from '@/components/common/LoadingState.vue'
import PageSection from '@/components/common/PageSection.vue'
import StateBanner from '@/components/common/StateBanner.vue'
import { watchDialogFocus } from '@/composables/useFocusReturn'
import { useStorageStore } from '@/stores/storage'
import { formatBytesIEC } from '@/utils/formatting'

const store = useStorageStore()
const confirmed = ref(false)
const confirmVisible = ref(false)
const historyKeep = ref(50)
const previewingId = ref('')
watchDialogFocus(confirmVisible)

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

const resultSeverity = computed(() => {
  const outcome = store.result?.outcome
  if (outcome === 'completed') return 'success'
  if (outcome === 'partial') return 'warning'
  return 'error'
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

function badgeLabel(category) {
  if (category.cleanable_bytes) return `${formatBytesIEC(category.cleanable_bytes)} reclaimable`
  return category.cleanable ? 'Nothing to reclaim' : 'Report only'
}

function badgeSeverity(category) {
  return category.cleanable_bytes ? 'info' : 'secondary'
}

function sharePercent(category) {
  const total = Number(store.measuredBytes || 0)
  const bytes = Number(category.bytes || 0)
  if (!total || !bytes) return 0
  return Math.max(1, Math.round((bytes / total) * 100))
}

function refresh() {
  store.loadInventory({ refresh: true })
}

async function startPreview(category) {
  if (store.busy) return
  previewingId.value = category.id
  store.clearResult()
  const keep = category.id === 'history_backups' ? Number(historyKeep.value) : null
  try {
    await store.previewCleanup(category.id, [], keep)
  } finally {
    previewingId.value = ''
  }
}

async function applyPlan() {
  await store.applyPlan()
}

function cancelPlan() {
  store.clearPlan()
}

function onDialogHide() {
  if (store.plan) store.clearPlan()
}

onMounted(() => {
  store.loadInventory()
})
</script>

<style scoped>
.storage-filesystem {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(min(100%, 13rem), 1fr));
  gap: 0.85rem;
  margin: 0;
}

.storage-filesystem__item {
  min-width: 0;
}

.storage-filesystem dt {
  color: var(--text-secondary);
  font-size: var(--text-xs);
}

.storage-filesystem dd {
  margin: 0.25rem 0 0;
  overflow-wrap: anywhere;
}

.storage-path {
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
  font-size: var(--text-xs);
}

.storage-categories {
  display: flex;
  flex-direction: column;
  gap: 0;
  margin: 0;
  padding: 0;
  list-style: none;
}

.storage-category {
  display: flex;
  flex-direction: column;
  gap: 0.65rem;
  min-width: 0;
  padding: 1rem 0;
  border-bottom: 1px solid var(--border-primary);
}

.storage-category__head {
  display: flex;
  flex-wrap: wrap;
  align-items: flex-start;
  justify-content: space-between;
  gap: 0.5rem;
}

.storage-category__identity {
  display: grid;
  gap: 0.2rem;
  min-width: 0;
}

.storage-category__size {
  margin: 0;
  color: var(--text-secondary);
  font-size: var(--text-sm);
  font-variant-numeric: tabular-nums;
}

.storage-share {
  height: 0.4rem;
  border-radius: 999px;
  background: var(--bg-tertiary);
  overflow: hidden;
}

.storage-share__fill {
  display: block;
  height: 100%;
  border-radius: inherit;
  background: var(--accent-primary);
}

.storage-category__actions {
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
}

.storage-units summary {
  cursor: pointer;
  font-size: var(--text-sm);
}

.storage-unit-list,
.storage-plan-units,
.storage-result__list {
  display: flex;
  flex-direction: column;
  gap: 0.4rem;
  margin: 0.5rem 0 0;
  padding: 0;
  list-style: none;
}

.storage-unit {
  display: flex;
  flex-wrap: wrap;
  align-items: baseline;
  justify-content: space-between;
  gap: 0.5rem;
  min-width: 0;
  padding-bottom: 0.35rem;
  border-bottom: 1px solid var(--border-secondary);
  font-size: var(--text-sm);
}

.storage-unit__label {
  overflow-wrap: anywhere;
}

.storage-unit__size {
  color: var(--text-secondary);
  white-space: nowrap;
}

.storage-unit__blocked {
  flex-basis: 100%;
  color: var(--status-warning);
  overflow-wrap: anywhere;
}

.storage-unit__cleanable {
  color: var(--text-secondary);
}

.storage-keep {
  display: grid;
  gap: 0.35rem;
  min-width: 0;
  max-width: 18rem;
}

.storage-result__heading {
  margin: 0;
  font-weight: 600;
}

.storage-result__list {
  font-size: var(--text-sm);
}

.storage-result__failed {
  color: var(--status-warning);
}

.storage-phrase {
  margin: 0;
  overflow-wrap: anywhere;
}

.storage-confirm {
  display: flex;
  align-items: flex-start;
  gap: 0.5rem;
}

.storage-plan-refused {
  color: var(--status-warning);
}

.storage-plan-refused p {
  margin: 0;
}

.storage-link {
  color: var(--accent-primary);
  margin-left: 0.35rem;
}
</style>
