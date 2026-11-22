<template>
  <div class="table-shell library-list">
    <div class="table-shell__scroll">
      <table aria-label="Model library">
        <thead>
          <tr>
            <th scope="col">Model</th>
            <th scope="col">Status</th>
            <th scope="col" class="library-list__size-heading">File size</th>
            <th scope="col" class="library-list__actions-heading">Actions</th>
          </tr>
        </thead>
        <tbody>
          <tr
            v-for="line in lines"
            :key="line.key"
            class="library-list__row"
            :class="{
              'library-list__row--variant': line.kind === 'variant',
              'is-running': line.running,
            }"
          >
            <td class="library-list__model">
              <div class="library-list__title">
                <input
                  v-if="line.quant"
                  type="checkbox"
                  class="library-list__select"
                  :checked="isSelected(line.quant.id)"
                  :aria-label="`Select ${line.title}`"
                  @change="$emit('toggle-selection', line.quant.id)"
                />
                <span class="library-list__name" :title="line.title">{{ line.title }}</span>
              </div>
              <div v-if="line.meta" class="library-list__meta">{{ line.meta }}</div>
            </td>
            <td class="library-list__state">
              <span
                v-if="line.status"
                class="library-list__status"
                :data-status="line.statusKey"
                :title="line.statusDetail"
                role="status"
                >{{ line.status }}</span
              >
              <span v-else class="library-list__status library-list__status--quiet">{{
                line.kind === 'group' ? 'Variants' : 'Stopped'
              }}</span>
            </td>
            <td class="library-list__size">
              {{ line.size || '—' }}
              <small v-if="line.kind === 'group'">Total on disk</small>
            </td>
            <td class="library-list__actions">
              <ModelCardActions
                v-if="line.quant"
                :model="line.quant"
                :name="line.title"
                compact
                :is-starting="isStarting(line.quant)"
                :is-stopping="isStopping(line.quant)"
                @start="$emit('start', line.quant.id)"
                @stop="$emit('stop', line.quant.id)"
                @configure="$emit('configure', line.quant.id)"
                @audio="$emit('audio', line.quant.id)"
                @connect="$emit('connect', line.quant)"
                @refresh="$emit('refresh')"
                @delete="$emit('delete', line.quant.id)"
              />
              <details v-else class="row-menu">
                <summary :aria-label="`More actions for ${line.title}`">
                  <i class="pi pi-ellipsis-v" aria-hidden="true" />
                </summary>
                <button type="button" @click="$emit('delete-group', line.groupId)">
                  Delete group
                </button>
              </details>
            </td>
          </tr>
        </tbody>
      </table>
    </div>
  </div>
</template>

<script setup>
import ModelCardActions from '@/components/models/ModelCardActions.vue'

defineProps({
  lines: { type: Array, default: () => [] },
  isSelected: { type: Function, default: () => false },
  isStarting: { type: Function, default: () => false },
  isStopping: { type: Function, default: () => false },
})

defineEmits([
  'toggle-selection',
  'start',
  'stop',
  'configure',
  'audio',
  'connect',
  'refresh',
  'delete',
  'delete-group',
])
</script>

<style scoped>
.library-list__title {
  display: flex;
  align-items: baseline;
  gap: 0.45rem;
  min-width: 0;
}

.library-list__name {
  font-weight: 650;
  overflow-wrap: anywhere;
}

.library-list__meta {
  margin-top: 0.1rem;
  color: var(--text-secondary);
  font-size: var(--text-xs);
  overflow-wrap: anywhere;
}

.library-list__row--variant .library-list__name {
  padding-left: 0.85rem;
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
  font-size: 0.82rem;
  font-weight: 550;
}

.library-list__row.is-running {
  box-shadow: inset 2px 0 0 var(--status-success);
}

.library-list__status {
  font-size: var(--text-xs);
  font-weight: 650;
}

.library-list__status[data-status='running'] {
  color: var(--status-success);
}

.library-list__status[data-status='starting'],
.library-list__status[data-status='stopping'],
.library-list__status[data-status='stale'],
.library-list__status[data-status='unknown'] {
  color: var(--status-warning);
}

.library-list__status[data-status='error'] {
  color: var(--status-error);
}

.library-list__status--quiet {
  color: var(--text-secondary);
  font-weight: 400;
}

.library-list__size {
  color: var(--text-secondary);
  font-size: 0.82rem;
  font-variant-numeric: tabular-nums;
  text-align: right;
  padding-right: 0.75rem;
  white-space: nowrap;
}

.library-list__size small {
  display: block;
  color: var(--text-secondary);
  font-size: var(--text-xs);
  font-weight: 400;
}

.library-list__size-heading {
  text-align: right;
}

.library-list__actions {
  min-width: 10rem;
}

.library-list__actions :deep(.model-actions) {
  margin: 0;
  padding: 0;
  border: 0;
  justify-content: flex-end;
}

.library-list__select {
  flex: none;
  width: 1rem;
  height: 1rem;
  margin: 0;
  accent-color: var(--accent-primary);
}

.row-menu summary {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  min-width: 2rem;
  min-height: 2rem;
  border: 1px solid var(--border-primary);
  border-radius: var(--radius-sm);
  background: var(--bg-secondary);
  cursor: pointer;
  list-style: none;
}

.row-menu summary::-webkit-details-marker {
  display: none;
}

.row-menu button {
  display: block;
  width: 100%;
  margin-top: 0.25rem;
  padding: 0.35rem 0.6rem;
  border: 1px solid var(--border-primary);
  border-radius: var(--radius-sm);
  background: var(--bg-secondary);
  color: var(--status-error);
  cursor: pointer;
}

@media (max-width: 1100px) {
  .library-list thead {
    position: absolute;
    width: 1px;
    height: 1px;
    overflow: hidden;
    clip-path: inset(50%);
    white-space: nowrap;
  }

  .library-list__row {
    display: grid;
    grid-template-columns: minmax(0, 1fr) auto;
    grid-template-areas:
      'model model'
      'state size'
      'actions actions';
    gap: 0.4rem 0.75rem;
    padding: 0.7rem 0.8rem;
  }

  .library-list__model {
    grid-area: model;
  }

  .library-list__state {
    grid-area: state;
  }

  .library-list__size {
    grid-area: size;
    text-align: right;
  }

  .library-list__actions {
    grid-area: actions;
  }
}

@media (max-width: 640px) {
  .library-list__size-heading,
  .library-list__actions-heading {
    white-space: nowrap;
  }
}
</style>
