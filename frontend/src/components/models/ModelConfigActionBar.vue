<template>
  <section class="config-action-bar" role="region" aria-label="Configuration actions">
    <PageToolbar plain label="Configuration status and actions">
      <template #summary>
        <div class="config-action-bar__identity">
          <span class="config-action-bar__name">{{ modelLabel }}</span>
          <span v-if="quantization" class="config-action-bar__meta">{{ quantization }}</span>
          <span v-if="format" class="config-action-bar__meta">{{ format }}</span>
          <span v-if="engineLabel" class="config-action-bar__meta">{{ engineLabel }}</span>
        </div>

        <div class="config-action-bar__status" role="status" aria-live="polite">
          <Tag
            :value="status.label"
            :severity="status.severity"
            :class="['config-action-bar__pill', `config-action-bar__pill--${status.key}`]"
            :data-status-key="status.key"
          />
          <Tag
            v-if="running && !['running', 'stale'].includes(status.key)"
            value="Running"
            severity="success"
            class="config-action-bar__pill"
            data-status-key="running-extra"
          />
        </div>
      </template>

      <template #actions>
        <Button
          label="Save changes"
          icon="pi pi-save"
          :severity="dirty ? 'success' : 'secondary'"
          :outlined="!dirty"
          :loading="saving"
          :disabled="!canSave"
          v-tooltip.top="saveHint"
          @click="$emit('save')"
        />
        <Button
          v-if="showApply"
          :label="applyLabel"
          icon="pi pi-bolt"
          severity="warning"
          :loading="applying"
          :disabled="applyDisabled"
          v-tooltip.top="applyHint"
          @click="$emit('apply')"
        />
        <span
          v-if="running"
          class="config-action-bar__primary"
          v-tooltip.top="'This model is active. Open the library to manage or connect to it.'"
        >
          <Button
            label="Open in library"
            icon="pi pi-th-large"
            severity="success"
            @click="$emit('open-library')"
          />
        </span>
        <span v-else class="config-action-bar__primary" v-tooltip.top="startDisabledReason || ''">
          <Button
            label="Start model"
            icon="pi pi-play"
            :severity="canStart ? 'success' : 'secondary'"
            :outlined="!canStart"
            :loading="starting"
            :disabled="!canStart"
            :aria-describedby="startDisabledReason ? 'config-action-bar-start-reason' : undefined"
            @click="$emit('start')"
          />
        </span>
        <details class="config-secondary-actions" @keydown.esc.stop.prevent="closeSecondaryActions">
          <summary>More actions</summary>
          <div class="config-secondary-actions__menu" @click="closeSecondaryActionsMenu">
            <Button
              :label="reviewLabel"
              icon="pi pi-list-check"
              size="small"
              severity="secondary"
              outlined
              :disabled="!canReview"
              @click="$emit('review')"
            />
            <Button
              label="Templates"
              icon="pi pi-bookmark"
              size="small"
              severity="secondary"
              outlined
              @click="$emit('templates')"
            />
            <Button
              v-if="showImportCommand"
              label="Import command"
              icon="pi pi-sparkles"
              size="small"
              severity="secondary"
              outlined
              @click="$emit('import-command')"
            />
            <Button
              label="Reset to saved"
              icon="pi pi-refresh"
              size="small"
              severity="secondary"
              outlined
              @click="$emit('reset')"
            />
          </div>
        </details>
      </template>
    </PageToolbar>

    <div class="config-action-bar__secondary">
      <p class="config-action-bar__detail runtime-state__detail" :title="status.detail">
        {{ status.detail }}
      </p>
      <span
        v-if="!running && startDisabledReason"
        id="config-action-bar-start-reason"
        class="sr-only"
        >{{ startDisabledReason }}</span
      >
    </div>
  </section>
</template>

<script setup>
import Button from 'primevue/button'
import Tag from 'primevue/tag'

import PageToolbar from '@/components/common/PageToolbar.vue'

defineProps({
  modelLabel: { type: String, default: 'Model' },
  quantization: { type: String, default: '' },
  format: { type: String, default: '' },
  engineLabel: { type: String, default: '' },
  status: {
    type: Object,
    default: () => ({ key: 'applied', label: 'Applied', severity: 'success', detail: '' }),
  },
  dirty: { type: Boolean, default: false },
  canSave: { type: Boolean, default: false },
  saveHint: { type: String, default: '' },
  saving: { type: Boolean, default: false },
  showApply: { type: Boolean, default: false },
  applyLabel: { type: String, default: 'Apply settings' },
  applying: { type: Boolean, default: false },
  applyDisabled: { type: Boolean, default: false },
  applyHint: { type: String, default: '' },
  running: { type: Boolean, default: false },
  starting: { type: Boolean, default: false },
  canStart: { type: Boolean, default: false },
  startDisabledReason: { type: String, default: '' },
  canReview: { type: Boolean, default: false },
  reviewLabel: { type: String, default: 'Review changes' },
  showImportCommand: { type: Boolean, default: false },
})

defineEmits([
  'save',
  'apply',
  'start',
  'open-library',
  'review',
  'templates',
  'import-command',
  'reset',
])

function closeSecondaryActions(event) {
  event.currentTarget.open = false
  event.currentTarget.querySelector('summary')?.focus()
}

function closeSecondaryActionsMenu(event) {
  event.currentTarget.closest('details')?.removeAttribute('open')
}
</script>

<style scoped>
.config-action-bar {
  position: sticky;
  top: var(--app-header-height);
  z-index: 20;
  display: flex;
  flex-direction: column;
  gap: 0.4rem;
  padding: 0.6rem 0.8rem;
  border: 1px solid var(--border-primary);
  border-radius: var(--radius-lg);
  background: var(--bg-card);
  box-shadow: var(--shadow-md);
}

.config-action-bar .page-toolbar {
  align-items: center;
  padding: 0;
}

.config-action-bar :deep(.page-toolbar__summary) {
  flex: 1 1 22rem;
  gap: 0.4rem 0.75rem;
}

.config-action-bar__identity {
  display: flex;
  flex-wrap: wrap;
  align-items: baseline;
  gap: 0.2rem 0.55rem;
  min-width: 0;
  flex: 1 1 15rem;
}

.config-action-bar__name {
  min-width: 0;
  max-width: 100%;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  font-size: var(--text-base);
  font-weight: 650;
  color: var(--text-primary);
}

.config-action-bar__meta {
  font-size: var(--text-xs);
  color: var(--text-secondary);
  white-space: nowrap;
}

.config-action-bar__meta + .config-action-bar__meta::before {
  content: '·';
  margin-right: 0.55rem;
  color: var(--text-secondary);
  opacity: 0.65;
}

.config-action-bar__status {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  justify-content: flex-end;
  gap: 0.35rem;
  min-width: 8.5rem;
}

.config-action-bar__pill {
  font-size: var(--text-xs);
  white-space: nowrap;
}

.config-action-bar__primary {
  display: inline-flex;
}

.config-action-bar__secondary {
  display: flex;
  align-items: center;
  gap: 0.5rem 0.75rem;
  flex-wrap: wrap;
}

.config-action-bar__detail {
  flex: 1 1 20rem;
  min-width: 0;
  margin: 0;
  font-size: var(--text-xs);
  line-height: 1.4;
  color: var(--text-secondary);
}

.config-action-bar .config-secondary-actions {
  position: relative;
  margin-left: auto;
}

.config-action-bar .config-secondary-actions summary {
  cursor: pointer;
  padding: 0.15rem 0.35rem;
  border-radius: var(--radius-sm);
  font-size: var(--text-xs);
  font-weight: 600;
  color: var(--text-secondary);
  list-style-position: inside;
  user-select: none;
}

.config-action-bar .config-secondary-actions summary:hover {
  color: var(--text-primary);
}

.config-action-bar .config-secondary-actions__menu {
  position: absolute;
  right: 0;
  top: calc(100% + 0.25rem);
  z-index: 30;
  display: flex;
  flex-direction: column;
  align-items: stretch;
  gap: 0.3rem;
  min-width: 12rem;
  padding: 0.45rem;
  border: 1px solid var(--border-primary);
  border-radius: var(--radius-md);
  background: var(--bg-card);
  box-shadow: var(--shadow-md);
}

.config-action-bar .config-secondary-actions__menu .p-button {
  justify-content: flex-start;
}

@media (max-width: 900px) {
  .config-action-bar {
    top: calc(var(--app-header-height) + 0.25rem);
  }

  .config-action-bar .page-toolbar {
    display: grid;
    grid-template-columns: minmax(0, 1fr) auto;
    align-items: center;
    row-gap: 0.45rem;
  }

  .config-action-bar :deep(.page-toolbar__summary) {
    grid-column: 1 / -1;
    display: grid;
    grid-template-columns: minmax(0, 1fr) auto;
    align-items: center;
    width: 100%;
  }

  .config-action-bar__identity {
    grid-column: 1;
    flex-basis: auto;
  }

  .config-action-bar__status {
    grid-column: 2;
    min-width: 0;
  }

  .config-action-bar :deep(.page-toolbar__actions) {
    grid-column: 1 / -1;
    width: 100%;
    margin-left: 0;
    justify-content: flex-start;
  }

  .config-action-bar :deep(.page-toolbar__actions) .p-button {
    flex: 1 1 auto;
    justify-content: center;
  }

  .config-action-bar__primary {
    flex: 1 1 auto;
  }

  .config-action-bar__primary .p-button {
    width: 100%;
  }

  .config-action-bar__secondary {
    display: block;
  }

  .config-action-bar__detail {
    display: -webkit-box;
    flex: none;
    overflow: hidden;
    -webkit-box-orient: vertical;
    -webkit-line-clamp: 2;
  }

  .config-action-bar .config-secondary-actions {
    margin-left: 0;
  }

  .config-action-bar .config-secondary-actions__menu {
    left: auto;
    right: 0;
  }
}
</style>
