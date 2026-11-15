<template>
  <details ref="panel" class="library-presets">
    <summary class="library-presets__toggle">
      Presets<span v-if="presets.length"> ({{ presets.length }})</span>
    </summary>
    <div class="library-presets__panel">
      <form class="library-presets__save" @submit.prevent="saveCurrent">
        <label class="sr-only" for="library-preset-name">Preset name</label>
        <input
          id="library-preset-name"
          v-model="draftName"
          type="text"
          :maxlength="MAX_PRESET_NAME_LENGTH"
          placeholder="Preset name"
          autocomplete="off"
        />
        <button type="submit" :disabled="!draftName.trim()">Save current filters</button>
      </form>
      <p v-if="error" class="library-presets__message library-presets__message--error" role="alert">
        {{ error }}
      </p>
      <p v-else-if="notice" class="library-presets__message" role="status">{{ notice }}</p>
      <p v-if="!presets.length" class="library-presets__empty">No saved presets yet.</p>
      <ul v-else class="library-presets__list">
        <li v-for="preset in presets" :key="preset.id" class="library-presets__item">
          <button
            type="button"
            class="library-presets__apply"
            :aria-label="`Apply preset ${preset.name}`"
            @click="applyPreset(preset)"
          >
            <span class="library-presets__name">{{ preset.name }}</span>
            <span class="library-presets__summary">{{
              describePresetFilters(preset.filters)
            }}</span>
          </button>
          <button
            type="button"
            class="library-presets__delete"
            :aria-label="`Delete preset ${preset.name}`"
            @click="removePreset(preset)"
          >
            Delete
          </button>
        </li>
      </ul>
    </div>
  </details>
  <p class="sr-only" role="status">{{ announcement }}</p>
</template>

<script setup>
import { ref } from 'vue'
import {
  MAX_PRESET_NAME_LENGTH,
  deletePreset,
  describePresetFilters,
  loadPresets,
  normalizePresetFilters,
  savePreset,
} from '@/utils/libraryFilterPresets'

const props = defineProps({
  /** Current library filters, in the same shape the URL query uses. */
  filters: { type: Object, default: () => ({}) },
})
const emit = defineEmits(['apply'])

const presets = ref(loadPresets())
const panel = ref(null)
const draftName = ref('')
const error = ref('')
const notice = ref('')
const announcement = ref('')

function showError(value) {
  error.value = value
  notice.value = ''
}

function saveCurrent() {
  error.value = ''
  notice.value = ''
  announcement.value = ''
  try {
    const { preset, presets: next } = savePreset(draftName.value, props.filters)
    presets.value = next
    draftName.value = ''
    notice.value = `Saved “${preset.name}”.`
  } catch (e) {
    showError(e?.message || 'Could not save the preset.')
  }
}

function applyPreset(preset) {
  error.value = ''
  notice.value = ''
  emit('apply', normalizePresetFilters(preset.filters))
  announcement.value = `Applied “${preset.name}”.`
  if (panel.value) {
    panel.value.open = false
    panel.value.querySelector('summary')?.focus()
  }
}

function removePreset(preset) {
  error.value = ''
  announcement.value = ''
  try {
    presets.value = deletePreset(preset.id)
    notice.value = `Deleted “${preset.name}”.`
  } catch (e) {
    showError(e?.message || 'Could not delete the preset.')
  }
}
</script>

<style scoped>
.library-presets {
  position: relative;
}

.library-presets__toggle {
  display: inline-flex;
  align-items: center;
  min-height: 2.25rem;
  padding: 0.2rem 0.55rem;
  border: 1px solid var(--border-primary);
  border-radius: var(--radius-md);
  background: var(--bg-secondary);
  color: var(--text-primary);
  font-size: 0.8rem;
  list-style: none;
  cursor: pointer;
}

.library-presets__toggle::-webkit-details-marker {
  display: none;
}

.library-presets[open] > .library-presets__toggle {
  border-color: var(--accent-cyan);
}

.library-presets__panel {
  position: absolute;
  top: calc(100% + 0.35rem);
  left: 0;
  z-index: 6;
  display: grid;
  gap: 0.55rem;
  width: min(22rem, calc(100vw - 2rem));
  padding: 0.75rem;
  border: 1px solid var(--border-secondary);
  border-radius: var(--radius-md);
  background: var(--bg-secondary);
  box-shadow: var(--shadow-md, 0 8px 24px rgb(0 0 0 / 0.28));
}

.library-presets__save {
  display: flex;
  gap: 0.4rem;
  align-items: center;
}

.library-presets__save input {
  flex: 1 1 8rem;
  min-width: 0;
  min-height: 2.25rem;
  padding: 0.2rem 0.5rem;
  border: 1px solid var(--border-primary);
  border-radius: var(--radius-md);
  background: var(--bg-tertiary);
  color: var(--text-primary);
  font: inherit;
  font-size: 0.8rem;
}

.library-presets__save button,
.library-presets__delete {
  min-height: 2.25rem;
  padding: 0.2rem 0.55rem;
  border: 1px solid var(--border-primary);
  border-radius: var(--radius-md);
  background: var(--bg-tertiary);
  color: var(--text-primary);
  font: inherit;
  font-size: 0.8rem;
  cursor: pointer;
}

.library-presets__save button:disabled {
  opacity: 0.6;
  cursor: default;
}

.library-presets__message {
  margin: 0;
  font-size: 0.8rem;
  color: var(--text-secondary);
}

.library-presets__message--error {
  color: var(--status-error);
}

.library-presets__empty {
  margin: 0;
  font-size: 0.8rem;
  color: var(--text-secondary);
}

.library-presets__list {
  display: grid;
  gap: 0.35rem;
  margin: 0;
  padding: 0;
  max-height: 16rem;
  overflow: auto;
  list-style: none;
}

.library-presets__item {
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto;
  gap: 0.35rem;
  align-items: center;
}

.library-presets__apply {
  display: grid;
  gap: 0.1rem;
  min-width: 0;
  padding: 0.35rem 0.5rem;
  border: 1px solid var(--border-primary);
  border-radius: var(--radius-md);
  background: var(--bg-tertiary);
  color: var(--text-primary);
  font: inherit;
  text-align: left;
  cursor: pointer;
}

.library-presets__apply:hover {
  border-color: var(--accent-cyan);
}

.library-presets__name {
  font-size: 0.85rem;
  font-weight: 600;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.library-presets__summary {
  font-size: 0.75rem;
  color: var(--text-secondary);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.library-presets__delete {
  color: var(--status-error);
}

@media (max-width: 900px) {
  .library-presets {
    grid-column: 1 / -1;
  }

  .library-presets__toggle {
    width: 100%;
    justify-content: center;
  }

  .library-presets__panel {
    right: 0;
  }
}
</style>
