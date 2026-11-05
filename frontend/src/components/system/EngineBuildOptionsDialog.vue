<template>
  <Dialog
    :visible="visible"
    :header="header"
    modal
    class="build-settings-dialog dialog-width-md"
    @update:visible="$emit('update:visible', $event)"
  >
    <div class="dialog-body build-settings-body">
      <PersistenceAlert :notice="notice" @retry="$emit('retry')" @refresh="$emit('refresh')" />
      <p
        v-for="note in notes"
        :key="note.text"
        :class="['build-note', note.tone === 'info' ? 'build-note--info' : '']"
        :role="note.role || undefined"
      >
        {{ note.text }}
      </p>
      <p v-if="frozen" class="build-note build-note--info">
        These CMake options are frozen to
        <strong>{{ frozenVersion }}</strong
        >. Saving does not change global Build settings.
      </p>

      <template v-if="!frozen">
        <div v-for="field in fields" :key="field.key" class="form-field">
          <label>
            {{ field.label }}
            <span v-if="field.optional" class="optional">(optional)</span>
          </label>
          <InputText
            :model-value="form[field.key] || ''"
            :placeholder="field.placeholder || ''"
            class="w-full"
            @update:model-value="setField(field.key, $event)"
          />
          <small v-if="field.hint">{{ field.hint }}</small>
        </div>
      </template>

      <div class="form-field">
        <label>{{ buildTypeLabel }}</label>
        <Select
          :model-value="config[buildTypeKey]"
          :options="buildTypes"
          option-label="label"
          option-value="value"
          :placeholder="buildTypePlaceholder"
          class="w-full"
          @update:model-value="setOption(buildTypeKey, $event)"
        />
      </div>

      <div v-if="optionsLoading" class="build-note build-note--info">Loading build options…</div>

      <template v-for="category in categories" :key="category.id">
        <div v-if="category.layout === 'grid'" class="form-field">
          <div v-if="category.note" class="build-note build-note--info">
            {{ category.note }}
          </div>
          <label class="build-options-section">{{ category.label }}</label>
          <div class="toggle-grid">
            <div v-for="opt in primaryOptions(category)" :key="opt.key" class="toggle-row">
              <ToggleSwitch
                :model-value="config[opt.key]"
                :disabled="Boolean(opt.disabled)"
                @update:model-value="setOption(opt.key, $event)"
              />
              <div>
                <span class="opt-label">{{ opt.label }}</span>
                <small class="opt-desc">{{ opt.desc }}</small>
                <small v-if="opt.disabledReason" class="opt-block-reason">
                  {{ opt.disabledReason }}
                </small>
              </div>
            </div>
          </div>
          <details v-if="secondaryOptions(category).length" class="build-options-details">
            <summary class="build-options-section">
              {{ category.moreLabel || 'More backends' }}
            </summary>
            <div class="toggle-grid">
              <div v-for="opt in secondaryOptions(category)" :key="opt.key" class="toggle-row">
                <ToggleSwitch
                  :model-value="config[opt.key]"
                  :disabled="Boolean(opt.disabled)"
                  @update:model-value="setOption(opt.key, $event)"
                />
                <div>
                  <span class="opt-label">{{ opt.label }}</span>
                  <small class="opt-desc">{{ opt.desc }}</small>
                  <small v-if="opt.disabledReason" class="opt-block-reason">
                    {{ opt.disabledReason }}
                  </small>
                </div>
              </div>
            </div>
          </details>
        </div>

        <details v-else class="build-options-details">
          <summary class="build-options-section">
            {{ category.label }}
            <span class="build-advanced-hint">{{ category.hint || 'advanced' }}</span>
          </summary>
          <div v-if="category.note" class="build-note build-note--info">
            {{ category.note }}
          </div>
          <div class="toggle-grid">
            <template v-for="opt in category.options || []" :key="opt.key">
              <div v-if="opt.type === 'bool'" class="toggle-row">
                <ToggleSwitch
                  :model-value="config[opt.key]"
                  :disabled="Boolean(opt.disabled)"
                  @update:model-value="setOption(opt.key, $event)"
                />
                <div>
                  <span class="opt-label">{{ opt.label }}</span>
                  <small class="opt-desc">{{ opt.desc }}</small>
                  <small v-if="opt.disabledReason" class="opt-block-reason">
                    {{ opt.disabledReason }}
                  </small>
                </div>
              </div>
              <div v-else-if="opt.type === 'int'" class="opt-string-field">
                <span class="opt-label">{{ opt.label }}</span>
                <small class="opt-desc">{{ opt.desc }}</small>
                <InputNumber
                  :model-value="config[opt.key]"
                  :min="opt.min ?? 0"
                  :max="opt.max ?? 256"
                  class="w-full mt-1"
                  @update:model-value="setOption(opt.key, $event)"
                />
              </div>
              <div v-else class="opt-string-field">
                <span class="opt-label">{{ opt.label }}</span>
                <small class="opt-desc">{{ opt.desc }}</small>
                <Select
                  v-if="opt.type === 'enum'"
                  :model-value="config[opt.key]"
                  :options="opt.enum_values || []"
                  class="w-full mt-1"
                  @update:model-value="setOption(opt.key, $event)"
                />
                <InputText
                  v-else
                  :model-value="config[opt.key]"
                  :placeholder="opt.placeholder || ''"
                  class="w-full mt-1"
                  @update:model-value="setOption(opt.key, $event)"
                />
              </div>
            </template>
          </div>
        </details>
      </template>

      <details class="build-options-details">
        <summary class="build-options-section">
          Custom flags
          <span class="build-advanced-hint">advanced</span>
        </summary>
        <div class="form-field">
          <label>Custom CMake args <span class="optional">(optional)</span></label>
          <InputText
            :model-value="config.custom_cmake_args"
            placeholder="e.g. -DFOO=ON -DBAR=OFF"
            class="w-full"
            @update:model-value="setOption('custom_cmake_args', $event)"
          />
        </div>
        <div class="form-field">
          <label>CFLAGS / CXXFLAGS <span class="optional">(optional)</span></label>
          <div class="flags-row">
            <InputText
              :model-value="config.cflags"
              placeholder="CFLAGS"
              class="flex-1"
              @update:model-value="setOption('cflags', $event)"
            />
            <InputText
              :model-value="config.cxxflags"
              placeholder="CXXFLAGS"
              class="flex-1"
              @update:model-value="setOption('cxxflags', $event)"
            />
          </div>
        </div>
      </details>
    </div>
    <template #footer>
      <Button
        label="Cancel"
        severity="secondary"
        outlined
        @click="$emit('update:visible', false)"
      />
      <Button
        :label="saveAction.label"
        icon="pi pi-save"
        severity="secondary"
        :loading="Boolean(saveAction.loading)"
        :disabled="Boolean(saveAction.disabled)"
        @click="$emit('save')"
      />
      <Button
        v-if="frozen && rebuildAction"
        :label="rebuildAction.label"
        icon="pi pi-refresh"
        severity="info"
        :loading="rebuildAction.loading"
        @click="$emit('rebuild')"
      />
      <Button
        v-if="!frozen"
        :label="primaryAction.label"
        :icon="primaryAction.icon"
        severity="info"
        :loading="primaryAction.loading"
        @click="$emit('build')"
      />
    </template>
  </Dialog>
</template>

<script setup>
/**
 * Schema-driven CMake build-options dialog shared by the engine Build settings
 * dialogs (llama-family and audio.cpp).
 *
 * The component is a pure renderer: the caller owns catalog fetching, engine
 * capability checks, version-freeze mode and every save/build/rebuild call.
 * It never references engine ids.
 *
 * `categories` is an annotated copy of the backend build-options payload:
 *   { id, label, layout: 'grid' | 'advanced', note, moreLabel,
 *     options: [{ key, type, label, desc, enum_values, min, max,
 *                 primary, disabled, disabledReason, placeholder }] }
 * `grid` renders open backend toggles (options with `primary === false` go
 * under "More backends"); `advanced` renders a collapsed <details> section.
 *
 * `fields` describes the editable global build fields (repo/ref/suffix):
 *   { key, label, optional, placeholder, hint } — hidden while frozen.
 *
 * `actions` carries button state as data:
 *   { save: { label, loading, disabled },
 *     primary: { label, icon, loading },   // Build now / Build from source
 *     rebuild: { label, loading } }        // optional: Save & retry/rebuild
 */
import { computed } from 'vue'
import Button from 'primevue/button'
import Dialog from 'primevue/dialog'
import InputText from 'primevue/inputtext'
import InputNumber from 'primevue/inputnumber'
import Select from 'primevue/select'
import ToggleSwitch from 'primevue/toggleswitch'
import PersistenceAlert from '@/components/common/PersistenceAlert.vue'

const props = defineProps({
  visible: { type: Boolean, default: false },
  header: { type: String, required: true },
  form: { type: Object, required: true },
  configKey: { type: String, default: 'build_config' },
  buildTypeKey: { type: String, default: 'build_type' },
  buildTypeLabel: { type: String, default: 'Build type' },
  buildTypes: { type: Array, default: () => [] },
  buildTypePlaceholder: { type: String, default: '' },
  fields: { type: Array, default: () => [] },
  categories: { type: Array, default: () => [] },
  notes: { type: Array, default: () => [] },
  notice: { type: Object, default: null },
  optionsLoading: { type: Boolean, default: false },
  frozen: { type: Boolean, default: false },
  frozenVersion: { type: String, default: '' },
  actions: { type: Object, default: () => ({}) },
})

const emit = defineEmits([
  'update:visible',
  'field',
  'option',
  'save',
  'build',
  'rebuild',
  'retry',
  'refresh',
])

const config = computed(() => props.form?.[props.configKey] || {})
const saveAction = computed(() => {
  const save = props.actions.save || {}
  return {
    label: save.label || (props.frozen ? 'Save' : 'Save settings'),
    loading: Boolean(save.loading),
    disabled: Boolean(save.disabled),
  }
})
const primaryAction = computed(() => {
  const primary = props.actions.primary || {}
  return {
    label: primary.label || 'Build now',
    icon: primary.icon || 'pi pi-cog',
    loading: Boolean(primary.loading),
  }
})
const rebuildAction = computed(() => {
  if (!props.frozen || !props.actions.rebuild) return null
  const rebuild = props.actions.rebuild
  return { label: rebuild.label || 'Save & rebuild', loading: Boolean(rebuild.loading) }
})

function primaryOptions(category) {
  return (category.options || []).filter((opt) => opt.primary !== false)
}

function secondaryOptions(category) {
  return (category.options || []).filter((opt) => opt.primary === false)
}

function setField(key, value) {
  emit('field', key, value)
}

function setOption(key, value) {
  emit('option', key, value)
}
</script>

<style scoped>
.toggle-grid {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}

.toggle-row {
  display: flex;
  align-items: flex-start;
  gap: 0.75rem;
}

.opt-label {
  font-size: 0.875rem;
  font-weight: 500;
  display: block;
}
.opt-desc {
  font-size: var(--text-xs);
  color: var(--text-secondary);
  display: block;
}
.opt-block-reason {
  font-size: var(--text-xs);
  color: var(--text-secondary);
  display: block;
  margin-top: 0.15rem;
}

.opt-string-field {
  display: flex;
  flex-direction: column;
  gap: 0.15rem;
  padding: 0.15rem 0;
}

.build-settings-body {
  max-height: min(70vh, 640px);
  overflow-y: auto;
  padding-right: 0.25rem;
}

.build-options-details {
  margin-bottom: 0.65rem;
  border: 1px solid var(--border-primary);
  border-radius: 6px;
  padding: 0.35rem 0.6rem 0.5rem;
  background: var(--bg-tertiary);
}

.build-options-details > summary {
  cursor: pointer;
  list-style: none;
  margin-bottom: 0.15rem;
  user-select: none;
}

.build-options-details > summary::-webkit-details-marker {
  display: none;
}
.build-options-details > summary::before {
  content: '▸ ';
  color: var(--text-secondary);
}
.build-options-details[open] > summary::before {
  content: '▾ ';
}
.build-options-details[open] > summary {
  margin-bottom: 0.45rem;
}

.build-advanced-hint {
  margin-left: 0.4rem;
  font-size: var(--text-xs);
  font-weight: 500;
  color: var(--text-secondary);
  opacity: 0.85;
  text-transform: uppercase;
  letter-spacing: 0.03em;
}

.build-options-section {
  font-size: 0.8rem;
  font-weight: 600;
  color: var(--text-secondary);
  margin-bottom: 0.25rem;
  display: block;
}
.flags-row {
  display: flex;
  gap: 0.5rem;
}

.build-note {
  font-size: 0.8rem;
  padding: 0.5rem 0.6rem;
  border-radius: 6px;
  margin-bottom: 0.5rem;
}
.build-note--info {
  background: var(--status-info-soft);
  color: var(--text-primary);
  border: 1px solid var(--border-primary);
}
.build-note strong {
  font-weight: 600;
}

@media (max-width: 768px) {
  .flags-row {
    flex-wrap: wrap;
  }
}
</style>
