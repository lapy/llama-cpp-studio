<template>
  <Dialog
    :visible="visible"
    :header="header"
    modal
    class="build-settings-dialog dialog-width-md"
    @update:visible="$emit('update:visible', $event)"
  >
    <div class="dialog-stack build-settings-body">
      <PersistenceAlert :notice="notice" @retry="$emit('retry')" @refresh="$emit('refresh')" />
      <StateBanner
        v-for="note in notes"
        :key="note.text"
        :severity="note.tone === 'info' ? 'info' : 'warning'"
        :role="note.role || undefined"
        class="build-settings-note"
        >{{ note.text }}</StateBanner
      >
      <StateBanner v-if="frozen" severity="info" role="note" class="build-settings-note">
        These CMake options are frozen to <strong>{{ frozenVersion }}</strong
        >. Saving does not change global Build settings.
      </StateBanner>

      <div class="page-form page-form--single">
        <template v-if="!frozen">
          <div v-for="field in fields" :key="field.key" class="page-form__field">
            <label class="page-form__label" :for="fieldId(field.key)">
              {{ field.label }}
              <template v-if="field.optional">(optional)</template>
            </label>
            <InputText
              :id="fieldId(field.key)"
              :model-value="form[field.key] || ''"
              :placeholder="field.placeholder || ''"
              class="w-full"
              @update:model-value="setField(field.key, $event)"
            />
            <p v-if="field.hint" class="page-form__hint">{{ field.hint }}</p>
          </div>
        </template>

        <div v-if="showToolkit" class="page-form__field">
          <label class="page-form__label" :for="toolkitFieldId">CUDA toolkit</label>
          <Select
            :input-id="toolkitFieldId"
            :model-value="config.cuda_toolkit || ''"
            :options="toolkitOptions"
            option-label="label"
            option-value="value"
            placeholder="Active Studio toolkit"
            class="w-full"
            data-toolkit-select
            @update:model-value="setOption('cuda_toolkit', $event)"
          />
          <p class="page-form__hint">
            Pin this engine's source build to an installed toolkit. Active uses the Studio-managed
            toolkit; a missing bound toolkit fails the build with a clear error.
          </p>
        </div>

        <div class="page-form__field">
          <label class="page-form__label" :for="buildTypeFieldId">{{ buildTypeLabel }}</label>
          <Select
            :input-id="buildTypeFieldId"
            :model-value="config[buildTypeKey]"
            :options="buildTypes"
            option-label="label"
            option-value="value"
            :placeholder="buildTypePlaceholder"
            class="w-full"
            @update:model-value="setOption(buildTypeKey, $event)"
          />
        </div>
      </div>

      <LoadingState v-if="optionsLoading" inline small message="Loading build options…" />

      <template v-for="category in categories" :key="category.id">
        <div v-if="category.layout === 'grid'" class="page-form__field">
          <StateBanner v-if="category.note" severity="info" role="note" class="build-settings-note">
            {{ category.note }}
          </StateBanner>
          <span class="page-form__label">{{ category.label }}</span>
          <div class="toggle-grid">
            <div v-for="opt in primaryOptions(category)" :key="opt.key" class="toggle-row">
              <ToggleSwitch
                :model-value="config[opt.key]"
                :disabled="Boolean(opt.disabled)"
                @update:model-value="setOption(opt.key, $event)"
              />
              <div class="toggle-copy">
                <span class="opt-label">{{ opt.label }}</span>
                <small class="page-form__hint">{{ opt.desc }}</small>
                <small v-if="opt.disabledReason" class="opt-block-reason">
                  {{ opt.disabledReason }}
                </small>
              </div>
            </div>
          </div>
          <details v-if="secondaryOptions(category).length" class="build-options-details">
            <summary class="page-form__label">
              {{ category.moreLabel || 'More backends' }}
            </summary>
            <div class="toggle-grid">
              <div v-for="opt in secondaryOptions(category)" :key="opt.key" class="toggle-row">
                <ToggleSwitch
                  :model-value="config[opt.key]"
                  :disabled="Boolean(opt.disabled)"
                  @update:model-value="setOption(opt.key, $event)"
                />
                <div class="toggle-copy">
                  <span class="opt-label">{{ opt.label }}</span>
                  <small class="page-form__hint">{{ opt.desc }}</small>
                  <small v-if="opt.disabledReason" class="opt-block-reason">
                    {{ opt.disabledReason }}
                  </small>
                </div>
              </div>
            </div>
          </details>
        </div>

        <details v-else class="build-options-details">
          <summary class="page-form__label">
            {{ category.label }}
            <span class="build-advanced-hint">{{ category.hint || 'advanced' }}</span>
          </summary>
          <StateBanner v-if="category.note" severity="info" role="note" class="build-settings-note">
            {{ category.note }}
          </StateBanner>
          <div class="toggle-grid">
            <template v-for="opt in category.options || []" :key="opt.key">
              <div v-if="opt.type === 'bool'" class="toggle-row">
                <ToggleSwitch
                  :model-value="config[opt.key]"
                  :disabled="Boolean(opt.disabled)"
                  @update:model-value="setOption(opt.key, $event)"
                />
                <div class="toggle-copy">
                  <span class="opt-label">{{ opt.label }}</span>
                  <small class="page-form__hint">{{ opt.desc }}</small>
                  <small v-if="opt.disabledReason" class="opt-block-reason">
                    {{ opt.disabledReason }}
                  </small>
                </div>
              </div>
              <div v-else-if="opt.type === 'int'" class="opt-string-field">
                <span class="opt-label">{{ opt.label }}</span>
                <small class="page-form__hint">{{ opt.desc }}</small>
                <InputNumber
                  :model-value="config[opt.key]"
                  :min="opt.min ?? 0"
                  :max="opt.max ?? 256"
                  :disabled="Boolean(opt.disabled)"
                  class="w-full mt-1"
                  @update:model-value="setOption(opt.key, $event)"
                />
              </div>
              <div v-else class="opt-string-field">
                <span class="opt-label">{{ opt.label }}</span>
                <small class="page-form__hint">{{ opt.desc }}</small>
                <Select
                  v-if="opt.type === 'enum'"
                  :model-value="config[opt.key]"
                  :options="opt.enum_values || []"
                  :disabled="Boolean(opt.disabled)"
                  class="w-full mt-1"
                  @update:model-value="setOption(opt.key, $event)"
                />
                <InputText
                  v-else
                  :model-value="config[opt.key]"
                  :placeholder="opt.placeholder || ''"
                  :disabled="Boolean(opt.disabled)"
                  class="w-full mt-1"
                  @update:model-value="setOption(opt.key, $event)"
                />
              </div>
            </template>
          </div>
        </details>
      </template>

      <details class="build-options-details">
        <summary class="page-form__label">
          Custom flags
          <span class="build-advanced-hint">advanced</span>
        </summary>
        <div class="page-form__field">
          <label class="page-form__label" :for="`${idSuffix}-cmake-args`">
            Custom CMake args (optional)
          </label>
          <InputText
            :id="`${idSuffix}-cmake-args`"
            :model-value="config.custom_cmake_args"
            placeholder="e.g. -DFOO=ON -DBAR=OFF"
            class="w-full"
            @update:model-value="setOption('custom_cmake_args', $event)"
          />
        </div>
        <div class="page-form__field">
          <span class="page-form__label">CFLAGS / CXXFLAGS (optional)</span>
          <div class="flags-row">
            <InputText
              :model-value="config.cflags"
              placeholder="CFLAGS"
              :aria-label="`CFLAGS`"
              class="flex-1"
              @update:model-value="setOption('cflags', $event)"
            />
            <InputText
              :model-value="config.cxxflags"
              placeholder="CXXFLAGS"
              :aria-label="`CXXFLAGS`"
              class="flex-1"
              @update:model-value="setOption('cxxflags', $event)"
            />
          </div>
        </div>
      </details>
    </div>
    <template #footer>
      <div class="dialog-actions">
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
      </div>
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
import { computed, useId } from 'vue'
import Button from 'primevue/button'
import Dialog from 'primevue/dialog'
import InputText from 'primevue/inputtext'
import InputNumber from 'primevue/inputnumber'
import Select from 'primevue/select'
import ToggleSwitch from 'primevue/toggleswitch'
import LoadingState from '@/components/common/LoadingState.vue'
import PersistenceAlert from '@/components/common/PersistenceAlert.vue'
import StateBanner from '@/components/common/StateBanner.vue'

const props = defineProps({
  visible: { type: Boolean, default: false },
  header: { type: String, required: true },
  form: { type: Object, required: true },
  configKey: { type: String, default: 'build_config' },
  buildTypeKey: { type: String, default: 'build_type' },
  buildTypeLabel: { type: String, default: 'Build type' },
  buildTypes: { type: Array, default: () => [] },
  buildTypePlaceholder: { type: String, default: '' },
  showToolkit: { type: Boolean, default: false },
  toolkitOptions: { type: Array, default: () => [] },
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

const idSuffix = `engine-build-${useId()}`
const toolkitFieldId = `${idSuffix}-toolkit`
const buildTypeFieldId = `${idSuffix}-build-type`

function fieldId(key) {
  return `${idSuffix}-field-${key}`
}

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
.build-settings-note {
  margin-bottom: 0;
  font-size: var(--text-sm);
}

.build-settings-body {
  max-height: min(70vh, 640px);
  overflow-y: auto;
  padding-right: 0.25rem;
}

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

.toggle-copy {
  display: flex;
  flex-direction: column;
  gap: 0.1rem;
  min-width: 0;
}

.opt-label {
  display: block;
  font-size: var(--text-sm);
  font-weight: 500;
}

.opt-block-reason {
  display: block;
  margin-top: 0.15rem;
  color: var(--status-warning);
  font-size: var(--text-xs);
}

.opt-string-field {
  display: flex;
  flex-direction: column;
  gap: 0.15rem;
  padding: 0.15rem 0;
}

.build-options-details {
  margin-bottom: 0.65rem;
  padding: 0.35rem 0.6rem 0.5rem;
  border: 1px solid var(--border-primary);
  border-radius: var(--radius-sm);
  background: var(--bg-tertiary);
}

.build-options-details > summary {
  margin-bottom: 0.15rem;
  list-style: none;
  cursor: pointer;
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
  color: var(--text-secondary);
  font-size: var(--text-xs);
  font-weight: 500;
  letter-spacing: 0.03em;
}

.flags-row {
  display: flex;
  gap: 0.5rem;
}

@media (max-width: 768px) {
  .flags-row {
    flex-wrap: wrap;
  }
}
</style>
