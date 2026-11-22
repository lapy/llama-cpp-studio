<template>
  <div class="param-groups">
    <PageToolbar plain label="Runtime parameter filters">
      <template #search>
        <span class="p-input-icon-left config-search-wrap">
          <i class="pi pi-search" aria-hidden="true" />
          <InputText
            :model-value="searchQuery"
            type="search"
            placeholder="Add a parameter…"
            class="config-search-input"
            aria-label="Search parameters to add"
            @update:model-value="emit('update:searchQuery', $event)"
          />
        </span>
        <Button
          v-if="searchQuery"
          icon="pi pi-times"
          text
          rounded
          severity="secondary"
          v-tooltip.top="'Clear search'"
          aria-label="Clear search"
          @click="emit('update:searchQuery', '')"
        />
      </template>
      <div class="param-filter" role="group" aria-label="Parameter filter">
        <button
          v-for="option in FILTER_OPTIONS"
          :key="option.value"
          type="button"
          class="param-filter__option"
          :class="{ 'param-filter__option--active': filterMode === option.value }"
          :aria-pressed="filterMode === option.value"
          @click="filterMode = option.value"
        >
          {{ option.label }}
        </button>
      </div>
      <div class="toggle-field">
        <ToggleSwitch
          :model-value="hideUnsupported"
          input-id="toggle-hide-unsupported"
          @update:model-value="emit('update:hideUnsupported', $event)"
        />
        <label for="toggle-hide-unsupported">Hide unsupported</label>
      </div>
      <template #summary>
        <span class="param-groups__summary" aria-live="polite">{{ totalChanged }} changed</span>
      </template>
      <template #actions>
        <Button
          v-if="autoFitOverrides.length"
          label="Restore auto settings"
          icon="pi pi-undo"
          size="small"
          severity="secondary"
          outlined
          type="button"
          data-auto-fit-action
          @click="emit('restore-auto-fit')"
        />
        <Button
          label="Expand all"
          size="small"
          severity="secondary"
          text
          type="button"
          @click="setAllGroupsOpen(true)"
        />
        <Button
          label="Collapse all"
          size="small"
          severity="secondary"
          text
          type="button"
          @click="setAllGroupsOpen(false)"
        />
      </template>
    </PageToolbar>

    <div v-if="addableMatches.length" class="param-tag-cloud-wrap">
      <h3 class="page-subheading">Add parameter</h3>
      <div class="param-tag-cloud" role="list">
        <button
          v-for="param in addableMatches"
          :key="param.key"
          type="button"
          class="param-search-tag"
          role="listitem"
          @click="addParam(param)"
        >
          <span class="param-search-tag__label">{{ param.label }}</span>
          <code class="param-search-tag__key">{{ param.key }}</code>
        </button>
      </div>
    </div>

    <div v-if="visibleGroups.length" class="param-groups__list">
      <section
        v-for="group in visibleGroups"
        :key="group.id"
        class="param-group"
        :class="{ 'param-group--open': isGroupOpen(group.id) }"
        :data-group="group.id"
      >
        <h3 class="param-group__heading">
          <button
            type="button"
            class="param-group__toggle"
            :aria-expanded="isGroupOpen(group.id) ? 'true' : 'false'"
            :aria-controls="groupBodyId(group.id)"
            @click="toggleGroup(group.id)"
          >
            <i :class="[group.icon, 'param-group__icon']" aria-hidden="true" />
            <span :id="groupTitleId(group.id)" class="param-group__title">{{ group.title }}</span>
            <span class="param-group__badge">
              {{ countLabel(group.count, 'parameter') }} · {{ group.modifiedCount }} changed
            </span>
            <i class="pi pi-chevron-down param-group__chevron" aria-hidden="true" />
          </button>
          <span :id="groupDescriptionId(group.id)" class="param-group__description">
            {{ group.description }}
          </span>
        </h3>
        <div
          v-show="isGroupOpen(group.id)"
          :id="groupBodyId(group.id)"
          class="param-group__body"
          role="region"
          :aria-labelledby="groupTitleId(group.id)"
          :aria-describedby="groupDescriptionId(group.id)"
        >
          <div class="params-grid">
            <div
              v-for="param in group.params"
              :key="`${group.id}-${param.key}`"
              class="param-field"
              :class="{ 'param-field--unsupported': param.supported === false }"
            >
              <div class="param-field__head">
                <label :for="inputId(param)" class="param-field__label">
                  <span class="param-field__name">
                    {{ param.label }}
                    <Tag
                      v-if="param.supported === false"
                      value="Not in this build"
                      severity="secondary"
                      class="param-supported-tag"
                    />
                    <span
                      class="pi pi-info-circle param-info"
                      role="img"
                      tabindex="0"
                      :aria-label="`About ${param.label}`"
                      v-tooltip.top="paramDescriptionTooltip(param)"
                    />
                  </span>
                  <code class="param-key-hint">{{ param.key }}</code>
                </label>
                <div class="param-field__actions">
                  <Tag
                    class="param-field__state"
                    :value="isModified(param) ? 'Override' : 'Default'"
                    :severity="isModified(param) ? 'info' : 'secondary'"
                  />
                  <Button
                    type="button"
                    icon="pi pi-times"
                    text
                    rounded
                    severity="secondary"
                    class="param-remove-btn"
                    aria-label="Remove parameter (reset to default)"
                    v-tooltip.top="'Remove from pane (reset to default)'"
                    @click="emit('remove:param', param.key)"
                  />
                </div>
              </div>

              <template
                v-if="
                  param.type === 'int' && (param.key === 'ctx_size' || param.key === 'session_len')
                "
              >
                <div class="param-slider-row">
                  <Slider
                    :model-value="valueFor(param)"
                    :min="512"
                    :max="maxContextSuggestion || 131072"
                    :step="256"
                    class="param-slider"
                    :disabled="param.supported === false"
                    @update:model-value="(value) => emit('update:param', param.key, value)"
                  />
                  <span v-if="maxContextSuggestion" class="param-hint">
                    Suggested max: {{ maxContextSuggestion.toLocaleString() }} tokens
                  </span>
                </div>
                <InputNumber
                  :input-id="inputId(param)"
                  :model-value="valueFor(param)"
                  :placeholder="String(param.default ?? '')"
                  class="param-input"
                  :disabled="param.supported === false"
                  :pt="describedByPt(param)"
                  @update:model-value="(value) => emit('update:param', param.key, value)"
                />
              </template>
              <template v-else-if="param.type === 'int' && param.key === 'n_gpu_layers'">
                <div class="param-slider-row">
                  <Slider
                    :model-value="valueFor(param)"
                    :min="0"
                    :max="layerCountSuggestion || 128"
                    :step="1"
                    class="param-slider"
                    :disabled="param.supported === false"
                    @update:model-value="(value) => emit('update:param', param.key, value)"
                  />
                  <span v-if="layerCountSuggestion" class="param-hint">
                    Detected layers: {{ layerCountSuggestion }}
                  </span>
                </div>
                <InputNumber
                  :input-id="inputId(param)"
                  :model-value="valueFor(param)"
                  :placeholder="String(param.default ?? '')"
                  class="param-input"
                  :disabled="param.supported === false"
                  :pt="describedByPt(param)"
                  @update:model-value="(value) => emit('update:param', param.key, value)"
                />
              </template>
              <Select
                v-else-if="param.value_kind === 'flag' && param.negative_flag"
                :input-id="inputId(param)"
                :model-value="valueFor(param)"
                :options="triStateOptions"
                optionLabel="label"
                optionValue="value"
                placeholder="Default"
                showClear
                class="param-input"
                :disabled="param.supported === false"
                @update:model-value="(value) => emit('update:param', param.key, value)"
              />
              <InputTags
                v-else-if="['repeatable', 'space_list'].includes(param.value_kind)"
                :input-id="inputId(param)"
                :model-value="valueFor(param)"
                delimiter=","
                class="param-input"
                :disabled="param.supported === false"
                @update:model-value="(value) => emit('update:param', param.key, value)"
              />
              <MultiSelect
                v-else-if="isDelimitedEnum(param)"
                :input-id="inputId(param)"
                :model-value="valueFor(param)"
                :options="param.options || []"
                optionLabel="label"
                optionValue="value"
                display="chip"
                :placeholder="csvEnumPlaceholder(param)"
                class="param-input w-full"
                :disabled="param.supported === false"
                @update:model-value="(value) => emit('update:param', param.key, value)"
              />
              <Select
                v-else-if="param.options && param.options.length"
                :input-id="inputId(param)"
                :model-value="valueFor(param)"
                :options="param.options"
                optionLabel="label"
                optionValue="value"
                :placeholder="param.default != null ? String(param.default) : ''"
                :showClear="param.required !== true"
                class="param-input"
                :disabled="param.supported === false"
                @update:model-value="(value) => emit('update:param', param.key, value)"
              />
              <InputNumber
                v-else-if="param.type === 'int'"
                :input-id="inputId(param)"
                :model-value="valueFor(param)"
                :min="param.min"
                :max="param.max"
                :placeholder="String(param.default ?? '')"
                class="param-input"
                :disabled="param.supported === false"
                :pt="describedByPt(param)"
                @update:model-value="(value) => emit('update:param', param.key, value)"
              />
              <InputNumber
                v-else-if="param.type === 'float'"
                :input-id="inputId(param)"
                :model-value="valueFor(param)"
                :min="param.min"
                :max="param.max"
                :minFractionDigits="1"
                :maxFractionDigits="4"
                :placeholder="String(param.default ?? '')"
                class="param-input"
                :disabled="param.supported === false"
                :pt="describedByPt(param)"
                @update:model-value="(value) => emit('update:param', param.key, value)"
              />
              <ToggleSwitch
                v-else-if="param.type === 'bool'"
                :input-id="inputId(param)"
                :model-value="valueFor(param)"
                :disabled="param.supported === false"
                @update:model-value="(value) => emit('update:param', param.key, value)"
              />
              <Textarea
                v-else-if="param.type === 'json'"
                :id="inputId(param)"
                :model-value="jsonDisplay(valueFor(param))"
                rows="4"
                class="w-full textarea-cli param-input"
                :placeholder="jsonPlaceholder(param)"
                :disabled="param.supported === false"
                autoResize
                @update:model-value="(value) => emit('update:json', param.key, value)"
              />
              <InputText
                v-else
                :id="inputId(param)"
                :model-value="valueFor(param)"
                :placeholder="param.default != null ? String(param.default) : ''"
                class="param-input"
                :disabled="param.supported === false"
                @update:model-value="(value) => emit('update:param', param.key, value)"
              />
              <small :id="helpId(param)" class="param-default-help">{{
                parameterHelp(param)
              }}</small>
              <button
                v-if="isModified(param)"
                class="param-reset"
                type="button"
                @click="emit('reset:param', param)"
              >
                Use engine default
              </button>
            </div>
          </div>
        </div>
      </section>
    </div>

    <Message
      v-else
      severity="secondary"
      :closable="false"
      class="config-scan-message param-groups__empty"
    >
      <div class="param-groups__empty-body">
        <span>{{ emptyMessage }}</span>
        <div class="param-groups__empty-actions">
          <Button
            v-if="searchQuery"
            label="Clear search"
            size="small"
            severity="secondary"
            outlined
            type="button"
            @click="emit('update:searchQuery', '')"
          />
          <Button
            v-if="filterMode === 'changed'"
            label="Show all"
            size="small"
            severity="secondary"
            outlined
            type="button"
            @click="filterMode = 'all'"
          />
          <Button
            v-if="hideUnsupported"
            label="Show unsupported"
            size="small"
            severity="secondary"
            outlined
            type="button"
            @click="emit('update:hideUnsupported', false)"
          />
        </div>
      </div>
    </Message>
  </div>
</template>

<script setup>
import { computed, ref, watch } from 'vue'
import Button from 'primevue/button'
import InputText from 'primevue/inputtext'
import InputNumber from 'primevue/inputnumber'
import ToggleSwitch from 'primevue/toggleswitch'
import Select from 'primevue/select'
import InputTags from 'primevue/inputtags'
import Message from 'primevue/message'
import Textarea from 'primevue/textarea'
import Slider from 'primevue/slider'
import MultiSelect from 'primevue/multiselect'
import Tag from 'primevue/tag'
import PageToolbar from '@/components/common/PageToolbar.vue'
import { groupForParam, groupParameters, paramMatchesSearch } from '@/utils/modelParamGroups'

const FILTER_OPTIONS = Object.freeze([
  Object.freeze({ label: 'All', value: 'all' }),
  Object.freeze({ label: 'Changed', value: 'changed' }),
])

const triStateOptions = [
  { label: 'Default', value: null },
  { label: 'Enabled', value: true },
  { label: 'Disabled', value: false },
]

const props = defineProps({
  /** Params to render; each row needs `key` and ideally `sectionId`, label, and type. */
  params: { type: Array, default: () => [] },
  /** Catalog params not currently shown; used by the add-by-search tag cloud. */
  addableParams: { type: Array, default: () => [] },
  /** Current working config values, keyed by parameter key. */
  config: { type: Object, default: () => ({}) },
  searchQuery: { type: String, default: '' },
  hideUnsupported: { type: Boolean, default: false },
  /** Returns true when the config holds an explicit value for the param. */
  isModified: { type: Function, default: (param) => param?.modified === true },
  /** Shared enum detection so save/apply and the row control agree. */
  isDelimitedEnum: { type: Function, default: () => false },
  maxContextSuggestion: { type: Number, default: null },
  layerCountSuggestion: { type: Number, default: null },
  /** Basic params keep their legacy `basic-*` ids for links and e2e selectors. */
  pinnedKeys: { type: Array, default: () => [] },
  /** Settings that currently defeat the engine's auto-fit; empty hides the action. */
  autoFitOverrides: { type: Array, default: () => [] },
})

const emit = defineEmits([
  'update:param',
  'update:searchQuery',
  'update:hideUnsupported',
  'update:json',
  'reset:param',
  'remove:param',
  'add:param',
  'restore-auto-fit',
])

const filterMode = ref('all')
const openGroups = ref({})
const lastModifiedCounts = new Map()

const searchQuery = computed(() => String(props.searchQuery || ''))
const searchActive = computed(() => Boolean(searchQuery.value.trim()))

const allGroups = computed(() =>
  groupParameters(props.params, { isModified: (param) => props.isModified(param) }),
)

const totalChanged = computed(() =>
  allGroups.value.reduce((total, group) => total + group.modifiedCount, 0),
)

const visibleGroups = computed(() => {
  const groups = []
  for (const group of allGroups.value) {
    let params = group.params
    if (props.hideUnsupported) {
      params = params.filter((param) => param.supported !== false)
    }
    if (searchActive.value) {
      params = params.filter((param) => paramMatchesSearch(param, searchQuery.value))
    }
    if (filterMode.value === 'changed') {
      params = params.filter((param) => props.isModified(param))
    }
    if (!params.length) continue
    groups.push({
      ...group,
      params,
      count: params.length,
      modifiedCount: params.filter((param) => props.isModified(param)).length,
    })
  }
  return groups
})

const addableMatches = computed(() => {
  if (!searchActive.value) return []
  const out = []
  for (const param of props.addableParams || []) {
    if (!paramMatchesSearch(param, searchQuery.value, { hideUnsupported: props.hideUnsupported })) {
      continue
    }
    out.push(param)
    if (out.length >= 100) break
  }
  return out
})

const emptyMessage = computed(() => {
  if (!allGroups.value.length) {
    return 'Search above to add parameters. Saved values that differ from the engine default appear here.'
  }
  if (searchActive.value) {
    return `No parameters match “${searchQuery.value.trim()}”. Clear the search to see every parameter.`
  }
  if (filterMode.value === 'changed') {
    return 'No parameters are changed from their engine default yet. Show all parameters to keep browsing.'
  }
  return 'No parameters are supported by this engine build. Turn off “Hide unsupported” to see them.'
})

// First render: open the two default groups plus any group holding an explicit
// override. Later, a group that gains its first override opens automatically so
// an added parameter never lands in a collapsed section.
watch(
  allGroups,
  (groups) => {
    for (const group of groups) {
      const previouslyModified = lastModifiedCounts.get(group.id)
      if (!(group.id in openGroups.value)) {
        openGroups.value[group.id] = group.defaultOpen || group.modifiedCount > 0
      } else if (previouslyModified === 0 && group.modifiedCount > 0) {
        openGroups.value[group.id] = true
      }
      lastModifiedCounts.set(group.id, group.modifiedCount)
    }
  },
  { immediate: true },
)

// A search only renders matching rows; expand those groups so the matches are
// visible without another click.
watch(searchQuery, (query) => {
  if (!query.trim()) return
  for (const group of visibleGroups.value) openGroups.value[group.id] = true
})

function countLabel(count, singular) {
  return `${count} ${singular}${count === 1 ? '' : 's'}`
}

function isGroupOpen(groupId) {
  return Boolean(openGroups.value[groupId])
}

function toggleGroup(groupId) {
  openGroups.value[groupId] = !openGroups.value[groupId]
}

function setAllGroupsOpen(open) {
  for (const group of allGroups.value) openGroups.value[group.id] = open
}

/** Adding a param from search opens its group before the row appears. */
function addParam(param) {
  openGroups.value[groupForParam(param).id] = true
  emit('add:param', param.key)
}

function groupTitleId(groupId) {
  return `param-group-title-${groupId}`
}

function groupDescriptionId(groupId) {
  return `param-group-description-${groupId}`
}

function groupBodyId(groupId) {
  return `param-group-${groupId}`
}

function isPinned(param) {
  return props.pinnedKeys.includes(param.key)
}

function inputId(param) {
  return isPinned(param) ? `basic-${param.key}` : `p-${param.sectionId || 'other'}-${param.key}`
}

function helpId(param) {
  return isPinned(param) ? `basic-help-${param.key}` : `${inputId(param)}-help`
}

function describedByPt(param) {
  return { pcInputText: { root: { 'aria-describedby': helpId(param) } } }
}

function valueFor(param) {
  return props.config?.[param.key]
}

function parameterHelp(param) {
  const value =
    param.default == null || param.default === ''
      ? 'chosen by the engine'
      : JSON.stringify(param.default)
  return `Engine default: ${value}`
}

function paramDescriptionTooltip(param) {
  const parts = [param.description].filter(Boolean)
  if (param.primary_flag) parts.push(`Primary flag: ${param.primary_flag}`)
  if (param.negative_flag) parts.push(`Negative flag: ${param.negative_flag}`)
  else if (param.flags?.length) parts.push(`CLI: ${param.flags.join(', ')}`)
  return parts.join('\n\n') || param.label || param.key
}

function csvEnumPlaceholder(param) {
  if (Array.isArray(param.default) && param.default.length) return param.default.join(', ')
  if (param.default != null && param.default !== '') return String(param.default)
  return 'Select one or more'
}

function jsonDisplay(value) {
  if (value == null || value === '') return ''
  if (typeof value === 'string') {
    const trimmed = value.trim()
    if (!trimmed) return ''
    try {
      return JSON.stringify(JSON.parse(trimmed), null, 2)
    } catch {
      return value
    }
  }
  if (typeof value === 'object') {
    try {
      return JSON.stringify(value, null, 2)
    } catch {
      return ''
    }
  }
  return String(value)
}

function jsonPlaceholder(param) {
  if (param?.default != null && typeof param.default === 'object') {
    try {
      return JSON.stringify(param.default, null, 2)
    } catch {
      return String(param.default)
    }
  }
  if (param?.default != null) return String(param.default)
  return '{"key": "value"}'
}
</script>

<style scoped>
.param-groups {
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
  min-width: 0;
}

.param-groups__summary {
  padding: 0.15rem 0.5rem;
  border-radius: 999px;
  background: var(--bg-surface);
  color: var(--text-secondary);
  font-size: var(--text-xs);
  font-weight: 600;
  white-space: nowrap;
}

.param-filter {
  display: inline-flex;
  flex-shrink: 0;
  padding: 0.15rem;
  border: 1px solid var(--border-primary);
  border-radius: var(--radius-md);
  background: var(--bg-surface);
}

.param-filter__option {
  border: 0;
  border-radius: calc(var(--radius-md) - 0.15rem);
  background: transparent;
  color: var(--text-secondary);
  padding: 0.2rem 0.6rem;
  font: inherit;
  font-size: 0.8125rem;
  cursor: pointer;
}

.param-filter__option:hover {
  color: var(--text-primary);
}

.param-filter__option--active {
  background: var(--accent-cyan-soft);
  color: var(--accent-primary);
  font-weight: 600;
}

.param-filter__option:focus-visible,
.param-group__toggle:focus-visible,
.param-reset:focus-visible,
.param-remove-btn:focus-visible {
  outline: 2px solid var(--accent-primary);
  outline-offset: 2px;
}

.param-tag-cloud-wrap {
  margin-top: 0.1rem;
}

.param-groups__list {
  display: flex;
  flex-direction: column;
}

.param-group {
  padding: 0.55rem 0;
  border-top: 1px solid var(--border-primary);
}

.param-group:first-child {
  border-top: 0;
  padding-top: 0.15rem;
}

.param-group__heading {
  display: flex;
  flex-direction: column;
  gap: 0.15rem;
  margin: 0;
  min-width: 0;
}

.param-group__toggle {
  display: grid;
  grid-template-columns: auto minmax(0, 1fr) auto auto;
  align-items: center;
  gap: 0.55rem;
  width: 100%;
  padding: 0.3rem 0.4rem;
  border: 0;
  border-radius: var(--radius-md, 0.5rem);
  background: transparent;
  color: var(--text-primary);
  font: inherit;
  text-align: left;
  cursor: pointer;
}

.param-group__toggle:hover {
  background: var(--bg-surface);
}

.param-group__icon {
  color: var(--accent-primary);
  font-size: 0.95rem;
}

.param-group__title {
  overflow: hidden;
  font-size: 0.9375rem;
  font-weight: 600;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.param-group__badge {
  flex-shrink: 0;
  color: var(--text-secondary);
  font-size: var(--text-xs);
  font-weight: 500;
}

.param-group__chevron {
  color: var(--text-secondary);
  font-size: 0.75rem;
  transition: transform 0.15s ease;
}

.param-group--open .param-group__chevron {
  transform: rotate(180deg);
}

.param-group__description {
  margin: 0 0 0 2.05rem;
  overflow: hidden;
  color: var(--text-secondary);
  font-size: var(--text-xs);
  line-height: 1.4;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.param-group__body {
  padding: 0.7rem 0.15rem 0.25rem;
}

.params-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(min(100%, 17rem), 1fr));
  gap: 0.75rem 1rem;
}

.param-field {
  display: flex;
  flex-direction: column;
  gap: 0.35rem;
  min-width: 0;
}

.param-field--unsupported {
  opacity: 0.88;
}

.param-field__head {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 0.5rem;
  margin-bottom: 0.35rem;
}

.param-field__label {
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto;
  align-items: center;
  column-gap: 0.5rem;
  row-gap: 0.15rem;
  flex: 1;
  min-width: 0;
  font-size: 0.875rem;
  font-weight: 500;
  color: var(--text-secondary);
}

.param-field__name {
  grid-column: 1;
  grid-row: 1;
  min-width: 0;
}

.param-field__label > .param-key-hint {
  grid-column: 1;
  grid-row: 2;
  margin-left: 0;
  justify-self: start;
}

.param-field__actions {
  display: flex;
  align-items: center;
  gap: 0.3rem;
  flex-shrink: 0;
}

.param-field__state {
  justify-self: end;
}

.param-supported-tag {
  margin-left: 0.35rem;
  vertical-align: middle;
  font-size: var(--text-xs) !important;
}

.param-remove-btn {
  flex-shrink: 0;
  margin-top: -0.15rem;
}

.param-input {
  width: 100%;
}

.param-key-hint {
  margin-left: 0;
  padding: 0.1rem 0.35rem;
  border-radius: 0.25rem;
  background: var(--bg-surface);
  color: var(--text-secondary);
  font-size: var(--text-xs);
  font-weight: 400;
  vertical-align: middle;
}

.param-info {
  font-size: var(--text-xs);
  cursor: help;
  opacity: 0.6;
}

.param-slider-row {
  display: flex;
  flex-direction: column;
  gap: 0.25rem;
  margin-bottom: 0.25rem;
}

.param-slider {
  width: 100%;
  max-width: 15rem;
}

.param-slider :deep(.p-slider) {
  height: 0.5rem;
}

.param-slider :deep(.p-slider-handle) {
  width: 1rem;
  height: 1rem;
  top: 50%;
  margin-top: -0.5rem;
}

.param-hint {
  color: var(--text-secondary);
  font-size: var(--text-xs);
}

.param-default-help {
  color: var(--text-secondary);
  font-size: var(--text-xs);
  line-height: 1.6;
}

.param-reset {
  justify-self: start;
  width: fit-content;
  padding: 0.25rem 0;
  border: 0;
  background: transparent;
  color: var(--accent-primary);
  font-size: var(--text-xs);
  cursor: pointer;
}

.param-groups__empty-body {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 0.5rem;
}

.param-groups__empty-actions {
  display: flex;
  flex-wrap: wrap;
  gap: 0.35rem;
}

@media (max-width: 768px) {
  .param-group__badge {
    display: none;
  }

  .param-group__toggle {
    grid-template-columns: auto minmax(0, 1fr) auto;
  }
}
</style>
