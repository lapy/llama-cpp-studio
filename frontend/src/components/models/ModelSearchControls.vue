<template>
  <section class="search-controls" aria-label="Catalog search">
    <div class="search-bar compact-toolbar">
      <div class="search-input-wrap">
        <i class="pi pi-search search-icon" />
        <label class="sr-only" for="catalog-search-query">Search models</label>
        <InputText
          id="catalog-search-query"
          :model-value="query"
          placeholder="Search models and audio packages…"
          class="search-input"
          @update:model-value="$emit('update:query', $event)"
          @keyup.enter="$emit('search')"
        />
        <Button
          v-if="query"
          icon="pi pi-times"
          text
          severity="secondary"
          class="clear-btn"
          aria-label="Clear search"
          @click="$emit('clear')"
        />
      </div>

      <Select
        :model-value="taskFilter"
        :options="taskFilterOptions"
        optionLabel="label"
        optionValue="value"
        placeholder="Any task"
        showClear
        class="catalog-filter catalog-filter--task"
        aria-label="Task"
        @update:model-value="$emit('update:taskFilter', $event)"
        @change="$emit('search')"
      />
      <Button
        label="Search"
        icon="pi pi-search"
        size="small"
        :loading="searching"
        @click="$emit('search')"
      />
      <Button
        :label="
          filtersOpen
            ? 'Hide filters'
            : `Filters${secondaryFilterCount ? ` (${secondaryFilterCount})` : ''}`
        "
        icon="pi pi-filter"
        size="small"
        severity="secondary"
        outlined
        :aria-expanded="filtersOpen ? 'true' : 'false'"
        aria-controls="catalog-secondary-filters"
        @click="$emit('update:filtersOpen', !filtersOpen)"
      />
      <Button
        v-if="secondaryFilterCount"
        label="Reset"
        size="small"
        severity="secondary"
        text
        @click="$emit('reset')"
      />
      <Button
        label="Import audio bundle"
        icon="pi pi-folder-open"
        size="small"
        severity="secondary"
        text
        class="catalog-filters__import"
        @click="$emit('import-audio')"
      />
    </div>

    <div v-if="activeFilterChips.length" class="search-chips" aria-label="Active filters">
      <button
        v-for="chip in activeFilterChips"
        :key="chip.key"
        type="button"
        class="search-chip"
        @click="chip.clear()"
      >
        {{ chip.label }}
        <span aria-hidden="true">×</span>
      </button>
    </div>

    <div
      v-show="filtersOpen"
      id="catalog-secondary-filters"
      class="catalog-filters catalog-filters--secondary compact-toolbar"
    >
      <Select
        v-if="showFormatSelect"
        :model-value="searchFormat"
        :options="formatOptions"
        optionLabel="label"
        optionValue="value"
        class="format-select"
        aria-label="Format"
        @update:model-value="$emit('update:searchFormat', $event)"
        @change="$emit('format-change')"
      />
      <Select
        :model-value="engineFilter"
        :options="engineFilterOptions"
        optionLabel="label"
        optionValue="value"
        placeholder="Any engine"
        showClear
        class="catalog-filter"
        aria-label="Engine"
        @update:model-value="$emit('update:engineFilter', $event)"
        @change="$emit('engine-change')"
      />
      <Select
        :model-value="inputModalityFilter"
        :options="inputModalityOptions"
        placeholder="Any input"
        showClear
        class="catalog-filter"
        aria-label="Input modality"
        @update:model-value="$emit('update:inputModalityFilter', $event)"
        @change="$emit('search')"
      />
      <Select
        :model-value="outputModalityFilter"
        :options="outputModalityOptions"
        placeholder="Any output"
        showClear
        class="catalog-filter"
        aria-label="Output modality"
        @update:model-value="$emit('update:outputModalityFilter', $event)"
        @change="$emit('search')"
      />
      <Select
        :model-value="providerFilter"
        :options="providerOptions"
        optionLabel="label"
        optionValue="value"
        placeholder="Any source"
        showClear
        class="catalog-filter"
        aria-label="Source"
        @update:model-value="$emit('update:providerFilter', $event)"
        @change="$emit('search')"
      />
      <Select
        v-if="installMethodOptions.length"
        :model-value="installMethodFilter"
        :options="installMethodOptions"
        optionLabel="label"
        optionValue="value"
        placeholder="Any install method"
        showClear
        class="catalog-filter"
        aria-label="Install method"
        @update:model-value="$emit('update:installMethodFilter', $event)"
        @change="$emit('search')"
      />
    </div>
  </section>
</template>

<script setup>
import Button from 'primevue/button'
import InputText from 'primevue/inputtext'
import Select from 'primevue/select'

defineProps({
  query: { type: String, default: '' },
  taskFilter: { type: String, default: null },
  filtersOpen: { type: Boolean, default: false },
  searchFormat: { type: String, default: null },
  engineFilter: { type: String, default: null },
  inputModalityFilter: { type: String, default: null },
  outputModalityFilter: { type: String, default: null },
  providerFilter: { type: String, default: null },
  installMethodFilter: { type: String, default: null },
  searching: { type: Boolean, default: false },
  secondaryFilterCount: { type: Number, default: 0 },
  activeFilterChips: { type: Array, default: () => [] },
  showFormatSelect: { type: Boolean, default: false },
  taskFilterOptions: { type: Array, default: () => [] },
  formatOptions: { type: Array, default: () => [] },
  engineFilterOptions: { type: Array, default: () => [] },
  inputModalityOptions: { type: Array, default: () => [] },
  outputModalityOptions: { type: Array, default: () => [] },
  providerOptions: { type: Array, default: () => [] },
  installMethodOptions: { type: Array, default: () => [] },
})

defineEmits([
  'update:query',
  'update:taskFilter',
  'update:filtersOpen',
  'update:searchFormat',
  'update:engineFilter',
  'update:inputModalityFilter',
  'update:outputModalityFilter',
  'update:providerFilter',
  'update:installMethodFilter',
  'search',
  'clear',
  'reset',
  'import-audio',
  'format-change',
  'engine-change',
])
</script>

<style scoped>
.search-controls {
  display: flex;
  flex-direction: column;
  gap: 0.4rem;
}
.search-bar {
  display: flex;
  gap: 0.5rem;
  align-items: center;
  flex-wrap: wrap;
}
.search-input-wrap {
  flex: 1;
  min-width: 200px;
  position: relative;
}
.search-icon {
  position: absolute;
  left: 0.75rem;
  top: 50%;
  transform: translateY(-50%);
  color: var(--text-secondary, #9ca3af);
  pointer-events: none;
}
.search-input {
  width: 100%;
  padding-left: 2.25rem !important;
}
.clear-btn {
  position: absolute;
  right: 0.25rem;
  top: 50%;
  transform: translateY(-50%);
}
.format-select {
  width: 140px;
}
.catalog-filters {
  display: flex;
  gap: 0.5rem;
  align-items: center;
  flex-wrap: wrap;
}
.catalog-filters--secondary {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(9.5rem, 1fr));
  gap: 0.35rem;
  padding-top: 0.35rem;
  border-top: 1px solid var(--border-primary);
}
.catalog-filters__import {
  margin-left: auto;
  white-space: nowrap;
}
.search-chips {
  display: flex;
  flex-wrap: wrap;
  gap: 0.35rem;
}
.search-chip {
  display: inline-flex;
  align-items: center;
  gap: 0.35rem;
  min-height: 1.75rem;
  padding: 0.1rem 0.55rem;
  border: 1px solid var(--border-primary);
  border-radius: 999px;
  background: var(--bg-tertiary);
  color: var(--text-primary);
  font: inherit;
  font-size: 0.8rem;
  cursor: pointer;
}
.catalog-filter {
  min-width: 0;
  width: 100%;
}
.catalog-filter--task {
  flex: 0 1 9.5rem;
  width: 9.5rem;
}

@media (max-width: 768px) {
  .format-select {
    width: 100%;
  }
  .catalog-filters {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
  .catalog-filters .catalog-filter {
    min-width: 0;
    max-width: none;
    flex: none;
    width: 100%;
  }
  .catalog-filters__import {
    margin-left: 0;
    grid-column: 1 / -1;
    white-space: normal;
  }
}

/* The compact utility stacks fields on touch sizes; the search row keeps its
   single-line wrap layout because its controls are inline actions. */
@media (max-width: 640px), (pointer: coarse) {
  .search-bar {
    flex-direction: row;
    align-items: center;
  }
}

@media (max-width: 480px) {
  .search-bar {
    flex-direction: row;
    flex-wrap: wrap;
    align-items: center;
  }
  .search-input-wrap {
    flex: 1 1 100%;
    min-width: 0;
    width: 100%;
  }
  .catalog-filter--task {
    flex: 1 1 8rem;
    width: auto;
  }
}
</style>
