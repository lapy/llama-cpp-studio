<template>
  <div class="search-controls">
    <PageToolbar label="Model search" class="search-bar compact-toolbar">
      <template #search>
        <div class="search-input-wrap">
          <i class="pi pi-search search-icon" aria-hidden="true" />
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
      </template>

      <Select
        :model-value="taskFilter"
        :options="taskFilterOptions"
        option-label="label"
        option-value="value"
        placeholder="Any task"
        show-clear
        aria-label="Task"
        class="catalog-filter--task"
        @update:model-value="$emit('update:taskFilter', $event)"
        @change="$emit('search')"
      />

      <template #actions>
        <Button
          label="Search"
          icon="pi pi-search"
          size="small"
          :loading="searching"
          @click="$emit('search')"
        />
        <Button
          :label="filtersOpen ? 'Hide filters' : filtersLabel"
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
          label="Reset filters"
          size="small"
          severity="secondary"
          text
          @click="$emit('reset')"
        />
      </template>
    </PageToolbar>

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
      class="catalog-filters catalog-filters--secondary compact-fields"
      role="group"
      aria-label="Search filters"
    >
      <Select
        v-if="showFormatSelect"
        :model-value="searchFormat"
        :options="formatOptions"
        option-label="label"
        option-value="value"
        aria-label="Format"
        @update:model-value="$emit('update:searchFormat', $event)"
        @change="$emit('format-change')"
      />
      <Select
        :model-value="engineFilter"
        :options="engineFilterOptions"
        option-label="label"
        option-value="value"
        placeholder="Any engine"
        show-clear
        aria-label="Engine"
        @update:model-value="$emit('update:engineFilter', $event)"
        @change="$emit('engine-change')"
      />
      <Select
        :model-value="inputModalityFilter"
        :options="inputModalityOptions"
        placeholder="Any input"
        show-clear
        aria-label="Input modality"
        @update:model-value="$emit('update:inputModalityFilter', $event)"
        @change="$emit('search')"
      />
      <Select
        :model-value="outputModalityFilter"
        :options="outputModalityOptions"
        placeholder="Any output"
        show-clear
        aria-label="Output modality"
        @update:model-value="$emit('update:outputModalityFilter', $event)"
        @change="$emit('search')"
      />
      <Select
        :model-value="providerFilter"
        :options="providerOptions"
        option-label="label"
        option-value="value"
        placeholder="Any source"
        show-clear
        aria-label="Source"
        @update:model-value="$emit('update:providerFilter', $event)"
        @change="$emit('search')"
      />
      <Select
        v-if="installMethodOptions.length"
        :model-value="installMethodFilter"
        :options="installMethodOptions"
        option-label="label"
        option-value="value"
        placeholder="Any install method"
        show-clear
        aria-label="Install method"
        @update:model-value="$emit('update:installMethodFilter', $event)"
        @change="$emit('search')"
      />
    </div>
  </div>
</template>

<script setup>
import { computed } from 'vue'
import Button from 'primevue/button'
import InputText from 'primevue/inputtext'
import Select from 'primevue/select'
import PageToolbar from '@/components/common/PageToolbar.vue'

const props = defineProps({
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
  'format-change',
  'engine-change',
])

const filtersLabel = computed(() =>
  props.secondaryFilterCount ? `Filters (${props.secondaryFilterCount})` : 'Filters',
)
</script>

<style scoped>
.search-controls {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}

.search-input-wrap {
  position: relative;
  flex: 1;
  min-width: 0;
}

.search-icon {
  position: absolute;
  left: 0.75rem;
  top: 50%;
  transform: translateY(-50%);
  color: var(--text-secondary);
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

.catalog-filter--task {
  min-width: 0;
}

.catalog-filters {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(min(100%, 10rem), 1fr));
  gap: var(--toolbar-gap);
}
</style>
