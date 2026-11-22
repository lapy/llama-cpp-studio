<template>
  <PageToolbar label="Library filters" class="library-toolbar compact-toolbar">
    <template #search>
      <label class="sr-only" for="library-search">Search library</label>
      <InputText
        id="library-search"
        :model-value="query"
        type="search"
        class="w-full"
        placeholder="Search name or repository"
        @update:model-value="$emit('update:query', $event)"
      />
    </template>

    <Select
      :model-value="statusFilter"
      :options="statusOptions"
      option-label="label"
      option-value="value"
      input-id="library-status"
      aria-label="Status"
      class="library-toolbar__select"
      @update:model-value="$emit('update:statusFilter', $event)"
    />

    <Button
      :label="secondaryFilterCount ? `Filters (${secondaryFilterCount})` : 'Filters'"
      icon="pi pi-filter"
      size="small"
      severity="secondary"
      outlined
      :aria-expanded="filtersOpen ? 'true' : 'false'"
      aria-controls="library-secondary-filters"
      @click="$emit('update:filtersOpen', !filtersOpen)"
    />

    <div
      v-show="filtersOpen"
      id="library-secondary-filters"
      class="library-toolbar__secondary"
      role="group"
      aria-label="Library filters"
    >
      <Select
        :model-value="engineFilter"
        :options="engineOptions"
        option-label="label"
        option-value="value"
        input-id="library-engine"
        aria-label="Engine"
        @update:model-value="$emit('update:engineFilter', $event)"
      />
      <Select
        :model-value="taskFilter"
        :options="taskOptions"
        option-label="label"
        option-value="value"
        input-id="library-task"
        aria-label="Task"
        @update:model-value="$emit('update:taskFilter', $event)"
      />
      <Select
        :model-value="sortBy"
        :options="sortOptions"
        option-label="label"
        option-value="value"
        input-id="library-sort"
        aria-label="Sort"
        @update:model-value="$emit('update:sortBy', $event)"
      />
    </div>

    <LibrarySavedFilters :filters="presetFilters" @apply="$emit('apply-preset', $event)" />

    <template #actions>
      <div class="library-view" role="group" aria-label="Library layout">
        <Button
          label="Cards"
          size="small"
          text
          severity="secondary"
          class="library-view__button"
          :aria-pressed="viewMode === 'cards'"
          @click="$emit('change-view', 'cards')"
        />
        <Button
          label="List"
          size="small"
          text
          severity="secondary"
          class="library-view__button"
          :aria-pressed="viewMode === 'list'"
          @click="$emit('change-view', 'list')"
        />
      </div>
    </template>
  </PageToolbar>
</template>

<script setup>
import Button from 'primevue/button'
import InputText from 'primevue/inputtext'
import Select from 'primevue/select'
import PageToolbar from '@/components/common/PageToolbar.vue'
import LibrarySavedFilters from '@/components/models/LibrarySavedFilters.vue'

defineProps({
  query: { type: String, default: '' },
  statusFilter: { type: String, default: 'all' },
  engineFilter: { type: String, default: 'all' },
  taskFilter: { type: String, default: 'all' },
  sortBy: { type: String, default: 'name' },
  filtersOpen: { type: Boolean, default: false },
  viewMode: { type: String, default: 'cards' },
  secondaryFilterCount: { type: Number, default: 0 },
  engineOptions: { type: Array, default: () => [] },
  taskOptions: { type: Array, default: () => [] },
  presetFilters: { type: Object, default: () => ({}) },
})

defineEmits([
  'update:query',
  'update:statusFilter',
  'update:engineFilter',
  'update:taskFilter',
  'update:sortBy',
  'update:filtersOpen',
  'apply-preset',
  'change-view',
])

const statusOptions = [
  { label: 'All statuses', value: 'all' },
  { label: 'Running', value: 'running' },
  { label: 'Stopped', value: 'stopped' },
  { label: 'Needs attention', value: 'attention' },
]

const sortOptions = [
  { label: 'Name', value: 'name' },
  { label: 'Size', value: 'size' },
  { label: 'Recent use', value: 'recent' },
]
</script>

<style scoped>
.library-toolbar__select {
  flex: 0 0 11rem;
  min-width: 0;
}

.library-toolbar__secondary {
  flex-basis: 100%;
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(min(100%, 10rem), 1fr));
  gap: var(--toolbar-gap);
}

.library-view {
  display: inline-flex;
  gap: 0.25rem;
}

.library-view__button[aria-pressed='true'] {
  color: var(--accent-primary);
  background: var(--accent-cyan-soft);
}

@media (max-width: 640px), (pointer: coarse) {
  .library-toolbar__select {
    flex: 1 1 auto;
    width: 100%;
  }

  .library-toolbar__secondary {
    grid-template-columns: minmax(0, 1fr);
  }
}
</style>
