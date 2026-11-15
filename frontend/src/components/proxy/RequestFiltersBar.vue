<template>
  <form
    class="request-filters compact-toolbar"
    role="search"
    aria-label="Request filters"
    @submit.prevent="emit('apply')"
  >
    <div class="compact-field compact-field--wide">
      <label class="compact-label sr-only" for="request-filter-search">Search requests</label>
      <input
        id="request-filter-search"
        v-model="search"
        class="field"
        placeholder="Search ID, alias, path or key"
      />
    </div>
    <div class="compact-field compact-field--wide">
      <label class="compact-label sr-only" for="request-filter-model">Model</label>
      <input
        id="request-filter-model"
        v-model="model"
        class="field"
        placeholder="Resolved model ID"
      />
    </div>
    <div class="compact-field">
      <label class="compact-label sr-only" for="request-filter-key">Client key</label>
      <select id="request-filter-key" v-model="keyId" class="field">
        <option value="">All clients</option>
        <option v-for="key in keys" :key="key.id" :value="key.id">{{ key.name }}</option>
        <option value="anonymous">Anonymous</option>
        <option value="internal">Studio</option>
      </select>
    </div>
    <div class="compact-field">
      <label class="compact-label sr-only" for="request-filter-state">State</label>
      <select id="request-filter-state" v-model="state" class="field">
        <option value="">All states</option>
        <option v-for="option in states" :key="option" :value="option">{{ option }}</option>
      </select>
    </div>
    <div class="compact-field">
      <label class="compact-label sr-only" for="request-filter-outcome">Outcome</label>
      <select id="request-filter-outcome" v-model="outcome" class="field">
        <option value="">All outcomes</option>
        <option value="4xx">HTTP 4xx</option>
        <option value="5xx">HTTP 5xx</option>
        <option value="errors">All HTTP 4xx / 5xx</option>
      </select>
    </div>
    <div class="compact-field compact-field--narrow">
      <label class="compact-label sr-only" for="request-filter-status">HTTP status</label>
      <input
        id="request-filter-status"
        v-model="status"
        class="field"
        type="number"
        min="100"
        max="599"
        placeholder="Any"
      />
    </div>
    <div class="request-filter-actions">
      <Button label="Apply" size="small" :loading="loading" @click="emit('apply')" />
      <Button
        label="Reset"
        size="small"
        severity="secondary"
        text
        :disabled="!dirty"
        @click="emit('reset')"
      />
    </div>
  </form>
</template>

<script setup>
import { computed } from 'vue'
import Button from 'primevue/button'

const props = defineProps({
  filters: { type: Object, required: true },
  keys: { type: Array, default: () => [] },
  states: { type: Array, default: () => [] },
  loading: { type: Boolean, default: false },
})
const emit = defineEmits(['apply', 'reset', 'update:filter'])
function bindFilter(key) {
  return computed({
    get: () => props.filters[key],
    set: (value) => emit('update:filter', key, value),
  })
}
const search = bindFilter('search')
const model = bindFilter('model')
const keyId = bindFilter('key_id')
const state = bindFilter('state')
const outcome = bindFilter('outcome')
const status = bindFilter('status')
const dirty = computed(() => Object.values(props.filters).some((value) => value !== ''))
</script>

<style scoped>
.request-filters {
  align-items: flex-end;
}

.request-filter-actions {
  display: flex;
  align-items: center;
  gap: 0.25rem;
  margin-left: auto;
}

.request-filter-actions :deep(.p-button.p-button-sm) {
  min-height: var(--compact-control-height);
}

@media (max-width: 640px), (pointer: coarse) {
  .request-filter-actions {
    margin-left: 0;
  }

  .request-filter-actions :deep(.p-button.p-button-sm) {
    flex: 1 1 auto;
  }
}
</style>
