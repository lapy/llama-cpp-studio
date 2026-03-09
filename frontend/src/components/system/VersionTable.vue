<template>
  <div>
    <div v-if="!versions.length" class="empty-state-mini">
      <i class="pi pi-code" />
      <span>No versions installed — use the buttons above to install one.</span>
    </div>
    <div v-else class="version-table">
      <div
        v-for="v in versions"
        :key="v.id ?? v.version"
        class="version-row"
        :class="{ active: v.is_active }"
      >
        <div class="version-info">
          <code class="version-name">{{ v.version }}</code>
          <Tag v-if="v.is_active" value="Active" severity="success" />
          <Tag :value="v.type || 'source'" severity="secondary" />
          <small v-if="v.repository_source" class="repo-label">{{ v.repository_source }}</small>
          <small v-if="v.build_config?.cuda" class="cuda-badge">CUDA</small>
        </div>
        <div class="version-actions">
          <Button
            v-if="!v.is_active"
            label="Activate"
            icon="pi pi-play"
            size="small"
            severity="success"
            outlined
            :loading="activating === (v.id ?? v.version)"
            @click="$emit('activate', v.id ?? v.version)"
          />
          <Button
            icon="pi pi-trash"
            text
            severity="danger"
            size="small"
            :disabled="v.is_active"
            v-tooltip.top="v.is_active ? 'Active versions cannot be deleted' : 'Delete version'"
            @click="$emit('delete', v.id ?? v.version)"
          />
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import Button from 'primevue/button'
import Tag from 'primevue/tag'

defineProps({
  versions: {
    type: Array,
    default: () => [],
  },
  activating: {
    type: [String, Number],
    default: null,
  },
})

defineEmits(['activate', 'delete'])
</script>

<style scoped>
.empty-state-mini {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  color: var(--text-secondary);
  font-size: 0.875rem;
  padding: 0.75rem 0;
}

.version-table {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}

.version-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0.5rem 0.75rem;
  background: var(--bg-surface);
  border: 1px solid var(--border-primary);
  border-radius: var(--radius-md);
  gap: 0.5rem;
  transition: border-color 0.15s;
}

.version-row.active {
  border-color: var(--accent-green);
}

.version-info {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  flex: 1;
  min-width: 0;
  flex-wrap: wrap;
}

.version-name {
  font-weight: 600;
  font-size: 0.875rem;
  font-family: monospace;
}

.repo-label {
  color: var(--text-secondary);
  font-size: 0.75rem;
}

.cuda-badge {
  background: rgba(34, 211, 238, 0.1);
  color: var(--accent-cyan);
  border: 1px solid rgba(34, 211, 238, 0.3);
  border-radius: 0.25rem;
  padding: 0.1em 0.4em;
  font-size: 0.7rem;
  font-weight: 600;
}

.version-actions {
  display: flex;
  gap: 0.25rem;
  flex-shrink: 0;
}
</style>
