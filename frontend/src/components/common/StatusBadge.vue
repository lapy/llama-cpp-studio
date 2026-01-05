<template>
  <span 
    class="status-badge" 
    :class="[`status-${status}`, badgeClass]"
    :title="tooltip"
  >
    <i v-if="icon" :class="icon" class="status-icon"></i>
    <span v-if="label">{{ label }}</span>
    <slot></slot>
  </span>
</template>

<script setup>
defineProps({
  status: {
    type: String,
    required: true,
    validator: (value) => ['success', 'warning', 'error', 'info', 'running', 'stopped', 'downloading'].includes(value)
  },
  label: {
    type: String,
    default: ''
  },
  icon: {
    type: String,
    default: ''
  },
  tooltip: {
    type: String,
    default: ''
  },
  badgeClass: {
    type: String,
    default: ''
  }
})
</script>

<style scoped>
.status-badge {
  display: inline-flex;
  align-items: center;
  gap: var(--spacing-xs);
  padding: var(--spacing-xs) var(--spacing-sm);
  border-radius: var(--radius-md);
  font-size: 0.75rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  white-space: nowrap;
}

.status-icon {
  font-size: 0.875rem;
}

.status-success {
  background: rgba(16, 185, 129, 0.1);
  color: var(--status-success);
  border: 1px solid rgba(16, 185, 129, 0.2);
}

.status-warning {
  background: rgba(245, 158, 11, 0.1);
  color: var(--status-warning);
  border: 1px solid rgba(245, 158, 11, 0.2);
}

.status-error {
  background: rgba(239, 68, 68, 0.1);
  color: var(--status-error);
  border: 1px solid rgba(239, 68, 68, 0.2);
}

.status-info {
  background: rgba(34, 211, 238, 0.1);
  color: var(--status-info);
  border: 1px solid rgba(34, 211, 238, 0.2);
}

.status-running {
  background: rgba(16, 185, 129, 0.1);
  color: var(--status-success);
  border: 1px solid rgba(16, 185, 129, 0.2);
}

.status-stopped {
  background: rgba(148, 163, 184, 0.1);
  color: var(--text-muted);
  border: 1px solid rgba(148, 163, 184, 0.2);
}

.status-downloading {
  background: rgba(34, 211, 238, 0.1);
  color: var(--accent-cyan);
  border: 1px solid rgba(34, 211, 238, 0.2);
}
</style>

