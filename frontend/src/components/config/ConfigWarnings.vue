<template>
  <div v-if="warnings.length > 0" class="config-warnings">
    <h3 class="section-title">
      <i class="pi pi-exclamation-triangle"></i>
      Configuration Warnings
    </h3>
    <div class="warnings-list">
      <div 
        v-for="warning in warnings" 
        :key="warning.field" 
        class="warning-item" 
        :class="warning.type"
      >
        <i :class="warning.type === 'error' ? 'pi pi-times-circle' : 'pi pi-exclamation-triangle'"></i>
        <span>{{ warning.message }}</span>
      </div>
    </div>
  </div>
</template>

<script setup>
defineProps({
  warnings: {
    type: Array,
    default: () => []
  }
})
</script>

<style scoped>
.config-warnings {
  margin: var(--spacing-lg) 0;
  padding: var(--spacing-lg);
  background: var(--bg-surface);
  border: 1px solid var(--border-primary);
  border-radius: var(--radius-lg);
}

.section-title {
  display: flex;
  align-items: center;
  gap: var(--spacing-sm);
  margin: 0 0 var(--spacing-md) 0;
  font-size: 1.1rem;
  font-weight: 600;
  color: var(--text-primary);
}

.section-title i {
  color: var(--status-warning);
}

.warnings-list {
  display: flex;
  flex-direction: column;
  gap: var(--spacing-sm);
}

.warning-item {
  display: flex;
  align-items: flex-start;
  gap: var(--spacing-sm);
  padding: var(--spacing-sm);
  border-radius: var(--radius-md);
  font-size: 0.875rem;
  line-height: 1.5;
}

.warning-item.error {
  background: var(--status-error-soft);
  color: var(--status-error);
  border-left: 3px solid var(--status-error);
}

.warning-item.warning {
  background: var(--status-warning-soft);
  color: var(--status-warning);
  border-left: 3px solid var(--status-warning);
}

.warning-item i {
  font-size: 1rem;
  flex-shrink: 0;
  margin-top: 2px;
}

.warning-item span {
  flex: 1;
}
</style>
