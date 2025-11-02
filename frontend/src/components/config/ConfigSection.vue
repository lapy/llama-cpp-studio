<template>
  <details 
    class="config-section" 
    :open="expanded" 
    v-bind="$attrs" 
    :aria-label="`${title} configuration section`"
    @toggle.capture.stop.prevent
    @click.capture.stop
  >
    <summary 
      class="section-title" 
      @click.stop.prevent="$emit('toggle')"
      @mousedown.stop
      :aria-expanded="expanded"
      :aria-controls="`section-${title.toLowerCase().replace(/\s+/g, '-')}`"
    >
      <div class="title-left">
        <i :class="icon" aria-hidden="true"></i>
        <span>{{ title }}</span>
      </div>
      <span v-if="badge" class="section-badge" :class="badgeClass" :aria-label="`${badge} section`">{{ badge }}</span>
    </summary>
    <div 
      v-show="expanded"
      class="section-grid" 
      :id="`section-${title.toLowerCase().replace(/\s+/g, '-')}`" 
      :aria-hidden="!expanded"
    >
      <slot></slot>
    </div>
  </details>
</template>

<script setup>
defineProps({
  title: {
    type: String,
    required: true
  },
  icon: {
    type: String,
    required: true
  },
  expanded: {
    type: Boolean,
    default: false
  },
  badge: {
    type: String,
    default: null
  },
  badgeClass: {
    type: String,
    default: ''
  }
})

defineEmits(['toggle'])

// Expose all attributes to allow data-section to be passed through
defineOptions({
  inheritAttrs: false
})
</script>

<style scoped>
.config-section {
  background: var(--gradient-card);
  border: 1px solid var(--border-primary);
  border-radius: var(--radius-xl);
  box-shadow: var(--shadow-md);
  transition: all var(--transition-normal);
  position: relative;
  overflow: hidden;
  backdrop-filter: blur(10px);
  animation: fadeIn 0.6s ease-out;
}

.config-section > summary {
  cursor: pointer;
  padding: var(--spacing-xl);
  list-style: none;
  position: relative;
  display: block;
}

.config-section > summary::-webkit-details-marker {
  display: none !important;
}

.config-section > summary::marker {
  display: none !important;
}

.config-section[open] > summary {
  border-bottom: 1px solid var(--border-primary);
}

.section-title {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin: 0;
  color: var(--text-primary);
  font-size: 1.1rem;
  font-weight: 600;
  user-select: none;
}

.title-left {
  display: flex;
  align-items: center;
  gap: var(--spacing-sm);
}

.section-badge {
  display: inline-block;
  padding: var(--spacing-xs) var(--spacing-sm);
  border-radius: var(--radius-sm);
  font-size: 0.75rem;
  font-weight: 600;
  letter-spacing: 0.5px;
  text-transform: uppercase;
}

.section-badge.essential-badge {
  background: rgba(34, 197, 94, 0.15);
  color: #22c55e;
}

.section-badge.advanced-badge {
  background: rgba(245, 158, 11, 0.15);
  color: #f59e0b;
}

.section-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: var(--spacing-lg);
  width: 100%;
  min-width: 0;
  padding: var(--spacing-xl);
}
</style>

