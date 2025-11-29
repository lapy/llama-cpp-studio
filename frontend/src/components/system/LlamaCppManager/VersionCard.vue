<template>
  <div 
    class="version-card"
    :class="{ 'active-version': version.is_active }"
  >
    <div class="version-header">
      <div class="version-info">
        <div class="version-title">
          <h4>{{ version.version }}</h4>
          <div v-if="version.is_active" class="active-indicators">
            <Tag 
              value="ACTIVE" 
              severity="success"
              class="active-badge"
            />
            <i class="pi pi-check-circle active-icon"></i>
          </div>
        </div>
        <div class="version-meta">
          <Tag 
            :value="version.install_type" 
            :severity="getInstallTypeSeverity(version.install_type)"
          />
          <span class="install-date">
            Installed: {{ formatDate(version.installed_at) }}
          </span>
        </div>
      </div>
      <div class="version-actions">
        <Button 
          v-if="!version.is_active"
          icon="pi pi-check"
          @click="$emit('activate', version.id)"
          severity="success"
          size="small"
          text
          :loading="activating === version.id"
        />
        <Button 
          icon="pi pi-trash"
          severity="danger"
          outlined
          @click="$emit('delete', version)"
          :disabled="version.is_active"
        />
      </div>
    </div>
    
    <div v-if="version.source_commit" class="version-details">
      <p><strong>Commit:</strong> {{ version.source_commit }}</p>
    </div>
    
    <div v-if="version.patches && version.patches.length > 0" class="version-patches">
      <p><strong>Patches Applied:</strong></p>
      <ul>
        <li v-for="patch in version.patches" :key="patch">
          <a :href="patch" target="_blank">{{ patch }}</a>
        </li>
      </ul>
    </div>
  </div>
</template>

<script setup>
import { formatDate } from '@/utils/formatting'
import Button from 'primevue/button'
import Tag from 'primevue/tag'

defineProps({
  version: {
    type: Object,
    required: true
  },
  activating: {
    type: [String, Number],
    default: null
  }
})

defineEmits(['activate', 'delete'])

const getInstallTypeSeverity = (type) => {
  switch (type) {
    case 'release': return 'info'
    case 'source': return 'success'
    case 'patched': return 'warning'
    default: return 'secondary'
  }
}
</script>

<style scoped>
.version-card {
  background: var(--gradient-card);
  border: 1px solid var(--border-primary);
  border-radius: var(--radius-xl);
  padding: var(--spacing-xl);
  box-shadow: var(--shadow-md);
  transition: all var(--transition-normal);
  position: relative;
  overflow: hidden;
  backdrop-filter: blur(10px);
  animation: fadeIn 0.6s ease-out;
}

.version-card::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  height: 3px;
  background: var(--gradient-primary);
  opacity: 0;
  transition: opacity var(--transition-normal);
}

.version-card:hover {
  transform: translateY(-5px) scale(1.02);
  box-shadow: var(--shadow-lg), var(--glow-primary);
}

.version-card:hover::before {
  opacity: 1;
}

.version-card.active-version {
  background: var(--gradient-card);
  border: 2px solid var(--status-success);
  box-shadow: var(--shadow-lg), var(--glow-success);
}

.version-card.active-version::before {
  background: var(--gradient-success);
  opacity: 1;
}

.version-card.active-version:hover {
  box-shadow: var(--shadow-xl), var(--glow-success);
}

.version-header {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  margin-bottom: 0.5rem;
}

.version-title {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  margin-bottom: 0.5rem;
}

.version-title h4 {
  margin: 0;
  color: var(--text-primary);
  font-weight: 700;
  font-size: 1.2rem;
}

.active-indicators {
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.active-badge {
  font-size: 0.75rem;
  font-weight: 600;
  animation: pulse 2s infinite;
}

.active-icon {
  color: var(--status-success);
  font-size: 1.25rem;
  animation: pulse 2s infinite;
}

@keyframes pulse {
  0%, 100% {
    opacity: 1;
  }
  50% {
    opacity: 0.7;
  }
}

.version-meta {
  display: flex;
  align-items: center;
  gap: 1rem;
}

.install-date {
  font-size: 0.9rem;
  color: var(--text-secondary);
  font-weight: 500;
}

.version-details,
.version-patches {
  margin-top: var(--spacing-md);
  font-size: 0.9rem;
  color: var(--text-secondary);
  line-height: 1.5;
}

.version-patches ul {
  margin: 0.5rem 0 0 1rem;
}

.version-patches a {
  color: var(--accent-cyan);
  text-decoration: none;
  font-weight: 500;
}

.version-patches a:hover {
  text-decoration: underline;
  color: var(--accent-blue);
}

.version-actions {
  display: flex;
  gap: var(--spacing-sm);
  align-items: center;
}

@media (max-width: 768px) {
  .version-header {
    flex-direction: column;
    gap: 1rem;
  }
  
  .version-meta {
    flex-direction: column;
    align-items: flex-start;
    gap: 0.5rem;
  }
}
</style>

