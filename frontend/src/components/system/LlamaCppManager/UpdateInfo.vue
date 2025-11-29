<template>
  <div v-if="updateInfo" class="update-info">
    <h3>Available Updates</h3>
    <div class="update-cards">
      <div class="update-card">
        <div class="update-header">
          <h4>Latest Release</h4>
          <Tag :value="updateInfo.latest_release?.tag_name || 'N/A'" severity="info" />
        </div>
        <p v-if="updateInfo.latest_release">
          Published: {{ formatDate(updateInfo.latest_release.published_at) }}
        </p>
        <Button 
          label="Install Release"
          icon="pi pi-download"
          @click="$emit('install-release', updateInfo.latest_release.tag_name)"
          :loading="installingRelease"
          :disabled="!updateInfo.latest_release"
        />
      </div>
      
      <div class="update-card">
        <div class="update-header">
          <h4>Latest Source</h4>
          <Tag :value="updateInfo.latest_commit?.sha?.substring(0, 8) || 'N/A'" severity="success" />
        </div>
        <p v-if="updateInfo.latest_commit">
          {{ updateInfo.latest_commit.message }}
        </p>
        <Button 
          label="Build from Source"
          icon="pi pi-code"
          @click="$emit('build-source')"
          :loading="buildingSource"
        />
      </div>
    </div>
  </div>
</template>

<script setup>
import { formatDate } from '@/utils/formatting'
import Button from 'primevue/button'
import Tag from 'primevue/tag'

defineProps({
  updateInfo: {
    type: Object,
    default: null
  },
  installingRelease: {
    type: Boolean,
    default: false
  },
  buildingSource: {
    type: Boolean,
    default: false
  }
})

defineEmits(['install-release', 'build-source'])
</script>

<style scoped>
.update-info {
  margin-bottom: 2rem;
}

.update-cards {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: 1rem;
  margin-top: 1rem;
}

.update-card {
  background: var(--gradient-card);
  border: 1px solid var(--border-primary);
  border-radius: var(--radius-xl);
  padding: var(--spacing-xl);
  box-shadow: var(--shadow-md);
  transition: all var(--transition-normal);
  position: relative;
  overflow: hidden;
}

.update-card::before {
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

.update-card:hover {
  transform: translateY(-3px);
  box-shadow: var(--shadow-lg);
}

.update-card:hover::before {
  opacity: 1;
}

.update-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 0.5rem;
}

.update-header h4 {
  margin: 0;
  color: var(--text-primary);
  font-weight: 700;
  font-size: 1.1rem;
}

.update-card p {
  margin: var(--spacing-md) 0 var(--spacing-lg);
  font-size: 0.9rem;
  color: var(--text-secondary);
  line-height: 1.5;
}

@media (max-width: 768px) {
  .update-cards {
    grid-template-columns: 1fr;
  }
}
</style>

