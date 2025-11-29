<template>
  <div class="installed-versions">
    <h3>Installed Versions</h3>
    <div v-if="versions.length === 0" class="empty-state">
      <i class="pi pi-code" style="font-size: 3rem; color: var(--text-secondary);"></i>
      <h4>No Versions Installed</h4>
      <p>Install a release or build from source to get started.</p>
    </div>
    
    <div v-else class="version-list">
      <VersionCard
        v-for="version in versions"
        :key="version.id"
        :version="version"
        :activating="activating"
        @activate="$emit('activate', $event)"
        @delete="$emit('delete', $event)"
      />
    </div>
  </div>
</template>

<script setup>
import VersionCard from './VersionCard.vue'

defineProps({
  versions: {
    type: Array,
    default: () => []
  },
  activating: {
    type: [String, Number],
    default: null
  }
})

defineEmits(['activate', 'delete'])
</script>

<style scoped>
.installed-versions {
  margin-top: 2rem;
}

.version-list {
  display: flex;
  flex-direction: column;
  gap: 1rem;
  margin-top: 1rem;
}
</style>

