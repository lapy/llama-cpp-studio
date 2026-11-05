<template>
  <Tag :value="status.label" :severity="status.severity" class="engine-version-tag" />
</template>

<script setup>
import { computed } from 'vue'
import Tag from 'primevue/tag'
import { useProgressStore } from '@/stores/progress'

const props = defineProps({
  versions: {
    type: Array,
    default: () => [],
  },
  engineId: {
    type: String,
    default: '',
  },
})

const progressStore = useProgressStore()

const status = computed(() => {
  const versions = Array.isArray(props.versions) ? props.versions : []
  const needle = String(props.engineId || '').toLowerCase()
  const tasks = Object.values(progressStore.tasks || {})
  const building = needle && tasks.some((task) => {
    const state = String(task?.status || '').toLowerCase()
    if (!['running', 'queued'].includes(state)) return false
    const blob = `${task?.type || ''} ${task?.description || ''} ${task?.metadata?.engine || ''}`.toLowerCase()
    return blob.includes(needle)
  })
  if (building) return { label: 'Building', severity: 'info' }
  if (versions.some((version) => version?.is_active)) return { label: 'Active', severity: 'success' }
  if (versions.some((version) => ['failed', 'error'].includes(String(version?.status || version?.build_status || '').toLowerCase()))) {
    return { label: 'Failed', severity: 'danger' }
  }
  if (versions.length) return { label: 'Installed, inactive', severity: 'warn' }
  return { label: 'Not installed', severity: 'secondary' }
})
</script>
