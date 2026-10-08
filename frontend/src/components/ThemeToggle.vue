<template>
  <Button
    :icon="isDark ? 'pi pi-sun' : 'pi pi-moon'"
    @click="toggleTheme"
    severity="secondary"
    text
    :aria-label="isDark ? 'Switch to Light Mode' : 'Switch to Dark Mode'"
    :title="isDark ? 'Switch to Light Mode' : 'Switch to Dark Mode'"
    :class="['theme-toggle', { 'is-dark': isDark }]"
  />
</template>

<script setup>
import { useTheme } from '@/composables/useTheme'
import Button from 'primevue/button'

const { isDark, toggleTheme } = useTheme()
</script>

<style scoped>
.theme-toggle {
  min-width: 2.75rem;
  min-height: 2.75rem;
  border-radius: var(--radius-md);
  color: var(--text-secondary);
}
.theme-toggle:hover {
  background: var(--hover-bg);
}
</style>
