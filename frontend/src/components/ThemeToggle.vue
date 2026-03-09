<template>
  <Button 
    :icon="isDark ? 'pi pi-sun' : 'pi pi-moon'"
    @click="toggleTheme"
    severity="secondary"
    text
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
  width: 2.4rem;
  height: 2.4rem;
  border-radius: 999px;
  position: relative;
  overflow: hidden;
  transition: transform var(--transition-normal), box-shadow var(--transition-normal), background-color var(--transition-normal), border-color var(--transition-normal);
}

.theme-toggle::before {
  content: '';
  position: absolute;
  top: 0;
  left: -100%;
  width: 100%;
  height: 100%;
  background: linear-gradient(
    90deg, 
    transparent, 
    color-mix(in srgb, var(--accent-cyan) 35%, transparent), 
    transparent
  );
  transition: left var(--transition-normal);
}

.theme-toggle:hover {
  transform: translateY(-1px);
  box-shadow: var(--shadow-md);
}

.theme-toggle:hover::before {
  left: 100%;
}

.theme-toggle:active {
  transform: scale(0.96);
}

.theme-toggle :deep(.p-button-icon) {
  transition: transform var(--transition-normal), color var(--transition-normal);
}

.theme-toggle:hover :deep(.p-button-icon) {
  transform: rotate(12deg) scale(1.08);
}

.theme-toggle.is-dark :deep(.p-button-icon) {
  color: #fbbf24;
}

.theme-toggle:not(.is-dark) :deep(.p-button-icon) {
  color: var(--accent-cyan);
}
</style>
