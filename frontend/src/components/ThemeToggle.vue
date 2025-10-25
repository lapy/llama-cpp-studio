<template>
  <Button 
    :icon="isDark ? 'pi pi-sun' : 'pi pi-moon'"
    @click="toggleTheme"
    severity="secondary"
    text
    :title="isDark ? 'Switch to Light Mode' : 'Switch to Dark Mode'"
    class="theme-toggle animate-pulse"
  />
</template>

<script setup>
import { useTheme } from '@/composables/useTheme'
import Button from 'primevue/button'

const { isDark, toggleTheme } = useTheme()
</script>

<style scoped>
.theme-toggle {
  transition: all var(--transition-normal);
  position: relative;
  overflow: hidden;
}

.theme-toggle::before {
  content: '';
  position: absolute;
  top: 0;
  left: -100%;
  width: 100%;
  height: 100%;
  background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
  transition: left var(--transition-normal);
}

.theme-toggle:hover {
  transform: rotate(180deg) scale(1.1);
  box-shadow: var(--shadow-lg), var(--glow-primary);
}

.theme-toggle:hover::before {
  left: 100%;
}

.theme-toggle:active {
  transform: rotate(180deg) scale(0.95);
}
</style>
