<template>
  <nav class="layout-nav animate-slide-in-up">
    <div class="nav-content">
      <Button 
        label="Models" 
        icon="pi pi-database"
        :class="{ 'p-button-outlined': $route.name !== 'models' }"
        @click="$router.push('/models')"
        class="nav-button"
      />
      <Button 
        label="Search" 
        icon="pi pi-search"
        :class="{ 'p-button-outlined': $route.name !== 'search' }"
        @click="$router.push('/search')"
        class="nav-button"
      />
      <Button 
        label="System" 
        icon="pi pi-desktop"
        :class="{ 'p-button-outlined': $route.name !== 'system' }"
        @click="$router.push('/system')"
        class="nav-button"
      />
    </div>
  </nav>
</template>

<script setup>
import { useRouter, useRoute } from 'vue-router'
import Button from 'primevue/button'

const router = useRouter()
const $route = useRoute()
</script>

<style scoped>
/* Navigation button styling */
.nav-content .p-button {
  display: flex;
  align-items: center;
  gap: var(--spacing-sm);
}

.nav-content .p-button .p-button-icon {
  margin-right: var(--spacing-sm);
  transition: transform var(--transition-normal);
}

.nav-content .p-button:hover .p-button-icon {
  transform: scale(1.1) rotate(5deg);
}

.nav-content .p-button:not(.p-button-outlined) {
  background: var(--gradient-primary);
  color: white;
  border: none;
  box-shadow: var(--shadow-md), var(--glow-primary);
}

.nav-content .p-button:not(.p-button-outlined):hover {
  transform: translateY(-2px);
  box-shadow: var(--shadow-lg), var(--glow-primary);
}

.nav-content .p-button.p-button-outlined:hover {
  background: var(--gradient-primary);
  color: white;
  border-color: var(--accent-cyan);
  transform: translateY(-2px);
}
</style>

