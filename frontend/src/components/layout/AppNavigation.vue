<template>
  <nav class="layout-nav" aria-label="Main">
    <div class="nav-content">
      <RouterLink
        v-for="item in items"
        :key="item.name"
        :to="item.to"
        class="p-button nav-button"
        :class="{ 'p-button-outlined': !isCurrent(item) }"
        :aria-current="isCurrent(item) ? 'page' : undefined"
      >
        <span :class="['p-button-icon', 'pi', item.iconClass]" aria-hidden="true" />
        <span class="p-button-label">{{ item.label }}</span>
      </RouterLink>
    </div>
  </nav>
</template>

<script setup>
import { useRoute } from 'vue-router'

const $route = useRoute()

function isCurrent(item) {
  if (item.name === 'models') return $route.name === 'models' || $route.name === 'model-config'
  return $route.name === item.name
}

const items = [
  { name: 'models', to: '/models', label: 'Models', iconClass: 'pi-database' },
  { name: 'audio', to: '/audio', label: 'Audio', iconClass: 'pi-volume-up' },
  { name: 'search', to: '/search', label: 'Search', iconClass: 'pi-search' },
  { name: 'engines', to: '/engines', label: 'Engines', iconClass: 'pi-cog' },
]
</script>

<style scoped>
.nav-content .p-button {
  display: flex;
  align-items: center;
  justify-content: flex-start;
  gap: 0.8rem;
  min-height: 2.875rem;
  padding: 0.7rem 0.85rem;
  border: 1px solid transparent;
  border-radius: var(--radius-md);
  color: var(--text-secondary);
  background: transparent;
  box-shadow: none;
  font-size: 0.875rem;
  font-weight: 500;
  text-decoration: none;
}
.nav-content .p-button-label {
  flex: none;
}
.nav-content .p-button:not(.p-button-outlined) {
  background: var(--nav-active-bg);
  color: var(--accent-primary);
  border-color: color-mix(in srgb, var(--accent-primary) 22%, transparent);
}
.nav-content .p-button:hover {
  background: var(--hover-bg);
}
.nav-content .p-button:focus-visible {
  outline: 2px solid var(--accent-primary);
  outline-offset: 2px;
}

@media (max-width: 900px) {
  .nav-caption,
  .nav-content .p-button {
    justify-content: center;
    flex-direction: column;
    gap: 0.35rem;
    padding: 0.5rem 0.25rem;
    font-size: 0.75rem;
    min-height: 3.5rem;
  }
}
</style>
