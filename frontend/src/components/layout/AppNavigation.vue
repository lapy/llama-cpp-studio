<template>
  <nav class="layout-nav animate-slide-in-up" aria-label="Main">
    <div class="nav-content">
      <RouterLink
        v-for="item in items"
        :key="item.name"
        :to="item.to"
        class="p-button nav-button"
        :class="{ 'p-button-outlined': $route.name !== item.name }"
        :aria-current="$route.name === item.name ? 'page' : undefined"
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

const items = [
  { name: 'models', to: '/models', label: 'Models', iconClass: 'pi-database' },
  { name: 'audio', to: '/audio', label: 'Audio', iconClass: 'pi-volume-up' },
  { name: 'search', to: '/search', label: 'Search', iconClass: 'pi-search' },
  { name: 'engines', to: '/engines', label: 'Engines', iconClass: 'pi-cog' },
]
</script>

<style scoped>
/* Navigation button styling */
.nav-content .p-button {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: var(--spacing-sm);
  text-decoration: none;
}

.nav-content .p-button:focus {
  outline: none;
}

.nav-content .p-button:focus-visible {
  outline: 2px solid var(--accent-cyan);
  outline-offset: 2px;
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

@media (max-width: 768px) {
  .nav-content .p-button {
    flex-direction: column;
    gap: 0.15rem;
    padding: 0.45rem 0.15rem;
    font-size: 0.68rem;
    min-height: 3.25rem;
    line-height: 1.15;
  }

  .nav-content .p-button .p-button-icon {
    margin-right: 0;
    font-size: 1rem;
  }

  .nav-content .p-button:hover .p-button-icon,
  .nav-content .p-button:not(.p-button-outlined):hover,
  .nav-content .p-button.p-button-outlined:hover {
    transform: none;
  }
}

@media (hover: none) {
  .nav-content .p-button:hover .p-button-icon,
  .nav-content .p-button:not(.p-button-outlined):hover,
  .nav-content .p-button.p-button-outlined:hover {
    transform: none;
  }
}
</style>
