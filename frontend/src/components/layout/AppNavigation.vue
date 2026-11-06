<template>
  <nav class="layout-nav" aria-label="Main">
    <div class="nav-content">
      <template v-for="(item, index) in items" :key="item.name">
        <span v-if="index === 0 || index === 3" class="nav-section-label">{{
          index === 0 ? 'Workspace' : 'Operations'
        }}</span>
        <RouterLink
          :to="item.to"
          :aria-label="item.ariaLabel || item.label"
          class="p-button nav-button"
          :class="{ 'p-button-outlined': !isCurrent(item) }"
          :aria-current="isCurrent(item) ? 'page' : undefined"
        >
          <span :class="['p-button-icon', 'pi', item.iconClass]" aria-hidden="true" />
          <span class="p-button-label">{{ item.label }}</span>
        </RouterLink>
      </template>
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
  { name: 'search', to: '/search', label: 'Search', ariaLabel: 'Find models', iconClass: 'pi-search' },
  { name: 'proxy', to: '/proxy', label: 'Proxy', iconClass: 'pi-share-alt' },
  { name: 'engines', to: '/engines', label: 'Engines', iconClass: 'pi-cog' },
  { name: 'system', to: '/system', label: 'System', iconClass: 'pi-sliders-h' },
]
</script>

<style scoped>
.nav-section-label {
  display: block;
  padding: 0.65rem 0.85rem 0.35rem;
  font-size: var(--text-xs);
  text-transform: uppercase;
  letter-spacing: 0.12em;
  color: var(--text-secondary);
}

@media (max-width: 900px) {
  .nav-section-label {
    display: none;
  }
}

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
  max-width: 100%;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
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
    font-size: var(--text-xs);
    line-height: 1.2;
    min-height: 3.5rem;
  }
}
</style>
