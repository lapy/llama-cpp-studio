<template>
  <header class="layout-header animate-slide-in-up">
    <div class="layout-header-content">
      <div class="logo">
        <span style="font-size: 1.5rem; margin-right: 0.5rem;">🎨</span>
        <span>llama.cpp Studio</span>
      </div>
      <div class="header-actions">
        <slot name="actions">
          <a
            class="llama-swap-link"
            href="http://localhost:2000/ui"
            target="_blank"
            rel="noopener noreferrer"
            v-tooltip.bottom="'Open llama-swap UI'"
          >
            <span
              class="status-light"
              :class="llamaSwapHealthy ? 'status-light--online' : 'status-light--offline'"
              aria-hidden="true"
            />
            <span class="llama-swap-label">llama-swap</span>
            <i class="pi pi-external-link" aria-hidden="true" />
          </a>
          <ThemeToggle />
        </slot>
      </div>
    </div>
  </header>
</template>

<script setup>
import { computed } from 'vue'
import ThemeToggle from '@/components/ThemeToggle.vue'

const props = defineProps({
  llamaSwapStatus: {
    type: Object,
    default: null
  }
})

const llamaSwapHealthy = computed(() => Boolean(props.llamaSwapStatus?.healthy))
</script>

<style scoped>
.llama-swap-link {
  display: inline-flex;
  align-items: center;
  gap: 0.45rem;
  padding: 0.45rem 0.7rem;
  border: 1px solid var(--border-primary);
  border-radius: 999px;
  color: var(--text-primary);
  text-decoration: none;
  background: var(--bg-surface);
  transition: border-color 0.15s ease, transform 0.15s ease, background 0.15s ease;
}

.llama-swap-link:hover {
  border-color: var(--accent-cyan);
  background: var(--bg-card-hover, rgba(255, 255, 255, 0.04));
  transform: translateY(-1px);
}

.status-light {
  width: 0.55rem;
  height: 0.55rem;
  border-radius: 999px;
  display: inline-block;
  box-shadow: 0 0 0 0.2rem rgba(255, 255, 255, 0.04);
}

.status-light--online {
  background: #22c55e;
  box-shadow: 0 0 0.45rem rgba(34, 197, 94, 0.55);
}

.status-light--offline {
  background: #ef4444;
  box-shadow: 0 0 0.45rem rgba(239, 68, 68, 0.45);
}

.llama-swap-label {
  font-size: 0.82rem;
  font-weight: 600;
}
</style>

