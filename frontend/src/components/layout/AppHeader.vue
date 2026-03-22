<template>
  <header class="layout-header animate-slide-in-up">
    <div class="layout-header-content">
      <div class="logo">
        <span class="logo-emoji" aria-hidden="true">🎨</span>
        <span>llama.cpp Studio</span>
      </div>
      <div class="header-actions">
        <slot name="actions">
          <a
            class="llama-swap-link"
            :href="llamaSwapUiUrl"
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
          <SwapConfigHeaderNotice />
          <ThemeToggle />
        </slot>
      </div>
    </div>
  </header>
</template>

<script setup>
import { computed } from 'vue'
import ThemeToggle from '@/components/ThemeToggle.vue'
import SwapConfigHeaderNotice from '@/components/layout/SwapConfigHeaderNotice.vue'

const props = defineProps({
  llamaSwapStatus: {
    type: Object,
    default: null
  }
})

const llamaSwapHealthy = computed(() => Boolean(props.llamaSwapStatus?.healthy))

/** Same host as this app, llama-swap proxy port (default 2000). */
const llamaSwapUiUrl = computed(() => {
  if (typeof window === 'undefined') {
    return 'http://localhost:2000/ui'
  }
  const { protocol, hostname } = window.location
  const host = hostname || 'localhost'
  return `${protocol}//${host}:2000/ui`
})
</script>

<style scoped>
.logo-emoji {
  font-size: 1.5rem;
  margin-right: 0.5rem;
}

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
  background: var(--status-success);
  box-shadow: 0 0 0.45rem color-mix(in srgb, var(--status-success) 55%, transparent);
}

.status-light--offline {
  background: var(--status-error);
  box-shadow: 0 0 0.45rem color-mix(in srgb, var(--status-error) 45%, transparent);
}

.llama-swap-label {
  font-size: 0.82rem;
  font-weight: 600;
}
</style>

