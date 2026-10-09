<template>
  <header class="layout-header">
    <div class="layout-header-content">
      <div class="logo">
        <span class="brand-mark" aria-hidden="true">
          <span class="brand-orbit">
            <span /><span /><span /><span /><span /><span /><span /><span />
          </span>
        </span>
        <span class="logo-copy">
          <strong>llama.cpp Studio</strong>
        </span>
      </div>
      <div class="header-actions">
        <slot name="actions">
          <a
            class="llama-swap-link"
            :href="llamaSwapUiUrl"
            target="_blank"
            rel="noopener noreferrer"
            :aria-label="llamaSwapLinkLabel"
            v-tooltip.bottom="'Open llama-swap UI'"
          >
            <span
              class="status-light"
              :class="`status-light--${llamaSwapState}`"
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
    default: null,
  },
})

const llamaSwapState = computed(() => {
  if (!props.llamaSwapStatus || props.llamaSwapStatus.healthy == null) return 'unknown'
  return props.llamaSwapStatus.healthy ? 'online' : 'offline'
})
const llamaSwapLinkLabel = computed(() => {
  if (llamaSwapState.value === 'online') return 'Open llama-swap UI, proxy online'
  if (llamaSwapState.value === 'offline') return 'Open llama-swap UI, proxy offline'
  return 'Open llama-swap UI, proxy status unknown'
})
const llamaSwapPort = computed(() => {
  const port = Number(props.llamaSwapStatus?.port)
  return Number.isFinite(port) && port > 0 ? port : 2000
})

/** Same host as this app, with the configured llama-swap proxy port. */
const llamaSwapUiUrl = computed(() => {
  if (typeof window === 'undefined') {
    return '#'
  }
  const { protocol, hostname } = window.location
  return `${protocol}//${hostname}:${llamaSwapPort.value}/ui`
})
</script>

<style scoped>
.brand-mark {
  position: relative;
  width: 2rem;
  height: 2rem;
  flex-shrink: 0;
}

.brand-orbit {
  position: absolute;
  inset: 0;
}

.brand-orbit span {
  position: absolute;
  top: 50%;
  left: 50%;
  width: 1.15rem;
  height: 1.15rem;
  margin: -0.575rem;
  border-radius: 50%;
  opacity: 0.72;
  transform: rotate(calc(var(--i) * 45deg)) translateX(0.38rem);
}

.brand-orbit span:nth-child(1) { --i: 0; background: #ff3b30; }
.brand-orbit span:nth-child(2) { --i: 1; background: #ff8a00; }
.brand-orbit span:nth-child(3) { --i: 2; background: #ffe14a; }
.brand-orbit span:nth-child(4) { --i: 3; background: #6dff4a; }
.brand-orbit span:nth-child(5) { --i: 4; background: #2ee6c7; }
.brand-orbit span:nth-child(6) { --i: 5; background: #3ecbff; }
.brand-orbit span:nth-child(7) { --i: 6; background: #6a6bff; }
.brand-orbit span:nth-child(8) { --i: 7; background: #d946ef; }

.logo-copy {
  display: flex;
  flex-direction: column;
  line-height: 1.05;
}

.logo-copy strong {
  font-size: 1rem;
  letter-spacing: -0.015em;
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
  background: color-mix(in srgb, var(--bg-surface) 82%, transparent);
  position: relative;
  transition:
    border-color 0.15s ease,
    transform 0.15s ease,
    background 0.15s ease;
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

.status-light--unknown {
  background: var(--text-muted);
  box-shadow: none;
}

.llama-swap-label {
  font-size: 0.82rem;
  font-weight: 600;
}

@media (max-width: 768px) {
  .logo {
    font-size: 1.05rem;
  }

  .brand-mark {
    width: 1.7rem;
    height: 1.7rem;
  }

  .brand-orbit span {
    width: 0.98rem;
    height: 0.98rem;
    margin: -0.49rem;
    transform: rotate(calc(var(--i) * 45deg)) translateX(0.32rem);
  }

  .llama-swap-label {
    position: absolute;
    width: 1px;
    height: 1px;
    padding: 0;
    margin: -1px;
    overflow: hidden;
    clip: rect(0, 0, 0, 0);
    white-space: nowrap;
    border: 0;
  }

  .llama-swap-link {
    padding: 0.4rem 0.55rem;
  }

  .llama-swap-link .pi-external-link {
    font-size: 0.75rem;
  }
}

@media (max-width: 480px) {
  .brand-mark {
    display: none;
  }
  .logo-copy strong {
    font-size: 0.8125rem;
    font-weight: 600;
  }
  .llama-swap-link {
    min-height: 2.75rem;
  }
}
</style>
